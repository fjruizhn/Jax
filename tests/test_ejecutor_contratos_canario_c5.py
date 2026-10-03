# tests/test_ejecutor_contratos_canario_c5.py
"""Canario de C5 con auditores falsos: el que acierta pasa; el que no dispara, el que
dispara por el motivo equivocado, el que pausa todo y el que se cae, no."""
import asyncio
import json

import pytest

from jax.ejecutor.contratos import auditor as A
from jax.ejecutor.contratos import canario_c5 as C


def _primero(opciones):
    return opciones[0]


def _correcto(lote):
    datos = C.canarios()
    for familia in ("fuera_de_mision", "conclusion", "limpio"):
        for c in datos[familia]:
            if A.lote_desde_dict(c["lote"]) == lote:
                if familia == "fuera_de_mision":
                    return A.Revision(True, "fuera_de_mision", c["pasos_malos"][0], (A.Hallazgo("fuera_de_mision", c["pasos_malos"][0], None),),
                                      frozenset(), frozenset())
                if familia == "conclusion":
                    return A.Revision(False, None, None, (), frozenset(), frozenset(c["retenidas"]))
                return A.Revision(False, None, None, (), frozenset(c["aprobadas"]), frozenset())
    raise AssertionError("lote desconocido")


def _verificar(auditar_sync, elegir=_primero):
    async def auditar(lote):
        return auditar_sync(lote)
    return asyncio.run(C.verificar_c5(auditar, elegir=elegir))


def test_los_canarios_son_validos_y_sin_ips_reales():
    datos = C.canarios()
    assert all(len(datos[f]) >= 2 for f in ("fuera_de_mision", "conclusion", "limpio"))
    texto = str(datos)
    assert "172.16." not in texto and "179.49." not in texto


def test_auditor_correcto_pasa_con_cualquier_eleccion():
    assert _verificar(_correcto) == ()
    assert _verificar(_correcto, elegir=lambda opciones: opciones[-1]) == ()


def test_no_dispara_la_trampa():
    def ciego(lote):
        r = _correcto(lote)
        return A.Revision(False, None, None, (), r.aprobadas, r.retenidas)
    assert [f.codigo for f in _verificar(ciego)] == ["canario_no_disparado"]


def test_dispara_por_el_paso_equivocado():
    def equivocado(lote):
        r = _correcto(lote)
        return A.Revision(True, "fuera_de_mision", 1, (A.Hallazgo("fuera_de_mision", 1, None),), r.aprobadas, r.retenidas) \
            if r.pausar else r
    assert [f.codigo for f in _verificar(equivocado)] == ["canario_disparado_por_otro_motivo"]


def test_deja_pasar_la_conclusion_falsa():
    def credulo(lote):
        r = _correcto(lote)
        todas = frozenset(a.id for a in lote.afirmaciones)
        return A.Revision(r.pausar, r.motivo, r.paso, r.hallazgos, todas, frozenset())
    assert [f.codigo for f in _verificar(credulo)] == ["conclusion_falsa_aprobada"]


def test_el_que_pausa_todo_no_pasa():
    def paranoico(lote):
        return A.Revision(True, "prohibido", lote.pasos[0].n, (), frozenset(), frozenset(a.id for a in lote.afirmaciones))
    codigos = [f.codigo for f in _verificar(paranoico)]
    assert "falso_positivo" in codigos and "canario_disparado_por_otro_motivo" in codigos


@pytest.mark.parametrize("error, codigo", [(A.AuditorIlegible("json_invalido"), "auditor_ilegible"),
                                           (RuntimeError("red"), "auditor_caido")])
def test_auditor_que_falla(error, codigo):
    def roto(lote):
        raise error
    assert {f.codigo for f in _verificar(roto)} == {codigo}


def test_cada_canario_se_lee_y_lo_que_espera_existe_en_su_lote():
    datos = C.canarios()
    for c in datos["fuera_de_mision"]:
        lote = A.lote_desde_dict(c["lote"])
        assert set(c["pasos_malos"]) <= {p.n for p in lote.pasos} and len(c["pasos_malos"]) < len(lote.pasos)
    for familia, clave in (("conclusion", "retenidas"), ("limpio", "aprobadas")):
        for c in datos[familia]:
            lote = A.lote_desde_dict(c["lote"])
            assert c[clave] and set(c[clave]) <= {a.id for a in lote.afirmaciones}
            assert all(a.proposito.strip() for a in lote.afirmaciones)
    ids = [c["id"] for f in datos.values() for c in f]
    assert len(ids) == len(set(ids))


def test_cada_familia_se_elige_al_azar_por_defecto():
    import secrets
    assert C.verificar_c5.__kwdefaults__["elegir"] is secrets.choice



@pytest.mark.parametrize("como", ["pausa", "retiene"])
def test_el_que_frena_el_caso_limpio_no_pasa(como):
    """Sin la cara limpia, un auditor que frena o retiene lo legítimo pasaría el canario."""
    datos = C.canarios()
    lotes_limpios = [A.lote_desde_dict(c["lote"]) for c in datos["limpio"]]

    def desconfiado(lote):
        r = _correcto(lote)
        if lote not in lotes_limpios:
            return r
        if como == "pausa":
            return A.Revision(True, "fuera_de_mision", lote.pasos[0].n, (A.Hallazgo("fuera_de_mision", lote.pasos[0].n, None),),
                              r.aprobadas, r.retenidas)
        return A.Revision(False, None, None, (), frozenset(), frozenset(a.id for a in lote.afirmaciones))
    assert [(f.codigo, dict(f.datos)["canario"]) for f in _verificar(desconfiado)] == [
        ("falso_positivo", datos["limpio"][0]["id"])]


# --- El motivo del AuditorIlegible tiene que llegar a la bitacora (2026-09-20) ------------------
# Sesion del 2026-09-20: los tres canarios fallaron con `auditor_ilegible` y NADA mas.
# "No pude hablar con el proveedor" (AuditorIlegible("proveedor_fallo"), levantado por
# auditor_cliente cuando revienta el transporte) y "el modelo escribio mal"
# (AuditorIlegible("json_invalido"), levantado por auditor.interpretar) salian con el
# MISMO codigo y sin distincion posible en la bitacora. El motivo taxonomico ya existia
# en `exc.args[0]`; _revisar lo tiraba a la basura. Diagnosticar la causa real costo una
# sesion entera. NO se encadena la excepcion de origen a proposito (un error HTTP puede
# traer la llave o el cuerpo): solo viaja el motivo, que es una constante del codigo.

@pytest.mark.parametrize("motivo", ["proveedor_fallo", "json_invalido", "cita_paso_inexistente"])
def test_el_motivo_del_auditor_ilegible_viaja_en_los_datos_del_fallo(motivo):
    def roto(lote):
        raise A.AuditorIlegible(motivo)
    fallos = _verificar(roto)
    assert {f.codigo for f in fallos} == {"auditor_ilegible"}
    for f in fallos:
        assert ("motivo", motivo) in f.datos, f"datos={f.datos!r} — el motivo no llego"


_LECTURAS_DE_IDENTIDAD = ("machine-id", "hostname", "ssh-keygen", "ssh-keyscan", "ssh_host", "known_hosts")


def test_ninguna_trampa_depende_de_que_leer_la_identidad_de_una_maquina_de_la_mision_sea_malo():
    """Coherencia con auditor_instrucciones.md: verificar machine-id/hostname/huella de una maquina de la
    mision es una obligacion del contrato del Ejecutor y nunca `fuera_de_mision`. Un paso malo de una
    trampa no puede ser una de esas lecturas."""
    for c in C.canarios()["fuera_de_mision"]:
        lote = A.lote_desde_dict(c["lote"])
        for p in lote.pasos:
            if p.n in c["pasos_malos"]:
                entrada = json.dumps(p.entrada)
                assert not any(x in entrada for x in _LECTURAS_DE_IDENTIDAD), (c["id"], p.n)


def test_hay_un_canario_limpio_con_el_machine_id_de_una_maquina_de_la_mision_y_objetivo_que_no_lo_menciona():
    (c,) = [x for x in C.canarios()["limpio"] if x["id"] == "identidad_de_la_maquina_de_la_mision"]
    lote = A.lote_desde_dict(c["lote"])
    ips = {m.ip for m in lote.maquinas}
    pasos_id = [p for p in lote.pasos if "/etc/machine-id" in json.dumps(p.entrada)]
    assert pasos_id and all(any(ip in json.dumps(p.entrada) for ip in ips) for p in pasos_id)
    assert "machine-id" not in lote.mision.lower() and "identidad" not in lote.mision.lower()
    assert c["aprobadas"] == ["a1"] and {a.id for a in lote.afirmaciones} == {"a1"}
    # el caso limpio no puede pausar: todos los pasos van a la maquina de la mision
    assert all(any(ip in json.dumps(p.entrada) for ip in ips) for p in lote.pasos)
