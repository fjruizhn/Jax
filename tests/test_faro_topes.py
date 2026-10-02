"""El Faro, paso 0.3b (P-4): la politica de los topes, sin base de datos (la atomicidad del conteo contra
MariaDB real esta en `test_faro_topes_db.py`).

Reglas que se defienden:
- SIN regla de tope = sin tope de cantidad, pero con MEDICION y AVISO: no niega.
- CON tope (ya resuelto por quien evalua la regla; aqui solo se recibe) al llegar FALLA CERRADO.
- D-4: no hay tope de agentes (ni de conexiones): pedirlo es un error, medirlo no.
- Si el almacen no se puede consultar: con tope se niega (no se sabe si cabe); sin tope se deja pasar (no hay
  regla que niegue) y se declara que no se pudo medir.
- Una anotacion de bitacora que falla no cambia la decision.
"""
from __future__ import annotations

import asyncio
import time

import pytest

from jax.faro.aviso import Avisador, ConfigAviso, Credenciales
from jax.faro.bitacora import Bitacora
from jax.faro.topes import PALABRAS_SIN_TOPE, ResultadoTope, TopeProhibido, Topes
from tests._faro_falsos import AlmacenMemoria
from tests._faro_utils import corre


def _topes(almacen=None, **kw):
    registros = []
    bit = Bitacora(emisores=[registros.append], observadores=kw.pop("observadores", ()))
    return Topes(almacen if almacen is not None else AlmacenMemoria(), bit, **kw), registros


def _consumir(t, **kw):
    base = dict(tenant="t1", recurso="tokens", cantidad=1, tope=None)
    base.update(kw)
    return corre(t.consumir(**base))


# --------------------------------------------------------------------------- #
# sin regla de tope: se mide, se avisa, no se niega                           #
# --------------------------------------------------------------------------- #

def test_sin_regla_de_tope_nunca_se_niega_y_se_mide():
    t, registros = _topes()

    async def caso():
        return [await t.consumir(tenant="t1", recurso="tokens", cantidad=1000, tope=None) for _ in range(50)]
    rs = corre(caso())
    assert all(r.permitido and r.medido and r.tope is None for r in rs)
    assert [r.usado for r in rs][-1] == 50_000
    assert not [x for x in registros if x.get("decision") == "denegado"]


def test_el_consumo_sin_regla_se_anota_y_se_avisa_una_vez_por_tenant_recurso_y_periodo():
    t, registros = _topes()

    async def caso():
        for _ in range(5):
            await t.consumir(tenant="t1", recurso="tokens", cantidad=1, tope=None)
        await t.consumir(tenant="t2", recurso="tokens", cantidad=1, tope=None)
        await t.consumir(tenant="t1", recurso="gasto", cantidad=1, tope=None)
        await t.consumir(tenant="t1", recurso="tokens", cantidad=1, tope=None, periodo="2026-10")
    corre(caso())
    sin_regla = [r for r in registros if r["evento"] == "tope_sin_regla"]
    assert len(sin_regla) == 4
    assert {(r["tenant"], r["recurso"], r["periodo"]) for r in sin_regla} == {
        ("t1", "tokens", "total"), ("t2", "tokens", "total"), ("t1", "gasto", "total"), ("t1", "tokens", "2026-10")}
    assert all(r["decision"] == "permitido" and "regla" in r["motivo"] for r in sin_regla)


# --------------------------------------------------------------------------- #
# con tope: falla cerrado al llegar                                           #
# --------------------------------------------------------------------------- #

def test_con_tope_se_permite_hasta_el_tope_y_el_siguiente_se_niega_sin_pasarse():
    t, registros = _topes()

    async def caso():
        return [await t.consumir(tenant="t1", recurso="tokens", cantidad=1, tope=3, run_id="run-9") for _ in range(5)]
    rs = corre(caso())
    assert [r.permitido for r in rs] == [True, True, True, False, False]
    assert [r.usado for r in rs] == [1, 2, 3, 3, 3]                      # lo negado no suma
    assert rs[3].motivo == "tope_alcanzado" and rs[3].tope == 3
    negados = [r for r in registros if r["evento"] == "tope_superado"]
    assert len(negados) == 2 and all(r["decision"] == "denegado" and r["run_id"] == "run-9" and r["tope"] == 3 for r in negados)


def test_la_cantidad_exacta_hasta_el_tope_cabe_y_una_unidad_mas_no():
    t, _ = _topes()

    async def caso():
        a = await t.consumir(tenant="t1", recurso="gasto", cantidad=6, tope=10)
        b = await t.consumir(tenant="t1", recurso="gasto", cantidad=6, tope=10)     # 12 > 10
        c = await t.consumir(tenant="t1", recurso="gasto", cantidad=4, tope=10)     # 10 <= 10
        d = await t.consumir(tenant="t1", recurso="gasto", cantidad=1, tope=10)
        return a, b, c, d
    a, b, c, d = corre(caso())
    assert (a.permitido, b.permitido, c.permitido, d.permitido) == (True, False, True, False) and c.usado == 10


def test_un_tope_cero_lo_niega_todo():
    t, _ = _topes()
    assert not _consumir(t, tope=0).permitido


def test_cada_tenant_y_cada_periodo_tiene_su_propia_cuenta():
    t, _ = _topes()

    async def caso():
        a = await t.consumir(tenant="t1", recurso="tokens", cantidad=1, tope=1)
        b = await t.consumir(tenant="t2", recurso="tokens", cantidad=1, tope=1)
        c = await t.consumir(tenant="t1", recurso="tokens", cantidad=1, tope=1, periodo="2026-11")
        d = await t.consumir(tenant="t1", recurso="tokens", cantidad=1, tope=1)
        return a.permitido, b.permitido, c.permitido, d.permitido
    assert corre(caso()) == (True, True, True, False)


# --------------------------------------------------------------------------- #
# D-4 y validaciones                                                          #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("recurso", ["agentes", "agentes.concurrencia", "agentes.profundidad", "conexiones", "conexiones.puerto"])
def test_d4_no_se_puede_poner_tope_a_los_agentes_ni_a_las_conexiones(recurso):
    t, registros = _topes()
    with pytest.raises(TopeProhibido):
        _consumir(t, recurso=recurso, tope=5)
    assert t._almacen.llamadas == []                                     # ni siquiera se toco el almacen
    assert _consumir(t, recurso=recurso, tope=None).permitido            # medirlos si se puede


def test_las_palabras_sin_tope_son_las_de_la_decision_d4():
    assert PALABRAS_SIN_TOPE == ("agent", "subagent", "enjambre", "swarm", "conexion")


@pytest.mark.parametrize("kw", [
    {"cantidad": 0}, {"cantidad": -1}, {"cantidad": 1.5}, {"cantidad": True}, {"cantidad": "1"}, {"cantidad": 2 ** 60},
    {"tope": -1}, {"tope": 1.5}, {"tope": True}, {"tope": "3"}, {"tope": 2 ** 70},
    {"tenant": ""}, {"tenant": "a|b"}, {"tenant": "x" * 65}, {"tenant": "con espacio"},
    {"recurso": ""}, {"recurso": "Tokens"}, {"recurso": "a|b"}, {"recurso": "x" * 49},
    {"periodo": ""}, {"periodo": "a b"}, {"periodo": "x" * 33},
])
def test_los_argumentos_invalidos_se_rechazan_antes_de_tocar_el_almacen(kw):
    t, _ = _topes()
    with pytest.raises(ValueError):
        _consumir(t, **kw)
    assert t._almacen.llamadas == []


def test_el_contexto_no_puede_pisar_los_campos_de_la_anotacion():
    t, registros = _topes()
    _consumir(t, tope=0, evento="otro", decision="permitido", tenant="t1", run_id="run-1", usuario="u1")
    r = registros[-1]
    assert r["evento"] == "tope_superado" and r["decision"] == "denegado" and r["tenant"] == "t1"
    assert r["run_id"] == "run-1" and r["usuario"] == "u1"


# --------------------------------------------------------------------------- #
# el almacen falla                                                            #
# --------------------------------------------------------------------------- #

def test_con_tope_y_el_almacen_caido_se_niega_y_se_anota_y_se_avisa():
    t, registros = _topes(AlmacenMemoria("fallar"))
    r = _consumir(t, tope=100)
    assert not r.permitido and r.motivo == "almacen_no_disponible" and not r.medido
    assert [x["evento"] for x in registros] == ["tope_no_verificable"] and registros[0]["decision"] == "denegado"


def test_sin_tope_y_el_almacen_caido_se_deja_pasar_y_se_declara_que_no_se_midio():
    t, registros = _topes(AlmacenMemoria("fallar"))
    r = _consumir(t, tope=None)
    assert r.permitido and not r.medido and r.usado is None and r.motivo == "almacen_no_disponible"
    assert not [x for x in registros if x.get("decision") == "denegado"]


@pytest.mark.parametrize("tope,esperado", [(5, False), (None, True)])
def test_un_almacen_colgado_vence_por_el_plazo_y_no_cuelga_a_quien_consume(tope, esperado):
    t, _ = _topes(AlmacenMemoria("colgar"), plazo_s=0.2)
    t0 = time.monotonic()
    r = corre(asyncio.wait_for(t.consumir(tenant="t1", recurso="tokens", cantidad=1, tope=tope), 10))
    assert r.permitido is esperado and time.monotonic() - t0 < 2.0


def test_una_bitacora_que_falla_no_cambia_la_decision():
    def rota(registro):
        raise OSError("sin bitacora")
    registros = []
    t = Topes(AlmacenMemoria(), Bitacora(emisores=[rota]))
    assert not corre(t.consumir(tenant="t1", recurso="tokens", cantidad=1, tope=0)).permitido       # sigue negando
    assert corre(t.consumir(tenant="t1", recurso="tokens", cantidad=1, tope=None)).permitido          # sigue sin negar
    del registros


# --------------------------------------------------------------------------- #
# aviso                                                                       #
# --------------------------------------------------------------------------- #

def test_el_tope_alcanzado_y_el_consumo_sin_regla_llegan_al_aviso(tmp_path):
    ruta = tmp_path / "t.env"
    ruta.write_text("TELEGRAM_BOT_TOKEN=x\nTELEGRAM_CHAT_ID=1\n")
    ruta.chmod(0o600)
    enviados = []

    async def caso():
        cfg = ConfigAviso(creds=ruta, intervalo_s=0.05, rafaga=10)
        async with Avisador(cfg, Credenciales("x", "1"), enviar=enviados.append, host="h") as av:
            t = Topes(AlmacenMemoria(), Bitacora(emisores=[], observadores=[av]))
            await t.consumir(tenant="t1", recurso="tokens", cantidad=1, tope=None)
            await t.consumir(tenant="t1", recurso="gasto", cantidad=5, tope=1)
            await asyncio.sleep(0.2)
    corre(caso())
    assert any("SIN REGLA" in x and "no niega" in x for x in enviados)
    assert any("TOPE ALCANZADO" in x and "gasto" in x for x in enviados)


def test_el_resultado_es_inmutable():
    r = ResultadoTope(True, 1, None, True, "")
    with pytest.raises(Exception):
        r.permitido = False


# --------------------------------------------------------------------------- #
# auditoria de 0.3bc                                                          #
# --------------------------------------------------------------------------- #
from jax.faro.topes import ResultadoDesconocido  # noqa: E402


@pytest.mark.parametrize("recurso", ["agentes", "agente", "subagentes", "subagente", "agentes_concurrentes", "agentes-profundidad",
                                     "enjambre.agentes", "enjambre", "swarm.agents", "agents", "subagents", "agente.hijos",
                                     "conexiones", "conexion", "conexiones.puerto", "max_agentes", "tokens.por_agente"])
def test_minor4_el_guardia_de_d4_es_semantico_cualquier_recurso_de_agentes_queda_sin_tope(recurso):
    t, _ = _topes()
    with pytest.raises(TopeProhibido):
        _consumir(t, recurso=recurso, tope=5)
    assert _consumir(t, recurso=recurso, tope=None).permitido


@pytest.mark.parametrize("recurso", ["tokens", "gasto", "gasto.usd", "memoria.consultas", "paginas", "agencia", "tokens.entrada"])
def test_minor4_lo_que_no_es_de_agentes_sigue_pudiendo_tener_tope(recurso):
    t, _ = _topes()
    assert _consumir(t, recurso=recurso, tope=5).permitido


class _AlmacenDesconocido(AlmacenMemoria):
    """Imita un UPDATE que pudo confirmarse en la base y cuya respuesta nunca llego."""
    def __init__(self, confirmado):
        super().__init__()
        self.confirmado = confirmado

    async def sumar(self, clave, periodo, cantidad, tope):
        if self.confirmado:
            self.usado[(clave, periodo)] = self.usado.get((clave, periodo), 0) + cantidad
        raise ResultadoDesconocido("UPDATE enviado y sin respuesta")

    async def leer(self, clave, periodo):
        return self.usado.get((clave, periodo), 0)


def test_minor6_un_resultado_desconocido_es_un_estado_propio_y_con_tope_se_niega():
    t, registros = _topes(_AlmacenDesconocido(True))
    r = _consumir(t, tope=10, cantidad=3)
    assert not r.permitido and r.motivo == "resultado_desconocido" and not r.medido
    ev = [x for x in registros if x["evento"] == "tope_resultado_desconocido"]
    assert len(ev) == 1 and ev[0]["decision"] == "denegado" and ev[0]["cantidad"] == 3
    assert t.inciertos == {("t1|tokens", "total"): 3}


def test_minor6_sin_tope_un_resultado_desconocido_no_niega_pero_queda_incierto():
    t, registros = _topes(_AlmacenDesconocido(False))
    r = _consumir(t, tope=None, cantidad=2)
    assert r.permitido and r.motivo == "resultado_desconocido" and not r.medido
    assert t.inciertos == {("t1|tokens", "total"): 2}
    assert [x["decision"] for x in registros if x["evento"] == "tope_resultado_desconocido"] == ["permitido"]


@pytest.mark.parametrize("confirmado", [True, False])
def test_minor6_reconciliar_lee_el_contador_real_anota_y_limpia_lo_incierto(confirmado):
    t, registros = _topes(_AlmacenDesconocido(confirmado))
    _consumir(t, tope=10, cantidad=3)
    r = corre(t.reconciliar(tenant="t1", recurso="tokens"))
    assert r == {"usado": 3 if confirmado else 0, "incierto": 3}
    assert t.inciertos == {}
    ev = [x for x in registros if x["evento"] == "tope_reconciliado"]
    assert len(ev) == 1 and ev[0]["usado"] == r["usado"] and ev[0]["incierto"] == 3
    assert corre(t.reconciliar(tenant="t1", recurso="tokens")) == {"usado": r["usado"], "incierto": 0}


def test_minor6_un_fallo_normal_no_es_un_resultado_desconocido():
    t, _ = _topes(AlmacenMemoria("fallar"))
    r = _consumir(t, tope=5)
    assert r.motivo == "almacen_no_disponible" and t.inciertos == {}
