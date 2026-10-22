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

from pathlib import Path

import asyncio
import time

import pytest

from jax.faro.aviso import Avisador, ConfigAviso, Credenciales
from jax.faro.bitacora import Bitacora
from jax.faro.catalogo_topes import es_de_catalogo, recursos_del_catalogo
from jax.faro.topes import AlmacenTopes, ResultadoTope, TopeProhibido, Topes

RAIZ = Path(__file__).resolve().parents[1]
# r7, MAJOR-1: el catalogo de las pruebas sale de un PIN de prueba (repo git
# minimo + snapshot), como en produccion — nunca de bytes del disco suelto
from tests.policy.catalogo_pin import catalogo_del_pin
CATALOGO = catalogo_del_pin()
from tests._faro_falsos import AlmacenMemoria
from tests._faro_utils import corre


def _topes(almacen=None, **kw):
    registros = []
    bit = Bitacora(emisores=[registros.append], observadores=kw.pop("observadores", ()))
    kw.setdefault("catalogo", CATALOGO)          # B-3/r7: sellado por el snapshot del pin de prueba
    return Topes(almacen if almacen is not None else AlmacenMemoria(), bit, **kw), registros


def _consumir(t, **kw):
    base = dict(tenant="t1", recurso="tokens_costo.tokens", cantidad=1, tope=None)
    base.update(kw)
    return corre(t.consumir(**base))


# --------------------------------------------------------------------------- #
# sin regla de tope: se mide, se avisa, no se niega                           #
# --------------------------------------------------------------------------- #

def test_sin_regla_de_tope_nunca_se_niega_y_se_mide():
    t, registros = _topes()

    async def caso():
        return [await t.consumir(tenant="t1", recurso="tokens_costo.tokens", cantidad=1000, tope=None) for _ in range(50)]
    rs = corre(caso())
    assert all(r.permitido and r.medido and r.tope is None for r in rs)
    assert [r.usado for r in rs][-1] == 50_000
    assert not [x for x in registros if x.get("decision") == "denegado"]


def test_el_consumo_sin_regla_se_anota_y_se_avisa_una_vez_por_tenant_recurso_y_periodo():
    t, registros = _topes()

    async def caso():
        for _ in range(5):
            await t.consumir(tenant="t1", recurso="tokens_costo.tokens", cantidad=1, tope=None)
        await t.consumir(tenant="t2", recurso="tokens_costo.tokens", cantidad=1, tope=None)
        await t.consumir(tenant="t1", recurso="monto_dinero.hnl", cantidad=1, tope=None)
        await t.consumir(tenant="t1", recurso="tokens_costo.tokens", cantidad=1, tope=None, periodo="2026-10")
    corre(caso())
    sin_regla = [r for r in registros if r["evento"] == "tope_sin_regla"]
    assert len(sin_regla) == 4
    assert {(r["tenant"], r["recurso"], r["periodo"]) for r in sin_regla} == {
        ("t1", "tokens_costo.tokens", "total"), ("t2", "tokens_costo.tokens", "total"), ("t1", "monto_dinero.hnl", "total"), ("t1", "tokens_costo.tokens", "2026-10")}
    assert all(r["decision"] == "permitido" and "regla" in r["motivo"] for r in sin_regla)


def test_resultado_y_bitacora_atan_el_consumo_al_catalogo_sellado_sin_permitir_suplantarlo():
    """Si se omite ``catalogo_oid`` de base o se deja entrar desde ``contexto``,
    un registro durable puede parecer atribuido a otro catálogo."""
    t, registros = _topes()
    resultado = corre(t.consumir(
        tenant="t1", recurso="tokens_costo.tokens", cantidad=1, tope=None,
        catalogo_oid="oid-forjado",
    ))

    assert resultado.catalogo_oid == CATALOGO.oid_pin
    evento = next(r for r in registros if r["evento"] == "tope_sin_regla")
    assert evento["catalogo_oid"] == CATALOGO.oid_pin
    assert evento["catalogo_oid"] != "oid-forjado"


def test_el_contrato_de_almacen_declara_lectura_para_reconciliar():
    assert callable(getattr(AlmacenTopes, "leer", None))


def test_reconciliacion_durable_conserva_el_oid_del_catalogo_sellado():
    """La reconciliación también es evidencia durable del contador y no puede
    perder la procedencia que llevaba el resultado original."""
    t, registros = _topes()
    corre(t.reconciliar(tenant="t1", recurso="tokens_costo.tokens"))

    evento = next(r for r in registros if r["evento"] == "tope_reconciliado")
    assert evento["catalogo_oid"] == CATALOGO.oid_pin


# --------------------------------------------------------------------------- #
# con tope: falla cerrado al llegar                                           #
# --------------------------------------------------------------------------- #

def test_con_tope_se_permite_hasta_el_tope_y_el_siguiente_se_niega_sin_pasarse():
    t, registros = _topes()

    async def caso():
        return [await t.consumir(tenant="t1", recurso="tokens_costo.tokens", cantidad=1, tope=3, run_id="run-9") for _ in range(5)]
    rs = corre(caso())
    assert [r.permitido for r in rs] == [True, True, True, False, False]
    assert [r.usado for r in rs] == [1, 2, 3, 3, 3]                      # lo negado no suma
    assert rs[3].motivo == "tope_alcanzado" and rs[3].tope == 3
    negados = [r for r in registros if r["evento"] == "tope_superado"]
    assert len(negados) == 2 and all(r["decision"] == "denegado" and r["run_id"] == "run-9" and r["tope"] == 3 for r in negados)


def test_la_cantidad_exacta_hasta_el_tope_cabe_y_una_unidad_mas_no():
    t, _ = _topes()

    async def caso():
        a = await t.consumir(tenant="t1", recurso="monto_dinero.hnl", cantidad=6, tope=10)
        b = await t.consumir(tenant="t1", recurso="monto_dinero.hnl", cantidad=6, tope=10)     # 12 > 10
        c = await t.consumir(tenant="t1", recurso="monto_dinero.hnl", cantidad=4, tope=10)     # 10 <= 10
        d = await t.consumir(tenant="t1", recurso="monto_dinero.hnl", cantidad=1, tope=10)
        return a, b, c, d
    a, b, c, d = corre(caso())
    assert (a.permitido, b.permitido, c.permitido, d.permitido) == (True, False, True, False) and c.usado == 10


def test_un_tope_cero_lo_niega_todo():
    t, _ = _topes()
    assert not _consumir(t, tope=0).permitido


def test_cada_tenant_y_cada_periodo_tiene_su_propia_cuenta():
    t, _ = _topes()

    async def caso():
        a = await t.consumir(tenant="t1", recurso="tokens_costo.tokens", cantidad=1, tope=1)
        b = await t.consumir(tenant="t2", recurso="tokens_costo.tokens", cantidad=1, tope=1)
        c = await t.consumir(tenant="t1", recurso="tokens_costo.tokens", cantidad=1, tope=1, periodo="2026-11")
        d = await t.consumir(tenant="t1", recurso="tokens_costo.tokens", cantidad=1, tope=1)
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


def test_el_catalogo_es_solo_de_actos_y_dinero_d4():
    # Decision de Fernando (2026-10-06): monto de dinero, actos externos,
    # frecuencia, duracion y tokens/costo. La lista negra por palabras murio:
    # fuera del catalogo no hay tope, punto.
    assert sorted(CATALOGO) == ["actos_externos", "duracion", "frecuencia",
                                "monto_dinero", "tokens_costo"]
    assert es_de_catalogo("actos_externos.mensajes", CATALOGO)
    assert not es_de_catalogo("actos_externos.sockets", CATALOGO)


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
    r = corre(asyncio.wait_for(t.consumir(tenant="t1", recurso="tokens_costo.tokens", cantidad=1, tope=tope), 10))
    assert r.permitido is esperado and time.monotonic() - t0 < 2.0


def test_una_bitacora_que_falla_no_cambia_la_decision():
    def rota(registro):
        raise OSError("sin bitacora")
    registros = []
    t = Topes(AlmacenMemoria(), Bitacora(emisores=[rota]), catalogo=CATALOGO)
    assert not corre(t.consumir(tenant="t1", recurso="tokens_costo.tokens", cantidad=1, tope=0)).permitido       # sigue negando
    assert corre(t.consumir(tenant="t1", recurso="tokens_costo.tokens", cantidad=1, tope=None)).permitido          # sigue sin negar
    del registros


def test_r6_el_runtime_exige_el_catalogo_sellado_del_snapshot():
    """r6: un Mapping cualquiera —dict, MappingProxyType, lo que sea— no abre el
    conteo con tope: solo el CatalogoTopes sellado que salio del snapshot
    verificado del pin. Fabricar el tipo a mano, tampoco — ni con el OID (r7)."""
    from types import MappingProxyType
    from jax.faro.catalogo_topes import CatalogoTopes, CatalogoTopesInvalido
    from jax.faro.config import ConfigFaroInvalida
    misma_forma = {c: list(s) for c, s in CATALOGO.items()}
    for falso in (misma_forma, MappingProxyType(misma_forma)):
        with pytest.raises(ConfigFaroInvalida):
            Topes(AlmacenMemoria(), Bitacora(emisores=[]), catalogo=falso)
    with pytest.raises(CatalogoTopesInvalido):
        es_de_catalogo("actos_externos.mensajes", MappingProxyType(misma_forma))
    with pytest.raises(CatalogoTopesInvalido):
        CatalogoTopes(misma_forma, oid_pin=CATALOGO.oid_pin)     # r7: ni con el OID del pin
    t, _ = _topes()   # con el sellado del pin de prueba: normal
    assert t.catalogo_oid == CATALOGO.oid_pin                    # r7, MAJOR-1: queda registrado


@pytest.mark.parametrize("recurso", ["conexiones.usd", "workers.tokens", "agentes.segundos"])
def test_r7_recurso_de_clase_prohibida_con_subid_ajeno_valido_niega(recurso):
    """MAJOR-3 / mutante AUD2: el subid existe (en OTRA clase) y el recurso
    parece del catalogo — no lo es: la clase prohibida nunca topea. Con el
    mutante (buscar el subid en CUALQUIER clase), esto pasaria y la prueba muere."""
    t, _ = _topes()
    assert not es_de_catalogo(recurso, CATALOGO)
    with pytest.raises(TopeProhibido):
        corre(t.consumir(tenant="t1", recurso=recurso, cantidad=1, tope=5))


def test_r7_topes_no_acepta_catalogo_de_bytes_sueltos():
    """MAJOR-1: la validacion privada devuelve las clases SIN sellar; Topes no
    las acepta. Solo el snapshot del pin emite CatalogoTopes."""
    import subprocess
    from jax.faro.config import ConfigFaroInvalida
    import jax.faro.catalogo_topes as CT
    suelto = CT._cargar_catalogo_bytes(
        subprocess.run(["git", "show", "HEAD:policy/faro/catalogo-topes.json"],
                       capture_output=True, check=True, cwd=RAIZ).stdout)
    assert type(suelto) is dict
    with pytest.raises(ConfigFaroInvalida):
        Topes(AlmacenMemoria(), Bitacora(emisores=[]), catalogo=suelto)  # type: ignore[arg-type]


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
            t = Topes(AlmacenMemoria(), Bitacora(emisores=[], observadores=[av]), catalogo=CATALOGO)
            await t.consumir(tenant="t1", recurso="tokens_costo.tokens", cantidad=1, tope=None)
            await t.consumir(tenant="t1", recurso="monto_dinero.hnl", cantidad=5, tope=1)
            await asyncio.sleep(0.2)
    corre(caso())
    assert any("SIN REGLA" in x and "no niega" in x for x in enviados)
    assert any("TOPE ALCANZADO" in x and "monto_dinero.hnl" in x for x in enviados)


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


@pytest.mark.parametrize("recurso", list(recursos_del_catalogo(CATALOGO)))
def test_r3_solo_el_catalogo_topea_la_decision_reemplazo_la_lista_negra(recurso):
    # Decision de Fernando (2026-10-06): topea SOLO el catalogo de actos y
    # dinero. La premisa vieja («todo lo que no es de agentes topea») murio.
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
    assert t.inciertos == {("t1|tokens_costo.tokens", "total"): 3}


def test_minor6_sin_tope_un_resultado_desconocido_no_niega_pero_queda_incierto():
    t, registros = _topes(_AlmacenDesconocido(False))
    r = _consumir(t, tope=None, cantidad=2)
    assert r.permitido and r.motivo == "resultado_desconocido" and not r.medido
    assert t.inciertos == {("t1|tokens_costo.tokens", "total"): 2}
    assert [x["decision"] for x in registros if x["evento"] == "tope_resultado_desconocido"] == ["permitido"]


@pytest.mark.parametrize("confirmado", [True, False])
def test_minor6_reconciliar_lee_el_contador_real_anota_y_limpia_lo_incierto(confirmado):
    t, registros = _topes(_AlmacenDesconocido(confirmado))
    _consumir(t, tope=10, cantidad=3)
    r = corre(t.reconciliar(tenant="t1", recurso="tokens_costo.tokens"))
    assert r == {"usado": 3 if confirmado else 0, "incierto": 3}
    assert t.inciertos == {}
    ev = [x for x in registros if x["evento"] == "tope_reconciliado"]
    assert len(ev) == 1 and ev[0]["usado"] == r["usado"] and ev[0]["incierto"] == 3
    assert corre(t.reconciliar(tenant="t1", recurso="tokens_costo.tokens")) == {"usado": r["usado"], "incierto": 0}


def test_minor6_un_fallo_normal_no_es_un_resultado_desconocido():
    t, _ = _topes(AlmacenMemoria("fallar"))
    r = _consumir(t, tope=5)
    assert r.motivo == "almacen_no_disponible" and t.inciertos == {}


def test_minor6_si_la_lectura_de_la_reconciliacion_falla_lo_incierto_se_conserva():
    almacen = _AlmacenDesconocido(True)
    t, _ = _topes(almacen)
    _consumir(t, tope=10, cantidad=3)

    async def leer_roto(clave, periodo):
        raise OSError("sin lectura")
    almacen.leer = leer_roto
    with pytest.raises(OSError):
        corre(t.reconciliar(tenant="t1", recurso="tokens_costo.tokens"))
    assert t.inciertos == {("t1|tokens_costo.tokens", "total"): 3}


# --------------------------------------------------------------------------- #
# Catalogo cerrado (R-4, decision de Fernando 2026-10-06; jax#370 r3)         #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("recurso", [
    "subprocess.spawn", "parallel.calls", "concurrent.requests", "multithread.jobs",
    "sockets.abiertos", "conns.db", "tasks.systemd", "subagentes",
    "multiagente.lanzados", "pids", "jobs.cola", "fork.hijos",
    "sesiones.mcp", "subprocesos", "multiagente.lanzados",
])
def test_r3_fabricados_e_infraestructura_nunca_llevan_tope(recurso):
    t, registros = _topes()
    with pytest.raises(TopeProhibido):
        _consumir(t, recurso=recurso, tope=5)
    assert t._almacen.llamadas == []
    assert _consumir(t, recurso=recurso, tope=None).permitido    # medir si se puede


@pytest.mark.parametrize("recurso", recursos_del_catalogo(CATALOGO))
def test_r3_todo_el_catalogo_lleva_tope_y_mide(recurso):
    t, registros = _topes()
    assert _consumir(t, recurso=recurso, tope=5).permitido       # del catalogo: topea


def test_r5_sin_catalogo_no_se_topea_nada():
    # B-3: el catalogo llega del snapshot evaluado; sin el, TODO tope niega
    # (TopeProhibido) y medir sigue permitido.
    t, _ = _topes()
    t._catalogo = None                                       # como si viniera sin catalogo
    with pytest.raises(TopeProhibido):
        _consumir(t, recurso="actos_externos.mensajes", tope=5)
    assert _consumir(t, recurso="actos_externos.mensajes", tope=None).permitido
