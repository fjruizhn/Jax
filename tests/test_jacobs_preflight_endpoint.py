"""POST /jacobs/preflight y el pre-vuelo obligatorio al crear (spec 2026-09-17
§4.7, §6.1, §8). Sin DB: el planificador, el pre-vuelo y el store van mockeados.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import asyncio
import os
from contextlib import ExitStack
from decimal import Decimal
from unittest.mock import AsyncMock, patch

os.environ["JAX_DB_NAME"] = "jax_memory_test"

import pytest  # noqa: E402
from fastapi import BackgroundTasks, HTTPException  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from jacobs import policy, routes  # noqa: E402
from jacobs.models import PipelineCreateRequest, Step, StepSpec  # noqa: E402
from jacobs.plan import PlanRejected, PlanViolation  # noqa: E402
from jacobs.prevuelo_reglas import CostoPaso, Veredicto, Violacion  # noqa: E402


def _veredicto(ok=True, usd="0.500000"):
    costo = CostoPaso(0, "jekyll", "deepseek-v4-flash", 1, 100, 8192, Decimal(usd), "acotado")
    violaciones = () if ok else (Violacion(0, "jekyll", "credencial_ausente", "sin credencial"),)
    return Veredicto(ok=ok, violaciones=violaciones, costo_max_usd=Decimal(usd),
                     pasos_costo=(costo,), sondeadas=())


def _pasos():
    return [Step(step_index=0, facet="jekyll", capability="research", input={"prompt": "p"})]


def _spec():
    return [StepSpec(facet="jekyll", capability="research", prompt="p")]


def _spec_n(n):
    return [StepSpec(facet="jekyll", capability="research", prompt="p") for _ in range(n)]


def _parches(pila, veredicto=None, prevuelo=None, build=None):
    m = {}
    for nombre in ("pipeline_create", "step_upsert", "event_append", "pipeline_update_status"):
        m[nombre] = pila.enter_context(patch.object(routes.store, nombre, AsyncMock()))
    m["pipeline_count_active"] = pila.enter_context(
        patch.object(routes.store, "pipeline_count_active", AsyncMock(return_value=0)))
    m["build"] = pila.enter_context(patch.object(
        routes._plan_builder, "build", build or AsyncMock(return_value=_pasos())))
    m["prevuelo"] = pila.enter_context(patch.object(
        routes, "prevuelo", prevuelo or AsyncMock(return_value=veredicto or _veredicto()), create=True))
    pila.enter_context(patch.object(policy, "check_kill_switch", return_value=False))
    # 2026-09-17: ya no hay candado. La reserva del cupo es un INSERT
    # condicionado y la conexión de la transacción sale del pool.
    m["conexion"] = ConexionFalsa()
    pila.enter_context(patch.object(routes.store, "conexion_del_pool", m["conexion"], create=True))
    m["reservar"] = pila.enter_context(
        patch.object(routes.cupo, "reservar_cupo", AsyncMock(return_value=True)))
    m["completar"] = pila.enter_context(
        patch.object(routes.cupo, "completar_reserva", AsyncMock(return_value=None)))
    m["soltar"] = pila.enter_context(
        patch.object(routes.cupo, "soltar_reserva", AsyncMock(return_value=1)))
    m["activos"] = pila.enter_context(
        patch.object(routes.cupo, "activos", AsyncMock(return_value=3)))
    m["transaccion"] = TransaccionFalsa(m["conexion"].orden)
    pila.enter_context(patch.object(routes.store, "transaccion", m["transaccion"], create=True))
    return m


class TransaccionFalsa:
    """Doble de store.transaccion (R38, fix round 1): la creación escribe en
    UNA transacción sobre la conexión del candado. Registra la conexión que
    recibió y si confirmó o se descartó."""

    def __init__(self, orden: list):
        self.orden, self.conexiones, self.confirmadas, self.descartadas = orden, [], 0, 0

    def __call__(self, conexion, estado=None):
        # estado (R41): EstadoDeTransaccion; este doble siempre confirma.
        self.conexiones.append(conexion)
        self.estado = estado
        return self

    async def __aenter__(self):
        self.orden.append("begin")
        return self.conexiones[-1]

    async def __aexit__(self, tipo, *exc):
        if tipo is None:
            if self.estado is not None:
                self.estado.enviando_commit = self.estado.confirmada = True
            self.confirmadas += 1
            self.orden.append("commit")
        else:
            self.descartadas += 1
            self.orden.append("descartada")
        return False


class ConexionFalsa:
    """Doble de `store.conexion_del_pool` (2026-09-17, sin candado): registra
    entradas y salidas y, con `falla`, se niega como un pool sin huecos."""

    def __init__(self, falla: Exception | None = None, orden: list | None = None):
        self.falla, self.orden = falla, orden if orden is not None else []
        self.entradas = self.salidas = 0

    def __call__(self, *a, **k):
        return self

    async def __aenter__(self):
        if self.falla is not None:
            raise self.falla
        self.entradas += 1
        self.orden.append("conexion")
        return "conexion-del-pool"

    async def __aexit__(self, *exc):
        self.salidas += 1
        self.orden.append("soltar")
        return False


class CandadoFalso:
    """Doble del candado entre procesos (F3): registra entradas y salidas y,
    con `falla`, se niega como un GET_LOCK que vence."""

    def __init__(self, falla: Exception | None = None, orden: list | None = None):
        self.falla, self.orden = falla, orden if orden is not None else []
        self.entradas = self.salidas = 0

    def __call__(self):
        return self

    async def __aenter__(self):
        if self.falla is not None:
            raise self.falla
        self.entradas += 1
        self.orden.append("candado")
        return "conexion-del-candado"

    async def __aexit__(self, *exc):
        self.salidas += 1
        self.orden.append("soltar")
        return False


def _crear(**cambios):
    datos = dict(name="t", objective="o", invoked_by="plataforma", mode="autonomous", steps=_spec())
    datos.update(cambios)
    return PipelineCreateRequest(**datos)


def test_preflight_devuelve_200_con_el_veredicto_aunque_rechace():
    with ExitStack() as pila:
        _parches(pila, veredicto=_veredicto(ok=False))
        r = asyncio.run(routes.preflight(routes.PreflightRequest(invoked_by="plataforma", steps=_spec())))
    assert r["ok"] is False
    assert r["violaciones"][0]["regla"] == "credencial_ausente"


def test_preflight_rechaza_un_invocador_no_valido_con_403():
    """Ruling R17 (fix round 1, Task 9): el rechazo de /preflight va como
    dict {code, detalle}, a diferencia de los rechazos de texto PREEXISTENTES
    de POST /jacobs/pipeline (que no se tocan en esta rama)."""
    with ExitStack() as pila:
        m = _parches(pila)
        with pytest.raises(HTTPException) as e:
            asyncio.run(routes.preflight(routes.PreflightRequest(invoked_by="Fernando", steps=_spec())))
    assert e.value.status_code == 403
    assert e.value.detail["code"] == "invocador_no_autorizado"
    assert "Fernando" in e.value.detail["detalle"]
    m["prevuelo"].assert_not_awaited()


def test_preflight_con_kill_switch_da_423_sin_planificar_ni_sondear():
    """I1 / Ruling R54: el pre-vuelo GASTA (la sonda paga y escribe
    axioma_usage), así que el kill switch también lo frena -- era el único
    camino que gastaba sin mirarlo. 423 con dict, como el resto de los
    rechazos de este endpoint (R17), después de validar invoked_by y antes de
    build(). Expected contra f24edc1: el pre-vuelo corre igual (200)."""
    with ExitStack() as pila:
        m = _parches(pila)
        pila.enter_context(patch.object(routes, "check_kill_switch", return_value=True))
        with pytest.raises(HTTPException) as e:
            asyncio.run(routes.preflight(routes.PreflightRequest(invoked_by="plataforma", steps=_spec())))
    assert e.value.status_code == 423
    assert e.value.detail["code"] == "kill_switch"
    assert "detenido" in e.value.detail["detalle"]
    m["build"].assert_not_awaited()
    m["prevuelo"].assert_not_awaited()


def test_preflight_con_invocador_invalido_y_kill_switch_responde_403_primero():
    """El orden del fix: invoked_by se valida ANTES del kill switch (un
    invocador no autorizado no aprende si Jacobs está frenado)."""
    with ExitStack() as pila:
        _parches(pila)
        pila.enter_context(patch.object(routes, "check_kill_switch", return_value=True))
        with pytest.raises(HTTPException) as e:
            asyncio.run(routes.preflight(routes.PreflightRequest(invoked_by="Fernando", steps=_spec())))
    assert e.value.status_code == 403


def test_preflight_rechaza_mas_de_20_pasos_con_422():
    """Ruling R17: el tope duro de 20 pasos también va como dict
    {code: "plan_rechazado", detalle}."""
    with ExitStack() as pila:
        m = _parches(pila)
        with pytest.raises(HTTPException) as e:
            asyncio.run(routes.preflight(routes.PreflightRequest(invoked_by="plataforma", steps=_spec_n(21))))
    assert e.value.status_code == 422
    assert e.value.detail["code"] == "plan_rechazado"
    assert "21" in e.value.detail["detalle"]
    m["build"].assert_not_awaited()
    m["prevuelo"].assert_not_awaited()


def test_preflight_no_escribe_nada_ni_con_plan_rechazado():
    with ExitStack() as pila:
        m = _parches(pila)
        asyncio.run(routes.preflight(routes.PreflightRequest(invoked_by="plataforma", steps=_spec())))
        for nombre in ("pipeline_create", "step_upsert", "event_append", "pipeline_update_status"):
            m[nombre].assert_not_awaited()
    rechazo = PlanRejected([PlanViolation(0, "jekyll", None, "research", "no va")])
    with ExitStack() as pila:
        m = _parches(pila, build=AsyncMock(side_effect=rechazo))
        with pytest.raises(HTTPException) as e:
            asyncio.run(routes.preflight(routes.PreflightRequest(invoked_by="plataforma", steps=_spec())))
        m["event_append"].assert_not_awaited()
    assert e.value.status_code == 422
    # Ruling R17: también dict acá, con el texto de PlanRejected en "detalle".
    assert e.value.detail["code"] == "plan_rechazado"
    assert e.value.detail["detalle"] == str(rechazo)


def test_preflight_con_la_base_caida_da_503():
    with ExitStack() as pila:
        _parches(pila, prevuelo=AsyncMock(side_effect=OSError("base caída")))
        with pytest.raises(HTTPException) as e:
            asyncio.run(routes.preflight(routes.PreflightRequest(invoked_by="plataforma", steps=_spec())))
    assert e.value.status_code == 503
    assert e.value.detail["code"] == "prevuelo_no_disponible"


def test_503_redacta_el_motivo_y_lo_recorta():
    """_prevuelo_o_503 pasa la excepción por recortar_redactado -- un secreto
    real (forma de key de Gemini) que llegara en el mensaje de una excepción
    NO puede aparecer en claro en el 503, y el motivo respeta el límite de
    300 caracteres que le pasa routes.py."""
    secreto = "AIzaSy" + "X" * 33
    with ExitStack() as pila:
        _parches(pila, prevuelo=AsyncMock(
            side_effect=RuntimeError(f"el proveedor devolvió {secreto} en el cuerpo")))
        with pytest.raises(HTTPException) as e:
            asyncio.run(routes.preflight(routes.PreflightRequest(invoked_by="plataforma", steps=_spec())))
    assert e.value.status_code == 503
    motivo = e.value.detail["motivo"]
    assert secreto not in motivo
    assert "***" in motivo
    assert len(motivo) <= 300


def test_crear_con_prevuelo_rechazado_da_422_sin_crear_y_deja_evento():
    with ExitStack() as pila:
        m = _parches(pila, veredicto=_veredicto(ok=False))
        with pytest.raises(HTTPException) as e:
            asyncio.run(routes.create_pipeline(_crear(), BackgroundTasks()))
        m["completar"].assert_not_awaited()
        tipos = [c.args[1] for c in m["event_append"].await_args_list]
    assert e.value.status_code == 422
    assert e.value.detail["code"] == "prevuelo_rechazado"
    assert tipos == ["PREVUELO_RECHAZADO"]


def test_evento_prevuelo_rechazado_tiene_el_mismo_contenido_que_el_422():
    """El payload que se guarda en el evento PREVUELO_RECHAZADO es EXACTAMENTE
    el mismo dict que se devuelve en el detail del 422 -- no dos cálculos que
    puedan divergir."""
    with ExitStack() as pila:
        m = _parches(pila, veredicto=_veredicto(ok=False))
        with pytest.raises(HTTPException) as e:
            asyncio.run(routes.create_pipeline(_crear(), BackgroundTasks()))
    eventos = [c.args[2] for c in m["event_append"].await_args_list if c.args[1] == "PREVUELO_RECHAZADO"]
    assert eventos == [e.value.detail]


def test_422_prevuelo_rechazado_incluye_pipeline_id_del_evento():
    """Ruling R19: el 422 lleva pipeline_id, y es el MISMO id con el que se
    escribió el evento PREVUELO_RECHAZADO."""
    with ExitStack() as pila:
        m = _parches(pila, veredicto=_veredicto(ok=False))
        with pytest.raises(HTTPException) as e:
            asyncio.run(routes.create_pipeline(_crear(), BackgroundTasks()))
    pipeline_id_evento = [
        c.args[0] for c in m["event_append"].await_args_list if c.args[1] == "PREVUELO_RECHAZADO"
    ][0]
    assert e.value.detail["pipeline_id"] == pipeline_id_evento


def test_prevuelo_recibe_identidad_y_objetivo_en_crear():
    """Sin esto, el evento de salud/uso de la sonda quedaría atribuido a
    nadie y el prompt armado no llevaría el objetivo real."""
    with ExitStack() as pila:
        m = _parches(pila)
        asyncio.run(routes.create_pipeline(
            _crear(objective="hacer cosas", user_id="u-1", tenant_id="t-1"), BackgroundTasks()))
    args, kwargs = m["prevuelo"].await_args
    assert args[1] == {"objective": "hacer cosas"}
    assert kwargs["user_id"] == "u-1"
    assert kwargs["tenant_id"] == "t-1"


def test_crear_con_costo_mayor_al_aceptado_da_409_sin_crear():
    with ExitStack() as pila:
        m = _parches(pila, veredicto=_veredicto(usd="0.500000"))
        with pytest.raises(HTTPException) as e:
            asyncio.run(routes.create_pipeline(
                _crear(costo_max_aceptado_usd=Decimal("0.40")), BackgroundTasks()))
        m["completar"].assert_not_awaited()
    assert e.value.status_code == 409
    assert e.value.detail["code"] == "costo_supera_lo_aceptado"
    assert e.value.detail["costo_max_usd"] == "0.500000"
    # Ruling R18: mismo grano de 6 decimales que el resto de los montos.
    assert e.value.detail["costo_max_aceptado_usd"] == "0.400000"


def test_crear_permite_cuando_el_costo_es_igual_al_aceptado():
    """El corte es `costo_max_usd > costo_max_aceptado_usd`, estricto: si
    coinciden, no es "supera lo aceptado" y la creación sigue."""
    with ExitStack() as pila:
        m = _parches(pila, veredicto=_veredicto(usd="0.500000"))
        r = asyncio.run(routes.create_pipeline(
            _crear(costo_max_aceptado_usd=Decimal("0.5")), BackgroundTasks()))
    m["completar"].assert_awaited_once()
    assert r["costo_max_usd"] == "0.500000"


def test_crear_costo_cero_no_queda_pelado_en_la_respuesta():
    """Ruling R18 (hallazgo propio, mismo bug que el eco del 409): el resumen
    de costo que alimenta PIPELINE_CREATED y el 200/dry_run pasa por el mismo
    formateo que Veredicto.to_dict() -- un veredicto sin costo real (p.ej.
    solo hyde/ollama) no puede devolver "0" pelado en ningún lado."""
    with ExitStack() as pila:
        m = _parches(pila, veredicto=_veredicto(usd="0"))
        r = asyncio.run(routes.create_pipeline(_crear(), BackgroundTasks()))
        creado = [c.args[2] for c in m["event_append"].await_args_list if c.args[1] == "PIPELINE_CREATED"]
    assert r["costo_max_usd"] == "0.000000"
    assert creado[0]["costo_max_usd"] == "0.000000"


def test_crear_ok_devuelve_el_costo_y_lo_deja_en_el_evento():
    bg = BackgroundTasks()
    with ExitStack() as pila:
        m = _parches(pila)
        r = asyncio.run(routes.create_pipeline(_crear(costo_max_aceptado_usd=Decimal("1")), bg))
        creado = [c.args[2] for c in m["event_append"].await_args_list if c.args[1] == "PIPELINE_CREATED"]
    assert r["costo_max_usd"] == "0.500000" and r["pasos_costo"][0]["paso"] == 0
    assert creado[0]["costo_max_usd"] == "0.500000"
    assert len(bg.tasks) == 1


def test_dry_run_tambien_corre_el_prevuelo():
    with ExitStack() as pila:
        m = _parches(pila)
        r = asyncio.run(routes.create_pipeline(_crear(mode="dry_run"), BackgroundTasks()))
        m["prevuelo"].assert_awaited_once()
    assert r["costo_max_usd"] == "0.500000"


def test_el_prevuelo_corre_despues_de_build_y_antes_de_crear():
    orden = []

    async def build(**kw):
        orden.append("build")
        return _pasos()

    async def prevuelo(*a, **kw):
        orden.append("prevuelo")
        return _veredicto()

    with ExitStack() as pila:
        m = _parches(pila, build=AsyncMock(side_effect=build), prevuelo=AsyncMock(side_effect=prevuelo))
        m["completar"].side_effect = lambda p, **kw: orden.append("crear")
        asyncio.run(routes.create_pipeline(_crear(), BackgroundTasks()))
    assert orden == ["build", "prevuelo", "crear"]


def test_crear_con_la_base_caida_da_503_sin_crear():
    with ExitStack() as pila:
        m = _parches(pila, prevuelo=AsyncMock(side_effect=OSError("base caída")))
        with pytest.raises(HTTPException) as e:
            asyncio.run(routes.create_pipeline(_crear(), BackgroundTasks()))
        m["completar"].assert_not_awaited()
    assert e.value.status_code == 503


def test_costo_max_aceptado_negativo_es_invalido():
    with pytest.raises(ValidationError):
        _crear(costo_max_aceptado_usd=Decimal("-1"))


# ---------------------------------------------------------------------------
# F3 (ola final, Ruling R31): el cupo de MAX_PARALLEL_PIPELINES se vuelve a
# contar y se escribe dentro del candado con nombre de MariaDB, que cruza
# procesos (el CLI de continuar corre en otro). El asyncio.Lock no alcanza.
# ---------------------------------------------------------------------------

def test_el_cupo_lleno_lo_rechaza_la_reserva_y_no_se_planifica():
    """2026-09-17: el cupo lo decide la RESERVA, que cuenta e inserta en la
    misma sentencia. Con el cupo lleno devuelve False, y la creación se corta
    ANTES de planificar y sondear -- no se gasta un LLM para un 422."""
    with ExitStack() as pila:
        m = _parches(pila)
        m["reservar"].return_value = False
        with pytest.raises(HTTPException) as e:
            asyncio.run(routes.create_pipeline(_crear(), BackgroundTasks()))
        m["completar"].assert_not_awaited()
        m["build"].assert_not_awaited()
        m["soltar"].assert_not_awaited()   # no hay reserva que soltar
    assert e.value.status_code == 422
    assert "Límite duro" in e.value.detail


def test_crear_reserva_antes_de_planificar_y_completa_en_la_transaccion():
    orden = []
    with ExitStack() as pila:
        m = _parches(pila)
        m["conexion"].orden = orden
        m["reservar"].side_effect = lambda p, **kw: orden.append("reservar") or True
        m["build"].side_effect = lambda **kw: orden.append("build") or _pasos()
        m["completar"].side_effect = lambda p, **kw: orden.append("completar")
        m["event_append"].side_effect = lambda *a, **kw: orden.append(a[1])
        asyncio.run(routes.create_pipeline(_crear(), BackgroundTasks()))
    # Sin "contar" al principio: crear ya no hace un recuento suelto -- la
    # reserva cuenta y escribe en la misma sentencia, y es la que decide.
    assert orden == ["reservar", "build", "conexion", "completar",
                     "PIPELINE_CREATED", "soltar"]


def test_si_la_escritura_falla_se_suelta_la_reserva_y_da_503():
    """La reserva ocupa cupo desde antes de planificar: si la transacción no
    llega a confirmar, hay que devolver el lugar o el cupo queda comido por un
    pipeline que nunca existió."""
    with ExitStack() as pila:
        m = _parches(pila)
        m["completar"].side_effect = OSError("base caída a mitad")
        with pytest.raises(HTTPException) as e:
            asyncio.run(routes.create_pipeline(_crear(), BackgroundTasks()))
        m["soltar"].assert_awaited_once()
    assert e.value.status_code == 503
    assert e.value.detail["code"] == "prevuelo_no_disponible"


def test_un_commit_incierto_no_suelta_la_reserva():
    """R41 con reserva: si el COMMIT salió y la respuesta no llegó, la fila
    PUEDE ser un pipeline completo. Soltarla borraría un pipeline real."""
    class TxIncierta(TransaccionFalsa):
        def __call__(self, conexion, estado=None):
            self.estado = estado
            return self

        async def __aenter__(self):
            return "tx"

        async def __aexit__(self, *exc):
            self.estado.enviando_commit = True
            raise OSError("se cortó confirmando")

    with ExitStack() as pila:
        m = _parches(pila)
        pila.enter_context(patch.object(routes.store, "transaccion", TxIncierta([])))
        with pytest.raises(HTTPException) as e:
            asyncio.run(routes.create_pipeline(_crear(), BackgroundTasks()))
        m["soltar"].assert_not_awaited()
    assert e.value.status_code == 503
    assert "puede existir" in e.value.detail["detalle"]
