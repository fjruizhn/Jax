"""Continuar un pipeline abortado o vencido (spec 2026-09-17 §5.1-§5.2;
desvíos 7, 10 y 11 del plan). Sin DB: store, pre-vuelo y gobernanza mockeados.
Las refs inline se leen de verdad; la ilegible es un artifact que no existe.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import asyncio
import time
import os
from contextlib import ExitStack, asynccontextmanager, contextmanager
from decimal import Decimal
from unittest.mock import AsyncMock, patch

os.environ["JAX_DB_NAME"] = "jax_memory_test"

import pytest  # noqa: E402

from jacobs import continuar  # noqa: E402
from jacobs.models import Pipeline, PipelineStatus, Step, StepStatus  # noqa: E402
from jacobs.prevuelo_reglas import CostoPaso, Veredicto, Violacion  # noqa: E402

_NADA = object()
_REF_A = 'inline:{"result": "a"}'
_REF_B = 'inline:{"result": "b"}'
_ILEGIBLE = "artifact://jacobs/no-existe/tampoco/output.json"


def _pasos(capability_2="research", facet_1="jekyll"):
    pasos = []
    for i in range(3):
        pasos.append(Step(
            step_id=f"s{i}", pipeline_id="p1", step_index=i,
            facet=facet_1 if i == 1 else "jekyll",
            capability=capability_2 if i == 2 else "research",
            input={"prompt": f"p{i}"}, depends_on=[i - 1] if i else [],
            status=StepStatus.completed if i < 2 else StepStatus.failed,
            error=None if i < 2 else "cortado",
            started_at=1.0, finished_at=2.0,
            output_ref=_REF_A if i == 0 else (_REF_B if i == 1 else None),
        ))
    return pasos


def _abortado(status=PipelineStatus.aborted, contexto=None, epoca=2):
    return Pipeline(
        pipeline_id="p1", name="t", invoked_by="plataforma", user_id="7", tenant_id="1",
        mode="autonomous", status=status, run_epoch=epoca,
        context=contexto if contexto is not None else
        {"objective": "o", "step_0_ref": _REF_A, "step_1_ref": _REF_B},
    )


def _veredicto(ok=True, usd="0.250000"):
    costo = CostoPaso(2, "jekyll", "deepseek-v4-flash", 1, 100, 8192, Decimal(usd), "acotado")
    violaciones = () if ok else (Violacion(2, "jekyll", "faceta_caida", "timeout de sonda (20s)"),)
    return Veredicto(ok=ok, violaciones=violaciones, costo_max_usd=Decimal(usd),
                     pasos_costo=(costo,), sondeadas=())


@contextmanager
def _entorno(pipeline=_NADA, pasos=None, activos=0, veredicto=None, transaccion=3, kill=False):
    with ExitStack() as pila:
        m = {}
        m["get"] = pila.enter_context(patch.object(continuar.store, "pipeline_get", AsyncMock(
            return_value=_abortado() if pipeline is _NADA else pipeline)))
        m["pasos"] = pila.enter_context(patch.object(continuar.store, "steps_by_pipeline", AsyncMock(
            return_value=pasos if pasos is not None else _pasos())))
        m["activos"] = pila.enter_context(patch.object(
            continuar.store, "pipeline_count_active", AsyncMock(return_value=activos)))
        m["tx"] = pila.enter_context(patch.object(
            continuar.store, "continuar_transaccion", AsyncMock(return_value=transaccion)))
        m["evento"] = pila.enter_context(patch.object(continuar.store, "event_append", AsyncMock()))
        m["prevuelo"] = pila.enter_context(patch.object(
            continuar, "prevuelo", AsyncMock(return_value=veredicto or _veredicto())))
        m["capacidades"] = pila.enter_context(patch.object(
            continuar, "_validate_plan_capabilities", AsyncMock()))
        pila.enter_context(patch.object(continuar, "check_kill_switch", return_value=kill))
        m["candado"] = _CandadoFalso()
        pila.enter_context(patch.object(continuar.store, "candado_de_activos", m["candado"], create=True))
        yield m


class _CandadoFalso:
    def __init__(self, falla: Exception | None = None):
        self.falla, self.orden = falla, []
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


def _continuar(**kw):
    return asyncio.run(continuar.continuar("p1", kw.pop("invoked_by", "plataforma"), **kw))


def _rechazo(**kw):
    with pytest.raises(continuar.ContinuarRechazado) as e:
        _continuar(**kw)
    return e.value


def test_un_abortado_se_continua_reusando_lo_que_tiene_ref():
    with _entorno() as m:
        r, p = _continuar()
        args = m["tx"].await_args.args
    assert r["pasos_reusados"] == [0, 1] and r["pasos_a_correr"] == [2]
    assert r["run_epoch"] == 3 and p.run_epoch == 3 and p.status == PipelineStatus.running
    assert args[:3] == ("p1", 2, PipelineStatus.aborted)
    assert [s.step_index for s in args[3]] == [2]


def test_un_vencido_tambien_se_continua():
    with _entorno(pipeline=_abortado(status=PipelineStatus.expired)) as m:
        r, _ = _continuar()
        assert m["tx"].await_args.args[2] == PipelineStatus.expired
    assert r["status"] == "running"


def test_los_demas_estados_dan_409_con_el_status():
    for status in (PipelineStatus.completed, PipelineStatus.running, PipelineStatus.pending,
                   PipelineStatus.interrupted, PipelineStatus.failed):
        with _entorno(pipeline=_abortado(status=status)) as m:
            e = _rechazo()
            m["tx"].assert_not_awaited()
        assert (e.status_code, e.code, e.cuerpo()["status"]) == (409, "estado_no_continuable", status.value)


def test_invocador_distinto_de_plataforma_403():
    with _entorno():
        e = _rechazo(invoked_by="ada")
    assert (e.status_code, e.code) == (403, "invocador_no_autorizado")


def test_pipeline_inexistente_404():
    with _entorno(pipeline=None):
        e = _rechazo()
    assert (e.status_code, e.code) == (404, "no_existe")


def test_kill_switch_423():
    with _entorno(kill=True) as m:
        e = _rechazo()
        m["tx"].assert_not_awaited()
    assert (e.status_code, e.code) == (423, "kill_switch")


def test_ref_ilegible_se_rehace_y_sale_del_contexto():
    contexto = {"objective": "o", "step_0_ref": _REF_A, "step_1_ref": _ILEGIBLE}
    with _entorno(pipeline=_abortado(contexto=contexto)) as m:
        r, _ = _continuar()
        ctx = m["tx"].await_args.args[5]
    assert r["pasos_reusados"] == [0] and r["pasos_a_correr"] == [1, 2]
    assert "step_1_ref" not in ctx and ctx["step_0_ref"] == _REF_A


def test_los_pasos_a_correr_quedan_pendientes_limpios():
    with _entorno() as m:
        _continuar()
        paso = m["tx"].await_args.args[3][0]
    assert (paso.status, paso.error, paso.started_at, paso.finished_at, paso.output_ref) == (
        StepStatus.pending, None, None, None, None)


def test_reasignar_cambia_la_faceta_y_recalcula_el_motor():
    with _entorno() as m:
        _continuar(reasignar={"2": "kimi"})
        paso = m["tx"].await_args.args[3][0]
        evento = m["tx"].await_args.kwargs["evento_payload"]
    assert (paso.facet, paso.motor) == ("kimi", "kimi")
    assert evento["reasignados"] == {"2": {"de": "jekyll", "a": "kimi"}}
    with _entorno() as m:
        _continuar(reasignar={"2": "thot"})
        paso = m["tx"].await_args.args[3][0]
    assert (paso.facet, paso.motor) == ("thot", None)


def test_reasignar_un_paso_reusado_da_422():
    with _entorno() as m:
        e = _rechazo(reasignar={"0": "thot"})
        m["tx"].assert_not_awaited()
        m["prevuelo"].assert_not_awaited()
    assert (e.status_code, e.code) == (422, "reasignacion_invalida")


def test_reasignar_a_una_faceta_desconocida_da_422():
    with _entorno() as m:
        e = _rechazo(reasignar={"2": "gpt"})
        m["tx"].assert_not_awaited()
    assert (e.status_code, e.code) == (422, "reasignacion_invalida")
    assert "gpt" in e.cuerpo()["detalle"][0]["motivo"]


def test_reasignacion_que_rompe_el_cleanroom_da_422_sin_escribir():
    pasos = _pasos(capability_2="critique", facet_1="thot")
    with _entorno(pasos=pasos) as m:
        e = _rechazo(reasignar={"2": "thot"})
        m["tx"].assert_not_awaited()
        m["evento"].assert_not_awaited()
    assert (e.status_code, e.code) == (422, "reasignacion_invalida")
    assert "cleanroom" in e.cuerpo()["detalle"][0]["reason"]


def test_limite_de_activos_da_429():
    with _entorno(activos=3) as m:
        e = _rechazo()
        m["tx"].assert_not_awaited()
    assert (e.status_code, e.code) == (429, "limite_de_activos")


def test_prevuelo_rechazado_da_422_sin_escribir_el_pipeline():
    with _entorno(veredicto=_veredicto(ok=False)) as m:
        e = _rechazo()
        m["tx"].assert_not_awaited()
        tipos = [c.args[1] for c in m["evento"].await_args_list]
    assert (e.status_code, e.code) == (422, "prevuelo_rechazado")
    assert tipos == ["PREVUELO_RECHAZADO"]


def test_costo_mayor_al_aceptado_da_409_sin_escribir():
    with _entorno() as m:
        e = _rechazo(costo_max_aceptado_usd=Decimal("0.10"))
        m["tx"].assert_not_awaited()
    assert (e.status_code, e.code) == (409, "costo_supera_lo_aceptado")
    assert e.cuerpo()["costo_max_aceptado_usd"] == "0.100000"


def test_el_costo_cero_no_queda_pelado():
    # R18: todo monto que sale de Jacobs es texto con 6 decimales -- un
    # costo_max_usd sin evaluables (Decimal(0), sin cuantizar) no puede
    # llegar como "0" pelado ni en la respuesta ni en el evento.
    veredicto = Veredicto(ok=True, violaciones=(), costo_max_usd=Decimal(0), pasos_costo=(), sondeadas=())
    with _entorno(veredicto=veredicto) as m:
        r, _ = _continuar()
        evento = m["tx"].await_args.kwargs["evento_payload"]
    assert r["costo_max_usd"] == "0.000000"
    assert evento["costo_max_usd"] == "0.000000"


def test_si_otro_pedido_gano_la_transaccion_da_409():
    with _entorno(transaccion=None) as m:
        e = _rechazo()
        tipos = [c.args[1] for c in m["evento"].await_args_list]
    assert (e.status_code, e.code) == (409, "estado_no_continuable")
    # PIPELINE_CONTINUED nunca pasa por event_append (R22: va dentro de la
    # transacción); lo que prueba que no ganó es que la ÚNICA escritura
    # intentada fue la propia transacción, que devolvió None.
    assert tipos == []
    m["tx"].assert_awaited_once()


def test_el_evento_continued_lleva_todo_el_detalle():
    with _entorno() as m:
        _continuar()
        evento = m["tx"].await_args.kwargs["evento_payload"]
    assert evento == {"by": "plataforma", "from_status": "aborted", "run_epoch": 3,
                      "pasos_a_correr": [2], "pasos_reusados": [0, 1], "reasignados": {},
                      "costo_max_usd": "0.250000"}


def test_la_marca_de_hyde_del_paso_a_rehacer_se_quita():
    contexto = {"objective": "o", "step_0_ref": _REF_A, "step_1_ref": _REF_B,
                "hyde_approved_s0": True, "hyde_approved_s2": True}
    with _entorno(pipeline=_abortado(contexto=contexto)) as m:
        _continuar()
        ctx = m["tx"].await_args.args[5]
    assert "hyde_approved_s2" not in ctx and ctx["hyde_approved_s0"] is True


def test_el_prevuelo_solo_mira_los_pendientes():
    with _entorno() as m:
        _continuar()
        kw = m["prevuelo"].await_args.kwargs
    assert kw["pendientes"] == {2}
    assert (kw["user_id"], kw["tenant_id"]) == ("7", "1")


def test_el_prevuelo_no_recibe_refs_de_pasos_a_rehacer():
    # Requisito (b): el contexto que se le pasa a prevuelo() es el MISMO
    # (podado) que se le pasa a la transacción -- no el crudo de
    # pipeline.context. Si _prevuelo_de volviera a pasar pipeline.context,
    # este assert vería la ref ilegible de step_1 todavía adentro.
    contexto = {"objective": "o", "step_0_ref": _REF_A, "step_1_ref": _ILEGIBLE}
    with _entorno(pipeline=_abortado(contexto=contexto)) as m:
        _continuar()
        ctx_prevuelo = m["prevuelo"].await_args.args[1]
    assert "step_1_ref" not in ctx_prevuelo and ctx_prevuelo["step_0_ref"] == _REF_A


def test_el_prevuelo_real_mide_el_peor_caso_del_dependiente_de_un_paso_a_rehacer(monkeypatch):
    """Requisito (b), con el prevuelo() REAL (catálogo y sonda mockeados,
    igual que test_prevuelo_orquestador.py): el step 0 tiene una ref VIEJA
    ilegible en el contexto guardado y se rehace; el step 1 depende de 0.
    Si la ref ilegible de step 0 no se sacara del contexto antes de llamar a
    prevuelo(), _chars_de_entrada() intentaría leerla (RefIlegible, ya que
    step 1 declara depends_on=[0] -- dependencia "full") y la corrida
    reventaría, o -- si no la declarara -- mediría un resumen corto en vez
    del peor caso. Con el contexto podado, mide MAX_DEP_CONTEXT_CHARS+1."""
    from jacobs import prevuelo as pv
    from jacobs.executor import MAX_DEP_CONTEXT_CHARS
    from jacobs.prevuelo_catalogo import Catalogo, FilaFaceta, FilaModelo
    from jacobs.sonda import ResultadoSonda

    ahora = 1_000_000.0
    catalogo = Catalogo(
        facetas={"jekyll": FilaFaceta("jekyll", "http_openai_compat", None, "deepseek",
                                      "https://api.deepseek.example/v1", "deepseek-v4-flash")},
        modelos={("deepseek", "deepseek-v4-flash"): FilaModelo("max_tokens", 8192, Decimal("0.27"), Decimal("1.10"))},
        min_output_tokens={}, proveedores_con_credencial=frozenset({"deepseek"}),
        salud={"jekyll": (ahora - 60, "ok")},
    )
    @asynccontextmanager
    async def conexion_de_prueba():  # Task 15b: sin conexión real, los lectores están mockeados
        yield object()

    monkeypatch.setattr(pv.store, "conexion_del_pool", conexion_de_prueba)
    monkeypatch.setattr(pv.prevuelo_catalogo, "leer_catalogo", AsyncMock(return_value=catalogo))
    monkeypatch.setattr(pv.prevuelo_catalogo, "resolver_motores", AsyncMock(return_value={}))
    monkeypatch.setattr(pv.sonda, "sondear", AsyncMock(return_value=ResultadoSonda(True, None)))
    monkeypatch.setattr(pv, "_ahora", lambda: ahora)
    monkeypatch.setattr(pv, "_max_iteraciones", lambda: 5)
    monkeypatch.setenv("JAX_PREVUELO_CHARS_POR_TOKEN", "2")

    pasos = [
        Step(step_id="s0", pipeline_id="p1", step_index=0, facet="jekyll", capability="research",
             input={"prompt": "p0"}, status=StepStatus.failed, error="cortado"),
        Step(step_id="s1", pipeline_id="p1", step_index=1, facet="jekyll", capability="research",
             input={"prompt": "p1"}, depends_on=[0], status=StepStatus.pending),
    ]
    contexto = {"objective": "o", "step_0_ref": _ILEGIBLE}
    pipeline = Pipeline(pipeline_id="p1", name="t", invoked_by="plataforma", mode="autonomous",
                        status=PipelineStatus.aborted, context=contexto)

    with patch.object(continuar.store, "pipeline_get", AsyncMock(return_value=pipeline)), \
         patch.object(continuar.store, "steps_by_pipeline", AsyncMock(return_value=pasos)), \
         patch.object(continuar.store, "pipeline_count_active", AsyncMock(return_value=0)), \
         patch.object(continuar, "_validate_plan_capabilities", AsyncMock()), \
         patch.object(continuar, "check_kill_switch", return_value=False):
        r = asyncio.run(continuar.previsualizar("p1", "plataforma"))

    costo_1 = next(c for c in r["veredicto"]["pasos_costo"] if c["paso"] == 1)
    assert costo_1["tokens_in_max"] >= MAX_DEP_CONTEXT_CHARS // 2


def test_plan_inconsistente_da_409():
    pasos = _pasos()
    pasos[1].step_index = 5  # hueco: no queda 0..N-1
    with _entorno(pasos=pasos) as m:
        e = _rechazo()
        m["tx"].assert_not_awaited()
        m["prevuelo"].assert_not_awaited()
    assert (e.status_code, e.code) == (409, "plan_inconsistente")


def test_plan_ya_invalido_sin_reasignar_da_plan_rechazado():
    # Sin ningún reasignar: el plan GUARDADO ya viola cleanroom (step 2,
    # capability critique, audita al step 1, mismo facet jekyll). El código
    # tiene que seguir siendo "plan_rechazado", no "reasignacion_invalida"
    # (ese código es solo cuando la reasignación PEDIDA rompe algo).
    pasos = _pasos(capability_2="critique")
    with _entorno(pasos=pasos) as m:
        e = _rechazo()
        m["tx"].assert_not_awaited()
    assert (e.status_code, e.code) == (422, "plan_rechazado")
    assert "cleanroom" in e.cuerpo()["detalle"][0]["reason"]


def test_validate_plan_capabilities_rechaza_da_422():
    from jacobs.plan import PlanRejected, PlanViolation
    rechazo = PlanRejected([PlanViolation(2, "jekyll", None, "research", "capability no soportada")])
    with _entorno() as m:
        m["capacidades"].side_effect = rechazo
        e = _rechazo()
        m["tx"].assert_not_awaited()
    assert (e.status_code, e.code) == (422, "plan_rechazado")
    assert e.cuerpo()["detalle"][0]["reason"] == "capability no soportada"


def test_costo_igual_al_aceptado_no_rechaza():
    with _entorno() as m:
        r, _ = _continuar(costo_max_aceptado_usd=Decimal("0.250000"))
        m["tx"].assert_awaited_once()
    assert r["status"] == "running"


def test_current_step_index_es_el_minimo_de_los_pasos_a_correr():
    with _entorno() as m:
        _, p = _continuar()
        indice = m["tx"].await_args.args[6]
    assert indice == 2 and p.current_step_index == 2


def test_user_id_tenant_id_explicitos_pisan_los_del_pipeline():
    with _entorno() as m:
        _continuar(user_id="99", tenant_id="55")
        kw = m["prevuelo"].await_args.kwargs
    assert (kw["user_id"], kw["tenant_id"]) == ("99", "55")


def test_la_lectura_de_ref_corre_por_asyncio_to_thread():
    # Requisito (f): nada bloqueante en async def -- _load_ref lee del disco.
    # El espía envuelve el asyncio.to_thread REAL (no cambia el resultado),
    # así que si `analizar()` llamara a `_ref_legible` directo, este test
    # dejaría de ver pasar la función por acá.
    real_to_thread = asyncio.to_thread
    llamadas = []

    async def espia(fn, *args, **kwargs):
        llamadas.append(fn)
        return await real_to_thread(fn, *args, **kwargs)

    with _entorno(), patch.object(continuar.asyncio, "to_thread", espia):
        _continuar()
    assert continuar._ref_legible in llamadas


def test_previsualizar_no_escribe_y_explica_por_que_no():
    with _entorno() as m:
        r = asyncio.run(continuar.previsualizar("p1", "plataforma"))
        m["tx"].assert_not_awaited()
        m["evento"].assert_not_awaited()
    assert r["continuable"] is True and r["motivo"] is None and r["veredicto"]["ok"] is True
    with _entorno(pipeline=_abortado(status=PipelineStatus.completed)):
        r = asyncio.run(continuar.previsualizar("p1", "plataforma"))
    # Enmienda de contrato con la Mesa: veredicto solo va con motivo.code
    # prevuelo_rechazado o limite_de_activos -- acá el rechazo es previo
    # (analizar() nunca llegó a correr el pre-vuelo), así que va null.
    assert r["continuable"] is False and r["motivo"]["code"] == "estado_no_continuable"
    assert r["veredicto"] is None
    with _entorno(activos=3):
        r = asyncio.run(continuar.previsualizar("p1", "plataforma"))
    assert r["continuable"] is False and r["motivo"]["code"] == "limite_de_activos"
    assert r["veredicto"] is not None and r["veredicto"]["ok"] is True


def test_previsualizar_explica_prevuelo_rechazado_y_lleva_el_veredicto():
    with _entorno(veredicto=_veredicto(ok=False)) as m:
        r = asyncio.run(continuar.previsualizar("p1", "plataforma"))
        m["tx"].assert_not_awaited()
        m["evento"].assert_not_awaited()
    assert r["continuable"] is False
    assert r["motivo"] == {"code": "prevuelo_rechazado",
                           "detalle": "el pre-vuelo encontró violaciones: ver `veredicto`"}
    assert r["veredicto"] is not None and r["veredicto"]["ok"] is False


def test_previsualizar_propaga_el_error_del_prevuelo():
    # Requisito (c): un error inesperado del pre-vuelo (DB caída,
    # FacetUnavailableError) NO se traga -- sube tal cual para que el
    # llamador responda 503 (§8, desvío 20 del plan).
    with _entorno() as m:
        m["prevuelo"].side_effect = RuntimeError("DB caída durante el pre-vuelo")
        with pytest.raises(RuntimeError, match="DB caída"):
            asyncio.run(continuar.previsualizar("p1", "plataforma"))


def test_previsualizar_relanza_403_y_404():
    with _entorno() as m:
        with pytest.raises(continuar.ContinuarRechazado) as e:
            asyncio.run(continuar.previsualizar("p1", "ada"))
        m["prevuelo"].assert_not_awaited()
    assert e.value.status_code == 403
    with _entorno(pipeline=None) as m:
        with pytest.raises(continuar.ContinuarRechazado) as e:
            asyncio.run(continuar.previsualizar("p1", "plataforma"))
        m["prevuelo"].assert_not_awaited()
    assert e.value.status_code == 404


# ---------------------------------------------------------------------------
# F3 (ola final, Ruling R31): continuar recuenta el cupo y escribe dentro del
# candado con nombre de MariaDB (el CLI corre en otro proceso que LAS MANOS).
# ---------------------------------------------------------------------------

def test_continuar_recuenta_bajo_el_candado_y_respeta_el_cupo_que_otro_proceso_lleno():
    with _entorno() as m:
        m["activos"].side_effect = [0, 3]
        e = _rechazo()
        m["tx"].assert_not_awaited()
        assert m["activos"].await_args.kwargs == {"conexion": "conexion-del-candado"}
        assert (m["candado"].entradas, m["candado"].salidas) == (1, 1)
    assert (e.status_code, e.code) == (429, "limite_de_activos")


def test_continuar_escribe_la_transaccion_dentro_del_candado():
    with _entorno() as m:
        orden = m["candado"].orden
        m["activos"].side_effect = lambda **kw: orden.append("contar") or 0

        async def tx(*a, **kw):
            orden.append("transaccion")
            return 3

        m["tx"].side_effect = tx
        _continuar()
    assert orden == ["contar", "candado", "contar", "transaccion", "soltar"]


def test_continuar_sin_candado_no_escribe_y_propaga():
    from jacobs import store

    with _entorno() as m:
        falso = _CandadoFalso(falla=store.CandadoNoDisponible("GET_LOCK venció"))
        with patch.object(continuar.store, "candado_de_activos", falso):
            with pytest.raises(store.CandadoNoDisponible):
                _continuar()
        m["tx"].assert_not_awaited()


def test_la_transaccion_de_continuar_tiene_plazo_y_no_retiene_el_candado(monkeypatch):
    """m1 de la re-revisión final: la transacción de continuar corría bajo el
    candado entre procesos SIN techo de tiempo, mientras que la de crear sí lo
    tiene. Un `SELECT ... FOR UPDATE` esperando un lock de fila (hasta
    innodb_lock_wait_timeout, 50 s por defecto) dejaba el candado
    `jacobs_crear_o_continuar:<base>` tomado y cualquier create o continue de
    cualquier proceso respondía 503. Ahora vence con el mismo plazo que crear
    (JAX_DB_CONNECT_TIMEOUT_SECONDS) y suelta el candado.
    Expected contra 1b124fe: el test se cuelga y lo corta su propio
    asyncio.wait_for (TimeoutError a los 5 s, con el candado retenido)."""
    monkeypatch.setenv("JAX_DB_CONNECT_TIMEOUT_SECONDS", "1")

    async def transaccion_colgada(*a, **kw):
        await asyncio.sleep(3600)

    async def cuerpo():
        with _entorno() as m:
            m["tx"].side_effect = transaccion_colgada
            inicio = time.monotonic()
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(
                    continuar.continuar("p1", "plataforma"), 5)
            return time.monotonic() - inicio, m["candado"]

    espera, candado = asyncio.run(cuerpo())
    assert espera < 3, espera
    assert candado.entradas == candado.salidas == 1, "el candado quedó tomado"
