"""Un solo ejecutor por pipeline (spec 2026-09-17 §5.3), sin DB ni red.

La tienda falsa implementa la API VIEJA (pipeline_update_status, step_upsert)
y la NUEVA (condicionales por época): así los tests corren contra el ejecutor
de hoy y fallan por comportamiento, no por un AttributeError.

Hoy cancelar no detiene nada: la escritura de fin de ola vuelve a poner
`running` y la ola siguiente se despacha.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import asyncio
import os

# Base de tests de ESTA sesión: respeta JAX_TEST_DB_SUFIJO en vez de
# clavar el nombre (mismo override incondicional que antes).
from base_de_test import fijar_base_de_test  # noqa: E402

fijar_base_de_test()

from unittest.mock import AsyncMock  # noqa: E402

from jacobs import executor  # noqa: E402
from jacobs.models import Pipeline, PipelineStatus, Step, StepStatus  # noqa: E402


class TiendaFalsa:
    def __init__(self, status="pending", epoca=0):
        self.status = status
        self.epoca = epoca
        self.contexto: dict = {}
        self.pasos: dict[int, str] = {}
        self.eventos: list[tuple[str, dict]] = []
        # Qué steps INTENTARON escribirse (con éxito o no) -- distingue "no se
        # intentó despachar/escribir" de "se intentó y la condición lo rechazó".
        self.step_upsert_intentos: list[int] = []

    async def pipeline_update_status(self, pipeline_id, status, current_step_index=None, context=None):
        self.status = status.value
        if context is not None:
            self.contexto = dict(context)

    async def pipeline_update_status_si_epoca(self, pipeline_id, epoca, status,
                                              current_step_index=None, context=None, *,
                                              desde=(PipelineStatus.running,)):
        if epoca != self.epoca or self.status not in {d.value for d in desde}:
            return False
        self.status = status.value
        if context is not None:
            self.contexto = dict(context)
        return True

    async def pipeline_epoca_y_status(self, pipeline_id):
        return self.epoca, PipelineStatus(self.status)

    async def step_upsert(self, s):
        self.pasos[s.step_index] = s.status.value

    async def step_upsert_si_epoca(self, s, epoca):
        self.step_upsert_intentos.append(s.step_index)
        if epoca != self.epoca or self.status != "running":
            return False
        self.pasos[s.step_index] = s.status.value
        return True

    async def event_append(self, pipeline_id, event_type, payload=None, step_id=None):
        self.eventos.append((event_type, payload or {}))

    def tipos(self):
        return [t for t, _ in self.eventos]


def _pipeline(n_pasos=2, epoca=0, modo="autonomous"):
    pasos = [
        Step(pipeline_id="p1", step_index=i, facet="jekyll", capability="research",
             input={"prompt": f"paso {i}"}, depends_on=[i - 1] if i else [])
        for i in range(n_pasos)
    ]
    return Pipeline(pipeline_id="p1", name="t", invoked_by="plataforma", mode=modo,
                    plan=pasos, context={"objective": "o"}, run_epoch=epoca)


def _pipeline_hyde(epoca=0):
    """Un solo step de faceta hyde, sin aprobar: dispara el gate en la ola 0."""
    pasos = [
        Step(pipeline_id="p1", step_index=0, facet="hyde", capability="ejecutar",
             input={"cmd": "algo"}, depends_on=[]),
    ]
    return Pipeline(pipeline_id="p1", name="t", invoked_by="plataforma", mode="autonomous",
                    plan=pasos, context={"objective": "o"}, run_epoch=epoca)


def _correr(monkeypatch, tienda, pipeline, despachar, *, kill_switch=False):
    monkeypatch.setattr(executor, "store", tienda)
    monkeypatch.setattr(executor, "check_kill_switch", lambda: kill_switch)
    monkeypatch.setattr(executor, "save_if_large", lambda pid, sid, raw, **kw: (None, raw))
    monkeypatch.setattr(executor, "_persist_step_to_repo", AsyncMock())
    monkeypatch.setattr(executor, "_dispatch_step", despachar)
    asyncio.run(executor.run_pipeline(pipeline))


def _fallar_desde_la_llamada(tienda, metodo: str, n: int):
    """Las primeras n-1 llamadas a `metodo` usan la lógica real de la tienda
    (dejan el estado como lo dejaría una corrida sana); desde la llamada n en
    adelante, la llamada devuelve False SIN tocar el estado -- simula que la
    época o el status se perdieron justo ahí (un /cancel, un /continue o el
    reaper concurrentes), para aislar el punto de escritura bajo prueba del
    resto de la corrida."""
    original = getattr(tienda, metodo)
    contador = {"n": 0}

    async def envoltorio(*a, **k):
        contador["n"] += 1
        if contador["n"] < n:
            return await original(*a, **k)
        return False

    setattr(tienda, metodo, envoltorio)


def test_cancelar_durante_la_ola_no_se_reescribe_running(monkeypatch):
    tienda, despachados = TiendaFalsa(), []

    async def despachar(step, pipeline):
        despachados.append(step.step_index)
        tienda.status = "aborted"  # POST /cancel mientras el paso corre
        return {"success": True, "result": "hecho"}

    _correr(monkeypatch, tienda, _pipeline(2), despachar)
    assert tienda.status == "aborted"
    assert despachados == [0]
    assert tienda.tipos().count("RUN_SUPERSEDED") == 1
    assert "PIPELINE_COMPLETED" not in tienda.tipos()


def test_continuado_por_otro_la_corrida_vieja_no_escribe_su_contexto(monkeypatch):
    tienda, despachados = TiendaFalsa(), []

    async def despachar(step, pipeline):
        despachados.append(step.step_index)
        tienda.epoca = 1  # POST /continue tomó el pipeline
        return {"success": True, "result": "tarde"}

    _correr(monkeypatch, tienda, _pipeline(2), despachar)
    assert "step_0_ref" not in tienda.contexto
    assert despachados == [0]
    assert tienda.tipos().count("RUN_SUPERSEDED") == 1


def test_un_solo_pipeline_aborted_con_los_errores(monkeypatch):
    tienda = TiendaFalsa()

    async def despachar(step, pipeline):
        raise RuntimeError("se cayó el proveedor")

    _correr(monkeypatch, tienda, _pipeline(1), despachar)
    abortados = [p for t, p in tienda.eventos if t == "PIPELINE_ABORTED"]
    assert len(abortados) == 1
    assert abortados[0] == {"at_wave": 0, "failed_steps": [0],
                            "errores": {"0": "se cayó el proveedor"}}
    assert tienda.status == "aborted"


def test_arranque_con_epoca_ajena_no_despacha(monkeypatch):
    tienda, despachados = TiendaFalsa(status="pending", epoca=1), []

    async def despachar(step, pipeline):
        despachados.append(step.step_index)
        return {"success": True, "result": "no debería"}

    _correr(monkeypatch, tienda, _pipeline(2, epoca=0), despachar)
    assert despachados == []
    assert tienda.status == "pending"
    assert tienda.tipos().count("RUN_SUPERSEDED") == 1


def test_relee_la_epoca_antes_de_cada_ola(monkeypatch):
    tienda, despachados = TiendaFalsa(), []

    def _cancelar_si_ya_persistio_la_ola_0():
        if "step_0_ref" in tienda.contexto and tienda.status == "running":
            tienda.status = "aborted"

    nuevo_original = tienda.pipeline_update_status_si_epoca
    viejo_original = tienda.pipeline_update_status

    async def nuevo(*a, **k):
        ok = await nuevo_original(*a, **k)
        if ok:
            _cancelar_si_ya_persistio_la_ola_0()
        return ok

    async def viejo(*a, **k):
        await viejo_original(*a, **k)
        _cancelar_si_ya_persistio_la_ola_0()

    tienda.pipeline_update_status_si_epoca = nuevo
    tienda.pipeline_update_status = viejo

    async def despachar(step, pipeline):
        despachados.append(step.step_index)
        return {"success": True, "result": "ok"}

    _correr(monkeypatch, tienda, _pipeline(2), despachar)
    assert despachados == [0]
    assert tienda.status == "aborted"
    assert tienda.tipos().count("RUN_SUPERSEDED") == 1
    # Sin la relectura de época al empezar la ola 1, el ejecutor igual termina
    # en (despachados=[0], status="aborted", RUN_SUPERSEDED=1) -- pero LLEGA
    # ahí por otro camino: abre la ola 1 (segundo WAVE_STARTED) e intenta
    # escribir el step 1 (que la condición de época rechaza). Estas dos
    # aserciones distinguen "nunca se abrió la ola 1" de "se abrió y se frenó
    # más tarde".
    assert tienda.tipos().count("WAVE_STARTED") == 1
    assert 1 not in tienda.step_upsert_intentos


def test_corrida_normal_completa_y_persiste_el_contexto(monkeypatch):
    """CONTROL: pasa también con el ejecutor de hoy."""
    tienda = TiendaFalsa()

    async def despachar(step, pipeline):
        return {"success": True, "result": f"r{step.step_index}"}

    _correr(monkeypatch, tienda, _pipeline(2), despachar)
    assert tienda.status == "completed"
    assert {"step_0_ref", "step_1_ref"} <= set(tienda.contexto)
    assert "RUN_SUPERSEDED" not in tienda.tipos()
    assert tienda.tipos().count("PIPELINE_COMPLETED") == 1


# ----------------------------------------------------------------
# Ronda de arreglo 1: cada salida temprana por época perdida también deja de
# escribir. Cada test aísla UN punto de escritura con _fallar_desde_la_llamada
# -- las llamadas anteriores usan la lógica real de la tienda (para llegar
# limpio hasta ahí) y la llamada bajo prueba se fuerza a devolver False.
# ----------------------------------------------------------------

def test_kill_switch_no_marca_el_step_si_perdio_la_epoca_antes_de_escribirlo(monkeypatch):
    tienda = TiendaFalsa()
    _fallar_desde_la_llamada(tienda, "step_upsert_si_epoca", 1)

    async def despachar(step, pipeline):
        raise AssertionError("el kill switch no debería despachar nada")

    _correr(monkeypatch, tienda, _pipeline(2), despachar, kill_switch=True)
    assert tienda.pasos == {}
    assert "KILL_SWITCH_ABORTED" not in tienda.tipos()
    assert tienda.status == "running"  # nunca llegó a escribirse `aborted`
    assert tienda.tipos().count("RUN_SUPERSEDED") == 1


def test_kill_switch_no_escribe_aborted_si_perdio_la_epoca_tras_marcar_los_steps(monkeypatch):
    tienda = TiendaFalsa()
    # Llamada 1 = el arranque (running); llamada 2 = el `aborted` del kill
    # switch, forzada a perder. El step SÍ se marca antes (lógica real).
    _fallar_desde_la_llamada(tienda, "pipeline_update_status_si_epoca", 2)

    async def despachar(step, pipeline):
        raise AssertionError("el kill switch no debería despachar nada")

    _correr(monkeypatch, tienda, _pipeline(2), despachar, kill_switch=True)
    assert tienda.pasos.get(0) == "failed"
    assert "KILL_SWITCH_ABORTED" not in tienda.tipos()
    assert tienda.status == "running"  # el `aborted` nunca se escribió
    assert tienda.tipos().count("RUN_SUPERSEDED") == 1


def test_hyde_gate_no_marca_el_step_si_perdio_la_epoca_antes_de_escribirlo(monkeypatch):
    tienda = TiendaFalsa()
    _fallar_desde_la_llamada(tienda, "step_upsert_si_epoca", 1)

    async def despachar(step, pipeline):
        raise AssertionError("el gate de hyde no debería despachar nada")

    _correr(monkeypatch, tienda, _pipeline_hyde(), despachar)
    assert tienda.pasos == {}
    assert "STEP_BLOCKED_HUMAN_GATE" not in tienda.tipos()
    assert "PIPELINE_INTERRUPTED" not in tienda.tipos()
    assert tienda.tipos().count("RUN_SUPERSEDED") == 1


def test_hyde_gate_no_escribe_interrupted_si_perdio_la_epoca_tras_marcar_el_step(monkeypatch):
    tienda = TiendaFalsa()
    # Llamada 1 = el arranque (running); llamada 2 = el `interrupted` del
    # gate de hyde, forzada a perder. El step SÍ se marca antes.
    _fallar_desde_la_llamada(tienda, "pipeline_update_status_si_epoca", 2)

    async def despachar(step, pipeline):
        raise AssertionError("el gate de hyde no debería despachar nada")

    _correr(monkeypatch, tienda, _pipeline_hyde(), despachar)
    assert tienda.pasos.get(0) == "blocked_human_gate"
    assert "STEP_BLOCKED_HUMAN_GATE" in tienda.tipos()  # esa SÍ se alcanzó a escribir
    assert "PIPELINE_INTERRUPTED" not in tienda.tipos()
    assert tienda.tipos().count("RUN_SUPERSEDED") == 1


def test_dry_run_no_escribe_el_evento_si_perdio_la_epoca(monkeypatch):
    tienda = TiendaFalsa()
    # dry_run no pasa por el arranque a `running`: su `completed` es la
    # primera llamada a pipeline_update_status_si_epoca de la corrida.
    _fallar_desde_la_llamada(tienda, "pipeline_update_status_si_epoca", 1)

    async def despachar(step, pipeline):
        raise AssertionError("dry_run no debería despachar nada")

    _correr(monkeypatch, tienda, _pipeline(2, modo="dry_run"), despachar)
    assert "DRY_RUN_COMPLETE" not in tienda.tipos()
    assert tienda.status == "pending"  # nunca se escribió `completed`
    assert tienda.tipos().count("RUN_SUPERSEDED") == 1


def test_supervised_no_escribe_interrupted_si_perdio_la_epoca_tras_la_ola(monkeypatch):
    tienda, despachados = TiendaFalsa(), []
    # Llamada 1 = arranque; llamada 2 = persistir avance de la ola 0 (real);
    # llamada 3 = el `interrupted` de supervised, forzada a perder.
    _fallar_desde_la_llamada(tienda, "pipeline_update_status_si_epoca", 3)

    async def despachar(step, pipeline):
        despachados.append(step.step_index)
        return {"success": True, "result": "ok"}

    _correr(monkeypatch, tienda, _pipeline(2, modo="supervised"), despachar)
    assert despachados == [0]
    assert "WAVE_COMPLETED" in tienda.tipos()  # esa SÍ se alcanzó a escribir
    assert "PIPELINE_INTERRUPTED" not in tienda.tipos()
    assert tienda.tipos().count("RUN_SUPERSEDED") == 1


def test_fail_step_no_escribe_step_failed_si_perdio_la_epoca(monkeypatch):
    """_fail_step (executor.py) aislado: si su propia escritura condicional
    del step se rechaza, no debe emitir STEP_FAILED."""
    tienda = TiendaFalsa(status="aborted")  # cancelado justo antes de reportar el error
    monkeypatch.setattr(executor, "store", tienda)
    step = Step(pipeline_id="p1", step_index=0, facet="jekyll", capability="research", input={})
    pipeline = Pipeline(pipeline_id="p1", name="t", invoked_by="plataforma", mode="autonomous",
                        plan=[step], context={}, run_epoch=0)

    asyncio.run(executor._fail_step(pipeline, step, 0, "se cayó el proveedor"))

    assert step.status == StepStatus.failed  # el objeto en memoria SÍ queda marcado
    assert tienda.pasos == {}  # pero la tienda nunca lo recibió
    assert "STEP_FAILED" not in tienda.tipos()


# ---------------------------------------------------------------------------
# F1 (ola final, 2026-09-17): el artifact (>60 KB) de una corrida superada no
# pisa el de la vigente. Antes la ruta era {pipeline}/{step}/output.json, sin
# época, y save_if_large corría ANTES de la escritura condicional.
# ---------------------------------------------------------------------------

def test_artifact_tardio_de_corrida_superada_no_pisa_el_de_la_vigente(monkeypatch, tmp_path):
    from jacobs import artifacts

    monkeypatch.setattr(artifacts, "ARTIFACTS_DIR", tmp_path)
    monkeypatch.setattr(executor, "_persist_step_to_repo", AsyncMock())
    tienda = TiendaFalsa(status="running", epoca=1)
    monkeypatch.setattr(executor, "store", tienda)
    grande = artifacts.SIZE_LIMIT + 10

    vieja_despachada, soltar_vieja = asyncio.Event(), asyncio.Event()

    async def despachar(step, pipeline):
        if pipeline.run_epoch == 1:
            vieja_despachada.set()
            await soltar_vieja.wait()
            return {"success": True, "result": "V" * grande}
        return {"success": True, "result": "N" * grande}

    monkeypatch.setattr(executor, "_dispatch_step", despachar)

    def _corrida(epoca):
        paso = Step(step_id="s0", pipeline_id="p1", step_index=0, facet="jekyll",
                    capability="research", input={"prompt": "x"}, timeout_seconds=30)
        return Pipeline(pipeline_id="p1", name="t", invoked_by="plataforma", mode="autonomous",
                        plan=[paso], context={"objective": "o"}, run_epoch=epoca)

    async def escenario():
        vieja = _corrida(1)
        tarea_vieja = asyncio.create_task(executor._run_one_step(vieja.plan[0], 0, vieja))
        await vieja_despachada.wait()
        tienda.epoca = 2  # POST /continue tomó el pipeline mientras la vieja esperaba al proveedor
        nueva = _corrida(2)
        assert await executor._run_one_step(nueva.plan[0], 0, nueva) is True
        ref_vigente = nueva.context["step_0_ref"]
        soltar_vieja.set()
        # m2 (2026-09-17): la corrida superada no devuelve False ("el paso
        # falló") sino PASO_SIN_ESCRITURA ("no pude escribir el paso").
        assert await tarea_vieja is executor.PASO_SIN_ESCRITURA
        return ref_vigente

    ref = asyncio.run(escenario())
    assert ref.startswith("artifact://")
    assert artifacts.read_artifact(ref)["result"].startswith("N")


def test_ref_de_artifact_sin_epoca_se_sigue_leyendo(monkeypatch, tmp_path):
    """Compatibilidad hacia atrás: los pipelines guardados antes de F1 tienen
    refs {pipeline}/{step}/output.json en context; continuar y los lectores
    las tienen que seguir leyendo."""
    from jacobs import artifacts

    monkeypatch.setattr(artifacts, "ARTIFACTS_DIR", tmp_path)
    viejo = tmp_path / "p1" / "s0"
    viejo.mkdir(parents=True)
    (viejo / "output.json").write_text('{"result": "de antes"}', encoding="utf-8")
    assert executor._load_ref("artifact://jacobs/p1/s0/output.json") == {"result": "de antes"}


def test_la_ruta_del_artifact_lleva_la_epoca(monkeypatch, tmp_path):
    from jacobs import artifacts

    monkeypatch.setattr(artifacts, "ARTIFACTS_DIR", tmp_path)
    datos = {"result": "x" * (artifacts.SIZE_LIMIT + 1)}
    ref, inline = artifacts.save_if_large("p1", "s0", datos, epoca=7)
    assert inline is None
    assert ref == "artifact://jacobs/p1/s0/e7/output.json"
    assert artifacts.read_artifact(ref) == datos


def test_un_paso_que_no_se_pudo_escribir_aborta_con_motivo_no_con_error_null(monkeypatch):
    """m2 de la re-revisión final: `_run_one_step` devolvía False tanto por
    "el paso falló" como por "no pude escribir el paso". step_upsert_si_epoca
    también da 0 filas si la fila del paso CAMBIÓ (borrada, otro step_id) con
    la corrida todavía vigente: ahí el pipeline aborta y el evento salía con
    `"errores": {"0": null}` -- un aborto sin motivo en la Mesa.
    Expected contra cfb0bd1: `{'0': None}`."""
    tienda = TiendaFalsa()

    async def despachar(step, pipeline):
        return {"result": "ok"}

    # La escritura de `running` pasa; la de `completed` (2ª llamada) devuelve
    # False SIN tocar época ni status: la fila del paso cambió, no la corrida.
    _fallar_desde_la_llamada(tienda, "step_upsert_si_epoca", 2)
    _correr(monkeypatch, tienda, _pipeline(1), despachar)

    abortados = [p for t, p in tienda.eventos if t == "PIPELINE_ABORTED"]
    assert len(abortados) == 1, tienda.tipos()
    assert abortados[0]["failed_steps"] == [0]
    motivo = abortados[0]["errores"]["0"]
    assert motivo is not None and "no se pudo escribir" in motivo, motivo
    assert "RUN_SUPERSEDED" not in tienda.tipos(), "la corrida seguía vigente"


def test_un_paso_sin_escribir_por_epoca_perdida_sigue_siendo_run_superseded(monkeypatch):
    """Control del otro lado: si la época SÍ se perdió, el camino no cambia --
    RUN_SUPERSEDED una vez y ningún PIPELINE_ABORTED."""
    tienda = TiendaFalsa()

    async def despachar(step, pipeline):
        tienda.epoca += 1  # otro pedido tomó la época mientras el paso corría
        return {"result": "ok"}

    _correr(monkeypatch, tienda, _pipeline(1), despachar)
    assert tienda.tipos().count("RUN_SUPERSEDED") == 1, tienda.tipos()
    assert "PIPELINE_ABORTED" not in tienda.tipos()
