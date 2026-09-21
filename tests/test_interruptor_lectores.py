"""LAS MANOS lee el freno desde JAX_KILL_SWITCH_PATH y falla cerrado (plan
2026-09-16-frente-b-kill-switch, Task 3). Sólo se mockea el borde de red (el
transporte), la credencial y la escritura de costo; JobStore y MotorCatalog
corren de verdad contra un JSONL temporal.

Corre con:
  PYTHONPATH=.:las_manos python -m pytest tests/test_interruptor_lectores.py -v

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import ast
import asyncio
import os
import tomllib
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

import interruptor
from motor_registry import worker
from motor_registry.catalog import MotorCatalog
from motor_registry.job_store import JobStore
from motor_registry.models import JobStatus
from workers import ssh_worker

RAIZ = Path(__file__).resolve().parents[1]
ES_ROOT = os.geteuid() == 0

_CAP = {
    "allowed_motors": ["kimi"], "allowed_callers": ["jacobs"], "risk_level": "low",
    "sandbox_only": True, "requires_human_gate": False, "max_execution_minutes": 15,
    "max_recursion_depth": 0, "output_schema": "",
}
_CFG = {
    "motors": {
        "kimi": {
            "enabled": True, "provider": "kimi", "transport": "http_openai_compat",
            "provider_id": "moonshot", "api_key_env": "KIMI_API_KEY",
            "api_url": "https://api.moonshot.ai/v1/chat/completions", "model": "kimi-k3",
            "max_context_tokens": 256000, "sandbox_only": True, "default_timeout_seconds": 600,
            "supports_reasoning": True, "reasoning_default_visibility": "audit_only",
            "max_tokens": 8000,
        },
    },
    "capabilities": {"implementation": _CAP},
}


async def _job(tmp_path, ruta_freno, transporte, durante=None):
    """Corre worker.run con `transporte`; `durante(tarea)` actúa mientras corre."""
    store = JobStore(str(tmp_path / "jobs.jsonl"))
    job_id = store.create(caller="jacobs", capability="implementation", motor="kimi",
                          trace_id="t", prompt="p", recursion_depth=0)
    parches = [
        patch.object(worker, "resolve_credential", AsyncMock(return_value="sk-fake")),
        patch("contrato_dispatch._leer_contrato", AsyncMock(return_value=("max_tokens", 131072))),
        patch("motor_registry.usage_writer.record_motor_usage", AsyncMock()),
        patch("httpx.AsyncClient.post", AsyncMock(side_effect=AssertionError("llamada de red real en un test"))),
        patch.dict(worker._TRANSPORT_DISPATCH, {"http_openai_compat": transporte}),
    ]
    for p in parches:
        p.start()
    try:
        tarea = asyncio.create_task(worker.run(
            job_id=job_id, motor="kimi", capability="implementation", prompt="p",
            context={}, store=store, catalog=MotorCatalog(_CFG), kill_switch_path=str(ruta_freno)))
        if durante is not None:
            await durante(tarea)
        await asyncio.wait_for(asyncio.gather(tarea, return_exceptions=True), timeout=3.0)
    finally:
        for p in reversed(parches):
            p.stop()
    return store._index[job_id]


def _transporte_lento(en_vuelo, cancelado):
    async def transporte(**kwargs):
        en_vuelo.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelado.append(True)
            raise
    return transporte


def test_motor_en_vuelo_queda_killed_by_switch(tmp_path, monkeypatch):
    """EL PEOR CASO del spec, del lado de LAS MANOS: el modelo está
    respondiendo, aparece el freno y el job termina killed_by_switch."""
    ruta = tmp_path / "interruptor" / "PAUSE"
    ruta.parent.mkdir()
    monkeypatch.setattr(worker, "_KILL_SWITCH_INTERVAL", 0.05)
    cancelado = []

    async def escenario():
        en_vuelo = asyncio.Event()

        async def durante(tarea):
            await asyncio.wait_for(en_vuelo.wait(), 2.0)
            interruptor.escribir_pausa(ruta, '{"user_id": "peor-caso"}')

        return await _job(tmp_path, ruta, _transporte_lento(en_vuelo, cancelado), durante)

    estado = asyncio.run(escenario())
    assert estado["status"] == JobStatus.FAILED.value
    assert "killed_by_switch" in estado["error"]
    assert cancelado == [True]


def _estados_de_uso(caplog):
    """Con qué status reportó el uso el job. `_report_usage` sólo llama a
    record_motor_usage con tokens > 0; con 0 tokens (el caso de estos tests,
    que cortan en el primer turno) deja un WARNING con el MISMO status, y esa
    es la salida observable de la etiqueta."""
    return [r.args[2] for r in caplog.records
            if r.name == worker.logger.name and "sin tokens acumulados" in r.getMessage()]


def test_motor_cancelado_con_el_freno_puesto_queda_killed_by_switch(tmp_path, monkeypatch, caplog):
    """Jacobs corta el step a los 250 ms y cancela el job ANTES del watcher de
    5 s: sin esto el job quedaba 'cancelado externamente' y el freno no
    aparecía como la causa."""
    ruta = tmp_path / "PAUSE"
    monkeypatch.setattr(worker, "_KILL_SWITCH_INTERVAL", 30.0)
    cancelado = []

    async def escenario():
        en_vuelo = asyncio.Event()

        async def durante(tarea):
            await asyncio.wait_for(en_vuelo.wait(), 2.0)
            interruptor.escribir_pausa(ruta, "{}")
            tarea.cancel()

        return await _job(tmp_path, ruta, _transporte_lento(en_vuelo, cancelado), durante)

    caplog.set_level("WARNING", logger=worker.logger.name)
    estado = asyncio.run(escenario())
    assert estado["status"] == JobStatus.FAILED.value
    assert "killed_by_switch" in estado["error"]
    # Revisión final del frente B (hallazgo 6): el uso se reporta "failed",
    # igual que el resto de los killed_by_switch.
    assert _estados_de_uso(caplog) == ["failed"]


def test_motor_cancelado_sin_freno_sigue_siendo_cancelado(tmp_path, monkeypatch, caplog):
    ruta = tmp_path / "PAUSE"
    monkeypatch.setattr(worker, "_KILL_SWITCH_INTERVAL", 30.0)

    async def escenario():
        en_vuelo = asyncio.Event()

        async def durante(tarea):
            await asyncio.wait_for(en_vuelo.wait(), 2.0)
            tarea.cancel()

        return await _job(tmp_path, ruta, _transporte_lento(en_vuelo, []), durante)

    caplog.set_level("WARNING", logger=worker.logger.name)
    estado = asyncio.run(escenario())
    assert estado["status"] == JobStatus.CANCELLED.value
    assert estado["error"] == "Job cancelado externamente"
    assert _estados_de_uso(caplog) == ["cancelled"]


@pytest.mark.skipif(ES_ROOT, reason="root atraviesa cualquier permiso")
def test_motor_con_directorio_ilegible_no_llama_al_modelo(tmp_path):
    carpeta = tmp_path / "interruptor"
    carpeta.mkdir()
    (carpeta / "PAUSE").write_text("")
    llamadas = []

    async def transporte(**kwargs):
        llamadas.append(kwargs)
        return {"choices": [{"message": {"content": "x"}, "finish_reason": "stop"}], "usage": {}}

    carpeta.chmod(0)
    try:
        estado = asyncio.run(_job(tmp_path, carpeta / "PAUSE", transporte))
    finally:
        carpeta.chmod(0o700)
    assert estado["status"] == JobStatus.FAILED.value
    assert "killed_by_switch" in estado["error"]
    assert llamadas == []


@pytest.mark.skipif(ES_ROOT, reason="root atraviesa cualquier permiso")
def test_watcher_ssh_con_directorio_ilegible_mata_el_proceso(tmp_path, monkeypatch):
    carpeta = tmp_path / "interruptor"
    carpeta.mkdir()
    (carpeta / "PAUSE").write_text("")
    monkeypatch.setattr(ssh_worker, "POLL_INTERVAL", 0.01)

    async def escenario():
        proc = await asyncio.create_subprocess_exec("sleep", "30")
        abortado = {"flag": False}
        carpeta.chmod(0)
        try:
            await asyncio.wait_for(
                ssh_worker._kill_switch_watcher(proc, str(carpeta / "PAUSE"), abortado), 2.0)
        finally:
            carpeta.chmod(0o700)
        await asyncio.wait_for(proc.wait(), 2.0)
        return abortado["flag"], proc.returncode

    flag, codigo = asyncio.run(escenario())
    assert flag is True
    assert codigo is not None


def test_ssh_exec_sin_variable_no_ejecuta(monkeypatch):
    monkeypatch.delenv("JAX_KILL_SWITCH_PATH", raising=False)
    lanzar = AsyncMock(side_effect=AssertionError("no debía lanzar ssh"))
    monkeypatch.setattr(asyncio, "create_subprocess_exec", lanzar)
    with pytest.raises(interruptor.InterruptorSinConfigurar):
        asyncio.run(ssh_worker.ssh_exec("127.0.0.1", "true"))
    lanzar.assert_not_called()


def test_ssh_worker_no_guarda_una_ruta_global():
    assert not hasattr(ssh_worker, "KILL_SWITCH_PATH")


def _fuente(relativa):
    return (RAIZ / relativa).read_text(encoding="utf-8")


def test_server_resuelve_la_ruta_al_importarse_antes_de_leer_config():
    fuente = _fuente("las_manos/server.py")
    arbol = ast.parse(fuente)
    asignacion = next(
        n for n in arbol.body
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "KILL_SWITCH" for t in n.targets))
    assert ast.unparse(asignacion.value) == "ruta_del_interruptor()"
    lectura_config = next(n for n in arbol.body if isinstance(n, ast.With))
    assert asignacion.lineno < lectura_config.lineno
    assert "kill_switch_path" not in fuente
    assert "ssh_worker.KILL_SWITCH_PATH" not in fuente


def _es_llamada_a_la_ruta(nodo):
    return (isinstance(nodo, ast.Call) and isinstance(nodo.func, ast.Name)
            and nodo.func.id == "ruta_del_interruptor")


def test_routes_resuelve_la_ruta_en_cada_governed_dispatch():
    """La ruta se resuelve DENTRO de `dispatch` (en cada pedido, no al
    importar) y lo que recibe el worker es esa resolución. Revisión final del
    frente B (hallazgo 5): se resuelve en una variable ANTES de crear el job,
    así que el keyword ya no es la llamada literal."""
    fuente = _fuente("las_manos/motor_registry/routes.py")
    assert "_KILL_SWITCH_PATH" not in fuente
    arbol = ast.parse(fuente)
    dispatch = next(n for n in arbol.body
                    if isinstance(n, ast.AsyncFunctionDef) and n.name == "governed_dispatch")
    fuera = [n for n in arbol.body if n is not dispatch and not (isinstance(n, ast.AsyncFunctionDef) and n.name == "dispatch")
             for m in ast.walk(n) if _es_llamada_a_la_ruta(m)]
    assert fuera == []
    asignaciones = [n for n in ast.walk(dispatch)
                    if isinstance(n, ast.Assign) and any(_es_llamada_a_la_ruta(m) for m in ast.walk(n.value))]
    assert len(asignaciones) == 1 and isinstance(asignaciones[0].targets[0], ast.Name)
    variable = asignaciones[0].targets[0].id
    valores = [k.value for k in ast.walk(dispatch)
               if isinstance(k, ast.keyword) and k.arg == "kill_switch_path"]
    assert len(valores) == 1
    assert {n.id for n in ast.walk(valores[0]) if isinstance(n, ast.Name)} & {variable}


def test_dispatch_sin_variable_no_deja_un_job_huerfano(tmp_path, monkeypatch):
    """Revisión final del frente B (hallazgo 5): la ruta se resolvía DESPUÉS
    de `_STORE.create`; sin JAX_KILL_SWITCH_PATH el pedido fallaba cerrado
    pero dejaba un job `pending` que ningún worker iba a terminar."""
    from motor_registry import routes
    from motor_registry.models import GovernedDispatchRequest
    from motor_registry.policy import MotorPolicy

    catalogo = MotorCatalog(_CFG)
    store = JobStore(str(tmp_path / "jobs.jsonl"))
    lanzado = AsyncMock()
    monkeypatch.setattr(routes, "_STORE", store)
    monkeypatch.setattr(routes, "_CATALOG", catalogo)
    monkeypatch.setattr(routes, "_POLICY", MotorPolicy(catalogo))
    monkeypatch.setattr(routes, "_ensure_catalog_fresh", AsyncMock())
    monkeypatch.setattr(routes.motor_worker, "run", lanzado)
    monkeypatch.delenv("JAX_KILL_SWITCH_PATH", raising=False)
    class Store:
        def load_execution(self, _):
            raise AssertionError("la ruta debe fallar antes de cargar")
    routes.configure_governed_execution_store(Store())
    pedido = GovernedDispatchRequest(execution_id="x")

    with pytest.raises(interruptor.InterruptorSinConfigurar):
        asyncio.run(routes.governed_dispatch(pedido))
    assert store._index == {}
    lanzado.assert_not_called()


def test_config_toml_de_las_manos_sin_ruta():
    with open(RAIZ / "las_manos" / "config.toml", "rb") as f:
        assert "kill_switch_path" not in tomllib.load(f)["server"]
