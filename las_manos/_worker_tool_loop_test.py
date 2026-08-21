#!/usr/bin/env python3
"""
Motor Registry — bucle multi-turno de tool-calling (GAP 2, Fase 3).

Corazón de la sesión 2026-08-19: el modelo pide una tool, el sistema la
ejecuta (re-autorizando en tool_authority.py en CADA turno, sin caché),
le devuelve el resultado como role:"tool", y el modelo sigue -- hasta que
responde sin tool_calls, o se agota una cota (iteraciones, tiempo,
detección de loop, bytes acumulados leídos).

Verificado a mano contra Ollama real antes de escribir esto (ver sesión):
formato exacto de historial que espera /v1/chat/completions -- un mensaje
assistant con tool_calls seguido de un mensaje {"role":"tool",
"tool_call_id":<id>,"content":<string>} por CADA tool_call. Esta suite lo
deja reproducible sin tocar Ollama ni la DB (workspace en tempdir,
httpx.AsyncClient.post mockeado con una secuencia de respuestas).

Corre desde /home/fruiz/jax/las_manos con:
  PYTHONPATH=/home/fruiz/jax/las_manos \
  /home/fruiz/jax/las_manos/.venv/bin/python \
  /home/fruiz/jax/las_manos/_worker_tool_loop_test.py

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from motor_registry import tool_authority, worker
from motor_registry.catalog import MotorCatalog
from motor_registry.job_store import JobStore

_FORBIDDEN = [".env", "secrets/", "private_keys/", "credentials/"]

_MOTOR_CFG = {
    "motors": {"jax_local": {
        "enabled": True, "provider": "ollama", "api_key_env": "",
        "api_url": "http://localhost:11434/v1", "model": "qwen3.6:35b-a3b-q4_K_M",
        "max_context_tokens": 0, "sandbox_only": True,
        "default_timeout_seconds": 300, "supports_reasoning": True,
        "transport": "ollama", "disable_reasoning": True,
        "has_tool_access": True,
    }},
    "capabilities": {
        "generate": {
            "allowed_callers": ["jacobs"], "risk_level": "low", "sandbox_only": True,
            "requires_human_gate": False, "max_execution_minutes": 15,
            "max_recursion_depth": 0, "output_schema": "",
        },
        "implementation": {
            "allowed_callers": ["jacobs"], "risk_level": "medium", "sandbox_only": True,
            "requires_human_gate": False, "max_execution_minutes": 30,
            "max_recursion_depth": 0, "output_schema": "code_patch.v1",
        },
        "file_read": {
            "allowed_callers": ["jacobs"], "risk_level": "medium", "sandbox_only": True,
            "requires_human_gate": False, "max_execution_minutes": 1,
            "max_recursion_depth": 0, "output_schema": "", "forbidden_paths": _FORBIDDEN,
        },
        "file_write": {
            "allowed_callers": ["jacobs"], "risk_level": "medium", "sandbox_only": True,
            "requires_human_gate": False, "max_execution_minutes": 1,
            "max_recursion_depth": 0, "output_schema": "", "forbidden_paths": _FORBIDDEN,
            "auditor_motor": "thot",
        },
    },
}
_MOTOR_CFG["motors"]["thot"] = {
    "enabled": True, "provider": "openai", "provider_id": "openai", "api_key_env": "",
    "api_url": "https://api.openai.com/v1", "model": "gpt-5.5", "max_context_tokens": 0,
    "sandbox_only": True, "default_timeout_seconds": 60, "supports_reasoning": False,
    "transport": "http_openai_compat", "disable_reasoning": False,
}


def _resp(*, content="", tool_calls=None, finish_reason="stop", usage=None):
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    payload = {
        "choices": [{"message": message, "finish_reason": finish_reason}],
        "usage": usage or {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
    }
    r = AsyncMock()
    r.json = lambda: payload
    r.raise_for_status = lambda: None
    return r


def _tc(name, arguments, call_id="call_1"):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}


class ToolLoopTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.workspace = Path(self._tmpdir.name)
        (self.workspace / "legit.txt").write_text("contenido de prueba")
        self.catalog = MotorCatalog(_MOTOR_CFG)
        self.store = JobStore(str(self.workspace / "jobs.jsonl"))
        self.kill_switch_path = str(self.workspace / "PAUSE")  # nunca existe
        ws_patch = patch.object(tool_authority, "WORKSPACE_ROOT", self.workspace.resolve())
        ws_patch.start()
        self.addCleanup(ws_patch.stop)

        import subprocess
        subprocess.run(["git", "init", "-q"], cwd=self.workspace, check=True)
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "add", "-A"], cwd=self.workspace, check=True)
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "fixture"], cwd=self.workspace, check=True)

        self.mock_telegram = AsyncMock(return_value={"ok": True, "message_id": 1, "error": None})
        telegram_patch = patch("jacobs.reaper.send_telegram_alert", self.mock_telegram)
        telegram_patch.start()
        self.addCleanup(telegram_patch.stop)

    async def _run(self, responses, context=None, capability="generate", motor="jax_local", **kwargs):
        job_id = self.store.create(
            caller="jacobs", capability=capability, motor=motor,
            trace_id="t", prompt="objetivo de prueba", recursion_depth=0,
        )
        with patch("httpx.AsyncClient.post", AsyncMock(side_effect=responses)) as mock_post:
            await worker.run(
                job_id=job_id, motor=motor, capability=capability,
                prompt="objetivo de prueba", context=context or {}, store=self.store,
                catalog=self.catalog, kill_switch_path=self.kill_switch_path,
                caller="jacobs", **kwargs,
            )
        return self.store._index[job_id], mock_post

    # --- 1. camino feliz: 2 turnos ---
    async def test_1_camino_feliz_dos_turnos(self):
        state, _ = await self._run([
            _resp(tool_calls=[_tc("read_file", {"path": "legit.txt"})], finish_reason="tool_calls"),
            _resp(content="El archivo dice: contenido de prueba", finish_reason="stop"),
        ])
        assert state["status"] == "completed", state
        assert state["_tool_loop_iterations"] == 2, state
        assert state["_tool_loop_history"][0]["results"][0]["decision"] == "executed", state
        assert state["result_summary"] == "El archivo dice: contenido de prueba", state

    # --- T1 (diagnóstico pipeline 19ad2c42-cdf): tools_for_call debe leer
    # motor_entry.has_tool_access, no un `if motor == "jax_local"` literal ---
    async def test_motor_sin_has_tool_access_no_recibe_catalogo_de_tools(self):
        """thot no tiene has_tool_access en el fixture (default False) --
        aunque el modelo "alucine" tool_calls en la respuesta mockeada, el
        payload real enviado a la API no debe incluir 'tools'."""
        with patch.object(worker, "resolve_credential_instrumented", AsyncMock(return_value="sk-fake")):
            state, mock_post = await self._run(
                [_resp(content="listo, sin tools", finish_reason="stop")],
                motor="thot", capability="critique",
            )
        assert state["status"] == "completed", state
        sent_payload = mock_post.call_args.kwargs["json"]
        assert "tools" not in sent_payload, sent_payload

    async def test_motor_con_has_tool_access_true_recibe_catalogo_aunque_no_sea_jax_local(self):
        """Prueba el mecanismo, no solo el resultado: si has_tool_access=True
        vive en la fuente de verdad (motor.has_tool_access, via catalog) para
        un motor que NO es 'jax_local', worker.py debe respetarlo -- si
        todavía comparara motor=="jax_local" a mano, este test fallaría
        aunque el dato real diga que sí puede."""
        self.catalog._motors["thot"].has_tool_access = True
        with patch.object(worker, "resolve_credential_instrumented", AsyncMock(return_value="sk-fake")):
            state, mock_post = await self._run(
                [_resp(content="listo, con tools ofrecidas", finish_reason="stop")],
                motor="thot", capability="critique",
            )
        assert state["status"] == "completed", state
        sent_payload = mock_post.call_args.kwargs["json"]
        assert "tools" in sent_payload, sent_payload

    # --- 2. cota de iteraciones ---
    async def test_2_agota_iteraciones_falla_explicito_no_completed(self):
        responses = [
            _resp(tool_calls=[_tc("read_file", {"path": f"x{i}.txt"})], finish_reason="tool_calls")
            for i in range(6)  # solo se consumen hasta la 5ta -- la 6ta no debe ni llamarse
        ]
        state, mock_post = await self._run(responses)
        assert state["status"] == "failed", state
        assert "5 iteraciones" in state["error"], state
        assert state["_tool_loop_iterations"] == 5, state
        assert mock_post.await_count == 5, mock_post.await_count

    # T1.b (2026-08-22, auditoria usage_writer): un job failed debe reportar
    # usage igual que uno completed -- antes la llamada vivía SOLO en la
    # rama de éxito.
    async def test_job_fallido_por_max_iteraciones_registra_usage_con_status_failed(self):
        responses = [
            _resp(tool_calls=[_tc("read_file", {"path": f"x{i}.txt"})], finish_reason="tool_calls")
            for i in range(6)
        ]
        with patch("motor_registry.usage_writer.record_motor_usage", AsyncMock()) as mock_usage:
            state, _ = await self._run(responses)
        assert state["status"] == "failed", state
        mock_usage.assert_awaited_once()
        args, kwargs = mock_usage.await_args
        assert kwargs["status"] == "failed", kwargs
        assert kwargs["job_id"] == state["job_id"], kwargs
        # 5 turnos completados antes del corte, 10/10 tokens cada uno (default de _resp)
        assert args[5] == 50, args  # tokens_in acumulados
        assert args[6] == 50, args  # tokens_out acumulados

    async def test_job_fallido_antes_de_llamar_al_llm_no_reporta_usage(self):
        """Kill switch activo ANTES de iniciar -- 0 tokens gastados, no debe
        haber llamada a record_motor_usage en absoluto (no fila vacía)."""
        import os as _os
        pause_path = str(self.workspace / "PAUSE_PRE")
        with open(pause_path, "w") as f:
            f.write("")
        try:
            with patch("motor_registry.usage_writer.record_motor_usage", AsyncMock()) as mock_usage:
                job_id = self.store.create(
                    caller="jacobs", capability="generate", motor="jax_local",
                    trace_id="t", prompt="x", recursion_depth=0,
                )
                await worker.run(
                    job_id=job_id, motor="jax_local", capability="generate", prompt="x",
                    context={}, store=self.store, catalog=self.catalog,
                    kill_switch_path=pause_path, caller="jacobs",
                )
            mock_usage.assert_not_awaited()
        finally:
            _os.remove(pause_path)

    # --- 3. presupuesto de tiempo ---
    async def test_3_presupuesto_de_tiempo_agotado_falla_explicito(self):
        responses = [
            _resp(tool_calls=[_tc("read_file", {"path": f"x{i}.txt"})], finish_reason="tool_calls")
            for i in range(5)
        ]
        # timeout_seconds=0 -> el deadline ya pasó antes de la primera
        # iteración: corta en la iteración 1, cero llamadas HTTP.
        state, mock_post = await self._run(responses, timeout_seconds=0)
        assert state["status"] == "failed", state
        assert "presupuesto de tiempo" in state["error"], state
        assert mock_post.await_count == 0, mock_post.await_count

    # --- 4. detección de loop ---
    async def test_4_deteccion_de_loop_corta_antes_de_agotar_iteraciones(self):
        responses = [
            _resp(tool_calls=[_tc("read_file", {"path": "x.txt"})], finish_reason="tool_calls")
            for _ in range(5)
        ]
        state, mock_post = await self._run(responses)
        assert state["status"] == "failed", state
        assert "Bucle detectado" in state["error"], state
        # LOOP_DETECTION_THRESHOLD=3 -> corta en la 3ra iteración, no llega a la 5ta
        assert mock_post.await_count == 3, mock_post.await_count

    # --- 5. tool rechazada por autoridad a mitad del bucle ---
    async def test_5_tool_rechazada_se_informa_al_modelo_y_el_bucle_sigue(self):
        state, _ = await self._run([
            _resp(tool_calls=[_tc("read_file", {"path": "/etc/passwd"})], finish_reason="tool_calls"),
            _resp(content="entendido, esa ruta no está permitida", finish_reason="stop"),
        ])
        assert state["status"] == "completed", state
        assert state["_tool_loop_history"][0]["results"][0]["decision"] == "rejected", state

    # --- 6. write_file (requires_human_gate) a mitad del bucle ---
    async def test_6_capability_con_requires_human_gate_se_informa_y_bucle_sigue(self):
        """T3 (Fase4): file_write ya NO tiene requires_human_gate=1 en el
        seed real, pero el MECANISMO sigue -- una capability que sí lo
        declare (T6: fuera del jail, efectos externos) debe rechazar mid-
        bucle igual que antes, informar al modelo, y el bucle sigue."""
        with patch.dict(self.catalog._capabilities["file_write"].__dict__, {"requires_human_gate": True}):
            state, _ = await self._run([
                _resp(tool_calls=[_tc("write_file", {"path": "x.txt", "content": "y"})], finish_reason="tool_calls"),
                _resp(content="no puedo escribir, continúo sin eso", finish_reason="stop"),
            ], context={"auditor": False})
        assert state["status"] == "completed", state
        reason = state["_tool_loop_history"][0]["results"][0]["reason"]
        assert "human_gate" in reason, reason
        assert not (self.workspace / "x.txt").exists()

    # --- 7. múltiples tool_calls en una respuesta: una autorizada, otra no ---
    async def test_7_multiples_tool_calls_una_autorizada_otra_rechazada(self):
        state, _ = await self._run([
            _resp(tool_calls=[
                _tc("read_file", {"path": "legit.txt"}, "call_a"),
                _tc("read_file", {"path": "/etc/passwd"}, "call_b"),
            ], finish_reason="tool_calls"),
            _resp(content="una funcionó, la otra no", finish_reason="stop"),
        ])
        results = state["_tool_loop_history"][0]["results"]
        assert results[0]["decision"] == "executed", results
        assert results[1]["decision"] == "rejected", results
        # ambas se ejecutaron pese al rechazo de la primera -- no se abortó el batch
        assert state["status"] == "completed", state

    # --- 8/9. integridad del historial: mensajes correctos + tool_call_id correlacionado ---
    async def test_8_9_historial_correcto_y_tool_call_id_correlacionado(self):
        # `messages` es la MISMA lista mutada in-place a lo largo del bucle
        # (payload["messages"] = messages, sin copiar -- correcto para
        # producción, donde httpx serializa a JSON al momento del POST real).
        # mock_post.call_args_list guarda una REFERENCIA, no una foto: para
        # ver el estado de CADA turno hay que copiar en el momento de la
        # llamada, no inspeccionar después de que el bucle ya mutó la lista.
        import copy
        snapshots = []
        responses = [
            _resp(tool_calls=[_tc("read_file", {"path": "legit.txt"}, "call_xyz")], finish_reason="tool_calls"),
            _resp(content="listo", finish_reason="stop"),
        ]
        call_iter = iter(responses)

        async def _capture_and_respond(*args, **kwargs):
            snapshots.append(copy.deepcopy(kwargs["json"]["messages"]))
            return next(call_iter)

        job_id = self.store.create(
            caller="jacobs", capability="generate", motor="jax_local",
            trace_id="t", prompt="objetivo de prueba", recursion_depth=0,
        )
        with patch("httpx.AsyncClient.post", AsyncMock(side_effect=_capture_and_respond)):
            await worker.run(
                job_id=job_id, motor="jax_local", capability="generate",
                prompt="objetivo de prueba", context={}, store=self.store,
                catalog=self.catalog, kill_switch_path=self.kill_switch_path,
                caller="jacobs",
            )

        assert len(snapshots) == 2, snapshots
        first_messages, second_messages = snapshots

        assert len(first_messages) == 1 and first_messages[0]["role"] == "user", first_messages

        # user original + assistant-con-tool_calls + tool-con-resultado
        assert len(second_messages) == 3, second_messages
        assert second_messages[0]["role"] == "user", second_messages
        assert second_messages[1]["role"] == "assistant", second_messages
        assert second_messages[1]["tool_calls"][0]["id"] == "call_xyz", second_messages
        assert second_messages[2]["role"] == "tool", second_messages
        assert second_messages[2]["tool_call_id"] == "call_xyz", second_messages
        assert second_messages[2]["content"] == "contenido de prueba", second_messages

    # --- extra: presupuesto acumulado de bytes leídos ---
    async def test_presupuesto_acumulado_de_lectura_corta_el_bucle(self):
        # cada archivo por debajo del cap POR-LLAMADA de tool_authority
        # (MAX_READ_BYTES=200_000) pero la SUMA de 3 supera el cap
        # ACUMULADO del bucle (MAX_TOTAL_READ_BYTES=500_000) -- confirma que
        # es un presupuesto distinto, no el mismo cap re-chequeado.
        assert tool_authority.MAX_READ_BYTES == 200_000, "ajustar el fixture si esto cambia"
        chunk = "x" * 180_000
        for name in ("big1.txt", "big2.txt", "big3.txt"):
            (self.workspace / name).write_text(chunk)
        state, mock_post = await self._run([
            _resp(tool_calls=[_tc("read_file", {"path": "big1.txt"}, "c1")], finish_reason="tool_calls"),
            _resp(tool_calls=[_tc("read_file", {"path": "big2.txt"}, "c2")], finish_reason="tool_calls"),
            _resp(tool_calls=[_tc("read_file", {"path": "big3.txt"}, "c3")], finish_reason="tool_calls"),
            _resp(content="no debería llegar acá", finish_reason="stop"),
        ])
        assert state["status"] == "failed", state
        assert "acumulado" in state["error"], state
        assert mock_post.await_count == 3, mock_post.await_count  # nunca llegó al 4to turno
        # confirma que los 2 primeros SÍ ejecutaron (180k cada uno, bajo el
        # cap por-llamada) -- el corte es por el acumulado, no por rechazo
        results = [it["results"][0]["decision"] for it in state["_tool_loop_history"]]
        assert results == ["executed", "executed", "executed"], results

    # --- extra: tool inventada a mitad del bucle ---
    async def test_tool_inventada_a_mitad_del_bucle_rechaza_y_sigue(self):
        state, _ = await self._run([
            _resp(tool_calls=[_tc("delete_everything", {"path": "x"})], finish_reason="tool_calls"),
            _resp(content="no tengo esa herramienta", finish_reason="stop"),
        ])
        assert state["status"] == "completed", state
        assert "no mapea" in state["_tool_loop_history"][0]["results"][0]["reason"], state

    # --- extra: usage se acumula across turnos, no solo el último ---
    async def test_usage_acumula_los_dos_turnos(self):
        state, _ = await self._run([
            _resp(tool_calls=[_tc("read_file", {"path": "legit.txt"})], finish_reason="tool_calls",
                  usage={"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}),
            _resp(content="listo", finish_reason="stop",
                  usage={"prompt_tokens": 150, "completion_tokens": 30, "total_tokens": 180}),
        ])
        assert state["_usage"]["prompt_tokens"] == 250, state
        assert state["_usage"]["completion_tokens"] == 50, state

    # --- extra: kill switch a mitad del bucle (no solo al inicio) ---
    async def test_kill_switch_a_mitad_del_bucle_falla_explicito(self):
        (self.workspace / "PAUSE").write_text("")
        state, _ = await self._run([
            _resp(tool_calls=[_tc("read_file", {"path": "legit.txt"})], finish_reason="tool_calls"),
        ])
        assert state["status"] == "failed", state
        assert "killed_by_switch" in state["error"], state

    # --- GAP2 Fase4: write_file en el bucle + auditor + notificación ---
    def _auditor_resp(self, verdict, reason="ok"):
        payload = {
            "choices": [{"message": {"role": "assistant", "content": json.dumps({"verdict": verdict, "reason": reason})},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
        }
        r = AsyncMock(); r.json = lambda: payload; r.raise_for_status = lambda: None
        return r

    async def test_write_file_ejecuta_commitea_y_auditor_pass_notifica_verde(self):
        with patch.object(worker, "resolve_credential_instrumented", AsyncMock(return_value="sk-fake")):
            state, _ = await self._run([
                _resp(tool_calls=[_tc("write_file", {"path": "out.html", "content": "<h1>hola</h1>"})], finish_reason="tool_calls"),
                _resp(content="listo, escribí el archivo", finish_reason="stop"),
                self._auditor_resp("pass", "se ve bien"),
            ])
        assert state["status"] == "completed", state
        assert (self.workspace / "out.html").read_text() == "<h1>hola</h1>"
        self.mock_telegram.assert_awaited_once()
        msg = self.mock_telegram.await_args.args[0]
        assert "🟢" in msg and "thot" in msg and "pass" in msg, msg

    async def test_auditor_revert_deshace_con_git_y_notifica_rojo(self):
        with patch.object(worker, "resolve_credential_instrumented", AsyncMock(return_value="sk-fake")):
            state, _ = await self._run([
                _resp(tool_calls=[_tc("write_file", {"path": "malo.html", "content": "contenido malo"})], finish_reason="tool_calls"),
                _resp(content="listo", finish_reason="stop"),
                self._auditor_resp("revert", "contenido inaceptable"),
            ])
        assert state["status"] == "completed", state  # el JOB sigue completed -- el revert es post-hoc
        assert not (self.workspace / "malo.html").exists(), "el revert debió deshacer el archivo"
        msg = self.mock_telegram.await_args.args[0]
        assert "🔴" in msg and "REVERTIDO" in msg, msg

    async def test_auditor_override_por_request_cambia_el_auditor(self):
        self.catalog._motors["kimi"] = self.catalog._motors["thot"]  # reusa fixture de motor http_openai_compat
        with patch.object(worker, "resolve_credential_instrumented", AsyncMock(return_value="sk-fake")):
            state, _ = await self._run([
                _resp(tool_calls=[_tc("write_file", {"path": "x.txt", "content": "y"})], finish_reason="tool_calls"),
                _resp(content="listo", finish_reason="stop"),
                self._auditor_resp("pass"),
            ], context={"auditor": "kimi"})
        assert state["status"] == "completed", state
        msg = self.mock_telegram.await_args.args[0]
        assert "kimi" in msg and "override por request" in msg, msg

    async def test_auditor_desactivado_por_request_no_llama_auditor(self):
        state, mock_post = await self._run([
            _resp(tool_calls=[_tc("write_file", {"path": "x.txt", "content": "y"})], finish_reason="tool_calls"),
            _resp(content="listo", finish_reason="stop"),
        ], context={"auditor": False})
        assert state["status"] == "completed", state
        assert mock_post.await_count == 2, mock_post.await_count  # nunca llamó al auditor
        msg = self.mock_telegram.await_args.args[0]
        assert "sin auditar" in msg or "ninguno" in msg, msg

    async def test_auto_revision_rechazada_si_auditor_es_el_mismo_motor(self):
        state, mock_post = await self._run([
            _resp(tool_calls=[_tc("write_file", {"path": "x.txt", "content": "y"})], finish_reason="tool_calls"),
            _resp(content="listo", finish_reason="stop"),
        ], context={"auditor": "jax_local"})  # mismo motor que produjo el trabajo
        assert state["status"] == "completed", state
        assert mock_post.await_count == 2, mock_post.await_count  # nunca se llamó a sí mismo como auditor
        msg = self.mock_telegram.await_args.args[0]
        assert "auto-revisión" in msg or "ninguno" in msg, msg

    async def test_read_after_write_lee_lo_escrito_no_cache(self):
        state, _ = await self._run([
            _resp(tool_calls=[_tc("write_file", {"path": "nuevo.txt", "content": "recien escrito"}, "c1")], finish_reason="tool_calls"),
            _resp(tool_calls=[_tc("read_file", {"path": "nuevo.txt"}, "c2")], finish_reason="tool_calls"),
            _resp(content="el archivo dice: recien escrito", finish_reason="stop"),
        ], context={"auditor": False})
        assert state["status"] == "completed", state
        read_result = state["_tool_loop_history"][1]["results"][0]
        assert read_result["decision"] == "executed", read_result
        assert read_result["content"] == "recien escrito", read_result


    # --- T2: fail-open de output_validator corregido ---
    async def test_t2_salida_invalida_reintenta_una_vez_y_completa_si_el_reintento_es_valido(self):
        valid_json = json.dumps({"diff": "x", "files_modified": ["a.py"], "description": "y"})
        state, mock_post = await self._run([
            _resp(content="esto no es JSON", finish_reason="stop"),
            _resp(content=valid_json, finish_reason="stop"),
        ], capability="implementation", context={"auditor": False})
        assert state["status"] == "completed", state
        assert mock_post.await_count == 2, mock_post.await_count  # confirma que SÍ reintentó

    async def test_t2_salida_invalida_tras_reintento_falla_explicito_nunca_completed(self):
        state, mock_post = await self._run([
            _resp(content="esto no es JSON", finish_reason="stop"),
            _resp(content="tampoco esto", finish_reason="stop"),
        ], capability="implementation", context={"auditor": False})
        assert state["status"] == "failed", state
        assert "schema" in state["error"], state
        assert mock_post.await_count == 2, mock_post.await_count  # un solo reintento, no más


if __name__ == "__main__":
    unittest.main(verbosity=2)
