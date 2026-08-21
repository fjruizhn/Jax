#!/usr/bin/env python3
"""
Motor Registry — worker: max_tokens explícito + captura de finish_reason/usage.

Bug real encontrado el 2026-08-10 revisando un pipeline: un job de kimi
(motor de razonamiento, supports_reasoning=true) devolvió `content` cortado
a mitad de palabra (861 bytes en la DB, contra 20-25KB de los otros facets
del mismo pipeline). Reproducido en vivo contra la API real de Moonshot:
sin `max_tokens` en el payload, el razonamiento ya consumía más de la mitad
del completion budget con un prompt trivial (usage real:
completion_tokens=866, reasoning_tokens=467). Con un prompt largo y
demandante como el que falló, es plausible que el razonamiento se coma todo
el budget antes de escribir la respuesta final.

Dos fixes cubiertos acá:
1. `_call_kimi` manda `max_tokens` en el payload cuando el motor lo declara
   en su config (`MotorEntry.max_tokens`).
2. El job guardado captura `_finish_reason`/`_usage` — antes se descartaban
   por completo, así que un corte como el de hoy era indiagnosticable
   después del hecho (no se podía saber si fue `finish_reason='length'` u
   otra causa).

Corre desde /home/fruiz/jax/las_manos con:
  PYTHONPATH=/home/fruiz/jax/las_manos \
  /home/fruiz/jax/las_manos/.venv/bin/python \
  /home/fruiz/jax/las_manos/_worker_max_tokens_test.py

Solo mockea el límite real de red (httpx.AsyncClient.post) y la resolución
de credencial (evita tocar la DB real) — JobStore y MotorCatalog corren de
verdad, contra un archivo JSONL temporal.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from motor_registry.catalog import MotorCatalog
from motor_registry.job_store import JobStore
from motor_registry import worker

_MOTOR_CFG = {
    "motors": {
        "kimi": {
            "enabled": True,
            "provider": "kimi",
            "transport": "http_openai_compat",
            # provider_id (2026-08-19, Task 3): en producción esto viene del
            # JOIN en MotorCatalog.from_db() (ver _catalog_from_db_test.py).
            # El constructor dict-based (usado solo por tests) no lo derivaba
            # de nada -- se agrega acá, mismo patrón que 'transport' arriba,
            # para que worker.py (que ahora lee motor_entry.provider_id en
            # vez del _MOTOR_PROVIDER_MAP hardcodeado eliminado) siga
            # resolviendo "moonshot" para kimi en este fixture.
            "provider_id": "moonshot",
            "api_key_env": "KIMI_API_KEY",
            "api_url": "https://api.moonshot.ai/v1/chat/completions",
            "model": "kimi-k2.7-code",
            "max_context_tokens": 256000,
            "sandbox_only": True,
            "default_timeout_seconds": 600,
            "supports_reasoning": True,
            "reasoning_default_visibility": "audit_only",
            "max_tokens": 8000,
        },
    },
    "capabilities": {
        "implementation": {
            "allowed_motors": ["kimi"],
            "allowed_callers": ["jacobs"],
            "risk_level": "low",
            "sandbox_only": True,
            "requires_human_gate": False,
            "max_execution_minutes": 10,
            "max_recursion_depth": 0,
            "output_schema": "",
        },
    },
}


def _fake_response(*, content, reasoning_content=None, reasoning=None, finish_reason="stop", usage=None):
    message = {"content": content}
    if reasoning_content is not None:
        message["reasoning_content"] = reasoning_content
    if reasoning is not None:
        message["reasoning"] = reasoning
    payload = {
        "choices": [{"message": message, "finish_reason": finish_reason}],
        "usage": usage or {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }
    resp = AsyncMock()
    resp.json = lambda: payload
    resp.raise_for_status = lambda: None
    return resp


class WorkerMaxTokensTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.store = JobStore(str(Path(self._tmpdir.name) / "jobs.jsonl"))
        self.catalog = MotorCatalog(_MOTOR_CFG)
        self.kill_switch_path = str(Path(self._tmpdir.name) / "PAUSE")  # nunca existe

    def tearDown(self):
        self._tmpdir.cleanup()

    async def _run_job(self, fake_resp, **run_kwargs):
        job_id = self.store.create(
            caller="jacobs", capability="implementation", motor="kimi",
            trace_id="t1", prompt="prompt de prueba", recursion_depth=0,
        )
        with patch.object(worker, "resolve_credential_instrumented", AsyncMock(return_value="sk-fake")), \
             patch("httpx.AsyncClient.post", AsyncMock(return_value=fake_resp)) as mock_post:
            await worker.run(
                job_id=job_id, motor="kimi", capability="implementation",
                prompt="prompt de prueba", context={}, store=self.store,
                catalog=self.catalog, kill_switch_path=self.kill_switch_path,
                **run_kwargs,
            )
        return job_id, mock_post

    async def test_payload_incluye_max_tokens_del_motor(self):
        job_id, mock_post = await self._run_job(_fake_response(content="respuesta completa"))
        _url, kwargs = mock_post.call_args
        sent_payload = kwargs["json"]
        assert sent_payload.get("max_tokens") == 8000, f"max_tokens no viajo en el payload: {sent_payload}"

    async def test_job_guarda_finish_reason_y_usage(self):
        job_id, _ = await self._run_job(_fake_response(
            content="respuesta completa", finish_reason="stop",
            usage={"prompt_tokens": 46, "completion_tokens": 866, "total_tokens": 912,
                   "completion_tokens_details": {"reasoning_tokens": 467}},
        ))
        state = self.store._index[job_id]
        assert state.get("_finish_reason") == "stop", state
        assert state.get("_usage", {}).get("completion_tokens") == 866, state

    async def test_run_con_identidad_llama_a_record_motor_usage(self):
        """Task 7: cuando el dispatch trae user_id/tenant_id (propagados por
        jacobs/executor.py, Task 6), worker.run debe llamar a
        record_motor_usage con provider_id/model/tokens reales -- sin tocar
        la logica de max_tokens/finish_reason/usage capturada arriba."""
        with patch("motor_registry.usage_writer.record_motor_usage", AsyncMock()) as mock_record:
            job_id, _ = await self._run_job(
                _fake_response(
                    content="respuesta completa", finish_reason="stop",
                    usage={"prompt_tokens": 46, "completion_tokens": 866, "total_tokens": 912},
                ),
                user_id="1", tenant_id="77",
            )
        mock_record.assert_awaited_once_with(
            "1", "77", "kimi", "moonshot", "kimi-k2.7-code", 46, 866,
            job_id=job_id, status="completed",
        )

    async def test_run_sin_identidad_pasa_none_a_record_motor_usage(self):
        """Compat con dispatches viejos / sin Jacobs: worker.run no filtra
        user_id/tenant_id ausentes antes de llamar -- el guard fail-soft
        (loguear y escribir con NULL, no descartar) vive en
        record_motor_usage, cubierto por _motor_usage_writer_test.py::
        test_record_motor_usage_sin_identidad_escribe_con_null_y_loguea.
        Aca solo confirmamos que worker sigue completando el job igual (ver
        test_job_guarda_finish_reason_y_usage) y que pasa None tal cual,
        sin inventar un default."""
        with patch("motor_registry.usage_writer.record_motor_usage", AsyncMock()) as mock_record:
            job_id, _ = await self._run_job(
                _fake_response(content="respuesta completa", finish_reason="stop"),
            )
        mock_record.assert_awaited_once_with(
            None, None, "kimi", "moonshot", "kimi-k2.7-code", 10, 20,
            job_id=job_id, status="completed",
        )
        assert self.store._index[job_id]["status"] == "completed"

    async def test_finish_reason_length_queda_registrado(self):
        """El caso real que motivo el fix: si Moonshot corta por limite de
        tokens, finish_reason='length' debe quedar capturado -- antes se
        perdia por completo y el corte era indiagnosticable despues."""
        job_id, _ = await self._run_job(_fake_response(
            content="respuesta a medi", finish_reason="length",
        ))
        state = self.store._index[job_id]
        assert state.get("_finish_reason") == "length", state
        assert state["status"] == "completed"  # el job igual se marca completed -- el dato queda para diagnostico, no bloquea

    async def test_transport_ollama_no_resuelve_credencial(self):
        """Guard igual a facet_resolver.py:81-82 -- transport='ollama' nunca
        llama a resolve_credential_instrumented. Si lo hiciera, este test
        lo detecta porque el mock de credential está seteado para explotar."""
        cfg = {
            "motors": {"jax_local": {
                "enabled": True, "provider": "ollama", "api_key_env": "",
                "api_url": "http://localhost:11434/v1", "model": "qwen3-coder:30b",
                "max_context_tokens": 0, "sandbox_only": True,
                "default_timeout_seconds": 300, "supports_reasoning": False,
                "transport": "ollama",
            }},
            "capabilities": {"implementation": _MOTOR_CFG["capabilities"]["implementation"]},
        }
        catalog = MotorCatalog(cfg)
        store = JobStore(str(Path(self._tmpdir.name) / "jobs2.jsonl"))
        job_id = store.create(
            caller="jacobs", capability="implementation", motor="jax_local",
            trace_id="t2", prompt="prompt de prueba", recursion_depth=0,
        )
        boom = AsyncMock(side_effect=AssertionError("no debería resolver credencial para ollama"))
        with patch.object(worker, "resolve_credential_instrumented", boom), \
             patch("httpx.AsyncClient.post", AsyncMock(return_value=_fake_response(content="listo"))):
            await worker.run(
                job_id=job_id, motor="jax_local", capability="implementation",
                prompt="prompt de prueba", context={}, store=store,
                catalog=catalog, kill_switch_path=self.kill_switch_path,
            )
        assert store._index[job_id]["status"] == "completed", store._index[job_id]

    async def test_transport_desconocido_falla_explicito_no_silencioso(self):
        cfg = {
            "motors": {"futuro": {
                "enabled": True, "provider": "x", "api_key_env": "", "api_url": "",
                "model": "x", "max_context_tokens": 0, "sandbox_only": True,
                "default_timeout_seconds": 300, "supports_reasoning": False,
                "transport": "subprocess",
            }},
            "capabilities": {"implementation": _MOTOR_CFG["capabilities"]["implementation"]},
        }
        catalog = MotorCatalog(cfg)
        store = JobStore(str(Path(self._tmpdir.name) / "jobs3.jsonl"))
        job_id = store.create(
            caller="jacobs", capability="implementation", motor="futuro",
            trace_id="t3", prompt="prompt de prueba", recursion_depth=0,
        )
        await worker.run(
            job_id=job_id, motor="futuro", capability="implementation",
            prompt="prompt de prueba", context={}, store=store,
            catalog=catalog, kill_switch_path=self.kill_switch_path,
        )
        state = store._index[job_id]
        assert state["status"] == "failed", state

    async def _run_ollama_job(self, *, disable_reasoning=True, context=None, tmp_suffix="4"):
        """T2 (2026-08-19): fixture ollama con disable_reasoning explicito --
        distinto del de test_transport_ollama_no_resuelve_credencial (ese no
        lo declara, se apoya en el default True de MotorEntry)."""
        cfg = {
            "motors": {"jax_local": {
                "enabled": True, "provider": "ollama", "api_key_env": "",
                "api_url": "http://localhost:11434/v1", "model": "qwen3.6:35b-a3b-q4_K_M",
                "max_context_tokens": 0, "sandbox_only": True,
                "default_timeout_seconds": 300, "supports_reasoning": True,
                "transport": "ollama", "disable_reasoning": disable_reasoning,
            }},
            "capabilities": {"implementation": _MOTOR_CFG["capabilities"]["implementation"]},
        }
        catalog = MotorCatalog(cfg)
        store = JobStore(str(Path(self._tmpdir.name) / f"jobs{tmp_suffix}.jsonl"))
        job_id = store.create(
            caller="jacobs", capability="implementation", motor="jax_local",
            trace_id=f"t{tmp_suffix}", prompt="prompt de prueba", recursion_depth=0,
        )
        with patch("httpx.AsyncClient.post", AsyncMock(return_value=_fake_response(content="listo"))) as mock_post:
            await worker.run(
                job_id=job_id, motor="jax_local", capability="implementation",
                prompt="prompt de prueba", context=context or {}, store=store,
                catalog=catalog, kill_switch_path=self.kill_switch_path,
            )
        return mock_post

    async def test_ollama_con_disable_reasoning_manda_reasoning_effort_none(self):
        """T2: motor.disable_reasoning=True (default) + transport='ollama'
        (unico camino verificado, ver docstring de _call_http_openai_compat)
        -> el payload real manda reasoning_effort='none'."""
        mock_post = await self._run_ollama_job(disable_reasoning=True)
        _url, kwargs = mock_post.call_args
        assert kwargs["json"].get("reasoning_effort") == "none", kwargs["json"]

    async def test_ollama_con_override_reasoning_no_manda_reasoning_effort(self):
        """T2: context={'reasoning': True} (por job/request) gana sobre el
        default del motor -- un caller que sí necesita razonamiento para ESE
        job no tiene que tocar la config del motor."""
        mock_post = await self._run_ollama_job(disable_reasoning=True, context={"reasoning": True})
        _url, kwargs = mock_post.call_args
        assert "reasoning_effort" not in kwargs["json"], kwargs["json"]

    async def test_ollama_con_motor_disable_reasoning_false_no_manda_nada(self):
        """T2: un motor ollama que declare disable_reasoning=False (ninguno
        hoy, pero el catalogo lo permite) no manda el parametro -- mismo
        comportamiento que antes de T2, cero regresion para ese caso."""
        mock_post = await self._run_ollama_job(disable_reasoning=False)
        _url, kwargs = mock_post.call_args
        assert "reasoning_effort" not in kwargs["json"], kwargs["json"]

    async def test_kimi_nunca_recibe_reasoning_effort(self):
        """T2: transport='http_openai_compat' (Kimi/Ada) queda FUERA del gate
        a propósito -- verificado real 2026-08-19 que Moonshot RECHAZA este
        motor exacto con 400 ('only type=enabled is allowed for this
        model') si se manda reasoning_effort. kimi tiene supports_reasoning
        =True en el fixture (_MOTOR_CFG), que es justo el caso que podría
        tentar a alguien a generalizar el gate por supports_reasoning en vez
        de por transport -- este test existe para que ese cambio falle acá
        primero, no en producción contra la API real de Moonshot."""
        job_id, mock_post = await self._run_job(_fake_response(content="respuesta completa"))
        _url, kwargs = mock_post.call_args
        assert "reasoning_effort" not in kwargs["json"], kwargs["json"]

    async def test_ollama_reasoning_key_se_captura_como_reasoning_content(self):
        """T3 (2026-08-19): Ollama devuelve la clave 'reasoning', no
        'reasoning_content' (verificado real contra /v1/chat/completions,
        qwen3.6:35b-a3b). Sin el fallback, _reasoning_content quedaba null
        SIEMPRE para jax_local pese a que el modelo sí razonó."""
        cfg = {
            "motors": {"jax_local": {
                "enabled": True, "provider": "ollama", "api_key_env": "",
                "api_url": "http://localhost:11434/v1", "model": "qwen3.6:35b-a3b-q4_K_M",
                "max_context_tokens": 0, "sandbox_only": True,
                "default_timeout_seconds": 300, "supports_reasoning": True,
                "transport": "ollama", "disable_reasoning": False,  # thinking encendido para este job
            }},
            "capabilities": {"implementation": _MOTOR_CFG["capabilities"]["implementation"]},
        }
        catalog = MotorCatalog(cfg)
        store = JobStore(str(Path(self._tmpdir.name) / "jobs5.jsonl"))
        job_id = store.create(
            caller="jacobs", capability="implementation", motor="jax_local",
            trace_id="t5", prompt="prompt de prueba", recursion_depth=0,
        )
        fake_resp = _fake_response(content="4", reasoning="1+1 es 2, 2+2 es 4...")
        with patch("httpx.AsyncClient.post", AsyncMock(return_value=fake_resp)):
            await worker.run(
                job_id=job_id, motor="jax_local", capability="implementation",
                prompt="prompt de prueba", context={}, store=store,
                catalog=catalog, kill_switch_path=self.kill_switch_path,
            )
        state = store._index[job_id]
        assert state.get("_reasoning_content") == "1+1 es 2, 2+2 es 4...", state


if __name__ == "__main__":
    unittest.main(verbosity=2)
