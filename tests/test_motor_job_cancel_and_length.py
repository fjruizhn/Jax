"""
Motor Registry — tres defectos que abortaron el pipeline b8f80733 (2026-09-12).

El paso 04 (kimi/generate) de "esquematizar el ERP" se perdió así:
  1. kimi gastó 7232 de sus 8000 tokens de salida razonando; la respuesta
     llegó cortada (finish_reason=length) como texto libre, no JSON.
  2. El worker la trató como "no cumple el schema" y REINTENTÓ una vez --
     una segunda llamada destinada a cortarse igual. Entre las dos se pasó
     de los 300 s del paso y Jacobs abortó el pipeline.
  3. Jacobs cortó su lado, pero el job siguió vivo en LAS MANOS: terminó
     10:02:59, se cobró ($0.216) y nadie usó la salida. `POST
     /motor/job/{id}/cancel` solo cambiaba la etiqueta; ningún código la
     leía.

Además, cada llamada al modelo usaba `motor.default_timeout_seconds` (600)
aunque al job le quedara menos presupuesto -- una llamada podía pasarse
del paso entero.

Solo se mockea el borde de red (el dispatcher de transporte), la
credencial y la escritura de costo. JobStore y MotorCatalog corren de
verdad, contra un JSONL temporal: nunca contra logs/motor_jobs.jsonl real.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from motor_registry import job_tasks, worker
from motor_registry.catalog import MotorCatalog
from motor_registry.job_store import JobStore
from motor_registry.models import JobStatus

_CAP_COMMON = {
    "allowed_motors": ["kimi"],
    "allowed_callers": ["jacobs"],
    "risk_level": "low",
    "sandbox_only": True,
    "requires_human_gate": False,
    "max_execution_minutes": 15,
    "max_recursion_depth": 0,
}

_CFG = {
    "motors": {
        "kimi": {
            "enabled": True,
            "provider": "kimi",
            "transport": "http_openai_compat",
            "provider_id": "moonshot",
            "api_key_env": "KIMI_API_KEY",
            "api_url": "https://api.moonshot.ai/v1/chat/completions",
            "model": "kimi-k3",
            "max_context_tokens": 256000,
            "sandbox_only": True,
            "default_timeout_seconds": 600,
            "supports_reasoning": True,
            "reasoning_default_visibility": "audit_only",
            "max_tokens": 8000,
        },
    },
    "capabilities": {
        # Schema real de producción: el caso del incidente.
        "generate": {**_CAP_COMMON, "output_schema": "generate.v1"},
        # Sin schema: aísla el presupuesto de tiempo de la validación.
        "implementation": {**_CAP_COMMON, "output_schema": ""},
        # Schema CON campos: desde 2026-09-12 los declarados-pendientes
        # (generate.v1, ...) aceptan texto libre, así que "schema inválido"
        # se prueba con uno que sí exige algo.
        "refactor": {**_CAP_COMMON, "output_schema": "code_patch.v1"},
    },
}


def _response(content: str, finish_reason: str = "stop", usage: dict | None = None) -> dict:
    return {
        "choices": [{"message": {"content": content}, "finish_reason": finish_reason}],
        "usage": usage or {"prompt_tokens": 10, "completion_tokens": 20},
    }


class _Base(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.store = JobStore(str(Path(self._tmpdir.name) / "jobs.jsonl"))
        self.catalog = MotorCatalog(_CFG)
        self.kill_switch_path = str(Path(self._tmpdir.name) / "PAUSE")  # nunca existe
        self.usage = AsyncMock()
        self._patches = [
            patch.object(worker, "resolve_credential_instrumented", AsyncMock(return_value="sk-fake")),
            patch("motor_registry.usage_writer.record_motor_usage", self.usage),
            # Red de seguridad: si el transporte falso no quedara puesto, el
            # worker saldría a la API real. Pasó en el primer borrador de este
            # archivo (el patch se cerraba antes del await) -- que falle fuerte.
            patch("httpx.AsyncClient.post", AsyncMock(side_effect=AssertionError("llamada de red real en un test"))),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()
        self._tmpdir.cleanup()

    def _new_job(self, capability: str) -> str:
        return self.store.create(
            caller="jacobs", capability=capability, motor="kimi",
            trace_id="t", prompt="prompt", recursion_depth=0,
        )

    async def _run(self, job_id: str, capability: str, call_fn, timeout_seconds=None):
        # El await va DENTRO del with: worker.run lee _TRANSPORT_DISPATCH al
        # ejecutarse, no al crear la corrutina.
        with patch.dict(worker._TRANSPORT_DISPATCH, {"http_openai_compat": call_fn}):
            return await worker.run(
                job_id=job_id, motor="kimi", capability=capability, prompt="prompt",
                context={}, store=self.store, catalog=self.catalog,
                kill_switch_path=self.kill_switch_path, timeout_seconds=timeout_seconds,
            )

    def _state(self, job_id: str) -> dict:
        return self.store._index[job_id]


class CorteDeTokensTest(_Base):
    async def test_corte_por_length_con_schema_invalido_falla_sin_reintentar(self):
        """El caso exacto del incidente: 7232 de 8000 tokens razonando,
        salida cortada como texto libre. Reintentar repite la misma llamada
        con el mismo techo: se corta igual y duplica el costo."""
        calls = []

        async def fake_call(**kwargs):
            calls.append(kwargs)
            return _response(
                "## Esquema del ERP\nMódulos: inventario, ventas, cont",
                finish_reason="length",
                usage={"prompt_tokens": 551, "completion_tokens": 8000,
                       "completion_tokens_details": {"reasoning_tokens": 7232}},
            )

        job_id = self._new_job("refactor")
        await self._run(job_id, "refactor", fake_call)

        state = self._state(job_id)
        assert len(calls) == 1, f"reintentó una salida cortada por tokens: {len(calls)} llamadas"
        assert state["status"] == JobStatus.FAILED.value, state
        assert "max_tokens" in (state["error"] or ""), state["error"]
        assert "7232" in state["error"], "el error debe decir cuánto se fue en razonamiento"

    async def test_corte_por_length_sin_schema_falla_no_sale_como_completo(self):
        """Decisión de Fernando, 2026-09-14 (DEUDA.md, anotados b8f80733):
        una salida cortada por tokens SIN schema quedaba `completed` desde el
        2026-08-10 ("el dato queda para diagnóstico"), y el paso siguiente la
        usaba sin saber que le faltaba el final -- fail-open (P10). Ahora falla
        igual que el caso con schema: sin reintento, diciendo qué subir."""
        calls = []

        async def fake_call(**kwargs):
            calls.append(kwargs)
            return _response(
                "El módulo de ventas registra cada factura y luego",
                finish_reason="length",
                usage={"prompt_tokens": 400, "completion_tokens": 8000,
                       "completion_tokens_details": {"reasoning_tokens": 6100}},
            )

        job_id = self._new_job("implementation")
        await self._run(job_id, "implementation", fake_call)

        state = self._state(job_id)
        assert len(calls) == 1, f"reintentó una salida cortada por tokens: {len(calls)} llamadas"
        assert state["status"] == JobStatus.FAILED.value, state
        assert "max_tokens" in (state["error"] or ""), state["error"]
        assert "6100" in state["error"], "el error debe decir cuánto se fue en razonamiento"

    async def test_corte_por_length_con_tool_calls_falla_sin_ejecutar_herramientas(self):
        """Hueco hermano, encontrado por la revisión de jax#152: si la respuesta
        cortada trae tool_calls, el chequeo de corte no corría y se ejecutaban
        herramientas con argumentos posiblemente truncados -- un write_file
        completo podía escribir antes de que llegara el siguiente, roto. Misma
        propiedad (P10), así que entra en el mismo arreglo."""
        calls = []

        async def fake_call(**kwargs):
            calls.append(kwargs)
            return {
                "choices": [{
                    "message": {"content": "", "tool_calls": [{
                        "id": "t1", "type": "function",
                        "function": {"name": "write_file", "arguments": '{"path": "a.txt", "content": "hola mu'},
                    }]},
                    "finish_reason": "length",
                }],
                "usage": {"prompt_tokens": 300, "completion_tokens": 8000},
            }

        ejecutar = AsyncMock()
        job_id = self._new_job("implementation")
        with patch.object(worker, "authorize_and_execute_tool_call", ejecutar):
            await self._run(job_id, "implementation", fake_call)

        state = self._state(job_id)
        assert len(calls) == 1, f"volvió a llamar al modelo: {len(calls)} llamadas"
        assert ejecutar.await_count == 0, "ejecutó una herramienta con argumentos de una salida cortada"
        assert state["status"] == JobStatus.FAILED.value, state
        assert "max_tokens" in (state["error"] or ""), state["error"]

    async def test_corte_con_motor_sin_max_tokens_no_dice_subir_cero(self):
        """Con max_tokens=0 el payload no lleva el campo y el corte viene del
        límite del proveedor o del contexto: "Subir motor.max_tokens (0)" no
        orienta a nadie. Revisión de jax#152."""
        cfg = {**_CFG, "motors": {"kimi": {**_CFG["motors"]["kimi"], "max_tokens": 0}}}
        self.catalog = MotorCatalog(cfg)

        async def fake_call(**kwargs):
            return _response("texto que se corta a mit", finish_reason="length")

        job_id = self._new_job("implementation")
        await self._run(job_id, "implementation", fake_call)

        state = self._state(job_id)
        assert state["status"] == JobStatus.FAILED.value, state
        assert "no declara max_tokens" in (state["error"] or ""), state["error"]
        assert "max_tokens (0)" not in state["error"], state["error"]

    async def test_sin_schema_y_sin_corte_sigue_completando(self):
        """Control: sin schema y con finish_reason=stop, el job se completa
        como siempre. El arreglo no puede tocar el caso normal."""
        async def fake_call(**kwargs):
            return _response("respuesta completa", finish_reason="stop")

        job_id = self._new_job("implementation")
        await self._run(job_id, "implementation", fake_call)
        assert self._state(job_id)["status"] == JobStatus.COMPLETED.value

    async def test_schema_invalido_sin_corte_sigue_reintentando_una_vez(self):
        """Control: el reintento de schema sigue existiendo para lo que sí
        arregla -- una respuesta completa que no respetó el formato."""
        calls = []

        async def fake_call(**kwargs):
            calls.append(kwargs)
            return _response("texto libre completo, no JSON", finish_reason="stop")

        job_id = self._new_job("refactor")
        await self._run(job_id, "refactor", fake_call)

        assert len(calls) == 2, f"se esperaba 1 reintento, hubo {len(calls) - 1}"
        assert self._state(job_id)["status"] == JobStatus.FAILED.value


class PresupuestoPorLlamadaTest(_Base):
    async def test_cada_llamada_recibe_lo_que_le_queda_al_job(self):
        seen = []

        async def fake_call(**kwargs):
            seen.append(kwargs["timeout"])
            return _response("ok")

        job_id = self._new_job("implementation")
        await self._run(job_id, "implementation", fake_call, timeout_seconds=100)

        assert seen, "no hubo llamada"
        assert 0 < seen[0] <= 100, f"la llamada recibió {seen[0]} s con 100 s de presupuesto"

    async def test_sin_presupuesto_usa_el_timeout_del_motor(self):
        seen = []

        async def fake_call(**kwargs):
            seen.append(kwargs["timeout"])
            return _response("ok")

        job_id = self._new_job("implementation")
        await self._run(job_id, "implementation", fake_call, timeout_seconds=None)

        assert seen == [600.0], seen


class CancelacionRealTest(_Base):
    async def test_cancel_corta_la_llamada_en_vuelo(self):
        started = asyncio.Event()
        outcome = {}

        async def slow_call(**kwargs):
            started.set()
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                outcome["call_cancelled"] = True
                raise
            return _response("no debería llegar")

        job_id = self._new_job("implementation")
        task = asyncio.create_task(self._run(job_id, "implementation", slow_call))
        job_tasks.register(job_id, task)
        await asyncio.wait_for(started.wait(), timeout=2)

        assert job_tasks.cancel(job_id) is True
        with self.assertRaises(asyncio.CancelledError):
            await task

        assert outcome.get("call_cancelled"), "la llamada al modelo siguió viva"
        assert self._state(job_id)["status"] == JobStatus.CANCELLED.value
        assert job_tasks.cancel(job_id) is False, "la tarea terminada debe salir del registro"

    async def test_cancel_de_un_job_desconocido_no_hace_nada(self):
        assert job_tasks.cancel("no-existe") is False

    async def test_completed_tardio_no_pisa_un_cancelled(self):
        """La cancelación puede llegar mientras la respuesta ya viene en
        camino. Lo que el caller canceló no puede reaparecer como
        `completed` -- que fue lo que pasó con la salida de kimi."""
        job_id = self._new_job("implementation")

        async def call_then_cancelled(**kwargs):
            self.store.update(job_id, status=JobStatus.CANCELLED.value)
            return _response("respuesta que llegó tarde")

        await self._run(job_id, "implementation", call_then_cancelled)

        assert self._state(job_id)["status"] == JobStatus.CANCELLED.value, self._state(job_id)
        self.usage.assert_awaited_once()
        assert self.usage.await_args.kwargs["status"] == "cancelled"


if __name__ == "__main__":
    unittest.main(verbosity=2)
