#!/usr/bin/env python3
"""Motor Registry — worker: detector de estructura (capas 2 y 4 anti-fabricación).

Antes de este cambio, una salida que AFIRMABA un archivo inexistente
(`files_modified: ["services/bodegas.py"]`, o un diff `--- a/no/existe.py`)
terminaba COMPLETED. Ahora, en modo `duro`, termina FAILED sin result_path; en
`sombra` no cambia el estado pero deja `_estructura` y `_unbacked_claims`.

Solo se parchea lo que ya se parchea en los demás tests del worker (red,
credencial, contrato de dispatch, Telegram) y las dos costuras del detector: las
FUENTES de la whitelist (así no hay DB ni git de producción) y la conexión de
`axioma_config`. Corre con:
  PYTHONPATH=.:las_manos python -m pytest las_manos/_worker_estructura_test.py

ANTI_FABRICACION_REAL = True le dice al conftest raíz que NO sustituya el
detector por el neutro que usan los demás tests del worker.
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from motor_registry import worker
from motor_registry.catalog import MotorCatalog
from motor_registry.job_store import JobStore

ANTI_FABRICACION_REAL = True

_CFG = {
    "motors": {
        "kimi": {
            "enabled": True, "provider": "kimi", "transport": "http_openai_compat",
            "provider_id": "moonshot", "api_key_env": "KIMI_API_KEY",
            "api_url": "https://api.moonshot.ai/v1/chat/completions", "model": "kimi-k2.7-code",
            "max_context_tokens": 256000, "sandbox_only": True, "default_timeout_seconds": 600,
            "supports_reasoning": True, "reasoning_default_visibility": "audit_only", "max_tokens": 8000,
        },
    },
    "capabilities": {
        "code_patch": {
            "allowed_motors": ["kimi"], "allowed_callers": ["jacobs"], "risk_level": "low",
            "sandbox_only": True, "requires_human_gate": False, "max_execution_minutes": 10,
            "max_recursion_depth": 0, "output_schema": "code_patch.v1",
        },
    },
}

_FABRICADA = json.dumps({"diff": "", "files_modified": ["services/bodegas.py"], "description": "agrega bodegas"})
_DIFF_FALSO = json.dumps({"diff": "--- a/no/existe.py\n+++ b/no/existe.py\n@@ -1 +1 @@\n-x\n+y\n",
                          "files_modified": [], "description": "arregla"})
_REAL = json.dumps({"diff": "", "files_modified": ["jax/core/x.py"], "description": "ok"})


def _resp(content):
    payload = {"choices": [{"message": {"content": content}, "finish_reason": "stop"}],
               "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}}
    r = AsyncMock()
    r.json = lambda: payload
    r.raise_for_status = lambda: None
    return r


def _fuentes(esquema_cae=False):
    from anti_fabricacion.whitelist import EsquemaDB, EstadoGit, Fuentes

    async def huella():
        if esquema_cae:
            raise ConnectionError("db caida")
        return "H"

    async def esquema():
        if esquema_cae:
            raise ConnectionError("db caida")
        return EsquemaDB(columnas=(("jacobs_steps", "status"),))

    async def git():
        return EstadoGit(head="h", status_hash="s", entradas=(("b1", "jax/core/x.py"),))

    async def simbolos(_e):
        return frozenset()

    return Fuentes(huella=huella, esquema=esquema, git=git, simbolos=simbolos)


class WorkerEstructuraTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        import contrato_dispatch
        import anti_fabricacion.whitelist as wl_mod
        wl_mod._MEMO.clear()
        for p in (
            patch.object(contrato_dispatch, "_leer_contrato", AsyncMock(return_value=("max_tokens", 131072))),
            patch.object(worker, "resolve_credential", AsyncMock(return_value="sk-fake")),
            patch("motor_registry.usage_writer.record_motor_usage", AsyncMock()),
        ):
            p.start()
            self.addCleanup(p.stop)
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.audit_path = str(Path(self._tmp.name) / "audit.jsonl")
        env = patch.dict(os.environ, {"JAX_AUDIT_LOG_PATH": self.audit_path})
        env.start()
        self.addCleanup(env.stop)
        self.store = JobStore(str(Path(self._tmp.name) / "jobs.jsonl"))
        self.catalog = MotorCatalog(_CFG)
        self.telegram = AsyncMock(return_value={"ok": True, "message_id": 1, "error": None})

    async def _correr(self, content, *, modo, esquema_cae=False, pipeline_id="pl-1"):
        from anti_fabricacion.consecuencia import Modo
        job_id = self.store.create(caller="jacobs", capability="code_patch", motor="kimi", trace_id="t",
                                   prompt="arregla bodegas", recursion_depth=0, pipeline_id=pipeline_id)
        conexion = patch("anti_fabricacion.consecuencia._db_conn", AsyncMock(side_effect=ConnectionError("config ilegible")), create=True)
        leer = patch("anti_fabricacion.consecuencia.leer_modo", AsyncMock(return_value=Modo(modo))) if modo else conexion
        with leer, \
             patch("anti_fabricacion.whitelist.fuentes_por_defecto", lambda *a, **k: _fuentes(esquema_cae)), \
             patch("jacobs.reaper.send_telegram_alert", self.telegram), \
             patch("httpx.AsyncClient.post", AsyncMock(return_value=_resp(content))) as post:
            await worker.run(job_id=job_id, motor="kimi", capability="code_patch", prompt="arregla bodegas",
                             context={}, store=self.store,
                             catalog=self.catalog, kill_switch_path=str(Path(self._tmp.name) / "PAUSE"),
                             pipeline_id=pipeline_id)
        return job_id, post

    def _auditoria(self):
        p = Path(self.audit_path)
        return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []

    # (a) lo que hoy termina COMPLETED, en duro tiene que terminar FAILED
    async def test_duro_files_modified_inexistente_falla(self):
        job_id, _ = await self._correr(_FABRICADA, modo="duro")
        s = self.store._index[job_id]
        assert s["status"] == "failed", s
        assert s["error"].startswith("estructura sin respaldo: ") and "services/bodegas.py" in s["error"]
        assert not s.get("result_path"), "un job marcado en duro no deja result_path"
        assert s["_estructura"]["estado"] == "marcado" and s["_unbacked_claims"]

    async def test_duro_diff_a_sobre_archivo_inexistente_falla(self):
        job_id, _ = await self._correr(_DIFF_FALSO, modo="duro")
        s = self.store._index[job_id]
        assert s["status"] == "failed" and "no/existe.py" in s["error"], s

    async def test_duro_con_salida_real_completa_y_expone_el_veredicto(self):
        job_id, _ = await self._correr(_REAL, modo="duro")
        s = self.store._index[job_id]
        assert s["status"] == "completed" and s["result_path"], s
        assert self.store.get(job_id).estructura_veredicto == "limpio"
        assert s["_estructura"]["estado"] == "limpio" and s["_unbacked_claims"] == []

    # (b) base caida en duro: 0 llamadas de transporte
    async def test_duro_con_la_base_caida_no_llama_al_motor(self):
        job_id, post = await self._correr(_REAL, modo="duro", esquema_cae=True)
        assert post.call_count == 0, "con el detector caido el motor NO se llama"
        s = self.store._index[job_id]
        assert s["status"] == "failed" and "detector_no_disponible" in s["error"], s
        assert s["_estructura"]["estado"] == "no_corrio"

    # (c) config ilegible -> duro
    async def test_config_ilegible_es_duro(self):
        job_id, _ = await self._correr(_FABRICADA, modo=None)
        assert self.store._index[job_id]["status"] == "failed"

    # (d) sombra: el estado no cambia, pero queda el rastro
    async def test_sombra_completa_y_guarda_estructura(self):
        job_id, _ = await self._correr(_FABRICADA, modo="sombra")
        s = self.store._index[job_id]
        assert s["status"] == "completed" and s["result_path"], s
        assert s["_estructura"]["estado"] == "marcado" and s["_unbacked_claims"]
        assert self.store.get(job_id).estructura_veredicto == "marcado"

    async def test_sombra_con_el_detector_caido_sigue_adelante_y_lo_dice(self):
        job_id, post = await self._correr(_REAL, modo="sombra", esquema_cae=True)
        s = self.store._index[job_id]
        assert post.call_count == 1 and s["status"] == "completed", s
        assert s["_estructura"]["estado"] == "no_corrio"

    # evento de auditoria y aviso, en ambos modos
    async def test_marcado_deja_evento_en_audit_y_avisa_en_ambos_modos(self):
        for modo in ("duro", "sombra"):
            self.telegram.reset_mock()
            if Path(self.audit_path).exists():
                Path(self.audit_path).write_text("")
            job_id, _ = await self._correr(_FABRICADA, modo=modo)
            eventos = [e for e in self._auditoria() if e.get("event") == "ESTRUCTURA_SIN_RESPALDO"]
            assert len(eventos) == 1, (modo, eventos)
            e = eventos[0]
            assert (e["job_id"], e["motor"], e["capability"], e["pipeline_id"]) == (job_id, "kimi", "code_patch", "pl-1")
            assert e["endpoint"] and e["modo"] == modo
            assert self.telegram.await_count == 1, (modo, self.telegram.await_args_list)
            assert "services/bodegas.py" in self.telegram.await_args.args[0]

    async def test_limpio_no_avisa_ni_audita(self):
        await self._correr(_REAL, modo="duro")
        assert self.telegram.await_count == 0
        assert not [e for e in self._auditoria() if e.get("event") == "ESTRUCTURA_SIN_RESPALDO"]


if __name__ == "__main__":
    unittest.main()
