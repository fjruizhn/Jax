"""E-12 / E-22 / E-18 (2026-09-16): la copia .md de cada step.

- E-22: el directorio sale de JAX_REPO_BASE (/etc/jax/.env), la MISMA
  variable con la que jax-platform lista y sirve esos documentos
  (api/admin/repository.py, REPO_BASE). Antes: ~/jax/repo/documents fijo en
  los dos repos.
- E-12: la escritura corre en asyncio.to_thread (mkdir incluido), sin
  aiofiles, una dependencia usada en un solo lugar.
- E-18: la marca fail-soft decía "nadie lee ese .md". Es falso: el admin de
  jax-platform lo lista.
"""
from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

os.environ["JAX_DB_NAME"] = "jax_memory_test"

RAIZ = Path(__file__).resolve().parents[1]


def test_el_documento_se_escribe_fuera_del_loop_en_el_directorio_configurado(tmp_path):
    from jacobs import executor
    hilos = []
    real = asyncio.to_thread

    async def espia(fn, *args, **kwargs):
        hilos.append(fn)
        return await real(fn, *args, **kwargs)

    async def correr():
        with patch.object(executor, "REPO_DOCUMENTS_DIR", tmp_path / "documents"), \
             patch.object(executor.asyncio, "to_thread", espia):
            await executor._persist_step_to_repo(
                pipeline_id="12345678-x", pipeline_name="t", step_index=1, facet="jekyll",
                capability="analysis", raw_output={"success": True, "result": "cuerpo", "model": "m"})

    asyncio.run(correr())
    assert hilos, "la escritura corrió dentro del event loop"
    assert (tmp_path / "documents" / "12345678_01_jekyll.md").read_text(encoding="utf-8").endswith("cuerpo")


def test_la_marca_fail_soft_de_la_copia_dice_la_verdad():
    fuente = (RAIZ / "jacobs" / "executor.py").read_text(encoding="utf-8")
    [linea] = [l for l in fuente.splitlines() if "except Exception as _persist_err" in l]
    assert "# fail-soft:" in linea
    assert "nadie lee" not in linea
    assert "/api/admin/repo" in linea and "output_ref" in linea


def test_los_tests_no_escriben_en_el_repo_de_produccion():
    base = Path(os.environ["JAX_REPO_BASE"])
    assert base.is_absolute() and str(base).startswith(tempfile.gettempdir())
    from jacobs import executor
    assert executor.REPO_DOCUMENTS_DIR == base / "documents"
