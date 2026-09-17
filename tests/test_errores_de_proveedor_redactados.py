"""E-16 (2026-09-16): el cuerpo de un error de proveedor se REDACTA (con la
credencial conocida) ANTES de recortarlo, en todos los caminos.

jax/core/redaccion.py:161 lo declara: recortar_redactado es "la única forma de
recortar un texto de error de proveedor en jax". Los caminos de Gemini lo
cumplían; estos no: executor (openai-compat y Ollama), REPL (DeepSeek, OpenAI,
Ollama) y el cerebro Ada del planner, que además lo mandaba a logger.error.
Recortar primero parte un secreto en el carácter 200 y el pedazo ya no lo
reconoce ninguna regla. run_task escribía el error de la tarea a disco sin
redactar. Todas las keys son FALSAS.
"""
from __future__ import annotations

import ast
import os
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx

os.environ["JAX_DB_NAME"] = "jax_memory_test"

RAIZ = Path(__file__).resolve().parents[1]
# Sin forma reconocible: SOLO se tapa si se pasa como secreto conocido.
SECRETO = "zq7Wn2-sin-forma-reconocible-88Hk"
CUERPO_CON_SECRETO = "p" * 192 + SECRETO + " fin"
# Con forma reconocible (Authorization): el token cruza el corte de 200.
CUERPO_CON_BEARER = "p" * 170 + "Authorization: Bearer tok-FAKE-e16-0123456789 fin"


def _cuerpos_recortados(fuente: str) -> list[int]:
    """Líneas con `X.text[:N]` o `body[:N]` (body = await resp.aread())."""
    lineas = set()
    for funcion in ast.walk(ast.parse(fuente)):
        if not isinstance(funcion, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        leidos = {
            t.id for n in ast.walk(funcion)
            if isinstance(n, ast.Assign) and isinstance(n.value, ast.Await) and isinstance(n.value.value, ast.Call)
            and isinstance(n.value.value.func, ast.Attribute) and n.value.value.func.attr == "aread"
            for t in n.targets if isinstance(t, ast.Name)
        }
        for n in ast.walk(funcion):
            if isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Slice):
                v = n.value
                if (isinstance(v, ast.Attribute) and v.attr == "text") or (isinstance(v, ast.Name) and v.id in leidos):
                    lineas.add(n.lineno)
    return sorted(lineas)


def _archivos_de_servicio():
    for arbol in ("jacobs", "jax", "las_manos"):
        for dirpath, dirnames, filenames in os.walk(RAIZ / arbol):
            dirnames[:] = [d for d in dirnames if d not in {".venv", "__pycache__", "tests"}]
            for nombre in filenames:
                if nombre.endswith(".py") and not nombre.startswith("test_") and not nombre.endswith("_test.py"):
                    yield Path(dirpath) / nombre


def test_ningun_cuerpo_de_proveedor_se_recorta_sin_redactar():
    hallazgos = {}
    for p in _archivos_de_servicio():
        lineas = _cuerpos_recortados(p.read_text(encoding="utf-8", errors="replace"))
        if lineas:
            hallazgos[str(p.relative_to(RAIZ))] = lineas
    assert hallazgos == {}, "usar recortar_redactado(texto, N, [credencial]) (jax/core/redaccion.py)"


def test_el_detector_ve_resp_text_recortado():
    assert _cuerpos_recortados("def f(resp):\n    return resp.text[:200]\n") == [2]


def test_el_detector_ve_un_cuerpo_leido_con_aread_y_recortado():
    fuente = "async def f(resp):\n    body = await resp.aread()\n    raise E(body[:200])\n"
    assert _cuerpos_recortados(fuente) == [3]


def test_el_detector_acepta_recortar_redactado():
    assert _cuerpos_recortados("def f(resp, k):\n    return recortar_redactado(resp.text, 200, [k])\n") == []


def _responder(status: int, texto: str):
    async def handle(transport_self, request):
        return httpx.Response(status, text=texto)
    return patch("httpx.AsyncHTTPTransport.handle_async_request", handle)


def _faceta(key="jekyll"):
    from facet_resolver import ResolvedFacet
    return ResolvedFacet(key=key, provider_id="deepseek", base_url="https://api.proveedor.example/v1", model="m",
                         credential=SECRETO, transport="http_openai_compat", persona=None, params=None)


class JacobsTest(unittest.IsolatedAsyncioTestCase):
    async def test_openai_compat_redacta_la_credencial_antes_de_recortar(self):
        from jacobs import executor
        with _responder(400, CUERPO_CON_SECRETO), patch.object(executor, "limite_de_salida", AsyncMock(return_value={})):
            with self.assertRaises(RuntimeError) as ctx:
                await executor._invoke_http_openai_compat(_faceta(), "hola", 10)
        mensaje = str(ctx.exception)
        self.assertIn("HTTP 400", mensaje)
        self.assertNotIn(SECRETO[:8], mensaje)
        self.assertIn("p" * 192 + "***", mensaje)

    async def test_ollama_redacta_antes_de_recortar(self):
        from jacobs import executor
        faceta = _faceta(key="jax_local")
        with _responder(500, CUERPO_CON_BEARER), patch.object(executor, "limite_de_salida", AsyncMock(return_value={})):
            with self.assertRaises(RuntimeError) as ctx:
                await executor._invoke_ollama(faceta, "hola", 10)
        self.assertNotIn("tok-FAKE", str(ctx.exception))

    async def test_ada_redacta_el_cuerpo_en_el_motivo_y_en_el_log(self):
        from jacobs import plan
        with _responder(400, CUERPO_CON_SECRETO), \
             patch.object(plan, "resolve_facet", AsyncMock(return_value=_faceta(key="ada"))), \
             patch.object(plan, "limite_de_salida", AsyncMock(return_value={"max_tokens": 10})), \
             self.assertLogs("jacobs.plan", level="ERROR") as logs:
            with self.assertRaises(plan.CerebroNoDisponible) as ctx:
                await plan.PlanBuilder()._ada_plan("objetivo", 3)
        self.assertIn("Ada HTTP 400", str(ctx.exception))
        self.assertNotIn(SECRETO[:8], str(ctx.exception))
        self.assertNotIn(SECRETO[:8], "\n".join(logs.output))


class ReplTest(unittest.IsolatedAsyncioTestCase):
    def _musculo(self, proveedor):
        from jax.muscles import base
        return base.HttpMuscle(name="jekyll", provider=proveedor, model_default="d", models_allowed=["d"],
                               system_prompt="s", timeout=10, api_url="https://api.proveedor.example/v1/chat/completions")

    async def _error(self, proveedor, metodo):
        from jax.muscles import base
        with _responder(400, CUERPO_CON_SECRETO), \
             patch.object(base.HttpMuscle, "_resolve_api_key", AsyncMock(return_value=SECRETO)), \
             patch.object(base.HttpMuscle, "_limite_de_salida", AsyncMock(return_value={})):
            with self.assertRaises(base.MuscleInvocationError) as ctx:
                await getattr(self._musculo(proveedor), metodo)("hola", "d")
        return str(ctx.exception)

    async def test_deepseek_redacta_la_credencial_conocida(self):
        mensaje = await self._error("deepseek", "_call_deepseek")
        self.assertIn("DeepSeek HTTP 400", mensaje)
        self.assertNotIn(SECRETO[:8], mensaje)

    async def test_openai_en_streaming_redacta_la_credencial_conocida(self):
        mensaje = await self._error("openai", "_call_openai")
        self.assertIn("OpenAI HTTP 400", mensaje)
        self.assertNotIn(SECRETO[:8], mensaje)

    async def test_ollama_local_redacta_antes_de_recortar(self):
        from jax.muscles.base import MuscleInvocationError
        from jax.muscles.ollama_muscle import OllamaMuscle
        musculo = OllamaMuscle("jax_local", "q", ["q"], "s", 10, api_url="http://ollama.example/api/chat")
        with _responder(500, CUERPO_CON_BEARER), \
             patch("jax.muscles.ollama_muscle.limite_de_salida", AsyncMock(return_value={"options": {"num_predict": 5}})):
            with self.assertRaises(MuscleInvocationError) as ctx:
                await musculo._call("hola", "q")
        self.assertNotIn("tok-FAKE", str(ctx.exception))


def test_el_error_de_una_tarea_se_escribe_redactado():
    from jax.core import main
    texto = main._texto_de_error_de_tarea(RuntimeError("fallo con api_key=sk-FAKE-tarea-0123456789"))
    assert "sk-FAKE-tarea" not in texto
    assert "api_key=***" in texto


def test_run_task_escribe_el_error_con_el_texto_redactado():
    arbol = ast.parse((RAIZ / "jax" / "core" / "main.py").read_text(encoding="utf-8"))
    run_task = next(n for n in ast.walk(arbol) if isinstance(n, ast.AsyncFunctionDef) and n.name == "run_task")
    asignaciones = [n for n in ast.walk(run_task) if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "error_msg" for t in n.targets)]
    assert asignaciones, "run_task ya no arma error_msg"
    for a in asignaciones:
        assert isinstance(a.value, ast.Call) and getattr(a.value.func, "id", None) == "_texto_de_error_de_tarea"
