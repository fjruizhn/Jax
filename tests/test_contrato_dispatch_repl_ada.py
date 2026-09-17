"""
PR-K (2026-09-14): el REPL y el planificador de Ada toman el NOMBRE y el TOPE
del límite de salida de la fila de `model` que despachan -- no el
`"max_tokens": 131072` fijo que tumbó a thot en la Mesa web (2026-08-24).

Puros: se parchea la lectura de la fila (`_leer_contrato`) y el HTTP. La
consulta real a la DB está en tests/test_contrato_dispatch_db.py.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import os
import unittest
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

_existing_db_name = os.environ.get("JAX_DB_NAME")
if _existing_db_name and _existing_db_name != "jax_memory_test":
    raise RuntimeError(
        f"JAX_DB_NAME={_existing_db_name!r} ya está seteado -- unset antes de correr este test."
    )
os.environ.setdefault("JAX_DB_NAME", "jax_memory_test")

import httpx  # noqa: E402

from jax.core import contrato_dispatch as cd  # noqa: E402
from jax.muscles import base  # noqa: E402


class _Resp:
    status_code = 200

    def __init__(self, payload):
        self._payload = payload
        self.text = ""

    def json(self):
        return self._payload

    def raise_for_status(self):
        return None


class _StreamResp:
    status_code = 200

    async def aiter_lines(self):
        yield 'data: {"model": "m", "choices": [{"delta": {"content": "hola"}}]}'
        yield "data: [DONE]"

    async def aread(self):
        return b""


def _muscle(provider, model, name="faceta"):
    return base.HttpMuscle(
        name=name, provider=provider, model_default=model,
        models_allowed=[model, "pesado"], system_prompt="s", timeout=10,
        api_url="https://api.proveedor.example/v1/chat/completions",
    )


class _Captura:
    """Parchea el HTTP (post y stream) y guarda el body enviado."""

    def __init__(self):
        self.bodies = []
        self.urls = []
        self.headers = []

    def patches(self):
        cap = self

        async def fake_post(client_self, url, json=None, **kw):
            cap.bodies.append(json)
            cap.urls.append(url)
            cap.headers.append(kw.get("headers"))
            if "generativelanguage" in url or ":generateContent" in url:
                return _Resp({"candidates": [{"content": {"parts": [{"text": "hola"}]}}]})
            if url.endswith("/api/chat"):
                return _Resp({"model": "m", "message": {"content": "hola"}})
            return _Resp({"model": "m", "choices": [{"message": {"content": "hola"}}]})

        @asynccontextmanager
        async def fake_stream(client_self, method, url, json=None, **kw):
            cap.bodies.append(json)
            cap.urls.append(url)
            cap.headers.append(kw.get("headers"))
            yield _StreamResp()

        return [
            patch.object(httpx.AsyncClient, "post", fake_post),
            patch.object(httpx.AsyncClient, "stream", fake_stream),
            patch.object(base, "resolve_credential_instrumented", AsyncMock(return_value="k")),
            patch.object(base, "record_resolved_version_safe", AsyncMock()),
        ]


class _Base(unittest.IsolatedAsyncioTestCase):
    def arrancar(self, fila):
        self.cap = _Captura()
        self.leer = AsyncMock(return_value=fila)
        for p in self.cap.patches() + [patch.object(cd, "_leer_contrato", self.leer)]:
            p.start()
            self.addCleanup(p.stop)


class ReplPayloadDelCatalogoTest(_Base):
    async def test_deepseek_manda_el_nombre_y_el_tope_de_su_fila(self):
        self.arrancar(("max_tokens", 393216))
        await _muscle("deepseek", "deepseek-flash").invoke("hola")
        body = self.cap.bodies[0]
        self.assertEqual(body["max_tokens"], 393216)
        self.assertNotIn("max_completion_tokens", body)
        self.leer.assert_awaited_once_with("deepseek", "deepseek-flash")

    async def test_openai_stream_manda_max_completion_tokens_con_el_tope_de_su_fila(self):
        # El caso real del incidente: gpt-5.6-terra exige el nombre nuevo y
        # acepta como mucho 128000.
        self.arrancar(("max_completion_tokens", 128000))
        await _muscle("openai", "gpt-5.6-terra").invoke("hola")
        body = self.cap.bodies[0]
        self.assertEqual(body["max_completion_tokens"], 128000)
        self.assertNotIn("max_tokens", body)
        self.leer.assert_awaited_once_with("openai", "gpt-5.6-terra")

    async def test_el_alias_kimi_lee_la_fila_de_moonshot(self):
        self.arrancar(("max_tokens", 131072))
        await _muscle("kimi", "kimi-k3").invoke("hola")
        self.leer.assert_awaited_once_with("moonshot", "kimi-k3")
        self.assertEqual(self.cap.bodies[0]["max_tokens"], 131072)

    async def test_lee_la_fila_del_modelo_que_despacha_no_la_del_asignado(self):
        # Modo pesado del REPL: invoke(model=MODELO_PESADO) despacha OTRO modelo.
        self.arrancar(("max_tokens", 1000))
        await _muscle("deepseek", "deepseek-flash").invoke("hola", model="pesado")
        self.leer.assert_awaited_once_with("deepseek", "pesado")


class ReplFailClosedTest(_Base):
    async def _falla(self, fila, *esperados):
        self.arrancar(fila)
        with self.assertRaises(base.MuscleInvocationError) as ctx:
            await _muscle("deepseek", "deepseek-flash").invoke("hola")
        self.assertEqual(self.cap.bodies, [], "no puede salir ningún request sin contrato")
        self.assertIsInstance(ctx.exception, cd.ModelDispatchConfigError)
        for texto in esperados:
            self.assertIn(texto, str(ctx.exception))
        return ctx.exception

    async def test_param_null_falla_con_el_update(self):
        await self._falla((None, 393216), "UPDATE model SET max_tokens_param")

    async def test_tope_null_falla_con_el_update(self):
        await self._falla(("max_tokens", None), "UPDATE model SET max_output_tokens")

    async def test_los_dos_null_juntan_los_dos_updates(self):
        await self._falla((None, None), "UPDATE model SET max_tokens_param",
                          "UPDATE model SET max_output_tokens")

    async def test_param_invalido_falla(self):
        await self._falla(("max_tokenz", 1000), "max_tokenz")

    async def test_tope_cero_falla(self):
        await self._falla(("max_tokens", 0), "entero positivo")

    async def test_tope_bool_falla(self):
        await self._falla(("max_tokens", True), "entero positivo")

    async def test_tope_texto_falla(self):
        await self._falla(("max_tokens", "8000"), "entero positivo")

    async def test_fila_inexistente_falla(self):
        await self._falla(None, "no está en el catálogo")

    async def test_el_repl_muestra_el_update_entero(self):
        # humanizar_error recortaba a 160 caracteres: el UPDATE no se veía.
        from jax.core.main import humanizar_error
        err = await self._falla((None, 393216), "UPDATE model SET max_tokens_param")
        visible = humanizar_error("Jekyll", err)
        self.assertIn("UPDATE model SET max_tokens_param='max_tokens'", visible)
        self.assertIn("WHERE model_id='deepseek-flash'", visible)


class NoOpenAICompatTest(_Base):
    async def test_gemini_no_lleva_max_tokens_ni_lee_el_contrato(self):
        self.arrancar(("max_tokens", 1))
        await _muscle("gemini", "gemini-x").invoke("hola")
        body = self.cap.bodies[0]
        self.assertNotIn("max_tokens", body)
        self.assertNotIn("max_completion_tokens", body)
        self.leer.assert_not_awaited()

    async def test_el_helper_rechaza_un_transporte_sin_limite_con_su_parametro(self):
        # Ronda 2 (I3): se decide por TRANSPORTE, como la Mesa web.
        self.arrancar(("max_tokens", 1))
        for transporte, propio in (("http_gemini", "generationConfig.maxOutputTokens"),
                                   ("subprocess", "max_tokens (obligatorio")):
            with self.assertRaises(cd.ModelDispatchConfigError) as ctx:
                await cd.limite_de_salida(transporte, "p", "x")
            self.assertIn(propio, str(ctx.exception), transporte)
        self.leer.assert_not_awaited()

    async def test_ollama_nativo_manda_num_predict_y_v1_max_tokens_solo_con_el_tope(self):
        # Ollama no usa max_tokens_param (NULL): el nombre lo fija el endpoint.
        self.arrancar((None, 4096))
        self.assertEqual(await cd.limite_de_salida("ollama", "ollama", "qwen"),
                         {"options": {"num_predict": 4096}})
        self.assertEqual(await cd.limite_de_salida("ollama", "ollama", "qwen", ollama_api=cd.OLLAMA_API_V1),
                         {"max_tokens": 4096})

    async def test_ollama_sin_tope_falla_con_el_update(self):
        self.arrancar(("max_tokens", None))
        with self.assertRaises(cd.ModelDispatchConfigError) as ctx:
            await cd.limite_de_salida("ollama", "ollama", "qwen")
        self.assertIn("UPDATE model SET max_output_tokens", str(ctx.exception))

    async def test_fila_ausente_no_sugiere_agregarla_bajo_otro_proveedor(self):
        # Ronda 2 (I1): "agregalo al catálogo" a secas habilitaba un dispatch cruzado.
        self.arrancar(None)
        with self.assertRaises(cd.ModelDispatchConfigError) as ctx:
            await cd.limite_de_salida("http_openai_compat", "deepseek", "gpt-x")
        texto = str(ctx.exception)
        self.assertIn("NO lo agregues bajo 'deepseek'", texto)
        self.assertNotIn("Agregalo al catálogo", texto)


def _facet_ada(**kw):
    from facet_resolver import ResolvedFacet
    datos = dict(key="ada", provider_id="zhipu", base_url="https://z.example/api/paas/v4",
                 model="glm-5.3", credential="cred-del-resolver", transport="http_openai_compat",
                 persona=None, params=None)
    datos.update(kw)
    return ResolvedFacet(**datos)


class AdaPlanDelBindingTest(unittest.IsolatedAsyncioTestCase):
    """PR-K ronda 1: Ada planifica con SU binding (resolve_facet("ada")) y el
    contrato de la fila de ese modelo. jacobs/plan.py importa
    `contrato_dispatch` y `facet_resolver` pelados (las_manos/): se parchean
    ESOS módulos, los de producción."""

    def arrancar(self, fila, facet=None, facet_error=None):
        import contrato_dispatch as cd_jacobs
        from jacobs import plan
        self.plan = plan
        self.cap = _Captura()
        self.leer = AsyncMock(return_value=fila)
        self.resolver = AsyncMock(return_value=facet or _facet_ada(), side_effect=facet_error)
        for p in [
            patch.object(httpx.AsyncClient, "stream", self.cap.patches()[1].new),
            patch.object(plan, "resolve_facet", self.resolver),
            patch.object(cd_jacobs, "_leer_contrato", self.leer),
        ]:
            p.start()
            self.addCleanup(p.stop)

    async def test_ada_manda_modelo_url_y_credencial_del_binding_y_el_tope_de_su_fila(self):
        self.arrancar(("max_tokens", 131072))
        await self.plan.PlanBuilder()._ada_plan("objetivo", 3, facetas_activas=frozenset({"ada", "thot"}))
        body = self.cap.bodies[0]
        self.assertEqual(body["model"], "glm-5.3")
        self.assertEqual(body["max_tokens"], 131072)
        self.assertEqual(self.cap.urls[0], "https://z.example/api/paas/v4/chat/completions")
        self.assertEqual(self.cap.headers[0]["Authorization"], "Bearer cred-del-resolver")
        self.resolver.assert_awaited_once_with("ada")
        self.leer.assert_awaited_once_with("zhipu", "glm-5.3")

    async def test_ada_con_otro_nombre_de_parametro_lo_respeta(self):
        self.arrancar(("max_completion_tokens", 5000))
        await self.plan.PlanBuilder()._ada_plan("objetivo", 3, facetas_activas=frozenset({"ada", "thot"}))
        body = self.cap.bodies[0]
        self.assertEqual(body["max_completion_tokens"], 5000)
        self.assertNotIn("max_tokens", body)

    async def test_binding_sin_contrato_no_despacha_y_da_error_con_el_update(self):
        self.arrancar((None, None))
        with self.assertLogs("jacobs.plan", level="ERROR") as logs, \
                self.assertRaises(self.plan.CerebroNoDisponible):
            await self.plan.PlanBuilder()._ada_plan("objetivo", 3, facetas_activas=frozenset({"ada", "thot"}))
        self.assertEqual(self.cap.bodies, [], "Ada no puede despachar sin contrato")
        salida = "\n".join(logs.output)
        self.assertIn("UPDATE model SET max_tokens_param", salida)
        self.assertIn("WHERE model_id='glm-5.3'", salida)

    async def test_faceta_no_resoluble_no_despacha_y_da_error(self):
        from facet_resolver import FacetUnavailableError
        self.arrancar(("max_tokens", 1), facet_error=FacetUnavailableError("ada"))
        with self.assertLogs("jacobs.plan", level="ERROR") as logs, \
                self.assertRaises(self.plan.CerebroNoDisponible):
            await self.plan.PlanBuilder()._ada_plan("objetivo", 3, facetas_activas=frozenset({"ada", "thot"}))
        self.assertEqual(self.cap.bodies, [])
        self.assertIn("faceta no resoluble", "\n".join(logs.output))
        self.leer.assert_not_awaited()

    async def test_binding_con_otro_transporte_no_despacha_y_da_error(self):
        self.arrancar(("max_tokens", 1), facet=_facet_ada(transport="http_gemini"))
        with self.assertLogs("jacobs.plan", level="ERROR") as logs, \
                self.assertRaises(self.plan.CerebroNoDisponible):
            await self.plan.PlanBuilder()._ada_plan("objetivo", 3, facetas_activas=frozenset({"ada", "thot"}))
        self.assertEqual(self.cap.bodies, [])
        self.assertIn("http_gemini", "\n".join(logs.output))

    async def test_el_fallback_a_qwen_sigue_pero_el_contrato_roto_queda_en_error(self):
        """El fallback que ya existía (Ada falla -> qwen) no puede enmascarar un
        contrato roto: el plan sale de qwen Y hay un ERROR con el UPDATE."""
        self.arrancar((None, None))
        b = self.plan.PlanBuilder()
        qwen = AsyncMock(return_value=[{"facet": "jekyll", "capability": "analysis", "prompt": "x"}])
        b._llm_plan = qwen
        b._from_spec = lambda pipeline_id, specs, caps: specs
        from jacobs import store
        evento = AsyncMock()
        with patch.object(self.plan, "_build_capability_hint", lambda g: ""), \
                patch.object(store, "event_append", evento), \
                self.assertLogs("jacobs.plan", level="INFO") as logs:
            specs = await b._from_objective("p", "x" * 250, 3, {"capabilities": {}, "facets": frozenset({"ada", "thot", "jekyll"})})
        self.assertEqual(specs[0]["facet"], "jekyll")
        qwen.assert_awaited_once()
        errores = [l for l in logs.output if l.startswith("ERROR")]
        self.assertTrue(any("UPDATE model SET max_tokens_param" in l for l in errores), logs.output)
        # Ronda 2 (M2): el pipeline registra que el plan salió de qwen, y por qué.
        evento.assert_awaited_once()
        pipeline_id, tipo, payload = evento.await_args.args
        self.assertEqual((pipeline_id, tipo), ("p", "PLAN_CEREBRO_FALLBACK"))
        self.assertEqual((payload["de"], payload["a"]), ("ada", "qwen"))
        self.assertIn("UPDATE model SET max_tokens_param", payload["motivo"])

    async def test_ada_http_no_200_es_error_y_queda_como_motivo_del_evento(self):
        """Ronda 2 (M2): un 400 del proveedor (p. ej. "max_tokens is too
        large") era un WARNING; ahora ERROR, y el motivo llega al evento."""
        self.arrancar(("max_tokens", 131072))

        class _Resp400:
            status_code = 400

            async def aread(self):
                return b"max_tokens is too large"

        @asynccontextmanager
        async def stream_400(client_self, method, url, json=None, **kw):
            yield _Resp400()

        b = self.plan.PlanBuilder()
        b._llm_plan = AsyncMock(return_value=[{"facet": "jekyll", "capability": "analysis", "prompt": "x"}])
        b._from_spec = lambda pipeline_id, specs, caps: specs
        from jacobs import store
        evento = AsyncMock()
        with patch.object(httpx.AsyncClient, "stream", stream_400), \
                patch.object(self.plan, "_build_capability_hint", lambda g: ""), \
                patch.object(store, "event_append", evento), \
                self.assertLogs("jacobs.plan", level="ERROR") as logs:
            await b._from_objective("p", "x" * 250, 3, {"capabilities": {}, "facets": frozenset({"ada", "thot", "jekyll"})})
        self.assertTrue(any(l.startswith("ERROR") and "Ada HTTP 400" in l for l in logs.output), logs.output)
        self.assertIn("Ada HTTP 400", evento.await_args.args[2]["motivo"])

    async def test_qwen_tambien_falla_cae_al_plan_fijo_y_lo_registra(self):
        self.arrancar(("max_tokens", 1))
        b = self.plan.PlanBuilder()
        b._llm_plan = AsyncMock(side_effect=self.plan.CerebroNoDisponible("qwen: sin contrato"))
        b._fallback_plan = lambda objective: [{"facet": "jekyll", "capability": "analysis", "prompt": "fijo"}]
        b._from_spec = lambda pipeline_id, specs, caps: specs
        from jacobs import store
        evento = AsyncMock()
        with patch.object(self.plan, "_build_capability_hint", lambda g: ""), \
                patch.object(store, "event_append", evento):
            specs = await b._from_objective("p", "corto", 3, {"capabilities": {}, "facets": frozenset({"ada", "thot", "jekyll"})})
        self.assertEqual(specs[0]["prompt"], "fijo")
        _, tipo, payload = evento.await_args.args
        self.assertEqual((tipo, payload["de"], payload["a"]), ("PLAN_CEREBRO_FALLBACK", "qwen", "fallback_plan"))
        self.assertIn("sin contrato", payload["motivo"])


class ExecutorOpenAICompatTest(unittest.IsolatedAsyncioTestCase):
    """jacobs/executor.py::_invoke_http_openai_compat (jekyll/thot/ada en
    pipelines) no mandaba límite de salida: ahora el de la fila del modelo."""

    def arrancar(self, fila):
        import contrato_dispatch as cd_jacobs
        from jacobs import executor
        self.executor = executor
        self.cap = _Captura()
        self.leer = AsyncMock(return_value=fila)
        for p in [
            patch.object(httpx.AsyncClient, "post", self.cap.patches()[0].new),
            patch.object(executor, "record_resolved_version_safe", AsyncMock()),
            patch.object(cd_jacobs, "_leer_contrato", self.leer),
        ]:
            p.start()
            self.addCleanup(p.stop)

    async def test_manda_nombre_y_tope_de_la_fila_del_modelo(self):
        self.arrancar(("max_completion_tokens", 128000))
        f = _facet_ada(key="thot", provider_id="openai", model="gpt-5.6-terra",
                       base_url="https://api.openai.com/v1")
        await self.executor._invoke_http_openai_compat(f, "hola", 10)
        body = self.cap.bodies[0]
        self.assertEqual(body["max_completion_tokens"], 128000)
        self.assertNotIn("max_tokens", body)
        self.leer.assert_awaited_once_with("openai", "gpt-5.6-terra")

    async def test_sin_contrato_no_despacha_y_sube_el_error_con_el_update(self):
        import contrato_dispatch as cd_jacobs
        self.arrancar(("max_tokens", None))
        with self.assertRaises(cd_jacobs.ModelDispatchConfigError) as ctx:
            await self.executor._invoke_http_openai_compat(_facet_ada(), "hola", 10)
        self.assertEqual(self.cap.bodies, [])
        self.assertIn("UPDATE model SET max_output_tokens", str(ctx.exception))


class PlanSinModeloLiteralTest(unittest.TestCase):
    """Tripwire: jacobs/plan.py no nombra ningún modelo en un string (el
    modelo sale del binding). AST: los comentarios no cuentan."""

    import re as _re
    PATRON = _re.compile(r"\b(glm|gpt|deepseek|kimi|gemini|claude|qwen|llama|sonnet|opus|haiku)[-:.]?[a-z]*[-:.]?\d")

    def _literales(self, fuente: str) -> list[str]:
        import ast
        return [n.value for n in ast.walk(ast.parse(fuente))
                if isinstance(n, ast.Constant) and isinstance(n.value, str)
                and self.PATRON.search(n.value.lower())]

    def test_plan_py_y_main_py_no_tienen_nombres_de_modelo_literales(self):
        # Ronda 2 (M4): también jax/core/main.py (el modelo pesado era literal).
        from pathlib import Path
        raiz = Path(__file__).resolve().parents[1]
        for rel in ("jacobs/plan.py", "jax/core/main.py"):
            fuente = (raiz / rel).read_text(encoding="utf-8")
            self.assertEqual(self._literales(fuente), [], rel)

    def test_el_detector_ve_un_modelo_literal(self):
        self.assertEqual(len(self._literales('A = "glm-5.2"\nB = "gpt-5.6-terra"\nC = "qwen3:14b"\n')), 3)
        self.assertEqual(self._literales('A = "kimi (coding)"\n# "glm-5.2" en comentario\n'), [])


# ---------------------------------------------------------------------------
# Ronda 2
# ---------------------------------------------------------------------------

def _cfg_repl(personalidades):
    return {"jax": {"timeout_seconds": 10}, "personalities": personalidades}


class ReplProveedorDelModeloTest(_Base):
    """I1: el REPL despacha al proveedor y la URL del MODELO del binding, nunca
    a la URL del proveedor que dice config.toml."""

    def _muscle(self, registro, personalidad):
        from jax.core.main import build_muscles
        from jax.core.registro_facetas import aplicar_registro
        cfg = _cfg_repl({"jekyll": {"system_prompt": "s", **personalidad}})
        aplicar_registro(cfg, registro)
        return build_muscles(cfg)["jekyll"]

    async def test_binding_de_otro_proveedor_despacha_a_la_url_del_modelo(self):
        self.arrancar(("max_completion_tokens", 128000))
        m = self._muscle(
            {"jekyll": {"model": "gpt-x", "models_allowed": ["gpt-x"], "transport": "http_openai_compat",
                        "provider_modelo": "openai", "base_url_modelo": "https://api.openai.example/v1"}},
            {"type": "http", "provider": "deepseek", "model_default": "deepseek-flash",
             "models_allowed": ["deepseek-flash"]},
        )
        await m.invoke("hola")
        self.assertEqual(self.cap.urls, ["https://api.openai.example/v1/chat/completions"])
        self.leer.assert_awaited_once_with("openai", "gpt-x")
        self.assertEqual(self.cap.bodies[0]["max_completion_tokens"], 128000)

    async def test_proveedor_no_configurado_da_error_visible_y_no_despacha(self):
        self.arrancar(("max_tokens", 1))
        m = self._muscle(
            {"jekyll": {"model": "claude-x", "models_allowed": ["claude-x"], "transport": "subprocess",
                        "provider_modelo": "anthropic", "base_url_modelo": None}},
            {"type": "http", "provider": "deepseek", "model_default": "deepseek-flash",
             "models_allowed": ["deepseek-flash"]},
        )
        with self.assertRaises(base.DispatchConfigMuscleError) as ctx:
            await m.invoke("hola")
        self.assertIn("subprocess", str(ctx.exception))
        self.assertEqual(self.cap.bodies, [], "nunca a la URL del TOML")
        self.leer.assert_not_awaited()


class OllamaNumPredictTest(_Base):
    """M3: los caminos Ollama nativos mandan options.num_predict del contrato."""

    async def test_ollama_muscle_manda_num_predict_de_la_fila(self):
        from jax.muscles.ollama_muscle import OllamaMuscle
        self.arrancar((None, 4096))
        m = OllamaMuscle("jax_local", "qwen-x", ["qwen-x"], "s", 10, api_url="http://ollama.example/api/chat")
        m.provider_id = "ollama"
        await m.invoke("hola")
        self.assertEqual(self.cap.bodies[0]["options"], {"num_predict": 4096})
        self.leer.assert_awaited_once_with("ollama", "qwen-x")

    async def test_ollama_muscle_sin_tope_no_despacha(self):
        from jax.muscles.ollama_muscle import OllamaMuscle
        self.arrancar((None, None))
        m = OllamaMuscle("jax_local", "qwen-x", ["qwen-x"], "s", 10, api_url="http://ollama.example/api/chat")
        m.provider_id = "ollama"
        with self.assertRaises(base.DispatchConfigMuscleError) as ctx:
            await m.invoke("hola")
        self.assertIn("UPDATE model SET max_output_tokens", str(ctx.exception))
        self.assertEqual(self.cap.bodies, [])


class JacobsOllamaNumPredictTest(unittest.IsolatedAsyncioTestCase):
    def arrancar(self, fila):
        import contrato_dispatch as cd_jacobs
        from jacobs import executor, plan
        self.executor, self.plan = executor, plan
        self.cap = _Captura()
        self.leer = AsyncMock(return_value=fila)
        for p in [
            patch.object(httpx.AsyncClient, "post", self.cap.patches()[0].new),
            patch.object(executor, "record_resolved_version_safe", AsyncMock()),
            patch.object(plan, "record_resolved_version_safe", AsyncMock()),
            patch.object(cd_jacobs, "_leer_contrato", self.leer),
        ]:
            p.start()
            self.addCleanup(p.stop)

    def _local(self):
        return _facet_ada(key="jax_local", provider_id="ollama", model="qwen-x", transport="ollama",
                          base_url="http://ollama.example/v1", credential="")

    async def test_invoke_ollama_manda_num_predict_de_la_fila(self):
        self.arrancar((None, 4096))
        await self.executor._invoke_ollama(self._local(), "hola", 10)
        self.assertEqual(self.cap.bodies[0]["options"], {"num_predict": 4096})
        self.leer.assert_awaited_once_with("ollama", "qwen-x")

    async def test_llm_plan_manda_el_menor_entre_su_presupuesto_y_el_tope(self):
        from jacobs import plan
        with patch.object(plan, "resolve_facet", AsyncMock(return_value=self._local())):
            self.arrancar((None, 1000))
            await plan.PlanBuilder()._llm_plan("o", 3, facetas_activas=frozenset({"hipatia"}))
            self.assertEqual(self.cap.bodies[0]["options"]["num_predict"], 1000)

    async def test_llm_plan_con_tope_mayor_manda_su_presupuesto(self):
        from jacobs import plan
        with patch.object(plan, "resolve_facet", AsyncMock(return_value=self._local())):
            self.arrancar((None, 10 ** 6))
            await plan.PlanBuilder()._llm_plan("o", 3, facetas_activas=frozenset({"hipatia"}))
            self.assertEqual(self.cap.bodies[0]["options"]["num_predict"], plan._LLM_PLAN_NUM_PREDICT)

    async def test_llm_plan_sin_contrato_no_despacha_y_da_error(self):
        from jacobs import plan
        with patch.object(plan, "resolve_facet", AsyncMock(return_value=self._local())):
            self.arrancar((None, None))
            with self.assertLogs("jacobs.plan", level="ERROR"), self.assertRaises(plan.CerebroNoDisponible):
                await plan.PlanBuilder()._llm_plan("o", 3, facetas_activas=frozenset({"hipatia"}))
            self.assertEqual(self.cap.bodies, [])


class ModoPesadoTest(unittest.TestCase):
    """M4: el modelo pesado es configuración y tiene que estar en el catálogo."""

    def _cfg(self, **modo):
        return {"jax": {"modo_pesado": modo},
                "personalities": {"jekyll": {"models_allowed": ["base-1", "pesado-1"]}}}

    def test_sale_de_la_configuracion(self):
        from jax.core.main import resolver_modo_pesado
        self.assertEqual(resolver_modo_pesado(self._cfg(faceta="jekyll", modelo="pesado-1")),
                         ("jekyll", "pesado-1", ""))

    def test_modelo_fuera_del_catalogo_no_se_activa(self):
        from jax.core.main import resolver_modo_pesado
        faceta, modelo, motivo = resolver_modo_pesado(self._cfg(faceta="jekyll", modelo="otro"))
        self.assertIsNone(faceta)
        self.assertIn("no está entre los modelos permitidos", motivo)

    def test_sin_configuracion_lo_dice(self):
        from jax.core.main import resolver_modo_pesado
        self.assertIn("no configurado", resolver_modo_pesado({"jax": {}, "personalities": {}})[2])

    def test_config_toml_lo_declara(self):
        import tomllib
        from pathlib import Path
        cfg = tomllib.loads((Path(__file__).resolve().parents[1] / "config" / "config.toml").read_text(encoding="utf-8"))
        modo = cfg["jax"]["modo_pesado"]
        self.assertIn(modo["faceta"], cfg["personalities"])
        self.assertTrue(modo["modelo"])


# ---------------------------------------------------------------------------
# Ronda 3
# ---------------------------------------------------------------------------

# provider.base_url REAL de producción (leído por el controller el 2026-09-14,
# solo lectura). Ollama trae '/v1' (camino OpenAI-compatible): un camino que
# armara el endpoint NATIVO pegándole /api/chat daría /v1/api/chat, que no existe.
_BASE_URL_PROD = {
    "deepseek": "https://api.deepseek.com/v1",
    "moonshot": "https://api.moonshot.ai/v1",
    "openai": "https://api.openai.com/v1",
    "zhipu": "https://api.z.ai/api/paas/v4",
    "gemini": "https://generativelanguage.googleapis.com/v1beta",
    "ollama": "http://localhost:11434/v1",
}


class TopeDeColumnaDePRLTest(_Base):
    """N1: el bloque verbatim es el de PR-L (tope de la columna INT)."""

    async def test_un_tope_mayor_que_la_columna_int_falla(self):
        self.arrancar(("max_tokens", 2 ** 31))
        with self.assertRaises(cd.ModelDispatchConfigError):
            await cd.limite_de_salida("http_openai_compat", "deepseek", "m")
        self.assertEqual(cd._MAX_OUTPUT_TOKENS_TOPE_COLUMNA, 2 ** 31 - 1)


class ClasificadorDelRouterTest(unittest.IsolatedAsyncioTestCase):
    """N2b: el clasificador del REPL nunca lanza, pero un contrato roto ya no
    se traga en silencio: WARNING con el motivo."""

    def _router(self, error):
        from jax.core.router import Router
        clasificador = AsyncMock()
        clasificador.invoke = AsyncMock(side_effect=error)
        return Router(classifier=clasificador)

    async def test_contrato_roto_deja_warning_con_el_update(self):
        err = base.DispatchConfigMuscleError(
            "[jax_local] dispatch abortado: modelo 'q': UPDATE model SET max_output_tokens=<tope>")
        with self.assertLogs("jax.router", level="WARNING") as logs:
            self.assertIsNone(await self._router(err)._classify("hola"))
        self.assertIn("UPDATE model SET max_output_tokens", "\n".join(logs.output))

    async def test_otra_falla_tambien_deja_rastro_pero_con_freno(self):
        """DECISION REVERTIDA el 2026-09-16 por Fernando, tras la auditoria P10.

        Este test afirmaba lo contrario —— se llamaba
        `test_otra_falla_sigue_cayendo_al_default_sin_ruido` y exigia
        `assertNoLogs`—— y la intencion era buena: separar la senal accionable
        (un contrato de dispatch roto, que trae el UPDATE que hay que correr)
        del ruido de un fallo de red transitorio.

        Lo que esa distincion no cubria: un fallo de red TRANSITORIO es ruido,
        pero uno PERMANENTE —— Ollama caido, el clasificador roto—— degrada el
        100 % del ruteo automatico a la faceta por defecto, indefinidamente y
        sin una sola linea que lo diga. Nadie se entera de que el router dejo
        de clasificar.

        El freno resuelve las dos cosas a la vez: se avisa la primera vez y
        despues cada _CLASIFICADOR_CADA_N, asi que un clasificador en bucle no
        inunda el log —— un log inundado se deja de leer, que es otra forma de
        callar—— pero la degradacion permanente si deja rastro, con el contador
        de cuantas veces fallo. Lo fija test_no_inunda_el_log en
        tests/test_degradaciones_declaradas.py.

        El test no se borro: se reescribio en su contrario, que es como esta
        casa cambia una decision fijada."""
        with self.assertLogs("jax.router", level="WARNING") as logs:
            self.assertIsNone(await self._router(RuntimeError("red caida"))._classify("hola"))
        salida = "\n".join(logs.output)
        self.assertIn("clasificador del router caido", salida)
        self.assertIn("NO esta clasificando", salida)

    async def test_el_contrato_roto_sigue_distinguiendose_de_una_falla_cualquiera(self):
        """La distincion original NO se perdio: el contrato roto sigue trayendo
        el UPDATE accionable, y una falla cualquiera no lo inventa."""
        err = base.DispatchConfigMuscleError(
            "[jax_local] dispatch abortado: modelo 'q': UPDATE model SET max_output_tokens=<tope>")
        with self.assertLogs("jax.router", level="WARNING") as contrato:
            await self._router(err)._classify("hola")
        with self.assertLogs("jax.router", level="WARNING") as cualquiera:
            await self._router(RuntimeError("red caida"))._classify("hola")
        self.assertIn("UPDATE model SET", "\n".join(contrato.output))
        self.assertNotIn("UPDATE model SET", "\n".join(cualquiera.output))


class UrlRealPorCaminoTest(_Base):
    """N4: con la provider.base_url de producción, cada camino llama al
    endpoint que corresponde y el parámetro del límite sigue a ESE endpoint."""

    def test_repl_http_arma_chat_completions_desde_la_base_url_real(self):
        from jax.core.registro_facetas import aplicar_registro
        for provider_id, clave in (("deepseek", "deepseek"), ("moonshot", "kimi"),
                                   ("openai", "openai"), ("zhipu", "zhipu")):
            with self.subTest(provider_id):
                cfg = {"personalities": {"f": {"type": "http", "provider": "x", "model_default": "m",
                                               "models_allowed": ["m"]}}}
                aplicar_registro(cfg, {"f": {"model": "m", "models_allowed": ["m"],
                                             "transport": "http_openai_compat",
                                             "provider_modelo": provider_id,
                                             "base_url_modelo": _BASE_URL_PROD[provider_id]}})
                self.assertEqual(cfg["personalities"]["f"]["api_url"],
                                 _BASE_URL_PROD[provider_id] + "/chat/completions")
                self.assertEqual(cfg["personalities"]["f"]["provider"], clave)

    async def test_repl_gemini_arma_generate_content_desde_la_base_url_real(self):
        from jax.core.main import build_muscles
        from jax.core.registro_facetas import aplicar_registro
        self.arrancar(("max_tokens", 1))
        cfg = _cfg_repl({"hipatia": {"type": "http", "provider": "gemini", "model_default": "g",
                                     "models_allowed": ["g"], "system_prompt": "s"}})
        aplicar_registro(cfg, {"hipatia": {"model": "gemini-x", "models_allowed": ["gemini-x"],
                                           "transport": "http_gemini", "provider_modelo": "gemini",
                                           "base_url_modelo": _BASE_URL_PROD["gemini"]}})
        await build_muscles(cfg)["hipatia"].invoke("hola")
        # Ruling T6-6 (2026-09-15): la key va en la cabecera x-goog-api-key,
        # nunca en la URL (httpx loguea la URL entera en INFO).
        self.assertEqual(
            self.cap.urls[0],
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-x:generateContent")
        self.assertIn("x-goog-api-key", self.cap.headers[0] or {})
        self.assertNotIn("max_tokens", self.cap.bodies[0])

    async def test_repl_ollama_usa_el_endpoint_nativo_del_entorno_y_num_predict(self):
        from jax.core.main import build_muscles
        from jax.core.registro_facetas import aplicar_registro
        self.arrancar((None, 262144))
        nativo = "http://ollama.invalid:11434/api/chat"  # JAX_OLLAMA_URL del conftest + /api/chat
        cfg = _cfg_repl({"jax_local": {"type": "ollama", "provider": "ollama", "model_default": "q",
                                       "models_allowed": ["q"], "system_prompt": "s"}})
        aplicar_registro(cfg, {"jax_local": {"model": "qwen-x", "models_allowed": ["qwen-x"],
                                             "transport": "ollama", "provider_modelo": "ollama",
                                             "base_url_modelo": _BASE_URL_PROD["ollama"]}})
        await build_muscles(cfg)["jax_local"].invoke("hola")
        self.assertEqual(self.cap.urls, [nativo])
        self.assertEqual(self.cap.bodies[0]["options"], {"num_predict": 262144})
        self.assertNotIn("max_tokens", self.cap.bodies[0])


class UrlRealJacobsYMotorTest(unittest.IsolatedAsyncioTestCase):
    """N4 en los caminos de Jacobs y del Motor Registry. Reusa el arranque de
    JacobsOllamaNumPredictTest sin heredar (y re-correr) sus tests."""

    arrancar = JacobsOllamaNumPredictTest.arrancar

    def _local_prod(self):
        return _facet_ada(key="jax_local", provider_id="ollama", model="qwen-x", transport="ollama",
                          base_url=_BASE_URL_PROD["ollama"], credential="")

    async def test_invoke_ollama_no_arma_el_nativo_desde_la_base_url(self):
        self.arrancar((None, 262144))
        await self.executor._invoke_ollama(self._local_prod(), "hola", 10)
        self.assertEqual(self.cap.urls, [self.executor.OLLAMA_URL])
        self.assertTrue(self.cap.urls[0].endswith("/api/chat") and "/v1/" not in self.cap.urls[0])
        self.assertEqual(self.cap.bodies[0]["options"], {"num_predict": 262144})

    async def test_llm_plan_no_arma_el_nativo_desde_la_base_url(self):
        from jacobs import plan
        with patch.object(plan, "resolve_facet", AsyncMock(return_value=self._local_prod())):
            self.arrancar((None, 262144))
            await plan.PlanBuilder()._llm_plan("o", 3, facetas_activas=frozenset({"hipatia"}))
        self.assertEqual(self.cap.urls, [plan.OLLAMA_URL])
        self.assertNotIn("/v1/", self.cap.urls[0])
        self.assertIn("num_predict", self.cap.bodies[0]["options"])

    async def test_motor_ollama_va_por_v1_chat_completions_con_max_tokens(self):
        from motor_registry import worker
        from motor_registry.catalog import MotorCatalog
        self.arrancar((None, 262144))
        entry = MotorCatalog({"motors": {"jax_local": {
            "enabled": True, "provider": "ollama", "transport": "ollama", "provider_id": "ollama",
            "api_key_env": "", "api_url": _BASE_URL_PROD["ollama"], "model": "qwen-x",
            "max_context_tokens": 0, "sandbox_only": True, "default_timeout_seconds": 60,
            "supports_reasoning": False, "reasoning_default_visibility": "none", "max_tokens": 0,
        }}, "capabilities": {}}).get_motor("jax_local")
        limite, _ = await worker._limite_del_motor(entry)
        await worker._call_http_openai_compat(
            api_url=entry.api_url, model=entry.model, api_key="",
            messages=[{"role": "user", "content": "x"}], timeout=5, limite=limite)
        self.assertEqual(self.cap.urls, ["http://localhost:11434/v1/chat/completions"])
        self.assertEqual(self.cap.bodies[0]["max_tokens"], 262144)
        self.assertNotIn("options", self.cap.bodies[0])


class ModuloUnicoTest(unittest.TestCase):
    def test_las_manos_lo_ve_por_symlink_al_mismo_archivo(self):
        from pathlib import Path
        raiz = Path(__file__).resolve().parents[1]
        link = raiz / "las_manos" / "contrato_dispatch.py"
        self.assertTrue(link.is_symlink(), "las_manos/contrato_dispatch.py debe ser symlink")
        self.assertEqual(link.resolve(), (raiz / "jax" / "core" / "contrato_dispatch.py").resolve())


if __name__ == "__main__":
    unittest.main()
