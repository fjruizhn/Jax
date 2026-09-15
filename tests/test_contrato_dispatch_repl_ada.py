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
    )


class _Captura:
    """Parchea el HTTP (post y stream) y guarda el body enviado."""

    def __init__(self):
        self.bodies = []

    def patches(self):
        cap = self

        async def fake_post(client_self, url, json=None, **kw):
            cap.bodies.append(json)
            if "generativelanguage" in url:
                return _Resp({"candidates": [{"content": {"parts": [{"text": "hola"}]}}]})
            return _Resp({"model": "m", "choices": [{"message": {"content": "hola"}}]})

        @asynccontextmanager
        async def fake_stream(client_self, method, url, json=None, **kw):
            cap.bodies.append(json)
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

    async def test_el_helper_rechaza_un_proveedor_no_openai_compat_con_su_parametro(self):
        self.arrancar(("max_tokens", 1))
        for proveedor, propio in (("gemini", "generationConfig.maxOutputTokens"),
                                  ("ollama", "options.num_predict"),
                                  ("anthropic", "max_tokens (obligatorio")):
            with self.assertRaises(cd.ModelDispatchConfigError) as ctx:
                await cd.limite_de_salida(proveedor, "x")
            self.assertIn(propio, str(ctx.exception), proveedor)
        self.leer.assert_not_awaited()


class AdaPlanDelCatalogoTest(unittest.IsolatedAsyncioTestCase):
    """jacobs/plan.py importa `contrato_dispatch` pelado (symlink de
    las_manos/): se parchea ESE módulo, que es el que usa en producción."""

    def arrancar(self, fila):
        import contrato_dispatch as cd_jacobs
        from jacobs import plan
        self.plan = plan
        self.cap = _Captura()
        self.leer = AsyncMock(return_value=fila)
        for p in [
            patch.object(httpx.AsyncClient, "stream", self.cap.patches()[1].new),
            patch.object(plan, "resolve_credential_instrumented", AsyncMock(return_value="k")),
            patch.object(cd_jacobs, "_leer_contrato", self.leer),
        ]:
            p.start()
            self.addCleanup(p.stop)

    async def test_ada_manda_el_nombre_y_el_tope_de_la_fila_de_ADA_MODEL(self):
        self.arrancar(("max_tokens", 131072))
        await self.plan.PlanBuilder()._ada_plan("objetivo", 3)
        body = self.cap.bodies[0]
        self.assertEqual(body["max_tokens"], 131072)
        self.assertEqual(body["model"], self.plan.ADA_MODEL)
        self.leer.assert_awaited_once_with("zhipu", self.plan.ADA_MODEL)

    async def test_ada_con_otro_nombre_de_parametro_lo_respeta(self):
        self.arrancar(("max_completion_tokens", 5000))
        await self.plan.PlanBuilder()._ada_plan("objetivo", 3)
        body = self.cap.bodies[0]
        self.assertEqual(body["max_completion_tokens"], 5000)
        self.assertNotIn("max_tokens", body)

    async def test_ada_sin_contrato_no_despacha_y_lo_dice_en_error(self):
        self.arrancar((None, None))
        with self.assertLogs("jacobs.plan", level="ERROR") as logs:
            resultado = await self.plan.PlanBuilder()._ada_plan("objetivo", 3)
        self.assertIsNone(resultado)
        self.assertEqual(self.cap.bodies, [], "Ada no puede despachar sin contrato")
        self.assertIn("UPDATE model SET max_tokens_param", "\n".join(logs.output))


class ModuloUnicoTest(unittest.TestCase):
    def test_las_manos_lo_ve_por_symlink_al_mismo_archivo(self):
        from pathlib import Path
        raiz = Path(__file__).resolve().parents[1]
        link = raiz / "las_manos" / "contrato_dispatch.py"
        self.assertTrue(link.is_symlink(), "las_manos/contrato_dispatch.py debe ser symlink")
        self.assertEqual(link.resolve(), (raiz / "jax" / "core" / "contrato_dispatch.py").resolve())


if __name__ == "__main__":
    unittest.main()
