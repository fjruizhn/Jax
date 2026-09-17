"""E-21 (2026-09-16): las URLs de servicio salen del entorno (/etc/jax/.env),
validadas al arrancar; las de proveedor salen del catálogo, sin default.

Antes: LAS_MANOS_BASE y OLLAMA_URL fijos en jacobs/executor.py y plan.py, el
embed de la memoria a http://localhost:11434 fijo, DEFAULT_OLLAMA_URL en el
músculo local, api_url en config.toml, y URLs de OpenAI/DeepSeek/Gemini como
default en jax/muscles/base.py (con la DB caída se despachaba a una URL que
nadie eligió).
"""
from __future__ import annotations

import ast
import asyncio
import inspect
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest

os.environ["JAX_DB_NAME"] = "jax_memory_test"

RAIZ = Path(__file__).resolve().parents[1]


def test_url_que_falta_es_error_que_nombra_la_variable(monkeypatch):
    from jax.core.config_entorno import EntornoInvalido, url_requerida
    monkeypatch.delenv("JAX_URL_DE_PRUEBA", raising=False)
    with pytest.raises(EntornoInvalido) as e:
        url_requerida("JAX_URL_DE_PRUEBA")
    assert "JAX_URL_DE_PRUEBA" in str(e.value) and "/etc/jax/.env" in str(e.value)


def test_url_sin_esquema_o_sin_host_es_error(monkeypatch):
    from jax.core.config_entorno import EntornoInvalido, url_requerida
    for valor in ("localhost:11434", "ftp://x", "http://"):
        monkeypatch.setenv("JAX_URL_DE_PRUEBA", valor)
        with pytest.raises(EntornoInvalido):
            url_requerida("JAX_URL_DE_PRUEBA")


def test_url_valida_vuelve_sin_barra_final(monkeypatch):
    from jax.core.config_entorno import url_requerida
    monkeypatch.setenv("JAX_URL_DE_PRUEBA", " https://servicio.test:8443/ ")
    assert url_requerida("JAX_URL_DE_PRUEBA") == "https://servicio.test:8443"


def test_ruta_relativa_es_error(monkeypatch):
    from jax.core.config_entorno import EntornoInvalido, ruta_absoluta_requerida
    monkeypatch.setenv("JAX_RUTA_DE_PRUEBA", "jax/repo")
    with pytest.raises(EntornoInvalido):
        ruta_absoluta_requerida("JAX_RUTA_DE_PRUEBA")


def test_ruta_absoluta_valida(monkeypatch):
    from jax.core.config_entorno import ruta_absoluta_requerida
    monkeypatch.setenv("JAX_RUTA_DE_PRUEBA", "/srv/algo")
    assert ruta_absoluta_requerida("JAX_RUTA_DE_PRUEBA") == Path("/srv/algo")


def _proceso_limpio(codigo: str, **entorno) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{RAIZ}{os.pathsep}{RAIZ / 'las_manos'}"
    for clave, valor in entorno.items():
        if valor is None:
            env.pop(clave, None)
        else:
            env[clave] = valor
    return subprocess.run([sys.executable, "-c", codigo], cwd=RAIZ, env=env,
                          capture_output=True, text=True, timeout=120)


def test_jacobs_toma_las_urls_del_entorno():
    r = _proceso_limpio(
        "from jacobs import executor, plan; print(executor.LAS_MANOS_BASE, executor.OLLAMA_URL, plan.OLLAMA_URL)",
        LAS_MANOS_URL="http://las-manos.test:7777/", JAX_OLLAMA_URL="http://ollama.test:11434")
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["http://las-manos.test:7777", "http://ollama.test:11434/api/chat",
                                "http://ollama.test:11434/api/chat"]


def test_jacobs_no_arranca_sin_LAS_MANOS_URL():
    r = _proceso_limpio("from jacobs import executor", LAS_MANOS_URL=None)
    assert r.returncode != 0
    assert "LAS_MANOS_URL" in r.stderr


def test_la_memoria_sin_JAX_OLLAMA_URL_falla_visible(monkeypatch):
    from jax.core.config_entorno import EntornoInvalido
    from jax.memory.db import MemoryDB
    monkeypatch.delenv("JAX_OLLAMA_URL", raising=False)
    with pytest.raises(EntornoInvalido):
        asyncio.run(MemoryDB().get_embedding("hola"))


def test_la_memoria_pide_el_embedding_a_JAX_OLLAMA_URL(monkeypatch):
    from jax.memory import db as dbmod
    monkeypatch.setenv("JAX_OLLAMA_URL", "http://ollama.test:11434/")
    urls = []

    async def handle(transport_self, request):
        urls.append(str(request.url))
        return httpx.Response(200, json={"embeddings": [[0.0] * dbmod.EMBED.dim]})

    async def correr():
        memoria = dbmod.MemoryDB()
        with patch("httpx.AsyncHTTPTransport.handle_async_request", handle):
            vector = await memoria.get_embedding("hola")
        await memoria.close()
        return vector

    assert len(asyncio.run(correr())) == dbmod.EMBED.dim
    assert urls == ["http://ollama.test:11434/api/embed"]


def test_el_musculo_local_no_tiene_url_por_defecto_y_el_repl_la_toma_del_entorno(monkeypatch):
    from jax.core.main import build_muscles
    from jax.muscles.ollama_muscle import OllamaMuscle
    assert inspect.signature(OllamaMuscle.__init__).parameters["api_url"].default is inspect.Parameter.empty
    monkeypatch.setenv("JAX_OLLAMA_URL", "http://ollama.test:11434")
    cfg = {"jax": {"timeout_seconds": 10}, "personalities": {"jax_local": {
        "type": "ollama", "provider": "ollama", "model_default": "q", "models_allowed": ["q"], "system_prompt": "s"}}}
    assert build_muscles(cfg)["jax_local"].api_url == "http://ollama.test:11434/api/chat"


@pytest.mark.parametrize("proveedor, metodo", [
    ("deepseek", "_call_deepseek"), ("openai", "_call_openai"), ("gemini", "_call_gemini"),
])
def test_sin_url_del_catalogo_el_musculo_http_no_despacha(proveedor, metodo):
    from jax.muscles import base
    musculo = base.HttpMuscle(name="f", provider=proveedor, model_default="x", models_allowed=["x"],
                              system_prompt="s", timeout=10)
    red = AsyncMock(side_effect=AssertionError("salió a la red sin URL del catálogo"))

    async def correr():
        with patch("httpx.AsyncHTTPTransport.handle_async_request", red), \
             patch.object(base.HttpMuscle, "_resolve_api_key", AsyncMock(return_value="k")), \
             patch.object(base.HttpMuscle, "_limite_de_salida", AsyncMock(return_value={})):
            await getattr(musculo, metodo)("hola", "x")

    with pytest.raises(base.MuscleInvocationError) as e:
        asyncio.run(correr())
    assert "catálogo" in str(e.value)


_SIN_URLS_LITERALES = ("jacobs/executor.py", "jacobs/plan.py", "jax/memory/db.py",
                       "jax/muscles/base.py", "jax/muscles/ollama_muscle.py")


def _urls_literales(fuente: str) -> list[int]:
    arbol = ast.parse(fuente)
    docstrings = {id(n.value) for n in ast.walk(arbol)
                  if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)}
    return [n.lineno for n in ast.walk(arbol)
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docstrings
            and re.search(r"\bhttps?://", n.value)]


def test_ningun_modulo_de_servicio_tiene_una_url_escrita():
    hallazgos = {rel: _urls_literales((RAIZ / rel).read_text(encoding="utf-8")) for rel in _SIN_URLS_LITERALES}
    assert {rel: l for rel, l in hallazgos.items() if l} == {}


def test_config_toml_de_jax_no_trae_urls():
    with open(RAIZ / "config" / "config.toml", "rb") as fh:
        cfg = tomllib.load(fh)
    assert [clave for clave, p in cfg["personalities"].items() if "api_url" in p] == []


def test_el_conftest_no_apunta_a_los_servicios_de_produccion():
    """Revisión final del frente E: el conftest raíz fijaba LAS_MANOS_URL y
    JAX_OLLAMA_URL a 127.0.0.1:7777 y localhost:11434, los mismos valores que
    producción. Un test que olvide parchear el transporte le pegaba a LAS MANOS
    o al Ollama vivos. Con un host `.invalid` (RFC 6761, nunca resuelve) ese
    olvido falla con un error de DNS en vez de salir a un servicio real."""
    from urllib.parse import urlsplit
    for nombre in ("LAS_MANOS_URL", "JAX_OLLAMA_URL"):
        host = urlsplit(os.environ[nombre]).hostname
        assert host.endswith(".invalid"), f"{nombre} del conftest apunta a {host!r}"


def test_url_base_con_path_query_o_fragmento_es_error(monkeypatch):
    """Revisión final del frente E: JAX_OLLAMA_URL=http://localhost:11434/v1
    pasaba y el código armaba /v1/api/chat. Las dos variables que usan
    url_requerida (LAS_MANOS_URL, JAX_OLLAMA_URL) son URLs BASE: el código les
    agrega la ruta, así que un path, una query o un fragmento es un error."""
    from jax.core.config_entorno import EntornoInvalido, url_requerida
    for valor in ("http://ollama.test:11434/v1", "http://ollama.test:11434/v1/",
                  "http://ollama.test:11434?x=1", "http://ollama.test:11434#f"):
        monkeypatch.setenv("JAX_URL_DE_PRUEBA", valor)
        with pytest.raises(EntornoInvalido) as e:
            url_requerida("JAX_URL_DE_PRUEBA")
        assert "JAX_URL_DE_PRUEBA" in str(e.value), valor


def test_la_copia_de_jax_platform_es_una_familia_de_espejos_completa():
    """Revisión final del frente E: jax-platform valida JAX_OLLAMA_URL al
    arrancar con una copia verbatim de este módulo (backend/config_entorno.py).
    Sin familia en check_mirror_sync, un arreglo de la regla en un repo no se
    vería en el otro. `compartidos` cubre TODOS los símbolos de nivel superior:
    una familia a medias da verde sin mirar lo que falta."""
    sys.path.insert(0, str(RAIZ / "scripts"))
    from check_mirror_sync import FAMILIAS  # noqa: E402

    familia = next((f for f in FAMILIAS if f.nombre == "config_entorno"), None)
    assert familia is not None, [f.nombre for f in FAMILIAS]
    assert familia.canonico == RAIZ / "jax" / "core" / "config_entorno.py"
    assert {e for e, _ in familia.espejos} == {"las_manos", "jax-platform"}
    arbol = ast.parse(familia.canonico.read_text(encoding="utf-8"))
    del_archivo = {n.name for n in arbol.body if hasattr(n, "name")}
    assert del_archivo - set(familia.compartidos) == set()
