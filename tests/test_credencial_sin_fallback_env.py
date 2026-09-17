"""B1.4 cerrado: la doble lectura DB->env se retiró (2026-09-17).

La ventana de medición existía con un criterio de salida escrito: 7 días
consecutivos sin ninguna línea `source=env_fallback`, incluyendo al menos una
rotación real. Medido el 2026-09-17 sobre 30 días de journal: 2.760 líneas
`source=db`, **0 líneas `source=env_fallback`**, con la rotación real de la
llave de Gemini del 2026-09-15 dentro de la ventana. Decisión de Fernando:
se retira el fallback en los dos repos (jax y jax-platform).

Lo que estos tests vigilan es que no vuelva: con la credencial AUSENTE en la DB
y la variable de entorno PRESENTE, el sistema falla cerrado
(`CredentialUnavailableError`) en vez de usar la env var. Un fallback a entorno
es un fail-open de la rotación: una llave revocada en la DB seguiría viva en el
proceso mientras la variable siguiera cargada.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from jax.core import credential_resolver
from jax.core.credential_resolver import CredentialUnavailableError
from jax.muscles.base import HttpMuscle, MuscleInvocationError

RAIZ = Path(__file__).resolve().parent.parent


class _MusculoDePrueba(HttpMuscle):
    async def _call(self, prompt, model, history=None):  # pragma: no cover - no se invoca
        raise AssertionError("este test no llega a _call")


def _musculo() -> _MusculoDePrueba:
    return _MusculoDePrueba(
        name="prueba",
        provider="gemini",
        model_default="m",
        models_allowed=["m"],
        system_prompt="",
        timeout=1.0,
    )


@pytest.fixture(autouse=True)
def _sin_cache():
    credential_resolver._cache.clear()
    yield
    credential_resolver._cache.clear()


def test_sin_credencial_en_db_no_se_usa_la_env_var(monkeypatch):
    """EL test del retiro. DB sin credencial activa + GEMINI_API_KEY presente
    en el entorno => fail-closed. Contra el código viejo este test PASABA la
    llave de la env var y no levantaba nada."""
    monkeypatch.setenv("GEMINI_API_KEY", "sk-de-la-env-var-no-usar")
    sin_credencial = AsyncMock(side_effect=CredentialUnavailableError("gemini"))
    with patch.object(credential_resolver, "_query_active_credential", sin_credencial):
        with pytest.raises(MuscleInvocationError) as exc:
            asyncio.run(_musculo()._resolve_api_key())
    assert not "sk-de-la-env-var-no-usar" not in str(exc.value)
    assert sin_credencial.await_count == 1


def test_el_camino_a_entorno_ya_no_existe_en_el_modulo():
    """El mapa proveedor->env var y la función instrumentada se retiraron: no
    quedan como código muerto esperando a que alguien los vuelva a llamar."""
    assert not hasattr(credential_resolver, "_PROVIDER_ENV_KEY_MAP")
    assert not hasattr(credential_resolver, "resolve_credential_instrumented")


def _fuentes() -> list[Path]:
    saltar = {".git", "node_modules", "__pycache__", "docs", ".superpowers"}
    archivos = []
    for p in RAIZ.rglob("*.py"):
        if saltar & set(p.relative_to(RAIZ).parts):
            continue
        if p.resolve() == Path(__file__).resolve():
            continue
        archivos.append(p)
    return archivos


def test_ningun_consumidor_sigue_llamando_a_la_version_instrumentada():
    """Los consumidores llaman a `resolve_credential()` directo. Si alguien
    reintroduce el nombre, este test lo nombra con archivo y línea."""
    hallazgos = []
    for archivo in _fuentes():
        for n, linea in enumerate(archivo.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if "resolve_credential_instrumented" in linea or "_PROVIDER_ENV_KEY_MAP" in linea:
                hallazgos.append(f"{archivo.relative_to(RAIZ)}:{n}")
    assert hallazgos == []
