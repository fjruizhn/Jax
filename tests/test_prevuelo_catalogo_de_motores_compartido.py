"""El pre-vuelo lee el catálogo de motores del PROCESO, el mismo del Motor
Registry, con SU invalidación por sello (ola final F8, Ruling R33).

Perfil (scripts/perfil_prevuelo.py cpu, 1.000 pedidos en serie, antes de F8):
`MotorCatalog.from_db()` por pedido era la pieza más grande de la CPU del
pre-vuelo -- 0,32 ms de 0,96 ms (fases) y 52 % en cProfile. El despacho real
ya tenía un catálogo compartido que se recarga sólo cuando el sello de
facet_resolver cambia (motor_registry/routes.py::_ensure_catalog_fresh). No
es una caché nueva: es la fuente del despacho, y el pre-vuelo tiene que
evaluar exactamente lo que el despacho va a usar.

Sólo se mockea `MotorCatalog.from_db`; el sello es un archivo real en un
directorio temporal (mismo arnés que tests/test_motor_catalog_sello.py).

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
from unittest.mock import AsyncMock

os.environ["JAX_DB_NAME"] = "jax_memory_test"

import pytest  # noqa: E402

import facet_resolver  # noqa: E402
from jacobs import prevuelo_catalogo as pc  # noqa: E402
from jacobs.models import Step  # noqa: E402
from motor_registry import routes  # noqa: E402
from motor_registry.catalog import MotorCatalog  # noqa: E402


def _catalogo(max_tokens: int) -> MotorCatalog:
    return MotorCatalog({
        "motors": {"kimi": {
            "enabled": True, "provider": "kimi", "transport": "http_openai_compat",
            "provider_id": "moonshot", "api_key_env": "", "api_url": "http://x",
            "model": "kimi-k3", "max_context_tokens": 0, "sandbox_only": True,
            "default_timeout_seconds": 600, "supports_reasoning": True,
            "reasoning_default_visibility": "audit_only", "max_tokens": max_tokens,
        }},
        "capabilities": {"generate": {
            "allowed_motors": ["kimi"], "allowed_callers": ["jacobs"],
            "risk_level": "low", "sandbox_only": True, "requires_human_gate": False,
            "max_execution_minutes": 15, "max_recursion_depth": 0,
            "output_schema": "generate.v1",
        }},
    })


_PASOS = [Step(pipeline_id="p", step_index=0, facet="kimi", capability="generate", input={"prompt": "x"})]
_CONEXION = object()


@pytest.fixture
def registro(monkeypatch, tmp_path):
    sello = Path(tmp_path) / "facet-cache-seal"
    monkeypatch.setattr(facet_resolver, "FACET_SEAL_PATH", str(sello))
    from_db = AsyncMock(return_value=_catalogo(8000))
    monkeypatch.setattr(MotorCatalog, "from_db", from_db)
    monkeypatch.setattr(routes, "_CATALOG", None)
    monkeypatch.setattr(routes, "_POLICY", None)
    monkeypatch.setattr(routes, "_CATALOG_LOADED_AT_WALL", None)
    monkeypatch.setattr(routes, "_CATALOG_LOCK", asyncio.Lock())
    return from_db


def _resolver():
    return asyncio.run(pc.resolver_motores(_PASOS, conexion=_CONEXION))


def test_sin_cambio_de_sello_el_prevuelo_no_vuelve_a_leer_el_catalogo(registro):
    _resolver()
    assert registro.await_count == 1
    registro.assert_awaited_with(conexion=_CONEXION)  # la carga usa la conexión del pre-vuelo
    for _ in range(3):
        motores = _resolver()
    assert registro.await_count == 1, "from_db() se llamó sin que el sello cambiara"
    assert motores[0].max_tokens == 8000


def test_con_el_sello_nuevo_el_prevuelo_recarga_y_ve_el_cambio(registro):
    assert _resolver()[0].max_tokens == 8000
    registro.return_value = _catalogo(2000)
    assert facet_resolver._tocar_sello()
    motores = _resolver()
    assert registro.await_count == 2
    assert motores[0].max_tokens == 2000


def test_el_prevuelo_y_el_despacho_comparten_el_mismo_catalogo(registro):
    _resolver()
    asyncio.run(routes._ensure_catalog_fresh())
    assert registro.await_count == 1
    assert routes._CATALOG is not None


def test_recarga_fallida_falla_cerrado_sin_usar_el_catalogo_viejo(registro):
    _resolver()
    registro.side_effect = OSError("DB caída")
    assert facet_resolver._tocar_sello()
    with pytest.raises(Exception) as e:
        _resolver()
    assert "no se pudo recargar" in str(e.value.detail)
