"""Fixed in-memory Faro package for the isolated MCP client smoke test."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import MappingProxyType

from jax.faro.paquete import PaqueteCargado

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "projects/las-voces/authority/faro_readonly_mcp.py"
spec = importlib.util.spec_from_file_location("lv_faro_readonly_fixture", PATH)
assert spec and spec.loader
bridge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge)

package = PaqueteCargado(
    sha="a" * 40,
    constitucion="# Constitución de prueba\n",
    skills=MappingProxyType({"alfa": MappingProxyType({"SKILL.md": b"---\ndescription: mide\n---\nlee"})}),
    agentes=MappingProxyType({"explorador": b"---\ndescription: descubre\nmodel: haiku\n---\nsecret-body"}),
)
raise SystemExit(bridge.serve(package))
