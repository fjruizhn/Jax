"""Fixed in-memory Faro package for the isolated MCP client smoke test."""
from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jax.faro.config import ConfigFaro
from jax.faro.paquete import construir_paquete, verificar_contra_arbol
from tests._faro_utils import repo_de_juguete

PATH = ROOT / "projects/las-voces/authority/faro_readonly_mcp.py"
spec = importlib.util.spec_from_file_location("lv_faro_readonly_fixture", PATH)
assert spec and spec.loader
bridge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge)

with tempfile.TemporaryDirectory(prefix="lv001b-faro-") as temp:
    base = Path(temp)
    repo = repo_de_juguete(base)
    sha = (repo / ".git/refs/heads/main").read_text().strip()
    cfg = ConfigFaro(repo=repo, sha=sha, destino=base / "ecosistema", uid_duenio=os.getuid())
    construir_paquete(cfg)
    assert not verificar_contra_arbol(cfg)
    raise SystemExit(bridge.serve_from_config(cfg))
