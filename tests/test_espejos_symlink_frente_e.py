"""E-10 / E-11 (2026-09-16): dentro de jax, cada módulo espejado es UN archivo.

las_manos/crypto_secrets.py, credential_resolver.py y model_catalog.py eran
copias reales de jax/core (crypto byte a byte; las otras dos distintas solo en
los imports), así que podían driftear DENTRO del repo, y model_catalog ni
siquiera estaba en FAMILIAS. Pasan a symlinks, como facet_resolver y redaccion.

El canónico importa BARE primero (LAS MANOS y Jacobs: WorkingDirectory o
PYTHONPATH con las_manos/, donde jax.core NO es importable) y cae a jax.core
(REPL y workers). En un contexto `.:las_manos` (CI) gana el bare:
jax.core.credential_resolver usa los módulos bare. Inocuo: ningún test parcha
jax.core.crypto_secrets.
"""
from __future__ import annotations

import ast
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("archivo", ["crypto_secrets.py", "credential_resolver.py", "model_catalog.py"])
def test_la_copia_de_las_manos_es_symlink_al_canonico(archivo):
    link = RAIZ / "las_manos" / archivo
    assert link.is_symlink(), f"las_manos/{archivo} debe ser symlink"
    assert link.resolve() == (RAIZ / "jax" / "core" / archivo).resolve()


@pytest.mark.parametrize("archivo, modulos", [
    ("credential_resolver.py", {"crypto_secrets", "db_connect_config"}),
    ("model_catalog.py", {"db_connect_config"}),
])
def test_el_canonico_importa_bare_primero_y_cae_a_jax_core(archivo, modulos):
    arbol = ast.parse((RAIZ / "jax" / "core" / archivo).read_text(encoding="utf-8"))
    vistos = set()
    for nodo in arbol.body:
        if not isinstance(nodo, ast.Try):
            continue
        bare = {n.module for n in nodo.body if isinstance(n, ast.ImportFrom)}
        for handler in nodo.handlers:
            if isinstance(handler.type, ast.Name) and handler.type.id == "ImportError":
                calificados = {n.module for n in handler.body if isinstance(n, ast.ImportFrom)}
                vistos |= {m for m in bare if f"jax.core.{m}" in calificados}
    assert modulos <= vistos


def _python(codigo: str, pythonpath: Path, cwd: Path) -> str:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(pythonpath)
    r = subprocess.run([sys.executable, "-c", codigo], cwd=cwd, env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


def test_en_el_repl_el_canonico_usa_jax_core():
    salida = _python(
        "import jax.core.credential_resolver as c, jax.core.model_catalog as m;"
        "print(c.decrypt_secret.__module__, c.db_connect_timeout_seconds.__module__, m.db_connect_timeout_seconds.__module__)",
        RAIZ, RAIZ)
    assert salida.split() == ["jax.core.crypto_secrets", "jax.core.db_connect_config", "jax.core.db_connect_config"]


def test_en_las_manos_el_bare_resuelve_al_archivo_canonico():
    salida = _python(
        "from pathlib import Path; import credential_resolver as c, model_catalog as m, crypto_secrets as s;"
        "print(Path(c.__file__).resolve(), Path(m.__file__).resolve(), Path(s.__file__).resolve(), c.decrypt_secret.__module__)",
        RAIZ / "las_manos", RAIZ / "las_manos")
    rutas = salida.split()
    assert rutas[:3] == [str((RAIZ / "jax" / "core" / n).resolve())
                         for n in ("credential_resolver.py", "model_catalog.py", "crypto_secrets.py")]
    assert rutas[3] == "crypto_secrets"


def test_las_notas_de_las_familias_dicen_symlink_y_no_tres_archivos():
    spec = importlib.util.spec_from_file_location("check_mirror_sync_e11", RAIZ / "scripts" / "check_mirror_sync.py")
    modulo = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = modulo
    spec.loader.exec_module(modulo)
    familias = {f.nombre: f for f in modulo.FAMILIAS}
    for nombre in ("crypto_secrets", "credential_resolver"):
        assert "TRES archivos reales" not in familias[nombre].nota
        assert "symlink" in familias[nombre].nota
