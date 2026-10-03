"""JAX_WORKSPACE_DIR sin valor por defecto: falla cerrado (2026-10-03).

El workspace de produccion se movio de /home/fruiz/jax-workspace a
/srv/jax-data/jax-workspace. Con el default viejo escrito en tres sitios, un
proceso que arrancara sin /etc/jax/.env escribia en silencio en la ruta
abandonada y partia los datos en dos. Ahora hay UNA funcion y no hay default.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from jax.core.workspace_dir import WorkspaceNoConfigurado, workspace_dir

RAIZ = Path(__file__).resolve().parents[1]
LITERAL_VIEJO = "/home/fruiz/" + "jax-workspace"


def test_ausente_lanza(monkeypatch):
    monkeypatch.delenv("JAX_WORKSPACE_DIR", raising=False)
    with pytest.raises(WorkspaceNoConfigurado) as exc:
        workspace_dir()
    assert "JAX_WORKSPACE_DIR no está configurada" in str(exc.value)
    assert "/etc/jax/.env" in str(exc.value)


@pytest.mark.parametrize("valor", ["", "   ", "\t\n"])
def test_vacia_o_solo_espacios_lanza(monkeypatch, valor):
    monkeypatch.setenv("JAX_WORKSPACE_DIR", valor)
    with pytest.raises(WorkspaceNoConfigurado):
        workspace_dir()


@pytest.mark.parametrize("valor", ["jax-workspace", "./ws", "../ws", "~/ws"])
def test_relativa_lanza_con_motivo(monkeypatch, valor):
    monkeypatch.setenv("JAX_WORKSPACE_DIR", valor)
    with pytest.raises(WorkspaceNoConfigurado) as exc:
        workspace_dir()
    assert "absoluta" in str(exc.value)


def test_absoluta_devuelve_path_resuelto(monkeypatch, tmp_path):
    monkeypatch.setenv("JAX_WORKSPACE_DIR", str(tmp_path / "a" / ".." / "ws"))
    r = workspace_dir()
    assert isinstance(r, Path)
    assert r == (tmp_path / "ws").resolve()


def test_symlink_se_resuelve_a_su_destino(monkeypatch, tmp_path):
    destino = tmp_path / "real"
    destino.mkdir()
    enlace = tmp_path / "enlace"
    enlace.symlink_to(destino)
    monkeypatch.setenv("JAX_WORKSPACE_DIR", str(enlace))
    assert workspace_dir() == destino.resolve()


def test_valor_con_espacios_alrededor_se_recorta(monkeypatch, tmp_path):
    monkeypatch.setenv("JAX_WORKSPACE_DIR", f"  {tmp_path}  ")
    assert workspace_dir() == tmp_path.resolve()


def test_los_tres_usos_fallan_cerrado_sin_variable(monkeypatch):
    """ocr falla al llamarla; los otros dos al importarse (a proposito)."""
    import sys
    sys.path.insert(0, str(RAIZ / "las_manos"))
    try:
        from procesamiento.extractores import ocr
    finally:
        sys.path.remove(str(RAIZ / "las_manos"))
    monkeypatch.delenv("JAX_WORKSPACE_DIR", raising=False)
    # RuntimeError y no la clase: ocr importa la funcion por el symlink bare
    # (`workspace_dir`) y aca se importa por `jax.core.workspace_dir`; son dos
    # modulos distintos con dos clases homonimas.
    with pytest.raises(RuntimeError, match="JAX_WORKSPACE_DIR no está configurada"):
        ocr._resolver_workspace_dir()


@pytest.mark.parametrize("modulo", ["jacobs.executor", "motor_registry.tool_authority"])
def test_import_a_nivel_de_modulo_sin_variable_no_arranca(modulo):
    codigo = (
        "import os, sys; os.environ.pop('JAX_WORKSPACE_DIR', None); "
        f"sys.path[:0] = [{str(RAIZ)!r}, {str(RAIZ / 'las_manos')!r}]; "
        f"import {modulo}"
    )
    # Sin conftest: proceso limpio. Las otras env obligatorias se fijan por si
    # el import falla antes por otra causa -- se exige la nuestra.
    env = {
        "PATH": "/usr/bin:/bin",
        "LAS_MANOS_URL": "http://x.invalid:1",
        "JAX_OLLAMA_URL": "http://y.invalid:1",
        "JAX_LAS_MANOS_CREDENCIAL_JACOBS": "x",
        "JAX_LAS_MANOS_CREDENCIAL_PLATAFORMA": "x",
        "JAX_REPO_BASE": "/tmp",
    }
    p = subprocess.run(["python3", "-c", codigo], env=env, capture_output=True, text=True)
    assert p.returncode != 0
    assert "JAX_WORKSPACE_DIR no está configurada" in p.stderr, p.stderr[-800:]


def test_literal_viejo_no_vuelve_a_codigo_python():
    """Barrido: el literal del workspace viejo no puede reaparecer en .py
    fuera de tests/ y docs/ (ni en los *_test.py junto al codigo, que son
    pruebas: se excluyen por nombre)."""
    salida = subprocess.run(
        ["git", "grep", "-l", "-F", LITERAL_VIEJO, "--", "*.py",
         ":!tests", ":!docs", ":!*_test.py", ":!policy/tests"],
        cwd=RAIZ, capture_output=True, text=True,
    )
    assert salida.stdout.strip() == "", f"literal viejo reaparecio en: {salida.stdout}"
