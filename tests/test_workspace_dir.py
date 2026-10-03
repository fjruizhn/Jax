"""JAX_WORKSPACE_DIR sin valor por defecto: falla cerrado (2026-10-03).

El workspace de produccion se movio de /home/fruiz/jax-workspace a
/srv/jax-data/jax-workspace. Con el default viejo escrito en tres sitios, un
proceso que arrancara sin /etc/jax/.env escribia en silencio en la ruta
abandonada y partia los datos en dos. Ahora hay UNA funcion y no hay default.
"""
from __future__ import annotations

import re
import subprocess
import sys
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


def test_absoluta_inexistente_lanza(monkeypatch, tmp_path):
    monkeypatch.setenv("JAX_WORKSPACE_DIR", str(tmp_path / "no-existe"))
    with pytest.raises(WorkspaceNoConfigurado) as exc:
        workspace_dir()
    assert "no existe o no es un directorio" in str(exc.value)


def test_archivo_regular_lanza(monkeypatch, tmp_path):
    archivo = tmp_path / "archivo"
    archivo.write_text("x")
    monkeypatch.setenv("JAX_WORKSPACE_DIR", str(archivo))
    with pytest.raises(WorkspaceNoConfigurado):
        workspace_dir()


def test_el_mensaje_trae_la_orden_para_correr_a_mano(monkeypatch):
    monkeypatch.delenv("JAX_WORKSPACE_DIR", raising=False)
    with pytest.raises(WorkspaceNoConfigurado) as exc:
        workspace_dir()
    assert "sudo -n grep '^JAX_WORKSPACE_DIR=' /etc/jax/.env | cut -d= -f2-" in str(exc.value)


@pytest.mark.parametrize("valor", ["jax-workspace", "./ws", "../ws", "~/ws"])
def test_relativa_lanza_con_motivo(monkeypatch, valor):
    monkeypatch.setenv("JAX_WORKSPACE_DIR", valor)
    with pytest.raises(WorkspaceNoConfigurado) as exc:
        workspace_dir()
    assert "absoluta" in str(exc.value)


def test_absoluta_devuelve_path_resuelto(monkeypatch, tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "ws").mkdir()
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
    """Los tres fallan al IMPORTAR (a proposito): ocr tiene `_WORKSPACE_DIR` a
    nivel de modulo (ocr.py), y con el van la compuerta y la ingesta. Aca se
    ejercita la funcion de resolucion de ocr, que es la que lanza."""
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
    p = subprocess.run([sys.executable, "-c", codigo], env=env, capture_output=True, text=True)
    assert p.returncode != 0
    assert "JAX_WORKSPACE_DIR no está configurada" in p.stderr, p.stderr[-800:]


_EXCLUIDOS = ("tests", "docs", ".git", "node_modules")
_LLAMADA_CON_DEFAULT = re.compile(
    r"""(?:getenv|environ\.get)\(\s*["']JAX_WORKSPACE_DIR["']\s*,"""
    r"""|(?:getenv|environ\.get)\(\s*["']JAX_WORKSPACE_DIR["']\s*\)[\s\\]*or[\s\\]*["'][^"']"""
    r"""|environ\.setdefault\(\s*["']JAX_WORKSPACE_DIR["']""")


def _violaciones(raiz: Path) -> list[str]:
    """.py bajo `raiz` (sin tests/, docs/, .git, policy/tests ni *_test.py)
    con el literal viejo o con un default en la lectura de la variable. Solo
    stdlib: no depende de git ni de que el checkout tenga .git.

    Detecta: `getenv/environ.get("JAX_WORKSPACE_DIR", <default>)` (tambien
    partido en lineas), `... ) or "<literal no vacio>"`,
    `environ.setdefault("JAX_WORKSPACE_DIR", ...)` y el literal viejo.

    NO detecta (alcance declarado, es un barrido de regex y no un analisis):
    `getenv(key="JAX_WORKSPACE_DIR", ...)`; una constante intermedia
    (`V = "JAX_WORKSPACE_DIR"; getenv(V, "/x")`);
    `Path("/home/fruiz") / "jax-workspace"` ni `"~/jax-workspace"` (el
    literal viejo se busca entero); y nada que no sea .py (shell, units
    systemd, YAML)."""
    malos = []
    for ruta in sorted(raiz.rglob("*.py")):
        rel = ruta.relative_to(raiz)
        partes = rel.parts
        if partes[0] in _EXCLUIDOS or partes[:2] == ("policy", "tests"):
            continue
        if ruta.name.endswith("_test.py") or any(p in (".git", "__pycache__", ".venv", "venv") for p in partes):
            continue
        texto = ruta.read_text(encoding="utf-8", errors="replace")
        # Con comentarios incluidos a proposito: un default que solo vive
        # en una linea comentada igual avisa de que alguien lo estaba pensando.
        if LITERAL_VIEJO in texto or _LLAMADA_CON_DEFAULT.search(texto):
            malos.append(str(rel))
    return malos


def test_el_barrido_detecta_el_literal_viejo(tmp_path):
    (tmp_path / "mod.py").write_text(f'RUTA = "{LITERAL_VIEJO}"\n')
    assert _violaciones(tmp_path) == ["mod.py"]


@pytest.mark.parametrize("llamada", [
    'os.getenv("JAX_WORKSPACE_DIR", "/srv/jax-data/jax-workspace")',
    "os.environ.get('JAX_WORKSPACE_DIR', '/srv/jax-data/jax-workspace')",
    'getenv( "JAX_WORKSPACE_DIR" ,None)',
])
def test_el_barrido_detecta_cualquier_default(tmp_path, llamada):
    (tmp_path / "otro.py").write_text(f"import os\nX = {llamada}\n")
    assert _violaciones(tmp_path) == ["otro.py"]


@pytest.mark.parametrize("llamada", [
    'os.environ.get("JAX_WORKSPACE_DIR") or "/srv/jax-data/jax-workspace"',
    "os.getenv('JAX_WORKSPACE_DIR') or '/srv/jax-data/jax-workspace'",
    'os.environ.get("JAX_WORKSPACE_DIR")\\\n    or "/srv/jax-data/jax-workspace"',
    'os.environ.setdefault("JAX_WORKSPACE_DIR", "/srv/jax-data/jax-workspace")',
    "os.environ.setdefault( 'JAX_WORKSPACE_DIR' ,\n    '/x')",
])
def test_el_barrido_detecta_or_y_setdefault(tmp_path, llamada):
    (tmp_path / "otro.py").write_text(f"import os\nX = {llamada}\n")
    assert _violaciones(tmp_path) == ["otro.py"]


def test_el_barrido_no_marca_or_con_literal_vacio(tmp_path):
    (tmp_path / "ok.py").write_text('import os\nX = os.environ.get("JAX_WORKSPACE_DIR") or ""\n')
    assert _violaciones(tmp_path) == []


def test_el_barrido_no_marca_la_lectura_sin_default_ni_las_exclusiones(tmp_path):
    (tmp_path / "ok.py").write_text('import os\nX = os.environ.get("JAX_WORKSPACE_DIR")\n')
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "t.py").write_text(f'X = "{LITERAL_VIEJO}"\n')
    (tmp_path / "a_test.py").write_text(f'X = "{LITERAL_VIEJO}"\n')
    assert _violaciones(tmp_path) == []


def test_literal_viejo_ni_defaults_vuelven_a_codigo_python():
    assert _violaciones(RAIZ) == []
