"""Sandbox de las instalaciones de dependencias (spec 2026-09-28 v1.2, §3.1; ruling del
controlador tras la re-revisión de escalón 3, BLOCK-1): pip, npm y composer ejecutan código de
terceros, y `jaxsvc` lee `/etc/jax/.env`. Filtrar el entorno no alcanza: la instalación corre en
bwrap por LISTA BLANCA -- solo lo del sistema que hace falta, de solo lectura, las copias de los
archivos de dependencias de solo lectura, y `deps/` como único lugar escribible."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from jax.ejecutor.codigo import sandbox as S

PROHIBIDAS = ("/etc/jax", "/var/lib/jaxsvc", "/srv", "/home")


def _montajes(argv):
    """[(opción, origen, destino)] de los montajes con origen del argv de bwrap."""
    salida = []
    i = 0
    fin = argv.index("--")
    while i < fin:
        if argv[i] in ("--bind", "--ro-bind", "--ro-bind-try", "--bind-try", "--dev-bind"):
            salida.append((argv[i], argv[i + 1], argv[i + 2]))
            i += 3
        else:
            i += 1
    return salida


def test_argv_por_lista_blanca(tmp_path):
    deps = tmp_path / "m" / "deps"
    (deps / "node" / "_raiz").mkdir(parents=True)
    copia = tmp_path / "copias" / "package.json"
    copia.parent.mkdir()
    copia.write_text("{}")
    argv = S.argv_sandbox(["npm", "ci"], deps=deps, cwd=deps / "node" / "_raiz",
                          solo_lectura=[(copia, deps / "node" / "_raiz" / "package.json")],
                          node_bin=Path("/opt/ejecutor/node-v24/bin"))
    assert argv[0] == "bwrap" and argv[-3:] == ["--", "npm", "ci"]
    for bandera in ("--unshare-all", "--share-net", "--die-with-parent", "--new-session", "--clearenv"):
        assert bandera in argv[:argv.index("--")], bandera
    montajes = _montajes(argv)
    assert [m for m in montajes if m[0] == "--bind"] == [("--bind", str(deps), str(deps))]
    assert ("--ro-bind", str(copia), str(deps / "node" / "_raiz" / "package.json")) in montajes
    assert ("--ro-bind-try", "/opt/ejecutor/node-v24", "/opt/ejecutor/node-v24") in montajes
    assert ("--ro-bind", "/usr", "/usr") in montajes
    for texto in argv:
        for prohibida in PROHIBIDAS:
            assert not (texto == prohibida or texto.startswith(prohibida + "/")), texto
    assert str(tmp_path / "m") not in argv, "el directorio de la misión no se monta: solo deps/"
    env = {argv[i + 1]: argv[i + 2] for i, a in enumerate(argv[:argv.index("--")]) if a == "--setenv"}
    assert env == {"PATH": "/opt/ejecutor/node-v24/bin:/usr/local/bin:/usr/bin:/bin", "HOME": "/tmp/h",
                   "LANG": "C.UTF-8"}
    assert argv[argv.index("--chdir") + 1] == str(deps / "node" / "_raiz")


@pytest.mark.parametrize("fuente", ["/etc/jax/.env", "/home/fruiz/x", "/srv/jax-data/y", "/var/lib/jaxsvc/z"])
def test_argv_rechaza_montar_una_ruta_prohibida(tmp_path, fuente):
    deps = tmp_path / "deps"
    deps.mkdir()
    with pytest.raises(ValueError, match="montaje_prohibido"):
        S.argv_sandbox(["true"], deps=deps, cwd=deps, solo_lectura=[(Path(fuente), deps / "x")])


def test_argv_rechaza_node_bajo_home(tmp_path):
    deps = tmp_path / "deps"
    deps.mkdir()
    with pytest.raises(ValueError, match="montaje_prohibido"):
        S.argv_sandbox(["true"], deps=deps, cwd=deps, node_bin=Path("/home/fruiz/.nvm/versions/node/v24/bin"))


@pytest.mark.parametrize("node_bin", ["/etc/node", "/node", "/var/lib/node", "/srv/node/bin"])
def test_argv_valida_lo_que_se_monta_para_node(tmp_path, node_bin):
    """MINOR-B: se monta el PADRE de node_bin; si ese padre contiene una ruta prohibida
    (`/etc` contiene `/etc/jax`, `/` contiene todo), se rechaza."""
    deps = tmp_path / "deps"
    deps.mkdir()
    with pytest.raises(ValueError, match="montaje_prohibido"):
        S.argv_sandbox(["true"], deps=deps, cwd=deps, node_bin=Path(node_bin))


def test_argv_rechaza_un_origen_que_contiene_una_prohibida(tmp_path):
    deps = tmp_path / "deps"
    deps.mkdir()
    for origen in ("/", "/etc", "/var/lib"):
        with pytest.raises(ValueError, match="montaje_prohibido"):
            S.argv_sandbox(["true"], deps=deps, cwd=deps, solo_lectura=[(Path(origen), deps / "x")])


def test_argv_rechaza_cwd_fuera_de_deps(tmp_path):
    deps = tmp_path / "deps"
    deps.mkdir()
    with pytest.raises(ValueError, match="cwd_fuera_de_deps"):
        S.argv_sandbox(["true"], deps=deps, cwd=tmp_path)


def _bwrap_utilizable() -> bool:
    if shutil.which("bwrap") is None:
        return False
    r = subprocess.run(["bwrap", "--unshare-all", "--ro-bind", "/usr", "/usr", "--symlink", "usr/bin", "/bin",
                        "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64", "--proc", "/proc",
                        "--dev", "/dev", "--", "/bin/true"], capture_output=True)
    return r.returncode == 0


requiere_bwrap = pytest.mark.skipif(not _bwrap_utilizable(), reason="bwrap o user namespaces no disponibles")


@requiere_bwrap
def test_bwrap_real_el_instalador_no_ve_secretos_ni_escribe_fuera(tmp_path, monkeypatch):
    monkeypatch.setenv("JAX_DB_PASSWORD", "secreto-del-entorno")
    secreto = tmp_path / "fuera" / "secreto.env"
    secreto.parent.mkdir()
    secreto.write_text("JAX_GITHUB_TOKEN=github_pat_SECRETO")
    deps = tmp_path / "m" / "deps"
    deps.mkdir(parents=True)
    (tmp_path / "m" / "espejo.git").mkdir()
    (tmp_path / "m" / "espejo.git" / "config").write_text("[remote]")
    copia = tmp_path / "copias" / "requirements.txt"
    copia.parent.mkdir()
    copia.write_text("x==1\n")
    home_real = Path.home()
    instalador = (
        f'cat {secreto} 2>/dev/null && echo LEI_SECRETO; '
        f'ls {home_real} >/dev/null 2>&1 && echo VI_HOME; '
        # Un archivo REAL fuera de /tmp (el checkout): el secreto de arriba vive en /tmp, que el
        # sandbox tapa con su propio tmpfs -- solo con él, montar "/" entero no se notaría.
        f'cat {Path(__file__).resolve()} >/dev/null 2>&1 && echo LEI_EL_REPO; '
        f'cat {tmp_path / "m" / "espejo.git" / "config"} 2>/dev/null && echo VI_EL_ESPEJO; '
        f'touch {tmp_path / "fuera" / "escrito"} 2>/dev/null && echo ESCRIBI_FUERA; '
        f'echo x > {copia} 2>/dev/null && echo ESCRIBI_LA_COPIA; '
        f'env | grep -q JAX_ && echo VI_JAX; '
        f'cat {copia} && echo hecho > {deps}/instalado && echo ESCRIBI_DEPS'
    )
    argv = S.argv_sandbox(["/bin/sh", "-c", instalador], deps=deps, cwd=deps, solo_lectura=[(copia, copia)])
    # bwrap recibe el entorno COMPLETO del proceso (con JAX_*): lo que lo limpia es --clearenv.
    r = subprocess.run(argv, capture_output=True, text=True, env=dict(os.environ))
    assert r.returncode == 0, r.stderr
    salida = r.stdout.split()
    for marca in ("LEI_SECRETO", "LEI_EL_REPO", "VI_HOME", "VI_EL_ESPEJO", "ESCRIBI_FUERA", "ESCRIBI_LA_COPIA", "VI_JAX"):
        assert marca not in salida, (marca, r.stdout)
    assert "x==1" in r.stdout and "ESCRIBI_DEPS" in salida
    assert (deps / "instalado").read_text() == "hecho\n"
    assert not (tmp_path / "fuera" / "escrito").exists() and copia.read_text() == "x==1\n"
