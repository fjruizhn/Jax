"""`docker rm -f` sin `-v` deja huérfano el volumen anónimo del contenedor.

La imagen de MariaDB declara `VOLUME /var/lib/mysql`. Un contenedor arrancado con
`--rm` y borrado con `docker rm -f` (sin `-v`) deja su datadir como volumen anónimo:
el 2026-10-04 había 341 así en hall9000 (56 GB, 22 con copias de `jax_memory`).
Medido ese día: `rm -f` → +1 volumen huérfano; `rm -fv` → 0.
"""
import re
import subprocess
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
# docker rm con -f pero sin v entre sus banderas cortas, en shell o como lista de Python.
SHELL = re.compile(r"docker\s+rm\s+(?:-(?!-)[a-uw-z]*f[a-uw-z]*)(?:\s|$)")
PYTHON = re.compile(r"""["']rm["']\s*,\s*["']-[a-uw-z]*f[a-uw-z]*["']""")
DOCKER_EN_LISTA = re.compile(r"docker")


def _archivos():
    salida = subprocess.run(["git", "ls-files", "*.sh", "*.py", "*.bash"], cwd=RAIZ,
                            capture_output=True, text=True, check=True).stdout.split()
    return [RAIZ / f for f in salida if f != "tests/test_docker_rm_sin_fuga_de_volumenes.py"]


def test_ningun_docker_rm_force_sin_borrar_volumenes():
    culpables = []
    for ruta in _archivos():
        try:
            lineas = ruta.read_text(encoding="utf-8").splitlines()
        except (UnicodeDecodeError, FileNotFoundError):
            continue
        for n, linea in enumerate(lineas, 1):
            if SHELL.search(linea) or (PYTHON.search(linea) and DOCKER_EN_LISTA.search(linea)):
                culpables.append(f"{ruta.relative_to(RAIZ)}:{n}: {linea.strip()}")
    assert not culpables, "docker rm -f sin -v (deja el volumen anónimo):\n" + "\n".join(culpables)


def test_el_patron_detecta_y_deja_pasar_lo_correcto():
    assert SHELL.search('sudo docker rm -f "$X"')
    assert not SHELL.search('sudo docker rm -fv "$X"')
    assert not SHELL.search('sudo docker rm -vf "$X"')
    assert PYTHON.search('[*docker, "rm", "-f", nombre]')
    assert not PYTHON.search('[*docker, "rm", "-fv", nombre]')
