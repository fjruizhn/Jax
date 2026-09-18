# tests/test_env_se_lee_con_sudo.py
"""`/etc/jax/.env` es `root:jaxsvc 640` desde el 2026-09-17: el operador NO lo lee directo.
Todo comando, guion o documento que diga cómo cargarlo tiene que hacerlo con `sudo -n cat`
(no interactivo a propósito: si el sudo pide contraseña, una corrida de tests se colgaría en vez
de fallar con un error legible). Este control lee el árbol: un `. /etc/jax/.env` que vuelva a
colarse en una instrucción o en un guion sale en rojo acá y no en medio de un despliegue.
"""
from __future__ import annotations

import re
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
# El archivo puede nombrarse (rutas, comentarios); lo que no puede es SOURCEARSE sin sudo.
SIN_SUDO = re.compile(r"(?:^|[;&|(`]\s*|\bset -a;?\s*&?&?\s*)(?:\.|source)\s+/etc/jax/\.env")
# Los planes de `docs/superpowers/plans/` son ACTAS fechadas: dicen lo que se corrió ese día, con
# el modo que el archivo tenía entonces. Reescribirlos sería falsear la historia; el control mira
# lo vivo (código, ops, scripts, tests, CONTEXT y DEUDA).
HISTORIA = ("docs/superpowers/plans/",)


# Sin `git grep`: en el runner el checkout puede no ser un repo utilizable por este proceso y el
# control se volvería verde sin mirar nada. Se camina el árbol y se saltan los directorios que no
# son fuente (.git, venvs, node_modules, caches).
SALTAR = {".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache", "dist", "build"}
EXTENSIONES = {".py", ".sh", ".md", ".service", ".conf", ".yml", ".yaml", ".txt", ".json", ""}


def _archivos() -> list[Path]:
    encontrados = []
    for p in RAIZ.rglob("*"):
        if not p.is_file() or any(parte in SALTAR for parte in p.parts):
            continue
        rel = str(p.relative_to(RAIZ))
        if rel.startswith(HISTORIA) or p.suffix not in EXTENSIONES:
            continue
        try:
            if "/etc/jax/.env" in p.read_text(errors="replace"):
                encontrados.append(p)
        except OSError:  # fail-soft: un archivo ilegible no es una instrucción; el control sigue con el resto
            continue
    return encontrados


def test_hay_archivos_que_nombran_el_env():
    # Si el grep no encuentra nada, el test de abajo pasa sin mirar nada.
    assert len(_archivos()) >= 5


def test_ningun_archivo_sourcea_el_env_sin_sudo():
    culpables = []
    for p in _archivos():
        if p.name == Path(__file__).name:
            continue
        for n, linea in enumerate(p.read_text(errors="replace").splitlines(), 1):
            if SIN_SUDO.search(linea):
                culpables.append(f"{p.relative_to(RAIZ)}:{n}: {linea.strip()[:90]}")
    assert culpables == []


def test_el_control_ve_la_forma_prohibida(tmp_path):
    """Un control que no falla no valida: la misma expresión, contra la línea vieja."""
    assert SIN_SUDO.search("set -a; . /etc/jax/.env; set +a")
    assert SIN_SUDO.search("set -a; source /etc/jax/.env; set +a")
    assert SIN_SUDO.search("cd /home/fruiz/jax && set -a && . /etc/jax/.env")
    assert not SIN_SUDO.search("set -a; . <(sudo -n cat /etc/jax/.env); set +a")
    assert not SIN_SUDO.search("# el archivo vive en /etc/jax/.env")
