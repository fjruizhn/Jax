"""Ninguna ruta del freno escrita a mano (plan 2026-09-16-frente-b-kill-switch,
Task 5). Test de CLASE sobre los archivos de código versionados: la ruta vive
sólo en /etc/jax/.env (JAX_KILL_SWITCH_PATH). La documentación (.md) queda
afuera a propósito: CONTEXT.md §9 y DEUDA.md cuentan la historia con la
ruta vieja, y eso es historia, no configuración."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
LITERAL = "/etc/jax/" + "PAUSE"
SUFIJOS = {".py", ".toml", ".sh", ".js", ".service", ".yml", ".yaml"}
EXCLUIDOS = ("_director_patch/",)
# La ruta vieja SÓLO puede estar en el módulo interruptor, como compatibilidad
# (requisito del controlador principal del frente B, 2026-09-17, ruling R14):
# gente y scripts pausan creándola y tiene que seguir frenando. El canónico y
# su symlink de las_manos son el mismo archivo. Ninguno más: lo fija
# test_la_excepcion_de_la_ruta_vieja_no_se_amplia.
PERMITIDOS_RUTA_VIEJA = frozenset({"jax/core/interruptor.py", "las_manos/interruptor.py"})


def _archivos():
    salida = subprocess.run(["git", "ls-files"], cwd=RAIZ, capture_output=True, text=True, check=True).stdout
    for relativa in salida.splitlines():
        ruta = RAIZ / relativa
        if ruta.suffix in SUFIJOS and not relativa.startswith(EXCLUIDOS) and ".backup-" not in relativa and ruta.is_file():
            yield relativa, ruta


def test_el_escaneo_mira_algo():
    assert len(list(_archivos())) > 100


def test_ningun_archivo_de_codigo_nombra_la_ruta_vieja():
    hallazgos = [r for r, ruta in _archivos()
                 if r not in PERMITIDOS_RUTA_VIEJA and LITERAL in ruta.read_text(encoding="utf-8", errors="replace")]
    assert hallazgos == []


def test_la_excepcion_de_la_ruta_vieja_no_se_amplia():
    assert PERMITIDOS_RUTA_VIEJA == {"jax/core/interruptor.py", "las_manos/interruptor.py"}
    assert (RAIZ / "las_manos" / "interruptor.py").resolve() == (RAIZ / "jax" / "core" / "interruptor.py").resolve()
    # Y la excepción se usa: la ruta vieja está en el módulo (si se fuera, la
    # compatibilidad se habría perdido y la excepción sobraría).
    con_la_ruta = {r for r, ruta in _archivos() if LITERAL in ruta.read_text(encoding="utf-8", errors="replace")}
    assert con_la_ruta == PERMITIDOS_RUTA_VIEJA


def test_ninguna_config_declara_la_ruta_del_freno():
    patron = re.compile(r"^\s*kill_switch_path\s*=", re.MULTILINE)
    hallazgos = [r for r, ruta in _archivos()
                 if ruta.suffix == ".toml" and patron.search(ruta.read_text(encoding="utf-8"))]
    assert hallazgos == []
