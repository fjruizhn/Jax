"""La config no declara la ruta del freno (plan 2026-09-16-frente-b-kill-switch, Task 5).

La ruta sale de la variable JAX_KILL_SWITCH_PATH (jax/core/interruptor.py), nunca de
config/config.toml: antes salia del TOML y se miraba una sola vez antes de invocar.

T16 (2026-10-02): el REPL y `jax --task` se retiraron, y con ellos las pruebas de
que cada invoke de jax/core/main.py corria bajo el interruptor. Queda esta, que
vigila la config y no al REPL; el nombre del archivo es historico.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import tomllib
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]


def test_la_config_no_declara_la_ruta_del_freno():
    with open(RAIZ / "config" / "config.toml", "rb") as f:
        assert "kill_switch_path" not in tomllib.load(f)["jax"]
