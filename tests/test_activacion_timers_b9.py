"""Contrato fail-closed de activación de los cinco timers B9.

No habilita ni inicia nada: un timer disabled/inactive es evidencia roja
explícita que requiere una decisión/operación humana fuera de esta prueba.
"""
from __future__ import annotations

import os
import shutil
import subprocess

import pytest

from manifiesto_arranque import ROOT, timers_b9

SCRIPT = ROOT / "ops" / "verificar-activacion-timers-b9.sh"
FALSO = ROOT / "tests" / "fixtures" / "systemctl-falso-timers-b9.sh"


def _correr(modo: str, tmp_path) -> subprocess.CompletedProcess[str]:
    doble = tmp_path / "bin" / "systemctl"
    doble.parent.mkdir()
    shutil.copy2(FALSO, doble)
    env = os.environ | {
        "JAX_TIMER_CONTRACT_SYSTEMCTL": str(doble),
        "SYSTEMCTL_FALSO_MODO": modo,
    }
    return subprocess.run([str(SCRIPT), str(tmp_path)], capture_output=True, text=True, env=env)


def test_el_manifiesto_declara_los_cinco_timers_b9():
    assert timers_b9() == [
        "jax-memory-embedding.timer", "jax-memory-lifecycle.timer",
        "jax-memory-synthesis.timer", "jax-memory-vector-health.timer",
        "jax-memory-worker.timer",
    ]


def test_estado_correcto_pasa_sin_mutar_nada(tmp_path):
    resultado = _correr("correcto", tmp_path)
    assert resultado.returncode == 0, resultado.stderr
    assert "5 timers B9" in resultado.stdout


def test_timer_monotonico_con_proximo_disparo_real_pasa(tmp_path):
    """Sin Realtime pero con un monotónico real (5min) SÍ hay próximo disparo."""
    resultado = _correr("correcto_monotonico", tmp_path)
    assert resultado.returncode == 0, resultado.stderr


def test_timer_running_mientras_corre_su_servicio_pasa(tmp_path):
    """m4 (ronda 3 de #356): un timer cuyo servicio esta corriendo en ese instante tiene
    SubState=running (systemd.timer: waiting | running | elapsed). Es un timer sano; exigir
    solo `waiting` daba un rojo falso justo cuando el worker esta trabajando. Se sigue
    exigiendo enabled, active y proximo disparo."""
    resultado = _correr("running_con_disparo", tmp_path)
    assert resultado.returncode == 0, resultado.stderr


@pytest.mark.parametrize("modo", [
    "disabled_inactive", "sin_next", "muerto_real", "falla",
    # m1 (ronda 3): enabled/active/waiting pero sin proximo disparo.
    "sin_next_monotonico_cero", "realtime_na",
])
def test_cualquier_estado_inactivo_o_incompleto_falla_cerrado(modo, tmp_path):
    resultado = _correr(modo, tmp_path)
    assert resultado.returncode != 0
    assert "TIMER B9 NO ACTIVADO" in resultado.stderr or "SYSTEMCTL SHOW FALLÓ" in resultado.stderr


def test_el_contrato_no_habilita_ni_arranca_timers(tmp_path):
    """El doble sólo admite `show`; una futura llamada enable/start/restart,
    incluso vía una variable o flags, devuelve error y vuelve rojo el caso."""
    resultado = _correr("correcto", tmp_path)
    assert resultado.returncode == 0, resultado.stderr
    contenido = SCRIPT.read_text(encoding="utf-8")
    codigo = "\n".join(line.split("#", 1)[0] for line in contenido.splitlines())
    assert '"$SYSTEMCTL_CMD" show -p UnitFileState -p ActiveState -p SubState -p NextElapseUSecRealtime -p NextElapseUSecMonotonic "$timer"' in codigo
    for verbo in ("enable", "start", "restart", "daemon-reload"):
        assert f'"$SYSTEMCTL_CMD" {verbo}' not in codigo


def test_el_doble_no_puede_escapar_de_raiz_prueba(tmp_path):
    env = os.environ | {"JAX_TIMER_CONTRACT_SYSTEMCTL": str(FALSO)}
    resultado = subprocess.run([str(SCRIPT), str(tmp_path)], capture_output=True, text=True, env=env)
    assert resultado.returncode != 0
    assert "fuera de RAIZ_PRUEBA" in resultado.stderr
