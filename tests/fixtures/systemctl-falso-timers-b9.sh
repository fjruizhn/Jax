#!/usr/bin/env bash
# Doble hermético para tests/test_activacion_timers_b9.py; nunca delega al
# systemctl real. MODO=correcto|disabled_inactive|sin_next|falla.
set -euo pipefail
modo="${SYSTEMCTL_FALSO_MODO:-correcto}"
[ "${1:-}" = show ] || exit 2
timer="${*: -1}"
case "$timer" in jax-memory-embedding.timer|jax-memory-lifecycle.timer|jax-memory-synthesis.timer|jax-memory-vector-health.timer|jax-memory-worker.timer) ;; *) exit 2;; esac
[ "$modo" != falla ] || { echo "unidad no encontrada" >&2; exit 1; }
if [ "$modo" = disabled_inactive ]; then
  printf 'UnitFileState=disabled\nActiveState=inactive\nSubState=dead\nNextElapseUSecRealtime=n/a\n'
elif [ "$modo" = sin_next ]; then
  printf 'UnitFileState=enabled\nActiveState=active\nSubState=waiting\nNextElapseUSecRealtime=\n'
else
  printf 'UnitFileState=enabled\nActiveState=active\nSubState=waiting\nNextElapseUSecRealtime=Sun 2026-09-27 12:00:00 CST\n'
fi
