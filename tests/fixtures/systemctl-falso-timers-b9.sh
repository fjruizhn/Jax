#!/usr/bin/env bash
# Doble hermético para tests/test_activacion_timers_b9.py; nunca delega al
# systemctl real. MODO=correcto|correcto_monotonico|disabled_inactive|sin_next|muerto_real|falla.
#
# Emite SIEMPRE los cinco campos que pide el guion, como el systemd real (el orden de
# salida real no respeta el orden en que se piden). `muerto_real` es la salida MEDIDA en
# hall9000 (2026-10-06) para un timer deshabilitado: Realtime vacío y Monotonic=infinity.
set -euo pipefail
modo="${SYSTEMCTL_FALSO_MODO:-correcto}"
[ "${1:-}" = show ] || exit 2
timer="${*: -1}"
case "$timer" in jax-memory-embedding.timer|jax-memory-lifecycle.timer|jax-memory-synthesis.timer|jax-memory-vector-health.timer|jax-memory-worker.timer) ;; *) exit 2;; esac
[ "$modo" != falla ] || { echo "unidad no encontrada" >&2; exit 1; }
case "$modo" in
  disabled_inactive)
    printf 'UnitFileState=disabled\nActiveState=inactive\nSubState=dead\nNextElapseUSecRealtime=n/a\nNextElapseUSecMonotonic=n/a\n' ;;
  muerto_real)
    printf 'ActiveState=inactive\nSubState=dead\nUnitFileState=disabled\nNextElapseUSecRealtime=\nNextElapseUSecMonotonic=infinity\n' ;;
  sin_next)
    printf 'UnitFileState=enabled\nActiveState=active\nSubState=waiting\nNextElapseUSecRealtime=\nNextElapseUSecMonotonic=infinity\n' ;;
  correcto_monotonico)
    # Timer de OnUnitActiveSec/OnBootSec: sin Realtime, con un monotónico real.
    printf 'UnitFileState=enabled\nActiveState=active\nSubState=waiting\nNextElapseUSecRealtime=\nNextElapseUSecMonotonic=5min\n' ;;
  *)
    # Timer de calendario activo: Realtime con fecha y Monotonic=0.
    printf 'UnitFileState=enabled\nActiveState=active\nSubState=waiting\nNextElapseUSecRealtime=Sun 2026-09-27 12:00:00 CST\nNextElapseUSecMonotonic=0\n' ;;
esac
