#!/usr/bin/env bash
# Installer helper only.  Production deployment owns enable/restart decisions;
# B9 does not restart platform services as a side effect of memory setup.
set -euo pipefail
ENVF=/etc/jax/.env
SD=/home/fruiz/jax/config/systemd

echo "== 1) JAX_REPL_* en $ENVF =="
if ! grep -q '^JAX_REPL_USER_ID=' "$ENVF"; then
  cat >> "$ENVF" <<'VARS'

# Identidad del REPL para scope de memoria de dos niveles (Fernando individual)
JAX_REPL_USER_ID=1
JAX_REPL_TENANT_ID=1
VARS
  echo "   (agregadas)"
else
  echo "   (ya existian — sin cambios)"
fi
grep -E 'JAX_REPL_USER_ID|JAX_REPL_TENANT_ID' "$ENVF"

echo "== 2) Unidades systemd del worker =="
install -m 644 "$SD/jax-memory-worker.service" /etc/systemd/system/
install -m 644 "$SD/jax-memory-worker.timer"  /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now jax-memory-worker.timer
systemctl list-timers jax-memory-worker.timer --no-pager || true

echo "LISTO."
