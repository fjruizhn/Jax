#!/usr/bin/env bash
# Instalador idempotente: env REPL + timer del worker + restart jax-platform.
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

echo "== 3) Reiniciar jax-platform (carga chat.py/main.py nuevos) =="
systemctl restart jax-platform
sleep 3
systemctl is-active jax-platform && echo "   jax-platform ACTIVO"

echo "LISTO."
