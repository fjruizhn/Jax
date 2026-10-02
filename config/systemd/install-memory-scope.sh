#!/usr/bin/env bash
# Installer helper only.  Production deployment owns enable/restart decisions;
# B9 does not restart platform services as a side effect of memory setup.
set -euo pipefail
SD=/home/fruiz/jax/config/systemd

echo "== 1) Unidades systemd del worker =="
install -m 644 "$SD/jax-memory-worker.service" /etc/systemd/system/
install -m 644 "$SD/jax-memory-worker.timer"  /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now jax-memory-worker.timer
systemctl list-timers jax-memory-worker.timer --no-pager || true

echo "LISTO."
