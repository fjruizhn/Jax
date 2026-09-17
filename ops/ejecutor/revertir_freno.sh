#!/usr/bin/env bash
# ops/ejecutor/revertir_freno.sh — deshace instalar_freno.sh en hall9000.
# Quita la unidad. Deja la llave del freno (sus públicas pueden estar en remotas: se quitan con
# revertir_en_maquina.sh) y cron.deny/at.deny/linger (son también de C6). La biblioteca queda:
# la usa el gancho de C1.
set -euo pipefail
: "${JAX_EJECUTOR_FRENO_ESTADO:?}"
sudo systemctl disable --now ejecutor-freno.service 2>/dev/null || true
sudo rm -f /etc/systemd/system/ejecutor-freno.service
sudo systemctl daemon-reload
test "$(systemctl is-active ejecutor-freno.service 2>/dev/null || true)" != active
echo "freno_revertido=true"
