#!/usr/bin/env bash
# ops/ejecutor/instalar_vigia.sh — plan 6 de SP1 en hall9000: la unidad del vigía por misión.
# Corre como fruiz desde el checkout de producción (master); sudo para lo de root. Idempotente.
# NO arranca ninguna misión: la unidad es una plantilla (ejecutor-vigia@<id>.service) y cada
# instancia exige los seis contratos antes de latir. Reversión:
#   sudo rm /etc/systemd/system/ejecutor-vigia@.service && sudo systemctl daemon-reload
set -euo pipefail
: "${JAX_EJECUTOR_MISIONES:?}" "${JAX_EJECUTOR_VIGIA_LATIDO:?}" "${JAX_EJECUTOR_VIGIA_LATIDO_MAX_S:?}"
: "${JAX_EJECUTOR_VIGIA_LATIDO_CADA_S:?}" "${JAX_EJECUTOR_PAUSA:?}" "${JAX_EJECUTOR_LLAVES_ROOT:?}"
REPO="$(git -C "$(dirname "$(readlink -f "$0")")" rev-parse --show-toplevel)"
test "$(git -C "$REPO" branch --show-current)" = master
python3 -c "import sys; c, m = float(sys.argv[1]), float(sys.argv[2]); sys.exit(0 if 0 < c < m else 1)" \
  "$JAX_EJECUTOR_VIGIA_LATIDO_CADA_S" "$JAX_EJECUTOR_VIGIA_LATIDO_MAX_S" \
  || { echo "codigo=latido_cada_invalido" >&2; exit 1; }
# Misiones: las escribe fruiz (el transporte de SP2); la cuenta del Ejecutor ni las lista.
sudo install -d -o jaxsvc -g jaxsvc -m 0750 "$JAX_EJECUTOR_MISIONES"
sudo install -o root -g root -m 0644 "$REPO/ops/ejecutor/ejecutor-vigia@.service" /etc/systemd/system/
sudo systemctl daemon-reload
systemctl cat ejecutor-vigia@verificacion.service >/dev/null
echo "vigia_instalado=true misiones=\"$JAX_EJECUTOR_MISIONES\" commit=\"$(git -C "$REPO" rev-parse --short HEAD)\""
