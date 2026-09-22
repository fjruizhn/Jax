#!/usr/bin/env bash
# ops/ejecutor/instalar_vigia.sh — plan 6 de SP1 en hall9000: la unidad del vigía por misión.
# Corre como fruiz desde el checkout de producción (master); sudo para lo de root. Idempotente.
# NO arranca ninguna misión: la unidad es una plantilla (ejecutor-vigia@<id>.service) y cada
# instancia exige los seis contratos antes de latir. Reversión:
#   sudo rm /etc/systemd/system/ejecutor-vigia@.service && sudo systemctl daemon-reload
set -euo pipefail
: "${JAX_EJECUTOR_MISIONES:?}" "${JAX_EJECUTOR_VIGIA_LATIDO:?}" "${JAX_EJECUTOR_VIGIA_LATIDO_MAX_S:?}"
: "${JAX_EJECUTOR_VIGIA_LATIDO_CADA_S:?}" "${JAX_EJECUTOR_PAUSA:?}" "${JAX_EJECUTOR_LLAVES_ROOT:?}"
: "${JAX_EJECUTOR_ADMIN_USUARIO:?}" "${JAX_EJECUTOR_CUENTA:?}"
REPO="$(git -C "$(dirname "$(readlink -f "$0")")" rev-parse --show-toplevel)"
test "$(git -C "$REPO" branch --show-current)" = master
python3 -c "import sys; c, m = float(sys.argv[1]), float(sys.argv[2]); sys.exit(0 if 0 < c < m else 1)" \
  "$JAX_EJECUTOR_VIGIA_LATIDO_CADA_S" "$JAX_EJECUTOR_VIGIA_LATIDO_MAX_S" \
  || { echo "codigo=latido_cada_invalido" >&2; exit 1; }
# Misiones: las escribe fruiz (el transporte de SP2) y también las mide (M-1, ronda 4: la
# huella la toma el controlador COMO FRUIZ). B-2 (ronda 4): ACL EXPLÍCITA, no sólo el modo
# -- ver ops/ejecutor/preparar_directorio_misiones.sh para el porqué completo. `axioma`
# recibe SÓLO travesía (--x, sin +r): puede entrar a `<mision_id>/claude-projects` si ya
# sabe el UUID exacto (se lo da la propia misión que está corriendo), pero no puede LISTAR
# este directorio ni leer `<id>.json` de ninguna misión. Cada subcarpeta la crea
# `cuenta_axioma.preparar_directorio_projects` con `sudo install -d -o axioma -g axioma -m
# 0700`, así que ni con el --x de acá alcanza para tocar la de OTRA misión sin adivinar su
# UUID Y que ese UUID sea, además, dueño=axioma -- que lo es siempre, así que el límite real
# es "conocer el UUID de la misión en curso", el mismo que ya protege todo lo demás del
# Ejecutor (machine-id, políticas, etc.).
"$REPO/ops/ejecutor/preparar_directorio_misiones.sh" "$JAX_EJECUTOR_MISIONES" "$JAX_EJECUTOR_ADMIN_USUARIO" \
  "$JAX_EJECUTOR_CUENTA"
sudo install -o root -g root -m 0644 "$REPO/ops/ejecutor/ejecutor-vigia@.service" /etc/systemd/system/
sudo systemctl daemon-reload
systemctl cat ejecutor-vigia@verificacion.service >/dev/null
echo "vigia_instalado=true misiones=\"$JAX_EJECUTOR_MISIONES\" commit=\"$(git -C "$REPO" rev-parse --short HEAD)\""
