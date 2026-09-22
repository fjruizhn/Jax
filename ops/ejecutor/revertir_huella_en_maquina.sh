#!/usr/bin/env bash
# ops/ejecutor/revertir_huella_en_maquina.sh <nombre> — deshace
# instalar_huella_en_maquina.sh en UNA máquina remota, en orden inverso. NUNCA borra la
# llave del servicio ni su known_hosts (son compartidos por TODO el inventario -- los
# borra a mano quien de verdad quiera dar de baja el mecanismo entero, no este guion).
# Se detiene ante el primer fallo; cada paso es idempotente.
set -euo pipefail
NOMBRE="${1:?uso: revertir_huella_en_maquina.sh <nombre>}"
: "${JAX_EJECUTOR_POLITICA:?}" "${JAX_EJECUTOR_ADMIN_USUARIO:?}" "${JAX_EJECUTOR_HUELLA_LLAVE:?}"
REPO="$(git -C "$(dirname "$(readlink -f "$0")")" rev-parse --show-toplevel)"
JAX_EJECUTOR_CUENTA="${JAX_EJECUTOR_CUENTA:-axioma}"
export JAX_EJECUTOR_CUENTA
. "$REPO/ops/ejecutor/_maquina.sh"

if [ "$LOCAL" = si ]; then
  echo "maquina_local_sin_remoto=\"$NOMBRE\""
  exit 0
fi

PUB="$(sudo cat "$JAX_EJECUTOR_HUELLA_LLAVE.pub")"
CLAVE="$(awk '{print $2}' <<<"$PUB")"

# 3. authorized_keys del administrador: quita SÓLO la línea con esta clave pública
#    (grep -vF por la clave, no por la línea entera -- from= pudo cambiar).
corre "test -f ~$ADMIN_LOCAL/.ssh/authorized_keys" || { echo "maquina_huella_revertida=\"$NOMBRE\" codigo=sin_authorized_keys"; exit 0; }
corre "cp ~$ADMIN_LOCAL/.ssh/authorized_keys ~$ADMIN_LOCAL/.ssh/authorized_keys.antes-de-revertir-huella \
  && { grep -vF $(printf %q "$CLAVE") ~$ADMIN_LOCAL/.ssh/authorized_keys.antes-de-revertir-huella > /tmp/ejecutor-authorized-sin-huella || true; } \
  && mv /tmp/ejecutor-authorized-sin-huella ~$ADMIN_LOCAL/.ssh/authorized_keys && chmod 0600 ~$ADMIN_LOCAL/.ssh/authorized_keys"

# 2. sudoers.d acotado.
corre "rm -f /etc/sudoers.d/50-ejecutor-huella && visudo -c >/dev/null"

# 1. El script.
corre "rm -f /usr/local/sbin/ejecutor-huella"

echo "maquina_huella_revertida=\"$NOMBRE\""
