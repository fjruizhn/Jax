#!/usr/bin/env bash
# ops/ejecutor/revertir_huella_en_maquina.sh <nombre> — deshace
# instalar_huella_en_maquina.sh en UNA máquina remota, en orden inverso. NUNCA borra la
# llave del servicio ni su known_hosts (son compartidos por TODO el inventario -- los
# borra a mano quien de verdad quiera dar de baja el mecanismo entero, no este guion).
# Se detiene ante el primer fallo; cada paso es idempotente.
#
# MAJOR-4 (ronda 2, auditoría adversarial 2026-09-22): "revertido" no se declara por
# haber CORRIDO los tres pasos -- se VERIFICA que ninguno de los tres sigue ahí (el
# script, el sudoers, la línea en authorized_keys) y se sale ≠0 si algo quedó. Y NO se
# deja ninguna copia de `authorized_keys` con la llave adentro (el paso 3 antes hacía
# un `cp ... .antes-de-revertir-huella` -- ESE archivo quedaba en la remota con la
# línea completa, command= y clave pública incluidos; se quitó).
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
#    (grep -vF por la clave, no por la línea entera -- from= pudo cambiar). Filtrado
#    y reemplazo EN UN SOLO comando remoto -- sin backup con la llave adentro.
if corre "test -f ~$ADMIN_LOCAL/.ssh/authorized_keys"; then
  corre "{ grep -vF $(printf %q "$CLAVE") ~$ADMIN_LOCAL/.ssh/authorized_keys > /tmp/ejecutor-authorized-sin-huella || true; }; \
    install -o $ADMIN_LOCAL -g $ADMIN_LOCAL -m 0600 /tmp/ejecutor-authorized-sin-huella ~$ADMIN_LOCAL/.ssh/authorized_keys; \
    rm -f /tmp/ejecutor-authorized-sin-huella"
fi

# 2. sudoers.d acotado.
corre "rm -f /etc/sudoers.d/50-ejecutor-huella && visudo -c >/dev/null"

# 1. El script.
corre "rm -f /usr/local/sbin/ejecutor-huella"

# MAJOR-4: verificar, no declarar. Un solo chequeo remoto -- si algo de los tres sigue
# ahí, esto sale ≠0 (el `&&` corta la cadena) y NUNCA se imprime "revertida".
corre "test ! -e /usr/local/sbin/ejecutor-huella \
  && test ! -e /etc/sudoers.d/50-ejecutor-huella \
  && { ! grep -qF $(printf %q "$CLAVE") ~$ADMIN_LOCAL/.ssh/authorized_keys 2>/dev/null; }"

echo "maquina_huella_revertida=\"$NOMBRE\" verificado=true"
