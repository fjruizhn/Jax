#!/usr/bin/env bash
# ops/ejecutor/revertir_en_maquina.sh <nombre> — deshace instalar_en_maquina.sh, en orden inverso.
# Después de revertir, sshd vuelve a leer ~cuenta/.ssh/authorized_keys (que la instalación NO tocó):
# la cuenta entra como antes. Se detiene ante el primer fallo; cada paso es idempotente.
set -euo pipefail
NOMBRE="${1:?uso: revertir_en_maquina.sh <nombre>}"
: "${JAX_EJECUTOR_POLITICA:?}" "${JAX_EJECUTOR_CUENTA:?}" "${JAX_EJECUTOR_ADMIN_USUARIO:?}" "${JAX_EJECUTOR_LLAVES_ROOT:?}"
REPO="$(git -C "$(dirname "$(readlink -f "$0")")" rev-parse --show-toplevel)"
. "$REPO/ops/ejecutor/_maquina.sh"
C="$JAX_EJECUTOR_CUENTA"
UNIDAD="$(corre "systemctl list-unit-files ssh.service sshd.service --no-legend" | awk 'NR==1{print $1}')"
test -n "$UNIDAD"

# 6. known_hosts del freno.
if [ "$LOCAL" = no ] && [ -n "${JAX_EJECUTOR_FRENO_KNOWN_HOSTS:-}" ] && sudo -n test -f "$JAX_EJECUTOR_FRENO_KNOWN_HOSTS"; then
  # Varias líneas por máquina (tipos de llave, hasheadas con `|`): se quitan por igualdad exacta
  # (grep -vxF), no con una regex armada a mano, que se rompía con más de una línea.
  ENTRADAS="$(mktemp)"
  ssh-keygen -F "[$IP]:$PUERTO" -f ~/.ssh/known_hosts | grep -v '^#' > "$ENTRADAS" || true
  if [ -s "$ENTRADAS" ]; then
    RC=0; sudo -n grep -vxFf "$ENTRADAS" "$JAX_EJECUTOR_FRENO_KNOWN_HOSTS" > "$ENTRADAS.quedan" || RC=$?
    [ "$RC" -le 1 ] || { rm -f "$ENTRADAS" "$ENTRADAS.quedan"; echo "codigo=known_hosts_ilegible" >&2; exit 1; }
    sudo -n sh -c "cat '$ENTRADAS.quedan' > '$JAX_EJECUTOR_FRENO_KNOWN_HOSTS'"
  fi
  rm -f "$ENTRADAS" "$ENTRADAS.quedan"
fi
# 5. cron, at (linger queda deshabilitado: era el estado previo verificado en la instalación).
corre "for f in /etc/cron.deny /etc/at.deny; do [ ! -f \$f ] || sed -i '/^$C\$/d' \$f; done"
# 4. sudoers.
corre "rm -f /etc/sudoers.d/50-ejecutor-axioma-registro && visudo -c >/dev/null"
# 3. sshd: quitar el drop-in, validar, recargar.
corre "rm -f /etc/ssh/sshd_config.d/50-ejecutor-axioma.conf && sshd -t && systemctl reload $UNIDAD"
esperar_sshd
# Sin la cuenta en un Match propio vuelve a leer su home (a archivo: con pipefail, grep -q + SIGPIPE da falso fallo).
SSHD_T="$(mktemp)"; corre "sshd -T -C user=$C,host=x,addr=127.0.0.1" > "$SSHD_T"
if grep -qi "^authorizedkeysfile .*authorized_keys.d" "$SSHD_T"; then rm -f "$SSHD_T"; echo "codigo=sshd_sigue_con_llaves_root" >&2; exit 1; fi
rm -f "$SSHD_T"
# 2. Archivo root de llaves: se guarda aparte, no se borra (es evidencia de qué llaves tenía).
corre "[ ! -e $JAX_EJECUTOR_LLAVES_ROOT ] || mv $JAX_EJECUTOR_LLAVES_ROOT $JAX_EJECUTOR_LLAVES_ROOT.revertido-\$(date -u +%Y%m%dT%H%M%SZ)"
# 1. Scripts root.
corre "rm -f /usr/local/sbin/ejecutor-revocar /usr/local/sbin/ejecutor-freno-remoto"
echo "maquina_revertida=\"$NOMBRE\""
