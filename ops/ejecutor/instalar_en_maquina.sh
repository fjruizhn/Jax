#!/usr/bin/env bash
# ops/ejecutor/instalar_en_maquina.sh <nombre> [--sin-freno]
# C6 (y la parte remota de C3/C4) en UNA máquina del inventario. Corre como fruiz en hall9000.
# Remoto: ssh de administrador + sudo -n. Local (hall9000): sudo directo.
# Ante el primer fallo se detiene (set -e). La reversión es ops/ejecutor/revertir_en_maquina.sh <nombre>.
# `--sin-freno`: la llave y el script del freno remoto son del plan 3 (C4); sin ellos hay que decirlo a propósito.
#
# Variables: JAX_EJECUTOR_POLITICA, _CUENTA, _ADMIN_USUARIO, _LLAVES_ROOT, _CONTROLADOR_LLAVE;
#            con freno además _FRENO_LLAVE y _FRENO_KNOWN_HOSTS.
set -euo pipefail
NOMBRE="${1:?uso: instalar_en_maquina.sh <nombre> [--sin-freno]}"
SIN_FRENO="${2:-}"
: "${JAX_EJECUTOR_POLITICA:?}" "${JAX_EJECUTOR_CUENTA:?}" "${JAX_EJECUTOR_ADMIN_USUARIO:?}" "${JAX_EJECUTOR_LLAVES_ROOT:?}"
: "${JAX_EJECUTOR_CONTROLADOR_LLAVE:?}"
REPO="$(git -C "$(dirname "$(readlink -f "$0")")" rev-parse --show-toplevel)"
if [ "$SIN_FRENO" != --sin-freno ]; then
  : "${JAX_EJECUTOR_FRENO_LLAVE:?}" "${JAX_EJECUTOR_FRENO_KNOWN_HOSTS:?}"
  test -f "$REPO/ops/ejecutor/ejecutor-freno-remoto"
fi
. "$REPO/ops/ejecutor/_maquina.sh"
C="$JAX_EJECUTOR_CUENTA"
DIR_LLAVES="$(dirname "$JAX_EJECUTOR_LLAVES_ROOT")"
DROPIN=/etc/ssh/sshd_config.d/50-ejecutor-axioma.conf
SUDOERS=/etc/sudoers.d/50-ejecutor-axioma-registro
ETAPA="$(mktemp -d)"; trap 'rm -rf "$ETAPA"' EXIT

# 0. Precondiciones: sudo sin contraseña, la cuenta existe, sin procesos vivos, nada instalado a medias.
corre "true"
corre "id $C >/dev/null"
test "$(corre "ps -u $C -o pid= | wc -l" | tr -d ' ')" = 0
UNIDAD="$(corre "systemctl list-unit-files ssh.service sshd.service --no-legend" | awk 'NR==1{print $1}')"
test -n "$UNIDAD"
for u in root "$ADMIN_LOCAL" "$C"; do
  corre "sshd -T -C user=$u,host=x,addr=127.0.0.1" | sort > "$ETAPA/sshd-$u-antes.txt"
done

# 1. Scripts root.
sube "$REPO/ops/ejecutor/ejecutor-revocar" /usr/local/sbin/ejecutor-revocar 0755
[ "$SIN_FRENO" = --sin-freno ] || sube "$REPO/ops/ejecutor/ejecutor-freno-remoto" /usr/local/sbin/ejecutor-freno-remoto 0755

# 2. Archivo root de llaves: las actuales de la cuenta, marcadas; más la del freno.
corre "cat ~$C/.ssh/authorized_keys" > "$ETAPA/actuales"
FRENO_PUB=""
[ "$SIN_FRENO" = --sin-freno ] || FRENO_PUB="$(sudo -n cat "$JAX_EJECUTOR_FRENO_LLAVE.pub")"
CTRL_PUB=""
[ "$LOCAL" = no ] || CTRL_PUB="$(cat "$JAX_EJECUTOR_CONTROLADOR_LLAVE.pub")"
( cd "$REPO" && PYTHONDONTWRITEBYTECODE=1 python3 -c '
import sys
from jax.ejecutor.contratos.revocacion import llaves_root
sys.stdout.write(llaves_root(open(sys.argv[1]).read(), controlador_pub=sys.argv[2] or None, freno_pub=sys.argv[3] or None))
' "$ETAPA/actuales" "$CTRL_PUB" "$FRENO_PUB" ) > "$ETAPA/llaves"
corre "install -d -o root -g root -m 0755 $DIR_LLAVES"
# Si ya hay un archivo root, sólo se acepta idéntico: pisarlo después de una revocación
# devolvería el acceso con las llaves del home de la cuenta.
if corre "test -e $JAX_EJECUTOR_LLAVES_ROOT"; then
  corre "cat $JAX_EJECUTOR_LLAVES_ROOT" | cmp - "$ETAPA/llaves"
else
  sube "$ETAPA/llaves" "$JAX_EJECUTOR_LLAVES_ROOT" 0644
fi

# 3. sshd: sólo para la cuenta; validar; comprobar que para root y el administrador no cambió NADA; recargar.
printf 'Match User %s\n    AuthorizedKeysFile %s/%%u\n' "$C" "$DIR_LLAVES" > "$ETAPA/50-ejecutor.conf"
sube "$ETAPA/50-ejecutor.conf" "$DROPIN" 0644
if ! corre "sshd -t"; then corre "rm -f $DROPIN"; echo "codigo=sshd_t_fallo" >&2; exit 1; fi
for u in root "$ADMIN_LOCAL"; do
  corre "sshd -T -C user=$u,host=x,addr=127.0.0.1" | sort > "$ETAPA/sshd-$u-despues.txt"
  diff "$ETAPA/sshd-$u-antes.txt" "$ETAPA/sshd-$u-despues.txt" \
    || { corre "rm -f $DROPIN"; echo "codigo=sshd_cambio_para_$u" >&2; exit 1; }
done
corre "sshd -T -C user=$C,host=x,addr=127.0.0.1" | grep -qx "authorizedkeysfile $DIR_LLAVES/%u" \
  || { corre "rm -f $DROPIN"; echo "codigo=sshd_no_aplica_a_la_cuenta" >&2; exit 1; }
corre "systemctl reload $UNIDAD"

# 4. Registro de sudo de la cuenta (C3), validado ANTES de instalar.
printf 'Defaults:%s log_output, iolog_dir=/var/log/sudo-io/%%{user}, logfile=/var/log/sudo-%s.log\n' "$C" "$C" > "$ETAPA/sudoers"
sube "$ETAPA/sudoers" /tmp/ejecutor-sudoers-prueba 0440
corre "visudo -cf /tmp/ejecutor-sudoers-prueba && install -o root -g root -m 0440 /tmp/ejecutor-sudoers-prueba $SUDOERS && rm /tmp/ejecutor-sudoers-prueba && visudo -c >/dev/null"

# 5. cron, at, linger.
corre "for f in /etc/cron.deny /etc/at.deny; do touch \$f; grep -qx $C \$f || echo $C >> \$f; done; test ! -e /etc/cron.allow; test ! -e /etc/at.allow; loginctl disable-linger $C"

# 6. known_hosts del freno (en hall9000), desde el known_hosts de fruiz ya confiado.
if [ "$LOCAL" = no ] && [ "$SIN_FRENO" != --sin-freno ]; then
  ENTRADA="$(ssh-keygen -F "[$IP]:$PUERTO" -f ~/.ssh/known_hosts | grep -v '^#')"
  test -n "$ENTRADA"
  sudo -n grep -qxF "$ENTRADA" "$JAX_EJECUTOR_FRENO_KNOWN_HOSTS" || echo "$ENTRADA" | sudo -n tee -a "$JAX_EJECUTOR_FRENO_KNOWN_HOSTS" >/dev/null
fi
echo "maquina_instalada=\"$NOMBRE\" llaves_sha256=\"$(corre "sha256sum $JAX_EJECUTOR_LLAVES_ROOT" | cut -d' ' -f1)\" freno=$([ "$SIN_FRENO" = --sin-freno ] && echo false || echo true)"
