#!/usr/bin/env bash
# ops/ejecutor/instalar_huella_en_maquina.sh <nombre>
#
# Arreglo del bug de producción jax#260 (2026-09-22): `jax-platform.service` corre como
# `jaxsvc` desde el 2026-09-17 (cuenta de servicio), no como `fruiz` -- y `jaxsvc` no
# puede leer la llave personal de `fruiz`, que es lo que el camino viejo de la huella
# necesitaba (`vigia_no_latio=true rc=2`, medido en producción). Este instalador da de
# alta, en UNA máquina remota del inventario, el camino nuevo: una llave PROPIA del
# servicio, autorizada con comando forzado hacia `ejecutor-huella` -- mismo patrón que
# ya usa C4 (`ejecutor-freno-remoto`, `ops/ejecutor/instalar_freno.sh` +
# `ops/ejecutor/instalar_en_maquina.sh`).
#
# Corre como fruiz en hall9000, con sudo (para lo local: la llave del servicio, que
# vive donde `jaxsvc` ya puede leer) y ssh de administrador + sudo -n (para lo remoto).
# Idempotente. Reversión: ops/ejecutor/revertir_huella_en_maquina.sh <nombre>.
#
# `hall9000` (la máquina LOCAL) no se instala: `hosts_con_sudo()` en
# `jax/ejecutor/contratos/vigia_servicio.py` sólo mide las REMOTAS -- hall9000 nunca es
# el destino de este ssh. Este guion, invocado con `hall9000` (o cualquier host
# `es_local`), sólo hace el paso 1 (la llave local) y termina ahí.
#
# Variables: JAX_EJECUTOR_POLITICA, _ADMIN_USUARIO, _HUELLA_LLAVE, _HUELLA_KNOWN_HOSTS,
#            _HUELLA_ORIGEN_IP (la IP de hall9000 vista desde la remota -- sin
#            hardcoding, Principio IV: cada máquina del ecosistema declara la suya).
set -euo pipefail
NOMBRE="${1:?uso: instalar_huella_en_maquina.sh <nombre>}"
: "${JAX_EJECUTOR_POLITICA:?}" "${JAX_EJECUTOR_ADMIN_USUARIO:?}" "${JAX_EJECUTOR_HUELLA_LLAVE:?}"
: "${JAX_EJECUTOR_HUELLA_KNOWN_HOSTS:?}" "${JAX_EJECUTOR_HUELLA_ORIGEN_IP:?}"
REPO="$(git -C "$(dirname "$(readlink -f "$0")")" rev-parse --show-toplevel)"
test -f "$REPO/ops/ejecutor/ejecutor-huella"
# Necesita JAX_EJECUTOR_CUENTA sólo porque _maquina.sh está pensado para instalar cosas
# DE la cuenta del Ejecutor -- acá no se usa para nada más que resolver IP/PUERTO/LOCAL.
JAX_EJECUTOR_CUENTA="${JAX_EJECUTOR_CUENTA:-axioma}"
export JAX_EJECUTOR_CUENTA
. "$REPO/ops/ejecutor/_maquina.sh"
ETAPA="$(mktemp -d)"; trap 'rm -rf "$ETAPA"' EXIT

# 1. La llave PROPIA del servicio, UNA sola vez para todo el inventario (idempotente:
#    si ya existe, se reusa tal cual -- no se regenera). jaxsvc:jaxsvc 700/600: el
#    mismo dueño que JAX_EJECUTOR_CONTROLADOR_LLAVE, un PAR de llaves DISTINTO.
DIR_LLAVE="$(dirname "$JAX_EJECUTOR_HUELLA_LLAVE")"
sudo install -d -o jaxsvc -g jaxsvc -m 0700 "$DIR_LLAVE"
if ! sudo test -e "$JAX_EJECUTOR_HUELLA_LLAVE"; then
  sudo ssh-keygen -q -t ed25519 -N '' -C ejecutor-huella-servicio -f "$JAX_EJECUTOR_HUELLA_LLAVE"
  sudo chown jaxsvc:jaxsvc "$JAX_EJECUTOR_HUELLA_LLAVE" "$JAX_EJECUTOR_HUELLA_LLAVE.pub"
fi
sudo test -e "$JAX_EJECUTOR_HUELLA_KNOWN_HOSTS" \
  || sudo install -o jaxsvc -g jaxsvc -m 0644 /dev/null "$JAX_EJECUTOR_HUELLA_KNOWN_HOSTS"
PUB="$(sudo cat "$JAX_EJECUTOR_HUELLA_LLAVE.pub")"
TIPO="$(awk '{print $1}' <<<"$PUB")"
CLAVE="$(awk '{print $2}' <<<"$PUB")"

if [ "$LOCAL" = si ]; then
  echo "maquina_local_sin_remoto=\"$NOMBRE\" llave_lista=true"
  exit 0
fi

# 2. known_hosts del servicio (jaxsvc), desde el known_hosts de fruiz ya confiado --
#    mismo criterio que el paso 6 de instalar_en_maquina.sh para el freno.
ssh-keygen -F "[$IP]:$PUERTO" -f ~/.ssh/known_hosts | grep -v '^#' > "$ETAPA/kh-entradas"
test -s "$ETAPA/kh-entradas"
while IFS= read -r LINEA_KH; do
  sudo grep -qxF -- "$LINEA_KH" "$JAX_EJECUTOR_HUELLA_KNOWN_HOSTS" \
    || printf '%s\n' "$LINEA_KH" | sudo tee -a "$JAX_EJECUTOR_HUELLA_KNOWN_HOSTS" >/dev/null
done < "$ETAPA/kh-entradas"

# 3. El script en la remota.
sube "$REPO/ops/ejecutor/ejecutor-huella" /usr/local/sbin/ejecutor-huella 0755

# 4. sudoers.d ACOTADO -- exactamente este binario, SIN argumentos, nunca ALL. MAJOR-5
#    (ronda 2, auditoría adversarial 2026-09-22): la cadena vacía `""` NO es decorativa
#    -- verificado en un contenedor Ubuntu 24.04 limpio (sudo 1.9.17p2, 2026-09-22) que
#    SIN ella sudo acepta CUALQUIER argumento («sudo -n cmd hostil» → rc=0) aunque el
#    binario no los use; sólo con `cmd ""` sudo exige que la invocación no tenga
#    argumentos. Sin esto, "sin argumentos: no hay superficie de ataque" sería una
#    afirmación falsa sobre el propio sudoers -- ver huella.py, _COMANDO_FORZADO_HUELLA.
printf '%s ALL=(root) NOPASSWD: /usr/local/sbin/ejecutor-huella ""\n' "$ADMIN_LOCAL" > "$ETAPA/sudoers-huella"
sube "$ETAPA/sudoers-huella" /tmp/ejecutor-sudoers-huella-prueba 0440
corre "visudo -cf /tmp/ejecutor-sudoers-huella-prueba && install -o root -g root -m 0440 /tmp/ejecutor-sudoers-huella-prueba /etc/sudoers.d/50-ejecutor-huella && rm /tmp/ejecutor-sudoers-huella-prueba && visudo -c >/dev/null"

# 5. authorized_keys del ADMINISTRADOR en la remota -- BLOCK-2/MAJOR-3 (ronda 2): la
#    línea la arma `huella.linea_authorized_keys_servicio` (Python, TESTEADO, no bash
#    suelto), y `huella.actualizar_authorized_keys_admin` CONVERGE: si ya hay una marca
#    `ejecutor-huella-servicio`, la REEMPLAZA (por ejemplo si `origen_ip` cambió) --
#    nunca la duplica. MINOR (ronda 2): `install -o/-g "$ADMIN_LOCAL"` en el directorio
#    Y el archivo -- antes, si `~$ADMIN_LOCAL/.ssh` no existía, `corre` (que ejecuta como
#    ROOT vía sudo -n) lo creaba dueño ROOT, dejando al administrador sin poder tocar su
#    propio `authorized_keys` nunca más.
corre "install -d -o $ADMIN_LOCAL -g $ADMIN_LOCAL -m 0700 ~$ADMIN_LOCAL/.ssh"
ACTUALES="$(corre "cat ~$ADMIN_LOCAL/.ssh/authorized_keys 2>/dev/null" || true)"
printf '%s' "$ACTUALES" > "$ETAPA/actuales"
( cd "$REPO" && PYTHONDONTWRITEBYTECODE=1 python3 -c '
import sys
from jax.ejecutor.contratos.huella import actualizar_authorized_keys_admin
actuales = open(sys.argv[1], encoding="utf-8").read()
sys.stdout.write(actualizar_authorized_keys_admin(
    actuales, tipo=sys.argv[2], clave=sys.argv[3], origen_ip=sys.argv[4]))
' "$ETAPA/actuales" "$TIPO" "$CLAVE" "$JAX_EJECUTOR_HUELLA_ORIGEN_IP" ) > "$ETAPA/nuevas"
if corre "cat ~$ADMIN_LOCAL/.ssh/authorized_keys 2>/dev/null" | cmp -s - "$ETAPA/nuevas"; then
  echo "codigo=llave_huella_ya_convergida host=\"$NOMBRE\""
else
  sube "$ETAPA/nuevas" /tmp/ejecutor-authorized-huella 0600
  corre "install -o $ADMIN_LOCAL -g $ADMIN_LOCAL -m 0600 /tmp/ejecutor-authorized-huella ~$ADMIN_LOCAL/.ssh/authorized_keys && rm -f /tmp/ejecutor-authorized-huella"
fi

echo "maquina_huella_instalada=\"$NOMBRE\" script_sha256=\"$(corre "sha256sum /usr/local/sbin/ejecutor-huella" | cut -d' ' -f1)\""
