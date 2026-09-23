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
# ADVERTENCIA (ronda 3, MINOR): instalar esto CAMBIA la huella de la máquina (agrega el
# script, el sudoers.d y la línea de authorized_keys -- todos parte de RUTAS_CONTROLES o
# medidos por la huella). Si hay una MISIÓN EN VUELO sobre esa máquina en ese momento, el
# PRÓXIMO cierre de turno va a ver ese cambio como `huella_cambio_no_declarado` y PAUSA
# el Ejecutor -- hay que aceptar esa huella después (`docs/ejecutor-huella-aceptar.md`).
# La forma correcta es instalar con el Ejecutor SIN misiones activas contra esa máquina.
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
# MINOR (ronda 5, auditoría adversarial 2026-09-22): `$ADMIN_LOCAL` viajaba SIN escapar
# dentro de los `corre "..."` remotos del paso 6 -- lo mismo que MAJOR-D (ronda 4) ya
# había corregido para los temporales. `ADMIN_Q` es la forma escapada, para usar en
# CUALQUIER `corre "..."`.
ADMIN_Q="$(printf %q "$ADMIN_LOCAL")"
ETAPA="$(mktemp -d)"; trap 'rm -rf "$ETAPA"' EXIT

# BLOCK-2 (ronda 3, auditoría adversarial 2026-09-22): un nombre FIJO en /tmp, escrito
# por root (vía `install`), permite que un usuario local de la remota plante un symlink
# en esa ruta ANTES de la subida -- root seguiría ese symlink al leer el contenido para
# instalarlo, así que el atacante decide qué instala root. `subir_sin_nombre_fijo` corta
# eso: el ADMINISTRADOR (nunca root) crea el destino con `mktemp` -- un archivo REAL, con
# un nombre que nadie pudo conocer de antemano, existente desde el instante mismo en que
# se crea -- y sube ahí con scp (mismo dueño, sin necesitar permisos de root). root sólo
# LEE ese archivo después, nunca escribe a una ruta que pudo haber sido plantada. Sólo
# aplica en el camino remoto (SSH_OPC existe recién después del `if LOCAL`, más abajo).
subir_sin_nombre_fijo() {
  local origen="$1" remoto
  remoto="$(ssh "${SSH_OPC[@]}" -p "$PUERTO" "$JAX_EJECUTOR_ADMIN_USUARIO@$IP" mktemp)"
  scp -q "${SSH_OPC[@]}" -P "$PUERTO" "$origen" "$JAX_EJECUTOR_ADMIN_USUARIO@$IP:$remoto"
  printf '%s' "$remoto"
}

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

# 3. El script en la remota -- BLOCK-2: nunca un nombre fijo (subir_sin_nombre_fijo).
# MAJOR-D (ronda 4, auditoría adversarial 2026-09-22): `$TMP_SCRIPT_REMOTO` sale de
# `mktemp` EN LA REMOTA -- si el `TMPDIR` de esa sesión estuviera bajo control de
# alguien (una variable de entorno hostil, un perfil de shell tocado), el nombre podría
# traer espacios o `;` e inyectar un comando extra al incrustarlo SIN COMILLAS en el
# `corre "..."` de abajo. `$(printf %q ...)` -- ya usado en el paso 5 para
# `$ADMIN_LOCAL` -- lo vuelve UN solo token opaco para el shell remoto. MINOR (ronda
# 4): `trap ... EXIT` en vez de `&& rm -f` -- si `install` fallara a mitad, el `&&`
# nunca llegaría al `rm`, y el temporal (con el CONTENIDO del script, no una llave,
# pero igual basura ajena) quedaría en la remota. El trap limpia SIEMPRE, éxito o no.
TMP_SCRIPT_REMOTO="$(subir_sin_nombre_fijo "$REPO/ops/ejecutor/ejecutor-huella")"
corre "trap 'rm -f $(printf %q "$TMP_SCRIPT_REMOTO")' EXIT; install -o root -g root -m 0755 $(printf %q "$TMP_SCRIPT_REMOTO") /usr/local/sbin/ejecutor-huella"

# 4. sudoers.d ACOTADO -- exactamente este binario, SIN argumentos, nunca ALL. MAJOR-5
#    (ronda 2, auditoría adversarial 2026-09-22): la cadena vacía `""` NO es decorativa
#    -- verificado en un contenedor Ubuntu 24.04 limpio (sudo 1.9.17p2, 2026-09-22) que
#    SIN ella sudo acepta CUALQUIER argumento («sudo -n cmd hostil» → rc=0) aunque el
#    binario no los use; sólo con `cmd ""` sudo exige que la invocación no tenga
#    argumentos. Sin esto, "sin argumentos: no hay superficie de ataque" sería una
#    afirmación falsa sobre el propio sudoers -- ver huella.py, _COMANDO_FORZADO_HUELLA.
#    BLOCK-2: staging por `subir_sin_nombre_fijo`, no un nombre fijo en /tmp.
printf '%s ALL=(root) NOPASSWD: /usr/local/sbin/ejecutor-huella ""\n' "$ADMIN_LOCAL" > "$ETAPA/sudoers-huella"
TMP_SUDOERS_REMOTO="$(subir_sin_nombre_fijo "$ETAPA/sudoers-huella")"
# MAJOR-D (ronda 4): mismo motivo que el paso 3 -- `$TMP_SUDOERS_REMOTO` sin comillas
# sería una inyección posible vía un `TMPDIR` hostil en la sesión del administrador.
# MINOR: `trap ... EXIT`, no `&& rm -f` -- si `visudo -cf` rechazara el archivo, el
# temporal (con una línea de sudoers, aunque de prueba) igual se limpia.
corre "trap 'rm -f $(printf %q "$TMP_SUDOERS_REMOTO")' EXIT; visudo -cf $(printf %q "$TMP_SUDOERS_REMOTO") && install -o root -g root -m 0440 $(printf %q "$TMP_SUDOERS_REMOTO") /etc/sudoers.d/50-ejecutor-huella"

# 5. MAJOR-4 (ronda 3): el guion remoto (ejecutor-huella) YA NO tiene el administrador
#    hardcodeado -- lo lee de este archivo, que sólo este instalador escribe (root
#    0644, una sola línea). Todo en UN solo comando remoto -- el `mktemp` y su uso
#    quedan dentro de la MISMA invocación de root, sin hop intermedio que necesite
#    `subir_sin_nombre_fijo` (BLOCK-2: no hay ventana entre crear el nombre y usarlo).
corre "install -d -o root -g root -m 0755 /etc/ejecutor-huella \
  && TMP_ADMIN=\$(mktemp) && trap 'rm -f \"\$TMP_ADMIN\"' EXIT \
  && printf '%s\n' $(printf %q "$ADMIN_LOCAL") > \"\$TMP_ADMIN\" \
  && install -o root -g root -m 0644 \"\$TMP_ADMIN\" /etc/ejecutor-huella/admin_usuario"

# 6. authorized_keys del ADMINISTRADOR en la remota -- BLOCK-2/MAJOR-3 (ronda 2/3): la
#    línea la arma `huella.linea_authorized_keys_servicio` (Python, TESTEADO, no bash
#    suelto), y `huella.actualizar_authorized_keys_admin` CONVERGE por MARCA
#    (`ejecutor-huella-servicio`), no por la clave pública actual -- si la llave del
#    servicio se REGENERA entre instalaciones, sigue reemplazando la línea vieja en vez
#    de dejarla huérfana (ver MAJOR-3, `revertir_huella_en_maquina.sh`, mismo criterio).
#    MINOR (ronda 2): `install -o/-g "$ADMIN_LOCAL"` en el directorio Y el archivo --
#    antes, si `~$ADMIN_LOCAL/.ssh` no existía, `corre` (que ejecuta como ROOT vía
#    sudo -n) lo creaba dueño ROOT, dejando al administrador sin poder tocar su propio
#    `authorized_keys` nunca más.
#
#    MINOR (ronda 5, auditoría adversarial 2026-09-22): `$ADMIN_LOCAL` viajaba SIN
#    escapar acá (`-o $ADMIN_LOCAL -g $ADMIN_LOCAL`, `~$ADMIN_LOCAL/...`) -- mismo tipo
#    de hueco que MAJOR-D ya había cerrado para los temporales. Pero `$(printf %q
#    "$ADMIN_LOCAL")` a secas NO alcanza para la parte `~$ADMIN_LOCAL`: la expansión de
#    `~usuario` la hace el shell ANTES de quitarle las comillas a lo que sigue, así que
#    un `~'fruiz'/.ssh` (comillado) YA NO expande el home -- se toma como un directorio
#    literal llamado `~'fruiz'`. La forma correcta es dejar de usar `~usuario` del todo:
#    el HOME se resuelve con `getent passwd` EN LA REMOTA (mismo criterio que
#    `tramo_admin()` en `ejecutor-huella`), y de ahí en más todo queda citado con
#    comillas normales, sin tilde de por medio.
corre "home=\$(getent passwd $ADMIN_Q | cut -d: -f6); test -n \"\$home\" \
  && install -d -o $ADMIN_Q -g $ADMIN_Q -m 0700 \"\$home/.ssh\""
ACTUALES="$(corre "home=\$(getent passwd $ADMIN_Q | cut -d: -f6); test -n \"\$home\" \
  && cat \"\$home/.ssh/authorized_keys\" 2>/dev/null" || true)"
printf '%s' "$ACTUALES" > "$ETAPA/actuales"
( cd "$REPO" && PYTHONDONTWRITEBYTECODE=1 python3 -c '
import sys
from jax.ejecutor.contratos.huella import actualizar_authorized_keys_admin
actuales = open(sys.argv[1], encoding="utf-8").read()
sys.stdout.write(actualizar_authorized_keys_admin(
    actuales, tipo=sys.argv[2], clave=sys.argv[3], origen_ip=sys.argv[4]))
' "$ETAPA/actuales" "$TIPO" "$CLAVE" "$JAX_EJECUTOR_HUELLA_ORIGEN_IP" ) > "$ETAPA/nuevas"
if corre "home=\$(getent passwd $ADMIN_Q | cut -d: -f6); test -n \"\$home\" \
  && cat \"\$home/.ssh/authorized_keys\" 2>/dev/null" | cmp -s - "$ETAPA/nuevas"; then
  echo "codigo=llave_huella_ya_convergida host=\"$NOMBRE\""
else
  TMP_AUTH_REMOTO="$(subir_sin_nombre_fijo "$ETAPA/nuevas")"
  # MAJOR-D (ronda 4): mismo motivo -- `$TMP_AUTH_REMOTO` sin comillas. MINOR: trap,
  # no `&& rm -f` -- este es el que más importa limpiar SIEMPRE: trae el contenido
  # completo del nuevo authorized_keys, con la clave del servicio adentro.
  corre "trap 'rm -f $(printf %q "$TMP_AUTH_REMOTO")' EXIT; \
    home=\$(getent passwd $ADMIN_Q | cut -d: -f6); test -n \"\$home\" \
    && install -o $ADMIN_Q -g $ADMIN_Q -m 0600 $(printf %q "$TMP_AUTH_REMOTO") \"\$home/.ssh/authorized_keys\""
fi

echo "maquina_huella_instalada=\"$NOMBRE\" script_sha256=\"$(corre "sha256sum /usr/local/sbin/ejecutor-huella" | cut -d' ' -f1)\""
