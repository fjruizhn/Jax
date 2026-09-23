#!/usr/bin/env bash
# ops/ejecutor/revertir_huella_en_maquina.sh <nombre> — deshace
# instalar_huella_en_maquina.sh en UNA máquina remota, en orden inverso. NUNCA borra la
# llave del servicio ni su known_hosts (son compartidos por TODO el inventario -- los
# borra a mano quien de verdad quiera dar de baja el mecanismo entero, no este guion).
# NUNCA borra /etc/ejecutor-huella/admin_usuario tampoco -- lo necesita cualquier huella
# que ejecutor-huella siga tomando en esta máquina mientras el script exista.
# Se detiene ante el primer fallo; cada paso es idempotente.
#
# MAJOR-4 (ronda 2, auditoría adversarial 2026-09-22): "revertido" no se declara por
# haber CORRIDO los tres pasos -- se VERIFICA que ninguno de los tres sigue ahí (el
# script, el sudoers, la línea en authorized_keys) y se sale ≠0 si algo quedó. Y NO se
# deja ninguna copia de `authorized_keys` con la llave adentro (el paso 3 antes hacía
# un `cp ... .antes-de-revertir-huella` -- ESE archivo quedaba en la remota con la
# línea completa, command= y clave pública incluidos; se quitó).
#
# MAJOR-3 (ronda 3, auditoría adversarial 2026-09-22): CORREGIDO -- filtraba y verificaba
# por la CLAVE PÚBLICA actual (`JAX_EJECUTOR_HUELLA_LLAVE.pub`). Si la llave del servicio
# se REGENERÓ entre la instalación y la reversión (rotación, reinstalación de la llave
# local), la clave pública YA NO COINCIDE con la que está en la remota -- el filtro no
# encuentra nada que quitar, la verificación tampoco encuentra nada (busca la clave
# NUEVA, no la vieja que sigue ahí), y el guion imprime "revertida" con la línea VIEJA
# todavía autorizada. Ahora filtra y verifica por la MARCA (`ejecutor-huella-servicio`,
# `huella.MARCA_HUELLA_SERVICIO`) -- fija, no depende de qué clave esté instalada. Esto
# también deja de necesitar `JAX_EJECUTOR_HUELLA_LLAVE` para revertir: funciona incluso
# si la llave local ya no existe.
set -euo pipefail
NOMBRE="${1:?uso: revertir_huella_en_maquina.sh <nombre>}"
: "${JAX_EJECUTOR_POLITICA:?}" "${JAX_EJECUTOR_ADMIN_USUARIO:?}"
REPO="$(git -C "$(dirname "$(readlink -f "$0")")" rev-parse --show-toplevel)"
JAX_EJECUTOR_CUENTA="${JAX_EJECUTOR_CUENTA:-axioma}"
export JAX_EJECUTOR_CUENTA
. "$REPO/ops/ejecutor/_maquina.sh"

if [ "$LOCAL" = si ]; then
  echo "maquina_local_sin_remoto=\"$NOMBRE\""
  exit 0
fi

# MAJOR-3: la MARCA, no la clave pública -- fuente única con huella.py, nunca repetida
# a mano (Principio IV).
MARCA="$(cd "$REPO" && PYTHONDONTWRITEBYTECODE=1 python3 -c \
  'from jax.ejecutor.contratos.huella import MARCA_HUELLA_SERVICIO; print(MARCA_HUELLA_SERVICIO)')"
test -n "$MARCA"

# MINOR (ronda 5, auditoría adversarial 2026-09-22): mismo arreglo que
# instalar_huella_en_maquina.sh -- `$ADMIN_LOCAL` escapado (`$ADMIN_Q`), y el HOME sale
# de `getent passwd` EN LA REMOTA en vez de `~$ADMIN_LOCAL` (que dejaría de expandir el
# home en cuanto `$ADMIN_LOCAL` estuviera comillado con `%q`, ver el comentario largo
# en el instalador).
ADMIN_Q="$(printf %q "$ADMIN_LOCAL")"

# 3. authorized_keys del administrador: quita SÓLO la(s) línea(s) marcadas -- filtrado y
#    reemplazo EN UN SOLO comando remoto, sin backup con la llave adentro. BLOCK-2
#    (ronda 3): el `mktemp` y su único uso quedan DENTRO de la misma invocación de root
#    (sin hop intermedio, sin nombre fijo expuesto entre crear y usar).
if corre "home=\$(getent passwd $ADMIN_Q | cut -d: -f6); test -n \"\$home\" \
  && test -f \"\$home/.ssh/authorized_keys\""; then
  corre "home=\$(getent passwd $ADMIN_Q | cut -d: -f6); test -n \"\$home\" \
    && TMP=\$(mktemp) \
    && { grep -v $(printf %q " $MARCA\$") \"\$home/.ssh/authorized_keys\" > \"\$TMP\" || true; } \
    && install -o $ADMIN_Q -g $ADMIN_Q -m 0600 \"\$TMP\" \"\$home/.ssh/authorized_keys\" \
    && rm -f \"\$TMP\""
fi

# 2. sudoers.d acotado.
corre "rm -f /etc/sudoers.d/50-ejecutor-huella && visudo -c >/dev/null"

# 1. El script.
corre "rm -f /usr/local/sbin/ejecutor-huella"

# MAJOR-4: verificar, no declarar. Un solo chequeo remoto -- si algo de los tres sigue
# ahí, esto sale ≠0 (el `&&` corta la cadena) y NUNCA se imprime "revertida". MAJOR-3:
# la verificación también es por MARCA, no por clave -- así una llave rotada entre medio
# no puede hacer que esto mienta. MINOR (ronda 5): si `getent` no resuelve el HOME, esto
# también tiene que fallar (fail-closed) -- nunca declarar "revertida" sin haber podido
# verificar de verdad la línea de authorized_keys.
corre "test ! -e /usr/local/sbin/ejecutor-huella \
  && test ! -e /etc/sudoers.d/50-ejecutor-huella \
  && home=\$(getent passwd $ADMIN_Q | cut -d: -f6) && test -n \"\$home\" \
  && { ! grep -q $(printf %q " $MARCA\$") \"\$home/.ssh/authorized_keys\" 2>/dev/null; }"

echo "maquina_huella_revertida=\"$NOMBRE\" verificado=true"
