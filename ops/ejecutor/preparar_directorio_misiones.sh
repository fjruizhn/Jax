#!/usr/bin/env bash
# ops/ejecutor/preparar_directorio_misiones.sh — B-2 (ronda 4, auditoría adversarial
# 2026-09-22; MINOR endurecido en la ronda 6; corregido en la ronda 7; mecanismo vuelto
# a separar en la ronda 8): JAX_EJECUTOR_MISIONES con una ACL EXPLÍCITA, no sólo el modo.
# Uso: preparar_directorio_misiones.sh <directorio> <admin> <cuenta>
#
# `<admin>` (JAX_EJECUTOR_ADMIN_USUARIO, fruiz) conserva rwx -- escribe la misión y
# toma/compara la huella (M-1). `<cuenta>` (JAX_EJECUTOR_CUENTA, axioma) recibe SÓLO
# travesía (--x, SIN +r): puede entrar a `<mision_id>/claude-projects` si ya sabe el
# UUID exacto (se lo da la propia misión que está corriendo), pero no puede LISTAR
# este directorio ni leer `<id>.json` de ninguna misión.
#
# Modo BASE `0700`, no `0750`: con `0750`, el bit de GRUPO de la propia jaxsvc
# ("group::r-x") le daría LISTAR y LEER este directorio a CUALQUIER cuenta que sea
# miembro del grupo `jaxsvc` -- no sólo a `admin`/`cuenta`, que son las dos únicas
# pensadas para entrar acá. Con `0700`, `group::` y `other::` quedan en `---`; el
# acceso de `admin` y `cuenta` sale ENTERO de sus entradas ACL nombradas, no del modo.
#
# Ronda 7 (BLOCK reproducido, 2026-09-22): la ronda 6 sólo aplicaba el modo/ACL al
# CREAR el directorio -- si ya existía (una instalación de ANTES de este script, o
# cualquier otra cosa que lo hubiera tocado), se quedaba con lo que tuviera puesto
# para siempre. La ACL real de producción, medida ese mismo día, tenía `drwxrwx---`
# con `group::rwx` -- exactamente el estado que la ronda 6 no corregía.
#
# Ronda 8 (MINOR): se vuelve a separar el mecanismo por caso, sin resignar la
# corrección de la ronda 7:
# - CREAR (el directorio no existe todavía): `install -d -m 0700 -o jaxsvc -g jaxsvc`,
#   UNA sola invocación que fija dueño+grupo+modo restrictivo de una vez -- sin ventana
#   donde el directorio quede con un modo más permisivo que el final, ni siquiera un
#   instante.
# - YA EXISTE: `chmod 0700` (converge el modo base de inmediato) seguido del MISMO
#   `setfacl -m` combinado de la ronda 7 -- base (`u::`,`g::`,`o::`,`m::`) Y nombradas
#   (`admin`,`cuenta`) Y sus default, todas en una sola invocación de `setfacl`. El
#   `setfacl` combinado es quien realmente corrige `group::` en el caso viejo (medido:
#   un `chmod` sobre un directorio que YA tiene una ACL con entradas nombradas sólo
#   toca el MASK, no `group::` -- por eso el `setfacl` no se resigna, se mantiene tal
#   cual la ronda 7 lo dejó); el `chmod` previo es la corrección inmediata para el caso,
#   más común, de un directorio SIN ACL extendida todavía (ahí sí gobierna `group::`
#   directamente) y deja explícito el paso que pide la ronda 8.
# `install -d` NO se usa para el caso "ya existe": comprobado que resetea el modo
# (700→755) incluso sobre un directorio YA EXISTENTE, aunque no se le pase `-m`.
set -euo pipefail
DESTINO="${1:?}"; ADMIN="${2:?}"; CUENTA="${3:?}"
ACL="u::rwx,g::---,o::---,m::rwx,u:$ADMIN:rwx,u:$CUENTA:--x,d:u::rwx,d:g::---,d:o::---,d:m::rwx,d:u:$ADMIN:rwx,d:u:$CUENTA:--x"

if sudo test -d "$DESTINO"; then
  sudo chown jaxsvc:jaxsvc "$DESTINO"
  sudo chmod 0700 "$DESTINO"
  sudo setfacl -m "$ACL" "$DESTINO"
else
  sudo install -d -m 0700 -o jaxsvc -g jaxsvc "$DESTINO"
  sudo setfacl -m "$ACL" "$DESTINO"
fi
echo "directorio_misiones_preparado=\"$DESTINO\" admin=\"$ADMIN\" cuenta=\"$CUENTA\""
