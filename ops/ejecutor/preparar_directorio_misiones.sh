#!/usr/bin/env bash
# ops/ejecutor/preparar_directorio_misiones.sh — B-2 (ronda 4, auditoría adversarial
# 2026-09-22; MINOR endurecido en la ronda 6): JAX_EJECUTOR_MISIONES con una ACL
# EXPLÍCITA, no sólo el modo.
# Uso: preparar_directorio_misiones.sh <directorio> <admin> <cuenta>
#
# `<admin>` (JAX_EJECUTOR_ADMIN_USUARIO, fruiz) conserva rwx -- escribe la misión y
# toma/compara la huella (M-1). `<cuenta>` (JAX_EJECUTOR_CUENTA, axioma) recibe SÓLO
# travesía (--x, SIN +r): puede entrar a `<mision_id>/claude-projects` si ya sabe el
# UUID exacto (se lo da la propia misión que está corriendo), pero no puede LISTAR
# este directorio ni leer `<id>.json` de ninguna misión.
#
# Modo BASE `0700`, no `0750` (MINOR, ronda 6 -- lo que decía el comentario viejo era
# incompleto sobre `group::`): con `0750`, el bit de GRUPO de la propia jaxsvc
# ("group::r-x") le daría LISTAR y LEER este directorio a CUALQUIER cuenta que sea
# miembro del grupo `jaxsvc` -- no sólo a `admin`/`cuenta`, que son las dos únicas
# pensadas para entrar acá. Con `0700`, `group::` y `other::` quedan en `---`; el
# acceso de `admin` y `cuenta` sale ENTERO de sus entradas ACL nombradas, no del modo.
#
# Sin ventana (MINOR, ronda 6): la primera vez (el directorio no existe) `install -d`
# lo crea con el modo final en UNA sola syscall -- no hay estado intermedio más
# permisivo. En una corrida IDEMPOTENTE (el directorio YA existe, con su ACL puesta),
# NO se vuelve a llamar a `install -d`: un `chmod` (que es lo que `install -d` hace
# sobre un directorio existente) recalcula el `mask::` de la ACL a partir de los bits
# de grupo pedidos, y por un instante reduce el permiso EFECTIVO de las entradas
# nombradas (a `admin` le tocaría, de rebote, perder el `w` hasta que `setfacl`
# reponga el mask) -- una escritura de misión que corriera justo en ese instante
# podría fallar. `setfacl -m` sobre una ACL sin cambios no toca el mask.
set -euo pipefail
DESTINO="${1:?}"; ADMIN="${2:?}"; CUENTA="${3:?}"
if [ ! -e "$DESTINO" ]; then
  sudo install -d -o jaxsvc -g jaxsvc -m 0700 "$DESTINO"
fi
sudo setfacl -m "u:$ADMIN:rwx" -m "d:u:$ADMIN:rwx" "$DESTINO"
sudo setfacl -m "u:$CUENTA:--x" -m "d:u:$CUENTA:--x" "$DESTINO"
echo "directorio_misiones_preparado=\"$DESTINO\" admin=\"$ADMIN\" cuenta=\"$CUENTA\""
