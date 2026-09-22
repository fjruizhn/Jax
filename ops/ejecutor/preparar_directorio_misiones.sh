#!/usr/bin/env bash
# ops/ejecutor/preparar_directorio_misiones.sh — B-2 (ronda 4, auditoría adversarial
# 2026-09-22; MINOR endurecido en la ronda 6; corregido en la ronda 7): JAX_EJECUTOR_MISIONES
# con una ACL EXPLÍCITA, no sólo el modo.
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
# con `group::rwx` -- exactamente el estado que la ronda 6 no corregía. Ahora el modo
# y la ACL se aplican SIEMPRE, exista o no el directorio.
#
# Sin ventana, de nuevo: la corrección es UN SOLO `setfacl -m` con TODAS las entradas
# juntas -- base (`u::`, `g::`, `o::`, `m::`) Y nombradas (`admin`, `cuenta`) Y sus
# default, todas en la MISMA invocación. `setfacl` recalcula la ACL completa a partir
# de esa única lista: no hay un `chmod` (que además, con una ACL puesta, sólo tocaría
# el MASK, no `group::` -- por eso un `chmod` suelto NO alcanza para corregir una
# `group::rwx` vieja) seguido de un `setfacl` por separado, así que no hay un estado
# intermedio distinto del inicial o del final. `mkdir -p` (no `install -d`: `install
# -d` SIEMPRE aplica un modo, incluso sin `-m` -- lo comprobado: resetea 700 a 755 en
# un directorio YA EXISTENTE) para no tocar el modo por su cuenta antes del setfacl
# atómico; `chown` es idempotente, se corre siempre.
set -euo pipefail
DESTINO="${1:?}"; ADMIN="${2:?}"; CUENTA="${3:?}"
sudo mkdir -p "$DESTINO"
sudo chown jaxsvc:jaxsvc "$DESTINO"
sudo setfacl -m "u::rwx,g::---,o::---,m::rwx,u:$ADMIN:rwx,u:$CUENTA:--x,d:u::rwx,d:g::---,d:o::---,d:m::rwx,d:u:$ADMIN:rwx,d:u:$CUENTA:--x" "$DESTINO"
echo "directorio_misiones_preparado=\"$DESTINO\" admin=\"$ADMIN\" cuenta=\"$CUENTA\""
