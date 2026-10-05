#!/usr/bin/env bash
# ops/ejecutor/respaldar-base-del-proxy.sh <MARCA> [DESTDIR]
#
# Respalda la unidad base INSTALADA del proxy del Ejecutor antes de que
# instalar_registro_y_cerco.sh la pise (m3, auditoría de #356: la base no tiene otro
# instalador y se pisaba sin copia).
#
#   origen   <DESTDIR>/etc/systemd/system/jax-ejecutor-proxy.service
#   destino  <DESTDIR>/etc/jax-ejecutor-cerco/respaldos/jax-ejecutor-proxy.service.<MARCA>
#
# El destino está fuera de /etc/systemd: systemd no lee ese directorio. Sin base instalada no
# hay nada que respaldar y sale 0. Imprime el comando para revertir.
#
# DESTDIR (default vacío = el sistema real, con sudo y dueño root): con DESTDIR puesto -- las
# pruebas, sobre un árbol falso del propio usuario -- no se usa sudo ni se cambia el dueño.
# MARCA: solo dígitos y guion (la que produce `date +%Y%m%d-%H%M%S`); cualquier otra cosa se
# rechaza, porque forma parte de una ruta.
set -euo pipefail

MARCA="${1:-}"
DESTDIR="${2:-}"
if ! [[ "$MARCA" =~ ^[0-9]+(-[0-9]+)*$ ]]; then
  echo "respaldar-base-del-proxy: MARCA inválida '$MARCA' (solo dígitos y guiones, como 'date +%Y%m%d-%H%M%S')" >&2
  exit 2
fi

origen="$DESTDIR/etc/systemd/system/jax-ejecutor-proxy.service"
dir_respaldos="$DESTDIR/etc/jax-ejecutor-cerco/respaldos"
destino="$dir_respaldos/jax-ejecutor-proxy.service.$MARCA"

if [ ! -f "$origen" ]; then
  echo "respaldar-base-del-proxy: no hay base instalada en $origen -- nada que respaldar"
  exit 0
fi

if [ -n "$DESTDIR" ]; then
  install -d -m 0755 "$dir_respaldos"
  cp -p -- "$origen" "$destino"
else
  sudo install -d -o root -g root -m 0755 "$dir_respaldos"
  sudo cp -p -- "$origen" "$destino"
fi
echo "respaldar-base-del-proxy: base respaldada en $destino"
echo "Para revertir la base del proxy: sudo cp -p $destino /etc/systemd/system/jax-ejecutor-proxy.service y recargar systemd"
