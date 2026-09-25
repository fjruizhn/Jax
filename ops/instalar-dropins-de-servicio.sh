#!/usr/bin/env bash
# ops/instalar-dropins-de-servicio.sh <unidad>.service.d <REPO> [DESTDIR]
#
# Instala en "$DESTDIR/etc/systemd/system/<unidad>.service.d/" los drop-ins
# de <unidad> que ops/manifiesto-arranque-instalado.tsv declara para ella --
# derivado del manifiesto, no una lista aparte: si el manifiesto gana o
# pierde un drop-in de este servicio, este guion lo sigue sin que nadie lo
# edite (ops/versionar-drop-ins, 2026-09-25).
#
# Lo llaman config/systemd/install-memory-scope.sh y
# ops/ejecutor/instalar_registro_y_cerco.sh, cada uno con su propia unidad,
# justo antes de su propio `daemon-reload` -- este guion NO hace
# daemon-reload ni systemctl: sólo copia archivos, igual que las líneas
# `install -m ...`/`sudo install ...` que ya tenían para la unidad base.
#
# DESTDIR (default vacío = comportamiento de siempre, /etc real): para
# probar sin tocar el /etc real, se invoca con DESTDIR=<directorio temporal>
# y las rutas quedan bajo "$DESTDIR/etc/systemd/system/...".
set -euo pipefail
UNIDAD_DIR="${1:?uso: instalar-dropins-de-servicio.sh <unidad>.service.d <REPO> [DESTDIR]}"
REPO="${2:?uso: instalar-dropins-de-servicio.sh <unidad>.service.d <REPO> [DESTDIR]}"
DESTDIR="${3:-}"

MANIFIESTO="$REPO/ops/manifiesto-arranque-instalado.tsv"
[ -f "$MANIFIESTO" ] || {
  echo "instalar-dropins-de-servicio: no se encontró el manifiesto: $MANIFIESTO" >&2
  exit 1
}

PREFIJO="/etc/systemd/system/$UNIDAD_DIR/"
instalados=0

while IFS=$'\t' read -r repo_rel instalada || [ -n "$repo_rel" ]; do
  [ -z "$repo_rel" ] && continue
  case "$instalada" in
    "$PREFIJO"*)
      sudo install -d -o root -g root -m 0755 "$DESTDIR/etc/systemd/system/$UNIDAD_DIR"
      sudo install -o root -g root -m 0644 "$REPO/$repo_rel" "$DESTDIR$instalada"
      instalados=$((instalados + 1))
      ;;
  esac
done < "$MANIFIESTO"

if [ "$instalados" -eq 0 ]; then
  echo "instalar-dropins-de-servicio: ningún drop-in de '$UNIDAD_DIR' en el manifiesto -- revisar el nombre de la unidad" >&2
  exit 1
fi

echo "instalar-dropins-de-servicio: $instalados drop-in(s) de $UNIDAD_DIR instalados en ${DESTDIR:-/} (sin daemon-reload -- lo hace quien invoca)"
