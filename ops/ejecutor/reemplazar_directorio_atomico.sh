#!/usr/bin/env bash
# ops/ejecutor/reemplazar_directorio_atomico.sh ETAPA DESTINO — deja DESTINO EXACTAMENTE
# igual a ETAPA (mismo conjunto de archivos, dueño root:root, modo 0644), root:root 0755
# en los directorios. Arma el árbol nuevo APARTE y lo intercambia con UN solo `mv` por
# lado (rename atómico dentro del mismo filesystem): nunca copia archivo por archivo
# ENCIMA de lo que ya está instalado, que es lo que dejaba basura (M4, auditoría
# adversarial 2026-09-22: el instalador viejo agregaba archivos nuevos pero nunca
# borraba los que sobraban de una skill retirada).
#
# Usa sudo para todo lo de root; el llamador (instalar_contexto.sh) ya exige sudo.
set -euo pipefail
ETAPA="${1:?uso: reemplazar_directorio_atomico.sh ETAPA DESTINO}"
DESTINO="${2:?uso: reemplazar_directorio_atomico.sh ETAPA DESTINO}"
test -d "$ETAPA"

NUEVO="${DESTINO}.nuevo.$$"
VIEJO="${DESTINO}.viejo.$$"
sudo rm -rf "$NUEVO" "$VIEJO"
trap 'sudo rm -rf "$NUEVO" "$VIEJO"' EXIT

sudo install -d -o root -g root -m 0755 "$NUEVO"
( cd "$ETAPA" && find . -type f ) | while IFS= read -r rel; do
  sudo install -D -o root -g root -m 0644 "$ETAPA/$rel" "$NUEVO/$rel"
done

sudo install -d -o root -g root -m 0755 "$(dirname "$DESTINO")"
if sudo test -e "$DESTINO"; then
  sudo mv -T "$DESTINO" "$VIEJO"
fi
sudo mv -T "$NUEVO" "$DESTINO"
