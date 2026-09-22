#!/usr/bin/env bash
# ops/ejecutor/reemplazar_directorio_atomico.sh ETAPA DESTINO — deja DESTINO EXACTAMENTE
# igual a ETAPA (mismo conjunto de archivos, dueño root:root, modo 0644), root:root 0755
# en los directorios. Arma el árbol nuevo APARTE y lo pone en el lugar de DESTINO con UN
# solo syscall atómico (`exch`, de util-linux -- envuelve `renameat2(RENAME_EXCHANGE)`,
# verificado disponible: `exch --version`, más abajo lo exige) cuando DESTINO ya existe;
# nunca copia archivo por archivo ENCIMA de lo que ya está instalado, que es lo que dejaba
# basura (M4, auditoría adversarial 2026-09-22: el instalador viejo agregaba archivos
# nuevos pero nunca borraba los que sobraban de una skill retirada).
#
# CORREGIDO (ronda 3, auditoría adversarial 2026-09-22): la versión anterior hacía dos
# `mv` separados (DESTINO -> VIEJO, después NUEVO -> DESTINO) -- entre esos dos `mv` hay
# una VENTANA real en la que DESTINO no existe (una jaula que intente montarlo en ese
# instante exacto falla). `exch` intercambia los dos nombres en UN solo syscall: no hay
# ventana. `exch` exige que los DOS caminos existan (es un intercambio, no un reemplazo);
# por eso, si DESTINO todavía no existe (primera instalación), se hace un `mv` simple --
# ESE caso sí es atómico con un solo `mv` (no hay nada que preservar del lado de DESTINO).
#
# Usa sudo para todo lo de root; el llamador (instalar_contexto.sh) ya exige sudo.
set -euo pipefail
ETAPA="${1:?uso: reemplazar_directorio_atomico.sh ETAPA DESTINO}"
DESTINO="${2:?uso: reemplazar_directorio_atomico.sh ETAPA DESTINO}"
test -d "$ETAPA"
command -v exch >/dev/null || { echo "codigo=exch_no_disponible" >&2; exit 1; }

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
  # Intercambio atómico: tras esto, DESTINO tiene el contenido NUEVO y $VIEJO tiene lo
  # que antes había en DESTINO -- se borra en el trap de arriba, ya fuera de la ventana
  # crítica (nadie deja de ver un DESTINO válido mientras tanto).
  sudo exch "$DESTINO" "$NUEVO"
  # Sin `mv -T` acá: tras el exch, "$NUEVO" ES el directorio viejo (nombre intercambiado,
  # no contenido movido) -- el trap de la línea de arriba ya lo borra al salir.
else
  sudo mv -T "$NUEVO" "$DESTINO"
fi
