#!/usr/bin/env bash
# ops/ejecutor/retirar_unidad_vigia.sh — retira `ejecutor-vigia@.service` (y su drop-in
# `ejecutor-vigia@.service.d/`) de /etc/systemd/system.
#
# Por qué (ronda 5, auditoría adversarial 2026-09-22, decisión de Fernando): el camino REAL
# del vigía es `jax/ejecutor/mision_servicio.py::abrir_vigia`, que lo lanza como SUBPROCESO
# DIRECTO de cada turno -- nunca por esta unidad. El journal de esta unidad (verificado el
# 2026-09-22, `journalctl -u 'ejecutor-vigia@*'`) muestra exactamente DOS arranques, los dos
# manuales de verificación del despliegue #190 (2026-09-17: `ejecutor-vigia@despliegue190-
# atemai` y `ejecutor-vigia@despliegue190-vm`, nombres de instancia que no son un UUID de
# misión real) -- cero desde entonces, cero de una misión real alguna vez. Además describía
# un camino falso: `User=jaxsvc`, cuando el vigía real hereda `fruiz` de jax-platform.
#
# Sin `[Install]` en la unidad (nunca tuvo `WantedBy`): no hace falta `systemctl disable`,
# sólo parar cualquier instancia que estuviera corriendo (no debería haber ninguna), borrar
# los archivos y recargar. Idempotente: correrlo dos veces no falla, la segunda vez no
# encuentra nada que retirar.
#
# Hace un respaldo ANTES de borrar nada, en /root/respaldo-ejecutor-vigia-<fecha>/ (0700,
# root:root) -- la unidad y el drop-in completos, tal como estaban instalados.
#
# NO lo corre esta sesión (de sólo lectura salvo SELECT/SHOW/EXPLAIN contra producción): lo
# corre la sesión principal, con sudo, después de revisar esta auditoría.
set -euo pipefail

UNIDAD=/etc/systemd/system/ejecutor-vigia@.service
DROPIN_DIR=/etc/systemd/system/ejecutor-vigia@.service.d
RESPALDO="/root/respaldo-ejecutor-vigia-$(date +%Y%m%d-%H%M%S)"

if [ ! -e "$UNIDAD" ] && [ ! -d "$DROPIN_DIR" ]; then
  echo "unidad_retirada=true respaldo=\"\" motivo=\"ya no estaba instalada\""
  exit 0
fi

# Por si quedara alguna instancia corriendo (no debería, el camino real nunca la usa):
# pararlas ANTES de tocar los archivos, para que un SIGTERM normal audite lo pendiente.
INSTANCIAS="$(systemctl list-units --plain --no-legend 'ejecutor-vigia@*' 2>/dev/null | awk '{print $1}')"
if [ -n "$INSTANCIAS" ]; then
  echo "$INSTANCIAS" | xargs -r sudo systemctl stop
fi

sudo mkdir -p "$RESPALDO"
sudo chmod 700 "$RESPALDO"
[ -e "$UNIDAD" ] && sudo cp -a "$UNIDAD" "$RESPALDO/"
[ -d "$DROPIN_DIR" ] && sudo cp -a "$DROPIN_DIR" "$RESPALDO/"

sudo rm -f "$UNIDAD"
sudo rm -rf "$DROPIN_DIR"
sudo systemctl daemon-reload

test ! -e "$UNIDAD"
test ! -d "$DROPIN_DIR"
echo "unidad_retirada=true respaldo=\"$RESPALDO\""
