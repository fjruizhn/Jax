#!/usr/bin/env bash
# OBSOLETO (auditoria Jax#354, 2026-10-05). Este instalador copiaba solo la unidad del worker, la
# habilitaba y arrancaba de una vez, y copiaba desde un checkout de TRABAJO (/home/fruiz/jax), no desde
# produccion. Las cinco unidades jax-memory-* se instalan segun docs/runbooks/memoria-cola-atascada.md y
# docs/runbooks/deployment-rollback.md; los timers los habilita el responsable, nunca un guion.
# Se conserva solo para reproducir la instalacion historica: exige la bandera --instalar-obsoleto y
# NO habilita ni arranca nada.
#
# 2026-10-05: ademas de worker y synthesis (unidad + timer), instala TODOS los drop-ins versionados de
# esas dos unidades (config/systemd/<unidad>.service.d/*.conf): el checkout de produccion con su freno,
# la cuenta de servicio, el PYTHONPATH y los limites de tiempo. Antes vivian solo en /etc/systemd/system
# y reinstalar desde el repo los perdia en silencio.
#
# JAX_SYSTEMD_SRC  origen (default: el checkout de trabajo historico).
# JAX_SYSTEMD_DEST destino (default: /etc/systemd/system). Con un destino distinto del real NO se recarga
#                  systemd: es la forma de ejercitar este guion en una prueba, sin tocar el servidor.
set -euo pipefail

if [[ "${1:-}" != "--instalar-obsoleto" ]]; then
  echo "install-memory-scope.sh esta OBSOLETO y no hace nada sin --instalar-obsoleto." >&2
  echo "Ver docs/runbooks/memoria-cola-atascada.md y docs/runbooks/deployment-rollback.md." >&2
  exit 2
fi

REAL="/etc/systemd/system"
SD="${JAX_SYSTEMD_SRC:-/home/fruiz/jax/config/systemd}"
DEST="${JAX_SYSTEMD_DEST:-$REAL}"
UNIDADES=(jax-memory-worker jax-memory-synthesis)

echo "== Copia las unidades, sus timers y sus drop-ins (sin habilitar ni arrancar nada) =="
for unidad in "${UNIDADES[@]}"; do
  install -m 644 "$SD/$unidad.service" "$DEST/"
  install -m 644 "$SD/$unidad.timer" "$DEST/"
  dropins="$SD/$unidad.service.d"
  if [[ ! -d "$dropins" ]]; then
    echo "FALTA $dropins: el repo no reproduciria la instalacion de $unidad" >&2
    exit 1
  fi
  install -d -m 755 "$DEST/$unidad.service.d"
  for conf in "$dropins"/*.conf; do
    install -m 644 "$conf" "$DEST/$unidad.service.d/"
  done
done

if [[ "$DEST" == "$REAL" ]]; then
  systemctl daemon-reload
else
  echo "(destino de prueba $DEST: no se recarga systemd)"
fi
echo "LISTO. Habilitar el timer (o no) es decision del responsable."
