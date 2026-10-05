#!/usr/bin/env bash
# OBSOLETO (auditoria Jax#354, 2026-10-05). Este instalador solo conoce la unidad del worker, la
# habilitaba y arrancaba de una vez, y copiaba desde un checkout de TRABAJO (/home/fruiz/jax), no desde
# produccion. Las cinco unidades jax-memory-* se instalan segun docs/runbooks/memoria-cola-atascada.md y
# docs/runbooks/deployment-rollback.md; los timers los habilita el responsable, nunca un guion.
# Se conserva solo para reproducir la instalacion historica: exige la bandera --instalar-obsoleto y
# NO habilita ni arranca nada.
set -euo pipefail

if [[ "${1:-}" != "--instalar-obsoleto" ]]; then
  echo "install-memory-scope.sh esta OBSOLETO y no hace nada sin --instalar-obsoleto." >&2
  echo "Ver docs/runbooks/memoria-cola-atascada.md y docs/runbooks/deployment-rollback.md." >&2
  exit 2
fi

SD="${JAX_SYSTEMD_SRC:-/home/fruiz/jax/config/systemd}"
echo "== Copia las unidades del worker (sin habilitar ni arrancar nada) =="
install -m 644 "$SD/jax-memory-worker.service" /etc/systemd/system/
install -m 644 "$SD/jax-memory-worker.timer"  /etc/systemd/system/
systemctl daemon-reload
echo "LISTO. Habilitar el timer (o no) es decision del responsable."
