#!/usr/bin/env bash
# ops/rutas-de-produccion.sh --verificar
#
# Envoltorio FINO, solo lectura: no escribe en /etc, /srv, /var ni
# /home/fruiz/jax, no arranca ni detiene ningun servicio. La logica real
# (parseo del .env, resolucion de symlinks, lista de permitidos, chequeo de
# jaxsvc) vive en ops/rutas_de_produccion_verificador.py -- un modulo Python
# importable y probado (tests/test_rutas_de_produccion_verificador.py), no
# en este guion. Ver el docstring de ese modulo para el porque (auditoria de
# escalon 3, PR jax#277, MAJOR-3: la version anterior de este guion en bash
# fallaba ABIERTO en 6 casos reales).
#
# La ruta de /etc/jax/.env esta FIJA acá, nunca configurable por variable de
# entorno: nadie desvia la verificacion de produccion con un ENV_FILE
# puesto por error. El texto del archivo se lee UNA vez con sudo -n cat y se
# le pasa al modulo Python por stdin.
set -euo pipefail

if [ "${1:-}" != "--verificar" ]; then
  echo "uso: $0 --verificar" >&2
  exit 2
fi

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$DIR/rutas_de_produccion_verificador.py" --verificar
