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
# La ruta de /etc/jax/.env esta FIJA en el modulo Python (nunca configurable
# por variable de entorno): nadie desvia la verificacion de produccion con
# un ENV_FILE puesto por error. Este guion NO lee el archivo ni se lo pasa
# por stdin -- ese comentario decia eso en una version anterior y ya no es
# cierto (ronda 3 de la auditoria de escalon 3, jax#277, MINOR-6a): el
# modulo hace su propio `sudo -n cat /etc/jax/.env` internamente
# (rutas_de_produccion_verificador.py:_main). Este guion sólo valida el
# argumento y exec-ea el modulo.
set -euo pipefail

if [ "${1:-}" != "--verificar" ]; then
  echo "uso: $0 --verificar" >&2
  exit 2
fi

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$DIR/rutas_de_produccion_verificador.py" --verificar
