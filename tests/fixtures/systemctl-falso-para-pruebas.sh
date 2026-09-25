#!/usr/bin/env bash
# Sólo para pruebas de tests/test_verificar_arranque_instalado.py -- NUNCA se
# instala ni se usa fuera de un PATH temporal de un test, y nunca escribe
# nada.
#
# Ronda 4: el guion real hace DOS consultas `show` distintas para
# jax-las-manos.service -- `-p FragmentPath` sola (lado DISCO, en
# obtener_reales_disco_produccion) y `-p FragmentPath -p DropInPaths
# -p NeedDaemonReload` (lado CARGADO, en obtener_reales_cargado). Este
# systemctl falso SÓLO intercepta la segunda (matchea por la presencia de
# "NeedDaemonReload" entre los argumentos, no sólo por el nombre de la
# unidad) -- así el lado DISCO sigue viendo el systemctl real (y por lo
# tanto el disco real, correcto) mientras el lado CARGADO ve la respuesta
# de mentira. Cualquier otra invocación delega al systemctl real.
#
# $SYSTEMCTL_FALSO_MODO:
#   incompleto (default): responde OK pero SIN z-pythonpath.conf -- para
#     probar que "lo cargado" y el manifiesto pueden desacordar aunque el
#     disco esté perfecto.
#   falla: sale con error, como si la unidad no existiera o systemctl
#     estuviera roto -- para probar que el guion no aborta, sigue con las
#     demás unidades y da rc=1 al final.
set -euo pipefail
REAL=/usr/bin/systemctl
MODO="${SYSTEMCTL_FALSO_MODO:-incompleto}"
ultimo="${*: -1}"
if [ "${1:-}" = show ] && [ "$ultimo" = jax-las-manos.service ] && [[ " $* " == *" NeedDaemonReload "* ]]; then
  if [ "$MODO" = falla ]; then
    echo "systemctl-falso: unidad simulada como inexistente" >&2
    exit 1
  fi
  printf '/etc/systemd/system/jax-las-manos.service\n'
  printf '/etc/systemd/system/jax-las-manos.service.d/checkout-de-produccion.conf /etc/systemd/system/jax-las-manos.service.d/cuenta-de-servicio.conf\n'
  printf 'no\n'
  exit 0
fi
exec "$REAL" "$@"
