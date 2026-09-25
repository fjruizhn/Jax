#!/usr/bin/env bash
# Sólo para pruebas de tests/test_verificar_arranque_instalado.py -- NUNCA se
# instala ni se usa fuera de un PATH temporal de un test, y nunca escribe
# nada. Para cualquier invocación que no sea `show ... jax-las-manos.service`
# delega al systemctl real. Para esa, el comportamiento depende de
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
if [ "${1:-}" = show ] && [ "$ultimo" = jax-las-manos.service ]; then
  if [ "$MODO" = falla ]; then
    echo "systemctl-falso: unidad simulada como inexistente" >&2
    exit 1
  fi
  printf '/etc/systemd/system/jax-las-manos.service\n'
  printf '/etc/systemd/system/jax-las-manos.service.d/checkout-de-produccion.conf /etc/systemd/system/jax-las-manos.service.d/cuenta-de-servicio.conf\n'
  exit 0
fi
exec "$REAL" "$@"
