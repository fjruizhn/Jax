#!/usr/bin/env bash
# Sólo para pruebas de tests/test_verificar_arranque_instalado.py -- NUNCA se
# instala ni se usa fuera de un PATH temporal de un test, y nunca escribe
# nada.
#
# Ronda 5: la capa DISCO ya NO llama a `systemctl` en absoluto (un solo
# camino -- `enumerar_dropins_en_disco` -- para producción y pruebas,
# nunca `systemd-delta` ni `systemctl show` de ese lado). La ÚNICA
# consulta real a `systemctl` que queda en todo el guion es la de la capa
# CARGADO (`-p FragmentPath -p DropInPaths -p NeedDaemonReload`, sólo
# producción) -- así que este systemctl falso sólo necesita interceptar
# ESA, para jax-las-manos.service (matchea por la presencia de
# "NeedDaemonReload" entre los argumentos, además del nombre de la
# unidad, para no interceptar por accidente otra consulta). Cualquier otra
# invocación delega al systemctl real.
#
# $SYSTEMCTL_FALSO_MODO:
#   incompleto (default): responde OK, NeedDaemonReload=no, pero SIN
#     z-pythonpath.conf -- para probar que "lo cargado" y el manifiesto
#     pueden desacordar aunque el disco esté perfecto.
#   falla: sale con error, como si la unidad no existiera o systemctl
#     estuviera roto -- para probar que el guion no aborta, sigue con las
#     demás unidades y da rc=1 al final.
#   necesita_reload: responde OK, con TODOS los drop-ins reales (nada
#     falta), pero NeedDaemonReload=yes -- para probar el chequeo de
#     NeedDaemonReload en aislamiento, sin que ningún otro desacuerdo
#     (DropInPaths incompleto, systemctl roto) sea lo que en realidad hace
#     fallar el test (MAJOR-2, ronda 5).
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
  if [ "$MODO" = necesita_reload ]; then
    printf '/etc/systemd/system/jax-las-manos.service.d/checkout-de-produccion.conf /etc/systemd/system/jax-las-manos.service.d/cuenta-de-servicio.conf /etc/systemd/system/jax-las-manos.service.d/z-pythonpath.conf\n'
    printf 'yes\n'
  else
    printf '/etc/systemd/system/jax-las-manos.service.d/checkout-de-produccion.conf /etc/systemd/system/jax-las-manos.service.d/cuenta-de-servicio.conf\n'
    printf 'no\n'
  fi
  exit 0
fi
exec "$REAL" "$@"
