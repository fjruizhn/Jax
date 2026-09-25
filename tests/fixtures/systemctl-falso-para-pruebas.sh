#!/usr/bin/env bash
# Sólo para pruebas de tests/test_verificar_arranque_instalado.py -- NUNCA se
# instala ni se usa fuera de un PATH temporal de un test, y nunca escribe
# nada.
#
# Ronda 5: la capa DISCO ya NO llama a `systemctl` en absoluto (un solo
# camino -- `enumerar_dropins_en_disco` -- para producción y pruebas,
# nunca `systemd-delta` ni `systemctl show` de ese lado). La ÚNICA
# consulta real a `systemctl` que queda en todo el guion es la de la capa
# CARGADO (`-p FragmentPath -p DropInPaths -p NeedDaemonReload -p Id
# -p Names`, ronda 6 -- sin `--value`, ver abajo).
#
# Ronda 6, MINOR-1: la capa cargado ahora puede activarse bajo
# RAIZ_PRUEBA (`SYSTEMCTL_DE_PRUEBA`), y el guion la corre para las 6
# unidades del manifiesto -- no sólo jax-las-manos.service. En un runner
# de CI limpio (ubuntu-latest, sin ningún jax*.service real cargado), un
# `systemctl show` real sobre las otras 5 unidades daría `FragmentPath=`
# y `DropInPaths=` VACÍOS (unidad no encontrada, pero `systemctl show` no
# falla por eso) -- una diferencia falsa, ajena a lo que el test intenta
# aislar. Por eso este systemctl falso responde con datos CORRECTOS
# (calcados del manifiesto) para las 6 unidades siempre, y sólo desvía el
# comportamiento de jax-las-manos.service según $SYSTEMCTL_FALSO_MODO --
# así la capa cargado de prueba es HERMÉTICA: nunca toca el systemctl
# real, en ningún runner. Cualquier OTRA unidad (fuera de las 6 del
# manifiesto) delega al systemctl real.
#
# Ronda 6, MINOR-2: el guion real dejó de pasar `--value` (medido en
# hall9000: `systemctl show` NO respeta el orden de los `-p` pedidos, así
# que el guion parsea `Clave=valor` por clave, no por posición) -- este
# systemctl falso imita el formato REAL (`Id=...`, `Names=...`,
# `FragmentPath=...`, `DropInPaths=...`, `NeedDaemonReload=...`, un orden
# DISTINTO del pedido, a propósito, para no esconder un regreso al
# parseo posicional).
#
# $SYSTEMCTL_FALSO_MODO (sólo afecta a jax-las-manos.service; las otras 5
# unidades SIEMPRE responden con datos correctos, sea cual sea el modo):
#   incompleto (default): responde OK, NeedDaemonReload=no, Names=Id
#     (sin alias), pero SIN z-pythonpath.conf -- para probar que "lo
#     cargado" y el manifiesto pueden desacordar aunque el disco esté
#     perfecto.
#   falla: sale con error, como si la unidad no existiera o systemctl
#     estuviera roto -- para probar que el guion no aborta, sigue con las
#     demás unidades y da rc=1 al final.
#   necesita_reload: responde OK, con TODOS los drop-ins reales (nada
#     falta) y Names=Id (sin alias), pero NeedDaemonReload=yes -- para
#     probar el chequeo de NeedDaemonReload en aislamiento, sin que
#     ningún otro desacuerdo sea lo que en realidad hace fallar el test
#     (MAJOR-2, ronda 5).
#   alias_cargado: responde OK, con TODOS los drop-ins reales, ambos
#     nombres reales (nada falta, nada sobra) y NeedDaemonReload=no, pero
#     `Names` trae un segundo nombre además del `Id` -- para probar el
#     chequeo `Names == Id` en aislamiento, sin que ningún otro
#     desacuerdo (DropInPaths incompleto, NeedDaemonReload) sea lo que en
#     realidad hace fallar el test (MAJOR-1, ronda 6).
#   correcto: responde OK, con TODOS los drop-ins reales, Names=Id (sin
#     alias) y NeedDaemonReload=no -- el caso feliz, para probar que
#     activar la capa cargado bajo RAIZ_PRUEBA (MINOR-1, ronda 6) por sí
#     solo no hace fallar nada cuando todo coincide de verdad.
set -euo pipefail
REAL=/usr/bin/systemctl
MODO="${SYSTEMCTL_FALSO_MODO:-incompleto}"
ultimo="${*: -1}"

# Drop-ins correctos por unidad, calcados de ops/manifiesto-arranque-instalado.tsv
# (los timers no tienen ninguno -- el manifiesto no declara timer.d/ para
# ellos).
dropins_correctos() {
  case "$1" in
    jax-las-manos.service|jax-memory-worker.service|jax-memory-synthesis.service|jax-ejecutor-proxy.service)
      printf '/etc/systemd/system/%s.d/checkout-de-produccion.conf /etc/systemd/system/%s.d/cuenta-de-servicio.conf /etc/systemd/system/%s.d/z-pythonpath.conf\n' "$1" "$1" "$1"
      ;;
    jax-memory-worker.timer|jax-memory-synthesis.timer)
      printf '\n'
      ;;
    *)
      return 1
      ;;
  esac
}

if [ "${1:-}" = show ] && [[ " $* " == *" NeedDaemonReload "* ]] && dropins_reales="$(dropins_correctos "$ultimo")"; then
  if [ "$ultimo" = jax-las-manos.service ] && [ "$MODO" = falla ]; then
    echo "systemctl-falso: unidad simulada como inexistente" >&2
    exit 1
  fi
  printf 'Id=%s\n' "$ultimo"
  if [ "$ultimo" = jax-las-manos.service ] && [ "$MODO" = alias_cargado ]; then
    printf 'Names=%s otro-alias.service\n' "$ultimo"
  else
    printf 'Names=%s\n' "$ultimo"
  fi
  printf 'FragmentPath=/etc/systemd/system/%s\n' "$ultimo"
  if [ "$ultimo" = jax-las-manos.service ] && [ "$MODO" = incompleto ]; then
    printf 'DropInPaths=/etc/systemd/system/jax-las-manos.service.d/checkout-de-produccion.conf /etc/systemd/system/jax-las-manos.service.d/cuenta-de-servicio.conf\n'
  else
    printf 'DropInPaths=%s\n' "$dropins_reales"
  fi
  if [ "$ultimo" = jax-las-manos.service ] && [ "$MODO" = necesita_reload ]; then
    printf 'NeedDaemonReload=yes\n'
  else
    printf 'NeedDaemonReload=no\n'
  fi
  exit 0
fi
exec "$REAL" "$@"
