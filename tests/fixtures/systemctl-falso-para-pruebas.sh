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
# unidades del manifiesto -- no sólo jax-las-manos.service. Por eso este
# systemctl falso responde con datos CORRECTOS (calcados del manifiesto)
# para las 6 unidades siempre, y sólo desvía el comportamiento de
# jax-las-manos.service según $SYSTEMCTL_FALSO_MODO.
#
# Ronda 7, MINOR-D: este archivo NUNCA delega a un `systemctl` real
# (`exec "$REAL" "$@"` se quitó del todo) -- cualquier invocación que no
# sea EXACTAMENTE la consulta cargado esperada, o que pida una unidad que
# no esté en el manifiesto, sale con error. Un fixture que a veces sí
# ejecuta el systemctl real es un fixture que a veces SÍ depende del
# estado real de la máquina -- justo lo que la capa cargado de prueba
# (ronda 6) existe para evitar. Además, las unidades "conocidas" ya NO
# están copiadas a mano en un `case` -- se leen de
# `ops/manifiesto-arranque-instalado.tsv` con el MISMO criterio que usa
# el guion real (`enumerar_dropins_en_disco`/la lista `unidades` del
# script principal), para que este fixture no pueda desincronizarse del
# manifiesto en silencio.
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
#   sin_id_names: responde OK, con TODOS los drop-ins reales y
#     NeedDaemonReload=no, pero `Id=` y `Names=` VACÍOS -- como un
#     `systemctl show` roto que omite esas dos propiedades sin fallar del
#     todo (MINOR-C, ronda 7: con la ronda 6 sola, `"" != ""` no detecta
#     esto).
#   correcto: responde OK, con TODOS los drop-ins reales, Names=Id (sin
#     alias) y NeedDaemonReload=no -- el caso feliz, para probar que
#     activar la capa cargado bajo RAIZ_PRUEBA (MINOR-1, ronda 6) por sí
#     solo no hace fallar nada cuando todo coincide de verdad.
set -euo pipefail
MODO="${SYSTEMCTL_FALSO_MODO:-incompleto}"
ultimo="${*: -1}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
MANIFIESTO="$REPO/ops/manifiesto-arranque-instalado.tsv"

# Drop-ins correctos por unidad, LEÍDOS del manifiesto (ronda 7,
# MINOR-D) -- mismo criterio que el guion real: cualquier fila cuyo
# segundo campo empiece con "/etc/systemd/system/<unidad>.d/". Sale
# distinto de 0 (sin imprimir nada) si la unidad no está entre las que
# declara el manifiesto.
dropins_correctos() {
  local unidad="$1"
  if ! awk -F'\t' '$2 ~ /^\/etc\/systemd\/system\/[^\/]+\.(service|timer)$/ { n=split($2,a,"/"); print a[n] }' "$MANIFIESTO" \
      | grep -Fxq -- "$unidad"; then
    return 1
  fi
  awk -F'\t' -v pref="/etc/systemd/system/$unidad.d/" 'index($2, pref) == 1 { print $2 }' "$MANIFIESTO" \
    | sort -u | tr '\n' ' ' | sed 's/ *$//'
}

if [ "${1:-}" != show ] || [[ " $* " != *" NeedDaemonReload "* ]]; then
  echo "systemctl-falso: invocación inesperada (no es la consulta CARGADO esperada) -- nunca delega al systemctl real (MINOR-D, ronda 7): $*" >&2
  exit 1
fi

if ! dropins_reales="$(dropins_correctos "$ultimo")"; then
  echo "systemctl-falso: unidad desconocida (no está en $MANIFIESTO) -- nunca delega al systemctl real (MINOR-D, ronda 7): $ultimo" >&2
  exit 1
fi

if [ "$ultimo" = jax-las-manos.service ] && [ "$MODO" = falla ]; then
  echo "systemctl-falso: unidad simulada como inexistente" >&2
  exit 1
fi

if [ "$ultimo" = jax-las-manos.service ] && [ "$MODO" = sin_id_names ]; then
  printf 'Id=\n'
  printf 'Names=\n'
else
  printf 'Id=%s\n' "$ultimo"
  if [ "$ultimo" = jax-las-manos.service ] && [ "$MODO" = alias_cargado ]; then
    printf 'Names=%s otro-alias.service\n' "$ultimo"
  else
    printf 'Names=%s\n' "$ultimo"
  fi
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
