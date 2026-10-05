#!/usr/bin/env bash
# Verifica, sin cambiar estado, que TODOS los timers B9 versionados estén
# habilitados, activos, esperando su próximo disparo y con uno programado.
#
# Es deliberadamente un contrato SEPARADO de verificar-arranque-instalado:
# aquel comprueba que repo, disco y carga de systemd coincidan; éste comprueba
# que los trabajos periódicos estén realmente activados. No contiene
# `enable`, `start`, `restart` ni `daemon-reload`; observar un timer apagado
# debe dar rojo, no corregirlo silenciosamente.
set -euo pipefail
export LC_ALL=C

REPO="$(git -C "$(dirname "$(readlink -f "$0")")" rev-parse --show-toplevel)"
MANIFIESTO="$REPO/ops/manifiesto-arranque-instalado.tsv"
[ -f "$MANIFIESTO" ] || { echo "verificar-activacion-timers-b9: falta $MANIFIESTO -- fallo cerrado" >&2; exit 1; }

# El doble de pruebas sólo puede entrar con RAIZ_PRUEBA explícita (nunca /)
# y tiene que vivir DENTRO de ella. Producción ignora por completo cualquier
# variable externa y resuelve systemctl por ruta absoluta confiable.
RAIZ_PRUEBA="${1:-}"
if [ -n "$RAIZ_PRUEBA" ]; then
  RAIZ_PRUEBA="$(realpath -m -- "$RAIZ_PRUEBA")"
  [ "$RAIZ_PRUEBA" = / ] && RAIZ_PRUEBA=""
fi
if [ -n "$RAIZ_PRUEBA" ]; then
  SYSTEMCTL_CMD="${JAX_TIMER_CONTRACT_SYSTEMCTL:-}"
  [ -n "$SYSTEMCTL_CMD" ] && [ -x "$SYSTEMCTL_CMD" ] || {
    echo "verificar-activacion-timers-b9: doble de systemctl inválido -- fallo cerrado" >&2; exit 1; }
  SYSTEMCTL_REAL="$(realpath -- "$SYSTEMCTL_CMD")"
  case "$SYSTEMCTL_REAL" in "$RAIZ_PRUEBA"/*) SYSTEMCTL_CMD="$SYSTEMCTL_REAL" ;; *)
    echo "verificar-activacion-timers-b9: doble fuera de RAIZ_PRUEBA -- fallo cerrado" >&2; exit 1 ;; esac
else
  SYSTEMCTL_CMD=""
  for candidata in /usr/bin/systemctl /bin/systemctl; do
    [ -x "$candidata" ] || continue
    read -r propietario grupo modo < <(stat -c '%U %G %a' "$candidata" 2>/dev/null) || continue
    [ "$propietario" = root ] && [ "$grupo" = root ] || continue
    modo_octal="0$modo"
    bits_grupo=$(( (modo_octal / 8) % 8 )); bits_otros=$(( modo_octal % 8 ))
    (( (bits_grupo & 2) == 0 && (bits_otros & 2) == 0 )) || continue
    SYSTEMCTL_CMD="$candidata"
    break
  done
  [ -n "$SYSTEMCTL_CMD" ] || { echo "verificar-activacion-timers-b9: no hay systemctl absoluto root:root y no escribible por grupo/otros -- fallo cerrado" >&2; exit 1; }
fi

mapfile -t timers < <(awk -F'\t' '$2 ~ /^\/etc\/systemd\/system\/jax-memory-[^\/]+\.timer$/ { n=split($2,a,"/"); print a[n] }' "$MANIFIESTO" | sort -u)
[ "${#timers[@]}" -eq 5 ] || {
  echo "verificar-activacion-timers-b9: se esperaban exactamente 5 timers B9 en el manifiesto, hay ${#timers[@]} -- fallo cerrado" >&2
  exit 1
}

fallo=0
for timer in "${timers[@]}"; do
  if ! salida="$("$SYSTEMCTL_CMD" show -p UnitFileState -p ActiveState -p SubState -p NextElapseUSecRealtime -p NextElapseUSecMonotonic "$timer" 2>&1)"; then
    echo "SYSTEMCTL SHOW FALLÓ para $timer: $salida" >&2
    fallo=1
    continue
  fi
  unit_file_state="" active_state="" sub_state="" next_elapse="" next_monotonic=""
  while IFS= read -r linea; do
    clave="${linea%%=*}"; valor="${linea#*=}"
    case "$clave" in
      UnitFileState) unit_file_state="$valor" ;;
      ActiveState) active_state="$valor" ;;
      SubState) sub_state="$valor" ;;
      NextElapseUSecRealtime) next_elapse="$valor" ;;
      NextElapseUSecMonotonic) next_monotonic="$valor" ;;
    esac
  done <<< "$salida"

  # Cada propiedad es obligatoria y se valida por valor exacto: una salida
  # truncada, "n/a" o un estado nuevo/desconocido es roja por diseño.
  #
  # "Sin próximo disparo" (B1, auditoría de #356): el systemd real imprime
  # NextElapseUSecRealtime= (vacío) y NextElapseUSecMonotonic=infinity para un timer
  # muerto -- medido en hall9000. Un timer de CALENDARIO activo muestra Realtime con
  # fecha y Monotonic=0, así que cuenta Realtime; un timer monotónico activo
  # (OnUnitActiveSec/OnBootSec) tiene Realtime vacío y un Monotonic real. Por eso solo
  # falta el disparo si Realtime está vacío o n/a Y el monotónico es "", n/a, infinity o 0.
  # Cada condición va en su propia variable: la forma anterior mezclaba `||` y `&&` sin
  # agrupar y se evaluaba como (A||B||C||D) && E.
  sin_realtime=0
  { [ -z "$next_elapse" ] || [ "$next_elapse" = n/a ]; } && sin_realtime=1
  sin_monotonico=0
  case "$next_monotonic" in "" | n/a | infinity | 0) sin_monotonico=1 ;; esac
  sin_disparo=0
  if [ "$sin_realtime" -eq 1 ] && [ "$sin_monotonico" -eq 1 ]; then sin_disparo=1; fi
  if [ "$unit_file_state" != enabled ] || [ "$active_state" != active ] || [ "$sub_state" != waiting ] || [ "$sin_disparo" -eq 1 ]; then
    echo "TIMER B9 NO ACTIVADO: $timer UnitFileState=${unit_file_state:-<vacío>} ActiveState=${active_state:-<vacío>} SubState=${sub_state:-<vacío>} NextElapseUSecRealtime=${next_elapse:-<vacío>} NextElapseUSecMonotonic=${next_monotonic:-<vacío>} -- se exige enabled/active/waiting/próximo disparo" >&2
    fallo=1
  fi
done

[ "$fallo" -eq 0 ] || exit 1
echo "verificar-activacion-timers-b9: los ${#timers[@]} timers B9 están enabled/active/waiting con próximo disparo"
