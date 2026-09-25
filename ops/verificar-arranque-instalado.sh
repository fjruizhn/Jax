#!/usr/bin/env bash
# ops/verificar-arranque-instalado.sh
#
# Compara, archivo por archivo, la configuración de arranque de systemd que
# vive en el repo contra la que está instalada en /etc y /usr/local/sbin de
# esta máquina. Sólo lectura: no escribe en /etc, no corre systemctl (ni
# siquiera `systemctl show`: la regla de esta rama es "systemctl cat" como
# máximo, y este guion se queda del lado seguro de esa regla usando listados
# de directorio en vez de preguntarle a systemd), no reinicia nada.
#
# Decisión de Fernando (2026-09-25, opción A): versionar en el repo el
# arranque de systemd que hoy sólo vivía en /etc de hall9000, y añadir esta
# verificación para que el drift entre repo e instalado se note solo.
#
# El manifiesto (repo -> instalado) vive en un archivo aparte,
# ops/manifiesto-arranque-instalado.tsv, para que este guion y
# tests/test_arranque_instalado.py lean EXACTAMENTE la misma lista -- dos
# listas que puedan divergir son peor que una sola que puede fallar.
#
# Salida: 0 si TODO lo de abajo se cumple. Distinto de 0, con CADA
# diferencia impresa, si no:
#   1. cada archivo del manifiesto existe en el repo y en lo instalado, y
#      son byte a byte idénticos (el guion .sh además con el bit ejecutable
#      en los dos lados);
#   2. lo instalado NUNCA es un symlink (auditoría escalón 3, m3: un symlink
#      podría apuntar a otra cosa sin que el cmp de arriba lo note si el
#      destino coincidiera hoy y cambiara mañana);
#   3. propietario root:root y modo 644 (755 para el .sh) en cada archivo instalado
#      (m3);
#   4. NINGÚN *.conf de más en los directorios `<unidad>.d/` de cada unidad
#      del manifiesto -- ni en las rutas específicas de la unidad
#      (/etc|/run|/usr/lib/systemd/system/<unidad>.d/) ni en las genéricas
#      que systemd aplica a TODO `.service` (`.../service.d/`) (auditoría
#      escalón 3, M2: el guion anterior sólo miraba "¿lo que el manifiesto
#      pide está instalado?", nunca "¿hay algo instalado que el manifiesto
#      no sabe que existe?").
set -euo pipefail

REPO="$(git -C "$(dirname "$(readlink -f "$0")")" rev-parse --show-toplevel)"
MANIFIESTO="$REPO/ops/manifiesto-arranque-instalado.tsv"

# RAIZ_PRUEBA: vacío por defecto -- SIN esto, ninguna línea de este guion
# cambia, sigue leyendo /etc, /run, /usr/lib reales. Existe SÓLO para poder
# ejercitar la lógica de este guion (symlink, dueño/modo, drop-ins de más)
# contra un árbol de prueba bajo /tmp, sin escribir NUNCA en el /etc real --
# la regla de esta rama es "producción, sólo lectura". No se documenta en el
# manifiesto ni en el runbook de instalación: nadie lo usa para instalar,
# sólo para probar este guion antes de confiar en él.
RAIZ_PRUEBA="${RAIZ_PRUEBA:-}"

[ -f "$MANIFIESTO" ] || {
  echo "verificar-arranque-instalado: no se encontró el manifiesto: $MANIFIESTO" >&2
  exit 1
}

fallo=0
revisados=0

verificar_propietario_y_modo() {
  # $1 = ruta real a stat-ear (con RAIZ_PRUEBA si aplica), $2 = ruta a
  # mostrar en los mensajes (siempre la de producción, nunca la de prueba),
  # $3 = "1" si es el .sh (755), si no 644.
  local real="$1" mostrar="$2" es_sh="$3" propietario grupo modo esperado
  read -r propietario grupo modo < <(stat -c '%U %G %a' "$real" 2>/dev/null) || {
    echo "NO SE PUDO LEER propietario/modo: $mostrar" >&2
    fallo=1
    return
  }
  if [ "$propietario" != root ] || [ "$grupo" != root ]; then
    echo "PROPIETARIO INCORRECTO: $mostrar es $propietario:$grupo -- se esperaba root:root" >&2
    fallo=1
  fi
  esperado=644
  [ "$es_sh" = 1 ] && esperado=755
  if [ "$modo" != "$esperado" ]; then
    echo "MODO INCORRECTO: $mostrar es $modo -- se esperaba $esperado" >&2
    fallo=1
  fi
}

while IFS=$'\t' read -r repo_rel instalada || [ -n "$repo_rel" ]; do
  [ -z "$repo_rel" ] && continue
  repo_abs="$REPO/$repo_rel"
  instalada_real="$RAIZ_PRUEBA$instalada"
  revisados=$((revisados + 1))

  if [ ! -f "$repo_abs" ]; then
    echo "FALTA en el repo: $repo_abs" >&2
    fallo=1
    continue
  fi

  if [ -L "$instalada_real" ]; then
    echo "SYMLINK instalado (se esperaba archivo regular): $instalada -> $(readlink "$instalada_real")" >&2
    fallo=1
    continue
  fi

  if [ ! -f "$instalada_real" ]; then
    echo "FALTA instalado: $instalada (repo: $repo_abs)" >&2
    fallo=1
    continue
  fi

  if ! cmp -s "$repo_abs" "$instalada_real"; then
    detalle="$(cmp "$repo_abs" "$instalada_real" 2>&1 || true)"
    echo "DIFIERE: $repo_abs != $instalada -- $detalle" >&2
    fallo=1
  fi

  case "$repo_rel" in
    *.sh)
      if [ ! -x "$repo_abs" ]; then
        echo "SIN BIT EJECUTABLE en el repo: $repo_abs" >&2
        fallo=1
      fi
      verificar_propietario_y_modo "$instalada_real" "$instalada" 1
      ;;
    *)
      verificar_propietario_y_modo "$instalada_real" "$instalada" 0
      ;;
  esac
done < "$MANIFIESTO"

if [ "$revisados" -eq 0 ]; then
  echo "verificar-arranque-instalado: el manifiesto está vacío -- $MANIFIESTO" >&2
  exit 1
fi

# --- Sin drop-ins de más (M2) ------------------------------------------
# Unidades del manifiesto: las filas cuya ruta instalada es la unidad base
# misma (/etc/systemd/system/<unidad>), nunca un .d/*.conf.
unidades="$(awk -F'\t' '$2 ~ /^\/etc\/systemd\/system\/[^\/]+\.(service|timer)$/ { n=split($2,a,"/"); print a[n] }' "$MANIFIESTO" | sort -u)"

while IFS= read -r unidad; do
  [ -z "$unidad" ] && continue

  esperados="$(awk -F'\t' -v pref="/etc/systemd/system/$unidad.d/" \
    'index($2, pref) == 1 { n=split($2,a,"/"); print a[n] }' "$MANIFIESTO" | sort -u)"

  dirs=("$RAIZ_PRUEBA/etc/systemd/system/$unidad.d" "$RAIZ_PRUEBA/run/systemd/system/$unidad.d" "$RAIZ_PRUEBA/usr/lib/systemd/system/$unidad.d")
  case "$unidad" in
    *.service)
      dirs+=("$RAIZ_PRUEBA/etc/systemd/system/service.d" "$RAIZ_PRUEBA/run/systemd/system/service.d" "$RAIZ_PRUEBA/usr/lib/systemd/system/service.d")
      ;;
  esac

  reales=""
  for dir in "${dirs[@]}"; do
    [ -d "$dir" ] || continue
    for conf in "$dir"/*.conf; do
      [ -e "$conf" ] || continue
      reales="$reales
$(basename "$conf")"
    done
  done
  reales="$(printf '%s\n' "$reales" | sed '/^$/d' | sort -u)"

  sobran="$(comm -13 <(printf '%s\n' "$esperados") <(printf '%s\n' "$reales"))"
  if [ -n "$sobran" ]; then
    echo "DROP-IN DE MÁS (no está en el manifiesto) para $unidad:" >&2
    while IFS= read -r extra; do
      echo "  $extra" >&2
    done <<< "$sobran"
    fallo=1
  fi
done <<< "$unidades"

if [ "$fallo" -ne 0 ]; then
  echo "verificar-arranque-instalado: hay diferencias entre el repo y lo instalado -- ver arriba" >&2
  exit 1
fi

echo "verificar-arranque-instalado: repo e instalado coinciden ($revisados archivos)"
