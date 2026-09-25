#!/usr/bin/env bash
# ops/verificar-arranque-instalado.sh
#
# Compara, archivo por archivo, la configuración de arranque de systemd que
# vive en el repo contra la que está instalada en /etc y /usr/local/sbin de
# esta máquina. Sólo lectura: no escribe en /etc, no corre systemctl salvo
# lo que ya haga el propio `cmp`/`test` (ninguno), no reinicia nada.
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
# Salida: 0 si repo e instalado coinciden byte a byte (y el .sh del
# manifiesto tiene el bit ejecutable en los dos lados). Distinto de 0, con
# CADA diferencia impresa, si no.
set -euo pipefail

REPO="$(git -C "$(dirname "$(readlink -f "$0")")" rev-parse --show-toplevel)"
MANIFIESTO="$REPO/ops/manifiesto-arranque-instalado.tsv"

[ -f "$MANIFIESTO" ] || {
  echo "verificar-arranque-instalado: no se encontró el manifiesto: $MANIFIESTO" >&2
  exit 1
}

fallo=0
revisados=0

while IFS=$'\t' read -r repo_rel instalada || [ -n "$repo_rel" ]; do
  [ -z "$repo_rel" ] && continue
  repo_abs="$REPO/$repo_rel"
  revisados=$((revisados + 1))

  if [ ! -f "$repo_abs" ]; then
    echo "FALTA en el repo: $repo_abs" >&2
    fallo=1
    continue
  fi
  if [ ! -f "$instalada" ]; then
    echo "FALTA instalado: $instalada (repo: $repo_abs)" >&2
    fallo=1
    continue
  fi

  if ! cmp -s "$repo_abs" "$instalada"; then
    detalle="$(cmp "$repo_abs" "$instalada" 2>&1 || true)"
    echo "DIFIERE: $repo_abs != $instalada -- $detalle" >&2
    fallo=1
  fi

  case "$repo_rel" in
    *.sh)
      if [ ! -x "$repo_abs" ]; then
        echo "SIN BIT EJECUTABLE en el repo: $repo_abs" >&2
        fallo=1
      fi
      if [ ! -x "$instalada" ]; then
        echo "SIN BIT EJECUTABLE instalado: $instalada" >&2
        fallo=1
      fi
      ;;
  esac
done < "$MANIFIESTO"

if [ "$revisados" -eq 0 ]; then
  echo "verificar-arranque-instalado: el manifiesto está vacío -- $MANIFIESTO" >&2
  exit 1
fi

if [ "$fallo" -ne 0 ]; then
  echo "verificar-arranque-instalado: hay diferencias entre el repo y lo instalado -- ver arriba" >&2
  exit 1
fi

echo "verificar-arranque-instalado: repo e instalado coinciden ($revisados archivos)"
