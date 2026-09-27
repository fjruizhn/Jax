#!/usr/bin/env bash
# ops/instalar-dropins-de-servicio.sh <unidad>.service.d|<ruta-absoluta> <REPO> [DESTDIR]
#
# Instala filas de ops/manifiesto-arranque-instalado.tsv -- derivado del
# manifiesto, NUNCA una lista de archivos aparte (ops/versionar-drop-ins,
# 2026-09-25). Dos modos, según la forma del primer argumento:
#
#   <unidad>.service.d   Todos los *.conf cuya ruta instalada empiece con
#                         /etc/systemd/system/<unidad>.service.d/ -- los
#                         drop-ins de esa unidad.
#   /ruta/absoluta        La UNA fila cuya ruta instalada sea exactamente
#                         esa -- hoy sólo se usa para
#                         /usr/local/sbin/jax-checkout-de-produccion-sano.sh
#                         (auditoría escalón 3, M5: el ExecStartPre de cada
#                         checkout-de-produccion.conf lo necesita para
#                         arrancar; un instalador que deje la unidad puesta
#                         sin este guion la deja sin poder arrancar nunca).
#
# Lo llaman config/systemd/install-memory-scope.sh y
# ops/ejecutor/instalar_registro_y_cerco.sh -- drop-ins (y el guion de
# sanidad) SIEMPRE ANTES que la unidad base, para no dejar nunca una base
# sola en /etc a mitad de una instalación interrumpida (M5). Este guion NO
# hace daemon-reload ni systemctl: sólo copia archivos.
#
# DESTDIR (default vacío = comportamiento de siempre, /etc real): para
# probar sin tocar el /etc real, se invoca con DESTDIR=<directorio temporal>
# y las rutas quedan bajo "$DESTDIR/etc/systemd/system/...". De los guiones
# de instalación de este árbol, SÓLO este acepta DESTDIR -- los otros dos
# tocan el sistema real de todos modos en otras líneas (nftables,
# /etc/jax/.env, systemctl restart de servicios reales), así que un DESTDIR
# ahí sería engañoso (auditoría escalón 3, M4).
set -euo pipefail
PATRON_O_RUTA="${1:?uso: instalar-dropins-de-servicio.sh <unidad>.service.d|/ruta/absoluta <REPO> [DESTDIR]}"
REPO="${2:?uso: instalar-dropins-de-servicio.sh <unidad>.service.d|/ruta/absoluta <REPO> [DESTDIR]}"
DESTDIR="${3:-}"

# MAJOR-2 (auditoría escalón 3, ronda 3): con DESTDIR vacío esto es una
# instalación REAL contra /etc -- exige lo mismo que install-memory-scope.sh/
# instalar_registro_y_cerco.sh: REPO es exactamente /srv/jax-prod/jax (no un
# checkout de trabajo), en master, y con el árbol limpio. Con DESTDIR puesto
# (pruebas) no se exige nada de esto -- un $REPO de mentira en un test no
# tiene por qué ser /srv/jax-prod/jax.
#
# /srv/jax-prod/jax es jaxsvc:jaxsvc: `git rev-parse`/`status` ahí dan
# "detected dubious ownership" (rc=128) tanto como fruiz como como root sin
# `-c safe.directory=...` -- se declara por invocación, nunca en la config
# global (mismo criterio que ops/sbin/jax-checkout-de-produccion-sano.sh).
# El chequeo de árbol limpio corre con `sudo` (root, con safe.directory):
# como fruiz, `git status --porcelain` da rc=0 pero con avisos
# "Permission denied" en stderr sobre subdirectorios de jaxsvc
# (las_manos/workspace, las_manos/repo, las_manos/missions) que fruiz no
# puede listar -- CUALQUIER stderr cuenta como fallo, no sólo lo que
# aparece en stdout.
RUTA_PRODUCCION=/srv/jax-prod/jax

# MINOR-1 (ronda 4): un DESTDIR que RESUELVE a "/" (p.ej. "", ".." desde
# la raíz, o un symlink que apunte ahí) tiene que tratarse EXACTAMENTE
# como "sin DESTDIR" -- si no, "$DESTDIR$instalada" más abajo termina
# escribiendo sobre el /etc real de todos modos, pero saltándose los
# frenos de esta sección porque `[ -z "$DESTDIR" ]` daba falso con algo
# que en los hechos apunta a la raíz. `realpath -m` no exige que exista
# (DESTDIR de prueba típicamente no existe todavía).
if [ -n "$DESTDIR" ]; then
  DESTDIR="$(realpath -m -- "$DESTDIR")"
  [ "$DESTDIR" = / ] && DESTDIR=""
fi

if [ -z "$DESTDIR" ]; then
  if [ "$REPO" != "$RUTA_PRODUCCION" ]; then
    echo "instalar-dropins-de-servicio: instalación real (sin DESTDIR) pero REPO=$REPO -- tiene que ser $RUTA_PRODUCCION. Abortando." >&2
    exit 1
  fi
  rama="$(git --no-optional-locks -c safe.directory="$RUTA_PRODUCCION" -C "$REPO" branch --show-current)"
  [ "$rama" = master ] || {
    echo "instalar-dropins-de-servicio: $REPO está en la rama '$rama', no en master. Abortando." >&2
    exit 1
  }
  archivo_err_status="$(mktemp)"
  sucio="$(sudo git --no-optional-locks -c safe.directory="$RUTA_PRODUCCION" -C "$REPO" status --porcelain 2>"$archivo_err_status")"
  err_status="$(cat -- "$archivo_err_status")"; rm -f -- "$archivo_err_status"
  if [ -n "$err_status" ]; then
    echo "instalar-dropins-de-servicio: git status avisó algo en $REPO (tratado como fallo): $err_status" >&2
    exit 1
  fi
  [ -z "$sucio" ] || {
    echo "instalar-dropins-de-servicio: $REPO tiene cambios sin comitear -- abortando:
$sucio" >&2
    exit 1
  }
fi

MANIFIESTO="$REPO/ops/manifiesto-arranque-instalado.tsv"
[ -f "$MANIFIESTO" ] || {
  echo "instalar-dropins-de-servicio: no se encontró el manifiesto: $MANIFIESTO" >&2
  exit 1
}

REPO_REAL="$(realpath "$REPO")"

case "$PATRON_O_RUTA" in
  /*) MODO=exacto; OBJETIVO="$PATRON_O_RUTA" ;;
  *) MODO=prefijo; OBJETIVO="/etc/systemd/system/$PATRON_O_RUTA/" ;;
esac

instalados=0

while IFS=$'\t' read -r repo_rel instalada || [ -n "$repo_rel" ]; do
  [ -z "$repo_rel" ] && continue

  case "$MODO" in
    exacto)
      [ "$instalada" = "$OBJETIVO" ] || continue
      ;;
    prefijo)
      case "$instalada" in
        "$OBJETIVO"*) ;;
        *) continue ;;
      esac
      ;;
  esac

  # m2 (auditoría escalón 3): el manifiesto es del propio repo, no un
  # insumo externo -- pero este guion corre con sudo, y una fila corrupta
  # (un merge mal resuelto, un edit a mano) no debería poder escribir fuera
  # de donde se espera sólo porque "es del repo". Defensa en profundidad,
  # no desconfianza del manifiesto de hoy.
  case "$repo_rel$instalada" in
    *..*)
      echo "instalar-dropins-de-servicio: fila con '..' -- rechazada: $repo_rel -> $instalada" >&2
      exit 1
      ;;
  esac

  base="$(basename -- "$instalada")"
  case "$MODO" in
    prefijo)
      if ! [[ "$base" =~ ^[A-Za-z0-9._-]+\.conf$ ]]; then
        echo "instalar-dropins-de-servicio: nombre de drop-in inválido -- rechazado: $base" >&2
        exit 1
      fi
      modo_archivo=0644
      ;;
    exacto)
      if ! [[ "$base" =~ ^[A-Za-z0-9._-]+\.sh$ ]]; then
        echo "instalar-dropins-de-servicio: nombre de guion inválido -- rechazado: $base" >&2
        exit 1
      fi
      modo_archivo=0755
      ;;
  esac

  repo_abs="$REPO/$repo_rel"
  [ -f "$repo_abs" ] || {
    echo "instalar-dropins-de-servicio: falta en el repo: $repo_abs" >&2
    exit 1
  }
  repo_abs_real="$(realpath -- "$repo_abs")"
  case "$repo_abs_real" in
    "$REPO_REAL"/*) ;;
    *)
      echo "instalar-dropins-de-servicio: $repo_abs se resuelve fuera del repo ($repo_abs_real) -- rechazado" >&2
      exit 1
      ;;
  esac

  sudo install -d -o root -g root -m 0755 "$DESTDIR$(dirname -- "$instalada")"
  sudo install -o root -g root -m "$modo_archivo" "$repo_abs" "$DESTDIR$instalada"
  instalados=$((instalados + 1))
done < "$MANIFIESTO"

if [ "$instalados" -eq 0 ]; then
  echo "instalar-dropins-de-servicio: ninguna fila coincide con '$PATRON_O_RUTA' -- revisar el argumento" >&2
  exit 1
fi

echo "instalar-dropins-de-servicio: $instalados archivo(s) de '$PATRON_O_RUTA' instalados en ${DESTDIR:-/} (sin daemon-reload -- lo hace quien invoca)"
