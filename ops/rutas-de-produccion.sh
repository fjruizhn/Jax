#!/usr/bin/env bash
# ops/rutas-de-produccion.sh --verificar
#
# Solo lectura: no escribe en /etc, /srv, /var ni /home/fruiz/jax, no arranca
# ni detiene ningun servicio, no reinicia nada. Dos preguntas sobre las
# claves de RUTA de /etc/jax/.env (las que terminan en _PATH, _DIR, _BASE,
# _BIN, _FILE o _LOG -- el mismo criterio con el que este guion decide que
# es seguro de imprimir: nunca un valor que no sea una ruta):
#
#   (A) alguna sigue apuntando al checkout de TRABAJO de un agente
#       (/home/fruiz/jax/) o al lanzador personal de fruiz (/home/fruiz/.local/)
#       en vez de a /srv/jax-prod (codigo) o /srv/jax-data (datos)? Si si,
#       FALLA listando cada clave=ruta.
#   (B) para las claves que este cambio SI mueve (JAX_CONFIG_PATH,
#       JAX_AUDIT_LOG_PATH, JAX_REPO_BASE): la ruta existe y jaxsvc -- la
#       cuenta de servicio real de jax-las-manos/jax-platform/etc, ver
#       EnvironmentFile=/etc/jax/.env en sus unidades -- la puede leer? Al
#       log de auditoria y a REPO_BASE/documents, ademas, se les exige que
#       jaxsvc pueda ESCRIBIR (AuditLog.log_execution() escribe en el primero,
#       jacobs/executor.py._persist_step_to_repo escribe en el segundo).
#
# EXCEPCIONES documentadas, con motivo -- nunca por omision (mismo criterio
# que policy/tests/test_archivos_de_test_wireados_en_ci.py):
#
#   JAX_MISSIONS_DIR, JAX_BIN -- SI tienen un consumidor real (verificado
#   2026-09-25, grep -rn en /srv/jax-prod/jax-platform/backend/api/command.py:
#   `MISSIONS_DIR = ruta_absoluta_requerida("JAX_MISSIONS_DIR")`,
#   `JAX_BIN = ruta_absoluta_requerida("JAX_BIN")`, invocado como
#   `$JAX_BIN --task <mission_file>`). No se mueven en este cambio: el
#   lanzador de JAX_BIN (/home/fruiz/.local/bin/jax) hace `cd $HOME/jax` y usa
#   SU PROPIO .venv -- moverlo decide DESDE QUE CHECKOUT corre cada mision
#   (el codigo que se ejecuta), una decision de arquitectura mayor y mas
#   riesgosa que mover un archivo de config o un log, y no es la que este
#   cambio (rutas de config/auditoria/documentos) vino a tomar. Sigue
#   apuntando a /home/fruiz/jax/ y /home/fruiz/.local/ HOY, a proposito;
#   queda como pendiente para que Fernando decida el destino. Ver el informe
#   de esta tarea (rama ops/rutas-de-produccion).
#
#   JAX_PIPELINE_DETAIL_PATH, PIPELINE_DETAIL_PATH -- terminan en _PATH pero
#   NO son una ruta de sistema de archivos: son una plantilla de ruta de
#   frontend (`/historial/{pipeline_id}`). `test -e` sobre eso no tiene
#   sentido; se excluyen de las dos fases.
#
# Salida: 0 si (A) no encuentra violaciones Y (B) pasa para las tres claves
# en alcance. Distinto de cero, con cada problema listado, si no.
set -euo pipefail

if [ "${1:-}" != "--verificar" ]; then
  echo "uso: $0 --verificar" >&2
  exit 2
fi

ENV_FILE="/etc/jax/.env"

# Claves que este cambio SI mueve -- Fase B corre solo sobre estas (ver
# comentario de arriba: JAX_MISSIONS_DIR/JAX_BIN quedan fuera a proposito).
KEYS_EN_ALCANCE="JAX_CONFIG_PATH JAX_AUDIT_LOG_PATH JAX_REPO_BASE"

# Claves excluidas de LAS DOS fases, con motivo (ver comentario de arriba).
es_excepcion() {
  case "$1" in
    JAX_MISSIONS_DIR|JAX_BIN|JAX_PIPELINE_DETAIL_PATH|PIPELINE_DETAIL_PATH) return 0 ;;
    *) return 1 ;;
  esac
}

# Es una clave de RUTA? Mismo criterio que la instruccion de esta tarea:
# solo *_PATH, *_DIR, *_BASE, *_BIN, *_FILE, *_LOG se leen o imprimen --
# cualquier otra clave de /etc/jax/.env (credenciales, tokens, hosts) nunca
# se toca ni se muestra.
es_clave_de_ruta() {
  case "$1" in
    *_PATH|*_DIR|*_BASE|*_BIN|*_FILE|*_LOG) return 0 ;;
    *) return 1 ;;
  esac
}

if ! sudo -n test -r "$ENV_FILE" 2>/dev/null; then
  echo "rutas-de-produccion: sudo -n no puede leer $ENV_FILE (falta la credencial cacheada de sudo -n?)" >&2
  exit 1
fi

ENV_CONTENIDO="$(sudo -n cat "$ENV_FILE")"

violaciones_home_fruiz=""
fase_b_fallas=""
revisadas=0

while IFS='=' read -r clave valor; do
  [ -z "$clave" ] && continue
  case "$clave" in \#*) continue ;; esac
  es_clave_de_ruta "$clave" || continue
  [ -z "$valor" ] && continue
  es_excepcion "$clave" && continue

  revisadas=$((revisadas + 1))

  # --- Fase A: sigue bajo el checkout de trabajo o el HOME de fruiz? ---
  case "$valor" in
    /home/fruiz/jax/*|/home/fruiz/.local/*)
      violaciones_home_fruiz="${violaciones_home_fruiz}${clave}=${valor}
"
      ;;
  esac

  # --- Fase B: solo para las claves que este cambio mueve ---
  case " $KEYS_EN_ALCANCE " in
    *" $clave "*) ;;
    *) continue ;;
  esac

  if ! sudo -n -u jaxsvc test -r "$valor" 2>/dev/null; then
    fase_b_fallas="${fase_b_fallas}${clave}=${valor}: jaxsvc no puede LEER esta ruta (existe? permiso?)
"
  fi

  case "$clave" in
    JAX_AUDIT_LOG_PATH)
      if ! sudo -n -u jaxsvc test -w "$valor" 2>/dev/null; then
        fase_b_fallas="${fase_b_fallas}${clave}=${valor}: jaxsvc no puede ESCRIBIR el log de auditoria
"
      fi
      ;;
    JAX_REPO_BASE)
      documentos="${valor%/}/documents"
      if ! sudo -n -u jaxsvc test -w "$documentos" 2>/dev/null; then
        fase_b_fallas="${fase_b_fallas}${clave}: jaxsvc no puede ESCRIBIR en ${documentos}
"
      fi
      ;;
  esac
done <<EOF_ENV
$ENV_CONTENIDO
EOF_ENV

if [ "$revisadas" -eq 0 ]; then
  echo "rutas-de-produccion: no se encontro NINGUNA clave de ruta en $ENV_FILE -- cambio el formato del archivo?" >&2
  exit 1
fi

fallo=0

if [ -n "$violaciones_home_fruiz" ]; then
  echo "FASE A -- claves de ruta que siguen apuntando al checkout de trabajo o al HOME de fruiz:" >&2
  printf '%s' "$violaciones_home_fruiz" >&2
  fallo=1
fi

if [ -n "$fase_b_fallas" ]; then
  echo "FASE B -- jaxsvc no tiene el acceso que necesita:" >&2
  printf '%s' "$fase_b_fallas" >&2
  fallo=1
fi

if [ "$fallo" -ne 0 ]; then
  echo "rutas-de-produccion: hay problemas -- ver arriba" >&2
  exit 1
fi

echo "rutas-de-produccion: todas las rutas de produccion en alcance estan fuera de /home/fruiz y jaxsvc tiene el acceso que necesita ($revisadas claves de ruta revisadas)"
