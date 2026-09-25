#!/usr/bin/env bash
# ops/permisos-proyectos.sh [--verificar|--aplicar] [RAIZ] — spec
# docs/superpowers/specs/2026-09-22-proyectos-y-selector-design.md §5: RAIZ/proyectos/ tiene
# que ser escribible por jaxsvc (LAS MANOS, jax-platform) y por fruiz (scripts/procesar_archivos.py),
# con herencia para todo lo que se cree después.
#
# Diseño (fijado, no se cambia acá): grupo jaxsvc en todo el árbol, bit setgid en directorios,
# ACL POSIX de acceso Y por defecto (u:fruiz:rwX, g:jaxsvc:rwX, m::rwx) en directorios. El dueño
# se queda fruiz -- no se toca (los hardlinks de fuente/ se publican con os.link).
#
# --verificar (por defecto): solo lectura, sale 0 si todo proyectos/ cumple (grupo, setgid en
# dirs, ACL de acceso y por defecto en dirs; grupo y ACL de acceso en archivos), 1 e imprime
# cada ruta que no cumple.
# --aplicar: antes de tocar nada, respalda el árbol completo (getfacl -R -p, restaurable con
# `sudo setfacl --restore=<archivo>`; verificado 2026-09-25 en hall9000 que ese restore repone
# owner/group/ACL/setgid de punta a punta) en ~/respaldos-permisos/proyectos-<fecha>.acl. Aplica
# y corre --verificar al final. Idempotente.
#
# ORDEN DE LAS OPERACIONES -- NO reordenar. Verificado empíricamente en hall9000 (2026-09-25,
# principio I: el que supone se equivoca): fruiz NO es miembro del grupo jaxsvc, así que
# CUALQUIER llamada de setfacl(1) corrida como fruiz sobre un directorio cuyo grupo es jaxsvc
# limpia en silencio el bit setgid que ya estuviera puesto (mismo mecanismo del kernel que
# limpia S_ISGID en chmod(2) cuando el llamador no pertenece al grupo del archivo y no tiene
# CAP_FSETID -- Linux lo aplica también a directorios, no solo a archivos regulares como sugiere
# la letra de chmod(2)). Por eso el chmod g+s va SIEMPRE al final, después de los dos setfacl, y
# SIEMPRE con sudo (root no pertenece a ningún grupo en particular, así que a root nunca se le
# limpia el bit). chgrp también necesita sudo: fruiz no puede chgrp a un grupo del que no es
# miembro (CAP_CHOWN). setfacl en cambio NO necesita sudo: el dueño (fruiz) puede fijar ACL sin
# pertenecer al grupo.
#
# ADVERTENCIA DE HOST -- ver docs/runbooks/workspace-proyectos.md. El /usr/bin/mkdir por
# defecto de este host (uutils-coreutils 0.8.0, paquete coreutils-from-uutils de Ubuntu) tiene
# un defecto verificado el 2026-09-25: al crear un directorio NUEVO dentro de un padre con ACL
# por defecto de entradas nombradas, agrega bits setuid y/o sticky espurios (proporcional a la
# cantidad de entradas nombradas) que GNU mkdir (/usr/bin/gnumkdir) y Python (os.mkdir) NO
# agregan sobre el mismo padre. El sticky espurio le impide a fruiz borrar un archivo de jaxsvc
# (y viceversa) dentro de ese directorio -- exactamente lo que este esquema de permisos existe
# para permitir. Los servicios de jaxsvc (LAS MANOS, jax-platform) crean directorios con
# Python, no con `mkdir`, así que no les pega; cualquier script de shell que use `mkdir` a
# secas bajo proyectos/ sí queda expuesto. Este guion no crea directorios nuevos, así que no lo
# sufre -- queda documentado para quien sí lo haga.
set -euo pipefail

MODO="--verificar"
RAIZ_ARG=""
RAIZ_DADA=0
for arg in "$@"; do
  case "$arg" in
    --verificar|--aplicar) MODO="$arg" ;;
    --*) echo "opción desconocida: $arg" >&2; exit 2 ;;
    *) RAIZ_ARG="$arg"; RAIZ_DADA=1 ;;
  esac
done

_raiz_por_defecto() {
  local desde_env
  desde_env="$(sudo -n grep '^JAX_WORKSPACE_DIR=' /etc/jax/.env 2>/dev/null | head -1 | cut -d= -f2-)" || desde_env=""
  if [[ -n "$desde_env" ]]; then
    printf '%s\n' "$desde_env"
  else
    printf '%s\n' "/home/fruiz/jax-workspace"
  fi
}

# RAIZ_DADA distingue "no se dio RAIZ posicional" (usa el valor por defecto) de "se dio una
# RAIZ vacía a propósito" (""), que la validación de abajo rechaza -- un `${RAIZ_ARG:-...}`
# a secas no puede distinguir los dos casos porque para bash una var vacía y una sin dar son
# lo mismo.
if [[ "$RAIZ_DADA" == 1 ]]; then
  RAIZ="$RAIZ_ARG"
else
  RAIZ="$(_raiz_por_defecto)"
fi

if [[ -z "$RAIZ" || "$RAIZ" == "/" ]]; then
  echo "RAIZ inválida: '$RAIZ'" >&2
  exit 2
fi

PROYECTOS="$RAIZ/proyectos"
if [[ ! -d "$PROYECTOS" ]]; then
  echo "RAIZ inválida: no existe $PROYECTOS (una ruta sin proyectos/)" >&2
  exit 2
fi

USUARIO="fruiz"
GRUPO="jaxsvc"

# --- verificación (solo lectura) --------------------------------------------------------
# Reglas exactas de "cumple" -- las mismas que deja --aplicar:
#   directorios: grupo=$GRUPO, setgid puesto, ACL de acceso user:$USUARIO:rwx y
#                group:$GRUPO:rwx, ACL por defecto default:user:$USUARIO:rwx y
#                default:group:$GRUPO:rwx.
#   archivos:    grupo=$GRUPO, ACL de acceso user:$USUARIO:rw* y group:$GRUPO:rw*
#                (sin exigir x: X solo la agrega si el archivo ya la tenía).
_verificar() {
  local fallos=0
  local ruta grupo_real acl

  while IFS= read -r -d '' ruta; do
    grupo_real="$(stat -c '%G' "$ruta")"
    if [[ "$grupo_real" != "$GRUPO" ]]; then
      echo "NO CUMPLE (grupo=$grupo_real, esperado=$GRUPO): $ruta"
      fallos=1
      continue
    fi

    acl="$(getfacl -p "$ruta" 2>/dev/null)"

    if [[ -d "$ruta" ]]; then
      if [[ ! -g "$ruta" ]]; then
        echo "NO CUMPLE (falta setgid): $ruta"
        fallos=1
        continue
      fi
      if ! grep -qE "^user:${USUARIO}:rwx$" <<<"$acl" || ! grep -qE "^group:${GRUPO}:rwx$" <<<"$acl"; then
        echo "NO CUMPLE (falta ACL de acceso rwx para ${USUARIO}/${GRUPO}): $ruta"
        fallos=1
        continue
      fi
      if ! grep -qE "^default:user:${USUARIO}:rwx$" <<<"$acl" || ! grep -qE "^default:group:${GRUPO}:rwx$" <<<"$acl"; then
        echo "NO CUMPLE (falta ACL por defecto rwx para ${USUARIO}/${GRUPO}): $ruta"
        fallos=1
        continue
      fi
    else
      if ! grep -qE "^user:${USUARIO}:rw" <<<"$acl" || ! grep -qE "^group:${GRUPO}:rw" <<<"$acl"; then
        echo "NO CUMPLE (falta ACL de acceso rw para ${USUARIO}/${GRUPO}): $ruta"
        fallos=1
        continue
      fi
    fi
  done < <(find "$PROYECTOS" \( -type d -o -type f \) -print0)

  return "$fallos"
}

# --- aplicación --------------------------------------------------------------------------
_aplicar() {
  command -v setfacl >/dev/null || { echo "falta setfacl" >&2; exit 2; }
  sudo -n true || { echo "sudo -n no disponible -- se necesita para chgrp/chmod g+s" >&2; exit 2; }

  local dir_respaldos="$HOME/respaldos-permisos"
  mkdir -p "$dir_respaldos"
  local archivo_respaldo="$dir_respaldos/proyectos-$(date +%Y%m%d-%H%M%S).acl"
  getfacl -R -p "$PROYECTOS" > "$archivo_respaldo"
  echo "Respaldo: $archivo_respaldo (revertir con: sudo setfacl --restore=\"$archivo_respaldo\")"

  # Orden fijo -- ver el comentario largo de arriba. NO reordenar.
  sudo -n chgrp -R "$GRUPO" "$PROYECTOS"
  setfacl -R -m "u:${USUARIO}:rwX,g:${GRUPO}:rwX,m::rwx" "$PROYECTOS"
  setfacl -R -d -m "u:${USUARIO}:rwX,g:${GRUPO}:rwX,m::rwx" "$PROYECTOS"
  sudo -n find "$PROYECTOS" -type d -exec chmod g+s {} +

  _verificar
}

case "$MODO" in
  --verificar)
    if _verificar; then
      echo "OK: $PROYECTOS cumple (grupo, setgid, ACL de acceso y por defecto)."
      exit 0
    else
      exit 1
    fi
    ;;
  --aplicar)
    if _aplicar; then
      echo "OK: $PROYECTOS aplicado y verificado."
      exit 0
    else
      echo "--aplicar terminó pero --verificar final encontró fallos (ver arriba)." >&2
      exit 1
    fi
    ;;
esac
