#!/usr/bin/env bash
# ops/ejecutor/instalar_contexto.sh [--rama-aprobada|--comprobar-frescura] — instala el
# CLAUDE.md GENERADO (spec §6.1: nunca a mano) y las skills declaradas (cerebros.toml
# `skills`) en JAX_EJECUTOR_LIB/contexto/, de solo lectura para la cuenta del Ejecutor,
# junto con el MANIFIESTO (sha256 de cada archivo) que el arranque de cada misión
# compara (M-2, ronda 6: `jax.ejecutor.contratos.arranque.verificar_contexto` ya NO
# regenera desde la constitución en cada misión -- lee este manifiesto, instalado por
# ESTE script, root, de sólo lectura para axioma). La jaula
# (jax.ejecutor.contratos.cuenta_axioma._jaula) monta CLAUDE.md y skills ro sobre
# "$HOME/.claude/CLAUDE.md" y "$HOME/.claude/skills".
#
# --comprobar-frescura: NO instala nada. Compara el manifiesto YA INSTALADO contra lo
# que el generador produciría AHORA MISMO y avisa si difieren -- INFORMATIVO, nunca
# bloquea ninguna misión (esa es la integridad, que sí bloquea, y la hace el arranque
# de cada misión contra el manifiesto). Pensado para engancharse a un timer periódico,
# no para correr en el camino de una misión.
#
# Corre como fruiz desde el checkout de producción (/srv/jax-prod/jax en master --
# JAX_REPO_PATH en /etc/jax/.env, NO /home/fruiz/jax: ese es el checkout de trabajo
# de Codex desde 2026-09-20, ver jax-checkout-de-produccion-separado); usa sudo para
# lo que es de root. Idempotente. Lee JAX_EJECUTOR_LIB del entorno
# (set -a; . <(sudo -n cat /etc/jax/.env)).
set -euo pipefail
: "${JAX_EJECUTOR_LIB:?}"
REPO="$(git -C "$(dirname "$(readlink -f "$0")")" rev-parse --show-toplevel)"

if [ "${1:-}" = --comprobar-frescura ]; then
  DESTINO="$JAX_EJECUTOR_LIB/contexto"
  MANIFIESTO_ACTUAL="$(cd "$REPO" && PYTHONDONTWRITEBYTECODE=1 python3 -m jax.ejecutor.contratos.contexto --manifiesto)"
  if ! sudo test -e "$DESTINO/MANIFIESTO.sha256.json"; then
    echo "frescura_ok=false motivo=\"manifiesto_no_instalado\""
    exit 0
  fi
  MANIFIESTO_INSTALADO="$(sudo cat "$DESTINO/MANIFIESTO.sha256.json")"
  if [ "$MANIFIESTO_ACTUAL" = "$MANIFIESTO_INSTALADO" ]; then
    echo "frescura_ok=true"
  else
    echo "frescura_ok=false motivo=\"desactualizado\""
  fi
  exit 0
fi

# Mismo candado que instalar_contratos.sh: producción se instala desde master;
# --rama-aprobada (sólo con GO/autonomía de Fernando) permite instalar desde la
# rama del checkout y lo deja dicho en la salida.
RAMA="$(git -C "$REPO" branch --show-current)"
test "$RAMA" = master || test "${1:-}" = --rama-aprobada

ETAPA="$(mktemp -d)"
trap 'rm -rf "$ETAPA"' EXIT
( cd "$REPO" && PYTHONDONTWRITEBYTECODE=1 python3 -m jax.ejecutor.contratos.contexto "$ETAPA" )

DESTINO="$JAX_EJECUTOR_LIB/contexto"

echo "--- diff CLAUDE.md (instalado -> nuevo) ---"
# El mensaje "sin CLAUDE.md instalado todavía" es SÓLO para cuando el archivo no
# existe -- antes salía igual cuando SÍ existía pero difería, porque `diff -u` sale
# con 1 al encontrar diferencias y el `||` de una sola cadena no distinguía los dos
# casos (arreglado 2026-09-22, auditoría adversarial, hallazgo MINOR). `|| true`
# porque el script corre con `set -e` y un diff con diferencias (rc=1) es el camino
# ESPERADO, no un error.
if sudo test -e "$DESTINO/CLAUDE.md"; then
  diff -u <(sudo cat "$DESTINO/CLAUDE.md") "$ETAPA/CLAUDE.md" || true
else
  echo "(sin CLAUDE.md instalado todavía)"
fi
echo "--- fin diff ---"

sudo install -d -o root -g root -m 0755 "$DESTINO"
sudo install -o root -g root -m 0644 "$ETAPA/CLAUDE.md" "$DESTINO/CLAUDE.md"
sudo install -o root -g root -m 0644 "$ETAPA/CLAUDE.md.sha256" "$DESTINO/CLAUDE.md.sha256"
sudo install -o root -g root -m 0644 "$ETAPA/MANIFIESTO.sha256.json" "$DESTINO/MANIFIESTO.sha256.json"

# Las skills: reemplazo ATÓMICO del árbol completo (M4, auditoría adversarial
# 2026-09-22). Instalar archivo por archivo, uno por uno, SÓLO agrega -- una skill
# retirada de cerebros.toml, o cualquier archivo que quedó de una versión vieja,
# nunca se borraba. reemplazar_directorio_atomico.sh arma el árbol nuevo aparte y lo
# intercambia con un `mv` (rename atómico): el destino queda EXACTAMENTE igual a la
# etapa, ni un archivo de más.
bash "$(dirname "$(readlink -f "$0")")/reemplazar_directorio_atomico.sh" "$ETAPA/skills" "$DESTINO/skills"

# El punto de montaje de la jaula tiene que existir en el HOME real de la cuenta:
# aunque bwrap crea el destino solo (comprobado empíricamente, bubblewrap 0.11.1,
# con --dev-bind / / antes del --ro-bind), este instalador no depende de eso --
# mismo criterio que ya usa instalar_contratos.sh con settings.json/settings.local.json
# (sólo si no existen: no se pisa nada). Sólo si JAX_EJECUTOR_CUENTA ya está
# configurada (Fase 3/4 en esta máquina); si no, es un no-op.
if [ -n "${JAX_EJECUTOR_CUENTA:-}" ]; then
  HOME_CUENTA="$(getent passwd "$JAX_EJECUTOR_CUENTA" | cut -d: -f6)"
  sudo -u "$JAX_EJECUTOR_CUENTA" test -e "$HOME_CUENTA/.claude/CLAUDE.md" \
    || sudo install -D -o "$JAX_EJECUTOR_CUENTA" -g "$JAX_EJECUTOR_CUENTA" -m 0600 /dev/null "$HOME_CUENTA/.claude/CLAUDE.md"
  sudo -u "$JAX_EJECUTOR_CUENTA" test -d "$HOME_CUENTA/.claude/skills" \
    || sudo install -d -o "$JAX_EJECUTOR_CUENTA" -g "$JAX_EJECUTOR_CUENTA" -m 0700 "$HOME_CUENTA/.claude/skills"
fi

echo "instalado=true lib=\"$DESTINO\" rama=\"$RAMA\" commit=\"$(git -C "$REPO" rev-parse --short HEAD)\" sha256=\"$(cat "$ETAPA/CLAUDE.md.sha256")\""
