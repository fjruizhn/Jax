#!/usr/bin/env bash
# ops/ejecutor/instalar_contexto.sh [--rama-aprobada] — instala el CLAUDE.md GENERADO
# (spec §6.1: nunca a mano) y las skills declaradas (cerebros.toml `skills`) en
# JAX_EJECUTOR_LIB/contexto/, de solo lectura para la cuenta del Ejecutor. La jaula
# (jax.ejecutor.contratos.cuenta_axioma._jaula) monta esos dos caminos ro sobre
# "$HOME/.claude/CLAUDE.md" y "$HOME/.claude/skills"; el arranque de cada misión
# (jax.ejecutor.contratos.arranque.verificar_contexto) rechaza la misión si lo
# instalado no coincide con lo que el generador produce HOY.
#
# Corre como fruiz desde el checkout de producción (/home/fruiz/jax en master); usa
# sudo para lo que es de root. Idempotente. Lee JAX_EJECUTOR_LIB del entorno
# (set -a; . <(sudo -n cat /etc/jax/.env)).
set -euo pipefail
: "${JAX_EJECUTOR_LIB:?}"
REPO="$(git -C "$(dirname "$(readlink -f "$0")")" rev-parse --show-toplevel)"
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
sudo test -e "$DESTINO/CLAUDE.md" \
  && diff -u <(sudo cat "$DESTINO/CLAUDE.md") "$ETAPA/CLAUDE.md" \
  || echo "(sin CLAUDE.md instalado todavía)"
echo "--- fin diff ---"

sudo install -d -o root -g root -m 0755 "$DESTINO"
sudo install -o root -g root -m 0644 "$ETAPA/CLAUDE.md" "$DESTINO/CLAUDE.md"
sudo install -o root -g root -m 0644 "$ETAPA/CLAUDE.md.sha256" "$DESTINO/CLAUDE.md.sha256"

# Las skills: carpeta completa (con subcarpetas), una por una desde la etapa.
sudo install -d -o root -g root -m 0755 "$DESTINO/skills"
( cd "$ETAPA/skills" && find . -type f ) | while IFS= read -r rel; do
  sudo install -D -o root -g root -m 0644 "$ETAPA/skills/$rel" "$DESTINO/skills/$rel"
done

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
