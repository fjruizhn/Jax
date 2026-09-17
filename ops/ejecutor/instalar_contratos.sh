#!/usr/bin/env bash
# ops/ejecutor/instalar_contratos.sh — instala C1/C2 del Ejecutor en hall9000.
# Corre como fruiz desde el checkout de producción (/home/fruiz/jax en master); usa sudo
# para lo que es de root. Idempotente. Lee JAX_EJECUTOR_* del entorno (set -a; . /etc/jax/.env).
set -euo pipefail
: "${JAX_EJECUTOR_LIB:?}" "${JAX_EJECUTOR_POLITICA:?}" "${JAX_EJECUTOR_GANCHO_TOPE_S:?}" "${JAX_EJECUTOR_CUENTA:?}"
REPO="$(git -C "$(dirname "$(readlink -f "$0")")" rev-parse --show-toplevel)"
# Producción se instala desde master. `--rama-aprobada` (sólo con GO/autonomía de Fernando) permite
# instalar desde la rama del checkout y lo deja dicho en la salida.
RAMA="$(git -C "$REPO" branch --show-current)"
test "$RAMA" = master || test "${1:-}" = --rama-aprobada
ETAPA="$(mktemp -d)"
trap 'rm -rf "$ETAPA"' EXIT
( cd "$REPO" && PYTHONDONTWRITEBYTECODE=1 python3 -m jax.ejecutor.contratos.instalacion "$ETAPA" )

sudo install -d -o root -g root -m 0755 "$JAX_EJECUTOR_LIB"
while IFS= read -r rel; do
  sudo install -D -o root -g root -m 0644 "$REPO/$rel" "$JAX_EJECUTOR_LIB/$rel"
done < "$ETAPA/instalables.txt"
sudo install -o root -g root -m 0755 "$ETAPA/gancho.sh" "$JAX_EJECUTOR_LIB/gancho.sh"
for f in managed-settings.json settings-usuario.json ejecutor-freno.service manifiesto.sha256; do
  sudo install -o root -g root -m 0644 "$ETAPA/$f" "$JAX_EJECUTOR_LIB/$f"
done

# Punto de montaje de la jaula: VACÍO en el host. Si alguien puso un managed-settings
# global, se para: afectaría a todas las sesiones de la máquina.
sudo install -d -o root -g root -m 0755 /etc/claude-code
test -z "$(sudo ls -A /etc/claude-code)"

# Política: la escribe fruiz, la lee la cuenta, nadie más.
sudo install -d -o fruiz -g "$JAX_EJECUTOR_CUENTA" -m 2750 "$(dirname "$JAX_EJECUTOR_POLITICA")"

# Puntos de montaje de los settings de la cuenta (sólo si no existen: no se pisa nada).
HOME_CUENTA="$(getent passwd "$JAX_EJECUTOR_CUENTA" | cut -d: -f6)"
for f in settings.json settings.local.json; do
  sudo -u "$JAX_EJECUTOR_CUENTA" test -e "$HOME_CUENTA/.claude/$f" \
    || sudo install -D -o "$JAX_EJECUTOR_CUENTA" -g "$JAX_EJECUTOR_CUENTA" -m 0600 /dev/null "$HOME_CUENTA/.claude/$f"
done

( cd "$JAX_EJECUTOR_LIB" && sudo sha256sum -c --quiet manifiesto.sha256 )
echo "instalado=true lib=\"$JAX_EJECUTOR_LIB\" rama=\"$RAMA\" commit=\"$(git -C "$REPO" rev-parse --short HEAD)\""
