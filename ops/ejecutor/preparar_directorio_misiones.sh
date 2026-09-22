#!/usr/bin/env bash
# ops/ejecutor/preparar_directorio_misiones.sh — B-2 (ronda 4, auditoría adversarial
# 2026-09-22): JAX_EJECUTOR_MISIONES con una ACL EXPLÍCITA, no sólo el modo.
# Uso: preparar_directorio_misiones.sh <directorio> <admin> <cuenta>
#
# `<admin>` (JAX_EJECUTOR_ADMIN_USUARIO, fruiz) conserva rwx -- escribe la misión y
# toma/compara la huella (M-1). `<cuenta>` (JAX_EJECUTOR_CUENTA, axioma) recibe SÓLO
# travesía (--x, SIN +r): puede entrar a `<mision_id>/claude-projects` si ya sabe el
# UUID exacto (se lo da la propia misión que está corriendo), pero no puede LISTAR
# este directorio ni leer `<id>.json` de ninguna misión. El resto ("other") queda sin
# nada -- la ACL real de producción medida el 2026-09-22 (`getfacl
# /var/lib/jax-ejecutor-misiones`) ya tenía `other::---`; esto la reproduce a
# propósito en vez de depender del bit +x de "otros" (0751, ronda 3), que le daba
# travesía a CUALQUIER cuenta del sistema, no sólo a la del Ejecutor.
#
# Idempotente: `install -d` y `setfacl -m` no fallan si ya están puestos.
set -euo pipefail
DESTINO="${1:?}"; ADMIN="${2:?}"; CUENTA="${3:?}"
sudo install -d -o jaxsvc -g jaxsvc -m 0750 "$DESTINO"
sudo setfacl -m "u:$ADMIN:rwx" -m "d:u:$ADMIN:rwx" "$DESTINO"
sudo setfacl -m "u:$CUENTA:--x" -m "d:u:$CUENTA:--x" "$DESTINO"
echo "directorio_misiones_preparado=\"$DESTINO\" admin=\"$ADMIN\" cuenta=\"$CUENTA\""
