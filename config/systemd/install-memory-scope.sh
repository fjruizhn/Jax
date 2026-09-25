#!/usr/bin/env bash
# Installer helper only.  Production deployment owns enable/restart decisions;
# B9 does not restart platform services as a side effect of memory setup.
set -euo pipefail
ENVF=/etc/jax/.env
SD=/home/fruiz/jax/config/systemd
REPO="$(dirname "$(dirname "$SD")")"
# DESTDIR (ops/versionar-drop-ins, 2026-09-25): vacío por defecto -- MISMO
# comportamiento de siempre, instala contra el /etc real. Sólo existe para
# poder probar la instalación de las unidades systemd (base + drop-ins) con
# una raíz temporal, sin tocar el sistema real -- el resto del guion
# (JAX_REPL_* en /etc/jax/.env, systemctl enable --now) no lo respeta y
# sigue tocando el sistema real siempre: no se prueba de punta a punta con
# DESTDIR, sólo la parte de instalación de unidades.
: "${DESTDIR:=}"

echo "== 1) JAX_REPL_* en $ENVF =="
if ! grep -q '^JAX_REPL_USER_ID=' "$ENVF"; then
  cat >> "$ENVF" <<'VARS'

# Identidad del REPL para scope de memoria de dos niveles (Fernando individual)
JAX_REPL_USER_ID=1
JAX_REPL_TENANT_ID=1
VARS
  echo "   (agregadas)"
else
  echo "   (ya existian — sin cambios)"
fi
grep -E 'JAX_REPL_USER_ID|JAX_REPL_TENANT_ID' "$ENVF"

echo "== 2) Unidades systemd del worker =="
install -m 644 "$SD/jax-memory-worker.service" "${DESTDIR}/etc/systemd/system/"
install -m 644 "$SD/jax-memory-worker.timer"  "${DESTDIR}/etc/systemd/system/"
# Drop-ins versionados del worker (ops/versionar-drop-ins, 2026-09-25):
# derivados de ops/manifiesto-arranque-instalado.tsv, no de una lista aparte acá.
"$REPO/ops/instalar-dropins-de-servicio.sh" jax-memory-worker.service.d "$REPO" "$DESTDIR"
systemctl daemon-reload
systemctl enable --now jax-memory-worker.timer
systemctl list-timers jax-memory-worker.timer --no-pager || true

echo "LISTO."
