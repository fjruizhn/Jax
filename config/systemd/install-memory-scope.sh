#!/usr/bin/env bash
# Installer helper only.  Production deployment owns enable/restart decisions;
# B9 does not restart platform services as a side effect of memory setup.
set -euo pipefail
ENVF=/etc/jax/.env

# MAJOR-3/ronda-2 (auditoría escalón 3): antes REPO/SD estaban fijos a
# /home/fruiz/jax -- el checkout de TRABAJO de un agente, en rama ajena, sin
# el guion nuevo. Corriendo como root eso copiaría lo que el agente tuviera
# puesto en ese momento, no lo que el repo versiona. REPO se deriva del
# propio guion (igual que ops/ejecutor/instalar_registro_y_cerco.sh) y se
# EXIGE que sea /srv/jax-prod/jax -- no "parecido", exactamente esa ruta.
#
# MAJOR-2 (ronda 3): /srv/jax-prod/jax es jaxsvc:jaxsvc -- `git rev-parse`
# ahí da "detected dubious ownership" (rc=128), tanto como fruiz como como
# root, sin `-c safe.directory=...`. Se declara por invocación (nunca en la
# config global), literal (todavía no se sabe si $REPO va a ser
# /srv/jax-prod/jax -- eso es justo lo que esta línea intenta averiguar), no
# variable: si el resultado no es exactamente esa ruta, la excepción de
# safe.directory declarada tampoco aplicaba a donde sea que $0 vivía, y el
# chequeo de abajo aborta igual.
RUTA_PRODUCCION=/srv/jax-prod/jax
REPO="$(git -c safe.directory="$RUTA_PRODUCCION" -C "$(dirname "$(readlink -f "$0")")" rev-parse --show-toplevel)"
if [ "$REPO" != "$RUTA_PRODUCCION" ]; then
  echo "install-memory-scope.sh: este guion corre desde $REPO -- tiene que ser $RUTA_PRODUCCION (el checkout de producción), no un checkout de trabajo. Abortando." >&2
  exit 1
fi
SD="$REPO/config/systemd"

# El mismo freno que ExecStartPre ya exige en caliente (master + árbol
# limpio), ACÁ TAMBIÉN, antes de copiar nada -- si /srv/jax-prod/jax no
# está en las condiciones que checkout-de-produccion.conf va a exigirle al
# servicio de todos modos, mejor que este guion aborte ahora que dejar
# unidades instaladas contra un checkout que ni siquiera va a arrancar.
/usr/local/sbin/jax-checkout-de-produccion-sano.sh "$REPO"

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
# Orden a propósito (auditoría escalón 3, M5): el guion de sanidad y los
# drop-ins ANTES que la unidad base -- así, si esto se interrumpe a mitad
# de camino, nunca queda una unidad base sola en /etc sin lo que necesita
# para arrancar bien (ni el guion de checkout-de-produccion.conf, ni
# User=jaxsvc de cuenta-de-servicio.conf, ni el PYTHONPATH de
# z-pythonpath.conf). Las dos líneas siguientes son la MISMA fuente que
# ops/manifiesto-arranque-instalado.tsv, nunca una lista aparte.
"$REPO/ops/instalar-dropins-de-servicio.sh" /usr/local/sbin/jax-checkout-de-produccion-sano.sh "$REPO"
"$REPO/ops/instalar-dropins-de-servicio.sh" jax-memory-worker.service.d "$REPO"
install -m 644 "$SD/jax-memory-worker.service" /etc/systemd/system/
install -m 644 "$SD/jax-memory-worker.timer"  /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now jax-memory-worker.timer
systemctl list-timers jax-memory-worker.timer --no-pager || true

echo "LISTO."
