#!/usr/bin/env bash
# ops/ejecutor/instalar_registro_y_cerco.sh — C3 en hall9000. Corre como fruiz; sudo para lo de root.
# Idempotente. Toca SOLO: el directorio del registro, los locks del carril, la tabla
# `inet ejecutor_cerco` de nftables (nada de otras tablas) y dos unidades systemd.
# Rollback del cerco: `sudo nft delete table inet ejecutor_cerco` (no toca ninguna otra tabla).
# Rollback total de la red: iptables-restore/ip6tables-restore de /var/backups/jax-ejecutor-cerco.
set -euo pipefail
: "${JAX_EJECUTOR_REGISTRO:?}" "${JAX_EJECUTOR_POLITICA:?}" "${JAX_EJECUTOR_CUENTA:?}"
: "${JAX_PROXY_CARRIL_PUERTO:?}" "${JAX_EJECUTOR_CANARIO_PUERTO:?}" "${JAX_PROXY_CARRIL_RAIZ:?}"
# C5 (plan 4): el proxy no arranca sin la pausa del Ejecutor y el latido del vigía configurados.
: "${JAX_EJECUTOR_PAUSA:?}" "${JAX_EJECUTOR_VIGIA_LATIDO:?}" "${JAX_EJECUTOR_VIGIA_LATIDO_MAX_S:?}"
# El cerco sólo deja alcanzar la local y las remotas con freno (cerco.py): la variable tiene que existir, aunque vacía.
: "${JAX_EJECUTOR_FRENO_REMOTOS?}"
# MAJOR-2 (auditoría escalón 3, ronda 3): /srv/jax-prod/jax es jaxsvc:jaxsvc
# -- `git rev-parse`/`status` ahí dan "detected dubious ownership" (rc=128),
# como fruiz y como root, sin `-c safe.directory=...`. Se declara por
# invocación (nunca en la config global), literal -- es la ruta que este
# guion EXIGE más abajo, no una que se calcula después.
RUTA_PRODUCCION=/srv/jax-prod/jax
REPO="$(git -c safe.directory="$RUTA_PRODUCCION" -C "$(dirname "$(readlink -f "$0")")" rev-parse --show-toplevel)"
if [ "$REPO" != "$RUTA_PRODUCCION" ]; then
  echo "instalar_registro_y_cerco.sh: este guion corre desde $REPO -- tiene que ser $RUTA_PRODUCCION. Abortando." >&2
  exit 1
fi
test "$(git -c safe.directory="$REPO" -C "$REPO" branch --show-current)" = master
# MAJOR-3/ronda-2: faltaba el árbol LIMPIO -- una rama correcta con cambios
# sin comitear (un experimento a medias, un archivo tocado a mano) también
# es un checkout que no es fiable copiar a unidades de producción.
#
# Corrido con `sudo` (root, con safe.directory) -- ronda 3, MAJOR-2: este
# guion corre como fruiz, y fruiz no puede LISTAR algunos subdirectorios de
# jaxsvc (las_manos/workspace, las_manos/repo, las_manos/missions) --
# `git status --porcelain` daba rc=0 igual, con esos "Permission denied"
# sólo en stderr, así que un `test -z "$(...)"` que sólo mirara stdout se
# habría tragado el aviso. Acá CUALQUIER stderr cuenta como fallo.
ARCHIVO_ERR_STATUS="$(mktemp)"
SUCIO="$(sudo git -c safe.directory="$REPO" -C "$REPO" status --porcelain 2>"$ARCHIVO_ERR_STATUS")"
ERR_STATUS="$(cat -- "$ARCHIVO_ERR_STATUS")"; rm -f -- "$ARCHIVO_ERR_STATUS"
if [ -n "$ERR_STATUS" ]; then
  echo "instalar_registro_y_cerco.sh: git status avisó algo en $REPO (tratado como fallo): $ERR_STATUS" >&2
  exit 1
fi
[ -z "$SUCIO" ] || {
  echo "instalar_registro_y_cerco.sh: $REPO tiene cambios sin comitear -- abortando:
$SUCIO" >&2
  exit 1
}
PY="$REPO/.venv/bin/python"
MARCA="$(date +%Y%m%d-%H%M%S)"

# Registro: directorio de la cuenta de SERVICIO (jaxsvc) 0750 -- la cuenta de la JAULA ni lo
# lista --, archivo 0640, append-only. Desde 2026-09-17 los servicios no corren como el
# operador: el proxy escribe el registro como jaxsvc (ver tests/test_ejecutor_cuenta_de_servicio.py).
DIR="$(dirname "$JAX_EJECUTOR_REGISTRO")"
sudo install -d -o jaxsvc -g jaxsvc -m 0750 "$DIR"
test -e "$JAX_EJECUTOR_REGISTRO" || install -m 0640 /dev/null "$JAX_EJECUTOR_REGISTRO"
sudo chattr +a "$JAX_EJECUTOR_REGISTRO"
lsattr "$JAX_EJECUTOR_REGISTRO" | cut -d' ' -f1 | grep -q a

# C5: la pausa vive junto al interruptor de JAX (directorio del frente B, root:fruiz 2770: la
# cuenta del Ejecutor no la puede borrar). El latido, en un directorio de fruiz 0750: si la
# cuenta pudiera tocarlo, fingiría un vigía vivo.
test -d "$(dirname "$JAX_EJECUTOR_PAUSA")"
sudo install -d -o jaxsvc -g jaxsvc -m 0750 "$(dirname "$JAX_EJECUTOR_VIGIA_LATIDO")"

# Locks del carril: de fruiz; la cuenta del Ejecutor nunca los toca (tests/test_ejecutor_carril_solo_en_el_proxy.py).
sudo install -d -o jaxsvc -g jaxsvc -m 0750 "$JAX_PROXY_CARRIL_RAIZ"

# Cerco: backup de TODA la red antes de tocar nada, y prueba de que se restaura.
# Medido 2026-09-17: `nft list ruleset` de hall9000 NO se puede volver a cargar (las tablas
# son de iptables-nft/ufw, con expresiones xt: «unsupported xtables compat expression»).
# El respaldo restaurable es iptables-save + ip6tables-save; se prueba en un netns
# descartable (no toca la red del host) y tiene que dar las mismas reglas y las mismas tablas.
RESPALDOS=/var/backups/jax-ejecutor-cerco
sudo install -d -o root -g root -m 0700 "$RESPALDOS"
R="$RESPALDOS/pre-cerco-$MARCA"
sudo sh -c "iptables-save > '$R.ipt4' && ip6tables-save > '$R.ipt6' && nft list ruleset > '$R.nft' 2>/dev/null"
normalizar() { grep -v '^#' | sed 's/\[[0-9]*:[0-9]*\]//'; }
diff <(sudo cat "$R.ipt4" "$R.ipt6" | normalizar) \
     <(sudo unshare -n sh -c "iptables-restore < '$R.ipt4' && ip6tables-restore < '$R.ipt6' && iptables-save && ip6tables-save" | normalizar)
diff <(sudo nft list tables | grep -v "inet ejecutor_cerco$" | sort) \
     <(sudo unshare -n sh -c "iptables-restore < '$R.ipt4' && ip6tables-restore < '$R.ipt6' && nft list tables" | sort)
echo "red_respaldada=$R.ipt4,$R.ipt6 restauracion_probada=true"

# Render como fruiz desde la política, chequeo de sintaxis, instalación root.
ETAPA="$(mktemp -d)"; trap 'rm -rf "$ETAPA"' EXIT
( cd "$REPO" && PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:las_manos "$PY" -m jax.ejecutor.contratos.cerco "$JAX_EJECUTOR_POLITICA" "$ETAPA/cerco.nft" )
sudo nft -c -f "$ETAPA/cerco.nft"
sudo install -d -o root -g root -m 0755 /etc/jax-ejecutor-cerco
sudo install -o root -g root -m 0644 "$ETAPA/cerco.nft" /etc/jax-ejecutor-cerco/cerco.nft
sudo install -o root -g root -m 0644 "$REPO/ops/ejecutor/ejecutor-cerco.service" /etc/systemd/system/
# Orden a propósito (auditoría escalón 3, M5): el guion de sanidad y los
# drop-ins del proxy ANTES que su unidad base -- así, si esto se interrumpe
# a mitad de camino, nunca queda la unidad base sola en /etc sin lo que
# necesita para arrancar bien. Las dos líneas siguientes son la MISMA
# fuente que ops/manifiesto-arranque-instalado.tsv, nunca una lista aparte.
"$REPO/ops/instalar-dropins-de-servicio.sh" /usr/local/sbin/jax-checkout-de-produccion-sano.sh "$REPO"
"$REPO/ops/instalar-dropins-de-servicio.sh" jax-ejecutor-proxy.service.d "$REPO"
sudo install -o root -g root -m 0644 "$REPO/ops/ejecutor/jax-ejecutor-proxy.service" /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable ejecutor-cerco.service
sudo systemctl restart ejecutor-cerco.service
sudo nft list table inet ejecutor_cerco >/dev/null
sudo systemctl enable jax-ejecutor-proxy.service
sudo systemctl restart jax-ejecutor-proxy.service
echo "c3_instalado=true"
