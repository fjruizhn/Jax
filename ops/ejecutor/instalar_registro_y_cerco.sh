#!/usr/bin/env bash
# ops/ejecutor/instalar_registro_y_cerco.sh — C3 en hall9000. Corre como fruiz; sudo para lo de root.
# Idempotente. Toca SOLO: el directorio del registro, los locks del carril, la tabla
# `inet ejecutor_cerco` de nftables (nada de otras tablas) y dos unidades systemd.
# Rollback del cerco: `sudo nft delete table inet ejecutor_cerco` (no toca ninguna otra tabla).
# Rollback total de la red: iptables-restore/ip6tables-restore de /var/backups/jax-ejecutor-cerco.
set -euo pipefail
: "${JAX_EJECUTOR_REGISTRO:?}" "${JAX_EJECUTOR_POLITICA:?}" "${JAX_EJECUTOR_CUENTA:?}"
: "${JAX_PROXY_CARRIL_PUERTO:?}" "${JAX_EJECUTOR_CANARIO_PUERTO:?}" "${JAX_PROXY_CARRIL_RAIZ:?}"
REPO="$(git -C "$(dirname "$(readlink -f "$0")")" rev-parse --show-toplevel)"
test "$(git -C "$REPO" branch --show-current)" = master
PY="$REPO/.venv/bin/python"
MARCA="$(date +%Y%m%d-%H%M%S)"

# Registro: directorio de fruiz 0750 (la cuenta ni lo lista), archivo 0640, append-only.
DIR="$(dirname "$JAX_EJECUTOR_REGISTRO")"
sudo install -d -o fruiz -g fruiz -m 0750 "$DIR"
test -e "$JAX_EJECUTOR_REGISTRO" || install -m 0640 /dev/null "$JAX_EJECUTOR_REGISTRO"
sudo chattr +a "$JAX_EJECUTOR_REGISTRO"
lsattr "$JAX_EJECUTOR_REGISTRO" | cut -d' ' -f1 | grep -q a

# Locks del carril: de fruiz; la cuenta del Ejecutor nunca los toca (tests/test_ejecutor_carril_solo_en_el_proxy.py).
sudo install -d -o fruiz -g fruiz -m 0750 "$JAX_PROXY_CARRIL_RAIZ"

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
sudo install -o root -g root -m 0644 "$REPO/ops/ejecutor/jax-ejecutor-proxy.service" /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable ejecutor-cerco.service
sudo systemctl restart ejecutor-cerco.service
sudo nft list table inet ejecutor_cerco >/dev/null
sudo systemctl enable jax-ejecutor-proxy.service
sudo systemctl restart jax-ejecutor-proxy.service
echo "c3_instalado=true"
