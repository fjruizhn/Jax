#!/usr/bin/env bash
# ops/ejecutor/instalar_carril_comun.sh — SP3: directorio de locks del carril con grupo común.
# Corre como fruiz; sudo para lo de root. Idempotente. NO reinicia nada y NO lo necesita: el dueño
# sigue siendo la cuenta de servicio que ya lo usa (los procesos en marcha no pierden acceso).
#
# Por qué grupo: la Mesa (jax-platform) y el proxy del Ejecutor toman `flock` sobre los MISMOS
# ficheros. Si algún servicio pasa a otra cuenta, el fichero nacería con el umask de quien llega
# primero. jax/ejecutor/prioridad.py ya abre los locks de sólo lectura (flock no pide escritura);
# esto les da a las cuentas de servicio el directorio (setgid, 2770) y los ficheros (0640, grupo).
#
# NUNCA la cuenta del Ejecutor (JAX_EJECUTOR_CUENTA): con acceso a mesa.lock la jaula podría
# retenerlo y la Mesa esperaría para siempre (el carril de la Mesa no tiene tope, a propósito).
# El script se niega antes de tocar nada — tests/test_ejecutor_sp3_config.py lo ejercita.
#
# Uso:     set -a; . <(sudo -n cat /etc/jax/.env); set +a; ops/ejecutor/instalar_carril_comun.sh
# Rollback: sudo chown fruiz:fruiz "$JAX_PROXY_CARRIL_RAIZ" && sudo chmod 0750 "$JAX_PROXY_CARRIL_RAIZ"
#           (y `sudo gpasswd -d <cuenta> "$JAX_CARRIL_GRUPO"` por cada cuenta agregada)
set -euo pipefail
: "${JAX_PROXY_CARRIL_RAIZ:?}" "${JAX_CARRIL_GRUPO:?}" "${JAX_CARRIL_CUENTAS:?}" "${JAX_EJECUTOR_CUENTA:?}"

case "$JAX_PROXY_CARRIL_RAIZ" in /*) ;; *) echo "carril_raiz_no_absoluta" >&2; exit 2 ;; esac
for cuenta in $JAX_CARRIL_CUENTAS; do
  if [ "$cuenta" = "$JAX_EJECUTOR_CUENTA" ]; then
    echo "cuenta_del_ejecutor_en_el_grupo_del_carril=$cuenta" >&2
    exit 2
  fi
done
DUENIO="${JAX_CARRIL_CUENTAS%% *}"

sudo groupadd -f "$JAX_CARRIL_GRUPO"
for cuenta in $JAX_CARRIL_CUENTAS; do
  sudo usermod -aG "$JAX_CARRIL_GRUPO" "$cuenta"
done
sudo install -d -o "$DUENIO" -g "$JAX_CARRIL_GRUPO" -m 2770 "$JAX_PROXY_CARRIL_RAIZ"
for nombre in mesa.lock ejecutor.lock; do
  f="$JAX_PROXY_CARRIL_RAIZ/$nombre"
  sudo sh -c "test -e '$f' || install -o '$DUENIO' -g '$JAX_CARRIL_GRUPO' -m 0640 /dev/null '$f'"
  sudo chgrp "$JAX_CARRIL_GRUPO" "$f"
  sudo chmod 0640 "$f"
done

# Verificación independiente: dueño, grupo y modos como quedaron, y la cuenta del Ejecutor sin acceso.
stat -c '%U:%G %a %n' "$JAX_PROXY_CARRIL_RAIZ" "$JAX_PROXY_CARRIL_RAIZ"/mesa.lock "$JAX_PROXY_CARRIL_RAIZ"/ejecutor.lock
if sudo -u "$JAX_EJECUTOR_CUENTA" test -r "$JAX_PROXY_CARRIL_RAIZ/mesa.lock"; then
  echo "carril_legible_por_el_ejecutor=true" >&2
  exit 1
fi
echo "carril_comun=true grupo=$JAX_CARRIL_GRUPO"
