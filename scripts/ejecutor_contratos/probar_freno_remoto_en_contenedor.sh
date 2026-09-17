#!/usr/bin/env bash
# scripts/ejecutor_contratos/probar_freno_remoto_en_contenedor.sh — C4 remoto, ensayado en un contenedor desechable.
#
# Ensaya en hall9000, SIN tocar .10/.11/.20, exactamente lo que Fernando corre contra una remota:
#   1. ops/ejecutor/instalar_en_maquina.sh <máquina>  (C6, SIN --sin-freno: sube ejecutor-freno-remoto y
#      agrega la llave del freno con command=…,restrict al archivo root de llaves de la cuenta);
#   2. el comando forzado responde y no deja correr otra cosa;
#   3. probar_c4.py --remoto <máquina> con el freno remoto habilitado → c4_vivo=true (control + dos rondas);
#   4. VERLO FALLAR: el mismo simulacro con el barrido remoto apagado → procesos_remotos_vivos y marca;
#   5. ops/ejecutor/revertir_en_maquina.sh <máquina> → la llave del freno ya no entra (Permission denied).
#
# Sin PerSourcePenalties en el contenedor: el ssh-keyscan cuenta como conexión sin autenticar y sshd
# (OpenSSH >= 9.8) rechazaba las siguientes por unos segundos (visto: «Connection refused» a mitad de la
# instalación). La llave del freno sí autentica, así que en una remota real esto no aplica.
# La "remota" es un contenedor (sshd en :58291, sudo clásico, usuario administrador con las llaves de
# fruiz, la cuenta con la llave de la cuenta de hall9000). No tiene systemd: `systemctl` y `loginctl`
# son dobles que recargan sshd con HUP y no hacen nada, respectivamente (lo de systemd ya lo ejercitó
# C6 en hall9000). Un freno root de PRUEBA (unidad transitoria) lleva la política de prueba y la remota.
#
# Cambios TRANSITORIOS en hall9000, todos con respaldo y `cmp` al restaurar (trap EXIT):
#   ~fruiz/.ssh/known_hosts y ~axioma/.ssh/known_hosts (+ la llave de host del contenedor),
#   nftables inet ejecutor_cerco @destinos_ssh (+ IP del contenedor . 58291, se quita).
# No toca /etc/jax/.env, la política de producción ni el freno de producción.
#
# Uso (como fruiz, desde el checkout): set -a; . /etc/jax/.env; set +a
#      scripts/ejecutor_contratos/probar_freno_remoto_en_contenedor.sh
set -euo pipefail
: "${JAX_EJECUTOR_POLITICA:?}" "${JAX_EJECUTOR_CUENTA:?}" "${JAX_EJECUTOR_ADMIN_USUARIO:?}" "${JAX_EJECUTOR_FRENO_LLAVE:?}"
REPO="$(git -C "$(dirname "$(readlink -f "$0")")" rev-parse --show-toplevel)"
C="$JAX_EJECUTOR_CUENTA"
ADMIN="$JAX_EJECUTOR_ADMIN_USUARIO"
PUERTO=58291
MAQUINA=prueba-freno
NOMBRE_CONTENEDOR="ejecutor-freno-prueba-$$"
UNIDAD_PRUEBA="ejecutor-freno-prueba-$$"
IMAGEN="${IMAGEN_PRUEBA:-python:3.14-slim}"
TMP="$(mktemp -d)"; chmod 0700 "$TMP"
HOME_C="$(getent passwd "$C" | cut -d: -f6)"
HOME_A="$(getent passwd "$ADMIN" | cut -d: -f6)"
IP=""

paso() { echo "== $*"; }

limpiar() {
  set +e
  sudo systemctl stop "$UNIDAD_PRUEBA" 2>/dev/null
  sudo docker rm -f "$NOMBRE_CONTENEDOR" >/dev/null 2>&1
  [ -z "$IP" ] || sudo nft delete element inet ejecutor_cerco destinos_ssh "{ $IP . $PUERTO }" 2>/dev/null
  sudo nft list set inet ejecutor_cerco destinos_ssh > "$TMP/cerco-despues.txt"
  cmp "$TMP/cerco-antes.txt" "$TMP/cerco-despues.txt" && echo "restaurado=cerco cmp=ok"
  if [ -f "$TMP/kh-admin.bak" ]; then cp "$TMP/kh-admin.bak" "$HOME_A/.ssh/known_hosts" && cmp "$TMP/kh-admin.bak" "$HOME_A/.ssh/known_hosts" && echo "restaurado=known_hosts_admin cmp=ok"; fi
  if [ -f "$TMP/kh-cuenta.bak" ]; then
    sudo install -o "$C" -g "$C" -m 0600 "$TMP/kh-cuenta.bak" "$HOME_C/.ssh/known_hosts"
    sudo cmp "$TMP/kh-cuenta.bak" "$HOME_C/.ssh/known_hosts" && echo "restaurado=known_hosts_cuenta cmp=ok"
  fi
  sudo rm -rf "$TMP"
}
trap limpiar EXIT

esperar_cuenta_vacia() {
  for _ in $(seq 60); do [ -z "$(ps -u "$C" -o pid=)" ] && return 0; python3 -c 'import time; time.sleep(1)'; done
  echo "codigo=cuenta_local_ocupada" >&2; return 1
}

sudo nft list set inet ejecutor_cerco destinos_ssh > "$TMP/cerco-antes.txt"
cp -p "$HOME_A/.ssh/known_hosts" "$TMP/kh-admin.bak"
sudo cp "$HOME_C/.ssh/known_hosts" "$TMP/kh-cuenta.bak"; sudo chown "$(id -u)" "$TMP/kh-cuenta.bak"

paso "contenedor que hace de máquina remota"
# --init: un init que cosecha (sin él, lo que mata el barrido queda zombi para siempre).
sudo docker run -d --init --name "$NOMBRE_CONTENEDOR" "$IMAGEN" sleep infinity >/dev/null
IP="$(sudo docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "$NOMBRE_CONTENEDOR")"
test -n "$IP"
dentro() { sudo docker exec -i "$NOMBRE_CONTENEDOR" sh -c "$1"; }
dentro "apt-get -qq update >/dev/null && DEBIAN_FRONTEND=noninteractive apt-get -qq install -y --no-install-recommends openssh-server sudo procps >/dev/null"
# uids que NO existen en hall9000: con el mismo uid, el freno root de hall9000 (que barre /proc por uid)
# mataría los procesos del contenedor y el barrido remoto parecería funcionar sin haber hecho nada.
UID_ADMIN_C=$(( $(getent passwd | cut -d: -f3 | sort -n | tail -1) + 1000 ))
test -z "$(getent passwd "$UID_ADMIN_C")" && test -z "$(getent passwd "$((UID_ADMIN_C + 1))")"
dentro "useradd -m -u $UID_ADMIN_C -s /bin/bash $ADMIN && useradd -m -u $((UID_ADMIN_C + 1)) -s /bin/bash $C && echo '$ADMIN ALL=(ALL) NOPASSWD: ALL' > /etc/sudoers.d/admin && chmod 0440 /etc/sudoers.d/admin"
cat "$HOME_A"/.ssh/*.pub | dentro "install -d -o $ADMIN -m 0700 /home/$ADMIN/.ssh && cat > /home/$ADMIN/.ssh/authorized_keys && chown $ADMIN /home/$ADMIN/.ssh/authorized_keys"
sudo cat "$HOME_C/.ssh/id_ed25519.pub" | dentro "install -d -o $C -m 0700 /home/$C/.ssh && cat > /home/$C/.ssh/authorized_keys && chown $C /home/$C/.ssh/authorized_keys"
printf '#!/bin/sh\ncase "$1" in\n  list-unit-files) echo "ssh.service enabled enabled" ;;\n  reload) pkill -HUP -x sshd ;;\nesac\n' | dentro "cat > /usr/local/bin/systemctl && chmod 0755 /usr/local/bin/systemctl"
printf '#!/bin/sh\nexit 0\n' | dentro "cat > /usr/local/bin/loginctl && chmod 0755 /usr/local/bin/loginctl"
dentro "mkdir -p /run/sshd && printf 'Port $PUERTO\\nPerSourcePenalties no\\n' > /etc/ssh/sshd_config.d/00-puerto.conf && /usr/sbin/sshd"

paso "confianza transitoria en la llave de host del contenedor (admin y cuenta) y cerco de la cuenta"
for _ in $(seq 20); do ssh-keyscan -p "$PUERTO" -t ed25519 "$IP" > "$TMP/host.key" 2>/dev/null && [ -s "$TMP/host.key" ] && break; python3 -c 'import time; time.sleep(0.5)'; done
test -s "$TMP/host.key"
cat "$TMP/host.key" >> "$HOME_A/.ssh/known_hosts"
sudo sh -c "cat '$TMP/host.key' >> '$HOME_C/.ssh/known_hosts'"
sudo nft add element inet ejecutor_cerco destinos_ssh "{ $IP . $PUERTO }"
for _ in $(seq 20); do
  ssh -o BatchMode=yes -o ConnectTimeout=3 -o StrictHostKeyChecking=yes -p "$PUERTO" "$ADMIN@$IP" true 2>/dev/null && break
  python3 -c 'import time; time.sleep(0.5)'
done
ssh -o BatchMode=yes -o ConnectTimeout=3 -o StrictHostKeyChecking=yes -p "$PUERTO" "$ADMIN@$IP" true

paso "política de prueba: la de producción + la máquina del contenedor"
( cd "$REPO" && PYTHONDONTWRITEBYTECODE=1 python3 - "$JAX_EJECUTOR_POLITICA" "$TMP/politica.json" "$MAQUINA" "$IP" "$PUERTO" <<'PY'
import json, sys
from jax.ejecutor.contratos import politica
origen, destino, nombre, ip, puerto = sys.argv[1:6]
doc = {k: v for k, v in json.load(open(origen)).items() if k != "sha256"}
doc["hosts"] = [*doc["hosts"], {"nombre": nombre, "ip": ip, "puerto": int(puerto), "rol": "desarrollo", "es_local": False}]
firmado = politica.firmar(doc)
politica.validar(firmado)
open(destino, "w").write(json.dumps(firmado, indent=2, sort_keys=True))
PY
)
chmod 0644 "$TMP/politica.json"
: > "$TMP/freno_known_hosts"
export JAX_EJECUTOR_POLITICA="$TMP/politica.json" JAX_EJECUTOR_FRENO_KNOWN_HOSTS="$TMP/freno_known_hosts"

paso "1. instalar_en_maquina.sh $MAQUINA (sin --sin-freno)"
"$REPO/ops/ejecutor/instalar_en_maquina.sh" "$MAQUINA"
dentro "grep -c 'command=\"/usr/local/sbin/ejecutor-freno-remoto\",restrict .* ejecutor-freno\$' /etc/ssh/authorized_keys.d/$C"

paso "2. comando forzado: pida lo que pida, sólo barre"
SALIDA="$(sudo ssh -i "$JAX_EJECUTOR_FRENO_LLAVE" -o BatchMode=yes -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes \
  -o UserKnownHostsFile="$TMP/freno_known_hosts" -p "$PUERTO" "$C@$IP" 'id; touch /tmp/no-deberia')"
echo "salida=\"$SALIDA\""
test "$SALIDA" = "freno_remoto=ok quedan=0"
dentro "test ! -e /tmp/no-deberia"

freno_de_prueba() {  # $1 = remotas habilitadas
  sudo systemctl stop "$UNIDAD_PRUEBA" 2>/dev/null || true
  sudo systemctl reset-failed "$UNIDAD_PRUEBA" 2>/dev/null || true
  sudo systemd-run --quiet --unit "$UNIDAD_PRUEBA" --property=RuntimeDirectory="$UNIDAD_PRUEBA" \
    --setenv=JAX_KILL_SWITCH_PATH="$JAX_KILL_SWITCH_PATH" --setenv=JAX_EJECUTOR_PAUSA="$JAX_EJECUTOR_PAUSA" \
    --setenv=JAX_EJECUTOR_CUENTA="$C" --setenv=JAX_EJECUTOR_ADMIN_USUARIO="$ADMIN" \
    --setenv=JAX_EJECUTOR_POLITICA="$TMP/politica.json" --setenv=JAX_EJECUTOR_FRENO_LLAVE="$JAX_EJECUTOR_FRENO_LLAVE" \
    --setenv=JAX_EJECUTOR_FRENO_KNOWN_HOSTS="$TMP/freno_known_hosts" --setenv=JAX_EJECUTOR_FRENO_REMOTOS="$1" \
    --setenv=JAX_EJECUTOR_FRENO_ESTADO="/run/$UNIDAD_PRUEBA/estado.json" --setenv=PYTHONDONTWRITEBYTECODE=1 \
    /usr/bin/python3 -I -c "import sys; sys.path.insert(0, '$REPO'); from jax.ejecutor.contratos.freno import principal; sys.exit(principal())"
  python3 -c 'import time; time.sleep(2)'
  sudo cat "/run/$UNIDAD_PRUEBA/estado.json"; echo
}

paso "3. simulacro remoto con el barrido habilitado (tiene que dar c4_vivo=true)"
freno_de_prueba "$MAQUINA"
esperar_cuenta_vacia
( cd "$REPO" && JAX_EJECUTOR_FRENO_REMOTOS="$MAQUINA" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:las_manos \
    python3 scripts/ejecutor_contratos/probar_c4.py --remoto "$MAQUINA" ) | tee "$TMP/verde.txt"
grep -qx 'c4_vivo=true freno="ejecutor" rondas=2 remoto="'"$MAQUINA"'"' "$TMP/verde.txt"

paso "4. VERLO FALLAR: el mismo simulacro con el barrido remoto apagado"
freno_de_prueba ""
esperar_cuenta_vacia
set +e
( cd "$REPO" && JAX_EJECUTOR_FRENO_REMOTOS="$MAQUINA" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:las_manos \
    python3 scripts/ejecutor_contratos/probar_c4.py --remoto "$MAQUINA" ) | tee "$TMP/rojo.txt"
set -e
grep -q 'codigo="procesos_remotos_vivos"' "$TMP/rojo.txt"
grep -q 'codigo="marca_creada"' "$TMP/rojo.txt"
grep -q '^c4_vivo=false' "$TMP/rojo.txt"
sudo systemctl stop "$UNIDAD_PRUEBA"
dentro "pkill -KILL -u $C; true"

paso "5. revertir_en_maquina.sh $MAQUINA: la llave del freno ya no entra"
"$REPO/ops/ejecutor/revertir_en_maquina.sh" "$MAQUINA"
set +e
sudo ssh -i "$JAX_EJECUTOR_FRENO_LLAVE" -o BatchMode=yes -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes \
  -o UserKnownHostsFile="$HOME_A/.ssh/known_hosts" -o PreferredAuthentications=publickey -p "$PUERTO" "$C@$IP" true 2> "$TMP/denegado.txt"
RC=$?
set -e
cat "$TMP/denegado.txt"
test "$RC" = 255 && grep -q "Permission denied (publickey" "$TMP/denegado.txt"
# La reversión también sacó la máquina del known_hosts del freno (con la llave de host del contenedor).
! grep -qxF -f "$TMP/host.key" "$TMP/freno_known_hosts"
dentro "test ! -e /usr/local/sbin/ejecutor-freno-remoto"
echo "freno_remoto_contenedor=ok"
