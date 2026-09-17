#!/usr/bin/env bash
# ops/ejecutor/instalar_freno.sh [--rama-aprobada] — C4 en hall9000: freno root de la cuenta del Ejecutor.
# Corre como fruiz; sudo para lo de root. Idempotente. Lee del entorno (set -a; . /etc/jax/.env).
# Reversión: ops/ejecutor/revertir_freno.sh.
set -euo pipefail
: "${JAX_EJECUTOR_LIB:?}" "${JAX_EJECUTOR_CUENTA:?}" "${JAX_EJECUTOR_ADMIN_USUARIO:?}" "${JAX_EJECUTOR_POLITICA:?}"
: "${JAX_EJECUTOR_FRENO_LLAVE:?}" "${JAX_EJECUTOR_FRENO_KNOWN_HOSTS:?}" "${JAX_EJECUTOR_FRENO_ESTADO:?}"
: "${JAX_KILL_SWITCH_PATH:?}" "${JAX_EJECUTOR_PAUSA:?}"
# Vacía hasta que la llave del freno esté en cada remota (instalar_en_maquina.sh sin --sin-freno).
: "${JAX_EJECUTOR_FRENO_REMOTOS?}"
REPO="$(git -C "$(dirname "$(readlink -f "$0")")" rev-parse --show-toplevel)"
RAMA="$(git -C "$REPO" branch --show-current)"
test "$RAMA" = master || test "${1:-}" = --rama-aprobada
C="$JAX_EJECUTOR_CUENTA"

# 0. Precondiciones. La cuenta no es root, de sistema ni el administrador (el freno se negaría
#    y no mataría nada). Ningún servicio corre COMO la cuenta: el freno lo mataría (el proxy de
#    C3 corre como el administrador, fuera de user-<uid>.slice).
UID_C="$(id -u "$C")"
test "$UID_C" -ge 1000 && test "$UID_C" != "$(id -u "$JAX_EJECUTOR_ADMIN_USUARIO")"
SERVICIOS_DE_LA_CUENTA="$(systemctl show '*' -p Id -p User --state=active 2>/dev/null | awk -v c="$C" '/^Id=/{id=$0} $0=="User="c{print id}')"
test -z "$SERVICIOS_DE_LA_CUENTA" || { echo "codigo=servicio_corre_como_la_cuenta $SERVICIOS_DE_LA_CUENTA" >&2; exit 1; }
test "$(dirname "$JAX_EJECUTOR_PAUSA")" = "$(dirname "$JAX_KILL_SWITCH_PATH")"
sudo test -d "$(dirname "$JAX_KILL_SWITCH_PATH")"

# 1. Biblioteca (gancho, freno, interruptor, pausa) y la unidad renderizada: el instalador de C1.
"$REPO/ops/ejecutor/instalar_contratos.sh" "${1:-}"

# 2. Llave del freno: root, nunca en la cuenta. La pública va a cada remota con comando forzado
#    (ops/ejecutor/instalar_en_maquina.sh <maquina>, sin --sin-freno).
sudo install -d -o root -g root -m 0700 "$(dirname "$JAX_EJECUTOR_FRENO_LLAVE")"
sudo test -e "$JAX_EJECUTOR_FRENO_LLAVE" || sudo ssh-keygen -q -t ed25519 -N '' -C ejecutor-freno -f "$JAX_EJECUTOR_FRENO_LLAVE"
sudo test -e "$JAX_EJECUTOR_FRENO_KNOWN_HOSTS" || sudo install -o root -g root -m 0644 /dev/null "$JAX_EJECUTOR_FRENO_KNOWN_HOSTS"

# 3. Lo que sacaría procesos de la cuenta de su cgroup: cron, at y linger.
for f in /etc/cron.deny /etc/at.deny; do
  sudo touch "$f"
  sudo grep -qx "$C" "$f" || echo "$C" | sudo tee -a "$f" >/dev/null
done
test ! -e /etc/cron.allow || { echo "codigo=cron_allow_existe" >&2; exit 1; }
test ! -e /etc/at.allow || { echo "codigo=at_allow_existe" >&2; exit 1; }
sudo loginctl disable-linger "$C"
test ! -e "/var/lib/systemd/linger/$C"

# 4. La unidad, desde la copia verificada por el manifiesto.
sudo install -o root -g root -m 0644 "$JAX_EJECUTOR_LIB/ejecutor-freno.service" /etc/systemd/system/ejecutor-freno.service
sudo systemctl daemon-reload
sudo systemctl enable ejecutor-freno.service
sudo systemctl restart ejecutor-freno.service
sleep 2
python3 - "$JAX_EJECUTOR_FRENO_ESTADO" <<'PY'
import json, sys, time
e = json.load(open(sys.argv[1]))
assert time.time() - e["momento"] < 2 and e["uid_resuelto"] is True, e
print(f"freno_latido=ok activo={str(e['activo']).lower()} remotos_cargados={str(e['remotos_cargados']).lower()} "
      f"remotos=\"{','.join(e['remotos'])}\"")
PY
