# ops/ejecutor/_maquina.sh — se incluye (.) desde instalar_en_maquina.sh y revertir_en_maquina.sh.
# Resuelve NOMBRE en la política (IP, PUERTO, LOCAL) y define `corre` (root en la máquina) y `sube`.
read -r IP PUERTO LOCAL < <(cd "$REPO" && PYTHONDONTWRITEBYTECODE=1 python3 - "$JAX_EJECUTOR_POLITICA" "$NOMBRE" <<'PY'
import json, sys
from jax.ejecutor.contratos import politica
(h,) = [h for h in politica.validar(json.load(open(sys.argv[1]))).hosts if h.nombre == sys.argv[2]]
print(h.ip, h.puerto, "si" if h.es_local else "no")
PY
)
test -n "$IP"
if [ "$LOCAL" = si ]; then
  ADMIN_LOCAL="$(id -un)"
  corre() { sudo -n sh -c "$1"; }
  sube() { sudo -n install -o root -g root -m "$3" "$1" "$2"; }
  esperar_sshd() { :; }
else
  ADMIN_LOCAL="$JAX_EJECUTOR_ADMIN_USUARIO"
  SSH_OPC=(-o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=yes)
  corre() { ssh "${SSH_OPC[@]}" -p "$PUERTO" "$JAX_EJECUTOR_ADMIN_USUARIO@$IP" "sudo -n sh -c $(printf %q "$1")"; }
  # Después de `systemctl reload` sshd se re-ejecuta (SIGHUP) y durante un instante rechaza conexiones:
  # la siguiente orden caía con «Connection refused» y dejaba la instalación a medias (visto en el
  # contenedor de prueba del freno, 2026-09-17). Se espera a que vuelva a atender, con tope.
  esperar_sshd() {
    local i
    for i in $(seq 20); do
      ssh "${SSH_OPC[@]}" -p "$PUERTO" "$JAX_EJECUTOR_ADMIN_USUARIO@$IP" true 2>/dev/null && return 0
      python3 -c 'import time; time.sleep(0.5)'
    done
    echo "codigo=sshd_no_vuelve" >&2
    return 1
  }
  sube() {
    local tmp="/tmp/ejecutor-subida-$$"
    scp -q "${SSH_OPC[@]}" -P "$PUERTO" "$1" "$JAX_EJECUTOR_ADMIN_USUARIO@$IP:$tmp" \
      && corre "install -o root -g root -m $3 $tmp $2 && rm -f $tmp"
  }
fi
