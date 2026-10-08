#!/usr/bin/env bash
# OBSOLETO (auditoria Jax#354, 2026-10-05). Este instalador copiaba solo la unidad del worker, la
# habilitaba y arrancaba de una vez, y copiaba desde un checkout de TRABAJO (/home/fruiz/jax), no desde
# produccion. Las cinco unidades jax-memory-* se instalan segun docs/runbooks/memoria-cola-atascada.md y
# docs/runbooks/deployment-rollback.md; los timers los habilita el responsable, nunca un guion.
# Se conserva solo para reproducir la instalacion historica: exige la bandera --instalar-obsoleto y
# NO habilita ni arranca nada (lo unico que llama de systemd es `daemon-reload`, y solo en el destino real).
#
# 2026-10-05: ademas de worker y synthesis (unidad + timer), instala TODOS los drop-ins versionados de
# esas dos unidades (config/systemd/<unidad>.service.d/*.conf): el checkout de produccion con su freno,
# la cuenta de servicio, el PYTHONPATH y los limites de tiempo. Antes vivian solo en /etc/systemd/system
# y reinstalar desde el repo los perdia en silencio.
#
# Auditoria Jax#355 (MINOR 5):
# - El origen por defecto es el checkout de PRODUCCION (/srv/jax-prod/jax), no el de trabajo.
# - Los destinos se normalizan (`realpath -m`): una barra final no hace que se salte el daemon-reload.
# - Un drop-in que ya no existe en el repo se retira, pero SOLO si lo instalo este guion: cada
#   directorio de drop-ins lleva un manifiesto `.instalado-por-install-memory-scope` (systemd ignora lo
#   que no termina en .conf) y nunca se toca un .conf ajeno.
#
# JAX_SYSTEMD_SRC  origen (default: /srv/jax-prod/jax/config/systemd).
# JAX_SYSTEMD_DEST destino (default: /etc/systemd/system). Con un destino distinto del real NO se recarga
#                  systemd: es la forma de ejercitar este guion en una prueba, sin tocar el servidor.
# JAX_SYSTEMD_REAL cual es el destino «real» (default: /etc/systemd/system); existe para poder probar la
#                  comparacion con una ruta de prueba.
set -euo pipefail

if [[ "${1:-}" != "--instalar-obsoleto" ]]; then
  echo "install-memory-scope.sh esta OBSOLETO y no hace nada sin --instalar-obsoleto." >&2
  echo "Ver docs/runbooks/memoria-cola-atascada.md y docs/runbooks/deployment-rollback.md." >&2
  exit 2
fi

# Argumentos: exactamente `--instalar-obsoleto` o `--instalar-obsoleto --print-origen`. Cualquier otra
# cosa (un segundo argumento con una errata, o un tercero) se rechaza AQUI, antes de resolver o tocar nada:
# un `--print-origin` mal escrito no puede terminar en una instalacion de verdad.
if [[ $# -gt 2 || ( $# -eq 2 && "$2" != "--print-origen" ) ]]; then
  echo "install-memory-scope.sh: argumento no reconocido: '${2}'${3:+ (y mas)}. Se admite solo --instalar-obsoleto [--print-origen]." >&2
  exit 2
fi

REAL="$(realpath -m -- "${JAX_SYSTEMD_REAL:-/etc/systemd/system}")"
SD="${JAX_SYSTEMD_SRC:-/srv/jax-prod/jax/config/systemd}"
DEST="$(realpath -m -- "${JAX_SYSTEMD_DEST:-$REAL}")"
UNIDADES=(jax-memory-worker jax-memory-synthesis)
MANIFIESTO=".instalado-por-install-memory-scope"

# `--instalar-obsoleto --print-origen`: imprime el origen y el destino YA resueltos y sale sin copiar nada.
# Existe para que las pruebas comprueben el valor real del default, no un texto del archivo.
if [[ "${2:-}" == "--print-origen" ]]; then
  echo "origen=$SD"
  echo "destino=$DEST"
  exit 0
fi

echo "== Copia las unidades, sus timers y sus drop-ins (sin habilitar ni arrancar nada) =="
for unidad in "${UNIDADES[@]}"; do
  dropins="$SD/$unidad.service.d"
  if [[ ! -d "$dropins" ]]; then
    echo "FALTA $dropins: el repo no reproduciria la instalacion de $unidad" >&2
    exit 1
  fi
  install -m 644 "$SD/$unidad.service" "$DEST/"
  install -m 644 "$SD/$unidad.timer" "$DEST/"
  destino_dropins="$DEST/$unidad.service.d"
  install -d -m 755 "$destino_dropins"

  # Retira lo que este guion instalo antes y el repo ya no tiene.
  if [[ -f "$destino_dropins/$MANIFIESTO" ]]; then
    while IFS= read -r previo; do
      [[ -n "$previo" && "$previo" == *.conf && "$previo" != */* ]] || continue
      if [[ ! -f "$dropins/$previo" && -f "$destino_dropins/$previo" ]]; then
        rm -f -- "$destino_dropins/$previo"
        echo "retirado: $destino_dropins/$previo (ya no esta en el repo)"
      fi
    done < "$destino_dropins/$MANIFIESTO"
  fi

  : > "$destino_dropins/$MANIFIESTO.nuevo"
  for conf in "$dropins"/*.conf; do
    install -m 644 "$conf" "$destino_dropins/"
    basename "$conf" >> "$destino_dropins/$MANIFIESTO.nuevo"
  done
  mv -f -- "$destino_dropins/$MANIFIESTO.nuevo" "$destino_dropins/$MANIFIESTO"
done

if [[ "$DEST" == "$REAL" ]]; then
  systemctl daemon-reload
else
  echo "(destino de prueba $DEST: no se recarga systemd)"
fi
echo "LISTO. Habilitar el timer (o no) es decision del responsable."
