#!/usr/bin/env bash
# ops/ejecutor/instalar_vigia.sh — prepara lo que el vigía necesita para correr. Corre como
# fruiz desde el checkout de producción (master); sudo para lo de root. Idempotente.
#
# NO instala ninguna unidad systemd (ronda 5, auditoría adversarial 2026-09-22: la plantilla
# `ejecutor-vigia@.service` que este script instalaba se RETIRÓ -- código muerto, el journal
# nunca mostró un solo arranque suyo, ver DEUDA.md). El vigía (`vigia_servicio.py`) lo lanza
# `abrir_vigia` (jax/ejecutor/mision_servicio.py) como SUBPROCESO DIRECTO de cada turno,
# heredando la identidad de `jax-platform` -- CORREGIDO (bug de producción jax#260,
# 2026-09-22, este comentario decía "fruiz"; ERA FALSO desde el 2026-09-17, cuando
# jax-platform.service pasó a `User=jaxsvc`): no hay nada que instalar para eso. Lo que
# SÍ sigue haciendo falta, y es lo que queda acá, es `JAX_EJECUTOR_MISIONES` con su ACL
# (B-2, ronda 4) y la validación de los tiempos de latido.
set -euo pipefail
: "${JAX_EJECUTOR_MISIONES:?}" "${JAX_EJECUTOR_VIGIA_LATIDO:?}" "${JAX_EJECUTOR_VIGIA_LATIDO_MAX_S:?}"
: "${JAX_EJECUTOR_VIGIA_LATIDO_CADA_S:?}" "${JAX_EJECUTOR_PAUSA:?}" "${JAX_EJECUTOR_LLAVES_ROOT:?}"
: "${JAX_EJECUTOR_ADMIN_USUARIO:?}" "${JAX_EJECUTOR_CUENTA:?}"
REPO="$(git -C "$(dirname "$(readlink -f "$0")")" rev-parse --show-toplevel)"
test "$(git -C "$REPO" branch --show-current)" = master
python3 -c "import sys; c, m = float(sys.argv[1]), float(sys.argv[2]); sys.exit(0 if 0 < c < m else 1)" \
  "$JAX_EJECUTOR_VIGIA_LATIDO_CADA_S" "$JAX_EJECUTOR_VIGIA_LATIDO_MAX_S" \
  || { echo "codigo=latido_cada_invalido" >&2; exit 1; }
# Misiones: las escribe fruiz (el transporte de SP2) y también las mide (M-1, ronda 4: la
# huella la toma el controlador COMO FRUIZ). B-2 (ronda 4): ACL EXPLÍCITA, no sólo el modo
# -- ver ops/ejecutor/preparar_directorio_misiones.sh para el porqué completo. `axioma`
# recibe SÓLO travesía (--x, sin +r): puede entrar a `<mision_id>/claude-projects` si ya
# sabe el UUID exacto (se lo da la propia misión que está corriendo), pero no puede LISTAR
# este directorio ni leer `<id>.json` de ninguna misión. Cada subcarpeta la crea
# `cuenta_axioma.preparar_directorio_projects` con `sudo install -d -o axioma -g axioma -m
# 0700`, así que ni con el --x de acá alcanza para tocar la de OTRA misión sin adivinar su
# UUID Y que ese UUID sea, además, dueño=axioma -- que lo es siempre, así que el límite real
# es "conocer el UUID de la misión en curso", el mismo que ya protege todo lo demás del
# Ejecutor (machine-id, políticas, etc.).
"$REPO/ops/ejecutor/preparar_directorio_misiones.sh" "$JAX_EJECUTOR_MISIONES" "$JAX_EJECUTOR_ADMIN_USUARIO" \
  "$JAX_EJECUTOR_CUENTA"
echo "vigia_instalado=true misiones=\"$JAX_EJECUTOR_MISIONES\" commit=\"$(git -C "$REPO" rev-parse --short HEAD)\""
