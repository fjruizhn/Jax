#!/usr/bin/env bash
# Se niega a arrancar si el checkout de produccion no esta limpio y en master.
#
# POR QUE EXISTE (2026-09-20). Servicios de produccion corrian desde checkouts
# DE TRABAJO (/home/fruiz/jax, que ese dia estaba en la rama
# phase1/pure-authority-resolver de otro agente; /home/fruiz/jax-platform).
# Cualquier reinicio -- incluso uno automatico por Restart= -- habria desplegado
# codigo sin revisar. Y lo leido de disco en caliente (instrucciones del
# auditor, canarios de C5) ya salia de esa rama, sin reinicio de por medio.
# La separacion de directorios es el arreglo; esto es el freno que lo cumple.
#
# safe.directory: el servicio corre como `jaxsvc` y el checkout del frontend es
# de `fruiz` (el vite dev corre como fruiz y necesita escribir). Sin esto git
# aborta con "detected dubious ownership" y el freno frenaria por el motivo
# EQUIVOCADO -- impidiendo arrancar para siempre. Se declara por invocacion, no
# en la config global de jaxsvc, y solo se corren `rev-parse` y `status`, que no
# ejecutan hooks del repo.
set -u
RUTA="${1:?uso: $0 <ruta del checkout>}"
G=(git -c "safe.directory=$RUTA" -C "$RUTA")

falla() { echo "jax-checkout-de-produccion-sano: $*" >&2; exit 1; }

[ -d "$RUTA/.git" ] || falla "$RUTA no es un checkout de git"

rama=$("${G[@]}" rev-parse --abbrev-ref HEAD 2>&1) || falla "no se pudo leer la rama: $rama"
[ "$rama" = "master" ] || falla "el checkout esta en '$rama', no en master -- no se arranca"

sucio=$("${G[@]}" status --porcelain 2>&1) || falla "no se pudo leer el estado: $sucio"
[ -z "$sucio" ] || falla "el checkout tiene cambios sin comitear -- no se arranca:
$(echo "$sucio" | head -5)"

echo "jax-checkout-de-produccion-sano: $RUTA limpio y en master ($("${G[@]}" rev-parse --short HEAD))"
