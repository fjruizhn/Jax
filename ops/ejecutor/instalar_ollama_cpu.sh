#!/usr/bin/env bash
# ops/ejecutor/instalar_ollama_cpu.sh — segundo Ollama, SOLO CPU: el auditor local de C5
# (spec 2026-09-18-auditor-local-opcion.md). Corre como fruiz desde el checkout de
# producción (master); sudo para lo de root. Idempotente. La garantía de "cero GPU"
# (las cuatro variables de ollama-cpu.env.plantilla) vive en el repo, no sólo en
# /etc/jax/ollama-cpu.env: un `git diff` la audita, y este instalador la vuelve a
# escribir completa en cada corrida -- nunca a mano, nunca parcial.
#
# Reversión: sudo systemctl disable --now ollama-cpu.service &&
#            sudo rm /etc/systemd/system/ollama-cpu.service /etc/jax/ollama-cpu.env &&
#            sudo systemctl daemon-reload
set -euo pipefail
: "${JAX_OLLAMA_CPU_PUERTO:?}" "${JAX_OLLAMA_CPU_MODELOS:?}" "${JAX_OLLAMA_CPU_MODELO:?}"
REPO="$(git -C "$(dirname "$(readlink -f "$0")")" rev-parse --show-toplevel)"
test "$(git -C "$REPO" branch --show-current)" = master

python3 -c "import sys; p = int(sys.argv[1]); sys.exit(0 if 0 < p < 65536 else 1)" "$JAX_OLLAMA_CPU_PUERTO" \
  || { echo "codigo=puerto_invalido puerto=\"$JAX_OLLAMA_CPU_PUERTO\"" >&2; exit 1; }
test -d "$JAX_OLLAMA_CPU_MODELOS" \
  || { echo "codigo=modelos_no_existe ruta=\"$JAX_OLLAMA_CPU_MODELOS\"" >&2; exit 1; }

sed -e "s#@PUERTO@#$JAX_OLLAMA_CPU_PUERTO#" -e "s#@MODELOS@#$JAX_OLLAMA_CPU_MODELOS#" \
  "$REPO/ops/ejecutor/ollama-cpu.env.plantilla" | sudo install -o root -g root -m 0644 /dev/stdin /etc/jax/ollama-cpu.env
sudo install -o root -g root -m 0644 "$REPO/ops/ejecutor/ollama-cpu.service" /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now ollama-cpu.service

# El canario, no un `sleep` a ciegas: espera a que el puerto responda y verifica en vivo
# que el proceso arrancó sin ningún backend de GPU (Principio VII -- un freno sin prueba
# no es freno). Si falla, el instalador falla: no queda un servicio "instalado" que en
# realidad esté compitiendo con la Mesa por la GPU.
for _ in $(seq 1 30); do
  curl -sf "http://127.0.0.1:${JAX_OLLAMA_CPU_PUERTO}/api/tags" > /dev/null 2>&1 && break
  sleep 1
done
JAX_OLLAMA_CPU_URL="http://127.0.0.1:${JAX_OLLAMA_CPU_PUERTO}" JAX_OLLAMA_CPU_MODELO="$JAX_OLLAMA_CPU_MODELO" \
  "$REPO/ops/ejecutor/canario_ollama_cpu.sh" 120

echo "ollama_cpu_instalado=true puerto=\"$JAX_OLLAMA_CPU_PUERTO\" commit=\"$(git -C "$REPO" rev-parse --short HEAD)\""
