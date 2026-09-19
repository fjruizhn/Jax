#!/usr/bin/env bash
# ops/ejecutor/canario_ollama_cpu.sh — el auditor local (ollama-cpu.service) arrancó
# SIN NINGÚN backend de GPU. No lee el journal (el formato del log es de Ollama, no
# nuestro, y puede cambiar): hace una generación REAL, mínima (`num_predict: 1`, para
# que sea rápida) y comprueba `size_vram=0` en /api/ps mientras el modelo sigue cargado.
# `size_vram > 0` significa que el modelo entró a la GPU -- exactamente lo que pasaba
# antes de agregar las cuatro variables (spec 2026-09-18-auditor-local-opcion.md, hallazgo
# de Vulkan): con sólo HIP_VISIBLE_DEVICES/ROCR_VISIBLE_DEVICES vacías, `ollama serve`
# seguía viendo la GPU discreta por el backend Vulkan.
#
# Uso: JAX_OLLAMA_CPU_URL=http://127.0.0.1:11435 JAX_OLLAMA_CPU_MODELO=qwen3:14b \
#      ops/ejecutor/canario_ollama_cpu.sh [tope_segundos]
# Sale 0 y una línea `canario_ollama_cpu=ok ...` si pasa; !=0 y `codigo=...` si no.
set -euo pipefail
: "${JAX_OLLAMA_CPU_URL:?}" "${JAX_OLLAMA_CPU_MODELO:?}"
TOPE_S="${1:-90}"

cuerpo=$(python3 -c '
import json, sys
print(json.dumps({"model": sys.argv[1], "prompt": "di OK", "stream": False, "think": False,
                   "options": {"num_predict": 1}}))
' "$JAX_OLLAMA_CPU_MODELO")

if ! salida=$(timeout "$TOPE_S" curl -sf "$JAX_OLLAMA_CPU_URL/api/generate" -d "$cuerpo"); then
  echo "codigo=generacion_fallida url=\"$JAX_OLLAMA_CPU_URL\" modelo=\"$JAX_OLLAMA_CPU_MODELO\"" >&2
  exit 1
fi
if ! echo "$salida" | jq -e '.done == true' > /dev/null 2>&1; then
  echo "codigo=respuesta_ilegible" >&2
  exit 1
fi

ps=$(curl -sf "$JAX_OLLAMA_CPU_URL/api/ps")
vram=$(echo "$ps" | jq -r --arg m "$JAX_OLLAMA_CPU_MODELO" '[.models[] | select(.model == $m)][0].size_vram // "ausente"')
if [ "$vram" = "ausente" ]; then
  echo "codigo=modelo_no_cargado modelo=\"$JAX_OLLAMA_CPU_MODELO\" ps=\"$ps\"" >&2
  exit 1
fi
if [ "$vram" != "0" ]; then
  echo "codigo=gpu_en_uso size_vram=$vram modelo=\"$JAX_OLLAMA_CPU_MODELO\"" >&2
  exit 1
fi
echo "canario_ollama_cpu=ok size_vram=$vram modelo=\"$JAX_OLLAMA_CPU_MODELO\""
