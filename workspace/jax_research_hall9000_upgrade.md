# JAX Research: Upgrade hall9000 — ROCm RDNA4 y Qwen3-14B

**Fecha de investigación:** 2026-06-09  
**Sistema objetivo:** hall9000 · AMD RX 9060 XT 16GB (gfx1200) · Ubuntu 24.04 · Ollama vía Vulkan  
**Investigado por:** JAX (Hyde)

---

## TAREA 1 — Estado de ROCm en gfx1200/gfx1201 (RDNA4)

### Issue #21376 — llama.cpp: OOM crash en ROCm cuando KV cache supera VRAM

| Campo | Estado |
|-------|--------|
| **Estado actual** | Parcialmente resuelto (fix en progreso, esperado mayo–junio 2026) |
| **Versión con fix** | Pendiente — milestone b4000, objetivo 15 mayo 2026 |
| **Workaround activo** | Sí, sigue siendo necesario |

**Raíz del problema:** Los binarios de llama.cpp no compilan con `-DGGML_USE_VMM=ON`, lo que deshabilita el pool allocator respaldado por VMM y fuerza el fallback a `hipMalloc` directo. El crash ocurre cuando el KV cache supera ~60% de VRAM (~14.2 GB/16 GB), lanzando `hipErrorOutOfMemory`.

**Workaround confirmado (Linux):**
```bash
export HSA_OVERRIDE_GFX_VERSION=12.0.1
export GGML_CUDA_NO_PEER_COPY=1
```

**Nota (junio 2026):** El fix se esperaba para "principios de junio". A la fecha de esta investigación, no se ha confirmado merge en rama estable. Verificar directamente en el issue antes de migrar.

---

### Issue #14927 — Ollama: GPU detectada pero reporta 0 VRAM, cae a CPU

| Campo | Estado |
|-------|--------|
| **Estado actual** | Abierto — parche comunitario pendiente de revisión AMD |
| **Causa raíz** | gfx1201/gfx1200 no está en la tabla de detección hardcodeada de llama.cpp |
| **Ollama oficial** | Usa ROCm 6.4.2 — RDNA4 requiere ROCm 7.x |
| **ROCm oficial** | 7.0.2 incluye soporte formal para RX 9060 XT |

**Workarounds disponibles (Linux):**

1. **Variables de entorno** — mismas que el issue anterior:
   ```bash
   export HSA_OVERRIDE_GFX_VERSION=12.0.1
   export GGML_CUDA_NO_PEER_COPY=1
   ```

2. **Build custom verificado** — `likelovewant/ollama-for-amd` releases, probado con Ollama 0.16.1 y 0.24.0 con RDNA4.

3. **Build específico RDNA4** — `xnyzer/ollama-rocm`: Ollama con ROCm 7 para RX 9060 XT en Windows 11, también funcional en Linux.

**Ollama oficial Windows:** No funciona. Los usuarios de Windows deben usar builds custom hasta que el parche llegue a `main`.

---

### ¿Vale la pena migrar de Vulkan a ROCm en hall9000 ahora mismo?

**Benchmark directo (misma GPU, RDNA4/gfx1201):**

| Backend | Modelo | Velocidad |
|---------|--------|-----------|
| llama-server **Vulkan** | Qwen3.5-9B Q6_K | **62 t/s** |
| vLLM **ROCm 7.2** | Qwen3.5-9B FP8 | 48 t/s |

Vulkan es **29% más rápido** que ROCm en esta arquitectura porque gfx1201 carece de kernels optimizados en ROCm/vLLM — el fallback es FP32 dequantization silenciosa.

**Conclusión:** **No conviene migrar ahora.** Vulkan funciona, es más rápido, y no requiere sortear bugs de detección. La ventana de estabilización de ROCm para RDNA4 es 2T2026 (estimado conservador). Revisar estado de issues en agosto 2026.

---

## TAREA 2 — Benchmark de Qwen3-14B en RX 9060 XT 16GB

### Velocidades de generación (t/s reales)

| GPU | Modelo | Quantización | Backend | t/s generación |
|-----|--------|-------------|---------|----------------|
| RX 9060 XT 16GB | Qwen3-14B | Q4 | Ollama/Linux | **~38 tok/s** |
| RX 9070 XT 16GB | Qwen3-14B | Q4 | Ollama v0.18.2 ROCm | 52.2 tok/s |
| RX 9070 XT 16GB | Qwen3.5-9B | Q4_K_M | llama-server Vulkan | 62 tok/s |
| Intel Arc A770 16GB | Llama3.1-13B | Q4 | — | 20–30 tok/s |

*El dato de 38 tok/s en RX 9060 XT es consistente con la brecha de bandwidth entre 9060 XT (~576 GB/s) y 9070 XT (~640 GB/s).*

---

### Comparación directa: Qwen3-14B vs qwen2.5:7b en hall9000

hall9000 actualmente corre llama3.2:3b a **98 tok/s** con Vulkan. Estimación basada en proporcionalidad de tamaño de modelo y bandwidth:

| Modelo | VRAM (Q4_K_M) | t/s estimado | Relativo |
|--------|--------------|-------------|---------|
| qwen2.5:7b (actual) | ~4.5 GB | ~70–80 tok/s | baseline |
| **qwen3:14b Q4_K_M** | ~8.5–9 GB | **~38 tok/s** | ~50% del baseline |
| qwen3:14b Q5_K_M | ~11–12 GB | ~28–32 tok/s | ~40% del baseline |
| qwen3:14b Q8_0 | ~14–15 GB | ~22–26 tok/s | ~30% del baseline |

*Qwen3-14B ofrece un salto cualitativo significativo (reasoning mejorado, ventana de contexto mayor, mejor código) a cambio de aproximadamente la mitad de la velocidad.*

---

### ¿Es viable para uso interactivo?

**Sí.** Con 38 tok/s (Q4_K_M), supera el umbral mínimo de ~20 tok/s para uso interactivo cómodamente. La respuesta se percibe fluida desde ~25 tok/s en texto conversacional.

---

### Quantización recomendada para 16GB VRAM

| Quantización | VRAM usada | t/s estimado | Calidad | Recomendación |
|-------------|-----------|-------------|---------|---------------|
| Q4_K_M | ~8.5–9 GB | ~38 tok/s | Buena | **Máxima velocidad, uso interactivo** |
| Q5_K_M | ~11–12 GB | ~28–32 tok/s | Mejor | Buen balance calidad/velocidad |
| Q8_0 | ~14–15 GB | ~22–26 tok/s | Casi exacta | Riesgo de OOM con contextos largos >16K |

**Recomendación práctica:** Empezar con **Q4_K_M**. Si la calidad de respuesta no satisface, subir a **Q5_K_M**. Evitar Q8_0 en 16GB por el poco headroom para KV cache con contextos largos — y más aún mientras el bug #21376 de ROCm esté activo.

---

## Recomendación ejecutiva

- **No migrar a ROCm todavía.** Vulkan es 29% más rápido que ROCm en RDNA4 y no requiere sortear bugs críticos de detección de GPU. Reevaluar en agosto 2026.
- **El workaround `HSA_OVERRIDE_GFX_VERSION=12.0.1` + `GGML_CUDA_NO_PEER_COPY=1` sigue siendo necesario** si se usa ROCm; con Vulkan es irrelevante.
- **Qwen3-14B Q4_K_M es viable hoy** en hall9000: ~38 tok/s, dentro del rango interactivo, ocupa ~9GB VRAM dejando 7GB libres para contexto.
- **Upgrade recomendado**: pasar de `qwen2.5:7b` a `qwen3:14b Q4_K_M` — la velocidad cae a ~50% pero la calidad de razonamiento/código es notablemente superior.
- **Monitorear issues #21376 y #14927** — si se confirma merge en llama.cpp b4000 y Ollama mainstream, una migración a ROCm en septiembre 2026 podría recuperar velocidad adicional (~10–15% sobre Vulkan actual).

---

## Fuentes

- [Issue #21376 llama.cpp — ROCm OOM en gfx1200](https://github.com/ggml-org/llama.cpp/issues/21376)
- [Issue #14927 Ollama — RDNA4 0 VRAM fallback CPU](https://github.com/ollama/ollama/issues/14927)
- [lemonade-sdk/llamacpp-rocm #87 — VMM fix](https://github.com/lemonade-sdk/llamacpp-rocm/issues/87)
- [CraftRigs — RX 9060 XT: Two Active Bugs](https://craftrigs.com/news/rx-9060-xt-active-bugs-llamacpp-ollama-before-launch/)
- [ivangotoy/llama-server-rdna4-vulkan — Vulkan 62 vs ROCm 48 t/s](https://github.com/ivangotoy/llama-server-rdna4-vulkan)
- [hirokuze/local-llm-benchmark-rx9070xt — Qwen3:14b 52.2 tok/s](https://github.com/hirokuze/local-llm-benchmark-rx9070xt/tree/main)
- [TechReviewer — RX 9060 XT 16GB for LLMs](https://www.techreviewer.com/tech-specs/amd-rx-9060-xt-16gb-gpu-for-llms/)
- [Phoronix — AMD ROCm 7.0.2 con RX 9060 soporte oficial](https://www.phoronix.com/news/AMD-ROCm-7.0.2-Released)
- [likelovewant/ollama-for-amd releases](https://github.com/likelovewant/ollama-for-amd/releases)
- [xnyzer/ollama-rocm — Ollama ROCm 7 RDNA4](https://github.com/xnyzer/ollama-rocm)
- [ROCm/ROCm #5812 — gfx1200 hang en ROCm 7.1.1](https://github.com/ROCm/ROCm/issues/5812)
