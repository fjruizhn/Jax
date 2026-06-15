# JAX — Investigación Técnica: Upgrade hall9000

**Para:** jax_local / Hipatia / Jekyll  
**De:** Fernando  
**Fecha:** 2026-06-09

---

## CONTEXTO DEL SISTEMA

hall9000 corre:
- Ubuntu 24.04.4, kernel 6.17
- Ryzen 5 8500G, 32GB RAM
- AMD RX 9060 XT 16GB RDNA4 (gfx1200)
- Ollama via **Vulkan** (ROCm diferido por bugs)
- Velocidad actual: ~98 tok/s con llama3.2:3b
- jax_local usa: qwen2.5:7b

---

## TAREA 1 — Estado de ROCm en gfx1200/gfx1201 (RX 9060 XT)

Investiga el estado actual de los siguientes issues para arquitectura RDNA4 gfx1200/gfx1201:

- **Issue #21376** (llama.cpp): OOM crash en ROCm cuando KV cache supera VRAM
- **Issue #14927** (Ollama): GPU detectada pero reporta 0 VRAM, cae a CPU

Preguntas específicas:

1. ¿Están resueltos, parcialmente resueltos, o siguen abiertos?
2. ¿Qué versión de Ollama y/o llama.cpp los resuelve si ya hay fix?
3. ¿El workaround `HSA_OVERRIDE_GFX_VERSION=12.0.1` + `GGML_CUDA_NO_PEER_COPY=1` sigue siendo necesario o ya está incorporado upstream?
4. ¿Vale la pena migrar de Vulkan a ROCm en hall9000 ahora mismo, o conviene esperar?

---

## TAREA 2 — Benchmark de Qwen3-14B en Vulkan con RX 9060 XT 16GB

Busca benchmarks reales (no teóricos) de Qwen3-14B corriendo en Ollama con Vulkan o ROCm sobre RX 9060 XT o hardware equivalente RDNA4 con 16GB VRAM.

Preguntas específicas:

1. ¿Cuántos tokens/segundo en generación (tg128 o equivalente)?
2. Comparación directa con qwen2.5:7b en el mismo hardware
3. ¿Es viable para uso interactivo en tiempo real? (mínimo aceptable: ~20 tok/s)
4. ¿Qué quantización recomendás — Q4_K_M, Q5_K_M, Q8 — para balancear velocidad y calidad dentro de 16GB VRAM?

---

## FORMATO DE ENTREGA

Genera tu respuesta como documento Markdown estructurado con:

- Título y fecha de investigación
- Una sección por tarea con subtítulos claros
- Tabla comparativa donde aplique
- Sección final **"Recomendación ejecutiva"** en máximo 5 bullets

Guarda el resultado como: `jax_research_hall9000_upgrade.md`

Usa fuentes recientes (2026). No suposiciones — solo lo que puedas verificar.
