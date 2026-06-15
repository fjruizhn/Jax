# JAX — Investigación Técnica: Upgrade hall9000
**Fecha:** 2026-06-09  
**Sistema:** Ryzen 5 8500G · 32GB RAM · AMD RX 9060 XT 16GB RDNA4 (gfx1200)  
**Equipo:** Hipatia (grounding) + Jekyll (contraste de fuentes) + jax_local (síntesis)

---

## TAREA 1 — Estado de ROCm en gfx1200/gfx1201 (RX 9060 XT)

### Contexto base: soporte upstream ROCm

| Versión ROCm | Cambio relevante |
|---|---|
| 7.0.2 | Primera versión con soporte oficial gfx1200/gfx1201 (hipBLAS, rocBLAS) |
| 7.2.0 | RX 9060 XT añadida explícitamente a la matriz de compatibilidad |
| 7.2+ | PyTorch y JAX framework soportan gfx1200/gfx1201 |

Soporte en ROCm ≠ soporte en binarios de Ollama/llama.cpp distribuidos.

---

### Issue #21376 — llama.cpp: OOM crash ROCm / KV cache supera VRAM

**Estado: Abierto. Sin fix en upstream oficial a junio 2026.**

**Causa raíz:** El allocator HIP sin `GGML_USE_VMM=ON` usa `hipMalloc` directo con pool que crece pero nunca libera páginas. La fragmentación acumulativa provoca crash `hipErrorOutOfMemory` cuando el KV cache supera ~60–78% de VRAM (~14.2 GB de 16 GB). Vulkan maneja el mismo escenario con spill graceful a RAM del sistema.

**Fix disponible:** Compilar llama.cpp desde fuente con:
```bash
cmake -DGGML_USE_VMM=ON ...
```
Reserva 32 GB de espacio de direcciones virtuales por device y mapea páginas físicas bajo demanda. Documentado en `lemonade-sdk/llamacpp-rocm#87`.

**Workaround sin compilar:**
```bash
export GGML_CUDA_NO_PEER_COPY=1
export HSA_OVERRIDE_GFX_VERSION=12.0.1
```
Extiende VRAM efectiva utilizable a ~78% (≈15.1 GB de 16 GB). **No incorporado upstream por defecto.** No tiene efecto equivalente en Windows.

**Milestone objetivo:** b4000 / mayo 2026 — no confirmado cerrado en release notes.

---

### Issue #14927 — Ollama: GPU detectada pero reporta 0 VRAM, cae a CPU

**Estado: Parcialmente resuelto en Linux con builds comunitarias. Sin fix en Ollama oficial para Windows.**

**Causa raíz (dos capas):**
1. Tabla de detección de Ollama no incluía gfx1201 originalmente.
2. Binarios distribuidos con ROCm 6.x no tienen kernels Tensile precompilados para gfx1201 (`TensileLibrary_lazy_gfx1201.dat` ausente).

**Historial:**

| Fecha | Evento |
|---|---|
| 2025-03-27 | PR #9878 mergeado a main — añade constante `GGML_CUDA_CC_RDNA4` y targets gfx1200/gfx1201 |
| 2026-02 | Issue #14927 abierto — el fix del PR no es suficiente en todos los escenarios |
| 2026-06 | `docs.ollama.com/gpu` lista gfx1200/gfx1201 como soportados en Linux bajo ROCm v7 |
| 2026-06 | Ollama 0.30.4–0.30.7: sin mención de fix específico para gfx1201 en release notes |

**Bug secundario:** `ROCm/rocm-libraries#7192` — rocBLASLt busca `gfx1200.dat` en lugar de `gfx1201.dat`, provoca SIGKILL en Ollama v0.7.22.1.

**Soluciones verificadas en Linux:**

| Proyecto | Versión Ollama | ROCm | Estado gfx1201 |
|---|---|---|---|
| `xnyzer/ollama-rocm` | 0.16.1 / 0.24.0 | 7.1.1 | Funcional — no requiere `HSA_OVERRIDE_GFX_VERSION` |
| `likelovewant/ollama-for-amd` | hasta v0.20.8 | 6.4.2 / 7.1.1 | Experimental |
| Level1Techs Docker build | 0.15.4 | 7.2 | Funcional — build manual |

**En Windows:** Ollama oficial usa ROCm 6.4.2. Sin fix para gfx1201 en el instalador oficial a junio 2026. Ruta viable: builds de `xnyzer/ollama-rocm` o LM Studio 0.3.12+ con selección manual de GPU.

---

### Estado de los workarounds

| Variable / Flag | ¿Necesario en junio 2026? | Estado |
|---|---|---|
| `HSA_OVERRIDE_GFX_VERSION=12.0.1` | Solo con ROCm 6.x o Ollama oficial sin parche | No necesario con ROCm 7.x + kernels nativos |
| `GGML_CUDA_NO_PEER_COPY=1` | Sí, para mitigar OOM (issue #21376) | No incorporado upstream por defecto |
| `-DGGML_USE_VMM=ON` (build flag) | Necesario al compilar desde fuente | No habilitado en binarios oficiales |

---

### ¿Vale la pena migrar de Vulkan a ROCm ahora mismo?

**Benchmarks verificados — llama.cpp Q4, métricas pp512/tg128 (knightli.com, abril 2026):**

| Backend | pp512 (t/s) | tg128 (t/s) |
|---|---|---|
| ROCm (sin FA) | 1419.67 ± 3.64 | 67.58 ± 0.24 |
| Vulkan (sin FA) | 2141.67 ± 6.87 | **70.54 ± 0.74** |

**RX 9070 XT — benchmark digtvbg.com:**

| Backend | tok/s |
|---|---|
| vLLM/ROCm | 48 |
| llama-server/Vulkan | **62** |

**Veredicto: No conviene migrar de Vulkan a ROCm en junio 2026.**

Vulkan supera a ROCm en pp512 (+50%) y tg128 (+4%) en RDNA4. La ventaja no es arquitectural — es que las builds ROCm aún no tienen kernels FP8 optimizados para gfx1200/gfx1201. Migrar a ROCm requiere: instalar ROCm 7.x manualmente + build comunitaria de Ollama/llama.cpp con kernels nativos + compilar con `-DGGML_USE_VMM=ON` + aplicar `GGML_CUDA_NO_PEER_COPY=1`. Alta fricción con rendimiento inferior actual.

---

## TAREA 2 — Benchmark de Qwen3-14B en Vulkan / ROCm con RDNA4 16GB

### Datos reales medidos

| Hardware | Backend | Quant | tok/s gen | Fuente |
|---|---|---|---|---|
| RX 9070 XT 16GB (gfx1201) | Ollama ROCm (Docker) | Q4 | **52.2** | github.com/hirokuze |
| RX 9070 XT 16GB (gfx1201) | Ollama ROCm | Q4 | **47–49** | craftrigs.com |
| AMD R9700 (gfx1201) | ROCm — tg128 | Q4_K_M | 24.03 | knightli.com |
| AMD R9700 (gfx1201) | Vulkan — tg128 | Q4_K_M | 28.20 | knightli.com |
| AMD BC-250 APU 16GB GDDR6 | Vulkan | Q4_K_M | 27 | github.com/akandr |
| RTX 4080 16GB (referencia) | CUDA | Q4_K_M | **61.85** | glukhov.org |

**Nota:** No existen benchmarks directos publicados de Qwen3-14B sobre RX 9060 XT (gfx1200) a la fecha. El hardware más cercano disponible es el RX 9070 XT (misma arquitectura, 36 CUs vs 32, ~640 vs ~384 GB/s ancho de banda de memoria).

**Proyección para RX 9060 XT:** La inferencia LLM es memory-bandwidth bound. Con ~60% del ancho de banda del 9070 XT, la proyección conservadora es **28–35 tok/s** en Qwen3-14B Q4_K_M. Supera el umbral de 20 tok/s con margen moderado.

---

### Comparación con qwen2.5:7b en RDNA4

No existe benchmark directo publicado de qwen2.5:7b en RDNA4 con Ollama. Dato de referencia:

| Hardware | Backend | Modelo | tok/s |
|---|---|---|---|
| RX 9070 XT | Ollama ROCm | qwen3:14b | 52.2 |
| RX 9070 XT | Ollama ROCm | qwen3.5:9b | 57.8 |

La diferencia 9B→14B es ~10% en el mismo hardware. Un modelo 7B estimado: **65–80 tok/s** — sin medición directa en RDNA4.

---

### ¿Es viable para uso interactivo en tiempo real? (umbral: 20 tok/s)

| Escenario | Viabilidad | Velocidad esperada |
|---|---|---|
| RX 9070 XT — Linux — ROCm Docker | Confirmado | ~50 tok/s |
| RDNA4 — Vulkan backend | Confirmado | ~28–30 tok/s |
| RX 9060 XT — Linux — ROCm 7.x comunitario | Viable con bugs activos | ~28–35 tok/s (proyectado) |
| RX 9060 XT — Windows — Ollama stock | No funcional | 4–8 tok/s (corre en CPU) |

**En hall9000 con Vulkan (configuración actual):** Viable. ~28–35 tok/s proyectados para Qwen3-14B, por encima del umbral de 20 tok/s.

---

### Quantización: tamaños reales y recomendación

**Tamaños GGUF de Qwen3-14B (bartowski/Qwen_Qwen3-14B-GGUF, HuggingFace):**

| Quant | Tamaño GGUF | VRAM estimada (KV 8K ctx) | Velocidad relativa |
|---|---|---|---|
| Q4_K_M | 9.00 GB | ~13–14 GB | referencia |
| **Q5_K_M** | **10.51 GB** | **~14–15 GB** | **~5% más lenta** |
| Q6_K | 12.12 GB | ~15.5 GB | ~10% más lenta |
| Q8_0 | 15.70 GB | >16 GB (no entra completo) | — |

**Q8_0 no entra completo en 16GB.** Con overhead del runtime y KV cache mínimo supera los 16 GB, provocando offload parcial a RAM y destruyendo la velocidad.

**Recomendación: Q5_K_M**

- Cabe con holgura: 10.5 GB modelo + ~3–4 GB KV cache 8K ctx ≈ 14 GB total
- Calidad notablemente superior a Q4_K_M (Qwen3 es sensible a cuantización en tareas de razonamiento)
- Penalización de velocidad marginal (~5%) — la inferencia LLM es memory-bandwidth bound, no compute bound
- Q4_K_M es la alternativa si se necesita contexto >8K o mayor margen de VRAM

---

## Recomendación Ejecutiva

- **No migrar de Vulkan a ROCm en junio 2026.** Vulkan supera a ROCm en pp512 (+50%) y tg128 (+4%) en RDNA4. ROCm en gfx1200/gfx1201 está en proceso de maduración; los bugs #21376 y #14927 siguen sin fix en binarios oficiales.

- **Qwen3-14B Q5_K_M es viable en hall9000.** Proyección: ~28–35 tok/s con Vulkan, sobre el umbral interactivo de 20 tok/s. El modelo entra completo en 16 GB con margen para KV cache de 8K ctx.

- **Para probar ROCm en el futuro:** Usar `xnyzer/ollama-rocm` (ROCm 7.1.1 con kernels gfx1200/gfx1201 nativos), compilar llama.cpp con `-DGGML_USE_VMM=ON`, y exportar `GGML_CUDA_NO_PEER_COPY=1`. No usar Ollama oficial mientras no incluya ROCm 7.x.

- **qwen2.5:7b sigue siendo la opción rápida.** Sin dato medido en RDNA4, pero a ~98 tok/s actual con llama3.2:3b Vulkan, un modelo 7B daría ~65–80 tok/s — claramente superior para uso interactivo de baja latencia.

- **Revisar en Q4 2026.** El soporte RDNA4 en ROCm está activo; es probable que los issues se cierren en los próximos meses. El milestone b4000 de llama.cpp y futuras versiones de Ollama son los puntos de revisión clave.

---

## Fuentes

| Fuente | Relevancia |
|---|---|
| [llama.cpp #21376](https://github.com/ggml-org/llama.cpp/issues/21376) | OOM crash ROCm gfx1200/1201 |
| [Ollama #14927](https://github.com/ollama/ollama/issues/14927) | 0 VRAM gfx1201, fallback CPU |
| [Ollama PR #9878](https://github.com/ollama/ollama/pull/9878) | Merge soporte gfx1200/gfx1201 Linux |
| [lemonade-sdk/llamacpp-rocm #87](https://github.com/lemonade-sdk/llamacpp-rocm/issues/87) | Fix GGML_USE_VMM |
| [xnyzer/ollama-rocm](https://github.com/xnyzer/ollama-rocm) | Build comunitaria ROCm 7 RDNA4 |
| [likelovewant/ollama-for-amd](https://github.com/likelovewant/ollama-for-amd/releases) | Build comunitaria gfx1201 experimental |
| [craftrigs.com — bugs RX 9060 XT](https://craftrigs.com/news/rx-9060-xt-active-bugs-llamacpp-ollama-before-launch/) | Bugs activos documentados + workarounds |
| [craftrigs.com — RX 9070 XT vs RTX 4080](https://craftrigs.com/comparisons/rx-9070-xt-vs-rtx-4080-super-local-llm/) | Qwen3:14b 47–49 tok/s AMD |
| [github.com/hirokuze](https://github.com/hirokuze/local-llm-benchmark-rx9070xt/tree/main) | Qwen3:14b 52.2 tok/s RX 9070 XT ROCm |
| [digtvbg.com](https://digtvbg.com/blog/llama-server-vulkan-rdna4-vllm-rocm-benchmark/) | Vulkan vs ROCm RDNA4: 62 vs 48 tok/s |
| [knightli.com](https://knightli.com/en/2026/04/23/llama-cpp-gpu-benchmark-cuda-rocm-vulkan-scoreboard/) | Scoreboard CUDA/ROCm/Vulkan abril 2026 |
| [glukhov.org](https://www.glukhov.org/post/2026/01/choosing-best-llm-for-ollama-on-16gb-vram-gpu/) | RTX 4080 referencia Qwen3:14b 61.85 tok/s |
| [github.com/akandr/bc250](https://github.com/akandr/bc250) | BC-250 APU Vulkan Qwen3:14b 27 tok/s |
| [bartowski/Qwen_Qwen3-14B-GGUF](https://huggingface.co/bartowski/Qwen_Qwen3-14B-GGUF) | Tamaños exactos por cuantización |
| [ROCm 7.0.2 release notes](https://rocm.docs.amd.com/en/docs-7.0.2/about/release-notes.html) | Primer soporte oficial gfx1200/1201 |
| [ROCm 7.2.0 release notes](https://rocm.docs.amd.com/en/docs-7.2.0/about/release-notes.html) | RX 9060 XT LP añadida a matriz |
| [docs.ollama.com/gpu](https://docs.ollama.com/gpu) | gfx1200/1201 listados soportados Linux |
| [ROCm/rocm-libraries #7192](https://github.com/ROCm/rocm-libraries/issues/7192) | rocBLASLt gfx1201 lookup error |
| [Level1Techs forum](https://forum.level1techs.com/t/ollama-0-15-4-with-rocm-7-2-and-gfx1201/245568) | Ollama 0.15.4 + ROCm 7.2 + gfx1201 funcional |
