# Umbral de decisión — GPU_SEMAPHORE cross-proceso

Definido **antes** de correr la medición real, para no elegir el umbral
después de ver el número y justificar el resultado que se prefiera.
Escrito el 2026-08-25, **ampliado el 2026-08-28 antes de medir** con el
discriminante de la sección 2 — la ampliación se hizo con el baseline de
`concurrency=1` ya corrido (74.0 tok/s) y **sin** haber corrido todavía
`concurrency=2` ni `3`, que es lo que el umbral decide.

## 1. Criterios originales (sin cambios)

Se considera que **SÍ** hace falta exclusión mutua cross-proceso (dejar
constancia de que el fix queda pendiente, fuera de este plan) si, en
`concurrency=2` o `concurrency=3` (mediana de 3 corridas cada uno):

- degradación de tok/s por request **> 25%** respecto al baseline de
  `concurrency=1`, **O**
- cualquier error HTTP / timeout / respuesta inválida de Ollama que **no**
  ocurra en `concurrency=1`, **O**
- pico de VRAM en una sola corrida **> 90% de los 32 GB** totales (riesgo
  real de OOM bajo carga de producción, no sólo del probe).

## 2. Ampliación obligatoria: el wall-clock, o el umbral no puede fallar

`OLLAMA_NUM_PARALLEL` está en **1** en este servidor (valor efectivo leído
del log de arranque de Ollama, no de la variable de entorno, que no está
seteada). Con Ollama serializando, dos requests concurrentes **se encolan**:
cada una reporta su propio `eval_count / elapsed` casi intacto, así que
**la degradación de tok/s daría ~0% y ninguno de los tres criterios de
arriba podría dispararse jamás** — no porque la GPU aguante, sino porque
nadie la usa en paralelo. Un umbral que no puede dar positivo no es un
umbral.

Por eso el veredicto se lee con el **factor de wall-clock** de la ronda
(`wall_clock(N) / wall_clock(1)`), que es lo único que distingue los dos
mundos:

| factor wall | degradación tok/s | Qué pasó de verdad |
|---|---|---|
| ≈ 1.0 | ≈ 0% | **Paralelismo real y sin contención.** Es el único caso en que "no hace falta semáforo" significa lo que parece. |
| ≈ 1.0 | > 25% | Paralelismo real **con** contención → hace falta exclusión mutua. |
| ≈ N | ≈ 0% | **Encolado.** Ollama ya serializa: la exclusión mutua **ya existe**, provista por Ollama, no por la GPU. Ver §3. |
| ≈ N | > 25% | Encolado **y** degradado: lo peor, hace falta exclusión mutua y además hay algo más pasando. |

## 3. Si el resultado es "encolado" — qué se puede concluir y qué no

**Se puede concluir:** las tres rutas que le pegan a Ollama sin pasar por
`GPU_SEMAPHORE` no pueden hoy sobrecargar la GPU en paralelo, porque Ollama
las serializa antes. El semáforo cross-proceso no cierra un riesgo activo.

**NO se puede concluir que el riesgo no exista.** La garantía la da un
valor de configuración (`OLLAMA_NUM_PARALLEL = 1`) que:
- **nadie fija explícitamente** — Ollama lo eligió solo, según la memoria
  disponible al arrancar. Con otro modelo residente más chico, la misma
  versión de Ollama puede elegir 4;
- **no está declarado en ningún `systemd` unit ni `.env`** del repo, así
  que no hay nada que lo defienda ni que avise si cambia;
- **no lo cubre ningún test ni guard.**

En ese caso el veredicto correcto NO es "cerrado" sino **"no hace falta
hoy, y depende de una condición que nadie vigila"** — y la acción que
corresponde es fijar `OLLAMA_NUM_PARALLEL=1` explícitamente en el unit de
Ollama, o registrar la dependencia. Es el mismo patrón que un guard cuyo
umbral lo desactiva (`min_context_tokens = 0`): el control existe, pasa, y
no miró nada.

## 4. Si ninguna condición de la §1 se cumple

Se documenta con los números reales en
`docs/superpowers/specs/2026-08-25-gpu-concurrency-resultado.md` (**no** en
`DEUDA.md` — política explícita del usuario para este plan), y se corrige
el comentario falso en `ollama_muscle.py` ("compartido por toda carga GPU
futura", siendo un `asyncio.Semaphore` de proceso). **Esa corrección es
incondicional: el comentario es falso hoy, sin importar lo que mida el
probe.**

## 5. Desvío del plan, declarado

El plan nombra `qwen3-coder:30b`. Se mide con **`qwen3.6:35b-a3b-q4_K_M`**,
el primario real de `jax_local` según `facet_binding` (verificado, no
supuesto). Dos razones, y las dos son de exactitud:
1. medir un modelo que producción no usa responde otra pregunta — la
   concurrencia depende del modelo (VRAM por instancia, KV cache, MoE vs
   denso);
2. `qwen3-coder:30b` (18.6 GB) **no entra** junto al residente (24 GB de
   32): cargarlo desalojaría el modelo de producción, que hoy tiene
   keep-alive infinito a propósito, y el siguiente turno de chat real
   pagaría la recarga.
