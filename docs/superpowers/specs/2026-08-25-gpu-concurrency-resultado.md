# Resultado de la medición de concurrencia de GPU — medido 2026-08-28

Umbral acordado **antes** de medir (ver
`2026-08-25-gpu-concurrency-umbral.md`): > 25% de degradación de tok/s,
cualquier error nuevo, o > 90% de VRAM en `concurrency=2/3` respecto a
`concurrency=1`; más el discriminante de wall-clock que la §2 de ese
documento agregó.

Medido con:

```
scripts/gpu_concurrency_probe.py --model qwen3.6:35b-a3b-q4_K_M \
    --concurrency 1 2 3 --repeats 3
```

**Entorno, porque el veredicto depende de él tanto como de la GPU:**
`OLLAMA_NUM_PARALLEL` efectivo = **1** (leído del log de arranque de
Ollama 0.31.1, la variable no está seteada en ningún lado); residente en
VRAM `qwen3.6:35b-a3b-q4_K_M` (24.0 GB de los 32 de la card0); GPU al 3% y
cero pipelines activos al arrancar la medición.

| concurrency | tok/s **generación** | degrad. gen | tok/s e2e | degrad. e2e | wall | factor wall | errores | pico VRAM |
|---|---|---|---|---|---|---|---|---|
| 1 | 76.8 | 0.0% | 74.5 | 0.0% | 7.5 s | ×1.00 | 0 | 24 366 MB |
| 2 | 76.7 | **0.1%** | 56.0 | 24.8% | 13.9 s | **×1.85** | 0 | 24 346 MB |
| 3 | 76.6 | **0.2%** | 35.8 | 52.0% | 21.8 s | **×2.90** | 0 | 24 364 MB |

## Veredicto: **por debajo del umbral — no hace falta exclusión mutua cross-proceso. Y la razón NO es la que el número sugiere a primera vista.**

Las tres condiciones de la §1 del umbral: degradación de generación 0.2%
(< 25%), cero errores, pico de VRAM 24.4 GB = **76%** de los 32 (< 90%).
Ninguna se cumple.

**La razón real es la fila 3 de la tabla del umbral: está ENCOLADO.** El
wall-clock crece ×1.85 y ×2.90 —lineal con la concurrencia— mientras la
velocidad de generación se mantiene clavada en ~76.7 tok/s. Ollama
serializa las requests: nunca hay dos inferencias compartiendo la GPU, así
que **la exclusión mutua ya existe, la provee Ollama, y no hay contención
que un semáforo pudiera evitar.**

## La trampa que este resultado tenía, y por qué importa

La primera corrida daba **24.0% de degradación en `concurrency=2` y 51.3% en
`concurrency=3`** — a un pelo de disparar el criterio de >25% y, en
`concurrency=3`, disparándolo de lleno. Ese número era un **artefacto de la
medición**: `tok/s` se calculaba como `eval_count / elapsed`, y `elapsed`
—el tiempo de la llamada HTTP— **incluye la espera en cola**. Una request
que espera 7 segundos y después genera a 76 tok/s se reportaba como si
generara a 38.

Con el reloj de generación de Ollama (`eval_duration`, que excluye la cola)
la degradación es **0.1% / 0.2%**. El veredicto se habría dado vuelta por
completo: "hace falta un semáforo cross-proceso" cuando lo que había era
una cola funcionando bien.

Es la décima lección de método aplicada a una medición en vez de a un
guard: **el número correcto por el motivo equivocado es peor que un número
que falla**, porque se lee como evidencia. Las dos columnas quedan en la
tabla —generación y extremo a extremo— justamente para que nadie las
confunda de nuevo: la de generación decide sobre la GPU; la e2e es lo que
siente el llamador, y es real (una request concurrente **sí** tarda el
triple), pero es latencia de cola, no contención.

## Qué NO cierra este resultado

La garantía depende de `OLLAMA_NUM_PARALLEL = 1`, y ese valor:

- **nadie lo fija**: Ollama lo eligió solo según la memoria disponible al
  arrancar — con un modelo residente más chico, la misma versión puede
  elegir 4;
- **no está declarado** en el unit de systemd de Ollama ni en ningún `.env`
  del repo, así que no hay nada que lo defienda ni que avise si cambia;
- **no lo cubre ningún test ni guard.**

O sea: el riesgo no es inexistente, está **dormido detrás de una condición
que nadie vigila**. La acción proporcionada es fijar
`OLLAMA_NUM_PARALLEL=1` explícitamente en el unit de Ollama —convertir un
accidente favorable en una decisión declarada— o registrar la dependencia
para que se sepa qué se rompe si cambia. **Queda como decisión de Fernando:
este plan no toca `DEUDA.md` (política explícita) ni configuración de
servicios.**

Fix evaluado y **no** implementado, para no reconstruirlo si el día de
mañana `num_parallel` sube: lock de archivo cross-proceso (`fcntl.flock`
sobre `/run/jax/gpu.lock`, envuelto en `asyncio.to_thread`, fail-closed —
el mismo patrón que ya se usa en `hyde_sandbox.py::run_sandboxed_claude`)
en los 3 call sites reales (`jax/muscles/ollama_muscle.py`,
`jacobs/plan.py::_llm_plan`, `jacobs/executor.py::_invoke_ollama`) más
`las_manos/motor_registry/worker.py::_call_http_openai_compat` cuando el
transporte es `ollama`.
