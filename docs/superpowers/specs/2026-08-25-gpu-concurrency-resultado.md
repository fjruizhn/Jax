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

## CONDICIONES DE VALIDEZ — leer ANTES de cambiar de modelo o de tocar Ollama

Esta sección existe para que **no haya que re-medir cada vez que cambia un
modelo**. Si las condiciones de abajo siguen siendo ciertas, el veredicto
sigue valiendo. Si alguna deja de serlo, dice qué se invalida y qué hay que
hacer.

### La condición que sostiene el VEREDICTO (una sola)

> **Este veredicto es válido mientras `OLLAMA_NUM_PARALLEL=1`, fijado en
> `/etc/systemd/system/ollama.service.d/override.conf`.**

Es la única condición de la que depende la *conclusión*, porque la
conclusión no es "la GPU aguanta la concurrencia" sino "no hay concurrencia
que aguantar: Ollama serializa". Con `num_parallel > 1` la exclusión mutua
que hoy "ya existe afuera" desaparece, y **este documento pasa a ser falso**.
Quien lo suba tiene que reabrir el veredicto y evaluar el lock cross-proceso
descrito al final, no sólo cambiar el número.

Desde el 2026-08-28 está **declarado** en el unit, con el porqué escrito al
lado. Antes era el default del binario — o sea, la decisión de nadie: un
upgrade de Ollama que cambiara ese default habría invalidado el veredicto en
silencio. Verificado después del restart leyendo el valor **efectivo** que
Ollama reporta, y probado al revés (poniendo 2 en el unit y viendo que
Ollama reportaba 2) — porque el valor `1` se leía igual antes de declararlo,
así que verlo no probaba que el unit se hubiera aplicado.

### Los NÚMEROS no son transferibles a otro modelo

Los 76.7 tok/s, los 24.4 GB de pico y los tiempos de pared son **de
`qwen3.6:35b-a3b-q4_K_M` y de nadie más**. Un modelo distinto cambia
tamaño en VRAM, KV cache, y si es MoE o denso. **No hay que re-medir para
cambiar de modelo** — el veredicto (que Ollama serializa) es independiente
del modelo. Hay que re-medir sólo si se quiere volver a afirmar algo sobre
*velocidad* o *margen de VRAM*.

### Parámetros de Ollama que NO deciden el veredicto pero sí los números

Todos salen del `server config` del log de arranque, y ninguno está
declarado (valores del 2026-08-28):

| Parámetro | Valor | Qué toca |
|---|---|---|
| `OLLAMA_MAX_LOADED_MODELS` | `0` = **derivado** | cuántos modelos coexisten antes de desalojar — ver la sección de desalojo |
| `OLLAMA_CONTEXT_LENGTH` | `0` = default | el KV cache escala con el contexto: cambia el margen de VRAM |
| `OLLAMA_KV_CACHE_TYPE` | vacío (f16) | VRAM por slot |
| `OLLAMA_FLASH_ATTENTION` | `false` | VRAM por slot y velocidad — activarlo mueve el baseline de tok/s |
| `OLLAMA_VULKAN` | `true` | el backend entero; otro backend, otros números |
| `OLLAMA_MAX_QUEUE` | `512` | **el modo de falla bajo carga**: con serialización las requests se encolan, y pasada esa cola devuelve 503. No se alcanzó en esta medición (3 concurrentes), pero es el límite real del diseño actual |
| `OLLAMA_KEEP_ALIVE` | `5m0s` | **no** explica la residencia actual: el modelo está residente con `keep_alive=-1` pedido por request, no por config. Si eso deja de mandarse, se descarga a los 5 min y el siguiente turno paga la recarga (**8.5 s medidos**) |

**Sólo `OLLAMA_NUM_PARALLEL` es carga estructural del veredicto.** El resto
mueve los números; ninguno convierte "serializado" en "paralelo".

### Desalojo por VRAM: no hay guard, y no hace falta que lo haya para el OOM

Encontrado al preparar la medición: `qwen3-coder:30b` (18.6 GB) **no entra**
junto al residente (24 GB de 32). Qué pasa en ese caso, verificado en el log
de este mismo servidor:

```
sched.go:557 msg="llama-server model predicted to exceed available memory, evicting" predicted="22.9 GiB"
```

Ollama **desaloja** al modelo residente para hacer lugar. O sea:
- **no hay riesgo de OOM** — el scheduler de Ollama lo maneja;
- **sí hay un costo silencioso**: el modelo de producción se descarga sin
  que nadie en JAX se entere, y el siguiente turno real de chat paga la
  recarga completa. Con `keep_alive=-1` la residencia es deliberada
  justamente para evitar eso.

**Guard del lado de JAX: NO EXISTE.** Verificado con `grep` sobre
`jax/`, `jacobs/`, `las_manos/` y `scripts/`: cero chequeos de VRAM, de
tamaño de modelo, o del residente antes de bindear o invocar. Nada impide
bindear `jax_local` (o cualquier facet) a un modelo que no entre junto al
residente, y nada avisa cuando el desalojo ocurre. **Está dormido detrás de
una condición que nadie vigila** — la misma forma que `num_parallel`, pero
con consecuencia menor (latencia, no corrección).

Qué haría falta si algún día se decide cerrarlo (**no construido, decisión
pendiente**): en el endpoint que aprueba un rebinding
(`POST /admin/models/proposals/{id}/approve`, único escritor de
`facet_binding`), comparar el tamaño del modelo propuesto contra
`VRAM total − residente` y avisar —no bloquear— cuando no entren juntos. El
dato ya está disponible por `/api/show` y `/api/ps` de Ollama, y el patrón
de "sonda después del rebinding" ya existe (`probe_after_rebind`).

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
