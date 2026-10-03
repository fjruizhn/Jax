# F2-D: medición reproducible de render reuse — 2026-10-03

## Diseño y pares medidos

Se comparó el mismo harness ASGI, el mismo código Platform y una salida de 16.000
caracteres. El harness ejecuta `project_provider_contract` → F2-C →
`prepare_governed_chat_response` → F2-D → ASGI send, con proveedor simulado por una
espera fija de 1 ms y un repositorio outbox determinista en memoria. No llama Ollama,
GPU, servicios externos, MariaDB ni credenciales productivas.

| Ejecución | JAX | Platform |
|---|---|---|
| Base | `7024529e1ede4b3a3de92eefe4f60e3d2f567f01` | `1e2888c32d7a6221a4b4395d2d2018845e31468f` |
| Optimizada | ver SHA del commit que incorpora este artefacto | `1e2888c32d7a6221a4b4395d2d2018845e31468f` |

Python 3.14.4, Hall9000. Se hicieron dos corridas secuenciales de 50 peticiones por
nivel, en concurrencias 1, 5, 10 y 25. La tabla muestra la mediana de cada métrica entre
las dos corridas. Resultados crudos por corrida y el código ejecutado están junto a este
documento en `docs/perf/f2d-render-reuse/`.

## Resultados de la frontera F2-C/F2-D

| Concurrencia | Base req/s | Optimizada req/s | Base p95 ms | Optimizada p95 ms | Degradación p95 > 500 ms |
|---:|---:|---:|---:|---:|---|
| 1 | 24,51 | 56,48 | 48,53 | 22,24 | no / no |
| 5 | 25,46 | 63,02 | 206,84 | 87,38 | no / no |
| 10 | 25,68 | 66,21 | 407,97 | 156,46 | no / no |
| 25 | 25,69 | 66,35 | 981,62 | 394,37 | sí / no |

En esta matriz, la base se degrada al llegar a 25 usuarios concurrentes; la rama
optimizada no supera p95 de 500 ms hasta el máximo probado de 25. El throughput observado
sube entre 2,3x y 2,6x. No se extrapola más allá de esa concurrencia ni a la plataforma
completa: esta medición aísla la ruta ASGI con outbox simulado.

## Micro-medición de revalidación

El script `micro_revalidate.py` mide 200 llamadas a `revalidate_for_transport` sobre la
misma respuesta de 16 KB y la misma Platform. Los resultados guardados son:

| | Base | Optimizada |
|---|---:|---:|
| Mediana por llamada | 6,79 ms | 0,11 ms |
| p95 por llamada | 10,05 ms | 0,14 ms |
| Cinco revalidaciones por turno, mediana | ~34,0 ms | ~0,6 ms |

La ruta optimizada reutiliza el `RenderedText` inmutable mintado y vuelve a verificar
digests, scope, referencias, acceso a citas y recibos F2-B en cada límite de transporte.
No hay caché global: descartar la unidad opaca al terminar el request invalida el reuso.

## Reproducción

Use los worktrees exactos de ambos pares. El benchmark solo necesita estas variables de
prueba; no cargue `/etc/jax/.env`:

```bash
export JAX_REPO_PATH=/path/to/jax-pair
export JAX_REPO_BASE=/path/to/jax-platform-pair
export JAX_CONFIG_PATH=/tmp/f2d-benchmark-config.toml
export JAX_TRUSTED_PROXIES=127.0.0.1
export JAX_JWT_SECRET=benchmark-only-not-a-secret
export F2D_LOAD_REQUESTS=50
export PYTHONPATH="$JAX_REPO_BASE/backend:$JAX_REPO_PATH"
python3 "$JAX_REPO_PATH/docs/perf/f2d-render-reuse/asgi_boundary_load.py"
python3 "$JAX_REPO_PATH/docs/perf/f2d-render-reuse/micro_revalidate.py" 16000
```

`JAX_CONFIG_PATH` puede ser un archivo vacío de prueba (`touch`); el script no lee ni
modifica configuración. `JAX_JWT_SECRET` es un valor desechable para importar el módulo,
no una credencial. Para recalcular la tabla se toman las medianas de los pares
`baseline-run-1/2.jsonl` y `optimized-run-1/2.jsonl`; los datos micro están en los dos
archivos `micro-*.txt`.

## Límites

No se ejecutó el orquestador HTTP+MariaDB del harness `carga/chat-f2d`: exige un entorno
aislado MariaDB que no estuvo disponible. No se usó producción y no se escribió a ninguna
base. La suite completa JAX tampoco pudo recolectarse en este entorno porque 15 módulos
Faro/LAS VOCES requieren el paquete `mcp`, ausente; las suites F2-C/F2-D/runtime-status
enfocadas sí se ejecutaron.
