# Proyectos E2b-1 — bloqueo de integración con Pipeline

- **Fecha:** 2026-10-06.
- **Fuente observada:** checkout de JAX en `origin/master`, commit
  `685579466c25771a4df5d5cfd2c51bc555a21fa6`.
- **Alcance:** lectura estática del spec E2b y de los contratos de ejecución de
  Jacobs/LAS MANOS. No se consultó `jax_memory`, producción ni datos de LACTOVI.
- **Tipo:** HECHO del código fuente a ese SHA; no describe el estado vivo de
  producción.

## Hallazgo

El diseño de E2b-1 (`docs/superpowers/specs/2026-10-02-proyectos-e2-adenda-design.md`,
§§4.1–4.4 y §6) coloca las herramientas de biblioteca en el worker gobernado de
LAS MANOS y exige que el proyecto venga fijado por la corrida de Pipeline.
El código de ese SHA no ofrece ese enlace:

1. `jacobs/executor.py::_invoke_motor` despacha el step a
   `POST /motor/dispatch`.
2. `las_manos/motor_registry/routes.py::dispatch` rechaza esa ruta con HTTP
   `410 GOVERNED_EXECUTION_REQUIRED` antes de ejecutar su lógica histórica.
3. La ruta `POST /motor/governed-dispatch` exige una ejecución Block 6. Al
   crear el job llama a `JobStore.create(... pipeline_id=None, project_id=None)`
   y al worker con `pipeline_id=None`.
4. `jacobs.models.Pipeline`, `PipelineCreateRequest` y el DDL de
   `jacobs_pipelines` no almacenan `project_id`; `pipeline_create` tampoco lo
   persiste.
5. El worker existente solo declara `read_file` y `write_file` en
   `TOOLS_CATALOG`; `TOOL_CAPABILITY_MAP` solo vincula esas herramientas con
   `file_read` y `file_write`. No existen `buscar_en_proyecto`,
   `leer_extracto` ni la capacidad `project_docs_read`.

El veredicto de lectura del escalón 3 confirma que no hay un camino alternativo
actual que permita integrar E2b-1 conservando la autoridad y el contexto de
proyecto requeridos. Añadir únicamente los endpoints y herramientas no los
habilitaría en Pipeline. Adaptarlos unilateralmente a Block 6 o reactivar el
endpoint legado abriría una decisión de arquitectura y podría exponer
documentos entre proyectos.

## Resultado de esta sesión

Se detuvo la implementación funcional en el límite del encargo: no se cambió
el transporte, los contratos de autoridad ni el spec. Esta rama entrega la
evidencia para resolver la incompatibilidad antes de retomar E2b-1. E2b-2 y
E2b-3 permanecen fuera de este alcance.

- **Pruebas funcionales y EXPLAIN:** no aplican; no se implementó una consulta
  ni una capacidad ejecutable.
- **Prueba de carga E2b §6:** no ejecutada; medirla antes de tener el camino
  autorizado de Pipeline no verificaría la funcionalidad pedida.
- **Pisos CI:** sin cambio; no se añadieron ni retiraron pruebas.
- **Acción necesaria para retomar:** resolver, en un cambio de diseño separado,
  cómo la plataforma fija y persiste `project_id` y cómo la ejecución gobernada
  de Pipeline transporta ese contexto hasta el worker sin aceptar selección
  del modelo.

## Ronda 2 — verificación del camino productivo (2026-10-06)

Se comprobó la premisa operativa de Ronda 2 antes de cambiar el código:

- `jax-las-manos.service` y `jax-platform.service` están activos en Hall9000.
- El checkout productivo de JAX está en
  `685579466c25771a4df5d5cfd2c51bc555a21fa6` (`master`, origin/master). Los
  SHA-256 de `jacobs/executor.py` y `las_manos/motor_registry/routes.py`
  desplegados coinciden con los del checkout de esta rama.
- `jacobs/executor.py::_invoke_motor` llama a `POST /motor/dispatch`; el
  handler desplegado rechaza esa ruta con 410 `GOVERNED_EXECUTION_REQUIRED`.
  No se encontró ningún
  llamador productivo de `POST /motor/governed-dispatch` ni del adaptador
  `governed_motor_payload`.
- En los registros de `jax-las-manos.service` desde su arranque
  (2026-10-05 21:06:15 CST) hasta esta observación no aparecen llamadas a
  `/jacobs/pipeline`, `/motor/dispatch` ni `/motor/governed-dispatch`. Por eso
  no hay una corrida reciente en los registros con la que acreditar cuál
  endpoint despachó una corrida viva; el código productivo desplegado sí
  acredita el 410 del camino de motor de Jacobs.
- El checkout productivo de jax-platform está en
  `d283d900e1da5ef557ddd53dcde2f0808306785c` (`master`, origin/master). Su
  `POST /api/pipelines` sobrescribe `user_id` y `tenant_id` autenticados y
  reenvía el cuerpo a `POST /jacobs/pipeline`; no resuelve ni fija `project_id`
  ni comprueba membresía del proyecto.
- `governed_dispatch` carga un `ExecutionRequest` sellado por `execution_id`,
  pero crea el job con `pipeline_id=None` y `project_id=None`. El request B6
  conserva `context` dentro de su hash, pero el repo no contiene un llamador
  productivo de `build_execution_request` que cree una ejecución por paso de
  Pipeline para transportar esos valores.

### Resolución

La instrucción de Ronda 2 de usar `governed_dispatch` no coincide con el
camino desplegado: `jacobs/executor.py` sigue en la ruta retirada, y no existe
un ciclo de creación de ejecución B6 por paso que permita cambiarla a
`governed_dispatch` con un `execution_id` auténtico. Crear ese ciclo, decidir
qué decisión B5 autoriza cada paso y ligar los nuevos campos al artefacto B6
excede el contrato escrito de E2b-1. No se inventó esa autoridad, no se
reactivó `/motor/dispatch` y no se agregó una ruta paralela. Se conserva la
evidencia en este PR para que el contrato de ejecución de Pipeline se resuelva
antes de integrar biblioteca accesible desde modelos.

- **Código funcional, EXPLAIN y carga:** no ejecutados; el punto de integración
  gobernado requerido no existe en el checkout desplegado.
- **Producción:** solo se consultaron estado de servicios, revisiones/archivos
  desplegados y registros del servicio. No se escribió en producción ni se usó
  `jax_memory` para pruebas.

## Procedencia de auditoría

Auditoría de escalón 3 de solo lectura, invocada en esta sesión sobre el mismo
checkout/SHA. Recomendación: no habilitar E2b-1 ni reinterpretar el contrato
hasta resolver la incompatibilidad. No se delegó la decisión a un agente ni se
atribuyó a Fernando una aprobación nueva.
