# Proyectos E2 — documentos del proyecto como biblioteca (adenda)

> Fecha: 2026-10-02 · Autor: Mr. Hyde (hall9000) · Decisiones: Fernando (misma fecha, en sesión)
> Continúa: `2026-09-22-proyectos-y-selector-design.md` (§5, §6, §7, §8, §11) y
> `2026-10-02-proyectos-e1-adenda-design.md` (tabla de entregas: E2 y E3).
> Repos: **jax** (LAS MANOS, procesador, esquema, worker de herramientas) y **jax-platform**
> (subida, pantallas, chat).
> Estado: dirección APROBADA por Fernando en conversación («B», «sí»); esta adenda es para
> su revisión escrita. Nada se implementa antes de esa revisión.

## 1 · Qué decidió Fernando

| Decisión | Cuándo | Palabras |
|---|---|---|
| Subir documentos es una herramienta de los **tres modos** (Chat, Pipeline, Ejecutor), no del chat | 2026-10-02 | «las herramientas de subir documentos también tienen que estar disponibles para las otras opciones… para que los modelos lo accesen de manera liviana» |
| **Opción C**: todo documento vive en un proyecto; selector de proyecto también en Pipeline y Ejecutor; en «Personal» el botón pide elegir o crear un proyecto | 2026-10-02 | «C» |
| Los documentos se ven y administran en la pestaña **Documentos** del proyecto | 2026-10-02 | idem |
| Topes: **100 MB por archivo; 250 archivos o 1 GB por lote** | 2026-09-25 | adenda E1, tabla de entregas |
| **B**: el modelo trabaja con **todo el proyecto como biblioteca** (busca y lee lo que necesita), no solo con lo adjuntado en el envío | 2026-10-02 | «yo me iría con B de un solo, creo que es más sólido» |
| Se entrega en dos partes, **E2a** (subida y Documentos) y **E2b** (biblioteca), con una sola revisión de esta adenda | 2026-10-02 | «sí» |

## 2 · Hechos de partida (verificados 2026-10-02 por Hyde)

| Hecho | Fuente |
|---|---|
| Producción: jax `eca7db4` (contiene E1.1 `27590a6`), jax-platform `9a2c90c` (#181) | `git log` en `/srv/jax-prod/*`, tras el reinicio de hall9000 de las 17:12 |
| `POST /procesamiento/trabajos` recibe `proyecto: str` y escribe en `proyectos/<slug>/` | `las_manos/procesamiento_routes.py:180-181,238-239` |
| El procesador deja un extracto por documento en `procesado/<sha256 de la fuente>/`: `ficha.json` + `texto.txt` / `texto.md` / `NN-hoja.csv` | `procesamiento/ficha.py`; `ls` de LACTOVI |
| LACTOVI: 120 archivos fuente, **295 MB**; 88 extractos, **1,5 MB** en total | `du`, `find` sobre `~/jax-workspace/proyectos/lacteos-victoria` |
| `proyectos/` es `fruiz:fruiz 775` sin setgid: `jaxsvc` no puede escribir | `ls -la`; arreglo en Jax#276 (abierto) |
| `/home/fruiz` (y con ella `jax-workspace/proyectos/`) entra al restic local y R2; ninguna exclusión de `SISTEMA_EXCLUYE` la toca | `claude-skills/bin/backup-hall9000.sh:154,194` |
| Restauración de `proyectos/` probada el 25-sep (250/250 sha256) | adenda Jax#275 — anterior al traslado de LACTOVI |
| **Chat**: jax-platform llama al modelo directamente (Ollama, OpenAI-compatible, Gemini). **No tiene bucle de herramientas**; Gemini solo lleva `googleSearch` | `jax-platform/backend/api/chat.py:829-900` |
| **Pipeline**: Jacobs → worker de LAS MANOS, que **sí** tiene un bucle gobernado (`read_file`, `write_file`), mapeo herramienta→capacidad que rechaza lo no mapeado, y envoltorio `<untrusted_source>` | `las_manos/motor_registry/tool_authority.py:70-78,419`; `worker.py:620,743` |
| **Ejecutor**: corre misiones con los seis contratos; **C5** decide qué auditor puede ver datos de clientes | `jax-platform/backend/ejecutor/misiones.py:1-25` |
| Papeles de proyecto en B9: `VIEWER` < `CONTRIBUTOR` < `OWNER` | `jax/memory/project_authority.py:198` |

**Consecuencia del hecho de los 1,5 MB:** los extractos de un solo expediente pesan del
orden de 375 000 tokens. Meterlos todos en un pedido no es liviano; por eso B (buscar y leer
lo necesario) es lo que corresponde.

## 3 · E2a — Subir documentos al proyecto

### 3.1 Datos
Tabla nueva **`project_documents`** en `jax_memory` (esquema en el repo jax):
`id`, `project_id` (FK), `sha256`, `nombre_original`, `ruta_fuente` (relativa a
`proyectos/<uuid>/fuente/`), `bytes`, `tipo`, `estado`
(`pendiente`/`procesando`/`listo`/`parcial`/`error`/`sin_extractor`), `job_id`,
`subido_por` (FK a `jax_users`), `created_at`, `updated_at`.
- Índice único `(project_id, sha256)`: el mismo contenido no entra dos veces al proyecto.
- Índice `(project_id, created_at)` para la lista de la pestaña.

### 3.2 LAS MANOS
- `POST /procesamiento/trabajos` cambia `proyecto: str` por **`project_uuid`** (spec §5):
  422 si el uuid no existe o el proyecto no está `ACTIVE`. La carpeta pasa a
  `proyectos/<project_uuid>/`; renombrar un proyecto no mueve archivos.
- La membresía **no** se verifica aquí: solo la credencial de plataforma llama a
  `/procesamiento`, y jax-platform ya verificó el papel (spec §5, sin cambios).
- Al terminar cada documento, el estado y el `sha256` del extracto quedan en el trabajo,
  como hoy. `scripts/procesar_archivos.py` también recibe `project_uuid`.

### 3.3 jax-platform
- **`POST /api/proyectos/{id}/documentos`** (multipart, varios archivos), papel mínimo
  **CONTRIBUTOR**. Antes de copiar, en este orden: tope por archivo (100 MB), tope del lote
  (250 archivos o 1 GB), reserva de disco con el mismo criterio que `adjuntos/cuota.py`.
  Escribe por streaming (`asyncio.to_thread`) con nombre seguro.
- **Duplicados:** mismo `sha256` en el proyecto → se salta y se informa; mismo nombre con
  otro contenido → se guarda con sufijo (`informe (2).pdf`). Nada se sobrescribe.
- Al terminar la copia, una llamada a `POST /procesamiento/trabajos` con las rutas; devuelve
  el `job_id` y las filas de `project_documents` en `pendiente`.
- **`GET /api/proyectos/{id}/documentos`** (VIEWER o más), paginado. El estado se actualiza
  consultando `GET /procesamiento/trabajos/{job_id}` mientras haya trabajos abiertos.
- Todos los topes salen de configuración (tabla de config de la plataforma), no del código.

### 3.4 Interfaz
- **Selector de proyecto** en las filas de Chat, Pipeline y Ejecutor (hoy solo en Chat).
- **Botón de documentos** en los tres modos. Sin proyecto elegido («Personal»), el botón
  abre «elige un proyecto o crea uno». Con papel VIEWER, el botón no sube y lo dice.
- **El Selector** (spec §7): `<input type="file" multiple>` y también carpeta
  (`webkitdirectory`). Antes de subir, ventana propia (`Dialogo.jsx`) con cuántos
  archivos, cuánto pesan, de qué tipos, cuáles se ignoran y por qué (tope, tipo, duplicado).
  Recién al confirmar, sube, con barra de avance.
- **Pestaña Documentos** en `ProyectoDetalle.jsx`: lista con nombre, tamaño, quién lo subió,
  estado y avance; botón «Agregar documentos» que abre el Selector.
- i18n `es`/`en`, tokens de color, claro y oscuro, foco y Escape como los demás modales.
  Ningún `confirm(`, `alert(` ni `prompt(`, tampoco desnudos.

### 3.5 Antes de habilitar la subida (Principio IX, spec §8)
1. Jax#276 aplicado: `proyectos/` con grupo `jaxsvc`, setgid y ACL por defecto; su prueba
   corriendo como `jaxsvc` pasa en el host.
2. **LACTOVI** se crea como proyecto (dueño: Fernando) y `proyectos/lacteos-victoria/` se
   **mueve** a `proyectos/<uuid>/`; `sha256` de `fuente/` iguales antes y después; sus 88
   extractos quedan registrados en `project_documents`.
3. **Restauración probada otra vez**, ya con la ruta nueva: un snapshot posterior al
   traslado, restaurado a una ruta aparte, `sha256` contra la ficha. Sin eso no se habilita.
4. Costo de R2: `du` de `proyectos/` antes y después; con el tope de 1 GB por lote, se
   anota el crecimiento esperado.

## 4 · E2b — El proyecto como biblioteca

### 4.1 Un solo servicio, en LAS MANOS
LAS MANOS es dueña de los extractos, así que la búsqueda vive ahí. Dos endpoints internos,
solo con la credencial de plataforma o desde el worker:
- `POST /procesamiento/proyectos/{uuid}/buscar` `{consulta, limite}` → fragmentos con
  `documento_id`, nombre, página o sección, y el texto del fragmento.
- `GET /procesamiento/proyectos/{uuid}/documentos/{id}/extracto?desde=&hasta=` → el
  extracto completo o un tramo.

### 4.2 Índice de búsqueda: texto, no vectores (decisión técnica de Hyde)
- Tabla **`project_document_fragments`** (`project_id`, `document_id`, `orden`,
  `pagina_o_seccion`, `texto`) con **índice FULLTEXT** de MariaDB. El procesador la llena al
  terminar cada extracto, en trozos de tamaño fijo con solapamiento.
- **Por qué texto y no bge-m3:** en estados financieros, avalúos y escrituras lo que se busca
  son cifras, nombres, RTN y números de escritura **exactos**, y ahí la búsqueda por texto
  acierta y la semántica se equivoca con seguridad. Además no agrega otro índice vectorial
  con el problema de borrado en cascada (MDEV-41227), que complicaría Destruir.
- La búsqueda semántica queda como segunda capa **si** una medición con preguntas reales de
  LACTOVI muestra que el texto se queda corto. Sin medición, no se agrega.

### 4.3 Las dos herramientas
- **`buscar_en_proyecto(consulta, limite)`** y **`leer_extracto(documento_id, desde, hasta)`**.
- **El proyecto lo fija el contexto, nunca el modelo.** Ninguna herramienta recibe
  `project_id` como argumento: sale del turno (chat), de la corrida (pipeline) o de la
  misión (ejecutor), que jax-platform fijó tras verificar el papel. Un modelo engañado por un
  documento no puede pedir otro proyecto.
- Todo lo que devuelven va dentro de `<untrusted_source>` (el envoltorio que ya existe en
  `tool_authority.py`): es DATO, no instrucción.
- **Topes por turno**, en configuración y no en el código: máximo de llamadas y máximo de
  caracteres leídos. Propuesta inicial: 8 llamadas y 60 000 caracteres; se ajusta con la
  medición de §6. Al llegar al tope, la herramienta lo dice y el modelo responde con lo que
  tiene.
- **Lo recién adjuntado:** el mensaje del envío lleva «documentos agregados en este envío:
  …», para que el modelo empiece por ahí. El envío espera a que esos extractos estén listos
  (o muestra el avance y se puede cancelar).
- Papel mínimo para usar la biblioteca: **VIEWER** (spec §2: el lector «lee extractos»).

### 4.4 Integración por modo — tres entregas auditadas por separado
Los tres modos usan motores distintos (§2), así que E2b se entrega en tres partes, en este
orden, cada una con su auditoría de escalón 3:

| Parte | Modo | Qué cambia |
|---|---|---|
| **E2b-1** | Servicio + **Pipeline** | Índice y endpoints (§4.1-4.2). En el worker de LAS MANOS, las dos herramientas entran a `TOOLS_CATALOG` y a `TOOL_CAPABILITY_MAP` con una capacidad nueva `project_docs_read` (solo lectura). La corrida de pipeline lleva `project_id`, fijado por jax-platform al lanzarla. Es la primera porque el bucle gobernado ya existe |
| **E2b-2** | **Chat** | jax-platform gana un bucle de herramientas acotado para los transportes `ollama` y `openai`; las herramientas llaman a los endpoints de §4.1 con el proyecto del turno. Gemini: se mide su llamada a funciones antes de incluirlo; si no se mide, en Gemini el chat no ofrece la biblioteca y lo dice |
| **E2b-3** | **Ejecutor** | La misión lleva `project_id`. **C5 manda:** los documentos de un proyecto son datos de clientes, así que una misión con proyecto solo corre en máquinas elegibles para datos de clientes y con el auditor que C5 admite. Se diseña en detalle al llegar, sobre el contrato C5 vigente ese día |

## 5 · Lo que no cambia
- Autorización en un solo punto (jax-platform, sobre B9). LAS MANOS no duplica papeles.
- Destruir un proyecto sigue fuera (adenda E1). Cuando llegue, borra también
  `project_documents`, `project_document_fragments` y `proyectos/<uuid>/`, y el índice
  FULLTEXT no tiene el riesgo de MDEV-41227.
- E3 (respaldo del chat: un PDF escaneado pegado en el chat va al procesador) sigue después
  de E2.

## 6 · LAS CUATRO DEL RENDIMIENTO
1. **Índices.** `EXPLAIN` sobre la consulta real de la pestaña Documentos, sobre `MATCH …
   AGAINST` acotado por `project_id` y sobre la verificación de papel de cada llamada a la
   biblioteca.
2. **Caché.** Ninguno nuevo sin medición. Si se cachea la verificación de papel, la
   invalidación (quitar miembro, cambiar papel, archivar) va en el mismo commit.
3. **Async.** La subida escribe por streaming fuera del bucle; el procesamiento ya es una
   cola (jax#255); las herramientas son HTTP asíncrono.
4. **Carga, antes de cada GO.** E2a: un lote del tamaño de LACTOVI (120 archivos, 295 MB)
   mientras 20 usuarios chatean, midiendo p95 del turno. E2b: 20 turnos simultáneos usando la
   biblioteca sobre LACTOVI, midiendo p95 de `buscar` y del turno completo. Los números van a
   la Biblioteca.
   **Calidad, además de velocidad:** un juego de 10 preguntas reales sobre LACTOVI con su
   respuesta y su documento conocidos; se mide cuántas acierta el modelo citando el documento
   correcto, con Qwen y con el modelo de nube del chat.

## 7 · Despliegue (cada paso en producción con GO de Fernando)
1. Respaldo verificado de `jax_memory`.
2. Esquema: `project_documents`, `project_document_fragments`.
3. Jax#276 en el host.
4. LACTOVI: proyecto, traslado, `sha256`, registro de sus extractos, índice.
5. Restauración probada de `proyectos/` con la ruta nueva.
6. jax (LAS MANOS con `project_uuid`) y después jax-platform (subida, pestaña, selectores).
7. E2b-1, E2b-2 y E2b-3, cada una con su auditoría, su carga y su GO.

## 8 · Preguntas para Fernando
1. **Quitar un documento del proyecto.** Esta adenda no lo incluye: hoy solo se destruye el
   proyecto entero. Propuesta: que un CONTRIBUTOR pueda **ocultar** un documento (deja de
   verse y de buscarse, se puede restaurar), y que borrarlo de verdad quede para Destruir.
   ¿Lo incluyo en E2a o lo dejamos fuera?
2. **E2b-3 (Ejecutor) dentro de E2 o después.** Por C5, meter documentos de clientes en una
   misión del Ejecutor es la parte con más gobierno. Propuesta: E2 cierra con Chat y Pipeline;
   el Ejecutor se diseña aparte cuando esas dos estén en producción. ¿De acuerdo, o lo
   quieres dentro de E2?

## 9 · Fuera de alcance
Búsqueda semántica (salvo que la medición la pida) · Destruir proyecto · invitar fuera del
tenant · editar documentos · reprocesar con otra versión del extractor desde la interfaz ·
E3.
