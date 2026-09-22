# Proyectos, miembros, Selector y respaldo del chat — diseño

> Fecha: 2026-09-22 · Autor: Mr. Hyde · Decisiones: Fernando (misma fecha)
> Repos: **jax** (esquema de `jax_memory`, LAS MANOS) y **jax-platform** (API, pantallas)
> Estado: spec para revisión de Fernando. Sin plan todavía.
> Continúa: `2026-09-20-procesamiento-archivos-design.md` (Plan 3, "Selector", nunca escrito).

## 1 · Por qué esto es más grande que "el Selector"

El Selector iba a mandar archivos a `POST /procesamiento/trabajos` (jax#255). Al
revisarlo contra producción el 2026-09-22 salieron cinco hechos, todos verificados:

1. **Dos cosas se llaman "proyecto" y no se tocan.** LAS MANOS recibe `proyecto` como
   texto libre y lo convierte en `proyectos/<slug>/`. jax-platform usa `project_id` de la
   tabla `projects`, donde sólo existe HAMURABI (id 1).
2. **Nadie crea proyectos.** No hay un solo `INSERT INTO projects` en jax ni en
   jax-platform. Tampoco hay tabla de miembros.
3. **`/chat` acepta cualquier `project_id` sin validarlo** (`backend/api/chat.py:1195`) — y
   **ya se usó**: hay 241 ids que no existen en `projects` (900001–900240 y 1400055), con
   241 conversaciones, ~700 mensajes, 10 `facts` y 6 `action_items`. Todos del usuario 1,
   `source = axioma-web-proyecto`, entre 19:56 y 21:23 del 2026-09-03, con preguntas de
   evaluación. Que sean de la evaluación de grounding de SP3 es **inferencia no verificada**.
4. **`jaxsvc` no puede escribir `proyectos/`.** La carpeta es `fruiz:fruiz 775` porque la
   creó la corrida manual de `scripts/procesar_archivos.py` del 2026-09-21 (17:21, sesión
   de Hyde — no "alguien a mano"). El primer trabajo real de LAS MANOS con un proyecto nuevo
   fallaría. Los tests no lo ven porque corren con el usuario que los lanza.
5. **Los documentos no tendrían respaldo.** El restic de hall9000 respalda sólo
   `/srv/backup-adata/staging` (`/opt/backup-scripts/backup-hall9000.sh:310`).
   `jax-workspace/` no está en ningún respaldo.

### Corrección al spec anterior, con el error al lado
El spec del 2026-09-20 §4 dice que el Selector "navega la máquina donde corre el navegador"
y que por eso "ya ve `/home/fruiz/atem-ai`". **Es falso como camino de datos:** ningún
navegador le entrega a una página la ruta de un archivo, sólo su contenido. El Selector
**sube bytes**; no entrega rutas. De ahí sale la subida por lotes de §6.

## 2 · Decisiones (Fernando, 2026-09-22)

- **Un proyecto es una fila de `projects`.** No texto libre ni una carpeta.
- **Cualquier usuario crea un proyecto y queda como dueño.** El **superadmin es dueño de
  todos** por su rol, sin fila de miembro: lo es desde el primer segundo de cualquier
  proyecto, y otro superadmin futuro también, sin migrar nada.
- **Tres papeles:**

  | | Chatea con la memoria del proyecto y lee extractos | Sube y procesa | Invita, quita, cambia papeles, renombra, archiva |
  |---|---|---|---|
  | **lector** | ✓ | — | — |
  | **editor** | ✓ | ✓ | — |
  | **dueño** | ✓ | ✓ | ✓ |

- **Un proyecto nunca se queda sin dueño.** No se puede quitar ni rebajar al último dueño
  (el superadmin sí puede).
- **Sólo se invita dentro del mismo tenant.**
- **Ciclo de vida, igual que los pipelines descartados:**

  | Estado | Quién lo ve | Quién lo deshace |
  |---|---|---|
  | activo | sus miembros | — |
  | **archivado** | sus miembros, en la vista "Archivados" | el dueño |
  | **oculto** ("Borrar") | sólo el superadmin | el superadmin |
  | **destruido** | nadie | **irreversible** |

  - Archivar: el dueño. Ocultar: sólo el superadmin, sólo desde archivado.
  - **Destruir**: sólo el superadmin, sólo desde oculto, con `ConfirmacionSuma` y **con un
    snapshot verificado que cubra el proyecto** (§8). La confirmación nombra el snapshot.
  - Archivados y ocultos **no** van en la lista principal: vistas propias, paginadas.
- **Interfaz:** pantalla nueva **Proyectos** + selector de proyecto en el chat.
- **Respaldo del chat:** un PDF sin texto va al procesador; el envío espera al extracto.
- **Reparto (enfoque A):** jax-platform maneja proyectos, miembros y subida; LAS MANOS
  recibe `project_uuid` y verifica que exista y esté activo.
- **Los 241 huérfanos** van a un proyecto archivado, "Evaluación grounding SP3 · 2026-09-03".

## 3 · Modelo de datos (jax_memory — el esquema vive en el repo **jax**)

### `projects` (existe; se amplía)
- `tenant_id INT NOT NULL` — HAMURABI y el proyecto de evaluación nacen con tenant 1 (el
  único que existe: Inversiones Diamante Negro).
- `estado ENUM('activo','archivado','oculto') NOT NULL DEFAULT 'activo'`. Es otro campo,
  no el `status` existente (`planning/active/...`), que describe el avance del trabajo y no
  su visibilidad. Mezclarlos daría un "completed" que no se sabe si está archivado.
- `creado_por`, `archivado_por/_at`, `oculto_por/_at`.

### `project_members` (nueva)
`(project_id, user_id)` como PK, `papel ENUM('dueño','editor','lector')`, `agregado_por`,
`agregado_at`. FK a `projects(id)`. El superadmin **no** tiene filas aquí.

### Llaves foráneas
`conversations`, `messages`, `facts`, `decisions` y `action_items` tienen `project_id` sin
FK. Se agrega la FK **después** de migrar los huérfanos (§9). Sin la FK, el hueco de `/chat`
vuelve a abrirse con cualquier camino nuevo que escriba `project_id`.

### Auditoría
Toda acción sobre un proyecto o sus miembros (crear, invitar, quitar, cambiar papel,
archivar, ocultar, restaurar, destruir) deja una fila en una tabla `project_audit`: quién,
qué, cuándo, antes y después.

## 4 · Autorización — un solo punto (jax-platform)

Una dependencia, `exigir_papel(project_id, minimo)`, que devuelve el papel efectivo
(superadmin → dueño) o responde **404** (el proyecto no existe, está oculto, o no eres
miembro: no se revela cuál) o **403** (eres miembro pero te falta el papel). La usan **todas**
las rutas que reciben un `project_id`:

- `/chat`: con `project_id`, mínimo **lector**. Hoy no valida nada.
- `/chat/upload`: con `project_id`, mínimo **editor** (el documento entra al proyecto).
- `/proyectos/...`: según la acción.
- Un proyecto **archivado** se puede leer pero no recibe documentos nuevos ni chat nuevo.

## 5 · LAS MANOS (jax)

`POST /procesamiento/trabajos` cambia `proyecto: str` por `project_uuid`:
- Rechaza un uuid inexistente o no activo (422), consultando `projects`.
- La carpeta es `proyectos/<project_uuid>/`: renombrar un proyecto no mueve archivos ni
  rompe rutas.
- **La membresía NO se verifica aquí.** Sólo la credencial de plataforma puede llamar a
  `/procesamiento`, y jax-platform ya verificó el papel. Duplicar la lógica de papeles en
  dos repos es la forma más segura de que se desalineen.
- `scripts/procesar_archivos.py` también pasa a recibir un `project_uuid`.

### Permisos del workspace
`proyectos/` y lo que cuelgue de ella: dueño `jaxsvc`, grupo con escritura para `fruiz`
(que corre el guion a mano), setgid para que lo nuevo herede el grupo. **Un test que
ejercita la escritura corriendo como `jaxsvc`**, no como el usuario de CI, y que falla
contra el estado de hoy.

## 6 · Subida por lotes (jax-platform)

Los topes del chat son para "un adjunto", no para "99 documentos". Camino propio:

- `POST /proyectos/{id}/documentos` (multipart, varios archivos), mínimo **editor**.
- Escribe en `proyectos/<uuid>/fuente/` por streaming, con reserva de disco antes de
  copiar (mismo criterio que `adjuntos/cuota.py`) y nombre seguro.
- Al terminar, llama a `POST /procesamiento/trabajos` con esas rutas y devuelve el `job_id`.
- **Topes: pregunta abierta** (§12). No se inventan: se fijan con el tamaño real de los
  expedientes de Nextcloud (184 `.xlsx` y los PDFs ya medidos).

## 7 · Interfaz (jax-platform)

### Pantalla "Proyectos"
- **Lista**: los proyectos activos donde eres miembro (todos si eres superadmin), paginada.
  Arriba, "Nuevo proyecto".
- **Pestañas** "Archivados" y (sólo superadmin) "Ocultos".
- **Dentro de un proyecto**:
  - **Documentos**: cada archivo con su estado de procesamiento (`pendiente`, `procesando`,
    `listo`, `parcial`, `error`), actualizado por consulta periódica a `GET
    /procesamiento/trabajos/{id}`; y **Agregar documentos**, que abre el Selector.
  - **Miembros**: invitar (sólo dentro del tenant), cambiar papel, quitar.
  - **Ajustes**: renombrar, archivar; si eres superadmin, ocultar, restaurar y destruir.

### El Selector
El diálogo nativo de abrir archivos (`<input type="file" multiple>`). Antes de subir,
**muestra cuántos archivos, cuánto pesan, de qué tipos y cuáles se van a ignorar**, y pide
confirmación en ventana propia (`Dialogo.jsx`). Recién después sube.

### Selector de proyecto en el chat
Arriba del chat: "Personal" o uno de tus proyectos activos. Es lo único que manda
`project_id` a `/chat`.

Todo con i18n (`es.js`/`en.js`), tokens de color, claro y oscuro. Ningún `confirm(`,
`alert(` ni `prompt(`, tampoco desnudos.

## 8 · Respaldo y destrucción — contrato previo (Principio IX)

**Esto va antes que el Selector.** El Selector habilita que entren al sistema documentos de
clientes que pueden no existir en ningún otro lado. Sin respaldo, el primer disco que falle
se los lleva.

- `jax-workspace/proyectos/` entra al restic de `backup-hall9000.sh`, local y R2.
- **Restauración probada** de un proyecto real (LACTOVI) a una ruta aparte, comparando
  `sha256` contra la ficha de `fuente/`. Sin eso, no se habilita la subida.
- **Destruir** exige un snapshot que contenga `proyectos/<uuid>/`, posterior al último
  documento agregado. La ventana de confirmación dice cuál es, y la fila de `project_audit`
  lo guarda.
- Destruir borra `proyectos/<uuid>/` y las filas del proyecto (miembros, conversaciones,
  mensajes, facts...). Las filas de uso (`axioma_usage`) se conservan, por el mismo motivo
  que en pipelines: el gasto no cambia retroactivamente.
- ⚠️ Borrar conversaciones en cascada es exactamente lo que envenena el HNSW (MDEV-41227,
  ver `docs/runbooks/` de jax#239). El plan tiene que resolverlo antes de habilitar
  Destruir — por ejemplo, borrar los mensajes uno por uno sin cascada y correr el detector
  después. No se habilita Destruir sin eso.
- El costo de R2 del respaldo nuevo **no está medido**. Se mide con `du` antes de activarlo.

## 9 · Migración (en este orden)

1. Respaldo verificado de `jax_memory` antes de tocar nada.
2. Columnas nuevas en `projects`; tabla `project_members`; `project_audit`.
3. Proyecto "Evaluación grounding SP3 · 2026-09-03" (archivado, tenant 1). Los 241 ids
   huérfanos se reescriben a su id en las cinco tablas. Conteos antes y después, iguales.
4. HAMURABI: tenant 1, activo.
5. LACTOVI: se crea el proyecto (dueño: Fernando) y `proyectos/lacteos-victoria/` se
   **mueve** a `proyectos/<uuid>/`. Se verifican los `sha256` de `fuente/` después.
6. Permisos de `proyectos/` (§5).
7. FKs de `project_id` en las cinco tablas.
8. `/chat` y `/chat/upload` con `exigir_papel`.

Los pasos 1–8 son escrituras en producción: cada uno con GO de Fernando.

## 10 · Respaldo del chat

Hoy un PDF sin texto responde `422 pdf_sin_texto` al instante (`backend/api/upload.py:149`).
Pasa a:

- **Con proyecto elegido y papel de editor o más**: el PDF entra a `fuente/` del proyecto
  y se encola en el procesador. El adjunto queda en estado "procesando".
- **Sin proyecto, o como lector**: se procesa en una carpeta personal del usuario, fuera de
  `proyectos/`, con la misma caducidad que los adjuntos de hoy.
- El chip del adjunto muestra el avance y se puede cancelar
  (`POST /procesamiento/trabajos/{id}/cancel`).
- **El botón de enviar espera al extracto.** El mensaje sale con el texto completo; las
  marcas de `parcial` y de cifras dudosas van dentro del extracto, como ya hace el procesador.
- Si el procesador falla o se cancela, el error se dice con la causa, en el chip.

## 11 · LAS CUATRO DEL RENDIMIENTO

1. **Índices.** `project_members (user_id, project_id)` para "mis proyectos";
   `projects (tenant_id, estado, ...)` para las listas. `EXPLAIN` sobre las consultas reales
   de cada lista y de `exigir_papel`, que corre en **cada** turno de chat con proyecto.
2. **Caché.** `exigir_papel` corre por turno. Sin medición no se cachea; si se cachea,
   la invalidación (cambio de papel, quitar miembro, archivar) va en el mismo commit y
   cruza procesos.
3. **Async.** La subida escribe por streaming con `asyncio.to_thread`; el procesamiento ya
   es una cola (jax#255). El chat no bloquea esperando al OCR: espera el cliente.
4. **Carga.** Antes del GO: subida de un lote del tamaño de LACTOVI (99 documentos)
   mientras corre el chat, midiendo la latencia p95 del turno. El número va a la Biblioteca.

## 12 · Preguntas abiertas

- **Topes de la subida por lotes** (por archivo y por lote): se proponen con los tamaños
  reales medidos de Nextcloud, y los aprueba Fernando.
- **Costo de R2** del respaldo de `proyectos/`: se mide antes de activarlo.

## 13 · Fuera de alcance

El Embudo de Nextcloud (Plan 2) · invitar a usuarios de otro tenant · papeles por
documento · notificaciones de invitación por correo · búsqueda semántica sobre extractos.
