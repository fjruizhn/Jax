# Proyectos E1 — adenda de reconciliación al spec del 2026-09-22

> Fecha: 2026-10-02 · Autor: Mr. Hyde (hall9000) · Decisiones: Fernando (misma fecha, en sesión)
> Continúa: `2026-09-22-proyectos-y-selector-design.md` (Jax#256) y la adenda de Jax#275
> (rama `docs/proyectos-sobre-b9`, decisiones D1–D5 del 2026-09-25 20:50).
> Repos: **jax-platform** (API, pantallas) y **jax** (lo que falte en la autoridad de B9).
> Estado: diseño APROBADO por Fernando en conversación; esta adenda es para su revisión escrita.

## 1 · Por qué una adenda

El spec del 22-sep se escribió antes de que B9 entrara a producción. Desde entonces:
B9 se desplegó (26-sep), Jax#280 implementó la autoridad de proyectos con D1–D5, el
hueco de `/chat` se cerró, y T16 / F2 cambiaron el chat de jax-platform. Esta adenda dice
qué del spec sigue, qué cambia y qué entra en la **primera entrega (E1)**.

## 2 · Hechos de partida (verificados 2026-10-02)

| Hecho | Fuente |
|---|---|
| Producción: jax `152cb23`, jax-platform `fff3006` | `git log` en `/srv/jax-prod/*` |
| `/chat` valida `project_id` con `ProjectScopeAuthorityResolver` | `jax-platform/backend/api/chat.py:1168` (en `fff3006`) |
| Autoridad B9 en jax: `create_project`, `bootstrap_existing_project`, `grant_member`, `change_project_role`, `revoke_member`, `set_project_lifecycle` | `jax/memory/project_authority.py:475,588,674,731,777,819` |
| B9 ya impide quitar o rebajar al último OWNER | `project_authority.py:763,802` |
| B9 impide cambiar la membresía de un admin activo del tenant | `project_authority.py:754,794` |
| REVIEWER no es asignable como papel de proyecto | `project_authority.py:733-737` |
| **No existe** renombrar un proyecto | sin `UPDATE projects SET name` en `project_authority.py` |
| HAMURABI (id 1) existe en `projects`, pero `jax_project_scope` y `jax_project_membership` tienen **0 filas** → `/chat` con ese proyecto da 403 a todos | consulta de solo lectura a `jax_memory` (explorador, 2026-10-02) — NO re-medido por Hyde |
| 241 `project_id` huérfanos en `conversations` (evaluación del 2026-09-03) y sin FK | idem — NO re-medido por Hyde |
| `jax-workspace/proyectos/` entra al restic local y R2; restauración probada 250/250 sha256 (25-sep) | adenda Jax#275; `backup-hall9000.sh:123` |
| `proyectos/` es `fruiz:fruiz 775` sin setgid: `jaxsvc` no puede escribir | `ls -ld` 2026-10-02; arreglo en Jax#276 (abierto) |
| LAS MANOS recibe `proyecto: str` y arma `proyectos/<slug>/`, sin validar contra B9 | `las_manos/procesamiento_routes.py:181,238` |
| jax-platform no tiene API ni pantalla de proyectos | `git grep` en `origin/master` |

## 3 · Qué cambia respecto del spec del 22-sep

- **Modelo de datos:** se adopta el de B9 (`jax_project_scope`, `jax_project_membership`,
  `jax_project_membership_event`). Quedan **sin efecto** `project_members`,
  `projects.estado` y `project_audit` del spec (§3). La auditoría es el registro de
  eventos de B9, que no se puede editar ni borrar.
- **Papeles:** lector = `VIEWER`, editor = `CONTRIBUTOR`, dueño = `OWNER`. `REVIEWER` no se
  ofrece en la interfaz.
- **Superadmin:** ya no es «dueño sin fila». Por D2 entra como miembro OWNER en la misma
  transacción que crea el proyecto, y por D4 el admin lo es **solo en su tenant**.
- **Estados:** `ACTIVE`, `ARCHIVED`, `HIDDEN` (D3). `DISABLED` no se expone en la interfaz.
  Manda `jax_project_scope.status`; `projects.status` es espejo (D5).
- **Destruir queda FUERA de esta versión** (Fernando, 2026-10-02). Ocultar basta para sacar
  un proyecto de la vista. Destruir exige borrar en cascada, que es lo que envenena el HNSW
  (MDEV-41227); tendrá su propio diseño. El §8 del spec sigue vigente para cuando llegue.

## 4 · Reparto en tres entregas

| Entrega | Contenido |
|---|---|
| **E1 — Proyectos** (esta adenda) | API y pantalla de proyectos y miembros, inicializar HAMURABI, huérfanos a su proyecto, FKs, selector de proyecto en el chat |
| **E2 — Selector** | subida por lotes (topes 100 MB/archivo; 250 archivos o 1 GB/lote, Fernando 2026-09-25), LAS MANOS con `project_uuid` validado contra B9, permisos (Jax#276), mover LACTOVI a `proyectos/<uuid>/` |
| **E3 — Respaldo del chat** | un PDF sin texto pegado en el chat va al procesador (spec §10) |

Cada entrega tiene su plan, su auditoría de escalón 3 y su despliegue.

## 5 · E1 — diseño

### 5.1 Enfoque
jax-platform usa la autoridad de B9 **en proceso**, como ya hacen `backend/api/chat.py` y
`backend/api/admin/users.py`. La plataforma no reimplementa papeles: toda mutación pasa por
`ProjectAuthorityAdmin` con un `MutationAuthorizationRequest`, y toda lectura por
`ProjectScopeAuthorityResolver`. Un solo lugar decide.

### 5.2 API (`backend/api/proyectos.py`)

| Ruta | Qué hace | Autoridad |
|---|---|---|
| `GET /proyectos?vista=activos\|archivados\|ocultos&cursor=` | lista paginada por cursor; solo proyectos con membresía ACTIVE del usuario; `ocultos` solo para el admin del tenant | resolvedor B9 |
| `POST /proyectos` (cabecera `Idempotency-Key`) | crea; el creador queda OWNER y el admin entra solo | `create_project` (D1/D2) |
| `GET /proyectos/{id}` | detalle y papel efectivo del usuario | resolvedor B9 |
| `PATCH /proyectos/{id}` | renombrar / descripción; mínimo OWNER | **nuevo en jax** (§5.3) |
| `POST /proyectos/{id}/estado` | archivar, restaurar, ocultar | `set_project_lifecycle` |
| `GET /proyectos/{id}/miembros` | lista de miembros | resolvedor B9 |
| `POST /proyectos/{id}/miembros` | invitar (mismo tenant) con papel lector/editor/dueño | `grant_member` |
| `PATCH /proyectos/{id}/miembros/{user_id}` | cambiar papel | `change_project_role` |
| `DELETE /proyectos/{id}/miembros/{user_id}` | quitar | `revoke_member` |
| `GET /proyectos/candidatos?q=` | usuarios activos del mismo tenant para invitar | tenant del usuario |

**Errores (spec §4, se mantiene):** **404** si el proyecto no existe, está oculto o no eres
miembro, sin revelar cuál; **403** si eres miembro y te falta el papel. Los errores de B9
(`LastOwnerRequired`, `TenantAdminMembershipProtected`, `MemberNotFound`…) se traducen a
códigos estables con texto i18n en el cliente; nunca se muestra el mensaje interno.

### 5.3 Lo que falta en jax (PR chico, aparte, antes de la plataforma)
- **Renombrar** (`rename_project`): mínimo OWNER, proyecto ACTIVE, actualiza `projects.name`
  y deja evento en `jax_project_membership_event` (o el registro equivalente que B9 acepte),
  con la misma disciplina de candados e idempotencia que el resto.
- **Consulta de «mis proyectos»** paginada por cursor, si el resolvedor no la ofrece. Se mide
  primero; solo se agrega lo que falte.

### 5.4 Migraciones en producción (por runbook, con respaldo verificado de `jax_memory` antes)
1. **HAMURABI:** `bootstrap_existing_project` con Fernando como único OWNER, alcance ACTIVE,
   tenant 1 (Fernando, 2026-10-02: «A»). Los demás se invitan desde la pantalla.
2. **Huérfanos:** se crea el proyecto archivado «Evaluación grounding SP3 · 2026-09-03»
   (tenant 1, dueño Fernando) y los 241 ids se reescriben a su id en `conversations`,
   `messages`, `facts`, `decisions` y `action_items`. Conteos antes y después, iguales;
   se re-miden el día de la migración (los de §2 no son de Hyde).
3. **FKs de `project_id`** en esas cinco tablas hacia `projects(id)`. ⚠️ Un `ALTER` sobre
   `messages` (lleva el índice vectorial) puede reconstruir la tabla. El plan lo ensaya
   primero en una copia y mide si toca el HNSW; si no hay forma segura, la FK de esa tabla
   se separa con su motivo escrito en `DEUDA.md` y la validación queda en la aplicación.

### 5.5 Interfaz
- **Ruta `/proyectos`:** pestañas Activos, Archivados y Ocultos (solo admin); «Nuevo
  proyecto»; lista paginada.
- **Dentro de un proyecto:** pestañas **Miembros** (invitar del mismo tenant, cambiar papel,
  quitar) y **Ajustes** (renombrar, archivar; el admin además ocultar y restaurar).
  **Documentos** llega con E2.
- **Selector «Personal / proyecto» en el chat:** solo ofrece proyectos activos donde eres
  miembro. Es lo único que manda `project_id` a `/chat`. Un proyecto archivado se lee pero
  no recibe chat nuevo.
- i18n (`es.js`/`en.js`), tokens de color, claro y oscuro. Toda confirmación en `Dialogo`;
  ningún `confirm(`, `alert(` ni `prompt(`, tampoco desnudos.

### 5.6 Las cuatro del rendimiento
1. **Índices:** `EXPLAIN` sobre las consultas reales de la lista, los miembros y los
   candidatos; sin `Using filesort` en el camino caliente.
2. **Caché:** ninguno nuevo sin medición previa.
3. **Async:** `aiomysql` y el pool existente; nada bloqueante en las rutas.
4. **Carga:** antes del GO, prueba sobre la lista de proyectos y sobre el turno de chat con
   proyecto (p95, concurrencia real); el número va a la Biblioteca.

### 5.7 Pruebas y cierre
- TDD contra una base de prueba real (`jax_test`), incluidos los casos de autoridad: no
  miembro → 404, lector que intenta invitar → 403, último dueño, admin protegido, otro tenant.
- Los pisos de CI se re-miden sobre `master` al integrar (hoy: 3071 con DB, 1055 vitest).
- Auditoría adversarial de escalón 3 antes de integrar cada PR.
- **Despliegue coordinado:** jax-platform tiene pendientes en master (#170–#174) sin
  desplegar; se despliega una sola vez, avisando antes a la sesión que lo coordina.

## 6 · Fuera de alcance de E1
Destruir · subida de documentos y Selector (E2) · respaldo del chat (E3) · invitar a otro
tenant · papeles por documento · notificaciones de invitación por correo.
