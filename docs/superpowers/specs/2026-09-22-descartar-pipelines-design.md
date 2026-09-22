# Descartar, recuperar y ocultar pipelines detenidos — diseño

> Fecha: 2026-09-22 · Autor: Mr. Hyde · Decisiones: Fernando (misma fecha)
> Repos: **jax** (máquina de estados de Jacobs) y **jax-platform** (API y panel)
> Estado: spec para revisión de Fernando. Sin plan todavía.

## 1 · El problema

El panel "Detenidos" de jax-platform muestra todo pipeline en `aborted` o `expired`
(`frontend/src/components/RightPanel/RightPanel.jsx:20`) y su única acción es
**Continuar**. No existe forma de quitar uno:

- `POST /pipeline/{id}/cancel` responde **409** sobre `aborted` porque lo trata como
  terminal (`jacobs/routes.py:760`).
- No existe ningún endpoint de borrado.
- Un detenido sólo deja de verse cuando suficientes pipelines nuevos lo empujan fuera de
  la primera página de `GET /pipelines` — eso es un accidente, no un diseño.

Captura de Fernando del 2026-09-20: *"como cancelo o borro esto"*.

## 2 · Decisiones (Fernando, 2026-09-22)

| Acción | Quién | Desde | Hacia | ¿Reversible? |
|---|---|---|---|---|
| **Descartar** | el dueño del pipeline | `aborted`, `expired` | `discarded` | sí |
| **Recuperar** | quien lo descartó, o el superadmin | `discarded` | el estado EXACTO previo | — |
| **Borrar** (= ocultar) | sólo el superadmin | `discarded` | `hidden` | sí, por el superadmin |
| **Restaurar** | sólo el superadmin | `hidden` | `discarded` | — |

- **"Borrar" no destruye nada.** Decisión de Fernando: *"nos da la posibilidad de
  arrepentirnos"*. Ninguna fila sale de la base — tampoco de `axioma_usage`, así los
  totales de gasto nunca cambian retroactivamente.
- **Nunca se oculta en un paso.** Sólo se oculta un `discarded`; el descarte es la antesala.
- **Los descartados NO van en la pantalla principal.** Van en una vista propia, paginada
  (*"van a ser muchos en el tiempo"*). Los ocultos, en una vista sólo del superadmin.

## 3 · Máquina de estados (jax)

Dos valores nuevos en `PipelineStatus` (`jacobs/models.py`): `discarded` y `hidden`.

- **Terminales y sin cupo:** se agregan a `ESTADOS_SIN_CUPO` (`jacobs/policy.py:47`).
- **El reaper no los toca:** su conjunto no-terminal (`pending/running/interrupted`,
  `jacobs/reaper.py:134`) no cambia; un test lo fija.
- **`/continue` los rechaza con 409** (sólo continúa `aborted`/`expired`). Para continuar
  un descartado, primero se recupera.
- **`/cancel` los rechaza con 409**, igual que a todo terminal.

### Columnas nuevas en `jacobs_pipelines`
(mismo patrón de `ALTER ... ADD COLUMN` idempotente de `jacobs/store.py:1037`)

| Columna | Tipo | Para qué |
|---|---|---|
| `status_previo` | VARCHAR(20) NULL | a qué vuelve al recuperar (`aborted`/`expired`) |
| `descartado_por` | VARCHAR(50) NULL | user_id — decide quién puede recuperar |
| `descartado_at` | DOUBLE NULL | orden de la vista de descartados |

### Transiciones: compare-and-set, siempre
Cada transición usa `pipeline_update_status_si_epoca` (`jacobs/store.py:1475`) con
`desde=` explícito. Una carrera (alguien continúa mientras otro descarta) produce 409, nunca
dos escrituras. Descartar escribe `status_previo`, `descartado_por` y `descartado_at` en el
**mismo** UPDATE; recuperar restaura `status_previo` y limpia las tres.

### Auditoría
Cada transición deja un evento en `jacobs_events`: `PIPELINE_DISCARDED`,
`PIPELINE_RECOVERED`, `PIPELINE_HIDDEN`, `PIPELINE_RESTORED`, con `user_id`, estado de
origen y de destino. `sql_eventos_de_causa` (jax-platform) filtra por `EVENTOS_DE_CAUSA`,
así que estos eventos no contaminan la causa de la detención; un test lo fija.

### Rutas nuevas en Jacobs
`POST /pipeline/{id}/discard`, `/recover`, `/hide`, `/restore`. Jacobs valida la
transición; **quién** puede pedirla lo valida jax-platform (§4), que es quien conoce el
papel del usuario — mismo reparto que `/cancel` hoy.

## 4 · API y permisos (jax-platform)

`backend/api/pipelines.py`, proxies con el mismo patrón que `cancel_pipeline` (`:992`):

| Ruta | Guardia |
|---|---|
| `POST /pipelines/{id}/discard` | `_require_pipeline_owner` |
| `POST /pipelines/{id}/recover` | `descartado_por == user` **o** superadmin |
| `POST /pipelines/{id}/hide` | superadmin |
| `POST /pipelines/{id}/restore` | superadmin |
| `GET /pipelines?estado=discarded` | dueño; paginado |
| `GET /admin/pipelines/ocultos` | superadmin; paginado, de todos los usuarios |

- `GET /pipelines` sin filtro **excluye** `discarded` y `hidden`. Hoy el panel filtra
  `aborted/expired` en el cliente sobre la primera página; con los descartados fuera del
  SQL, la página deja de gastarse en filas que nadie va a ver.
- Un `hidden` no aparece en ninguna lista de su dueño: para él, ya no existe.

## 5 · Interfaz

- **Panel "Detenidos"**: botón **Descartar** junto a Continuar. Confirmación en ventana
  propia (`components/Dialogo.jsx`), nunca `confirm()`.
- **Historial de pipelines**: pestaña **Descartados**, paginada, con **Recuperar** por
  fila y, si eres superadmin, **Borrar** (con `ConfirmacionSuma`).
- **Administración → Pipelines ocultos** (sólo superadmin): lista paginada de todos los
  usuarios, con **Restaurar**.
- Textos en `i18n/es.js` y `en.js`; colores por tokens; claro y oscuro.

## 6 · LAS CUATRO DEL RENDIMIENTO

1. **Índices.** Las listas nuevas filtran por `(user_id, tenant_id, status)` y ordenan por
   `created_at` o `descartado_at`. El índice existente `idx_jacobs_pipelines_duenio
   (user_id, tenant_id, created_at)` no cubre el filtro por estado. El plan decide entre
   ampliar ese índice o crear uno nuevo, **con `EXPLAIN` sobre la consulta real** y sin
   `Using filesort`. La vista de ocultos (todos los usuarios) va por `idx_pipelines_status`.
2. **Caché.** Ninguno nuevo.
3. **Async.** Todo por `aiomysql` y el cliente HTTP compartido; nada bloqueante.
4. **Carga.** Se mide `GET /pipelines` y la lista de descartados con el usuario de más
   pipelines, antes del GO. El número va a la Biblioteca.

## 7 · Pruebas (mínimo)

- Cada transición válida, y cada inválida con 409 (incluidas `continue` y `cancel` sobre
  `discarded`/`hidden`).
- Recuperar devuelve al estado **exacto** previo: un `expired` vuelve a `expired`, no a
  `aborted`.
- Carrera descartar ↔ continuar: sólo una gana. Con mutación que quita el `desde=`, el test
  cae.
- Permisos: otro usuario no descarta ni recupera; un no-superadmin no oculta ni restaura
  (403); el superadmin recupera lo que descartó otro.
- `axioma_usage` intacto tras ocultar: la suma de gasto del mes es la misma antes y después.
- El reaper ignora `discarded`/`hidden`.
- Escaneo de `confirm(`, `alert(`, `prompt(` desnudos en el frontend.

## 8 · Fuera de alcance

Destruir pipelines de verdad. No se pidió; si algún día hace falta, se diseña aparte,
con el mismo criterio que la destrucción de proyectos (sólo desde oculto, sólo superadmin,
con respaldo verificado).
