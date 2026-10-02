# Proyectos E1 — 2026-10-02

> HISTORIA. Autor: Mr. Hyde (hall9000). Decisiones: Fernando, en sesión.
> Spec: `docs/superpowers/specs/2026-10-02-proyectos-e1-adenda-design.md` · Plan: `docs/superpowers/plans/2026-10-02-proyectos-e1-plan.md`

## Qué se hizo
- **Coordinación.** El Selector llevaba la marca `[EN CURSO: macbook-pro]` desde el 23-sep. Se revisó la Mac (checkout y memoria): la UI nunca empezó. Fernando pasó las marcas del Selector y del respaldo del chat a `hall9000` (claude-skills `6ff0517`). Codex (LAS VOCES), Kimi (BinB) y la otra sesión de Claude (T16, El Faro) confirmaron que no tocaban este trabajo.
- **Reconciliación.** El spec del 22-sep quedó superado por B9 y D1–D5. Fernando decidió:
  - Destruir queda fuera de esta versión.
  - HAMURABI se inicializa con él como único OWNER (más los admins, por D2).
  - El trabajo se reparte en tres entregas: E1 Proyectos, E2 Selector y E3 respaldo del chat.
- **Parte A (jax), Jax#315, merge `2832cdd`.**
  - `rename_project`.
  - `jax/memory/project_queries.py`.
  - `scripts/proyectos_e1_migrar.py`: HAMURABI, huérfanos a un proyecto archivado, mapa de reversión 0600 y `--revertir` que contrasta el mapa con la base.
  - `scripts/proyectos_e1_fks.py`.
  - Runbook `docs/runbooks/proyectos-e1-produccion.md`.
  - **No desplegada.**
- **Parte B (jax-platform), jax-platform#176.**
  - API `/api/proyectos`.
  - Pantallas `/proyectos` y detalle.
  - Selector «Personal / proyecto» en el chat.
  - Prueba de carga `loadtest/proyectos_e1.py`.

## Las Cuatro — números medidos (VERDAD OPERACIONAL a la fecha; caduca si cambia esquema, volumen o infraestructura)
Todo medido en hall9000 con un proceso uvicorn y cliente, backend y MariaDB en el mismo host, contra una base de prueba (nunca `jax_memory`).
- **Lista** `GET /api/proyectos`: 1.000 proyectos sembrados; el usuario es miembro de 300. Con 50 usuarios durante 2 min: 794,6 rps, p50 42 ms, p95 188,8 ms, p99 302,9 ms, 0 errores.
  - El throughput toca techo cerca de c=10–25 (1.273 rps de pico).
  - El p95 pasa de 250 ms entre c=50 y c=100.
- **Peor caso del chat con proyecto.** Es la ruta real `retrieve_authorized`, que toma FOR UPDATE en `jax_users`, en `jax_project_scope` de P y en la membresía. 50 miembros de un mismo P mientras el OWNER renombra y archiva a 1 Hz.
  - p95 de los turnos permitidos: 72,7 ms sin escrituras y 70,7 ms con escrituras.
  - 0 errores 1213 (deadlock), 0 errores 1205 (lock timeout), 0 otros.
  - El p50 del turno (~40 ms) se explica al menos a medias por la **espera del pool** (10 conexiones, ~5 pedidos por turno, 50 clientes). La parte atribuible al candado de la fila de P **no se midió por separado**.
  - Una primera medición (que usaba `resolve_scope`, sin candado) fue **inválida** y la auditoría la rechazó.
- **Mutaciones** con 20 clientes: crear p95 43 ms; invitar p95 42 ms; quitar p95 42 ms.
- **Índices.** La lista usa `idx_jax_project_membership_user_list` (`range`, sin filesort ni temporary). La lista de miembros usa temporary+filesort, acotado a un proyecto.

**Riesgos que no se cerraron:**
- En producción el pool se comparte con todo el backend.
- No se midió la lectura con memoria real sembrada.
- Hubo un solo escritor.
- Hay 503 posibles por la carrera entre archivar y un turno en vuelo; no se midieron.
- La lectura de B9 con FOR UPDATE serializa los turnos de un mismo proyecto. Es diseño previo de B9; E1 lo expone más.

## Lecciones técnicas
- MariaDB 12.3.3 rechaza `ADD FOREIGN KEY ... ALGORITHM=INPLACE` con `foreign_key_checks=1` (error 1846). Por eso las FKs se crean con los checks apagados solo en esa sesión, `LOCK=SHARED` y un conteo de huérfanos antes y después que se repite en cada corrida. COPY habría reconstruido `messages`, que tiene el HNSW (MDEV-41227).
- En Tailwind 3.4, `min-h-11` junto a `min-h-6` da 24px: gana el que se emite último. El alto de 44px va en una sola constante (`TAMANO_BOTON_44`), con una prueba sobre el CSS compilado.
- Una prueba de carga «verde» puede serlo por cómo está armada. Hay que medir la ruta que toma el candado, no una vecina sin candado.
- Las guardas del repo (fail-soft marcado, `connect_timeout`, `.env` con sudo, archivos de test en CI, pisos exactos) atrapan en CI lo que la revisión no ve. No se esquivan: se cumplen.

## Pendientes
- **Despliegue de E1:** espera el GO de Fernando. Orden: respaldo con restauración probada → jax → jax-platform (junto con #170–#174, una sola vez) → índice → migración → ensayo y FKs → verificación.
- **E2 (Selector) y E3 (respaldo del chat):** sus líneas siguen en `PENDIENTES.md` con `[EN CURSO: hall9000]`.
- **Hallazgos ajenos a E1, para Fernando:**
  - `frontend/src/api/websocket.js`, en dev con host local, fija el WebSocket a `:8080` (la plataforma de producción en hall9000).
  - El `conftest.py` de jax intenta conectarse a `jax_memory` cuando no corre con `CI=true`.
