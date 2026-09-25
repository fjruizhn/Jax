# Proyectos sobre la autoridad de B9 — adenda al spec del 2026-09-22

> Fecha: 2026-09-25 · Autor: Mr. Hyde · Continúa: `2026-09-22-proyectos-y-selector-design.md`
> Estado: adenda para Fernando. **v2 tras la auditoría de escalón 3 del 2026-09-25, que RECHAZÓ
> la v1 con 3 BLOCK.** La v1 describía B9 por su código y sus docstrings, no por lo que corre.
> Esta versión lo corrige: Proyectos queda BLOQUEADO detrás de B9 (§5).

## 1 · Por qué hace falta

El spec del 22 propone su propio modelo de miembros (`project_members`, `project_audit`,
`estado` en `projects`). El **23-sep** entró B9 (`97e2abe`, «harden B9 project authority
boundary», y la remediación jax#269 integrada el 25-sep) con OTRO modelo, ya auditado:

| Spec del 22 | B9 (verificado en `origin/master` `a723964`) |
|---|---|
| `project_members (papel: dueño/editor/lector)` | `jax_project_membership (project_role: OWNER/CONTRIBUTOR/REVIEWER/VIEWER, status ACTIVE/REVOKED)` — `jax/memory/b9_migrations/003_project_scope_authority.sql` |
| `projects.tenant_id` + `projects.estado` | `jax_project_scope (project_id, tenant_id, status ACTIVE/DISABLED)` con FK a `projects(id)` y a `jax_tenants` |
| `project_audit` | `jax_project_membership_event`, append-only por trigger |
| `exigir_papel` nuevo en jax-platform | `ProjectScopeAuthorityResolver` (`jax/memory/scope_authority.py`), que `/chat` YA usa (`backend/api/chat.py:1162-1193`) |
| alta, invitar, cambiar papel, quitar | `ProjectAuthorityAdmin` (`jax/memory/project_authority.py`): `grant_member`, `change_project_role`, `revoke_member`, `set_project_scope_status`, `bind_legacy_project_scope` |

**Construir el modelo del 22 al lado del de B9 daría dos fuentes de verdad de autoridad**
(Principio IX). Esta adenda adopta el modelo de B9 y lista dónde el spec del 22 y B9
dicen cosas distintas.

## 2 · Hechos nuevos que corrigen el spec del 22

- **§1.3 SIGUE SIENDO CIERTO EN PRODUCCIÓN — hueco de seguridad vivo.** La validación de B9
  (`_scope_for_chat`, jax-platform#155) está en master pero **no desplegada**. Producción
  corre `1c600fd` (#160, arrancado el 2026-09-23 09:28); verificado el 2026-09-25 que
  `grep -c _scope_for_chat /srv/jax-prod/jax-platform/backend/api/chat.py` da 0. Cualquier
  usuario activo que arme a mano un `/chat` con `project_id=1` lee y escribe la memoria de
  HAMURABI, y los huérfanos pueden seguir creciendo. La interfaz nunca manda `project_id`.
- **B9 no está desplegado en ninguna de sus partes.** En `jax_memory` de producción no existe
  ninguna tabla `jax_project_%` ni `memory_%`, verificado el 2026-09-25. `projects` tiene una
  sola fila: HAMURABI (id 1). La migración 003 está escrita para no aplicarse sola.
- **§1.5 y §8 «entra al restic»: ya estaba.** `/home/fruiz/jax-workspace` entra en el respaldo
  de SISTEMA desde el 2026-09-22 (`SISTEMA_RUTAS` de `backup-hall9000.sh` incluye
  `/home/fruiz`). Verificado el 2026-09-25: `restic ls --recursive latest` de
  `.../jax-workspace/proyectos` da 360 entradas en local y en R2, igual que el disco.
- **§8 restauración probada: HECHA el 2026-09-25 03:42.** `restic restore` del snapshot de
  sistema más reciente, desde local (`9745f866`) y desde R2 (`46e8b42f`), a
  `/srv/backup-adata/restauracion-prueba/`: **250/250 archivos iguales por sha256** en los dos
  (LACTOVI incluido). Se borró la copia al terminar.
- **§12 costo de R2: $0.** La tarifa oficial de Cloudflare (developers.cloudflare.com/r2/pricing,
  leída el 2026-09-25) es $0,015/GB-mes con **10 GB-mes gratis**. El repositorio mide 1,92 GiB
  y `jax-workspace` 302 MB en total.
- **§12 topes de subida: APROBADOS por Fernando el 2026-09-25**: 100 MB por archivo; 250
  archivos o 1 GB por lote. Medido en Nextcloud: 1.165 documentos, p99 41 MB, máximo 91 MB;
  el expediente mayor pesa 761 MB y el más numeroso tiene 217 archivos.
- **Orden: opción B (el del spec), elegida por Fernando el 2026-09-25.**

## 3 · Correspondencia que se adopta

| Spec | B9 |
|---|---|
| lector | `VIEWER` |
| editor | `CONTRIBUTOR` |
| dueño | `OWNER` |
| — | `REVIEWER`: verifica memoria. No se ofrece en la pantalla de Proyectos: no está en el spec |
| crear proyecto | fila en `projects` + `jax_project_scope` ACTIVE + membresía `OWNER` del creador, en UNA transacción, con su evento |
| invitar / cambiar / quitar | `ProjectAuthorityAdmin`, sin reimplementar nada |
| «un proyecto nunca se queda sin dueño» | **B9 NO lo impide hoy** (`revoke_member` y `change_project_role` no cuentan OWNERs). Hace falta: bloquear `jax_project_scope` del proyecto OBJETIVO con `FOR UPDATE` antes de contar (sin eso hay *write skew*), contar solo OWNER ACTIVE cuyo `jax_users` esté ACTIVE, y decidir la excepción del superadmin que da el spec del 22. Es código de B9: reabre su auditoría |
| LAS MANOS recibe `project_uuid` | valida `projects` + `jax_project_scope.status='ACTIVE'`. Que el alcance esté ACTIVE es propiedad del PROYECTO, no permiso del llamante: el papel lo verifica jax-platform al encolar, y hay que decidir si se vuelve a verificar al ejecutar (TOCTOU) o si se usa una política de servicio de B9. `scripts/procesar_archivos.py` queda fuera del modelo de papeles |

## 3-bis · Defecto de B9: no hay forma de crear el primer alcance ni el primer miembro

Lo encontró la auditoría de escalón 3 corriendo el resolvedor real en memoria, con un cursor
falso. **Hyde NO lo reprodujo por su cuenta**: entra como hallazgo de auditoría, no como hecho
verificado por él. Lo que sí verificó Hyde son B1 y B2 (§2).
- Todo método de `ProjectAuthorityAdmin` pide `PROJECT_SHARED` (`project_authority.py:38-40`).
- El resolvedor real, con `project_id`, exige que el alcance exista y esté ACTIVE y que haya
  membresía; sin `project_id`, rechaza (`scope_authority.py:96-107, 219`).
- Resultado:
  - `bind_legacy_project_scope` da `PROJECT_SCOPE_UNBOUND`;
  - que un superadmin se agregue como miembro da `project membership is missing`;
  - reactivar un alcance DISABLED da `project scope is disabled`.
- Los tests (`tests/test_project_scope_authority.py:10-15`) usan `_TrustedAdminResolver`, que
  devuelve `memory:admin` sin leer nada: el control no puede fallar.
- **Lavado de autoridad:** un superadmin que sea VIEWER del proyecto 1 puede enlazar el 2,
  porque se valida el alcance de la PETICIÓN y `_admin_shape` deja pasar `memory:admin` para
  cualquier `project_id` (`project_authority.py:24`).
- Otros agujeros del mismo código:
  - `bind_legacy_project_scope` acepta un `tenant_id` que elige quien llama;
  - `grant_member` no mira el `status` del alcance;
  - `change_project_role` puede promover a OWNER una fila REVOKED.

**Es código de B9, que lleva Codex** (`[EN CURSO: macbook-pro.codex]`). Se le deja anotado.
Hyde no lo reescribe por su cuenta.

## 4 · Decisiones de AUTORIDAD que son de Fernando

Sus decisiones del 22 y el contrato de B9 del 23 (los dos integrados por él) se contradicen.
No se resuelven por inferencia.

**D1 · ¿Quién crea un proyecto?**
- Spec: «cualquier usuario crea un proyecto y queda como dueño».
- B9: crear el alcance exige administrador GLOBAL (`set_project_scope_status`/`bind_…` con
  `bootstrap=True`: «an OWNER cannot bootstrap scope state»).
- (a) El spec: una operación nueva, `CREATE_PROJECT`, que cualquier usuario ACTIVE del tenant
  puede invocar y que solo crea un proyecto nuevo con él como OWNER.
- (b) B9: solo el superadmin crea; el usuario pide y el superadmin da de alta.

**D2 · ¿El superadmin ve y chatea en cualquier proyecto sin ser miembro?**
- Spec: «dueño de todos por su rol, sin fila de miembro».
- B9: el resolvedor exige membresía ACTIVE para leer; el administrador global solo tiene
  operaciones de gestión.
- (a) El spec: el resolvedor trata a un superadmin DB-backed y ACTIVE del mismo tenant como
  OWNER, con `membership_id=None` y la procedencia `jax_users.superadmin`.
- (b) B9: el superadmin se agrega como miembro cuando lo necesita, y queda registrado.

**D3 · Archivado y oculto.**
- `jax_project_scope.status` solo admite ACTIVE/DISABLED, y DISABLED cierra TODO, lectura incluida.
- El spec quiere «archivado: los miembros lo leen, pero no recibe documentos ni chat nuevo»
  y «oculto: solo lo ve el superadmin».
- (a) Estados nuevos en el CHECK, ARCHIVED y HIDDEN, con su semántica en el resolvedor:
  cambia el contrato de B9.
- (b) Sin «archivado legible»: archivar = DISABLED, y nadie lo lee hasta que se reactive.

**D4 · ¿Cómo nace la autoridad, y `admin` es lo mismo que `superadmin`?**
- B9 trata `admin`, `superadmin` y `super_admin` como administración global
  (`scope_authority.py:46`, `project_authority.py:54`); el spec habla solo de superadmin.
- Hace falta una operación de arranque que se resuelva contra el proyecto OBJETIVO, con
  `CREATE_PROJECT` atómico (`projects` + alcance + OWNER + evento en una sola transacción),
  y que exija que el tenant del administrador sea el del proyecto.

**D5 · ¿Qué campo manda en «archivado»?**
Hay tres: `projects.status` (ENUM que ya incluye `archived`), `jax_project_scope.status`
y el `estado` que proponía el spec del 22. Uno solo tiene que mandar.

**Recomendación de Hyde:**
- D1 (a) y D3 (a): son las decisiones que Fernando ya tomó el 22, y B9 no las consideró
  porque se escribió después sin tenerlas delante.
- D2 (b): el acceso implícito es la forma más común de que un permiso se escape, y agregar
  al superadmin como miembro cuesta un clic que queda en el registro. **D2 (b) contradice lo
  que Fernando decidió el 22**, así que se la presento, no la aplico.

## 5 · Orden real (reemplaza la lista «se construye YA» de la v1)

0. **Cerrar el hueco de producción de `/chat`**: desplegar jax-platform#155. Antes hay que
   aplicar B9 001-004 en producción con respaldo verificado y enlazar HAMURABI (spec §9.4).
   Si no, todo chat de proyecto da 403 (falla cerrado). **B9 es de Codex; el GO, de Fernando.**
1. Codex arregla el arranque de B9 (§3-bis) con tests contra el resolvedor REAL, y pasa
   auditoría nueva.
2. Fernando decide D1–D5.
3. Recién entonces: API de Proyectos, miembros, subida por lotes, Selector, selector de
   proyecto en el chat, respaldo del chat con proyecto, y LACTOVI.

**Lo único independiente de B9, y que se hace ya:** permisos de `proyectos/` para `jaxsvc`
(§5 del spec del 22). Sin eso LAS MANOS no puede escribir, con o sin B9.

**Invariante hasta migrar los huérfanos (m2 de la auditoría):** ningún `projects.id` en el
rango 900001–1400055. Si no, un proyecto nuevo heredaría memoria ajena. Hoy
`AUTO_INCREMENT` va en 2.
