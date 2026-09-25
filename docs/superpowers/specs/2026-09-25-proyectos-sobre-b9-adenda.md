# Proyectos sobre la autoridad de B9 — adenda al spec del 2026-09-22

> Fecha: 2026-09-25 · Autor: Mr. Hyde · Continúa: `2026-09-22-proyectos-y-selector-design.md`
> Estado: adenda para Fernando. Separa lo que se construye YA (compatible con las dos
> lecturas) de tres decisiones de AUTORIDAD que son suyas.

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

- **§1.3 ya no es cierto.** `/chat` valida `project_id` contra B9 desde jax-platform#155:
  responde 403 `project_scope_denied` sin membresía ACTIVE. Los 241 huérfanos siguen en la
  base, pero ya no pueden crecer por `/chat`.
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
| «un proyecto nunca se queda sin dueño» | **B9 NO lo impide hoy** (`revoke_member` y `change_project_role` no cuentan OWNERs). Se agrega en `project_authority.py`, bajo el mismo `FOR UPDATE` |
| LAS MANOS recibe `project_uuid` | valida `projects` + `jax_project_scope.status='ACTIVE'` |

## 4 · Tres decisiones de AUTORIDAD que son de Fernando

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

**Recomendación de Hyde:**
- D1 (a) y D3 (a): son las decisiones que Fernando ya tomó el 22, y B9 no las consideró
  porque se escribió después sin tenerlas delante.
- D2 (b): el acceso implícito es la forma más común de que un permiso se escape, y agregar
  al superadmin como miembro cuesta un clic que queda en el registro. **D2 (b) contradice lo
  que Fernando decidió el 22**, así que se la presento, no la aplico.

## 5 · Qué se construye YA (vale con cualquier respuesta a D1–D3)

1. **Permisos de `proyectos/`** para `jaxsvc` (§5 del spec), con un test que escribe como `jaxsvc`.
2. **LAS MANOS con `project_uuid`**: valida `projects` + scope ACTIVE; carpeta `proyectos/<uuid>/`.
3. **API de Proyectos** en jax-platform:
   - crear, SOLO superadmin por ahora (lo más estricto de D1; se abre si Fernando elige D1-a);
   - «mis proyectos»;
   - miembros vía `ProjectAuthorityAdmin`, con la regla de «nunca sin dueño».
4. **Subida por lotes** con los topes aprobados, y el **Selector**.
5. **Selector de proyecto en el chat.**
6. **Respaldo del chat** (§10 del spec).
7. **LACTOVI** como primer proyecto real: se crea, Fernando queda como OWNER y su carpeta
   pasa a `proyectos/<uuid>/`, con verificación de sha256.

**Queda para después de D1–D3:** archivar, ocultar y destruir; los 241 huérfanos (van a un
proyecto archivado: D3); las FKs de `project_id` (van después de los huérfanos).
