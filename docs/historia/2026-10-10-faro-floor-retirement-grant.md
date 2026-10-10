# Registro de coordinación — grant y revert de Faro F1.1

**Archivado:** 2026-10-10. Fuente: `TRASPASO.md` de la rama `codex/faro-floor-retirement-grant`. Este registro conserva estado, verificaciones y siguientes pasos; no otorga autoridad.

**Objetivo:** cerrar el permiso temporal de piso (#390), integrar el revert requerido de #385 (#387) y dejar registrado si producción puede habilitarse.

## Estado verificado al 2026-10-10

- `master` incluye #393 (`f4227ab007344e34551437b66f20803b1e19ac60`), cuyo guard `38055164507` terminó SUCCESS sin carrera, y #394, merge commit `834170c5b7443eb16e1713378ef69103595dfec2`. El `post-merge-guard` de #394 terminó SUCCESS en policy push run `38056397779`; `master@834170c5b7443eb16e1713378ef69103595dfec2`, sin carrera.
- #394 (`f0dcdaf9c187935fa9c0ad3800f30a570838a6f5`) pasó auditoría Tier 3 APROBADO, 0/0/0; sus 32 checks terminaron SUCCESS. La ejecución exacta del piso pasó `238 passed` en Python 3.12.
- Este branch quedó rebasado localmente sobre `master@834170c5b7443eb16e1713378ef69103595dfec2`. Su diff actual respecto de master contiene solo el registro del grant y `TRASPASO.md`; el merge dedicado debe cambiar únicamente el registro. El árbol `834170c5…^{tree}` es idéntico al head auditado de #394 (`f0dcdaf9…^{tree}`), por eso la simulación sobre ese árbol produjo la huella de candidato `9f020c472765aba2ea0428fcf00ad0734c1b6145c2e0643f9781e7859043d4a5` y el comparador la aceptó. Recalcular tras archivar el handoff y actualizar la base.
- El grant de este branch apunta a la definición exacta del piso 20 introducida por #392: merge `8d1d1c49c5795fd6416085084265006fbcb6d69b`, padres `0d6484f2f56059e9fb0d11b04de8990375ee7c53` y `597b9a75d32874a84613e1ed7a0de40ea9e620ab`, SHA de definición `394bf2567f47c8d6942c3e30019376f34fedc15c175bc100e9cc830f2fdc3595`.
- Simulación local sobre el árbol exacto de #394: merge dedicado del grant + revert adaptado de #385, 11 rutas; el comparador de esa base aceptó la transición y el diff exacto fue `9f020c472765aba2ea0428fcf00ad0734c1b6145c2e0643f9781e7859043d4a5`. La simulación no es merge ni CI oficial.
- #387 sigue OPEN sobre el merge #385; su head remoto observado era `b5f70fef88a6f7c2a19a9cfca4b7dd2716a04a51`. Hay que reapilarlo sobre el merge real del grant, revisar el diff final y recalcular la huella.
- #388 sigue bajo el owner de su sesión; su rama y worktree están fuera de alcance.
- `/etc/jax/authority/trusted-root.json` no existe. La búsqueda local de ese nombre no encontró copia. Dos repositorios restic conocidos también se consultaron antes: no encontraron esa ruta en sus snapshots listados. No crear ni sustituir una raíz; producción sigue bloqueada hasta recibir/restaurar material autorizado y verificar el ledger.

## Secuencia de cierre

1. Rebasar este branch sobre `master@834170c5b7443eb16e1713378ef69103595dfec2`; comprobar que solo modifica el registro de grants y recalcular simulación/hash sobre ese árbol exacto.
3. Archivar este handoff en `docs/historia` mediante un PR separado, integrarlo con auditoría/CI/guard y quitar `TRASPASO.md` de este branch; el merge del grant debe cambiar solo el registro.
4. Auditar Tier 3 el SHA exacto del grant; publicar/actualizar PR #390, CI verde, preflight exacto, merge dedicado y post-merge-guard.
5. Rebasar #387 sobre el merge real #390. Reauditar SHA exacto, CI verde, preflight, merge con ventana de Fernando abierta y guard.
6. No declarar producción lista mientras falte la raíz confiable. No tocar #388 ni fases con otro owner.
