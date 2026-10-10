# Registro de coordinación — grant y revert de Faro F1.1

**Archivado:** 2026-10-10. Fuente: `TRASPASO.md` de la rama `codex/faro-floor-retirement-grant`. Este registro conserva estado, verificaciones y siguientes pasos; no otorga autoridad.

El traspaso de la rama `codex/faro-grant-handoff-refresh` hasta `44704db54070a455d12819006e4be5ef13e21284` también se archiva aquí antes de la auditoría final de su PR #397: objetivo, evidencia, corrección del dictamen, estado de autoridad, decisiones, ownership y secuencia están asentados en las secciones siguientes. El archivo operativo `TRASPASO.md` se elimina del SHA final.

## Actualización verificada — 2026-10-10

- #396 se integró como `3643568c045caa44cfcaf08e3101f1ce5126e8ce`; `post-merge-guard` terminó SUCCESS, default `master@3643568c`, sin carrera.
- #395 se integró como `ec01b2191b1a7c1e05854a345a610d72f8b643ea`; `post-merge-guard` terminó SUCCESS, default `master@ec01b219`, sin carrera. Este es el tip oficial de esta actualización.
- #390 continúa OPEN en head `3e74c61463d1c7c6727b0fd40134bd774015b3f8`, base antigua `0d6484f2f56059e9fb0d11b04de8990375ee7c53`, estado UNSTABLE. El run `38049519208` falló el único caso pendiente del piso MariaDB durante `_apply_migration`, con `OperationalError 2013` (conexión perdida durante consulta); 11 casos pasaron y uno terminó con error. No se atribuye causa raíz al servidor ni se repite DDL automáticamente.
- #387 continúa OPEN/DIRTY en head `b5f70fef88a6f7c2a19a9cfca4b7dd2716a04a51`, base `9b2744c33c37fdfeec87bac03ddd00c9b76e8fb5`. El trabajo local de preparación del revert permanece separado y requiere reapilado sobre el grant real y nueva auditoría/CI.
- #388 sigue OPEN/DRAFT/CONFLICTING; su rama pertenece a otra sesión. El owner debe preservar los cambios Step 6/7 y reapilar luego de #390/#387, conservando el hardening y las ocho regresiones de #392 que ya están en `master`.
- El registro anterior decía que ocho repositorios Restic no contenían `trusted-root.json`, pero no conservó hora, snapshots, comandos ni salida: queda corregido como **reporte histórico no verificado**, no como hecho sobre respaldos. El 2026-10-10T19:12:56Z, en `hall9000`, `sudo -n test -e /etc/jax/authority/trusted-root.json` retornó 1; `sudo -n find /etc/jax /srv/faro /var/lib/jax -maxdepth 5 -type f \( -name trusted-root.json -o -name trusted-checkpoint-bootstrap-receipt.json -o -name trusted-checkpoints.log \)` no encontró esos archivos. El mirror local `/srv/faro/jax.git` declara `master@7024529e1ede4b3a3de92eefe4f60e3d2f567f01`; no se encontró `jaxctl` en PATH, así que el ledger no pudo verificarse por la interfaz operativa. Esta comprobación del host no determina qué contienen los backups. No crear ni sustituir anclas: producción sigue bloqueada hasta localizar material aprobado, verificar el ledger conforme a `docs/runbooks/authority-root-recovery.md` y conservar la procedencia.
- La auditoría Tier 3 inicial de este refresco documental, SHA `cf3c8b1140e2ff0156c4f74341485356dde3aef9`, fue RECHAZADO (1 BLOCK, 1 MAJOR): el SHA aún contenía `TRASPASO.md` y la búsqueda Restic estaba presentada como verificada sin alcance reproducible. Se corrigió la búsqueda a reporte no verificado y se registró la inspección del host; antes de la auditoría final se eliminará el traspaso de la rama y esta actualización quedará archivada aquí.
- La sesión principal pidió revisión read-only a GLM/ZCode; Kimi continúa sin poder responder por límite temporal HTTP 403. La revisión externa no sustituye la verificación en GitHub ni las auditorías obligatorias.

La siguiente verificación exacta debe actualizar este registro si los PR o el tip cambian; los puntos anteriores son historia fechada, no estado operacional perpetuo.

**Objetivo:** cerrar el permiso temporal de piso (#390), integrar el revert requerido de #385 (#387) y dejar registrado si producción puede habilitarse.

## Estado verificado al 2026-10-10

- `master` incluye #393 (`f4227ab007344e34551437b66f20803b1e19ac60`), cuyo guard `38055164507` terminó SUCCESS sin carrera, y #394, merge commit `834170c5b7443eb16e1713378ef69103595dfec2`. El `post-merge-guard` de #394 terminó SUCCESS en policy push run `38056397779`; `master@834170c5b7443eb16e1713378ef69103595dfec2`, sin carrera.
- #394 (`f0dcdaf9c187935fa9c0ad3800f30a570838a6f5`) pasó auditoría Tier 3 APROBADO, 0/0/0; sus 32 checks terminaron SUCCESS. La ejecución exacta del piso pasó `238 passed` en Python 3.12.
- Este branch quedó rebasado localmente sobre `master@834170c5b7443eb16e1713378ef69103595dfec2`. Su diff actual respecto de master contiene solo el registro del grant y `TRASPASO.md`; el merge dedicado debe cambiar únicamente el registro. El árbol `834170c5…^{tree}` es idéntico al head auditado de #394 (`f0dcdaf9…^{tree}`), por eso la simulación sobre ese árbol produjo la huella de candidato `9f020c472765aba2ea0428fcf00ad0734c1b6145c2e0643f9781e7859043d4a5` y el comparador la aceptó. Recalcular tras archivar el handoff y actualizar la base.
- El grant de este branch apunta a la definición exacta del piso 20 introducida por #392: merge `8d1d1c49c5795fd6416085084265006fbcb6d69b`, padres `0d6484f2f56059e9fb0d11b04de8990375ee7c53` y `597b9a75d32874a84613e1ed7a0de40ea9e620ab`, SHA de definición `394bf2567f47c8d6942c3e30019376f34fedc15c175bc100e9cc830f2fdc3595`.
- Simulación local sobre el árbol exacto de #394: merge dedicado del grant + revert adaptado de #385, 11 rutas; el comparador de esa base aceptó la transición y el diff exacto fue `9f020c472765aba2ea0428fcf00ad0734c1b6145c2e0643f9781e7859043d4a5`. La simulación no es merge ni CI oficial.
- #387 sigue OPEN sobre el merge #385; su head remoto observado era `b5f70fef88a6f7c2a19a9cfca4b7dd2716a04a51`. Hay que reapilarlo sobre el merge real del grant, revisar el diff final y recalcular la huella.
- #388 sigue bajo el owner de su sesión; su rama y worktree están fuera de alcance.
- Estado reportado en la primera versión del traspaso: `/etc/jax/authority/trusted-root.json` no existía en el host y dos repositorios Restic no mostraron esa ruta. No se preservó evidencia reproducible de los backups; la parte Restic queda **no verificada** y fue corregida en la actualización fechada al inicio de este documento. No crear ni sustituir una raíz.

## Secuencia de cierre registrada

1. Cerrar #397 con SHA limpio de `TRASPASO.md`, Tier 3, CI, verify-integration, merge y post-merge-guard.
2. Rebasar #390 una sola vez sobre el tip oficial post-#397, quitar su `TRASPASO.md` y conservar únicamente el cambio del registry. Recalcular `expected_diff_sha256` por simulación sobre el árbol exacto; después auditoría, CI, preflight, merge dedicado y guard.
3. Rebasar #387 sobre el merge real de #390. Revisar el revert completo y la huella de piso; reauditar, repetir CI, preflight, merge con ventana abierta y post-merge-guard.
4. Coordinar el siguiente rebase de #388 con su owner; conservar el hardening y las regresiones de #392, combinar Step 6/7, re-medidir el piso y auditar/validar el SHA final.
5. No desplegar mientras falten root/receipt/checkpoint válidos o no se pueda verificar el ledger. No sintetizar anclas; reanudar con la recuperación aprobada y el runbook `docs/runbooks/authority-root-recovery.md`.

Esta secuencia es un traspaso histórico, no prueba del estado actual. Cada paso parte del tip oficial y de la evidencia exacta vigente.
