# Traspaso continuo · Faro F1.1 paso 6

## Objetivo

Completar en esta rama aislada el permiso `RulePermit` de un solo uso y su auditoría
durable, conforme a `docs/superpowers/specs/2026-10-05-faro-f1-1-rule-authority-kernel-design.md`
§9–§15. No integrar ni publicar hasta cerrar los gates y resolver la pregunta humana de
`OVERLAY_ISSUED`.

## Estado verificado

- Rama: `codex/faro-f11-kernel-step6`.
- Base actual: `bf6b05809e1c3505efd7447b33685796ca42e3b6` (master posterior a #386).
- Worktree: `/home/fruiz/jax/.worktrees/codex-faro-f11-kernel-step6`.
- Incremento local: checkpoint externo hash-chained con genesis explícito, CAS por head
  anterior, lock `flock`, escritura atómica, `fsync` de archivo/directorios y relectura.
  `MariaDBRuleDecisionStore` puede usarlo: mantiene el lock externo antes del lock DB,
  coteja el checkpoint con el audit head bloqueado antes de mutar, publica tras commit y
  falla cerrado si el head está atrasado o la publicación no se confirma.
- Modelo `RulePermitDraft` + `RulePermit`: proyección canónica hash-bound, campos cerrados,
  objetos profundamente inmutables; el store solo sella un permiso asociado a una evaluación
  sellada del kernel. El adapter en memoria aplica idempotencia exacta y rechaza reintentos
  con otro permiso.
- Verificación local: las suites de checkpoint, providers, modelos, permisos y store en
  memoria pasan (`303 passed`, checkpoint 9); MariaDB colecciona 17 pruebas. `compileall` y
  `git diff --check` pasan.
- Las pruebas MariaDB no se ejecutaron: el fixture Docker no puede abrir
  `/var/run/docker.sock` por permiso denegado.
- GLM/ZCode revisa read-only el SHA `76c8ecd0` (checkpoint/almacenamiento externo); su
  veredicto final aún está pendiente. Observó que fallo post-commit/post-anclaje queda
  cerrado hasta reconciliación explícita y que adapters futuros deben tomar primero el
  mismo `flock`; falta runbook auditado de reconciliación antes de wiring productivo.
  Kimi CLI no está disponible por límite 403.
- Añadí regresión que forja una cadena hash-válida con un head histórico repetido. Pasa con
  el validador vigente; mutando temporalmente esa guardia, falla como se espera.
- Agregué inyección de fallo de `fsync` antes de `replace` (el head viejo permanece) y de
  `fsync` de directorio después de `replace` (error de publicación incierta; el nuevo archivo
  es visible). Las nueve pruebas de checkpoint y las cinco suites cercanas pasan: 303 total.
- El checkout compartido claude-skills está sincronizado; `claude-skills-sync pull`
  informó una divergencia local de `settings.json` de Claude, sin cambios en PENDIENTES.

## Límites de autoridad y alcance

- La decisión directa de Fernando sigue pendiente: si `OVERLAY_ISSUED` requiere una
  ratificación previa no revocada del mismo hash y cuarentena permanente para overlays
  históricos inválidos.
- El kernel de este paso debe ignorar overlays y ratificaciones del corpus; solo usa el
  grant individual de Block 4 derivado del ledger verificado.
- No tocar el worktree `/home/fruiz/wt/jax-ledger-checkpoint` ni `codex/faro-r2`.
- No publicar ni integrar hasta resolver la pregunta de OVERLAY, cerrar auditoría/CI y
  revalidar la ventana inmediatamente antes de publicar/integrar. El trabajo aislado y el
  rebase técnico pueden continuar sin decidir la semántica de OVERLAY.

## Siguiente acción exacta

En este worktree, continuar TDD desde la migración `policy/rule_authority/migrations/001_rule_authority_kernel.sql`:

1. Terminar emisión durable de decisión + permiso y validar migration/tests en MariaDB CI.
2. Resolver la frontera transaccional compartida con el ledger Block 4; luego implementar
   consumo atómico, carrera de doble consumo y kernel `evaluate`/`consume`, excluyendo overlays.
3. Ejecutar las suites disponibles, registrar límites del runner, actualizar el handoff,
   delta-auditar el SHA exacto y dejar el cambio listo para Fernando, sin integrar.

## Archivos tocados

- `policy/rule_authority/trusted_checkpoint.py`
- `policy/rule_authority/permit.py`
- `policy/rule_authority/__init__.py`
- `policy/rule_authority/storage.py`
- `policy/rule_authority/store.py`
- `policy/rule_authority/migrations/002_enable_atomic_permits.sql`
- `tests/policy/test_faro_rule_permit.py`
- `tests/policy/test_faro_rule_authority_store.py`
- `tests/policy/test_rule_authority_checkpoint.py`
- `tests/policy/test_faro_rule_authority_storage_mariadb.py`
- `docs/superpowers/plans/2026-10-09-faro-f11-step6-checkpoint.md`
- `TRASPASO.md`
