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
- Verificación local del incremento: `python3 -m pytest tests/policy/test_rule_authority_checkpoint.py tests/policy/test_rule_authority_providers.py tests/policy/test_faro_rule_authority_models.py -q`
  → 289 passed; `git diff --check` limpio.
- La suite MariaDB no se revalidó en este incremento; en la medición anterior el fixture
  Docker no pudo abrir `/var/run/docker.sock` por permiso denegado.
- El checkout compartido claude-skills está sincronizado; `claude-skills-sync pull`
  informó una divergencia local de `settings.json` de Claude, sin cambios en PENDIENTES.

## Límites de autoridad y alcance

- La decisión directa de Fernando sigue pendiente: si `OVERLAY_ISSUED` requiere una
  ratificación previa no revocada del mismo hash y cuarentena permanente para overlays
  históricos inválidos.
- El kernel de este paso debe ignorar overlays y ratificaciones del corpus; solo usa el
  grant individual de Block 4 derivado del ledger verificado.
- No tocar el worktree `/home/fruiz/wt/jax-ledger-checkpoint` ni `codex/faro-r2`.
- No publicar, integrar ni rebasear sobre otra base hasta acordarlo y revalidar la ventana
  inmediatamente antes de publicar/integrar.

## Siguiente acción exacta

En este worktree, continuar TDD desde la migración `policy/rule_authority/migrations/001_rule_authority_kernel.sql`:

1. Hacer que las decisiones `DENY`/`MISSING_RULE` comparen el audit head DB con el
   checkpoint externo bajo lock compartido y publiquen el nuevo head solo después del
   commit; inyectar fallos antes/después del commit y durante el `fsync`.
2. Añadir permiso + consumo atómicos, usando el mismo audit head y el lock del ledger
   Block 4 en una transacción MariaDB.
3. Añadir kernel `evaluate`/`consume`, carreras y migration incremental; excluir overlays.
4. Medir local/CI, completar `TRASPASO.md`, solicitar Tier 3 sobre SHA exacto y dejar
   listo para Fernando, sin integrar.

## Archivos tocados

- `policy/rule_authority/trusted_checkpoint.py`
- `tests/policy/test_rule_authority_checkpoint.py`
- `docs/superpowers/plans/2026-10-09-faro-f11-step6-checkpoint.md`
- `TRASPASO.md`
