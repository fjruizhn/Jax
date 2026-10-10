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
  objetos profundamente inmutables; solo el helper interno del store sella un permiso
  confiable. Ocho pruebas unitarias pasan.
- Verificación local del incremento: `python3 -m pytest tests/policy/test_rule_authority_checkpoint.py tests/policy/test_rule_authority_providers.py tests/policy/test_faro_rule_authority_models.py -q`
  → 289 passed; `python3 -m pytest tests/policy/test_faro_rule_permit.py -q` → 8
  passed; `git diff --check` limpio.
- La suite MariaDB colecciona 14 pruebas, incluidas 2 nuevas para el checkpoint, pero no
  se ejecutó: el fixture Docker no puede abrir `/var/run/docker.sock` por permiso denegado.
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

1. Ejecutar en CI las pruebas MariaDB nuevas de mismatch, idempotencia y publicación
   incierta; luego inyectar fallos adicionales alrededor del `fsync`.
2. Añadir permiso + consumo atómicos, usando el mismo audit head y el lock del ledger
   Block 4 en una transacción MariaDB.
3. Añadir kernel `evaluate`/`consume`, carreras y migration incremental; excluir overlays.
4. Medir local/CI, completar `TRASPASO.md`, solicitar Tier 3 sobre SHA exacto y dejar
   listo para Fernando, sin integrar.

## Archivos tocados

- `policy/rule_authority/trusted_checkpoint.py`
- `policy/rule_authority/permit.py`
- `policy/rule_authority/__init__.py`
- `policy/rule_authority/storage.py`
- `tests/policy/test_faro_rule_permit.py`
- `tests/policy/test_rule_authority_checkpoint.py`
- `tests/policy/test_faro_rule_authority_storage_mariadb.py`
- `docs/superpowers/plans/2026-10-09-faro-f11-step6-checkpoint.md`
- `TRASPASO.md`
