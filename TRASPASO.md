# Traspaso · Rule Authority checkpoint DURABILITY_UNKNOWN

- Fecha: 2026-10-10 (America/Tegucigalpa).
- Rama/worktree: `fix/faro-checkpoint-restart-recovery` en `/home/fruiz/jax/.worktrees/codex-faro-f11-kernel-step6`, base inicial `765406f4475ac5babb8ec6af2e6c92b91aa65344`.
- Objetivo: cerrar el BLOCK post-`os.replace` del checkpoint Rule Authority y demostrar retry/recovery tras fallo `fsync`, reinicio y crash con rename perdido, para PERMIT y consumo.
- Decisión de diseño: `CheckpointDurabilityUnknown` representa la falla en el checkpoint; el servicio expone `RuleAuthorityDurabilityUnknown` hasta `evaluate()`/`consume()`. No hay marcador persistido porque su escritura tendría la misma incertidumbre. Reconciliar significa comparar la cadena exacta observada, hacer `fsync` del archivo + cadena de directorios, releer y exigir identidad/hash-chain exactos.
- Auditoría previa: Tier 3 del SHA `08e3f9d6aa2ca483b187cd27e5387854fbea23e0` = APROBADO CON CAMBIOS, 0 BLOCK / 1 MAJOR / 1 MINOR / 0 LOW. Delta local corrige ambos hallazgos: `RuleAuthorityDurabilityUnknown` llega al caller de evaluación/consumo y se cubre checkpoint anterior tras crash de consumo. El nuevo SHA requiere auditoría exacta.
- Pruebas verificadas tras esos fixes: suite unitaria Rule Authority exacta, 58 passed; MariaDB 47 collected; `compileall`, JSON y `git diff --check` limpios.
- Prueba end-to-end MariaDB: cubre retry exacto y otro `consumption_request_id`, tipo público, retención de filas/idempotency key, reconciliación después del restart y ambos outcomes del crash para evaluación y consumo. La ejecución local no arrancó por permiso denegado en `/var/run/docker.sock`; debe pasar en CI exacta.
- Pisos: MariaDB Block 4 colecciona ahora 47 (antes 45); `ci/pisos.json`, `docs/ci/pisos.md` y comentario workflow en 47. Confirmar el run real de CI, no inferir pasadas de collection.
- Biblioteca actualizada en `docs/historia/2026-10-10-faro-checkpoint-unknown-recovery.md`; conserva el veredicto anterior y los cambios actuales.
- Próximo paso: revisar el diff, comprometer con `Co-Authored-By: Codex <noreply@openai.com>`, retirar TRASPASO archivando hechos antes de auditoría, y solicitar delta Tier 3 del nuevo SHA. Antes del push ejecutar `/home/fruiz/claude-skills/bin/ventana estado`.
