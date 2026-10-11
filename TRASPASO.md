# Traspaso · Rule Authority checkpoint durability

- Rama: `fix/faro-checkpoint-restart-recovery`.
- Base auditada: `88000c65e9378809af159a9f51bb6120741ef112`.
- Auditoría exacta: APROBADO CON CAMBIOS, 0 BLOCK / 0 MAJOR / 1 MINOR. MINOR único: no había regresión versionada para detectar el cambio de inode si el archivo se reemplaza con bytes idénticos durante `confirmar()`.
- Cambio para cerrar MINOR: nuevo test inyecta fallo de directory fsync tras replace, sustituye el checkpoint por otro inode idéntico durante reconciliación, exige `CheckpointDurabilityUnknown` y conserva el marcador. `ci/pisos.json` y `docs/ci/pisos.md` suben authority-rule-unit de 58 a 59.
- Verificación actual: comando exacto unitario `59 passed`; `piso.py verificar authority-rule-unit/tests`, JSON y diff-check pasan. Integración MariaDB colecciona 47; el runtime local está bloqueado por `permission denied` de Docker.
- Próximo comando: volver a ejecutar el comando exacto de suite y `git diff --check`, medir el piso, compilar, y revisar el diff.
- Falta: commit con footer Codex; borrar este archivo archivando su contenido en `docs/historia/2026-10-10-faro-checkpoint-unknown-recovery.md`; solicitar delta-audit exacta del nuevo SHA. Si aprueba, comprobar ventana antes de publicar y esperar CI exacta. No integrar sin procedimiento reservado y guard.
