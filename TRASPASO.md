# Traspaso · Rule Authority checkpoint DURABILITY_UNKNOWN

- Fecha: 2026-10-10 (America/Tegucigalpa).
- Objetivo: cerrar el BLOCK post-`os.replace` del checkpoint Rule Authority y probar fallo `fsync`, retry/recovery y reinicio/crash simulado para evaluación PERMIT y consumo.
- Rama/worktree: `fix/faro-checkpoint-restart-recovery` en `/home/fruiz/jax/.worktrees/codex-faro-f11-kernel-step6`, base inicial `765406f4475ac5babb8ec6af2e6c92b91aa65344`.
- Decisión técnica: `DURABILITY_UNKNOWN` es un error tipado transitorio, no un marcador persistido. Escribir el marcador podría fallar con la misma incertidumbre. El reintento compara la cadena íntegra observada, hace `fsync` del archivo y de la cadena de directorios, vuelve a leer y exige identidad exacta antes de recuperar resultado alguno.
- Hecho verificado: la prueba unitaria de reinicio falló primero porque el código emitía solo `CheckpointInvalido`; tras el cambio a `CheckpointDurabilityUnknown`, la suite exacta del workflow (`test_faro_rule_authority_store.py`, permit, kernel y checkpoint) pasa (`58 passed`). `compileall`, JSON y `git diff --check` pasan.
- Pendiente de verificación: la regresión end-to-end MariaDB cubre evaluación PERMIT, consumo, retry exacto/ID distinto, recreación store/kernel y checkpoint anterior restaurado tras crash simulado. No se pudo iniciar localmente por `permission denied` al socket Docker `/var/run/docker.sock`; la CI exacta debe ejecutar los 46 casos.
- Pisos recolectados: `authority-ledger-mariadb/integration` ahora colecciona 46 (antes 45); piso/docs/comentario de workflow actualizados a 46. La colección Rule Authority MariaDB es 28 y sin cambio.
- Archivos modificados: `errors.py`, `trusted_checkpoint.py`, pruebas de checkpoint y de integración MariaDB, `ci/pisos.json`, `docs/ci/pisos.md`, comentario de conteo en `.github/workflows/policy.yml`.
- Siguiente acción exacta: hacer revisión adversarial de este diff y, si no encuentra BLOCK/MAJOR, preparar el commit con coautor Codex; después correr `/home/fruiz/claude-skills/bin/ventana estado` inmediatamente antes de publicar para obtener CI exacta.
