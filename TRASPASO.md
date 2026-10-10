# Faro #387 rebase continuo

- Objetivo: reapilar el revert de #385 sobre el master oficial actual, validar el retiro exacto autorizado por #390, auditar el SHA final e integrar #387 solo después de preflight y CI verdes.
- Estado verificado: `origin/master` es `250b82c82deaac8811936ebbd26872d9d7b3e26b`; #390 está MERGED y su post-merge-guard pasó. #387 sigue OPEN en `b5f70fef88a6f7c2a19a9cfca4b7dd2716a04a51`, base vieja `9b2744c33c37fdfeec87bac03ddd00c9b76e8fb5`; `pisos-no-bajan` es el único check fallido observado.
- Rama propia: `codex/revert-faro385`, worktree `/home/fruiz/jax/.worktrees/codex-faro387-rebase-20261010`, iniciado desde `origin/master`.
- Decisión: conservar el rollback de #385 y quitar exactamente su piso F1.1 solo mediante el grant registrado en master; no cambiar otros pisos. No tocar #388/#375 ni worktrees de sus owners.
- Progreso: el revert original aplicado sobre master produjo conflictos solo en el piso, su documentación y el test de MariaDB; se resolvieron quitando el piso F1.1 retirado y eliminando su test junto con la implementación, conservando los cambios de master posteriores a #385.
- Siguiente: confirmar el diff y `git diff --check`, correr `.github/ci/comparar_pisos.py` contra el grant #390 y verificar que el hash exacto esperado coincide; después auditoría Tier 3, CI, `verify-integration`, merge exacto y `post-merge-guard`.
