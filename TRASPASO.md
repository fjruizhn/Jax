# Traspaso Codex · PR #381

## Objetivo

Cerrar #381 sobre el `origin/master` vigente, con el writer sellado de ratificaciones de #379 intacto y con cada checkpoint histórico verificado contra su evento. La ventana de Fernando está abierta en Hall9000; revisar su vigencia antes de publicar o integrar.

## Estado confirmado

- SHA publicado al inicio: `4d493b988c2cb116c6236ca868a52aa2da90f413`; PR #381 está abierto y su base era la rama #379 ya integrada.
- `origin/master` observado: `8c850bcec5cd0233db5487343210c0ba96689f97`; #379 (`7cf27a09`) es ancestro de master.
- Auditor Tier 3 rechazó `4d493b98`: (1) preservar el writer dedicado de #379, y (2) validar cada fila de checkpoint contra `events[sequence-1]`.
- Se resolvió `git merge --no-commit --no-ff origin/master` en esta rama; no quedan conflictos. Se conservaron los cambios recientes de Faro de master y se integraron solo los deltas #381.
- `service.py` mantiene `append_ratification_from_candidate` como único writer ratificador, bajo el mismo lock/checkpoint durable. Los writers genéricos rechazan `RATIFICATION_GRANTED`.
- `replay.py` usa `checkpoint_store.checkpoints()` y valida cada fila positiva contra `events[sequence-1]` (event id/hash). La regresión seq0/seq2 falsificado/seq3 head reprodujo RED antes de la corrección y quedó GREEN.
- Piso `authority-ledger-mariadb/integration`: 9 (no se baja a 6). Pisos de loader seal e Identity de master también se conservan.
- Commit local de reapilado y cierres: `dd645ec7` (merge commit, contiene `Co-Authored-By: Codex`).
- Pruebas locales: focales 42 passed; ratification + checkpoint 37 passed; MariaDB **10 passed** al ejecutar con `sudo -n -E` por el acceso al socket Docker. El verificador del piso 10 terminó con rc 0; `git diff --cached --check` estuvo limpio.
- Kimi agotó su límite temporal; GLM está trabajando en #378 y no se le interrumpe.

## Falta

1. Confirmar que la rama remota sigue en `4d493b98`; publicar `dd645ec7` como fast-forward.
2. Cambiar la base de #381 a `master`; esperar CI completa verde del SHA exacto.
3. Pedir auditoría Tier 3 independiente del nuevo SHA; incorporar el veredicto al PR.
4. Ejecutar `verify-integration <SHA>` desde el checkout asociado a PR, inmediatamente antes del merge. Seguir `docs/runbooks/integracion-github-free.md` literalmente.
5. Tras integrar, correr enseguida `post-merge-guard`, registrar el resultado en Biblioteca y actualizar PENDIENTES. Borrar este handoff antes de la auditoría final; no borrar su registro histórico.

## Decisiones y siguiente comando

Fernando dijo: «si recomiendas hacerlo así hazlo». Decisión operativa: resolver el rechazo sobre la rama actual, preservando #379 y re-midiendo pisos; #378 queda con GLM. No integrar hasta tener auditoría Tier 3 y CI verde.

Al retomar: revisar la ventana con `/home/fruiz/claude-skills/bin/ventana estado`, luego `git status --short --branch && git diff --cached origin/master -- policy/authority_ledger/service.py policy/authority_ledger/replay.py tests/policy/test_authority_ledger_checkpoint.py ci/pisos.json` desde `/home/fruiz/wt/jax-ledger-checkpoint`.
