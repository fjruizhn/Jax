# Traspaso continuo — cerco de proyectos para herramientas del worker

Fecha: 2026-10-06  
Rama: `fix/cerco-proyectos-tool-authority`  
Base: `origin/master` (`f47820f5af36d7b0e4e0d5b2156ac6275c10462c`)  
Estado: auditorías APROBADAS en cc455e9e por rutas y coordinación. El CI de PR #364 descubrió que fixtures E2A temporales no tenían .git, un test de permisos invocaba una API interna retirada y dos escáneres estáticos exigían motivos fail-soft y no usar subprocess en el archivo de pruebas. Ajusté esos fixtures y el piso del job subpipeline-contrato-db (+1). Los cambios aún requieren auditoría del SHA nuevo y CI completo.

## Alcance

`authorize_and_execute_tool_call` rechaza y audita herramientas de archivo (`file_read` / `file_write`) cuya ruta canónica resuelve dentro de `WORKSPACE_ROOT/proyectos`. Lectura y escritura recorren la ruta original desde el workspace con descriptores de directorio y `O_NOFOLLOW`; se niegan symlinks de componentes y la raíz léxica `proyectos/`. El worker y los movimientos de carpetas de E2a-LACTOVI comparten `flock` en el inode del git-dir común, accesible por ambas cuentas, así un traslado legítimo a `proyectos/` no puede intercalarse entre validar y escribir/leer. `resolve_jailed_path` queda intacta para el endpoint de Procesamiento.

## Verificación

- TDD rojo en master: los nuevos casos ejecutaban `read_file`/`write_file` sobre rutas de proyecto y symlink; el resolver compartido seguía aceptándolas.
- Verde enfocado: 2 pruebas pytest, 8 subcasos.
- Rojo carrera: la sustitución de `safe/` por symlink a `proyectos/` permitió lectura antes del cambio.
- `las_manos/_tool_authority_test.py`, `las_manos/_procesamiento_routes_test.py` y `tests/test_proyectos_e2a_lactovi_mariadb.py` (sin DB): 180 passed, 13 skipped, 8 warnings, 8 subtests; incluye linked worktrees compartiendo el inode del lock y E2A tomando el lock del workspace de datos aunque el script viva en otro checkout. Además, una comprobación entre usuarios confirmó que jaxsvc no puede tomar el flock exclusivo mientras fruiz lo sostiene. `py_compile` y `git diff --check` pasan.
- Suite `tests-puros`: 3578 passed, 3 skipped, 3 xfailed, 1 fallo ambiental ajeno: `test_arranque_real_no_colisiona_con_policy_de_la_raiz` no puede leer `/etc/jax/build/implementation-identity.json` con el usuario local. No se usó esta salida para fijar el piso CI.
- Piso CI esperado tras medir los dos tests nuevos: `tests-puros/out` 3545 → 3547; `las_manos/_tool_authority_test.py` 74 → 75; los 45 skips del runner no cambian. CI deberá confirmar el piso.
- El job subpipeline-contrato-db/out incluye el nuevo test E2A: piso 295 → 296. CI deberá confirmar el conteo.

## Límites y pendientes

- No se consultó ni modificó MariaDB, política, producción ni `PENDIENTES.md`.
- PR #364 abierto contra master; no integrar. Cada commit posterior requiere auditoría del SHA exacto.
- Runbook `docs/constitucion/runbooks/traspaso-continuo.md` ausente en este checkout; se mantuvo este registro conforme al requisito general de continuidad.
