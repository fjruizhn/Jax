# Traspaso continuo — cerco de proyectos para herramientas del worker

Fecha: 2026-10-06  
Rama: `fix/cerco-proyectos-tool-authority`  
Base: `origin/master` (`f47820f5af36d7b0e4e0d5b2156ac6275c10462c`)  
Estado: implementación y pruebas focalizadas terminadas; falta auditoría adversarial satisfactoria, publicar PR y registrar SHA/URL.

## Alcance

`authorize_and_execute_tool_call` rechaza y audita herramientas de archivo (`file_read` / `file_write`) cuya ruta canónica resuelve dentro de `WORKSPACE_ROOT/proyectos`. Lectura y escritura recorren la ruta original desde el workspace con descriptores de directorio y `O_NOFOLLOW`; se niegan symlinks de componentes y la raíz léxica `proyectos/`. El worker y los movimientos de carpetas de E2a-LACTOVI comparten `flock` en `.git/project-tree.lock`, así un traslado legítimo a `proyectos/` no puede intercalarse entre validar y escribir/leer. `resolve_jailed_path` queda intacta para el endpoint de Procesamiento.

## Verificación

- TDD rojo en master: los nuevos casos ejecutaban `read_file`/`write_file` sobre rutas de proyecto y symlink; el resolver compartido seguía aceptándolas.
- Verde enfocado: 2 pruebas pytest, 8 subcasos.
- Rojo carrera: la sustitución de `safe/` por symlink a `proyectos/` permitió lectura antes del cambio.
- `las_manos/_tool_authority_test.py` + `las_manos/_procesamiento_routes_test.py`: 177 passed, 8 warnings, 8 subtests; cubre carrera entre `os.replace` y persistencia Git, symlink final y que el movimiento coordinado espere hasta después de la lectura/escritura.
- Suite `tests-puros`: 3578 passed, 3 skipped, 3 xfailed, 1 fallo ambiental ajeno: `test_arranque_real_no_colisiona_con_policy_de_la_raiz` no puede leer `/etc/jax/build/implementation-identity.json` con el usuario local. No se usó esta salida para fijar el piso CI.
- Piso CI actualizado +8: `tests-puros/out` 3537 → 3545; `las_manos/_tool_authority_test.py` 66 → 74 contra `origin/master`; los 45 skips del runner no cambian.

## Límites y pendientes

- No se consultó ni modificó MariaDB, política, producción ni `PENDIENTES.md`.
- No integrar. Crear PR independiente tras auditoría adversarial.
- Runbook `docs/constitucion/runbooks/traspaso-continuo.md` ausente en este checkout; se mantuvo este registro conforme al requisito general de continuidad.
