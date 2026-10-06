# Traspaso continuo — cerco de proyectos para herramientas del worker

Fecha: 2026-10-06  
Rama: `fix/cerco-proyectos-tool-authority`  
Base: `origin/master` (`f47820f5af36d7b0e4e0d5b2156ac6275c10462c`)  
Estado: implementación y pruebas focalizadas terminadas; falta auditoría adversarial satisfactoria, publicar PR y registrar SHA/URL.

## Alcance

`authorize_and_execute_tool_call` rechaza y audita herramientas de archivo (`file_read` / `file_write`) cuya ruta canónica resuelve dentro de `WORKSPACE_ROOT/proyectos`. Lectura y escritura recorren la ruta original desde el workspace con descriptores de directorio y `O_NOFOLLOW`; se niegan symlinks de componentes y la raíz léxica `proyectos/`, sin volver a resolver el nombre protegido después de autorizar. `resolve_jailed_path` queda intacta para el endpoint de Procesamiento.

## Verificación

- TDD rojo en master: los nuevos casos ejecutaban `read_file`/`write_file` sobre rutas de proyecto y symlink; el resolver compartido seguía aceptándolas.
- Verde enfocado: 2 pruebas pytest, 8 subcasos.
- Rojo carrera: la sustitución de `safe/` por symlink a `proyectos/` permitió lectura antes del cambio.
- `las_manos/_tool_authority_test.py` + `las_manos/_procesamiento_routes_test.py`: 175 passed, 8 subtests; cubre lectura/escritura bajo carrera y demuestra que Git confirma solo los bytes autorizados aunque el pathname pase a apuntar a `proyectos/` después de `os.replace`.
- Suite `tests-puros`: 3578 passed, 3 skipped, 3 xfailed, 1 fallo ambiental ajeno: `test_arranque_real_no_colisiona_con_policy_de_la_raiz` no puede leer `/etc/jax/build/implementation-identity.json` con el usuario local. No se usó esta salida para fijar el piso CI.
- Piso CI actualizado +6: `tests-puros/out` 3537 → 3543; `las_manos/_tool_authority_test.py` 66 → 72 contra `origin/master`; los 45 skips del runner no cambian.

## Límites y pendientes

- No se consultó ni modificó MariaDB, política, producción ni `PENDIENTES.md`.
- No integrar. Crear PR independiente tras auditoría adversarial.
- Runbook `docs/constitucion/runbooks/traspaso-continuo.md` ausente en este checkout; se mantuvo este registro conforme al requisito general de continuidad.
