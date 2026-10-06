# Traspaso continuo — cerco de proyectos para herramientas del worker

Fecha: 2026-10-06  
Rama: `fix/cerco-proyectos-tool-authority`  
Base: `origin/master` (`f47820f5af36d7b0e4e0d5b2156ac6275c10462c`)  
Estado: implementación y pruebas terminadas; falta auditoría adversarial, publicar PR y registrar el SHA/URL.

## Alcance

`authorize_and_execute_tool_call` rechaza y audita herramientas de archivo (`file_read` / `file_write`) cuya ruta canónica resuelve dentro de `WORKSPACE_ROOT/proyectos`. `resolve_jailed_path` queda intacta para el endpoint de Procesamiento.

## Verificación

- TDD rojo en master: los nuevos casos ejecutaban `read_file`/`write_file` sobre rutas de proyecto y symlink; el resolver compartido seguía aceptándolas.
- Verde enfocado: 2 pruebas pytest, 8 subcasos.
- `las_manos/_tool_authority_test.py`: 68 passed; `origin/master`: 66 passed.
- `las_manos/_procesamiento_routes_test.py`: 103 passed.
- Suite `tests-puros`: 3578 passed, 3 skipped, 3 xfailed, 1 fallo ambiental ajeno: `test_arranque_real_no_colisiona_con_policy_de_la_raiz` no puede leer `/etc/jax/build/implementation-identity.json` con el usuario local. No se usó esta salida para fijar el piso CI.
- Piso CI actualizado +2: `tests-puros/out` 3537 → 3539, medido por diferencia del archivo de prueba contra `origin/master`; los 45 skips del runner no cambian.

## Límites y pendientes

- No se consultó ni modificó MariaDB, política, producción ni `PENDIENTES.md`.
- No integrar. Crear PR independiente tras auditoría adversarial.
- Runbook `docs/constitucion/runbooks/traspaso-continuo.md` ausente en este checkout; se mantuvo este registro conforme al requisito general de continuidad.
