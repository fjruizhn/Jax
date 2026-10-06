# Traspaso continuo — pipeline sin despacho motor legacy

Fecha: 2026-10-06  
Rama: `fix/pipeline-motor-no-disponible`  
Base: `origin/master` (`f47820f5af36d7b0e4e0d5b2156ac6275c10462c`)  
Estado: cambios y pruebas focalizadas listos; solicitar auditoría adversarial del SHA final antes de publicar el PR.

## Alcance

El predicado `faceta_ejecutable_en_pipeline` excluye `MOTOR_FACETS` y Hyde de planes/menús y de validación pre-vuelo. `/plan`, `/preflight` y `/continue` responden 422 `motor_gobernado_no_disponible`; devolución automática registra el rechazo y no sondea ni reencola. `_invoke_motor` lanza `GovernedExecutionRequiredError` antes de obtener cliente HTTP. El endpoint legacy conserva 410 y evidencia B7 y se eliminó su cuerpo inalcanzable.

## Verificación

- Rojo en master antes de la implementación para menú, plan/pre-vuelo y despacho HTTP directo.
- `tests/test_plan_facetas_de_la_tabla.py`, `tests/test_invoke_motor_rechazo.py`, `tests/test_jacobs_continuar.py`, `jacobs/_devolucion_test.py`: 75 passed.
- Contratos de despacho directo retirado en `tests/test_motor_contexto_y_salida.py`, `tests/test_invoke_motor_cancel_on_timeout.py` y `tests/test_cliente_http_compartido.py`: 21 passed.
- Tests con DB no ejecutables en este host: no están definidos `JAX_DB_HOST/JAX_DB_PORT`; no se cargaron credenciales ni se conectó a `jax_memory` de producción.
- Piso objetivo `tests-puros/out`: +3 tests netos, 3537 → 3540; 45 skips conservados. Requiere confirmación del runner CI.

## Límites y pendientes

- No se consultó ni modificó MariaDB de producción, `policy/**` ni `PENDIENTES.md`.
- No integrar. Publicar en PR independiente después de auditoría adversarial satisfactoria.
- Runbook `docs/constitucion/runbooks/traspaso-continuo.md` ausente en este checkout.
