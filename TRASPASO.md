# Traspaso — Jax Faro 0.4 memoria.buscar

- **Objetivo:** implementar `memoria.buscar` como lectura por el Puerto, con Guardia y bitácora; scope de tenant/usuario fijado por la identidad del socket; resultados marcados `untrusted_source`.
- **Hecho:** rama `feat/faro-0.4-memoria-buscar` actualizada a `origin/master` `2925fa8c587f7effc0c5bbbfa263e9da75f75e25`; leído el plan 0 y el runbook central de traspaso. Decisión de Fernando recibida por coordinación: B9 no está en producción; el perfil de servicio de memoria queda apagado por defecto y solo admite el test DB local.
- **Pruebas:** `tests/test_faro_memoria.py` escrito primero. RED confirmado: colección falla porque todavía no existen `ConfigMemoria` ni `jax.faro.herramientas.memoria`.
- **Falta:** implementar el adaptador y la compuerta de configuración; probar scope, salida no confiable, freno/bitácora, error cerrado, consulta SQL solo lectura contra DB de prueba `jax_memory_test` en `127.0.0.1:3308` con `jax_test`; cablear el archivo a CI y remedir `ci/pisos.json`; documentar límite de B9 en spec/PR; ejecutar suite y mutaciones relevantes; borrar este archivo y registrar el cierre en `docs/historia/` antes de la auditoría final.
- **Siguiente comando:** `python3 -m pytest tests/test_faro_memoria.py -q` después de implementar `ConfigMemoria` y el adaptador.
