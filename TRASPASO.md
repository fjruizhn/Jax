# Traspaso continuo — índices para auditoría global de descartes

- Objetivo: añadir a `jacobs_events` los índices que permiten paginar el feed global por fecha y filtrar por pipeline sin ordenar filas temporalmente. Esto habilita el PR de pantalla admin en `jax-platform`.
- Hecho: worktree `feat/auditoria-eventos-descartes` creado desde `origin/master` en `68557946`; esquema y escritor revisados en `jacobs/store.py` y `jacobs/routes.py`; pruebas nuevas fallaron con el código viejo (índices ausentes); ambos índices aditivos ya están declarados en `_INDICES`.
- Falta: prueba verde, EXPLAIN sin filesort/temporary y medición de piso CI si cambia; commit, push y PR de `jax` antes del PR de plataforma.
- Decisión: índice global `(event_type, ts, id)` para cuatro consultas por tipo y merge acotado; índice de pipeline `(pipeline_id, event_type, ts, id)` para consultas filtradas por pipeline. Decisión técnica de Codex basada en los filtros y orden del encargo.
- Siguiente comando exacto: ejecutar las dos pruebas con `CI=1 GITHUB_ACTIONS=true JAX_DB_HOST=127.0.0.1 JAX_DB_PORT=33316 JAX_DB_USER=jax_test JAX_DB_PASSWORD=codex-test-db-only PYTHONPATH=las_manos python3 -m pytest -q tests/test_jacobs_events_indice_auditoria_global.py` desde `/home/fruiz/wt/jax-auditoria-descarte` (MariaDB desechable `codex-jxp-auditoria-db-20261006`; no se conecta a 3308).
