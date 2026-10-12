# Traspaso B4 producción

Fecha: 2026-10-12

## Alcance propio

- `policy/authority_ledger/provisioning.py`: cuentas B4 reader/writer con host
  TCP exacto y privilegios mínimos.
- `jaxctl/runtime.py`: fábrica dedicada `JAX_AUTHORITY_DB_*` para todo uso de
  `MariaDBAuthorityLedgerStore`; nunca hace fallback a `JAX_DB_*`.
- `policy/authority_ledger/bootstrap.py`: inspección conservadora, staging
  público y límite de primer-init que sólo llama los límites existentes de
  initialize/verify; no firma ni aprueba material.
- `docs/runbooks/authority-first-bootstrap.md`: procedimiento operador.

## Evidencia local

`python3 -m pytest tests/policy/test_authority_ledger_bootstrap_helpers.py
tests/policy/test_authority_ledger_runtime_composition.py
tests/policy/test_block8_adapters_docs.py
policy/tests/test_archivos_de_test_wireados_en_ci.py -q` → 35 passed.

`python3 -m pytest tests/policy/test_authority_ledger_events.py
tests/policy/test_authority_ledger_checkpoint.py
tests/policy/test_authority_ledger_authentication.py -q` → 34 passed.

La corrida aislada real de MariaDB (`mariadb:12.3.3`, `--network none`, socket
temporal) previa a los últimos endurecimientos completó 49 passed en 122.57 s:
`/tmp/authority-ledger-mariadb-b4`. La selección exacta identity previa
completó 1064 passed en 6.12 s: `/tmp/identity-foundation-b4-final2`.

## Límites

No se aplicaron GRANTs, migraciones, servicios, archivos en `/etc` o `/var`,
claves ni material de raíz. Sigue pendiente la fuente externa verificada de la
clave pública de Fernando y la ejecución revisada del runbook en el entorno
operativo.
