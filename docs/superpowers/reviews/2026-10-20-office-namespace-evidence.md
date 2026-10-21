# Namespace Office · evidencia de desarrollo

2026-10-20 · OP#343/345 · Codex. Entorno MariaDB12.3.3 desechable,127.0.0.1:33991, CI=true sin GITHUB_ACTIONS ni override de guardas productivas. No producción.

Sol namespace_review RECHAZÓ 28c1cb3e3f42c0527fd14071e40fe91896798e6c: LASMANOS aceptabaOffice, rollback ignoraba nuevaDDL, faltaban upgradepoblado/rollbackentreinserts y registro decontención. Implementador separado corrigió todos en fixround1; dictamen sobre corrección todavía pendiente.

Raíz corrió tras corrección:
`python3 /tmp/cv-office-run-tests.py jax -q --tb=short --disable-warnings tests/test_office_namespace.py tests/test_office_namespace_mariadb.py tests/test_legacy_adoption.py tests/test_project_scope_authority.py tests/test_b9_scope_authority.py tests/test_project_authority_input_validation.py tests/test_proyecto_activo_mariadb.py tests/test_project_authority_mariadb.py::test_migracion_apply_apply_revert_apply_y_falla_cerrado_con_fila_archivada`

Resultado:58passed,67warnings0.43s. WarningsDDLidempotente. Pruebas cubren guardasOffice/request-onlyantesDDL, reversiónvacía/reaplicar, backfillCLASSICconfilaprevia, triggerreal fallainsertScope sinhuérfano y LASMANOSOfficeinvisible conownershipcoincidente.

MEDICIÓN del implementador, comando reproducible:
`python3 /tmp/cv-office-run-tests.py jax -q -s --tb=short tests/test_office_namespace_mariadb.py::test_office_same_key_concurrency_creates_one_project_and_stable_replay`
Samekeyp95=0.002938s;differentkeyp95=0.004604s. Fuente task-1-report.md; raíz confirmó suiteconcurrentepasa pero no repitió esos valores exactos. Son tiempos locales efímeros, no cifra de carga del sistema de producción.

No cliente/APIOffice activado. Consumidor JXP pendiente de revisiónpropia. Autoría cambios:Tier2namespace_implement; coordinación/verificaciónCodex; auditoríaSolreadonly independiente.
