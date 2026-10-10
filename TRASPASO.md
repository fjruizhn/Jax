# TRASPASO — soporte del grant exacto de retiro JAX

- Objetivo: dejar que el registro de grants exactos de pisos opere bajo CI sin debilitar el rechazo de retiros no autorizados.
- Hecho: PR #390 (`7890a4fe76219425933598ad002920d99c7532a2`) creó el grant exacto para el rollback #387; auditoría Tier 3 APROBADO. Su CI falló `archivos-de-test-en-ci` porque el test de soporte exigía `grants=[]`. El registro del PR #390 produjo 232/233 en esa sub-suite.
- Hecho: esta rama `codex/faro-retirement-grant-schema`, basada en `master@0f9b79633f4b9ca6fdf89cfb8b1a124d5f05027d`, sustituye esa aserción por esquema versionado y campos exactos, sin añadir un test. Prueba observada RED con el grant real de #390; GREEN con el cambio. La suite exacta local quedó `233 passed`.
- Decisión: mantener validación profunda del comparador; la prueba superficial confirma versión, tipo de colección y conjunto cerrado de claves. Fernando autorizó el grant exacto de #390; la corrección de soporte no amplía el grant.
- Falta: push/PR del soporte; CI exacta; auditoría Tier 3 del SHA final; integrar por procedimiento y post-merge-guard. Luego regenerar CI de #390 contra la nueva base y auditarlo otra vez. Después actualizar #387 sobre el merge real del grant y completar CI/auditoría/guard.
- Producción: Block 4 sigue UNAVAILABLE por ausencia de raíz confiable; no sintetizarla ni desplegar hasta restaurar una fuente verificada del dueño de autoridad.
- Siguiente comando: `git status --short --branch && git diff --check && python3 -m pytest -q policy/tests/test_pisos_fuera_del_workflow.py policy/tests/test_pisos_migracion_desde_master.py policy/tests/test_comparar_pisos.py`.
