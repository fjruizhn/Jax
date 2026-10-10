# TRASPASO — soporte para grant exacto de retiro JAX

**Fecha:** 2026-10-10 (America/Tegucigalpa)  
**Rama/worktree:** `codex/faro-retirement-grant-schema` / `/home/fruiz/worktrees/jax-faro-retirement-grant-schema`  
**Base:** `master@0f9b79633f4b9ca6fdf89cfb8b1a124d5f05027d`  
**Último commit:** `03a33bd9`  

## Alcance

El soporte genérico de #389 está integrado. Grant exacto #390 (`7890a4fe76219425933598ad002920d99c7532a2`) fue aprobado por auditoría Tier 3 y ya está publicado, pero su CI reveló una contradicción: `test_registro_de_retiros_vigente_inicia_vacio_y_con_esquema_versionado` exige que `grants` esté siempre vacío. Con el grant real, `archivos-de-test-en-ci` dio `232 passed, 1 failed`; el piso exterior es 233.

Esta rama sustituye esa aserción por una que valida versión 1, lista de grants y conjunto cerrado de claves por grant. La validación profunda de valores, identidad, hash y topología continúa en el comparador base. Se corrigió el texto de `docs/ci/pisos.md`; no se añade ni retira una prueba, por lo que el piso sigue en 233.

## Verificación

- RED observado: ejecutar el test original con el JSON real de #390 falló por la aserción `grants=[]`.
- GREEN observado: el test actualizado pasó con ese mismo JSON real.
- Suite exacta del job: `python3 -m pytest -q policy/tests/test_pisos_fuera_del_workflow.py policy/tests/test_pisos_migracion_desde_master.py policy/tests/test_comparar_pisos.py` → `233 passed`.
- `git diff --check` pasó.

## Pendiente

1. Abrir PR de soporte #391; esperar CI verde.
2. Actualizar la rama PR #391 para retirar este `TRASPASO.md` y dejar su registro en `docs/ci/pisos.md` antes de la auditoría exacta.
3. Tier 3 audita el SHA final; integrar bajo el procedimiento normal y ejecutar `post-merge-guard`.
4. Reintentar #390 sobre la nueva base real, obtener CI exacta verde y nueva auditoría; después actualizar #387 al merge real del grant y completar sus controles.
5. No tocar #388, propiedad de su sesión; ya se le informó del cambio de soporte.
6. Producción continúa bloqueada: Block 4 no tiene raíz confiable disponible. No inventar ni recrear esa raíz.

**Siguiente paso:** `gh pr create --base master --head codex/faro-retirement-grant-schema ...`
