# TRASPASO — transición exacta del piso en grant de retiro

**Objetivo:** permitir que el grant autorice el retiro de una definición de piso que fue reemplazada exactamente por otra definición en un merge de dos padres, manteniendo todos los bindings y rechazos cerrados existentes.

## Hecho

- Confirmé en `CONTEXT.md`, `AGENTS.md`, el runbook `traspaso-continuo.md` y el comparador actual que la falla de #390 nace de asumir que la clave del piso debía estar ausente en el primer padre.
- Tier 3 ya recomendó validar la transición exacta: definición del padre 1 distinta del objetivo; padre 2 igual al objetivo; merge igual al objetivo; el merge y sus padres exactos siguen ligados por SHA.
- Añadí pruebas primero: creación y reemplazo exacto pasan; padre1 ya igual al objetivo, propuesta distinta o resultado aterrizado distinto se rechazan.
- Verificación local actual: `pytest -q policy/tests/test_comparar_pisos.py` → 88 passed; suite exacta del workflow → 238 passed en Python 3.14.4; `git diff --check` limpio. CI en Python 3.12 aún debe confirmar el nuevo piso.
- Kimi hizo revisión adversarial de solo lectura: sin blockers; marcó como LOW una ambigüedad sobre si cambiar solo `output_file` cuenta como definición distinta. Tier 3 confirmó que `floor_definition` es `{entry, output_file}` y que este cambio afecta la evidencia inspeccionada; aceptar esa transición completa es coherente y no requiere cambio. Caso real #392: patrón 12→20, mismo output_file.
- ZCode/GLM no produjo una revisión: su CLI omitió la corrida por `Built-in skipped (not-due)` en ambos intentos. Esto no se contará como aprobación.
- PR #393 tiene varios checks en verde; el bridge rechazó el intento de preflight porque `tests-puros` seguía activo. La corrida 38053864753 está siendo vigilada.

## Decisiones

- Autoridad arquitectónica: Tier 3 confirma que la diferencia se compara sobre la definición completa `{entry, output_file}`. El usuario dio GO general para completar el Faro y delegar decisiones técnicas.
- El grant sigue ligado a merge+padres exactos, SHA de definición, SHA de base, evento del PR, un solo retiro y hash completo del diff.
- No tocar #388 ni sus worktrees. La raíz confiable de producción no se fabrica.

## Falta

1. Auditoría adversarial final sobre el SHA exacto; confirmar el piso exacto en CI Python 3.12.
2. Auditoría Tier 3 del SHA final y PR separado para el validador.
3. Integrar #393 y luego el soporte de transición con `verify-integration` y `post-merge-guard`.
4. Recalcular y publicar el grant dedicado #390 como único archivo del diff; luego reapilar, auditar e integrar #387.
5. Coordinar con el owner de #388, sin tomar su rama.
6. Restaurar y verificar la fuente autorizada de `/etc/jax/authority/trusted-root.json` antes de declarar producción.

## Siguiente comando

`gh run watch 38053864753 --repo fjruizhn/Jax --exit-status`

## Cierre del traspaso — 2026-10-10

- La auditoría Tier 3 del SHA `5ac7a74cc3072531505d8bdcff359df0dc39cebe` fue `RECHAZADO (BLOCK 1)` por conservar `TRASPASO.md` en el árbol. No identificó fallos de autoridad/código. Este archivo archiva ese traspaso y `TRASPASO.md` se quita antes de la auditoría final.
- Kimi revisó en solo lectura; sin blockers. Su observación LOW sobre cambio aislado de `output_file` fue resuelta por Tier 3: `output_file` es parte de la identidad completa de `floor_definition`.
- La CLI ZCode/GLM no produjo veredicto; emitió `Built-in skipped (not-due)` y se canceló sin modificar archivos.
- Corrida CI #393 `38053864753` seguía con `tests-puros` en curso al último chequeo; el preflight oficial rechazó #393 hasta que termine verde.
- El CI Python 3.12 del cambio de transición, el PR de soporte y el nuevo preflight/guard aún están pendientes.
- El grant local #390 debe apuntar a la introducción de #392: merge `8d1d1c49c5795fd6416085084265006fbcb6d69b`, parent1 `0d6484f2f56059e9fb0d11b04de8990375ee7c53`, parent2 `597b9a75d32874a84613e1ed7a0de40ea9e620ab`; recalcular `expected_diff_sha256` tras el rebase final.
