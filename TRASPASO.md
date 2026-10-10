# TRASPASO — transición exacta del piso en grant de retiro

**Objetivo:** permitir que el grant autorice el retiro de una definición de piso que fue reemplazada exactamente por otra definición en un merge de dos padres, manteniendo todos los bindings y rechazos cerrados existentes.

## Hecho

- Confirmé en `CONTEXT.md`, `AGENTS.md`, el runbook `traspaso-continuo.md` y el comparador actual que la falla de #390 nace de asumir que la clave del piso debía estar ausente en el primer padre.
- Tier 3 ya recomendó validar la transición exacta: definición del padre 1 distinta del objetivo; padre 2 igual al objetivo; merge igual al objetivo; el merge y sus padres exactos siguen ligados por SHA.
- Añadí pruebas primero: creación y reemplazo exacto pasan; padre1 ya igual al objetivo, propuesta distinta o resultado aterrizado distinto se rechazan.
- Verificación local actual: `pytest -q policy/tests/test_comparar_pisos.py` → 88 passed; suite exacta del workflow → 238 passed en Python 3.14.4; `git diff --check` limpio. CI en Python 3.12 aún debe confirmar el nuevo piso.
- PR #393 tiene varios checks en verde; el bridge rechazó el intento de preflight porque `tests-puros` seguía activo. La corrida 38053864753 está siendo vigilada.

## Decisiones

- Autoridad arquitectónica: veredicto Tier 3 existente para la comparación de las tres definiciones. El usuario dio GO general para completar el Faro y delegar decisiones técnicas.
- El grant sigue ligado a merge+padres exactos, SHA de definición, SHA de base, evento del PR, un solo retiro y hash completo del diff.
- No tocar #388 ni sus worktrees. La raíz confiable de producción no se fabrica.

## Falta

1. Completar la suite y actualizar el piso exacto CI y su documentación.
2. Auditoría Tier 3 del SHA final y PR separado para el validador.
3. Integrar #393 y luego el soporte de transición con `verify-integration` y `post-merge-guard`.
4. Recalcular y publicar el grant dedicado #390 como único archivo del diff; luego reapilar, auditar e integrar #387.
5. Coordinar con el owner de #388, sin tomar su rama.
6. Restaurar y verificar la fuente autorizada de `/etc/jax/authority/trusted-root.json` antes de declarar producción.

## Siguiente comando

`gh run watch 38053864753 --repo fjruizhn/Jax --exit-status`
