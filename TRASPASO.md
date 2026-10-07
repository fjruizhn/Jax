# Traspaso Faro F1.1 #371

## Estado (2026-10-07)

- Rama `feat/faro-f1.1-evaluador`, worktree `/home/fruiz/wt/jax-faro-f11-paso4`.
- Integrados localmente #370 r5 y #372; pendiente incluir los fixes de #369 en el árbol actualizado.
- `RuleEvaluationRequest` y `RuleLimits` requieren el catálogo cargado, validan unidades/subids y normalizan argumentos NFC antes del hash.
- `RuleDecisionStore.get(request)` devuelve `None` si el hash no coincide; constructor no acepta `_decisions`.
- Workflow Python 3.14 instala `pymysql`; el piso de modelos es 29.
- Suite focal schema/snapshot/ataques/modelos: 242 passed.

## Siguiente

1. Cambiar compatibilidad `MappingProxyType` por el `CatalogoTopes` sellado cuando esté el SHA final r6 de #370.
2. Incorporar el estado corregido #369 y completar pruebas/mutantes V4/V6/V8/V9 y regresiones.
3. Verificar pisos 19/12/1 y modelo, merge-tree contra #370/#372, publicar commits Codex y actualizar el cuerpo del PR #371.
