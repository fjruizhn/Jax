# Traspaso Faro F1.1 #371

## Estado (2026-10-07)

- Rama `feat/faro-f1.1-evaluador`, worktree `/home/fruiz/wt/jax-faro-f11-paso4`.
- Integración #370 r5 está resuelta localmente; #372 todavía pendiente.
- `RuleEvaluationRequest` y `RuleLimits` requieren el catálogo cargado, validan límites/subids y calculan hash tras normalizar argumentos NFC.
- `RuleDecisionStore.get` recibe el request y coteja `request_hash`; constructor no acepta `_decisions`.
- Workflow Python 3.14 instala `pymysql`; paso de modelos mide 29.
- Corrió el test focal de modelos (29 passed). Los tests combinados de #370 aún dependen de que el merge de #370 esté comprometido porque verifican objetos con `git show HEAD:`.

## Siguiente

1. Terminar y registrar el merge de #370; integrar #372 resolviendo `__init__.py` y workflow.
2. Reemplazar `MappingProxyType` de compatibilidad por el `CatalogoTopes` sellado de r6 cuando Hyde confirme el SHA final de #370; reapilar luego sobre ese SHA.
3. Cerrar mutantes V4, V6, V8, V9 y regresiones; correr pisos 19/12/1 y MaríaDB según encargo.
4. Actualizar base y cuerpo de PR #371, verificar merge-tree contra #370/#372, publicar commits Codex y actualizar este archivo antes de cierre.
