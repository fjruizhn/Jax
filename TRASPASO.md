# Traspaso Faro F1.1 #368/#369/#371

## Estado (2026-10-07)

- Rama #371 `feat/faro-f1.1-evaluador`, worktree `/home/fruiz/wt/jax-faro-f11-paso4`.
- #370 r5 y #372 providers están integrados; #369 corregido está integrado en este commit de merge.
- #368/#369: K3, K8, D2, D3, E4 y E10 fueron verificados en sus ramas propias. Pisos #369: codec 19, MariaDB 1, ratificaciones 12.
- #371: límites requieren catálogo cargado, topes `2**53`, unidades limitadas al catálogo, NFC antes del hash y store que coteja hash en `get(request)`; constructor no permite `_decisions`.
- Python 3.14 instala `pymysql`; piso de modelos 29 y piso combinado Identity provisional 577.
- En #371 pasaron modelos + schema/snapshot/ataques: 242 tests antes del merge de #369.

## Pendiente

1. Cambiar temporal `MappingProxyType` a `CatalogoTopes` sellado del SHA final r6 de #370 y reapilar sobre dicho SHA.
2. Correr la suite completa de ledger y mutantes V4/V6/V8/V9/regresiones sobre la pila final; medir Identity exacto y corregir su piso.
3. Correr `piso.py verificar` para 19/12/1/29 y actualizar PR #371, merge-tree y reporte de entrega.
