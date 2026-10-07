# Traspaso Faro F1.1 #368/#369/#371

## Estado (2026-10-07)

- Rama #371 `feat/faro-f1.1-evaluador`, worktree `/home/fruiz/wt/jax-faro-f11-paso4`.
- #370 r5 y #372 providers están integrados; #369 corregido y el último commit CI de #368 están integrados en este árbol.
- #368/#369: K3, K8, D2, D3, E4 y E10 verificados en sus ramas. Pisos ledger: codec 19, MariaDB 1, ratificaciones 12.
- #371: límites requieren catálogo cargado, topes `2**53`, NFC antes del hash, unidades limitadas al catálogo, `get(request)` liga el hash y el store no acepta `_decisions` por constructor.
- Python 3.14 instala `pymysql`; pisos de modelos 30 e Identity Foundation Shadow 579.
- Suite ledger: 53 passed; modelos/schema/snapshot/ataques/providers: 264 passed.
- Mutantes V4, V6, V8 y V9 mueren en sus pruebas específicas.

## Pendiente

1. Cambiar `MappingProxyType` a `CatalogoTopes` sellado cuando se confirme el SHA r6 final de #370 y reapilar sobre él.
2. Volver a correr suites y pisos sobre el SHA final, actualizar PR #371 y dejar el reporte en `~/encargos-codex/entrega-codex-faro-f11-368-369-371-r3.md`.
- CI GitHub en #371 r5: `faro-bitacora-db` falla en `tests/test_faro_topes_db.py`; los casos crean `Topes` sin catálogo y usan el recurso legado `tokens`, por lo que D-4 los niega. El mismo fallo está en #370 r5. Se informó a Hyde para corregirlo en la rama de origen, sin alterar ese trabajo.
