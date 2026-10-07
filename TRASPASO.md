# Traspaso Faro F1.1 · #371 (paso 4) · ronda 3

## Estado (2026-10-07)

- Rama `feat/faro-f1.1-evaluador`, worktree `/home/fruiz/wt/jax-371-r3`. Base: #370 `feat/faro-f1.1-schema-snapshot` @ 05183cb7 (fusionada con merge commit; conflictos de TRASPASO.md, ci/pisos.json y docs/ci/pisos.md resueltos conservando todos los pisos).
- Hecho en la ronda 3 (cierra el veredicto de ronda 2): catálogo por tipo exacto `CatalogoTopes`; `RuleLimits` = envoltorio de `LimitesObligatorios`/`Tope` de #370, solo vía `limites_de(regla, catalogo)`; `RuleEvaluation` exige el mismo catálogo y el `request_hash` lleva el OID; `type(x) is str` en unidades/moneda; enteros de `arguments` con rango; `store.record` exige `RuleDecision`; sin campo huérfano ni import duplicado; `pymysql==1.2.0` con hashes (`requirements-faro-mariadb-ci.txt`); historia corregida.
- Pisos medidos en hall9000 (Python 3.14.4, pytest 9.1.1): modelos 132, identity 613, codec 19, ratificaciones 12, MariaDB 4. `tests/policy` completo con MariaDB efímera: 803 passed, 15 skipped.
- Mutantes: 28 mutaciones de `models.py`/`store.py`, todas muertas (tabla en `~/encargos-codex/entrega-faro-f11-371-r3.md`).

## Pendiente

- Auditoría de escalón 3 sobre el SHA final (otra invocación). Nada se declara aprobado aquí.
- Si #370 cambia de SHA, repetir la fusión y re-medir identity (el piso depende de la lista del paso).
- No hay evaluador en #371 (solo modelos y store): vigencia, STOP, snapshot y grants son pasos posteriores.

## Siguiente comando

```sh
JAX_AUTHORITY_LEDGER_DOCKER_CMD='sudo -n docker' PYTHONPATH=. python3 -B -m pytest -q tests/policy -p no:cacheprovider
```
