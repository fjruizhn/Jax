# Traspaso · Faro F1.1 paso 7 (jax#373) ronda 3

## Continuación 2026-10-08 · rebase sobre #370 final

El usuario pidió integrar toda la cadena pendiente. Esta rama tiene una preparación local
sin commit que reaplica el snapshot de #370 final (`3692173f`) sobre el trabajo r3 de
providers. Cambios re-medidos en el árbol combinado:

- Faro Fase0: **729 passed**, sin skips, con el comando del workflow en Python 3.12.15.
- Identity Foundation Shadow: **778 passed**, sin skips, medido previamente con Python
  3.14.4 y el comando exacto del workflow; el entorno local de esta sesión solo ofrece
  Python 3.12.15, por lo que CI debe repetir la medición en 3.14.
- Se actualizaron `ci/pisos.json`, comentarios del workflow y el spec/handoff de #370.
- `git diff --check` limpio.

Antes de integrar #373 falta completar el commit de esta reaplicación, empujar la rama,
actualizar la base de su PR a `master`, esperar CI nueva y pedir auditoría Tier 3 del SHA
exacto. No usar la CI del SHA previo como evidencia para el nuevo commit.

Estado auditoría: el PR #371 c36c0543 fue **RECHAZADO** por tres BLOCK de providers,
un MAJOR en la cola de avisos de #376 y un piso documental desactualizado. La cola se
está corrigiendo en el worktree `jax-faro-f11-paso4`; incorpora límite/rotación por
segmento y cuarentena de segmentos legados >16 MiB. El proveedor r3 de esta rama contiene
el contrato de leases, guard y checkpoint endurecido que el auditor pide para sustituir
los providers iniciales rechazados de #371. Coordinar ese reemplazo al reapilar #371 y
mantener auditoría exacta después del cambio.

Los 4 tests de MariaDB de `tests/policy/test_authority_ledger_storage_mariadb.py`
requieren Docker; CI aislada los corre en runner. No se hicieron cambios ni despliegues a
producción.

Rama `feat/faro-f1.1-providers-r3` (local, SIN empujar): b16216d7 (r2, RECHAZADO) + merge de
`origin/feat/faro-f1.1-schema-snapshot` (#370 @ 05183cb7) + 3 commits de r3. Veredicto de entrada:
`~/encargos-codex/veredicto-jax373-r2.md`; encargo: `encargo-glm-g2-faro-f11-paso7-r3.md` (B).

Hecho en r3: BLOCK (sin `except...pass`; STOP ilegible niega), MAJOR-1 (guard probado con lease valido
y solo el dato malo), MAJOR-2 (procedencia `refs/heads|tags/...` anclada e igual a la del pin,
`type(pin) is TrustedPolicyPin`), MAJOR-3 (reloj `utcoffset() == 0`), MAJOR-4 (sin lecturas sueltas:
todo es valor dentro de `VistaLease`, validado por tipo), MAJOR-5 (centinela, §13), MINOR (exclusion
determinista de hilo a hilo, `confirmar` = el log contiene el head, `__delattr__`, forma de limites,
W3 declarado y cerrado en falso). Piso Identity 704.

Falta (no es de esta ronda): CI de GitHub sin verificar (no hay push); los 4 tests MariaDB de
`tests/policy/test_authority_ledger_storage_mariadb.py` necesitan docker y no corrieron aqui.
Interpretacion a confirmar: `FormaLimites` (enum cerrado) es mia — §10.7 solo dice «forma de limites».

---

# Traspaso · Faro F1.1 #368/#369/#371

## Objetivo

Cerrar los hallazgos del veredicto de escalón 3 para #368, #369 y #371. El veredicto de
entrada está en `~/encargos-codex/jax-faro-f11-veredicto-368-369-371-r2.md`.

## Hecho en esta rama (#368)

- El modelo compara ambos sellos por identidad; el writer exige el sello de snapshot para
  `RATIFICATION_GRANTED` y sigue rechazando el sello de storage.
- `previous_event_hash` se lee y contrasta junto con las demás columnas denormalizadas.
- `canonical_intent` es NOT NULL en la migración inicial y en la migración de upgrade.
- Se agregó `policy.authority_ledger.provisioning` para crear la cuenta de aplicación y sus
  grants limitados, usando credenciales del entorno.
- Las pruebas K3, K8, D2 y D3 fallan al retirar el control correspondiente.
- MariaDB 12.3.3 efímera pasó migración, provisión, append/replay y triggers con un principal
  que tiene SELECT/UPDATE/DELETE.
- Suite `tests/policy/test_authority_ledger*.py`: 41 passed.
- El codec corrió 19 pruebas; el piso `authority-ledger-codec/codec` quedó en 19.
- El paso Python 3.14 del workflow instala `pymysql`, requerido por el job MariaDB aislado.
- Ronda 3, dos hallazgos menores: el provisioning ahora hace `REVOKE ALL PRIVILEGES, GRANT OPTION`
  antes de los GRANT (la cuenta preexistente con ALL converge al contrato) y la migración 002
  aborta con `SIGNAL 45000` si hay filas NULL (independiente de `sql_mode`) y corre bajo
  `STRICT_ALL_TABLES`. Tres pruebas MariaDB nuevas (SHOW GRANTS exacto; 002 con NULL y
  `sql_mode=''`; 002 sin NULL). Piso `authority-ledger-mariadb/integration`: 1 -> 4.

## Hecho en esta rama (#369)

- Se integró el kernel corregido de #368 conservando las pruebas de las ocho variantes del
  ledger.
- La activación solo acepta una ratificación de corpus; una ratificación individual no puede
  servir como su destino (E4).
- `current_rule_ratification` se renombró a `latest_unrevoked_rule_ratification`; su contrato
  aclara que la vigencia temporal la evalúa Rule Authority.
- Los sellos de las ratificaciones individuales también se validan por identidad; una
  instancia `AlwaysEqual` es rechazada.
- La prueba E10 llega al constructor y mata la mutación que elimina el rechazo de payload Block
  4 heredado.
- Pisos verificados en esta rama: codec 19, MariaDB 1, ratificaciones 12.
- Se incorporó desde #368 la instalación de `pymysql` en el paso Python 3.14, necesaria para el test MariaDB del workflow.
- Suite `tests/policy/test_authority_ledger*.py`: 53 passed. K3, K8, D2, D3, E4 y E10 muertos.

## Falta

- #371 espera el SHA final de #370 antes del rebase sobre `origin/feat/faro-f1.1-schema-snapshot`.
- Re-medición final de los pisos sobre la base real apilada, merge-tree y reporte de entrega.

## Decisiones

- Se siguen los cierres obligatorios del veredicto r2 y la decisión indicada por Fernando para
  consumir el catálogo de topes de #370. No se escribe ni publica un veredicto de auditoría.
- No se usan llaves de Fernando ni bases persistentes; las pruebas MariaDB usan contenedor
  desechable, `--network none` y socket Unix.

## Siguiente comando

Desde el worktree #369, ejecutar:

```sh
JAX_AUTHORITY_LEDGER_DOCKER_CMD='sudo -n docker' PYTHONPATH=. python3 -B -m pytest -q tests/policy/test_authority_ledger*.py
```
