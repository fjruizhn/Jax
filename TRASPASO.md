# Traspaso · feat/faro-f1.1-schema-unidades

## Retoma Codex · 2026-10-09

- Reapilado en worktree propio `/home/fruiz/jax/.worktrees/codex-faro-378-rebase-20261009`, rama `codex/faro-378-rebase-20261009`; la rama/worktree ZCode y el PR #378 original siguen intactos.
- Base actual: `origin/master@da99e6e22ccd1689950ed5ef13278fbd14ffb03c`, que incluye #381 (`ec89ef1e`) y el registro documental #384. Se reaplicaron los seis commits del delta de #378 desde `4c3064a2`.
- Conflictos resueltos preservando el workflow vigente y los cambios ya integrados de #381; el piso identity se suma a 1003 de master (+34 schema, +1 prueba de catálogo forjado, +2 paridad JSON/Python).
- Medición exacta del paso `Identity Foundation Shadow -- piso exacto de tests CORRIDOS`, extraído del workflow, con Python 3.14.4: `1040 passed`, cero skipped; `piso.py verificar` rc=0. `ci/pisos.json`, `.github/workflows/policy.yml` y `docs/ci/pisos.md` reflejan ese conteo.
- PR #378 original sigue OPEN, con head `4c3064a2` y base `master@8c850bce`; no se cambió su rama ni su head remoto. La rama Codex reapilada aún no está publicada ni auditada.
- Decisión de coordinación: continuar sin esperar respuesta de GLM/ZCode, que ya terminó la revisión; preservar sus worktrees y preparar un head nuevo para revisión exacta.

### Siguiente gate

1. Ejecutar `git diff --check` y la suite focal `tests/policy/test_rule_authority_schema.py` sobre esta rama.
2. Registrar esta evidencia, borrar `TRASPASO.md` y dejar el traspaso durable en `docs/historia/` antes de solicitar auditoría Tier 3 final.
3. Publicar la rama Codex sin reescribir `feat/faro-f1.1-schema-unidades`, abrir un PR de reemplazo y esperar CI completa y auditoría exacta.
4. Inmediatamente antes de cualquier integración, ejecutar `bin/ventana estado`, `verify-integration` para el SHA auditado y `post-merge-guard` después del merge.


## Objetivo

Cerrar el hallazgo del auditor de #371: `policy/rule_authority/schema.py` no validaba
`Cantidad.unit` ni `Monto.currency` contra el catálogo sellado (solo `limites_de`
re-validaba al evaluar), y la fixture `regla-ejemplo-tope.yaml` usaba `unit: llamadas`,
fuera del catálogo. Encargo: `~/encargos-codex/encargo-glm-g2-schema-unidades.md`.

**Reapilado 2026-10-09 sobre `origin/master@8c850bce`** (post-merge de #371). Base
original de la rama: `origin/feat/faro-f1.1-evaluador @ d4ffe62f`; #371 llegó a master
por la rama reconstruida `feat/faro-f1.1-paso4-rebuild` (tip c86582af), por eso el PR
estaba CONFLICTING y los SHAs del evaluador viejo no son ancestros de master. El delta
propio (4 commits: schema+fixture+pruebas, pisos, docs, r2) se reapiló tal cual;
`schema.py` y la fixture quedaron byte-idénticos a la rama original y el archivo de
pruebas final = pruebas de master + las 35 de esta rama (los 3 nombres extra del archivo
son pruebas nuevas de master, presentes y contadas en su piso 968).

## Hecho en esta rama

- `schema.py`: helper `_validar_subid` al estilo de `_validar_periodo` — `type is str`
  (subclases con `__eq__`/`lower` forjados no pasan), NFC, pertenencia exacta al
  catálogo SELLADO (`CatalogoTopes` del pin) y `tras_lower` para la forma ISO de las
  monedas (`USD` ↔ subid `usd`, igual que `limites_de`).
- `_validar_cantidad` exige `unit` subid de `actos_externos`; `_validar_monto` mantiene
  la forma ISO 4217 (3 mayúsculas) y exige `currency` subid de `monto_dinero`. Una regla
  con cantidad o monto exige el catálogo del pin: sin él (o sin sellar), niega (B-3
  extendido a los límites). La regla base sin límites sigue validando sin catálogo.
- `_RE_UNIDAD` desaparece (sin uso): el catálogo cierra más que la forma.
- Fixture corregida: `unit: llamadas` → `unit: mensajes` (subid real de actos_externos).
- 34 pruebas nuevas en `tests/policy/test_rule_authority_schema.py`: 16 negativas de
  unit (llamadas, workers, mayúscula, espacios, ancho cero, int, None, True, lista,
  subclase de str, subids de OTRAS clases), 8 negativas de currency (EUR —forma ISO
  válida ausente del catálogo—, USD con ancho cero, espacio final, int, subclase),
  5+2 positivas del catálogo real (los 5 subids de actos, HNL/USD), 1 sin catálogo /
  sin sellar, 1 de pin alterno (el vocabulario es dato del pin), 1 de no-vacío.
- `limites_de` NO se toca: sigue re-validando al evaluar (defensa en profundidad).

## Mutantes muertos

| Mutante | Pruebas que lo matan | Aserción |
|---|---|---|
| «sin validar unit» (vuelve a solo-forma `_RE_UNIDAD`) | 7 fallos: negativas `llamadas`/`workers`/etc., sin-catálogo y pin alterno | `pytest.raises` con «quantity.unit» ya no ocurre: la forma regex pasa |
| «sin validar currency» (vuelve a solo-forma ISO) | 3 fallos: `[EUR]`, `[USD\u200b]`, pin alterno | EUR es forma ISO válida: sin membresía al catálogo firma |
| M6: `_validar_subid` con `if catalogo is None: return valor` | `test_cantidad_o_monto_sin_catalogo_del_pin_niega` (1 fallo) | regla SIN tope; `match=^obligation_limits.quantity.unit: sin catalogo` (y `amount.currency`). Con tope, `_validar_tope` negaba igual con «sin catalogo» y el mutante sobrevivía (corregido en r2) |
| M5: quitar el chequeo de sellado del helper | esa misma prueba + `test_un_catalogo_forjado_por_subclase_no_cuenta_como_sellado` (2 fallos) | prefijo `obligation_limits.quantity.unit: ... SELLADO`; subclase forjada con `object.__new__` |
| schema.py con `isinstance` (código de 220fc977) | `test_un_catalogo_forjado_por_subclase_no_cuenta_como_sellado` (1 fallo) | la subclase forjada pasaba el `isinstance`; ahora `type(catalogo) is not CatalogoTopes`, en el helper y en `_validar_tope` |

M5/M6 comprobados con copias `git archive` del árbol (mutante aplicado sobre el schema nuevo, pruebas nuevas).

## Pisos (re-medidos 2026-10-09 sobre el árbol reapilado, venv-faro Python 3.14.4)

- `identity-foundation-shadow/policy`: 968 (master) → **1005** (+34 del schema de
  unidades + 1 de la subclase forjada, r2 + 2 de paridad del espejo, MINOR Tier 3).
  Medido con el comando exacto del paso (lista extraída del propio workflow):
  `1005 passed in 5.27s`, 0 skipped, `piso.py verificar` rc=0. En master este piso
  ya no es 659: #373/#379/#371 lo subieron a 968 y la clave
  `authority-rule-models/models` (151) desapareció de `ci/pisos.json`
  (los modelos entraron a la lista identity como `test_faro_rule_authority_models.py`).
- `faro-fase0/faro`: **780** en master; esta rama no toca sus pruebas ni su job → sin
  re-medición local, CI confirma.
- Guardas de CI sobre el árbol final: `archivos-de-test-en-ci/pisos` **227 passed**
  (`piso.py verificar` rc=0), `archivos-de-test-en-ci/wireados` **7 passed** (rc=0),
  `comparar_pisos.py` contra origin/master OK (54 pisos, ninguno baja).

## MINOR del auditor Tier 3 (cerrado 2026-10-09, r3)

- El espejo `policy/faro/schemas/rule-v1.schema.json` aceptaba unidades/monedas que
  Python rechaza: `quantity.unit` solo tenía `pattern` de forma y `amount.currency`
  solo `^[A-Z]{3}$`. Ahora son enums del catálogo sellado (actos_externos: compras,
  correos, mensajes, pagos, publicaciones; monto_dinero en forma ISO: HNL, USD), igual
  que `tope.period`/`resource` ya lo eran.
- Pruebas: 2 vectores de paridad negativos (`llamadas`, `EUR`) en
  `test_el_espejo_json_y_python_aceptan_los_mismos_vectores_de_regla` + binding del
  test de vocabularios (los enums del espejo == catálogo real, unidades y monedas).
- Mutante comprobado: con el espejo revertido a `pattern`, exactamente esos 2 vectores
  fallan (stash/restore verificado); con el enum, 167 passed en el archivo.

## Pendiente

- Auditoría de Hyde y merge de Fernando. #377 (ledger sello+append) y #379 ya están en
  master; el PR quedó apuntando a master tras el reapilado.
