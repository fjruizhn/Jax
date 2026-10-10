# Traspaso · feat/faro-f1.1-schema-unidades

## Objetivo

Cerrar el hallazgo del auditor de #371: `policy/rule_authority/schema.py` no validaba
`Cantidad.unit` ni `Monto.currency` contra el catálogo sellado (solo `limites_de`
re-validaba al evaluar), y la fixture `regla-ejemplo-tope.yaml` usaba `unit: llamadas`,
fuera del catálogo. Encargo: `~/encargos-codex/encargo-glm-g2-schema-unidades.md`.
Base: `origin/feat/faro-f1.1-evaluador` @ d4ffe62f.

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

## Pisos

- `identity-foundation-shadow/policy`: 659 → **693** (medido con el comando exacto del
  paso, `piso.py verificar` OK; comentario del paso actualizado con el desglose).
- `authority-rule-models/models`: **151** sin cambios (verificado).
- `faro-fase0/faro`: **727** sin cambios (verificado con `~/tmp-glm/venv-faro`).
- Guardas de CI: wireados + bash válido + pisos fuera del workflow = 237 passed.

## Pendiente

- Auditoría de Hyde y merge de Fernando (el ledger sello+append va aparte, por #377).
