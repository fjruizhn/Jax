# Traspaso — Faro F1.1 RULE AUTHORITY KERNEL

## Objetivo

Implementar únicamente el núcleo de autoridad de reglas aprobado por Fernando el 2026-10-05: esquema `policy/faro/*.yaml`, carga desde snapshot confiable, ratificación individual Block 4 ligada al hash exacto, firma Ed25519 `human:fernando`, vigencia, presencia en el SHA de policy, STOP, decisiones `PERMIT`/`DENY`/`MISSING_RULE`, `RulePermit` corto y de un solo uso, y auditoría durable.

## Hecho

- La cadena de recuperación A–D terminó en JAX master `f69f045a35fc5d6143b1d6a17934ee01b42d55d1`, con CI post-merge `37322257433` verde.
- La rama `feat/faro-f1.1-rule-authority` parte exactamente de ese master.
- La coordinación está publicada en `claude-skills` commit `d0132ac7` con `[EN CURSO: hall9000.codex]`.
- Se inició descubrimiento de solo lectura del código vigente y de los contratos Block 4/5/6/Faro.

## Falta

- Terminar el mapa de contratos existentes.
- Obtener diseño Tier 3 y resolver cualquier ambigüedad.
- Escribir y someter a revisión humana la spec y el plan exigidos por Superpowers.
- Implementar con TDD mediante Tier 2, verificar, auditar el SHA exacto y abrir PR.

## Decisiones

- Fernando aprobó en persona el 2026-10-05 el contrato de ratificación individual descrito en el encargo de Gobernanza; `RATIFICATION_GRANTED` del corpus no cuenta como ratificación de una regla.
- `policy/**` sigue reservado: su integración corresponde a Fernando.
- No entran motor real, jaula nueva, Qwen, SSH, GitHub, producción ni activación de RL01.

## Siguiente comando

```bash
git status --short && rg -n "RATIFICATION_GRANTED|human:fernando|policy_corpus_hash|effective_authority" policy tests/policy
```
