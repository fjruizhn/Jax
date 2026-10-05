# Traspaso — Faro F1.1 RULE AUTHORITY KERNEL

## Objetivo

Implementar únicamente el núcleo de autoridad de reglas aprobado por Fernando el 2026-10-05: esquema `policy/faro/*.yaml`, carga desde snapshot confiable, ratificación individual Block 4 ligada al hash exacto, firma Ed25519 `human:fernando`, vigencia, presencia en el SHA de policy, STOP, decisiones `PERMIT`/`DENY`/`MISSING_RULE`, `RulePermit` corto y de un solo uso, y auditoría durable.

## Hecho

- La cadena de recuperación A–D terminó en JAX master `f69f045a35fc5d6143b1d6a17934ee01b42d55d1`, con CI post-merge `37322257433` verde.
- La rama `feat/faro-f1.1-rule-authority` parte exactamente de ese master.
- La coordinación está publicada en `claude-skills` commit `d0132ac7` con `[EN CURSO: hall9000.codex]`.
- Se completó el descubrimiento de solo lectura del código vigente y de los contratos Block 4/5/6/Faro.
- La arquitectura Tier 3 emitió `APROBADO PARA IMPLEMENTAR CON CONDICIONES` y confirmó como condición previa la corrección del adapter completo de persistencia de Block 4.
- La revisión Tier 4 del diseño final emitió `APROBADO PARA IMPLEMENTAR`, sin bloqueadores; se incorporaron sus precisiones sobre leases, vigencia, clasificación de capabilities, checkpoint externo y recuperación idempotente.
- Se escribió `docs/superpowers/specs/2026-10-05-faro-f1-1-rule-authority-kernel-design.md` para revisión humana.
- Baseline focal: `729 passed, 39 skipped, 1 failed`; el único fallo es la prueba MariaDB que exige variables restringidas de `jax_memory_test`, ausentes en esta sesión. No se cambió host/base/user ni se accedió a producción.
- El intento de enrutamiento `agentdb_route` de Ruflo falló con `EACCES` al crear `/.claude-flow/policy`; no produjo análisis ni cambió estado del repo.

## Falta

- Someter la spec escrita a revisión humana, como exige el flujo de brainstorming de Superpowers.
- Tras aprobación humana, escribir y revisar el plan exigido por Superpowers.
- Implementar con TDD mediante Tier 2, verificar, auditar el SHA exacto y abrir PR.

## Decisiones

- Fernando aprobó en persona el 2026-10-05 el contrato de ratificación individual descrito en el encargo de Gobernanza; `RATIFICATION_GRANTED` del corpus no cuenta como ratificación de una regla.
- `policy/**` sigue reservado: su integración corresponde a Fernando.
- No entran motor real, jaula nueva, Qwen, SSH, GitHub, producción ni activación de RL01.

## Siguiente comando

```bash
sed -n '1,560p' docs/superpowers/specs/2026-10-05-faro-f1-1-rule-authority-kernel-design.md
```
