# Traspaso — Faro F1.1 RULE AUTHORITY KERNEL

## Objetivo

Registrar las decisiones de implementación del contrato F1.1 aprobado por Fernando
el 2026-10-05 y el estado verificable de esta ronda. Este archivo es un handoff;
no concede autoridad ni prueba el estado operativo actual.

## Estado al 2026-10-07 · Hall9000

- La spec F1.1 está en
  `docs/superpowers/specs/2026-10-05-faro-f1-1-rule-authority-kernel-design.md`;
  su sección 16 registra nueve decisiones de implementación.
- PR #376 (`a3bf02c4`) fue integrado por solicitud explícita de Fernando el
  2026-10-07 en su rama padre `feat/faro-f1.1-evaluador` mediante el merge
  `5cfc44c6610c32f036cd4e978e88195dd5f91520`. La rama padre quedó en ese SHA.
  Esto completa la integración en esa rama apilada; aún no afirma que el código
  esté en `master`.
- PR #382 contiene la alineación documental. Su head `3d307197` está sincronizado
  con `master` y su diff hacia `master` consta de `TRASPASO.md` y 82 líneas de la
  spec. Los checks de ese SHA estaban verdes. La auditoría independiente aprobó
  con un hallazgo menor: este handoff tenía instrucciones vencidas. Este commit
  corrige ese estado; el SHA nuevo requiere CI y auditoría exacta antes del merge.
- Fernando instruyó integrar #376 y #382 el 2026-10-07. Para #382, el merge debe
  usar el SHA nuevo aprobado y la secuencia de integración verificada para PRs
  dirigidos a `master`.
- La cadena de código F1.1 conserva dependencias apiladas; el merge de #376 en
  `feat/faro-f1.1-evaluador` no integra por sí solo esa cadena en `master`.

## Decisiones y límites

- Faro decide permisos antes de ejecutar; el aviso comunica una decisión y no
  modifica autoridad.
- No inferir que una PR apilada quedó en `master` solo porque se integró en su
  rama padre.
- No reutilizar auditorías de un SHA anterior tras modificar el PR.
