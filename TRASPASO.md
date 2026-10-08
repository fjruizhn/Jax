# Traspaso — Faro F1.1 RULE AUTHORITY KERNEL

## Objetivo

Registrar las decisiones de implementación del contrato F1.1 aprobado por Fernando
el 2026-10-05 y el estado verificable de esta ronda. Este archivo es un handoff;
no concede autoridad ni prueba el estado operativo actual.

## Estado verificado al 2026-10-07 · Hall9000

- La spec F1.1 está en
  `docs/superpowers/specs/2026-10-05-faro-f1-1-rule-authority-kernel-design.md`;
  su sección 16 registra nueve decisiones de implementación.
- PR #376 (`a3bf02c4`) se integró por solicitud explícita de Fernando en su rama
  padre `feat/faro-f1.1-evaluador` mediante el merge
  `5cfc44c6610c32f036cd4e978e88195dd5f91520`. Esta rama forma parte del PR #371;
  el merge de #376 no pone por sí solo el código en `master`.
- PR #382 integró la alineación documental en `master` mediante el merge
  `1f9c2fa5eb8ec297bed7e70fa43dbef1e2ef52a5`. Su CI `push` sobre el merge terminó
  verde y el `post-merge-guard` verificó el SHA, el tip de `master` y la ausencia
  de carrera. El guard primero encontró una lectura inestable mientras nacía CI;
  se repitió tras finalizar los jobs y entonces confirmó éxito.
- Al verificar este handoff, los PRs de código F1.1 #370, #371, #373, #375 y #378
  seguían abiertos y apilados. La cadena de código aún no estaba integrada en
  `master`; comprobar sus refs y CI actuales antes de retomarlos.
- La auditoría inicial de #382 detectó que un handoff anterior decía que #376
  necesitaba sincronizarse y que #382 no debía integrarse. Ese texto fue corregido
  en #382 antes de su merge; la copia de este archivo actualiza el estado posterior.

## Decisiones y límites

- Faro decide permisos antes de ejecutar; el aviso comunica una decisión y no
  modifica autoridad.
- No inferir que una PR apilada quedó en `master` solo porque se integró en su
  rama padre.
- No reutilizar auditorías de un SHA anterior tras modificar el PR.
