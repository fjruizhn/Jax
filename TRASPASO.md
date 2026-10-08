# Traspaso — Faro F1.1 RULE AUTHORITY KERNEL

## Objetivo

Registrar las decisiones de implementación del contrato F1.1 aprobado por Fernando
el 2026-10-05 y mantener el estado de la cadena de PRs de Faro. Este archivo es
handoff, no autoridad ni evidencia operativa.

## Estado verificado · 2026-10-07

- La spec F1.1 existe en `docs/superpowers/specs/2026-10-05-faro-f1-1-rule-authority-kernel-design.md`.
- La sección 16 registra nueve decisiones y sus PR/SHA de procedencia.
- PR #382 contiene esa alineación documental. Su base original apuntaba a una rama
  F1.1 antigua, por lo que el job `pisos-no-bajan` comparaba una historia incompleta
  con `master`. Para mantener una comparación válida se sincroniza primero esta rama
  con `master`; luego el PR apunta a `master`.
- PR #376 contiene los avisos del kernel. Su padre avanzó de `b967dff` a `d4ffe62`;
  la rama requiere incorporar esos cambios y resolver únicamente los conflictos de
  `ci/pisos.json` y `docs/ci/pisos.md`, conservando los pisos medidos.
- PR #370, #371, #373, #375, #376 y #378 siguen abiertos. Sus checks estaban verdes
  en la última consulta; #376 aún no tenía mergeable tree antes de esta reparación.
- PR #381, separado de estas dos reparaciones, fue rechazado por auditoría en
  `aa8d31e3` y su CI falla. No integrar ni usar ese SHA como aprobación de ledger.

## Pendiente de esta ronda

1. Terminar el merge de `master` en la rama de #382, actualizar el estado de este
   handoff, y confirmar que el diff hacia `master` solo contenga documentación.
2. Cambiar la base de #382 a `master`, validar `pisos-no-bajan` y todos los checks
   del nuevo merge SHA.
3. Incorporar la punta `d4ffe62` en la rama de #376; resolver los pisos por medición,
   ejecutar la suite `faro-fase0` y comprobar que el PR queda mergeable.
4. Esperar CI nuevo de #376 y comprobar que los cambios conservan la separación
   entre aviso y decisión de autoridad.

No integrar estos PRs como parte de esta ronda. La cadena de F1.1 requiere su propia
revisión y el GO de Fernando.
