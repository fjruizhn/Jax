# F2-E tramo 2 — reconciliación después de SR2

**Fecha:** 2026-10-05
**Tipo:** HISTORIA
**Fuente:** merge regular de `origin/master`
`fff343d15c1aca16b318f16f109dbd59d07c732e` en JAX PR #351.

## Qué se hizo

Se incorporó el master que contiene SR2 STEP_STATUS a la rama F2-E structured
projection mediante un merge regular, sin rebase, reset ni limpieza. Hubo dos
conflictos: el workflow de política y `jacobs/store.py`.

El workflow conserva ambos pasos exactos e independientes: structured
projection (`governance/f2e-structured-projection`) y la salida externa Jacobs
(`governance/f2e-external-output`). En `jacobs/store.py` permanecen tanto la
lectura por lote canónica de `PIPELINE_STATUS` que requiere la proyección como
la instantánea canónica de paso y dueño que requiere SR2.

## Reconciliación de contratos

SR2 elevó el contrato compartido de runtime status a `.4` y el dominio F2-C a
`f2-c.domain.7`. La composición de aviso Jacobs y la proyección estructurada se
alinearon con esos valores. Los dobles de prueba siguen la lectura por lote
canónica, por lo que ya no pueden acreditar el estado con un objeto del
llamador ni con una proyección de Platform.

## Verificación

- `tests/test_f2e_structured_projection.py`: `15 passed` y piso exacto verde.
- `tests/test_external_output.py` + `jacobs/_governed_aviso_test.py`:
  `10 passed` y piso exacto verde.
- `tests/test_f2e_step_status.py`: `26 passed` y piso exacto verde.
- `tests/test_f2e_runtime_status.py`, `tests/test_governance_loaders.py` y
  `tests/test_governance_validator.py`: `55 passed`.
- `.github/workflows/policy.yml` se cargó como YAML y no contiene marcadores
  de conflicto.

## Límites

La CI completa y la auditoría sobre el SHA publicado siguen obligatorias antes
de integrar. No se redujo ningún piso, no se desplegó y no se modificaron los
triggers de `las-voces-sync.yml`.
