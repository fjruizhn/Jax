# Traspaso Codex · JAX Faro #373 · 2026-10-08

## Decisión y estado

Fernando decidió alinear `FormaLimites` al schema vigente: enum cerrado con `NINGUNA`,
`CANTIDAD`, `MONTO`; una acción `OBLIGATING` requiere exactamente cantidad o monto, y
`unidad`/`moneda` deben coincidir con la forma. No se amplía schema ni evaluador.

En `feat/faro-f1.1-providers-r2`, el rechazo Tier 3 anterior incluía prioridad de writer,
catálogo cerrado, tipos escalares exactos y esta incoherencia. El cambio local actual elimina
`CANTIDAD_Y_MONTO`, valida coherencia, ajusta las regresiones y el piso Identity Shadow a 791.
Evidencia local Python 3.14.4: providers 130 passed; lista exacta Identity Shadow 791 passed,
cero skipped; py_compile, JSON y diff-check limpios. El CI remoto todavía no cubre estos cambios.

## Dependencias y próximo paso

PR #373 debe actualizar su base a `master` después de mergear #379, incluir los cambios locales,
remedir el piso sobre el árbol final y recibir nueva auditoría Tier 3 del SHA exacto antes de
integrar. #371 consume #373 y además necesita la corrección de cola de #376; #375 y #378 están
apilados sobre #371. No desplegar.

```sh
cd /home/fruiz/wt/jax-faro-f11-providers && git status --short --branch
```
