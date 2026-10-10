# TRASPASO · #381 tras auditoría exacta

- Objetivo: cerrar los dos BLOCK del veredicto Tier 3 sobre `c8421337e3c380ae4347acb14f6b4e12628b6915` antes de solicitar auditoría nueva.
- Hecho verificado: auditoría RECHAZADO por writer privado que aceptaba intent arbitrario y piso Identity Foundation desactualizado (968 vs incremento neto esperado de 34).
- Corrección en curso: flujo de firma ratification inline dentro de `append_ratification_from_candidate`, eliminado el helper privado; regresión testifica ausencia del helper.
- Falta: correr pruebas focales y el comando exacto Identity Foundation del workflow, actualizar pisos y documentación según medición; documentar el rechazo; commit/push, auditoría nueva del SHA, CI verde e integración reglamentaria.
- Decisión: preservar la frontera de autoridad y medir con la lista exacta del workflow, no estimar el piso.
- Siguiente comando: extraer la lista de paths entre líneas 2681–2756 de `.github/workflows/policy.yml`, ejecutar `python -B -m pytest -q <paths> --confcutdir=tests/policy -p no:cacheprovider` y verificar el piso actualizado.
