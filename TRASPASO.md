# Traspaso — JAX#357 ronda 4

- **Estado:** scanner basado en `master`; whitelist, recorrido Compose, banco diferencial y piso revisados. Auditoría de escalón 3 aprobó `08d7d200`; revalidar solo si cambia el SHA antes de empujar.
- **Rama:** `codex/jax353-docker-rm-followup`.
- **Base:** `8da1473939af2c8178b8b632e2d77b9120556cab`.
- **Cambios:** base del scanner restaurada desde `68557946`; whitelist literal de 11 subcomandos, sintaxis Python inválida falla cerrada con ruta y línea, y búsqueda hacia atrás hasta separador para Compose con opciones largas. No se conserva `&` como separador: la auditoría confirmó que eso omite una marca de `master` fuera de whitelist.
- **Auditoría anterior:** SHA `02e57d75` rechazado por Compose largo y expectativas no exigidas por el runner. El auditor retiró su hallazgo sobre `&` tras verificar que añadirlo causa regresión frente a master. El scanner 8da completo fue descartado: auditoría comparativa encontró 65 fugas respecto a master.
- **Banco:** 673 casos (507 PR + 136 auditor + 30 propios), corridos con master `68557946` y scanner `3d08bd92`: 0 fugas, 0 marcas inesperadas, 0 expectativas incumplidas, 20 marcas nuevas esperadas de Compose largo, 5 diferencias autorizadas por whitelist. Salida en `docs/ci/jax357-differential.out`.
- **Piso:** medición previa con comando exacto del workflow: 3562 passed, 45 skipped, 1 xfailed, 16 subtests; master: 3537, delta +25. Solo se añadieron aserciones a un test existente después; no cambió el conteo. `ci/pisos.json` y `docs/ci/pisos.md` documentan la medición.
- **Pendiente de cierre:** empujar la rama existente de PR #357 y recoger SHA/checks de CI. No integrar el PR en esta tarea.
- **Restricción del encargo:** `PENDIENTES.md` permanece intacto.
