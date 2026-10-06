# Traspaso — JAX#357 ronda 4

- **Estado:** scanner simplificado desde `master`; whitelist, banco diferencial y piso re-medidos. Listo para auditoría de escalón 3 sobre el próximo SHA.
- **Rama:** `codex/jax353-docker-rm-followup`.
- **Base:** `8da1473939af2c8178b8b632e2d77b9120556cab`.
- **Cambios:** restaurado el scanner de `68557946`; solo se eximen los 11 subcomandos Docker autorizados cuando los tokens `docker`, subcomando y `rm` son literales y contiguos. Se conserva el fallo cerrado de Python con ruta y línea.
- **Banco:** 673 casos (507 PR + 136 auditor + 30 propios); 0 fugas, 0 marcas nuevas, 5 diferencias de whitelist. Salida en `docs/ci/jax357-differential.out`; scanner SHA `9f3329b3`.
- **Piso:** comando exacto del workflow: 3562 passed, 45 skipped, 1 xfailed, 16 subtests; base master: 3537, delta real: +25. Piso y nota actualizados.
- **Pendiente de cierre:** auditoría sobre SHA final, push a la rama existente de PR #357 y recoger SHA/checks.
- **Restricción del encargo:** `PENDIENTES.md` permanece intacto.
