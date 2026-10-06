# Traspaso — JAX#357 ronda 4

- **Estado:** scanner simplificado desde `master` y whitelist de falsos positivos implementados; prueba enfocada verde (28 passed).
- **Rama:** `codex/jax353-docker-rm-followup`.
- **Base:** `8da1473939af2c8178b8b632e2d77b9120556cab`.
- **Cambios:** restaurado el scanner de `68557946`; solo se eximen los 11 subcomandos Docker autorizados cuando los tokens `docker`, subcomando y `rm` son literales y contiguos. Se conserva el fallo cerrado de Python con ruta y línea.
- **Banco:** runner diferencial actualizado para los 507 casos previos del PR, 136 casos del auditor y 30 nuevos; falta correr contra el commit candidato y guardar la salida.
- **Piso:** falta correr el comando exacto `tests-puros/out`, actualizar `ci/pisos.json` y `docs/ci/pisos.md` con la medición contra 3537 de master.
- **Pendiente de cierre:** revisar cambios, commit con firma Codex, correr bancos y piso, push a la rama y recoger SHA/PR/checks.
- **Restricción del encargo:** `PENDIENTES.md` permanece intacto.
