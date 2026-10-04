# Traspaso continuo — F2-E Tramo 1

- Objetivo: reconciliar JAX PR #344 (Jacobs aviso Telegram por texto gobernado) con master actual, volver a medir CI y entregar el SHA a jax-fe para reauditoría de c3.
- Hecho: master `52e63c9bd1fb6153584edced55b64749fabdf2f5` se incorporó en `a1868fc`; master avanzó después con #345 a `3a605bd0b123ff2312225ce3f231728fd535ea8e` y ese delta se incorporó localmente sin conflicto. Se preservaron el test genérico de #340, el paso F2-E, el piso separado de 10 y se adoptó el piso medido `permisos-proyectos/permisos_proyectos = 306 passed, 1 skipped` de #345.
- Verificado local sobre la combinación: F2-E tests `10 passed`; pruebas de migración/workflow/pisos `155 passed`; suite afectada `tests/test_permisos_proyectos.py`: `307 passed in 125.13s` en este host (el runner mide `306 passed, 1 skipped`); `git diff --check` limpio.
- CI: #344 sobre `a1868fc` estaba verde; ese resultado no certifica el merge local de #345. El merge nuevo debe publicarse y su CI completa verificarse en el SHA resultante.
- Decisiones: se mantiene `governance/gov = 263`; F2-E conserva la clave de piso independiente `governance/f2e-external-output = 10`. No se sumaron deltas de pisos a ciegas.
- Falta: crear/publicar el merge commit de #345, esperar CI completa verde, verificar master, enviar el SHA final a jax-fe para reauditoría independiente de c3. No iniciar Tramo 2 hasta cerrar este paso.
- Siguiente comando: `git push origin HEAD:phase2/f2e-general-tramo1-20261004`; después consultar `gh pr checks 344` y esperar todos los checks.
- Límites: no merge, no despliegue, no auditoría propia. Tramo 2 aún no iniciado.
