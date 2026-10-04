# Traspaso continuo — F2-E Tramo 1

- Objetivo: reconciliar JAX PR #344 (Jacobs aviso Telegram por texto gobernado) con master actual, volver a medir CI y entregar el SHA a jax-fe para reauditoría de c3.
- Hecho: #345 (master `3a605bd0b123ff2312225ce3f231728fd535ea8e`) se incorporó a #344. Luego master avanzó con #348 a `f7386d8c3690832e5d9c21c79cab776998a4bfa1`; delta revisado e incorporado sin conflicto: comparador de pisos permite el endurecimiento `^N passed` → `^N' passed, M skipped`, fija OCR `201 passed, 2 skipped` y sube el piso de tests de pisos 223→227. Cambios de `.github/ci/*.py` vienen íntegros del merge de master, sin edición manual.
- Verificado local sobre combinación: F2-E tests `10 passed`; pruebas de comparador/migración/workflow/pisos `234 passed`; suite afectada `tests/test_permisos_proyectos.py`: `307 passed in 125.13s` en este host (CI mide `306 passed, 1 skipped`); OCR con el `PYTHONPATH=.:las_manos` del job: `203 passed` en hall9000 (runner fija `201 passed, 2 skipped`); `git diff --check` limpio.
- CI: el SHA de #344 sobre #345 aún corría cuando master avanzó con #348 y no certifica la combinación final. El merge actual debe publicarse y su CI completa verificarse en el SHA resultante.
- Decisiones: se mantiene `governance/gov = 263`; F2-E conserva `governance/f2e-external-output = 10`. Se adoptan exactamente los pisos 306/1 y 201/2 de master #345/#348; no se sumaron deltas a ciegas.
- Falta: crear/publicar merge commit de #348, esperar CI completa verde, verificar master, enviar el SHA final a jax-fe para reauditoría independiente de c3. No iniciar Tramo 2 hasta cerrar este paso.
- Siguiente comando: `git push origin HEAD:phase2/f2e-general-tramo1-20261004`; después consultar `gh pr checks 344` y esperar todos los checks.
- Límites: no merge, no despliegue, no auditoría propia. Tramo 2 aún no iniciado.
