# Traspaso continuo — F2-E Tramo 1

- Objetivo: reconciliar JAX PR #344 (Jacobs aviso Telegram por texto gobernado) con master actual, re-medición de pisos y CI verde para reauditoría corta de c3.
- Hecho: PR head previo `44864db9ab65ecba3dd6288614c682328e372ba6`; `origin/master` observado `142c91f13b200d41e38c214a46c266e56749c7eb` (#350 OCR y otros cambios). Merge local creado como `50e4d50a8c58678e8e6f77f236dd11e86452b6aa`. `ci/pisos.json` conserva `governance/f2e-external-output = ^10 passed` y adopta OCR `^219 passed, 2 skipped`; no se editaron `.github/ci/*.py` ni `pisos-no-bajan`.
- Verificado sobre la combinación: F2-E: `10 passed`; OCR con `PYTHONPATH=.:las_manos`: `221 passed` local (las 2 imágenes LACTOVI existen en hall9000; el runner no las tiene, así que su piso es `219 passed, 2 skipped`); `git diff --check` pendiente tras actualizar este archivo.
- CI del SHA publicado anterior estaba verde, pero no certifica el merge nuevo. Falta correr el comparador/migración/cobertura de archivos, publicar la rama y esperar todas las comprobaciones del SHA nuevo.
- Decisión de coordinación: jax-fe autorizó merge de master en #344, re-medir y entregar el SHA para reauditoría corta. No mergear PR ni desplegar.
- Tramo 2 de F2-E continúa en el worktree separado `f2e-structured-projection-jax` y su par de Plataforma; no reutilizar ni mezclar archivos con este branch.
- Siguiente comandos: `python3 -m pytest -q tests/test_pisos_migracion_desde_master.py tests/test_pisos_fuera_del_workflow.py tests/test_archivos_de_test_wireados_en_ci.py`; luego `git push origin HEAD:phase2/f2e-general-tramo1-20261004`; verificar con `gh pr checks 344 --repo fjruizhn/Jax` hasta que todos estén verdes y reportar el SHA final.
- Límites: sin integración a master, sin despliegue, sin auditoría propia.
