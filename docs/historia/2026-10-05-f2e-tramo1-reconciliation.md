# F2-E tramo 1 — reconciliación tras el corte eléctrico

**Fecha:** 2026-10-05. **Decisión operativa:** Codex, siguiendo el orden y
los límites indicados por Fernando Ruiz en la recuperación de Axioma 3.0.

## Qué se hizo y por qué

El PR JAX #344 (`phase2/f2e-general-tramo1-20261004`) partió del HEAD
publicado `56fe7cc037e1d62c696ca04a336170f08a185e6d`. Se incorporó
`origin/master` en `d136cce5d88b3420a5066b40e7786c158118f29e` mediante
un merge regular, para incluir #353 sin sustituir los contratos F2-E.

Los blobs de `policy/governance/external_output.py`,
`jacobs/governed_aviso.py`, `jacobs/aviso.py` y las pruebas específicas
permanecieron idénticos a los del HEAD F2-E publicado. Los contratos F2-B,
F2-C y F2-D no se modificaron. El cambio de master añadió el barrido de
`docker rm -fv` y sus tres pruebas al workflow y elevó el piso de
`tests-puros/out` a `^3473 passed, 45 skipped`; el paso aislado
`governance/f2e-external-output` conserva `^10 passed`. Ningún piso común
se rebajó.

## Verificación y lecciones técnicas

- Las pruebas enfocadas F2-E y Docker dieron `13 passed` con `python3`.
- La medición de pisos, migración y cobertura del workflow dio `155 passed`.
- La corrida amplia local de `tests-puros` alcanzó `3512 passed, 3 skipped,
  3 xfailed`, pero falló `test_arranque_las_manos_no_shadowea_policy` porque
  un proceso hijo no pudo leer `/etc/jax/build/implementation-identity.json`
  (`PermissionError`). Este resultado local no sustituye la CI del SHA final.
- El comparador local contra `refs/pisos-base/master` no estaba disponible;
  la prueba de migración pasó. La prueba completa de CI sigue siendo obligatoria.
- La lista de `tests-puros` aparece dos veces en el workflow: corrida y paso
  de piso. El mismo archivo de prueba debe quedar en ambas listas.

## Pendientes y alternativas descartadas

Falta CI verde para el SHA final, una reauditoría de escalón 3 del delta desde
el último SHA auditado `44864db9ab65ecba3dd6288614c682328e372ba6`, y
la verificación de integración antes de cualquier merge. No se desplegó.
F2-E tramo 2 permanece en PRs separados. Se descartó bajar el piso para
acomodar el fallo de permisos local: no mide el runner y ocultaría una
regresión. También se descartó mezclar #344 con SR2 o con tramo 2: sus pares
y auditorías deben cerrarse sobre SHAs propios.
