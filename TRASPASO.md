# Traspaso Codex · Faro F1.1 · 2026-10-09

## Estado

- #379 fue aprobado por auditoría Tier 3 e integrado como `d0c8a68f131849fb298c8cda9b5519e304088e11`; su `post-merge-guard` confirmó CI verde y sin carrera.
- #373 fue aprobado por auditoría Tier 3 sobre `42b05c025d452cd163b1a5bf463661e6c1f197f6`, pasó todos los checks y se integró como `35991faa81b2d7ac5d85a5bedf67eea168e98cd7`. `post-merge-guard` confirmó `master@35991faa` y ninguna carrera.
- Esta reconstrucción de #371 contiene modelos/store en memoria, exports y pruebas, más los cambios de cola en `jax/faro/aviso.py` y `tests/test_faro_aviso.py`. No agrega evaluador ni modifica MariaDB de autoridad. El test de contrato comprueba que el futuro llamador mantenga `store.record()` dentro de `leases_de_emision(pin → clasificación → STOP)` y que STOP siga bloqueado hasta que retorne; no hay cableado de producción a un evaluador en este paso.
- La rama local `feat/faro-f1.1-paso4-rebuild` quedó rebasada sobre `master@35991faa`. El commit `3d08076c` corrige los MAJOR de #371 en la cola de resumen diario: presupuesto agregado de bytes/segmentos/candidatos/líneas/mensajes, lock-file regular del usuario con modo 0600 y apertura no bloqueante, y prueba dinámica que demuestra que la lectura no retiene el candado. Si se parte un segmento, el resto se sincroniza y publica antes del prefijo; un crash puede reentregar, pero no perder avisos. Verificación local exacta: `tests/test_faro_aviso.py` → `115 passed`; `py_compile` y `git diff --check` limpios. El piso Identity Shadow exacto del workflow sigue siendo `968 passed`, cero skipped, porque los archivos medidos no cambiaron.
- `tests/test_faro_requirements.py` ya está en el job Faro de `.github/workflows/policy.yml`; no falta portarla.
- #381 tuvo veredicto RECHAZADO sobre `4d493b988c2cb116c6236ca868a52aa2da90f413`: falta conservar el writer sellado de #379 al reconstruir su rama, y el replay debe cotejar cada fila de checkpoint positiva con el evento correspondiente del ledger.
- Fernando decidió conservar la semántica de #381: `OVERLAY_ISSUED` requiere ratificación activa, no revocada, del mismo corpus.
- #375 y #378 siguen apilados sobre evaluador/step4; rebasar después de cerrar #371.

## Evidencia local de la reconstrucción original

- Modelos + providers: 283 passed.
- Avisos con dependencias Faro fijadas bajo `/tmp/jax-faro-testdeps`: 111 passed.
- `py_compile` y `git diff --check` limpios.
- Identity Shadow combinado con #373 y #371: 968 passed, cero skipped, medido desde el comando exacto del workflow con `python3` (el host no expone alias `python`).

## Siguientes pasos

1. Confirmar coordinación del head remoto viejo de #371 (`c36c0543`) y actualizar el PR con esta reconstrucción, preservando el `master@35991faa` exacto.
2. Obtener auditoría Tier 3 y CI verde para el SHA exacto de #371; después integrar mediante preflight y post-merge guard.
3. Reconstruir #381 sobre el nuevo master, cerrar los dos BLOCKs, mantener la compuerta de ratificación aprobada por Fernando, medir pisos, auditar y completar CI.
4. Reapilar #375 y #378 sobre #371 integrado; auditar y completar CI de cada PR en orden.

No desplegar. No declarar cerrado Faro F1.1 hasta completar #371, #381, #375 y #378.
