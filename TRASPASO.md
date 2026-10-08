# Traspaso — Faro F1.1 RULE AUTHORITY KERNEL

## Objetivo

Registrar las decisiones de implementación del contrato F1.1 aprobado por Fernando
el 2026-10-05 y el estado verificable de esta ronda. Este archivo es un handoff;
no concede autoridad ni prueba el estado operativo actual.

## Estado verificado al 2026-10-07 · Hall9000

- La spec F1.1 está en
  `docs/superpowers/specs/2026-10-05-faro-f1-1-rule-authority-kernel-design.md`;
  su sección 16 registra nueve decisiones de implementación.
- PR #376 (`a3bf02c4`) se integró por solicitud explícita de Fernando en su rama
  padre `feat/faro-f1.1-evaluador` mediante el merge
  `5cfc44c6610c32f036cd4e978e88195dd5f91520`. Esta rama forma parte del PR #371;
  el merge de #376 no pone por sí solo el código en `master`.
- PR #382 integró la alineación documental en `master` mediante el merge
  `1f9c2fa5eb8ec297bed7e70fa43dbef1e2ef52a5`. Su CI `push` sobre el merge terminó
  verde y el `post-merge-guard` verificó el SHA, el tip de `master` y la ausencia
  de carrera. El guard primero encontró una lectura inestable mientras nacía CI;
  se repitió tras finalizar los jobs y entonces confirmó éxito.
- Al verificar este handoff, los PRs de código F1.1 #370, #371, #373, #375 y #378
  seguían abiertos y apilados. La cadena de código aún no estaba integrada en
  `master`; comprobar sus refs y CI actuales antes de retomarlos.
- La auditoría inicial de #382 detectó que un handoff anterior decía que #376
  necesitaba sincronizarse y que #382 no debía integrarse. Ese texto fue corregido
  en #382 antes de su merge; la copia de este archivo actualiza el estado posterior.

## Decisiones y límites

- Faro decide permisos antes de ejecutar; el aviso comunica una decisión y no
  modifica autoridad.
- No inferir que una PR apilada quedó en `master` solo porque se integró en su
  rama padre.
- No reutilizar auditorías de un SHA anterior tras modificar el PR.

## Reanudación 2026-10-07 · actualización de base de #370

- Para satisfacer el preflight, se incorporó `master@a5460c886e8ff0fde447aff8a0adf97b07b89761`
  al head anterior de #370 `28f1eac7fbb6523cf569eb53bd236a28fc228a10`.
- El merge no produjo conflictos de código; tocó `TRASPASO.md` y añadió la
  spec de decisión de #376 que ya estaba en master. El SHA resultante del merge
  es `ce614b667d002545d82d4df0ed4d169d89cca69e`.
- Aún no integrar: falta publicar el SHA actualizado, esperar CI exacta y pedir
  auditoría escalón 3 del nuevo SHA. La auditoría aprobada de `28f1eac7` no cubre
  este head actualizado.

## Rechazo y cierres de auditoría 2026-10-07/08

- La auditoría escalón 3 de `d0206d029b9e7e3930c1a0f09d5bfd9a1595fb9f` fue
  **RECHAZADO**: BLOCK por trees sueltos adulterados bajo el mismo OID; MAJOR por
  atribución incompleta del catálogo y discrepancia JSON/Python; MAJOR por falta
  de presupuesto agregado; MINOR por piso que toleraba pruebas saltadas y escape
  `\010` accidental en el hash.
- Cierres implementados: commit y árboles raw se verifican con `git hash-object`
  y se interpretan desde esos mismos bytes; sin `ls-tree` en el camino confiable;
  SHA-1/SHA-256 del repositorio; sin lazy fetch; límites de 1.024 objetos,
  profundidad 16, 8 MiB agregados y 1 MiB por blob, comprobados antes de cargar
  el lote; salida de `cat-file` validada y acotada.
- `tope:null` conserva semántica de ausencia; JSON Schema y Python comparten seis
  vectores, incluidos casos OBLIGATING positivo/negativo, periodos y catálogo.
  `Topes` lleva el `catalogo_oid` sellado en resultados y eventos, reservado
  contra suplantación por contexto.
- La serialización usa `catalogo-topes.json\0` + `100644\0`; se actualizó el
  vector dorado. Identity Foundation usa el piso exacto de 643 y rechaza
  cualquier `skipped`; el piso de Faro pasa 726→728 por dos pruebas.
- Verificación local actual: snapshot+ataques+schema → **307 passed**;
  `compileall`, JSON Schema meta-valid y `git diff --check` limpios. No se pudo
  colectar localmente `tests/test_faro_topes.py` porque este intérprete no tiene
  la dependencia `mcp` ya instalada; el workflow de CI instala las dependencias
  declaradas y ejecutará la suite completa.
- Pendiente: documentar cierres en README/spec, archivar y quitar este traspaso
  antes de la auditoría final, publicar un SHA candidato, esperar CI verde del
  SHA exacto, auditarlo en escalón 3 y luego integrar #370. No reutilizar el
  veredicto sobre `d0206d02`.


## Cierre del handoff · 2026-10-08

- Cierres tras la auditoría RECHAZADO de `d0206d02`: autenticación de objetos
  commit/tree/blob desde sus bytes, presupuestos globales, procedencia del
  catálogo y paridad de esquema Python/JSON; vector de serialización corregido.
- Verificación local actual: snapshot+ataques+schema, **307 passed**;
  `compileall` y `git diff --check` limpios. La CI completa y la auditoría
  Tier 3 del SHA final aún son necesarias antes de integrar.
- Este archivo se archiva antes de la auditoría final según el runbook de
  traspaso continuo.


## Reanudación tras auditoría de 68f10eae · 2026-10-08

- La auditoría Tier 3 confirmó BLOCK: la lectura tenía precheck de tamaño y luego
  `subprocess.run(capture_output=True)` materializaba otro batch; un objeto suelto
  cambiado entre ambos procesos podía exceder el presupuesto antes de rechazo.
- Corrección local: un solo `cat-file --batch` con stdout sin buffer; la cabecera
  real define tipo y tamaño y se compara contra presupuesto por objeto y agregado
  antes del body. Errores matan y esperan el proceso; stderr va a DEVNULL; se
  comprueban separador, salida sobrante y OID. Commit, trees y blobs pasan su
  presupuesto restante a esta lectura. Regresiones RED/GREEN para blob, commit y
  tree.
- La auditoría también encontró que el espejo JSON Schema no expresa fechas
  imposibles ni compara extremos. Arquitectura concluyó que el validator Python
  es la única autoridad vigente: añadí `format: date-time` y pruebas con checker
  explícito para calendario; documenté que la comparación `not_after >
  not_before` solo la garantiza `validar_regla()`.
- Verificación focal actual: snapshot+ataques+schema **312 passed**;
  `comparar_pisos.py origin/master` OK; `compileall`, meta-schema y `git diff
  --check` están pendientes de confirmar después del último ajuste.
- Este worktree vuelve a tener `TRASPASO.md` hasta el cierre; publicarlo, esperar
  CI, recibir auditoría Tier 3 del SHA final y archivar/quitar el handoff antes
  de esa auditoría. El head publicado remoto todavía es `68f10eae`; no audita
  los cambios locales.


## Deadline del lector de objetos · 2026-10-08

- Auditoría adversarial señaló que streaming aún podía bloquearse si `cat-file`
  entregaba cabecera y retenía la tubería. Corregido con deadline monotónico de
  120 s para el batch, `select` + `os.read` no bufferizados en las lecturas,
  deadline al escribir, stderr a DEVNULL y kill/wait al timeout/error.
- Regresión ejecuta proceso real que emite cabecera y retiene body; a 20 ms debe
  expirar, matar y esperar. Verificación completa focal snapshot+ataques+schema:
  **313 passed**; compileall, JSON Schema meta-valid, comparador de pisos y
  `git diff --check` limpios.
- Ramas locales aún no contienen este fix; publicar commit de trabajo y pedir
  auditoría nueva del SHA final. Luego esperar CI exacta; antes de la auditoría
  final, archivar el handoff actualizado y quitar `TRASPASO.md`.
