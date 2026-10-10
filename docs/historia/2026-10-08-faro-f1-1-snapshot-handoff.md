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

## Fallos de CI y corrección final · 2026-10-08

- CI del SHA `137ef5d568ec35d475bc0c44c5a5ea65cc59ce6a` reportó dos defectos:
  `no-fail-open-except` exigía documentar tres `except` de limpieza como fail-soft;
  `faro-fase0` fallaba porque el fake `AlmacenMemoria` no implementaba `leer`,
  aunque `Topes.reconciliar()` y `AlmacenMariaDB` sí dependen de esa lectura.
- Se declararon las tres razones de cleanup junto a sus `except`; se completó el
  protocolo `AlmacenTopes.leer()` y el fake, incluyendo los modos de error y timeout.
- Verificación local: suite Faro fase 0 completa **728 passed** (Python 3.12,
  dependencias instaladas desde requirements fijados con hash); controles de
  snapshot/schema/ataques **313 passed** (Python 3.14, Unicode 16); control de
  excepts **21 passed**; `git diff --check` limpio.
- El nuevo candidate aún no está comprometido ni publicado. Cometer código con
  este handoff, luego archivar su contenido y retirar `TRASPASO.md`, publicar el
  SHA resultante, repetir auditoría Tier 3 exacta y esperar CI verde antes de
  integrar #370.

## Contrato de lectura completado · 2026-10-08

- La revisión Tier 3 de `92c13ff2471aa4cbbd518542d7a0eb6eb15d2682` detectó que
  `AlmacenTopes` (protocolo de producción) tampoco declaraba `leer`, aunque
  `Topes.reconciliar()` lo llama y el almacén MariaDB ya lo implementa.
- Se añadió una regresión que comprobó RED (`AlmacenTopes.leer` ausente); el
  protocolo ahora declara la lectura y la prueba, junto con `tests/test_faro_topes.py`,
  pasa: **113 passed**.
- El PR remoto ya recibió `92c13ff`; este nuevo cambio vuelve a cambiar el SHA.
  Falta cometerlo con este handoff, retirar/archivar el handoff actualizado,
  publicar el SHA final, pedir auditoría exacta y esperar CI verde.

## Piso exacto de CI · 2026-10-08

- Auditoría de `e941c3b59befa6347bcb126fd85deaa66f3e05c1` detectó que la nueva
  prueba del protocolo suma uno al set Faro y el piso permanecía en 728.
- Re-medición de la suite completa Fase 0 con Python 3.12 y requirements fijados
  por hash: **729 passed**. `ci/pisos.json` y el comentario del workflow quedan
  actualizados a 729; `comparar_pisos.py origin/master` confirma que no baja
  ninguno de los pisos heredados.
- Este SHA todavía debe probarse y auditarse en CI; el nuevo trabajo requiere
  archivar y retirar el handoff antes de pedir auditoría final exacta.

## Desglose del piso revisado · 2026-10-08

- Auditoría de `6699c6d4fcc49a1477284e955826e7c51a64f350` confirmó que el
  comparador y piso 729 son correctos; encontró que el comentario aún atribuía
  109 pruebas a `topes`, mientras que su suite mide 113.
- El comentario se ajustó a 113; el desglose ahora suma 729. Este cambio crea un
  nuevo SHA, que requiere auditoría y CI exactas después de archivar este handoff.

## Justificación de stderr acotado · 2026-10-08

- CI exacta de `5d72b97e23603ad5ec40e0b7e1bd8fe97978e4ba` pasó `no-fail-open-except`
  pero falló `test_stderr_no_se_tira`: `jax/faro/git_objetos.py` usa
  `stderr=DEVNULL` sin motivo en el registro controlado.
- Añadido el permiso con justificación específica: no se captura la salida del
  recorrido Git, el código de salida se valida y el caller falla cerrado con error
  tipado. Control local: **4 passed**, diff-check limpio.
- Falta cometer, archivar/retirar handoff, publicar el nuevo SHA, auditarlo y
  esperar CI exacta.

## Desfase de piso Identity Foundation Shadow · 2026-10-08

- CI del SHA `49b0e1bbf1fe1b735f9dfd660240b0a5c68b9c8f` ejecutó 657 pruebas,
  pero `ci/pisos.json` todavía exigía 643. El conjunto incluye las nuevas pruebas de
  schema/snapshot Faro F1.1.
- Se actualizó el piso a 657 y el comentario de `policy.yml` a Ronda 12 (+14).
  La ejecución exacta local, con Python 3.14.4 y la misma lista de pruebas del
  workflow, dio `657 passed, 1 warning`; el matcher acepta el conteo con o sin
  advertencias. El workflow ya verifica cero skipped por separado.
- `python3 .github/ci/comparar_pisos.py origin/master` pasó: 50 pisos y 1 mínimo
  heredados sin bajar, 51 pisos totales en el head.
- Este cambio aún necesita auditoría Tier 3 exacta, CI completa verde y el protocolo
  de integración antes de integrar #370.

## Matcher del piso sin prefijos permisivos · 2026-10-08

- Auditoría Tier 3 de `347b1e9adcff9f72ea397cb11c082813817c1d2a` rechazó el matcher
  `^657 passed`: aceptaba resúmenes con `xfailed` o `xpassed`.
- Se restauró el matcher estricto `^657 passed in ` y se silenció únicamente
  `PytestCollectionWarning` para `TestEvidenceIngester`, una clase auxiliar que pytest
  anunciaba como no coleccionable. La lista exacta del workflow ahora da
  `657 passed in 4.73s`; el piso y la verificación independiente de cero skipped pasan.
- Casos sintéticos del piso: acepta `657 passed in ...`; rechaza `657 passed` con
  `xfailed`, `xpassed` o `skipped`. El comparador contra `origin/master` no baja pisos.
- El SHA previo no se debe integrar: esta corrección exige nuevo commit, auditoría Tier 3
  y CI completa exacta antes del merge.

## Corrección: advertencia de colección sin filtro global · 2026-10-08

- Auditoría Tier 3 rechazó la supresión global `-W ignore::pytest.PytestCollectionWarning`:
  habría ocultado advertencias futuras de clases mal nombradas.
- Corrección aplicada: `policy/enforcement_evidence/test_evidence.py` marca
  `TestEvidenceIngester.__test__ = False`, porque es una clase auxiliar importada en
  la suite y no una clase de prueba. Se eliminó el filtro global. Cualquier otra
  advertencia de colección sigue visible y hace que falle el piso exacto.
- La lista exacta de Identity Foundation Shadow pasó localmente: `657 passed in 5.23s`;
  cero skipped, comparador contra `origin/master` y `git diff --check` limpios.
- Ese SHA fue auditado, integrado como PR #370 y pasó `post-merge-guard`; el
  recibo final quedó en el registro de integración de esa fecha.

## Archivo del traspaso raíz · 2026-10-08

- El traspaso operativo que reapareció al actualizar la rama de #377 sobre
  `master` se retiró del cambio candidato. Era una instrucción de continuación
  ligada a un worktree y a SHAs ya superados; conservarlo en `master` habría
  publicado un pendiente falso.
- Este documento conserva el contexto histórico de Faro F1.1. El estado vivo de
  cualquier PR o CI debe consultarse en sus refs, checks y auditorías del SHA
  exacto; este archivo no es autoridad operativa.

## #379 · sello del loader de autoridad · 2026-10-08

- #379 se abrió apilado sobre el head anterior de #377. Tras integrarse #377, su
  PR se cambió a `master` y la rama incorporó el nuevo tip antes de la revisión
  final.
- La CI anterior detectó una carrera en `_ephemeral_mariadb`: la imagen publica
  un socket del servidor temporal mientras inicializa, antes de iniciar el
  servidor definitivo. La espera ahora exige el fin de inicialización del
  entrypoint y una conexión autenticada con `SELECT 1`; las reconexiones son
  acotadas y solo aplican a errores de socket/servidor transitorios. Fallos de
  credenciales como `1045` se propagan de inmediato.
- Tres pruebas unitarias verifican clasificación, reintento y propagación de
  errores permanentes: `3 passed`; `py_compile` y `git diff --check` pasaron.
  No se pudo ejecutar MariaDB localmente por falta de permiso al socket Docker;
  la integración real se comprueba en CI.
- En el primer candidato de #379, la lista exacta de Identity Foundation
  Shadow creció de 657 a 673 al incluir 16 casos del loader en ambos pasos.
  La auditoría del SHA posterior encontró una regresión 17 del loader y exigió
  corregir ambos pisos: loader 17 e Identity Shadow 674. Ese estado posterior
  se registra en `docs/ci/pisos.md`; esta nota conserva solo la medición histórica
  del primer candidato, no una afirmación del piso vigente.
- CI del SHA `3371ecd0` reveló que MariaDB 12.3.3 registra `ready for connections`
  para el servidor de inicialización en `port: 0` antes del servidor final en
  `port: 3306`; esperar solo `init process done` vencía aunque el servidor final
  ya estuviera listo. La detección ahora reconoce el marcador de init o la pareja
  ready+port 3306, luego exige conexión autenticada y `SELECT 1`. Piso de integración
  MariaDB: 8 -> 9. La regresión unitaria del detector pasa localmente; CI del nuevo
  SHA debe volver a medir las nueve pruebas y el resto del workflow.
- El `TRASPASO.md` operacional de la rama se retiró del candidato para evitar
  publicar instrucciones de una sesión ya terminada. Este registro guarda el
  contexto histórico; no declara estado vivo de PR, CI ni autoridad.

## #371 · auditorías y piso de Fase 0 · 2026-10-09

- Hubo dos auditorías Tier 3 independientes sobre el SHA exacto
  `bf0a11d940af4c67a6f8b015fc454c005685316e`. `audit_373_final` informó
  APROBADO CON CAMBIOS, 0 BLOCK, 0 MAJOR y 2 MINOR: escaneo de resumen O(n)
  fuera del flock y falta de presupuesto estructural futuro para
  `RuleEvaluationRequest.arguments` ([veredicto](https://github.com/fjruizhn/Jax/pull/371#issuecomment-6091232740)).
  `audit_faro_heads` informó APROBADO funcional, 0 BLOCK, 0 MAJOR y 3 MINOR:
  scan O(n) sin p95/RSS; `tests/test_faro_aviso.py:1081–1124` no aísla reinicio
  y procesos cortos ni LRU > 32; y `test_faro_rule_authority_models` no invoca
  el evaluador/wiring futuro ([veredicto formal](https://github.com/fjruizhn/Jax/pull/371#issuecomment-6092620311)).
  Ambos informes comparten el hallazgo O(n); los otros difieren. Una nota
  intermedia parafraseó el tercer punto de `audit_faro_heads` como una
  afirmación futura sobre `store.record()`/lease; el comentario formal precisa
  la falta de invocación del evaluador/wiring ([nota de coordinación](https://github.com/fjruizhn/Jax/pull/371#issuecomment-6091653546)).
- La CI de `bf0a11d…` midió 780 pruebas frente al piso anterior de 729; falló
  `faro-fase0` únicamente por ese desfase, mientras `tests-puros` pasó. El
  cambio `c86582af06e62888df649658300600e6ce5ccf14` actualizó
  `ci/pisos.json` y `docs/ci/pisos.md` a 780. Su delta-audit exacta fue
  APROBADO CON CAMBIOS, 0 BLOCK/MAJOR y funcionalmente correcto; la CI de ese
  SHA terminó verde. GitHub registra #371 integrado en `master` como
  `8c850bcec5cd0233db5487343210c0ba96689f97`.
