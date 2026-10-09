# Faro F1.1 · cierre de contrato loader y ratificación

Fecha: 2026-10-09 · host `hall9000` · ejecución Codex, revisión consultiva Kimi y auditoría Tier 3 Sol.

## Estado observado

El PR #379, head `a678b75ad896b648e0f89ee3532ea225fda3d05d`, recibió rechazo Tier 3 exacto. Se reprodujo que las fábricas `_from_validated_snapshot` de `ValidatedCandidateCorpus` y `AuthorityEventIntent` permitían acuñar sellos con valores arbitrarios; el intent podía llegar a append firmado. También se confirmó una separación entre validar el candidate y volver a leer sus campos al construir la vista.

La corrección local en curso elimina ambas fábricas sellantes, deja el sello del candidate en el loader después de C14N/3 y captura los campos una sola vez al construir la vista. Se añadió el writer `append_ratification_from_candidate`; el append genérico rechaza intents `RATIFICATION_GRANTED` construidos por caller. `copy`/`deepcopy` retornan la instancia inmutable; pickle falla cerrado. El piso loader-seal pasó de 17 a 18.

Kimi confirmó los hechos pero consideró menor su severidad por el TCB en proceso. Sol confirmó el rechazo para el SHA auditado, el bypass de ambas fábricas y la carrera; determinó que Python hostil dentro del mismo proceso está fuera del threat model escrito. Se alineó el comentario del modelo con ese límite. No hubo contradicción factual que requiriera apelación.

Verificación local reportada por el implementador: 56 pruebas focales y 126 resolver/ledger sin MariaDB pasan; `py_compile` y `git diff --check` pasan. No se pudieron ejecutar las pruebas MariaDB porque la cuenta local no tiene permiso sobre `/var/run/docker.sock`; quedan para CI con Docker. Todavía no hay SHA de commit corregido, auditoría nueva ni integración.

## Dependencias y pendientes

- Auditar y cerrar la corrección de #379 sobre su SHA final, con CI verde incluyendo MariaDB.
- #373 fue rechazado en `ce0043bff28957efd337ca715c0323e534b4bd2f`: se está corrigiendo una subclase hostil de `str` que falsea la comparación de procedencia del pin. Reapilar y auditar #373 después de #379.
- #371 tiene que reapilarse sobre #373 y el arreglo de cola; #375 y #378 dependen del evaluador final.
- #381 tiene un hallazgo previo de rollback de checkpoint y una decisión de semántica de `OVERLAY_ISSUED` aún pendiente de Fernando.
- GLM/ZCode no devolvió revisión: la petición terminó por error de red del proveedor. No se considera una aprobación.

## Decisión técnica

Quitar del modelo las fábricas generales que convertían datos de caller en objetos sellados. El límite es la API soportada dentro del TCB Python; no se presenta el sello como protección contra ejecución hostil dentro del mismo intérprete. La firma Ed25519 y el replay completo siguen siendo los controles criptográficos.

## Responsable

Fernando pidió a Codex liderar el cierre de Faro F1.1 hoy. La revisión Kimi fue consultiva; Sol emitió la auditoría de escalón 3. Codex implementa, verifica y coordina la secuencia.

## Actualización · bloqueo Tier 3 sobre `bc4089f`

El auditor detectó que la prueba MariaDB de round-trip aún enviaba el primer
`RATIFICATION_GRANTED` al writer genérico. Se cambió esa primera escritura a
`append_ratification_from_candidate`, con el mismo candidate usado para
construir el overlay; las siete escrituras restantes siguen por el writer
genérico, conservando la cadena de ocho eventos.

La lista exacta de Identity Foundation Shadow se volvió a medir con el comando
del workflow: `677 passed in 4.43s`, cero skipped. El incremento desde 674 es
de tres pruebas: una de `copy`/`deepcopy`/pickle del loader (el archivo ahora
tiene 18) y dos regresiones del writer de ratificación. Por ello el piso se
actualizó a `^677 passed in ` en `ci/pisos.json`, el workflow y esta memoria.
La integración MariaDB sigue pendiente de CI: el host local no permite acceder
al socket Docker.
