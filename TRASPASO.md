# Traspaso · fix/authority-resolution-loader-seal

## Objetivo

Cerrar en Block 3 el mismo patrón que la falla 1 de #377 (hallazgo del auditor
de #377, fuera de alcance allí): `ValidatedCandidateCorpus._loader_seal` era
`init=True` y `dataclasses.replace(corpus, ...)` conservaba el sello — un corpus
alterado pasaba por validado. Encargo:
`~/encargos-codex/encargo-glm-g2-loader-seal.md`.
Base: `origin/fix/authority-ledger-sello-y-append` @ d082cc89 (#377 + cierre de
auditoría de Hyde).

## Hecho en esta rama

- `policy/authority_resolution/models.py`:
  - `_loader_seal` pasa a `init=False`: ni el constructor ni `replace` lo
    transportan; solo lo estampa `_from_validated_snapshot` (el loader).
  - Nuevo `_content_binding` (`init=False`): digest del contenido congelado
    (dominio `JAX-VALIDATED-CORPUS-CONTENT`) estampado por el loader;
    `_was_loader_validated()` lo RE-DERIVA y compara — comparación de campos
    honesta, NO recalcula el `policy_corpus_hash` de C14N/3 (cubre archivos del
    snapshot, no el modelo). Todo consumidor existente (adapter,
    `ratification_intent_from_candidate`) lo hereda sin cambios.
  - Congelado profundo: las colecciones de todo el árbol (scope, relationships,
    precedence, normative_sources, protected_metanorms, manifest.members,
    normative_documents) entran como tuple/list y se guardan SIEMPRE como tuple,
    desacopladas de la fuente; `normative_documents` exige FrozenNormativeDocument.
  - `__post_init__` ya no exige el sello (no puede verlo): la puerta es el
    consumidor, como en #377 tras el cierre de Hyde.
- Tests: `test_authority_resolution_loader_seal.py` (6 ataques: replace con
  hash alterado, replace con contenido alterado, construcción directa,
  congelado profundo con listas desacopladas, drift de contenido detectado por
  el digest, positivo del loader con el corpus real);
  `candidate_boundary` ajustada (la construcción directa ya no rechaza en
  constructor: produce un objeto que todo consumidor niega).
- Bytes firmados INTACTOS: golden de #377 verde y diff vacío del corpus real
  (hash/view/intent) contra d082cc89.

## Mutantes muertos

| Mutante | Fallos | Aserto que explota |
|---|---|---|
| «init=True» (sello vuelve a viajar) | 2 | `forged._loader_seal is None` ya no; replace transporta el sello |
| «sin congelar» (_tupla_congelada no coerce) | 1 | el corpus guarda la lista viva: isinstance tuple / desacople explotan |
| «sin comparar hash» (_was_loader_validated solo sello) | 1 | el drift de contenido pasa por válido |

## Pisos

- Los cinco de #377 verificados con `piso.py verificar`: codec 19,
  ratificaciones 12, seal 4, golden 7, integration (MariaDB efímera) 5 — sin
  cambios.
- Identity Foundation Shadow (sin piso propio en esta línea): A/B medido,
  344 (d082cc89) → **350** en la rama (+6 del archivo nuevo, cableado en la
  lista del paso).
- Guardas CI: wireados + bash válido + pisos fuera del workflow = 237 passed.

## Pendiente

- Auditoría de Hyde y merge (apilado sobre #377 — fusionar en orden).

## Ronda de CI #379 · 2026-10-08

- La CI anterior falló en
  `test_upgrade_migration_fails_closed_on_null_intent_even_with_permissive_sql_mode`:
  el socket Unix de la MariaDB efímera existía durante la inicialización, pero
  desaparecía antes de la primera conexión que usaba el caso de prueba.
- Causa trazada: la imagen inicia un servidor temporal para preparar su
  directorio de datos y luego lo sustituye por el servidor final. La sonda
  anterior comprobaba solo socket/una conexión, sin esperar el marcador del
  entrypoint `init process done`; por tanto podía devolver el factory durante
  esa transición.
- Corrección local pendiente de auditoría: esperar el marcador y `SELECT 1`
  del servidor final. El factory entregado después solo reintenta durante 10 s
  errores transitorios de socket o MariaDB (2002, 2003, 2006, 2013); permisos,
  autenticación y demás fallos permanentes se relanzan de inmediato. El timeout
  incluye el último error y la espera de readiness incluye los últimos logs del
  contenedor.
- Verificación sin Docker: tres pruebas unitarias RED→GREEN para clasificación
  restringida, reintento transitorio y propagación inmediata de `1045`; `3
  passed`, `git diff --check` limpio. En Hall9000 el usuario de esta sesión no
  puede acceder a `/var/run/docker.sock`, de modo que el ejercicio MariaDB real
  queda para CI.
- Antes de auditoría final: incorporar esta nota en el handoff histórico
  `docs/historia/2026-10-08-faro-f1-1-snapshot-handoff.md` sobre la base que ya
  lo contiene y retirar este traspaso operativo del candidato a `master`.
