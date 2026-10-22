# B9: disposición humana final y límites de procedencia

**HISTORIA / 28-sep-2026 (America/Tegucigalpa).** Este es un registro
aditivo de una determinación de Fernando, Human Authority de JAX/Axioma. No
concede capacidad automática, no certifica el estado presente y no reemplaza
los eventos canónicos ni las fuentes operacionales vigentes.

## Disposición de reconciliación B9-2026-09-28

Fernando revisó la reconciliación forense de la adopción manual de memoria
legacy y de las tres ejecuciones manuales de embeddings del 26-sep-2026.

- **Execution intent:** acknowledged by Human Authority after the fact.
- **Technical authorization:** valid at execution time.
- **Contemporaneous human authorization evidence:** absent.
- **Forensic origin:** fully reconciled.

La determinación es un **POST-HOC HUMAN ACKNOWLEDGMENT**. No afirma que
existiera evidencia contemporánea de autorización explícita y no convierte la
aceptación posterior en autorización histórica.

Los clusters afectados son la adopción manual de 136 fuentes legacy y las tres
ejecuciones manuales de `jax-memory-embedding.service` que persistieron 137
generaciones. La reconciliación observó 137 objetos, 273 revisiones, 410
eventos, 137 proyecciones, 136 bindings legacy, un espacio de embeddings y 137
generaciones. Sus cadenas son un `CREATE -> RE_EMBED` y 136
`IMPORT_LEGACY -> RE_SCOPE -> RE_EMBED`.

No se reescribe la procedencia histórica: este registro no modifica
`memory_events`, no retroalimenta metadatos de autorización y no agrega una
migración. La evidencia forense conserva la distinción entre el actor técnico
registrado y la ausencia de prueba contemporánea de intención humana.

## Identidad de embedding

El espacio productivo observado mantiene `model_version_or_digest = NULL`.
El digest de Ollama observado durante prueba controlada,
`7907646426070047a77226ac3e684fbbe8410524f7b4a74d02837e43f2146bab`, no se
aplica ni se atribuye retroactivamente a ese espacio.

Si en el futuro una versión o digest concreto pasa a formar parte de
`EmbeddingSpaceIdentity`, debe derivar un **nuevo espacio inmutable**. Nunca
se reescribe en sitio la identidad existente `NULL`/`UNKNOWN`.
