# Fase 1 / Bloque 9 — remediation acotada

**Fecha:** 2026-09-24. **Tipo:** HISTORIA y PENDIENTE. **Decisión:** Fernando autorizó remediar exclusivamente B9-AUD-001..007, ejecutar sus regresiones y hacer después un targeted recheck. El último veredicto formal anterior a estos cambios era **AUDIT FAIL — REMEDIATION REQUIRED**.

## Qué se hizo y por qué

- B9-AUD-001 y 006: la síntesis conserva el alcance efectivo de revisiones fuente validadas dentro de la transacción; las pruebas comprueban el aislamiento entre proyectos y la ausencia de promoción a alcance de tenant.
- B9-AUD-002: `PromptMemoryContext.render()` serializa el contenido de memoria como un campo de datos, manteniendo los encabezados de confianza fuera del contenido.
- B9-AUD-003: `retrieve_authorized()` consulta con el alcance ya resuelto. Una regresión usa membresía activa y MariaDB; la lectura directa de proyecto sigue rechazada.
- B9-AUD-004: `CONTENT_PURGE` elimina contenido de revisiones históricas, payloads separables y vectores. La regresión inspecciona físicamente tablas B9 en una base temporal de CI.
- B9-AUD-005: el binding legacy usa tenant en mapa, consulta y claves; la migración 004 deriva el tenant del objeto ligado y falla si no puede determinarlo.
- B9-AUD-007: verificar exige un estado de ciclo de vida elegible y no revive memoria vencida o retirada.

La implementación inicial está en `f44ad93`; las pruebas con MariaDB real y las correcciones de sus guardas están en `f693a33`, `1bc9810`, `6101107` y `0e21027`. Los diagnósticos de CI están en `07c85d5` y `79b504f`; `ec93329` ajustó el piso del job de memoria tras la medición.

## Evidencia y lecciones técnicas

**HECHO verificado:** el runner del HEAD `79b504f` registró **186 passed** en la segunda corrida de `memory-vector-zero-io`; ese job falló porque todavía exigía el piso anterior de 159. La primera corrida de ese job pasó con MariaDB. El piso se cambió a 186 en `ec93329`; su CI de HEAD exacto aún debe confirmarse.

Las pruebas de integración deben usar `connect_timeout` y `exigir_base_de_test()`; dos controles existentes detectaron esas omisiones. Una tabla temporal de MariaDB puede desaparecer cuando se libera una conexión tras un `ALTER` fallido: la prueba del binding ambiguo ahora verifica el resultado en la misma conexión. Ninguna de estas observaciones equivale por sí sola al targeted recheck formal.

## Pendientes y límites

- **2026-09-24:** confirmar CI completo de los HEAD exactos de JAX y jax-platform, incluidos los jobs de memoria y backend con DB.
- **2026-09-24:** solicitar targeted recheck formal de AUD-001..007 sobre el par exacto de HEADs y registrar su veredicto.
- **Después del recheck y con la autoridad aplicable:** resolver los bloqueos de PR #269 y #155. No se hizo merge ni despliegue.

**Alternativas descartadas:** nueva auditoría general, reabrir B9-R0 y rediseñar la arquitectura; exceden la autorización de remediation acotada de Fernando.
