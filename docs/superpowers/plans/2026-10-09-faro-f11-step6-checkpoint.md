# Faro F1.1 · Paso 6: permiso de un solo uso y checkpoint durable

## Objetivo

Completar el límite durable de §10–§11 del diseño aprobado: las decisiones, los
`RulePermit` y los consumos forman una cadena de auditoría; sus escrituras usan una
transacción MariaDB, y el resultado no se reconoce hasta que un checkpoint externo
monotónico ancla exactamente el head confirmado.

La semántica de `OVERLAY_ISSUED` queda fuera de este cambio. El kernel no consulta ni
interpreta overlays, cuarentenas ni ratificaciones de corpus. El permiso requiere el
grant individual vigente que resulte del replay verificado del ledger Block 4. La
integración final queda pendiente de la respuesta de Fernando sobre overlays.

## Restricciones

- Base de implementación: `master@bf6b05809e1c3505efd7447b33685796ca42e3b6`.
- No modificar migraciones publicadas; agregar migración nueva si el esquema lo exige.
- Reutilizar el protocolo de lock y el orden de adquisición ya fijado por providers.
- El audit head DB y el checkpoint externo deben coincidir antes de mutar.
- Tras commit, publicar y releer el head externo exacto antes de responder éxito.
- Un checkpoint ausente, truncado, adelantado, contradictorio o no durable cierra emisión y consumo.
- No conectar el kernel a dispatch, procesos, red ni efectos externos.
- No publicar ni integrar mientras la decisión de overlays esté pendiente.

## Fases TDD

1. Escribir pruebas del checkpoint externo: genesis explícito, CAS de anterior exacto,
   idempotencia exacta, monotonicidad, detección de truncamiento/filas parciales y
   confirmación de cualquier head ya anclado. Ejecutar y confirmar que fallan.
2. Implementar el checkpoint durable como log canónico encadenado, con lock estable,
   `fsync` de archivo/directorio y error tipado para resultado de publicación desconocido.
3. Probar el almacenamiento de auditoría: append transaccional bajo audit-head lock,
   verificación de cadena, y ningún éxito cuando falla el anclaje externo.
4. Añadir modelos/proyecciones canónicas de permiso y consumo; cubrir hashes, round-trip,
   procedencia del store e idempotencia exacta por `request_id`.
5. Añadir modelo draft y objeto `RulePermit` con hash de dominio, proyección cerrada y
   sello del store; un draft del caller nunca sirve para consumir.
6. Implementar persistencia atómica de decisión `PERMIT` + permiso y el consumo con
   `permit_id` único; verificar carrera de doble consumo y orden con el head Block 4.
7. Implementar kernel `evaluate`/`consume` con providers y leases ya existentes, sin
   wiring operativo; cubrir negativas, STOP, expiración, cambio de pin/capability, fallo
   del checkpoint y resultado idempotente.
8. Añadir migración incremental necesaria y tests MariaDB de triggers, índices con
   `EXPLAIN`, rollback y fallos inyectados.
9. Ejecutar suites relacionadas, registrar limitaciones del runner local, revisar diff,
   delta-auditar el SHA exacto y dejar PR listo sin integrar.

## Decisiones técnicas que deben quedar comprobables

- El checkpoint externo es un ancla local duradera; no sustituye la cadena de auditoría
  DB ni el checkpoint Block 4.
- El orden de locks debe seguir: leases pin → capability → STOP → permit → authority
  ledger head → audit head. Todo adapter de checkpoint evita invertirlo.
- El retry de emisión recupera solo el permiso cuyo registro y checkpoint exactos ya
  estén confirmados. No emite otro permiso para el mismo request.
- Un fallo después del commit deja la operación no reconocida y bloquea nuevos cambios
  por mismatch hasta reconciliación explícita; no se autocorrige el checkpoint desde DB.

## Estado

- 2026-10-09: rama aislada creada desde master `bf6b0580`.
- Incremento local: `RuleAuditCheckpointStore` con genesis explícito, CAS, log
  hash-chained, exclusión `flock`, reemplazo atómico, `fsync` y relectura. La escritura
  existente de decisiones puede usar el checkpoint: bloquea archivo externo antes de
  MariaDB, exige head coincidente antes de mutar, confirma después del commit, y niega
  mismatch o publicación incierta.
- Verificación inicial antes del incremento de permisos: tests de checkpoint, providers y
  modelos, 289 passed; MariaDB coleccionaba 14 pruebas y Docker no podía abrir
  `/var/run/docker.sock`.
- En progreso local: se añadió migration 002 y el adapter MariaDB para emitir decisión
  `PERMIT` + `RulePermit` en una transacción, junto con publicación del head externo; el
  store en memoria prueba el boundary sellado e idempotencia. Las suites unitarias cercanas
  pasan (300 tests); MariaDB solo colecciona 17 tests porque Docker no puede abrir su socket.
- Bloqueo técnico abierto antes del wiring: la emisión todavía no comparte la misma
  transacción/row lock con el ledger Block 4. Resolver el contrato de conexión y orden de
  locks, implementar consumo y kernel, y validar en MariaDB CI.
- GLM/ZCode tiene auditoría read-only en curso sobre `76c8ecd0`; no cubre los cambios
  posteriores. La semántica `OVERLAY_ISSUED` permanece como decisión directa pendiente de
  Fernando y bloquea integración, no el trabajo aislado.
