# Memoria B9: validación previa a reactivación

HISTORIA / Codex · 25-sep-2026 (America/Tegucigalpa). Fuente: encargo de
Fernando por la sesión Hyde, los dos archivos de encargo/auditoría conservados
en `b9-remediation`, SQL observado por principal y pruebas descritas aquí.
Este registro documenta observaciones; no concede autoridad ni certifica el
estado presente. Consultar las fuentes vigentes de la guía operacional.

El baseline a723964 reprodujo escrituras parciales, duplicación al reintentar,
lectura fallida marcada como procesada y bloqueo de conversaciones sanas por
una conversación inválida. Se decidió usar claims durables y salida congelada,
y confirmar lote B9, origen, processed y COMPLETED en una sola transacción.
El auditor independiente exigió además autoridad antes del proveedor, arranque
sin migración legacy y revalidación del dueño del proyecto al adoptar.

La principal restauró un respaldo completo de 62 tablas en un contenedor
MariaDB 12.3.3 sin red, con datos en tmpfs. El manifiesto y dump permanecen
protegidos en Hall9000. Conteos restaurados: facts151, decisions22,
action_items24, usuarios6, mensajes1658, conversaciones364 y objetos B9/eventos1.
No se considera este ejercicio una prueba de arranque completo de la aplicación.

El plan privado de adopción, creado con una conexión de sólo lectura, clasificó
197 filas:136 elegibles; cuarentena47 SUPERSEDED,5 EXPIRED,9 PROJECT_SCOPE_UNBOUND.
No se inventó dueño ni se publicó ninguna fila en producción en esta fase.
Los datos legacy y las exclusiones permanecen respaldados; la verificación
legacy no se transforma en verdad verificada B9.

Pruebas end-to-end: driver explícito contra una base fresca
`jax_memory_test_memb9_20260925235135`, usuario restringido `jax_test` y esquema
real con migraciones y triggers.19 escenarios PASS: atomicidad, reintentos,
concurrencia, ACK perdido, lectura fallida, cuarentena, leases/fuente cambiada,
adopción, aislamiento, síntesis verificada/revocada, embedding y lifecycle.
Los nueve módulos de regresión ejecutados sobre MariaDB real dieron102 passed,
0 skips. El sufijo explícito de la base impide que módulos importados elijan
una base automática distinta. Sólo el proveedor está simulado. No se usan credenciales productivas ni el
bootstrap de gobernanza de conftest para estas pruebas.

Rendimiento:1000 objetos canónicos,500 por tenant, pool8, limit20. Sobre la misma
base, p95 SQL original/nuevo con concurrencia1:4,77/0,56ms;10:9,23/3,01ms;
25:27,01/9,93ms. A25, RPS1362,93/5432,9. Reader completo nuevo a25:p95 56,47ms,
619,45RPS. Son medidas de datos sintéticos; no acreditan capacidad productiva
ni el peor volumen futuro. EXPLAIN nuevo usa ix_memory_revision_private_feed,
los joins restantes PRIMARY y elimina temporary/filesort. La versión sin orden
de join explícito no mejoró fuera del ruido; se descartó esa variante.

La denormalización de tenant en revisiones se obtiene exclusivamente del objeto
y queda protegida por FK compuesta. Un trigger deriva ese mismo valor cuando
un llamador anterior omite la columna. Una prueba real bajo STRICT_TRANS_TABLES
confirmó la compatibilidad del INSERT VALUES usado por la API anterior y el
rechazo de tenant explícito distinto, también sobre el esquema006 real. No
se extiende esa garantía a INSERT SELECT que omita la columna.

Coordinación: Hyde conserva sus módulos de autoridad y migración005. PR274
recibió aviso; los límites de duración van en drop-ins nuevos propios, sin editar
las unidades base que lleva esa sesión. La marca de este frente está publicada
en PENDIENTES.md. Los subagentes no operaron SSH, bases ni producción.

Pendiente dentro de este encargo: revisión del SHA exacto, integración por PR,
aplicación productiva explícita, adopción idempotente y ciclo observado con un
turno autenticado que recupere un hecho legacy conocido. No se declara cierre.

Continuación del26-sep: CI detectó cuatro módulos nuevos sin ejecución y doce
capturas de error sin contrato escrito. Se añadieron los comentarios de fallo
visible sin cambiar comportamiento y un job propio con MariaDB12.3.3 vacío,
jax_test limitado a su base y102 pruebas sin skips. La misma receta pasó en
una base vacía real.31 controles locales de CI pasan. Otro job reveló que la
migración B9 de jax-platform exige JAX_REPO_PATH; CI ahora apunta al checkout
actual mediante github.workspace. El cambio de workflow queda sujeto a la
regla de integración de Fernando, sin usar la excepción automática de Codex.
