# Reactivación observada de memoria B9

Este procedimiento no concede permiso de producción. Requiere autorización del
responsable y un SHA integrado con revisión independiente; los snapshots no
sustituyen la autoridad y el estado vigente consultables mediante `jaxctl`.

1. Coordinar la propiedad de archivos y verificar que los timers estén detenidos.
   Conservar los horarios y drop-ins instalados; registrar el SHA desplegado.
2. Respaldar íntegramente la base, restaurarla en una instancia sin red y comprobar
   tablas y conteos. Conservar dump, manifiesto y planes con permisos privados.
3. En una base fresca de pruebas, aplicar las migraciones y ejecutar
   `tests/memory_b9_regression_driver.py` con `jax_test`. El proveedor está simulado;
   SQL, API, permisos, triggers, locks y commits son reales. Complementar con las
   regresiones unitarias y las pruebas SQL de atomicidad.
4. Aplicar la migración de jobs una vez, de forma explícita. El arranque de los
   workers no debe ejecutar migraciones ni modificar ownership legacy.
5. Generar y revisar el plan privado de adopción. Aplicarlo mediante la API y un
   administrador real; repetirlo para verificar que no crea eventos adicionales.
   Verificar lectura del dueño, aislamiento entre usuarios y reconciliación.
6. Ejecutar unidades manualmente antes de habilitar timers. Verificar estado de
   salida, registros durables y duración; un lote con fallos debe terminar no cero
   aunque haya completado conversaciones sanas. Observar embedding, lifecycle,
   síntesis y salud vectorial, incluyendo los casos sin trabajo elegible.
7. Con una sesión autenticada, comprobar un recuerdo legacy y una conversación
   nueva. Cerrar únicamente la conversación controlada mediante su frontera
   autorizada; observar extracción y lectura B9, sin marcar procesado a mano.
8. Habilitar timers sólo tras resultados satisfactorios; registrar próximos
   disparos, identidad del código, evidencia y límites de lo realmente probado.

Ante fallo, detener la unidad afectada y conservar jobs, datos y evidencia.
No borrar resultados ni reiniciar estados para obtener una salida verde. La
recuperación debe respetar el resultado congelado y los marcadores de commit.

La migración 006 incorpora DDL de tenant, FK, trigger e índices. Tras una
interrupción no debe ejecutarse completa a ciegas: inspeccionar columnas,
triggers, constraints e índices y completar exclusivamente los pasos faltantes.
La validación de readiness detecta una instalación incompleta sin repararla.
Síntesis excluye scopes con user_id NULL y exige fuentes B9 verificadas; un
ciclo sin fuentes elegibles no demuestra síntesis de memoria de proyecto.
