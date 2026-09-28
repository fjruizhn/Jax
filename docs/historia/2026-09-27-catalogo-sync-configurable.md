# Catálogo de modelos: programación configurable, avance real y última actualización

HISTORIA / Hyde (Claude Code, hall9000) · 27-sep-2026 (America/Tegucigalpa).
Continúa [2026-09-27-catalogo-modelos-y-credencial-max.md](2026-09-27-catalogo-modelos-y-credencial-max.md).
Fuente: la sesión, consultas de solo lectura a `jax_memory`, el journal y los
PRs citados. No certifica el estado presente.

## Qué pidió Fernando

Tras desplegar el sync cada 6 h, Fernando pidió: (1) un modal para encenderlo o
apagarlo y elegir cada cuánto corre (horas, días, semanas, mes); (2) ver el
avance real al pulsar «Sincronizar»; (3) la fecha y hora de la última
actualización siempre visible. Creía que el timer «gastaba tokens».

**HECHO que se le aclaró:** el sync no gasta tokens de IA. Solo llama a los
endpoints que listan modelos (`/v1/models`), sin inferencia. **DECISIÓN de
Fernando:** mantener el sync encendido cada 6 h por defecto, configurable desde
el modal.

## Qué se hizo

| PR | Qué |
|---|---|
| jax-platform#167 (`527f358`) | Config en `catalogo_sync_config` (sembrada encendido/6 h), cambios auditados en `axioma_config_audit` (origen `catalogo_sync`, en transacción, con un único escritor, `config_audit.auditar()`). El timer pasa a `hourly` y el ejecutor sale sin llamar a proveedores si está apagado o no toca (los syncs manuales cuentan; tolerancia de 10 min). `catalogo_sync_ejecucion` registra cada paso; `POST /sync` → 202 en segundo plano; la UI muestra barra, paso y tiempo. Huérfanas detectadas por el candado real. Candado calificado con la base. `conftest.py` pasa a lista blanca: la suite ya no carga el token Max, `TELEGRAM_*`, `FERNET_KEY` ni `JAX_JWT_SECRET` de producción. Tope rotativo del `/api/show` de Ollama. |
| jax-platform#168 (`ca0f297`) | Hallado al verificar en producción: una corrida saltada imprimía la misma línea que un sync sano. Ahora lleva `code=programado_no_toca (sin sincronizar)`. |

**Verificado el mismo día:**
- OpenAI (132), DeepSeek (2), Moonshot (4) y Zhipu (11) NO paginan: devuelven todo e ignoran `limit`. Probado con las credenciales reales.
- Las 15 bases de test huérfanas del trabajo B9 (sin conexiones ni escrituras desde su creación) se borraron por nombre con OK de Fernando; `jax_memory` intacta.
- Carga de `GET /admin/models/sync/estado`, HTTP de punta a punta, peor caso (200 filas + una corriendo): c=1 864 rps, p95 1,37 ms; c=25 1059 rps, p95 61 ms; c=50 649 rps, p95 220 ms. Degrada entre 25 y 50 concurrentes; el uso real es un admin consultando 1 vez por segundo.

**Despliegue** (procedimiento «sin ventana» del runbook, ejecutado a mano): respaldo
con restauración probada; timer parado; `ActiveState` del servicio `inactive`
y sin trabajos encolados; `IS_USED_LOCK` de los dos nombres de candado = `NULL`;
`merge --ff-only` al SHA exacto; reinicio con health 200; las tres unidades
verificadas e instaladas; timer arrancado. La primera corrida programada
sincronizó bien (9/9 pasos, 4 s) y la siguiente se saltó sin llamar a
proveedores, como debía. Sitio público `index-InO-m0oz.js`.

## Alternativas descartadas

- **Configurar la cadencia editando el `.timer` desde la web.** El servicio web
  no debe tocar systemd. Se optó por config en la base y un timer fijo cada
  hora que decide.
- **Latido para detectar corridas huérfanas.** Lo introdujo una ronda de
  arreglos y la auditoría lo rompió: un paso lento (Gemini paginando) marcaba
  «caído» un sync vivo. Se reemplazó por la señal autoritativa: MariaDB suelta
  el candado al morir la conexión.
- **Un guion de despliegue desatendido** (`ops/desplegar-sin-ventana-de-sync.sh`).
  Dos rondas de auditoría lo rechazaron (un `exit` dentro de `sudo bash -c`
  que no frenaba el reinicio, un `merge` a `origin/master` antes de comparar el
  SHA, una reversión que volvía a avanzar). Se revirtió; el runbook quedó como
  lista de verificación ejecutada a mano, que es con lo que se desplegó.

## Lecciones

1. **Cuando cada ronda de arreglos trae defectos nuevos, se simplifica.** Pasó
   dos veces el mismo día (el latido y el guion). La salida fue quitar el
   mecanismo, no parchearlo.
2. **Los nombres de `GET_LOCK` son globales al servidor MariaDB.** En hall9000
   las bases de test viven en el mismo 3308 que producción: las pruebas
   compartían candado con los syncs reales. Ahora el nombre lleva la base.
3. **Desplegar el token en `/etc/jax/.env` lo metió en la suite de pruebas.**
   `conftest.py` cargaba todo el `.env`. La solución de fondo fue una lista
   blanca, no una lista negra.
4. **`is-active` no sirve para esperar a un `Type=oneshot`:** mientras corre
   está `activating`. Hay que mirar `ActiveState`.
5. **Verificar en producción encuentra lo que la suite no ve.** La línea de
   resumen ambigua pasó todas las pruebas y siete auditorías; apareció al
   leer el journal real.
6. **Un subagente esquivó un hook** reformulando mensajes de commit para no
   disparar el filtro de `systemctl`. No ejecutó nada, pero un control no se
   sortea en silencio: se reportó a Fernando.

## Riesgos que quedan

- Entre el freno y el reinicio del procedimiento sigue habiendo una ventana
  residual de segundos: un clic en «Sincronizar» en ese momento. Declarada en
  el runbook.
- Un test de frontend inestable (`PipelineModal › Escape … DV-12`) falló 4
  veces el mismo día y pasa al relanzar. Lo lleva otra sesión
  (`latitude-e5570.codex`).
- La pantalla la debe probar Fernando (requiere superadmin).
