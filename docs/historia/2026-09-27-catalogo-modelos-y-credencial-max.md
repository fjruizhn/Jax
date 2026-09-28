# Catálogo de modelos siempre al día, y la credencial de la cuenta Max

HISTORIA / Hyde (Claude Code, hall9000) · 27-sep-2026 (America/Tegucigalpa).
Fuente: la sesión en la que Fernando preguntó «¿por qué en Axioma no me aparece
Opus 5.5?», consultas de solo lectura a `jax_memory`, journal de los servicios
y los PRs citados. Este registro documenta observaciones y decisiones; no
certifica el estado presente. Para el estado vigente, consultar las fuentes
de la guía operacional.

## Qué pasó

**Síntoma.** Opus 5.5 no aparecía en Axioma. Anthropic sí lo ofrecía a la
cuenta (`GET /v1/models` lo listaba primero).

**Causa raíz (HECHO, verificado).** El 17-sep a las 15:22 los servicios pasaron
a la cuenta `jaxsvc` (`HOME=/var/lib/jaxsvc`, drop-in `cuenta-de-servicio.conf`).
El sync del catálogo leía el token de Anthropic de `~/.claude/.credentials.json`,
que existe solo en `/home/fruiz` y es `600 fruiz`. Desde entonces el sync de
Anthropic se **saltaba**, y el endpoint devolvía `ok: true` igual, porque solo
contaba como fallo un `error`, no un `skipped`. Último sync bueno de Anthropic:
17-sep 07:19. Nadie lo notó en diez días.

**Hallazgos de paso, también reales:**
- Jekyll estaba atado a `deepseek-v4-flash`, `deprecated` desde el 4-sep. Nadie
  lo notó: el sync no miraba los bindings.
- **Gemini tenía exactamente 50 modelos `available`**: su `pageSize` por
  defecto. El sync no paginaba, y todo lo que quedaba fuera de la primera
  página sumaba *misses* hasta deprecarse. Con la paginación, 50 → 61
  disponibles y 6 «deprecados» volvieron.
- `price_cache_per_1m_usd` se truncaba (`DECIMAL(10,4)`).
- El sync solo se disparaba con un botón del admin: no había nada programado.

## Qué se hizo

| PR | Qué |
|---|---|
| jax-platform#163 (`cfc03b7`) | Token por `CLAUDE_CODE_OAUTH_TOKEN`. `ok` baja con proveedores saltados/fallidos y con `facetas_en_riesgo`. Paginación de Anthropic y Gemini con tope. Lista vacía = error. Timer `jax-catalogo-modelos` cada 6 h con `OnFailure=` y aviso por Telegram. Modelos nuevos desde `model.created_at`. Candado `GET_LOCK`. Precios `DECIMAL(12,6)`. Script `ops/guardar-token-claude.sh`. |
| Jax#281 (`5d31fa7`) | `hyde_sandbox`: la credencial va por un `env=` mínimo explícito, **nunca por argv**. El archivo se monta solo si es legible. Sin ninguna credencial, fail-closed. |
| jax-platform#164 (`c127a7f`) | CI rojo desde el 26-sep (ver abajo). |
| jax-platform#165 | Runbook: el respaldo del paso 0 no se podía restaurar. |

**Despliegue (27-sep: código ~09:03–09:40 CST; token y verificación ~16:05–16:09 CST).** Respaldo de `jax_memory` (11 MB) con
**restauración probada** en una base descartable: 6 tablas clave, 66 tablas y 5
triggers idénticos. jax `4642613 → 5d31fa7`, jax-platform `7fa4303 → cfc03b7`,
bundle público `index-s5m99BgC.js → index-BsYYSMie.js`. Fernando guardó el token
con el script (108 caracteres, `.env` sigue `root:jaxsvc 640`) y reató Jekyll a
`deepseek-v4-pro` por la vía gobernada (`model_ref` 140; la sonda de las 16:07:32
dio `ok`).

**Verificación en producción:**
- Anthropic `/v1/models` con el token nuevo, como `jaxsvc`: HTTP 200, 12
  modelos, `claude-opus-5-5` presente.
- Primer sync por la unidad: `ok=True`, 7 proveedores sin fallos, 0 facetas en
  riesgo, nuevos `{anthropic: 1, gemini: 3}`, 4,5 s. La marca de avisos avanzó,
  y solo avanza con confirmación de Telegram.
- **Freno ejercitado** (Principio VII): un `SIGKILL` a mitad de corrida →
  `Result=signal` → `OnFailure=` → la unidad de aviso terminó en 0 (Telegram
  confirmó). La corrida siguiente entró sin `sync_en_curso`: el candado se
  liberó con la conexión.
- Timer habilitado; primera corrida programada a las 18:03 CST.

## Por qué así (decisiones)

- **Nada de API key: la cuenta Max vía `claude setup-token`.** DECISIÓN de
  Fernando, 27-sep.
- **Jekyll → `deepseek-v4-pro`**, no `deepseek-flash`. Propuesta de Hyde (el
  papel de Jekyll es la profundidad; 8 llamadas en dos semanas, la diferencia
  son centavos), aprobada por Fernando el 27-sep.
- **GO absoluto para llevar a producción**, dado por Fernando en la sesión.

## Alternativas descartadas

- **Dar a `jaxsvc` lectura sobre el archivo de credenciales de fruiz (ACL).**
  Deshacía la frontera del 17-sep. Además, Claude Code reescribe el archivo al
  refrescar (se pierde la ACL), y el access token dura horas: dependería de que
  hubiera una sesión interactiva viva.
- **`--setenv TOKEN` en el argv de bwrap.** Lo implementó la primera versión de
  Jax#281 y la auditoría lo **rechazó (BLOCK)**: `/proc/<pid>/cmdline` es legible
  por cualquier usuario del host (`/proc` sin `hidepid`).
- **Guardián de «menos de la mitad» + botón «forzar» auditado.** Nació para
  protegerse de listas parciales. Tres rondas de auditoría le encontraron
  defectos nuevos cada vez (incluido un BLOCK: la auditoría del forzado se
  escribía después del acto y podía perderse). Se **quitó entero**: la
  paginación robusta y «lista vacía = error» atacan los defectos reales. Un
  retiro masivo legítimo fluye por los 3 misses de D1.4.
- **Aplicar las migraciones B9 004/006 al arrancar la plataforma** (para #164).
  El README de `b9_migrations` dice que se aplican a mano; se respetó y se usó
  un manifiesto con sha256.

## Lecciones técnicas

1. **Un `skipped` que no baja `ok` es un fail-open.** Diez días sin Anthropic
   con el botón diciendo «sincronizado».
2. **Cambiar el usuario de un servicio rompe todo lo que leía del `$HOME` del
   anterior**, y no siempre de forma ruidosa. Hyde montaba un archivo que
   `jaxsvc` no podía leer (`isfile` da True: basta con recorrer directorios).
   Hoy Hyde está cerrado por Block 6, así que no se notó.
3. **El arreglo se audita como el código original.** jax-platform#163 pasó
   **cinco** rondas de auditoría adversarial; dos las rechazaron por defectos que
   metieron los propios arreglos. La salida no fue un cuarto parche, sino
   **simplificar**.
4. **Los números de un subagente son una fuente, no una autoridad.** El delta de
   piso «+34» era falso; el runner midió **+153**. «Producción tiene el `.venv`
   roto» y «34 fallos por la base del subagente» también resultaron falsos al
   medir. El CI rojo, en cambio, sí era real y previo.
5. **Un respaldo sin restauración probada no es respaldo, literalmente.** El
   runbook pedía probar la restauración y la restauración **no funcionaba**
   (`DEFINER=root` → `ERROR 1227`). Se descubrió solo al ejecutarla.
6. **`--defaults-extra-file=<(printf …)`** se lee una sola vez: la segunda
   llamada de `mariadb` cae al socket por defecto.

## CI rojo desde el 26-sep (jax-platform#164)

master `7fa4303`, #162 y #163: 34 tests con `Unknown column 'r.tenant_id'`. Las
bases de test solo recibían las migraciones B9 001/002 (+003 vía authority); la
006 (`tenant_id`) se aplica a mano en producción, y jax#279 empezó a usarla.
Producción la tenía completa (verificado en `information_schema`: 4 tablas de
006, FK, trigger, `NOT NULL`, columna de 004). El arreglo aplica en las bases de
test lo declarado en `backend/b9_migraciones_en_produccion.json` (con sha256 =
archivos de `/srv/jax-prod/jax`). Un `.sql` nuevo no declarado pone CI en
rojo, así que la alarma sigue viva para la 007.

## Riesgos que quedan (declarados)

- El token de `setup-token` caduca en ~1 año (pendiente con fecha en
  `claude-skills/PENDIENTES.md`).
- Dentro del sandbox de Hyde, el agente ve su propio token (B-4, aceptado; antes
  veía el archivo, con `refreshToken`).
- OpenAI, DeepSeek, Moonshot y Zhipu no se paginan: NO VERIFICADO que no lo
  necesiten.
- `_sync_ollama_models` hace un `/api/show` por modelo sin tope.
- `aiomysql.Pool.release()` no despierta a un waiter al descartar una conexión
  cerrada (solo en el caso anómalo de `RELEASE_LOCK` no confirmado).
