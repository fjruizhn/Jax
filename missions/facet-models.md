/using-superpowers
/ruflo

# Misión: facet_models — selector de modelo por faceta en Axioma

## Contexto (leer antes de tocar nada)

Hoy Axioma tiene tres fuentes de verdad sobre qué modelo usa cada faceta, y
las tres divergen entre sí:
1. `jacobs/executor.py` — cada `_invoke_*` tiene `model = "<literal>"` hardcodeado.
2. `backend/api/admin/keys.py` — diccionario `PROVIDERS` con un campo `"model"`
   fijo que alimenta la columna MODELO en /admin/api-keys (confirmado
   desactualizado: dice `gpt-4o`, la realidad en executor.py es `gpt-5.5`).
3. `config/config.toml` — `[personalities.<facet>]` con `model_default`/
   `models_allowed`, que solo lee el CLI viejo (`jax/core/main.py`), no Jacobs.

Objetivo: una sola fuente de verdad en MariaDB (`jax_memory`), con un
dropdown en el frontend que permita cambiar el modelo activo por faceta sin
tocar código ni redeployar, más alta/baja de modelos permitidos.

Facetas HTTP en Jacobs (`_HTTP_FACETS` en executor.py): hipatia, jekyll,
thot, ada. Facet local: jax_local (Ollama). Facet subprocess: hyde (Claude
Code CLI — NO tocar su invocación en esta misión, solo agregar su fila de
modelos; la conexión real a Jacobs es una misión aparte, `hyde-jacobs.md`,
NO ejecutar todavía).

## 1. Reconocimiento (read-only, reportar antes de tocar nada)

```bash
sudo mysql jax_memory -e "SHOW TABLES;"
cat /home/fruiz/jax-platform/backend/api/admin/keys.py
cat /home/fruiz/jax-platform/backend/main.py | grep -n "include_router\|APIRouter"
cat /home/fruiz/jax-platform/frontend/src/pages/admin/AdminApiKeys.jsx
grep -n "^async def _invoke_\|^_HTTP_FACETS\|^_MOTOR_FACETS" /home/fruiz/jax/jacobs/executor.py
```

Reportar: ¿existe ya algo llamado `facet_models` o similar? ¿Cómo registra
`main.py` los routers existentes (para replicar el patrón exacto)? ¿El
`AdminApiKeys.jsx` real coincide con lo que ya vimos, o cambió desde
entonces?

## 2. Verificar soporte de índice parcial en MariaDB 11.4

```bash
sudo mysql jax_memory -e "
CREATE TABLE facet_models_test (
    id INT AUTO_INCREMENT PRIMARY KEY,
    facet VARCHAR(50) NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT FALSE
);
CREATE UNIQUE INDEX uniq_active_test ON facet_models_test (facet, is_active) WHERE is_active = TRUE;
"
echo "Exit: $?"
sudo mysql jax_memory -e "DROP TABLE IF EXISTS facet_models_test;"
```

Si falla (exit != 0 o error de sintaxis): usar un `TRIGGER BEFORE INSERT/UPDATE`
que fuerce `UPDATE facet_models SET is_active=FALSE WHERE facet=NEW.facet AND
id != NEW.id` cuando `NEW.is_active=TRUE`, en vez del índice parcial. Declarar
en el reporte final cuál de los dos caminos se tomó y por qué.

## 3. Crear la tabla real + seed

Seed con los valores REALES confirmados en el reconocimiento del paso 1 (no
los de este mission file a ciegas — si `_invoke_thot` ya no dice `gpt-5.5`
para cuando corras esto, usar lo que encuentres en disco):

```sql
CREATE TABLE facet_models (
    id           INT AUTO_INCREMENT PRIMARY KEY,
    facet        VARCHAR(50) NOT NULL,
    provider_id  VARCHAR(50) NOT NULL,
    model_name   VARCHAR(100) NOT NULL,
    is_active    BOOLEAN NOT NULL DEFAULT FALSE,
    added_by     VARCHAR(100) DEFAULT NULL,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uniq_facet_model (facet, model_name)
);

INSERT INTO facet_models (facet, provider_id, model_name, is_active) VALUES
  ('thot',      'openai',    'gpt-5.5',            TRUE),
  ('thot',      'openai',    'gpt-5.6-sol',        FALSE),
  ('thot',      'openai',    'gpt-5.6-terra',      FALSE),
  ('thot',      'openai',    'gpt-5.6-luna',       FALSE),
  ('jekyll',    'deepseek',  'deepseek-v4-flash',  TRUE),
  ('jekyll',    'deepseek',  'deepseek-v4-pro',    FALSE),
  ('hipatia',   'gemini',    'gemini-2.5-flash',   TRUE),
  ('hipatia',   'gemini',    'gemini-3.1-pro',     FALSE),
  ('hipatia',   'gemini',    'gemini-3.5-flash',   FALSE),
  ('kimi',      'moonshot',  'kimi-k2.7-code',     TRUE),
  ('ada',       'zhipu',     'glm-5.2',            TRUE),
  ('jax_local', 'ollama',    'qwen3-coder:30b',    TRUE),
  ('jax_local', 'ollama',    'qwen3:14b',          FALSE),
  ('jax_local', 'ollama',    'qwen2.5:7b',         FALSE),
  ('jax_local', 'ollama',    'llama3.2:3b',        FALSE),
  ('hyde',      'anthropic', 'sonnet',             TRUE),
  ('hyde',      'anthropic', 'opus',               FALSE),
  ('hyde',      'anthropic', 'haiku',              FALSE);
```

## 4. Backend — nuevo router `backend/api/admin/facet_models.py`

Cuatro endpoints, todos con `Depends(require_superadmin)` (mismo patrón que
`config_admin.py`):
- `GET /api/admin/facet-models/{facet}` — lista modelos + cuál está activo
- `POST /api/admin/facet-models/{facet}` — agrega modelo nuevo (`is_active=FALSE`)
- `PUT /api/admin/facet-models/{facet}/active/{model_id}` — marca activo (y
  desactiva los demás de esa faceta, respetando el índice/trigger del paso 2)
- `DELETE /api/admin/facet-models/{facet}/{model_id}` — elimina, PERO rechaza
  con 400 si `is_active=TRUE` ("no se puede eliminar el modelo activo —
  activá otro primero")

Registrar el router en `main.py` siguiendo el patrón exacto que ya usan los
demás routers de `api/admin/`.

## 5. Corregir `keys.py` para que deje de mentir

En `list_keys`, reemplazar `"model": p["model"]` (el dict fijo `PROVIDERS`)
por un `SELECT model_name FROM facet_models WHERE facet=%s AND is_active=TRUE`
por cada provider. Si no hay fila activa, fallback al valor viejo del dict
(fail-open, no romper la pantalla si la tabla nueva está vacía por algún motivo).

## 6. Frontend — `AdminApiKeys.jsx`

Reemplazar la celda de texto plano `MODELO` por un `<select>` poblado con
`GET /api/admin/facet-models/{facet}`, que dispare
`PUT /api/admin/facet-models/{facet}/active/{model_id}` en `onChange`. Agregar
un botón pequeño "+" que abra un input inline para `POST` un modelo nuevo
(provider_id + model_name), y un ícono de basura por fila que dispare
`DELETE` (deshabilitado/oculto en la fila que está activa).

Mantener el estilo Tailwind existente del archivo (slate-800/purple-600, ya
visto en `AdminSettings.jsx`) — no introducir un sistema de diseño nuevo.

## 7. Gate de prueba (obligatorio, no cerrar sin esto)

1. `curl` a los 4 endpoints nuevos con una faceta de prueba, confirmar
   status codes correctos (200 en los normales, 400 en el DELETE del activo,
   404 en faceta/modelo inexistente).
2. Cambiar el modelo activo de `thot` vía PUT, confirmar con GET que se
   reflejó, y confirmar que `/api/admin/keys` (el endpoint viejo) ahora
   muestra el modelo nuevo en la columna `model` — prueba de que dejó de
   leer el dict fijo.
3. Abrir `/admin/api-keys` en el navegador (o describir el HTML resultante
   si no hay acceso visual), confirmar que el dropdown aparece y funciona.
4. Intentar `DELETE` sobre el modelo activo de una faceta, confirmar que
   rechaza con 400 y el mensaje esperado.

## 8. Explícitamente NO hacer en esta misión

- NO tocar `jacobs/executor.py` todavía (eso es la próxima misión: leer
  `facet_models` desde ahí en cada `_invoke_*`).
- NO conectar Hyde a Jacobs (misión aparte).
- NO tocar `config/config.toml` (sigue siendo la fuente de gobernanza de
  `models_allowed` para el CLI viejo; no lo reemplazamos, coexiste).

## 9. Reporte final

Igual que las misiones anteriores: archivos tocados + diff, resultado del
gate (todos los casos), incertidumbres declaradas explícitamente, path del
rollback (backups con timestamp de cada archivo tocado).
