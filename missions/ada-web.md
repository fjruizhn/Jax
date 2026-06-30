# Misión Hyde — Cerrar a Ada en la web (axioma-ia.io)

> Ejecutar con: `jax --task ~/jax/missions/ada-web.md`
> Hipatia inicia con `/using-superpowers` y `/ruflo`.
> Plugins: superpowers, ui-ux-pro-max, ruflo-core, impeccable, token-optimizer, napkin.

---

## 1. Objetivo

Desbloquear a **Ada** en el backend de `jax-platform` para que responda por
`/api/chat` con **GLM-5.2** (igual que ya lo hace en consola), y alinear la
identidad de Ada como **ANALISTA (rigor / formalización)** en toda la plataforma.

La causa raíz del bloqueo es un **placeholder temporal** en `chat.py` que
intercepta a Ada y devuelve un texto fijo "está pendiente" sin llamar al motor.
Se elimina y se la enruta por el patrón OpenAI-compatible (idéntico a Kimi).

## 2. Principios (PROTOCOLO HYDE activo)

- **No suponer.** Leer cada archivo antes de editar; confirmar el reconocimiento.
- **Backup antes de cada modificación.**
- **Ningún comando sin output conocido.** Probar incremental; la prueba final es el gate.
- **Causa raíz, no parche.** Se *elimina* el placeholder; no se rodea.
- **Toda incertidumbre se declara explícita** en el reporte.

## 3. Identidad fijada

**Ada = ANALISTA.** Dominio: rigor, formalización, demostración (la voz de Lovelace, ⚛️).
La consola YA está alineada (`system_prompt` en `~/jax/config/config.toml`).
Esta misión alinea el **frontend**, que aún dice "Síntesis final".

## 4. Mapa verificado (líneas exactas, ya confirmadas en vivo)

| # | Archivo:línea | Estado actual | Objetivo |
|---|---|---|---|
| A | `backend/api/chat.py` ~324-327 | Placeholder `if facet=="ada"` con texto fijo | **Eliminar** |
| B | `backend/api/chat.py` `_invoke_facet` (~263-272 es Kimi) | No hay rama `ada` → cae al fallback ollama | **Agregar rama `ada`** (clon de Kimi) |
| C | `backend/api/admin/keys.py:22` | `glm-4-flash`, `ZHIPU_API_KEY`, `test_url:None` | `glm-5.2`, `ZAI_API_KEY`, test_url real |
| D | `backend/api/admin/dashboard.py:17` | `"ZHIPU_API_KEY"` | `"ZAI_API_KEY"` |
| E | `backend/api/admin/usage.py:15` | `glm-4-flash` $0.00/$0.00 | `glm-5.2` $1.40/$4.40 |
| F | `frontend/src/i18n/es.js` + `en.js` | `descAda: 'Síntesis final'` / `'Final synthesis'` | `'Análisis y rigor'` / `'Analysis & rigor'` |

**Dato clave confirmado:** `backend/api/chat.py:18` carga
`CONFIG_PATH = ~/jax/config/config.toml` — el **mismo** config que el REPL.
Por eso `[personalities.ada]` (api_url Z.ai, model_default `glm-5.2`, system_prompt
analista) ya existe y el backend lo hereda. La rama nueva solo lo consume.

**Firma de `_call_openai_compat`** (confirmada en la rama Kimi):
`_call_openai_compat(base_url, api_key, model, system_prompt, history, message)`.

## 5. Tareas (orden estricto)

### 5.0 — Reconocimiento (read-only, confirmar supuestos)
Ejecutar y revisar antes de tocar nada:

```bash
# a) Confirmar la personalidad de Ada en el config central
grep -n -A14 '\[personalities.ada\]' ~/jax/config/config.toml
#   Esperado: api_url con .../api/paas/v4/chat/completions, model_default = glm-5.2,
#   system_prompt analista. Si falta api_url o model_default → DECLARAR y parar.

# b) Ver la función de test de keys.py (para saber si usa Bearer o ?key=)
sed -n '1,140p' ~/jax-platform/backend/api/admin/keys.py

# c) Cómo corre el backend (para reiniciarlo bien después)
systemctl list-units --type=service 2>/dev/null | grep -iE 'axioma|jax|uvicorn'
pm2 list 2>/dev/null

# d) Confirmar que el backend carga /etc/jax/.env en su entorno
grep -rniE 'load_dotenv|dotenv|EnvironmentFile|/etc/jax/.env' \
  ~/jax-platform/backend/ /etc/systemd/system/ 2>/dev/null | grep -viE 'venv|site-packages'

# e) ¿"synthesis"/capability se valida o mapea en el backend, o es etiqueta libre?
grep -rniE 'synthesis|capability' ~/jax-platform/backend/ --include=*.py \
  | grep -viE 'venv|site-packages|__pycache__'

# f) Unidad de precios en usage.py (per 1M vs per 1K) — ver el cálculo que consume MODEL_PRICES
grep -n -B2 -A8 'MODEL_PRICES' ~/jax-platform/backend/api/admin/usage.py
```

**No avanzar si (a) revela que falta `api_url` o `model_default` en Ada.**

### 5.1 — Backups (timestamp único)
```bash
TS=$(date +%Y%m%d-%H%M%S)
for f in api/chat.py api/admin/keys.py api/admin/dashboard.py api/admin/usage.py; do
  cp ~/jax-platform/backend/$f ~/jax-platform/backend/$f.backup-adaweb-$TS
done
cp ~/jax-platform/frontend/src/i18n/es.js ~/jax-platform/frontend/src/i18n/es.js.backup-adaweb-$TS
cp ~/jax-platform/frontend/src/i18n/en.js ~/jax-platform/frontend/src/i18n/en.js.backup-adaweb-$TS
```

### 5.2 — chat.py (A + B): el fix central
**A — eliminar** el bloque placeholder de Ada (~324-327):
```python
if facet == "ada":
    resp = "Ada está pendiente — su key Z.ai estará disponible en la semana del 22-jun."
    await _fire_completed(facet, tenant_id, user_id, resp)
    return ChatResponse(facet=facet, response=resp, timestamp=timestamp)
```

**B — agregar** la rama `ada` dentro de `_invoke_facet`, inmediatamente después
del bloque de Kimi, replicando su patrón:
```python
    if facet == "ada":
        api_url = personality.get("api_url", "https://api.z.ai/api/paas/v4/chat/completions")
        # Normalizar a base URL sin /chat/completions
        base_url = api_url[:-len("/chat/completions")] if api_url.endswith("/chat/completions") else api_url
        return await _call_openai_compat(
            base_url,
            os.getenv("ZAI_API_KEY", ""),
            personality.get("model_default", "glm-5.2"),
            system_prompt, history, message,
        )
```

### 5.3 — admin/keys.py:22 (C): coherencia del panel
Reemplazar la entrada de Z.ai. Usar el `test_url` según el formato que reveló 5.0(b):
- Si el test usa cabecera `Authorization: Bearer` → `test_url: "https://api.z.ai/api/paas/v4/models"`.
- Si usa otro formato → ajustar acorde (declarar la decisión).

```python
{"id": "zhipu", "name": "Z.ai", "facet": "ada", "model": "glm-5.2", "env_key": "ZAI_API_KEY", "test_url": "https://api.z.ai/api/paas/v4/models"},
```
> Mantener `id: "zhipu"` salvo que 5.0 confirme que NO se referencia en rutas
> (`/admin/keys/{id}/...`). Es etiqueta interna; el visible es `name: "Z.ai"`.

### 5.4 — admin/dashboard.py:17 (D)
En `_PROVIDERS_KEYS`: `"ZHIPU_API_KEY"` → `"ZAI_API_KEY"`.

### 5.5 — admin/usage.py:15 (E)
Reemplazar la entrada de precio (respetando la unidad confirmada en 5.0(f)):
```python
"glm-5.2": {"in": 1.40, "out": 4.40},
```

### 5.6 — Frontend (F): identidad analista
- `frontend/src/i18n/es.js`: `descAda: 'Análisis y rigor'`
- `frontend/src/i18n/en.js`: `descAda: 'Analysis & rigor'`
- **Capability del pipeline builder:** localizar el objeto de Ada con
  `capability: "synthesis"`. Si 5.0(e) confirmó que es **etiqueta libre** →
  cambiar a `"analysis"` por coherencia. Si está **tipada/mapeada** a un handler →
  **dejar como está y reportarlo** (no romper el pipeline).
- Rebuild + deploy del frontend según el flujo documentado:
  ```bash
  cd ~/jax-platform/frontend && npm run build
  ```
  (desplegar `dist/` al destino que sirve Nginx, según el procedimiento del proyecto).

### 5.7 — Reinicio + PRUEBA REAL (gate)
1. Reiniciar el backend con el método identificado en 5.0(c).
2. **Prueba end-to-end por la web:** loguearse en axioma-ia.io, seleccionar a **Ada**,
   enviar un mensaje corto (ej. *"presentate en una línea"*).
   - **Output esperado:** respuesta de **glm-5.2** con voz analista + sello ⚛️,
     **NO** el texto placeholder.
   - Si devuelve el placeholder → el backend no recargó; reiniciar de verdad.
   - Si tira error de auth (502 / 401) → revisar que `ZAI_API_KEY` esté en el
     entorno del backend (ver 5.0(d): EnvironmentFile o load_dotenv).
3. **Confirmar persistencia** en MariaDB:
   ```bash
   ( eval "$(grep -E '^JAX_DB_(USER|PASSWORD|NAME)=' /etc/jax/.env)"
     MYSQL_PWD="$JAX_DB_PASSWORD" mariadb -u "$JAX_DB_USER" "$JAX_DB_NAME" -e \
     "SELECT id, role, facet_used, model, LEFT(content,30) txt, created_at \
      FROM messages WHERE role='ada' ORDER BY id DESC LIMIT 3;" )
   ```
   Esperado: fila nueva con `role='ada'`, `model='glm-5.2'`.
4. **Panel admin:** verificar que Z.ai ya aparece con key **configurada** (no "sin key")
   y que el botón *Probar* da OK.

### 5.8 — Reporte final
Archivos tocados + diffs, resultado del gate 5.7 (respuesta cruda real),
incertidumbres declaradas, rollback disponible.

## 6. Rollback
Restaurar los 6 backups `.backup-adaweb-$TS`. Si se desplegó el frontend,
rebuild desde el `es.js`/`en.js` restaurados y redeploy.

## 7. Criterios de aceptación
- [ ] Placeholder de Ada eliminado de `chat.py`.
- [ ] Rama `ada` en `_invoke_facet` (OpenAI-compatible, glm-5.2, ZAI_API_KEY).
- [ ] `keys.py`, `dashboard.py`, `usage.py` coherentes (glm-5.2 / ZAI_API_KEY / precio real).
- [ ] `descAda` = analista en ES y EN; frontend rebuildeado y desplegado.
- [ ] Ada responde por la web con glm-5.2 (no placeholder).
- [ ] Fila en MariaDB con `role='ada'`, `model='glm-5.2'`.
- [ ] Panel admin: Z.ai con key OK.
