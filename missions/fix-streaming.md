# Misión Hyde — Streaming (SSE) en el worker OpenAI-compat

> Ejecutar con: `jax --task ~/jax/missions/fix-streaming.md --facet hyde`
> **CAMBIO DELICADO.** Backup obligatorio. Si el gate (§7) falla en cualquier caso → rollback inmediato y reportar. NO dejar el worker a medias.
> Hipatia inicia con `/using-superpowers` y `/ruflo`.

---

## 1. Objetivo y causa raíz

El worker OpenAI-compat (`_call_openai` en `~/jax/jax/muscles/base.py`, usado por
**Ada** y **Kimi**) manda `"stream": False`. Con streaming apagado, el modelo genera
toda la respuesta en silencio y la entrega de un golpe al final; durante esa
generación (decenas de segundos en salidas largas) la conexión HTTP no transmite
nada, y Z.ai/un proxy la corta → `httpx.ReadError('')`. Por eso el contrato v3
completo falla aunque el input sea chico y `max_tokens` esté alto.

**Fix de raíz:** activar streaming (`"stream": True`) y leer la respuesta por chunks
SSE, acumulándolos. La conexión nunca queda en silencio → no se corta. Es el
mecanismo correcto para respuestas largas. El interfaz de `_call_openai` NO cambia:
sigue devolviendo un string; solo cambian las tripas de request/lectura.

**Alcance:** SOLO `_call_openai` (Ada + Kimi). NO tocar `_call_deepseek` (Jekyll) ni
el worker Gemini (Hipatia) — quedan como están; DeepSeek streaming es un follow-up
aparte con su propio gate.

## 2. Principios (HYDE activo)
- No suponer; leer antes de editar. Backup antes de modificar.
- Ningún cambio sin prueba. Gate fuerte (§7). Rollback ante cualquier fallo del gate.
- Declarar incertidumbres.

## 3. Reconocimiento (read-only) — NO EDITAR TODAVÍA
```bash
# a) El bloque EXACTO de _call_openai (request + lectura de respuesta)
grep -n "_call_openai\|stream\|httpx.AsyncClient\|resp.json\|choices\|message\|Origen" ~/jax/jax/muscles/base.py
sed -n '225,260p' ~/jax/jax/muscles/base.py

# b) ¿json y httpx están importados arriba del archivo?
grep -n "^import\|^from" ~/jax/jax/muscles/base.py | grep -E "json|httpx"

# c) DE DÓNDE sale el valor de self.timeout y CUÁNTO es (clave para streaming largo)
grep -n "timeout" ~/jax/jax/muscles/base.py | head
grep -rn "timeout" ~/jax/jax/core/main.py | head
grep -rn "timeout" ~/jax/config/config.toml 2>/dev/null | head

# d) El wrapper asyncio.wait_for que pone un techo TOTAL de tiempo (línea ~117)
sed -n '113,125p' ~/jax/jax/muscles/base.py
```
**Reportá antes de editar:** (1) líneas exactas del bloque request/lectura de
`_call_openai`; (2) si `import json` existe (si no, hay que agregarlo); (3) el VALOR
actual de `self.timeout` y de dónde viene; (4) confirmá que `asyncio.wait_for` en
~117 envuelve `_call` con `timeout=self.timeout` (techo total).

## 4. Fix A — streaming en `_call_openai`
- Backup `base.py.backup-streaming-$TS`.
- Si falta `import json` arriba del archivo, agregarlo.
- En el payload de `_call_openai`, cambiar `"stream": False` → `"stream": True`
  (dejar `max_tokens` como está).
- **Reemplazar** el bloque de request+lectura (el `async with httpx.AsyncClient...`
  que hace `resp = await client.post(...)`, chequea status, `data = resp.json()`,
  extrae `choices[0].message.content`) por esta versión streaming:

```python
texto = ""
async with httpx.AsyncClient(timeout=self.timeout) as client:
    async with client.stream("POST", url, headers=headers, json=payload) as resp:
        if resp.status_code != 200:
            body = await resp.aread()
            raise MuscleInvocationError(
                f"[{self.name}] OpenAI HTTP {resp.status_code}: {body[:200]!r}"
            )
        partes = []
        async for linea in resp.aiter_lines():
            if not linea or not linea.startswith("data:"):
                continue
            payload_str = linea[5:].strip()      # quita "data:"
            if payload_str == "[DONE]":
                break
            try:
                chunk = json.loads(payload_str)
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            pieza = delta.get("content")
            if pieza:
                partes.append(pieza)
        texto = "".join(partes)

# Post-proceso existente (mantener idéntico): limpiar auto-etiquetas del modelo.
lineas = [l for l in texto.splitlines()
          if not l.strip().startswith("⚙️ *Origen")]
return "\n".join(lineas).strip()
```
> Mantener EXACTAMENTE igual: el armado de `messages` (system_prompt + history +
> prompt), `headers`, `url`, `payload` (salvo `stream`), y el post-proceso final.
> Solo cambia cómo se envía y se lee la respuesta. `reasoning_content` (Kimi) se
> ignora naturalmente porque solo acumulamos `delta.content`.

## 5. Fix B — techo de tiempo total para generaciones largas
El `asyncio.wait_for(_call, timeout=self.timeout)` (~117) corta la llamada COMPLETA a
los `self.timeout` segundos. Streaming evita el `ReadError`, pero si `self.timeout` es
bajo, una generación larga ahora moriría con `MuscleTimeoutError`.
- Si en §3c `self.timeout` < 600s, **subirlo a 600** en su fuente (config.toml o el
  default donde se construyen los músculos). Si ya es ≥ 600, dejarlo.
- Subir el techo es seguro: solo cambia cuánto se está dispuesto a esperar; las
  respuestas rápidas siguen volviendo rápido.
- Reportar el valor anterior y el nuevo.

## 6. Incertidumbres a declarar (no suponer)
- Si `client.stream` o `aiter_lines` no existen en la versión de httpx instalada,
  declararlo y reportar la versión (`pip show httpx`). NO improvisar otra API.
- Si el endpoint de Z.ai devuelve el error 200-no-stream de forma distinta, capturar
  y reportar el cuerpo real (ahora se ve, gracias al fix de error previo).
- Si `self.timeout` no se puede ubicar/editar limpio, declararlo y NO forzar.

## 7. Gate de prueba (OBLIGATORIO — rollback si falla cualquiera)
```bash
# 1) NO-REGRESIÓN: una salida corta sigue funcionando
cat > ~/stream-gate-short.md <<'EOF'
ada: respondé solo STREAM-OK
EOF
jax --task ~/stream-gate-short.md --facet ada
cat ~/stream-gate-short_result.md
#   ESPERADO: "STREAM-OK". Si falla → ROLLBACK.

# 2) EL CASO QUE FALLABA: salida LARGA que antes daba ReadError
cat > ~/stream-gate-long.md <<'EOF'
ada: generá una lista numerada del 1 al 500, un número por línea, sin texto extra.
Al final, en la última línea, escribí exactamente: STREAM-LARGO-OK
EOF
jax --task ~/stream-gate-long.md --facet ada
tail -8 ~/stream-gate-long_result.md
wc -l ~/stream-gate-long_result.md
#   ESPERADO: la lista llega a 500 Y la última línea dice STREAM-LARGO-OK
#   (prueba que la respuesta llegó COMPLETA por streaming, sin corte). Si hay
#   ReadError o se corta antes de 500 → ROLLBACK.

# 3) Kimi también usa este worker — verificar que no se rompió
cat > ~/stream-gate-kimi.md <<'EOF'
kimi: respondé solo KIMI-STREAM-OK
EOF
jax --task ~/stream-gate-kimi.md --facet kimi
cat ~/stream-gate-kimi_result.md
#   ESPERADO: "KIMI-STREAM-OK". Si falla → ROLLBACK.
```
**Criterio de aceptación (los 3 obligatorios):**
1. Salida corta de Ada → STREAM-OK.
2. Salida larga de Ada → llega a 500 + STREAM-LARGO-OK (sin ReadError, sin corte).
3. Kimi → KIMI-STREAM-OK.
Si CUALQUIERA falla: `cp base.py.backup-streaming-$TS base.py`, reportar el fallo
exacto, NO dejar el worker modificado.

## 8. Reporte final
Archivo(s) tocado(s) + diff del bloque de `_call_openai`, valor de timeout antes/después,
resultado de los 3 casos del gate (con conteos reales), versión de httpx, incertidumbres,
rollback disponible (`base.py.backup-streaming-$TS`).
