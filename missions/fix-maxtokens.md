# Misión Hyde — Fix `max_tokens` en worker OpenAI-compat + error vacío en `--task`

> Ejecutar con: `jax --task ~/jax/missions/fix-maxtokens.md --facet hyde`
> (forzar Hyde — la tarea es ejecución de código). Hipatia inicia con `/using-superpowers` y `/ruflo`.

---

## 1. Objetivo

Dos fixes de raíz en un solo paso:

**Fix A — `max_tokens` ausente (causa del `[Ada fallo]`).** El payload del worker
OpenAI-compat (`~/jax/jax/muscles/base.py`, ~línea 241) no envía `max_tokens`. Z.ai
(y otros OpenAI-compat) aplican un default de salida bajo (~4096), así que toda
respuesta larga de Ada o Kimi se trunca y el parseo posterior revienta. Es un límite
latente en TODO el sistema, no solo en un contrato.

**Fix B — mensaje de error vacío.** En `~/jax/jax/core/main.py` (~línea 134), el
fallo se reporta como `[label fallo] ` con el detalle vacío (`corto` llega vacío),
aunque el worker captura el error real con `status_code` y `resp.text[:200]`. Hay que
propagar el detalle real al `_result.md`.

## 2. Principios (HYDE activo)
- No suponer; leer antes de editar. Backup antes de modificar.
- Ningún cambio sin prueba (gate §6). Declarar incertidumbres.

## 3. Reconocimiento (read-only) — NO EDITAR TODAVÍA
```bash
# Fix A — ver el payload OpenAI-compat exacto y si hay max_tokens en otros workers
grep -n 'payload\|max_tokens\|stream' ~/jax/jax/muscles/base.py

# ¿El worker DeepSeek (también OpenAI-compat) ya manda max_tokens? (consistencia)
sed -n '200,260p' ~/jax/jax/muscles/base.py

# Fix B — la función que arma "[label fallo] {corto}" y de dónde sale 'corto'
sed -n '110,140p' ~/jax/jax/core/main.py

# Fix B — cómo se llama esa función desde el except de run_task (línea ~316)
sed -n '310,330p' ~/jax/jax/core/main.py
```
**Reportá** antes de editar: (1) el nombre exacto de la variable del payload en el
bloque OpenAI-compat y si hay un atributo de config tipo `self.max_tokens`; (2) si
DeepSeek/otros ya mandan `max_tokens` (para usar el mismo patrón); (3) qué función
genera `[label fallo]` y por qué `corto` queda vacío.

## 4. Fix A — `max_tokens` en el payload OpenAI-compat
- Backup `base.py.backup-maxtokens-$TS`.
- En el bloque OpenAI-compat (el `payload = {...}` con `"stream": False`, ~línea 241),
  agregar `max_tokens`:
```python
payload = {
    "model": model,
    "messages": messages,
    "stream": False,
    "max_tokens": 32000,   # evita truncado en salidas largas (Ada/Kimi). glm-5.2 lo soporta.
}
```
> Si existe un atributo de config por personalidad (ej. `self.max_tokens` leído del
> `config.toml`), preferir `payload["max_tokens"] = self.max_tokens or 32000` y
> reportarlo. Si NO existe, hardcodear 32000 como default seguro y declararlo.
> Aplicar el MISMO fix al worker DeepSeek si tampoco lo manda (mismo síntoma latente).

## 5. Fix B — propagar el error real al `_result.md`
- Backup `main.py.backup-errmsg-$TS` (si no se respaldó ya en esta corrida).
- En la función que arma `[{label} fallo] {corto}` (~línea 134), asegurar que `corto`
  contenga el texto real de la excepción. Identificar por qué llega vacío:
  - Si `corto` se deriva de `str(e)` y la excepción se construyó sin mensaje, propagar
    `repr(e)` o el `error_msg` completo.
  - En el `except (MuscleError, Exception) as e` de `run_task` (~línea 316), garantizar
    que `error_msg` incluya `str(e)` (que ya trae `[nombre] OpenAI HTTP NNN: <texto>`):
```python
except (MuscleError, Exception) as e:
    error_msg = str(e) or repr(e) or "error sin detalle"
    # ...escribe f"# Error en tarea: {task_file.name}\n\n{error_msg}\n"
```
> NO cambiar el flujo de control ni el formato del archivo — solo garantizar que el
> detalle real (HTTP status + cuerpo) llegue al `_result.md`. Reportar la causa exacta
> del vacío encontrada en §3.

## 6. Gate de prueba (obligatorio)
```bash
# 1) Fix A — una salida LARGA que antes reventaba ahora completa:
cat > ~/maxtokens-gate.md <<'EOF'
ada: generá una lista numerada del 1 al 200, un número por línea, sin texto extra.
EOF
jax --task ~/maxtokens-gate.md --facet ada
tail -5 ~/maxtokens-gate_result.md
#   ESPERADO: la lista llega hasta 200 (antes se cortaba ~línea por límite de salida).
#   Si llega a 200 → max_tokens funciona.

# 2) Fix B — forzar un error y ver que el detalle YA NO está vacío:
cat > ~/errmsg-gate.md <<'EOF'
ada: PING
EOF
#   Forzar fallo temporal: usar un modelo inexistente vía un task que dispare HTTP != 200.
#   (Si no hay forma simple de forzarlo, Hipatia documenta que el fix está aplicado y
#    verifica leyendo el código que error_msg = str(e) propaga el detalle.)
grep -n "max_tokens" ~/jax/jax/muscles/base.py
grep -n "error_msg = str(e)" ~/jax/jax/core/main.py
```

**Criterio de aceptación:** (1) la lista del gate llega a 200 sin cortarse; (2) el
código muestra `max_tokens` en el payload y `error_msg` propagando `str(e)`.

## 7. Reporte final
Archivos tocados + diffs, resultado del gate (¿llegó a 200?), causa exacta del `corto`
vacío, incertidumbres declaradas, rollback (`*.backup-maxtokens-$TS`, `*.backup-errmsg-$TS`).
