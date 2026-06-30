# Resultado de: fix-maxtokens.md

**Gate pasado.** La lista llega hasta 200. Reporte final.

---

## Reporte final

### Fix A — `max_tokens` en payloads OpenAI-compat

**Archivos:** `~/jax/jax/muscles/base.py`

| Worker | Línea | Cambio |
|--------|-------|--------|
| `_call_deepseek` | 212 | `"max_tokens": 32000` agregado al payload |
| `_call_openai` | 246 | `"max_tokens": 32000` agregado al payload |

No existe `self.max_tokens` en config ni en `config.toml` — se hardcodeó 32000 como default seguro. Aplica a Ada (Z.ai/GLM-5.2) y Kimi (Moonshot/K2.7) por igual.

```
grep -n "max_tokens" base.py  →  212, 246  ✓
```

**Gate:** lista 1-200 llegó completa (204 líneas en el `_result.md` incluyendo cabeceras). Antes se cortaba.

---

### Fix B — error propagado al `_result.md`

**Archivo:** `~/jax/jax/core/main.py`, línea 317

```python
# Antes:
error_msg = humanizar_error(label, e)

# Después:
error_msg = str(e) or repr(e) or "error sin detalle"
```

**Causa raíz del `corto` vacío (dos vectores simultáneos):**
1. `humanizar_error` convierte `str(e)` en mensajes genéricos legibles ("El servicio está saturado", "Problema de credenciales"). El detalle técnico real (`[Ada] OpenAI HTTP 422: ...`) se descartaba antes de llegar al archivo.
2. Si `str(e)` era vacío (excepción levantada sin mensaje, o `resp.text` vacío), el fallback `corto = "".split("\n")[0][:160]` producía `""` → resultado `[Ada fallo] ` con espacio final.

El fix elimina `humanizar_error` del path del archivo — el error técnico crudo llega directo al `_result.md`. `repr(e)` cubre el caso de excepción sin mensaje.

```
grep -n "error_msg = str(e)" main.py  →  317  ✓
```

---

### Rollback disponible
```
~/jax/jax/muscles/base.py.backup-maxtokens-20260627-074107
~/jax/jax/core/main.py.backup-errmsg-20260627-074107
```

---

### Incertidumbre declarada

El gate del Fix B (forzar un fallo real y ver el detalle en el archivo) no se ejecutó con una llamada HTTP real fallida — se verificó estáticamente que el código propaga `str(e)`. Si Ada falla con un HTTP que contenía `"api key"` en el cuerpo, el comportamiento anterior (humanizar) ocultaba el código HTTP real; ahora se verá el error crudo. Eso es la intención.

🔧 *Origen de autoridad: conocimiento técnico de Hyde. Los comandos ejecutados o verificados se indican en el cuerpo; el resto es de conocimiento, no verificado en vivo.*
