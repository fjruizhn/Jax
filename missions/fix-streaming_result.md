# Resultado de: fix-streaming.md

Gate 3: **PASS** — `KIMI-STREAM-OK`.

---

## Reporte Final

### Archivos modificados

| Archivo | Cambio |
|---|---|
| `~/jax/jax/muscles/base.py` | Fix A: streaming SSE en `_call_openai` + `import json` |
| `~/jax/config/config.toml` | Fix B: `timeout_seconds` 180 → 600 |

**Rollback disponible:** `base.py.backup-streaming-20260627-080208`

### Diff del bloque `_call_openai`

- `"stream": False` → `"stream": True`
- `async with httpx.AsyncClient... client.post()` + `resp.json()` + `choices[0]["message"]` → reemplazado por `client.stream("POST", ...)` + `aiter_lines()` + acumulación por `delta.content`
- `import json` agregado (no existía)

### Timeout

| | Antes | Después |
|---|---|---|
| Modo REPL (`timeout_seconds`) | 180s | **600s** |
| Modo `--task` (`task_timeout_seconds`) | 3600s | 3600s (sin cambio) |

### Gate (3/3 obligatorios — todos PASS)

| Caso | Esperado | Resultado |
|---|---|---|
| Ada corta | `STREAM-OK` | ✓ `STREAM-OK` |
| Ada larga (era ReadError) | 500 líneas + `STREAM-LARGO-OK` | ✓ 505 líneas totales, `STREAM-LARGO-OK` en línea 501 |
| Kimi | `KIMI-STREAM-OK` | ✓ `KIMI-STREAM-OK` |

### Versión httpx: 0.28.1 — `client.stream()` + `aiter_lines()` disponibles. Sin improviso.

### Incertidumbres resueltas
- `import json` no existía → se agregó.
- `reasoning_content` de Kimi no llega en `delta.content` → se ignora naturalmente, confirmado por gate 3.
- Z.ai devuelve SSE estándar de OpenAI → funciona sin adaptación.

🔧 *Origen de autoridad: conocimiento técnico de Hyde. Los comandos ejecutados o verificados se indican en el cuerpo; el resto es de conocimiento, no verificado en vivo.*
