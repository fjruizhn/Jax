# Resultado de: fix-maxtokens-128k.md

Los tres criterios de aceptación verificados con evidencia real:

---

## Reporte Final

**Archivo tocado:** `~/jax/jax/muscles/base.py`
**Backup:** `base.py.backup-128k-20260627-090957`

**Diff (las dos líneas):**
```
L213 (DeepSeek/Jekyll):   "max_tokens": 32000  →  131072
L247 (OpenAI-compat/Ada+Kimi): "max_tokens": 32000  →  131072
```

**Gate §5 — resultados crudos:**

| Criterio | Resultado |
|---|---|
| Lista 1→1500 + GATE-128K-OK sin corte | ✅ — `tail -4` muestra `1500 / GATE-128K-OK` — 1505 líneas totales (lista + marcador + footer Ada) |
| Salida corta SHORT-OK | ✅ — sin regresión |
| `131072` en disco | ✅ — `grep -n max_tokens` confirma ambas líneas |

**DeepSeek/Jekyll:** aplicado el mismo cambio (131072). INCÓGNITA pendiente: no se probó contra el endpoint real de DeepSeek (requeriría una tarea `--facet jekyll`). Si rechazara 131072, el rollback es `cp base.py.backup-128k-20260627-090957 base.py`.

Ada ahora tiene pista completa de 128K — razona a fondo y escribe documentos largos sin pelear por presupuesto.

🔧 *Origen de autoridad: conocimiento técnico de Hyde. Los comandos ejecutados o verificados se indican en el cuerpo; el resto es de conocimiento, no verificado en vivo.*
