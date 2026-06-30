# Resultado de: jacobs-fase-c.md

Todo correcto. Fernando, cuando puedas correr el restart en tu terminal, luego ejecutá el gate completo:

```bash
# 1. Restart + health
sudo systemctl restart jax-las-manos && sleep 8 && systemctl is-active jax-las-manos

# 2. Verificación estática
grep -n "MAX_DEP_CONTEXT_CHARS\|depends_on\|full" ~/jax/jacobs/executor.py | head

# 3. Gate funcional (usando "autonomous" — "execute" no es modo válido)
curl -s --max-time 120 http://127.0.0.1:7777/jacobs/pipeline -X POST -H "Content-Type: application/json" \
  -d '{
    "name":"gate-faseC",
    "objective":"prueba de contexto completo",
    "invoked_by":"Fernando",
    "mode":"autonomous",
    "steps":[
      {"facet":"jax_local","capability":"reason","prompt":"Escribí exactamente este texto largo único: BLOQUE-ALFA seguido de 100 palabras inventadas distintas.","depends_on":[]},
      {"facet":"jax_local","capability":"reason","prompt":"Repetí TEXTUALMENTE el contenido completo que recibiste del step 0. Si solo viste un resumen cortado, decí TRUNCADO.","depends_on":[0]}
    ]
  }' | python3 -m json.tool | head -20

# 4. Logs de tamaño de contexto
sudo journalctl -u jax-las-manos -n 30 --no-pager | grep -i "contexto\|deps\|step" | tail -10
```

**Nota importante:** El gate usa `"mode":"autonomous"` no `"execute"` — `execute` no existe en `VALID_MODES` y fallaría con 422. El task original tenía ese valor incorrecto.

---

**Resumen de los 4 cambios aplicados:**

| Archivo | Cambio |
|---------|--------|
| `models.py` | `StepSpec` gana `depends_on: list[int]` — sin esto, el campo se descartaba en `.model_dump()` antes de llegar al executor |
| `executor.py` | `MAX_DEP_CONTEXT_CHARS = 60_000` — tope configurable (C3) |
| `executor.py` | `_build_context_input`: si `depends_on` no vacío → carga COMPLETO solo esos índices; si vacío → resumen 500 de todos (caso simple intacto) + log de tamaño (C1) |
| `executor.py` | `_enrich_prompt`: encabezado "Dependencias declaradas" + nota `[TRUNCADO]` si aplica (C2) |

Backups en: `jacobs/executor.py.backup-faseC-20260629-041736` y `jacobs/models.py.backup-faseC-20260629-041736`.

🔧 *Origen de autoridad: conocimiento técnico de Hyde. Los comandos ejecutados o verificados se indican en el cuerpo; el resto es de conocimiento, no verificado en vivo.*
