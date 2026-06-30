# Resultado de: jacobs-faseC-dbfix.md

`sudo` requiere terminal interactiva. Necesito que ejecutes el restart:

```bash
sudo systemctl restart jax-las-manos && sleep 8 && systemctl is-active jax-las-manos
```

Cuando lo tengas activo, ejecutá el gate completo:

```bash
PIPE=$(curl -s --max-time 120 http://127.0.0.1:7777/jacobs/pipeline -X POST -H "Content-Type: application/json" \
  -d '{
    "name":"gate-faseC-2",
    "objective":"prueba contexto completo",
    "invoked_by":"Fernando",
    "mode":"autonomous",
    "steps":[
      {"facet":"jax_local","capability":"reason","prompt":"Escribí exactamente: BLOQUE-ALFA-XYZ-123 y nada más.","depends_on":[]},
      {"facet":"jax_local","capability":"reason","prompt":"¿Qué texto exacto recibiste del step 0? Repetilo. Si no lo viste, decí NADA.","depends_on":[0]}
    ]
  }' | python3 -c "import sys,json; print(json.load(sys.stdin)['pipeline_id'])")
echo "pipeline: $PIPE"

sleep 30

curl -s http://127.0.0.1:7777/jacobs/pipeline/$PIPE | python3 -c "
import sys,json; d=json.load(sys.stdin)
for s in d.get('steps',[]):
    print('step', s['step_index'], 'depends_on=', s['depends_on'], 'status=', s['status'])
    if s.get('output_ref'): print('   output_ref:', s['output_ref'][:80])
"

sudo journalctl -u jax-las-manos -n 20 --no-pager | grep -i "contexto\|deps" | tail -6
```

---

## Resumen de lo ejecutado

**Paso 1 — Columna DB:**
- `depends_on` **NO existía** en `jacobs_steps` → fue **AGREGADA** como `longtext NULL`.

**Paso 2 — store.py (2 cambios):**

`step_upsert` (INSERT):
- Agregada columna `depends_on` a la lista de columnas y a `VALUES` (16 parámetros ahora).
- Serializada como `json.dumps(s.depends_on, ensure_ascii=False)`.
- También en `ON DUPLICATE KEY UPDATE` para actualizaciones posteriores.

`steps_by_pipeline` (lectura):
- Parseo de `depends_on` del row de DB antes de construir el Step.
- Guard `try/except` igual que el de `input_ref`.
- `depends_on=depends_on` pasado al constructor de Step.

**Ruta que usa el executor:** `run_pipeline()` itera `pipeline.plan` que viene de `_row_to_pipeline()` → lee el campo `plan` JSON de `jacobs_pipelines` → `depends_on` ya viajaba correctamente aquí (porque `model_dump()` lo incluye). El bug era que `step_upsert` no persistía la columna en `jacobs_steps`, y `steps_by_pipeline` (usada por el endpoint GET y por la reanudación de supervisados) no la leía.

**Backup:** `/home/fruiz/jax/jacobs/store.py.backup-dependson-20260629-042958`

Pegame el output del gate cuando tengas el restart hecho.

🔧 *Origen de autoridad: conocimiento técnico de Hyde. Los comandos ejecutados o verificados se indican en el cuerpo; el resto es de conocimiento, no verificado en vivo.*
