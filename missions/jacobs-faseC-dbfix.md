faceta: hyde

# Jacobs Fase C — Fix: persistir `depends_on` en la base de datos

> Editar `~/jax/jacobs/store.py`. HYDE: backup, gate, rollback si falla.
> Hipatia inicia con `/using-superpowers` y `/ruflo`.

## Causa raíz (verificada)
El campo `depends_on` viaja bien por todo Python (request → StepSpec → _from_spec → Step,
todo confirmado con pruebas directas del intérprete). PERO `store.py` NO lo persiste:
- `step_upsert` (INSERT) no incluye la columna `depends_on`.
- `steps_by_pipeline` (lectura) reconstruye el Step sin `depends_on` → vuelve al default [].
Resultado: el Step se guarda en MariaDB sin dependencias, y al releerlo para ejecutar,
el executor ve `depends_on=[]`. Por eso el contexto completo de Fase C no se cargaba.

Además, la tabla `jacobs_steps` se creó antes de que existiera el campo → probablemente
le falta la columna (el INSERT fallaría o la descartaría).

## Paso 1 — Verificar/crear la columna en la tabla
El usuario `fruiz` no tiene acceso MySQL directo; usar las credenciales del propio store.
Hipatia: leé la config de conexión de `store.py` (`_db_cfg`) y ejecutá vía Python con esas
credenciales (NO con `mysql -u root`).

```bash
cd ~/jax/las_manos
~/jax/las_manos/.venv/bin/python -c "
import asyncio, sys; sys.path.insert(0,'.')
from jacobs import store
async def go():
    conn = await store.get_conn()
    cur = await conn.cursor()
    # ¿existe la columna?
    await cur.execute(\"SHOW COLUMNS FROM jacobs_steps LIKE 'depends_on'\")
    existe = await cur.fetchone()
    if existe:
        print('columna depends_on YA existe')
    else:
        await cur.execute('ALTER TABLE jacobs_steps ADD COLUMN depends_on JSON NULL')
        await conn.commit()
        print('columna depends_on AGREGADA')
    await cur.execute('DESCRIBE jacobs_steps')
    for row in await cur.fetchall():
        print(row)
    conn.close()
asyncio.run(go())
"
```
> Reportar el resultado (si existía o se agregó) y el esquema final de la tabla.

## Paso 2 — `store.py`: guardar y leer `depends_on`
Backup: `store.py.backup-dependson-$(date +%Y%m%d-%H%M%S)`

### 2a — En `step_upsert` (INSERT):
Agregar `depends_on` a la lista de columnas, al VALUES, y a los parámetros (como JSON,
igual que `input`). Patrón: `json.dumps(s.depends_on, ensure_ascii=False)`.
```sql
INSERT INTO jacobs_steps
    (step_id, pipeline_id, step_index, facet, capability,
     input_ref, output_ref, status, timeout_seconds,
     retries_allowed, skip_on_fail, trace_id,
     started_at, finished_at, error, depends_on)
VALUES (%s,%s,%s,%s,%s, %s,%s,%s,%s, %s,%s,%s, %s,%s,%s, %s)
```
Y en los parámetros, agregar al final:
```python
    json.dumps(s.depends_on, ensure_ascii=False),
```
> NOTA: el ON DUPLICATE KEY UPDATE no necesita actualizar depends_on (no cambia tras crear),
> pero si querés robustez, agregá `depends_on=VALUES(depends_on)` ahí también.

### 2b — En `steps_by_pipeline` (lectura):
Al reconstruir el Step, parsear y pasar `depends_on`:
```python
        deps_raw = row.get("depends_on")
        try:
            depends_on = json.loads(deps_raw) if deps_raw else []
        except (json.JSONDecodeError, TypeError):
            depends_on = []
        result.append(Step(
            ...
            depends_on=depends_on,    # AGREGAR esta línea al constructor del Step
            ...
        ))
```
> Verificar si hay OTRA función que reconstruya Step desde fila (ej. en _row_to_pipeline
> o pipeline_get que carga el plan). Si el plan del pipeline se guarda como JSON completo
> (campo `plan` JSON en jacobs_pipelines), ahí depends_on ya viaja dentro del JSON del Step
> y NO se pierde — verificar cuál ruta usa el executor para cargar los steps. Reportar
> ambas rutas y asegurar que la que usa el executor preserve depends_on.

## Paso 3 — Gate (el mismo de Fase C, ahora debe pasar)
```bash
sudo systemctl restart jax-las-manos && sleep 8 && systemctl is-active jax-las-manos

# Pipeline 2 steps: el 2º depende del 1º. Debe ver el output COMPLETO del step 0.
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

# Esperar ejecución (qwen local ya está caliente en GPU)
sleep 30

# Ver el depends_on PERSISTIDO del step 1 (la prueba directa del fix)
curl -s http://127.0.0.1:7777/jacobs/pipeline/$PIPE | python3 -c "
import sys,json; d=json.load(sys.stdin)
for s in d.get('steps',[]):
    print('step', s['step_index'], 'depends_on=', s['depends_on'], 'status=', s['status'])
    if s.get('output_ref'): print('   output_ref:', s['output_ref'][:80])
"

# Log del contexto que recibió el step 1
sudo journalctl -u jax-las-manos -n 20 --no-pager | grep -i "contexto\|deps" | tail -6
```
**Criterio de aceptación:**
- El step 1 persiste con `depends_on=[0]` (NO []) → el fix de DB funciona.
- El log muestra `step 1 deps=[0] contexto=N chars` con N > 0 → recibió el contexto.
- (Si qwen completó) el output del step 1 contiene "BLOQUE-ALFA-XYZ-123" → vio el contenido real.
- Servicio 'active'.

## Reporte final
Resultado del Paso 1 (columna existía/agregada + esquema), diff de store.py (INSERT + lectura),
qué ruta usa el executor para cargar steps (jacobs_steps directo vs plan JSON del pipeline),
resultado del gate (¿depends_on=[0] persistido? ¿contexto>0? ¿output correcto?),
incertidumbres, rollback (*.backup-dependson-*).
