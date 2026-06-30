faceta: hyde

# Jacobs — Fix: outputs grandes van a artifact, no revientan la columna output_ref

> Editar el Jacobs CANÓNICO `~/jax/jacobs/`. HYDE: backup, gate, rollback si falla.
> Hipatia inicia con `/using-superpowers` y `/ruflo`.

## Causa raíz (verificada por traceback)
El ensamble mecánico generó 139.141 chars (0.13 MB). `save_if_large` usa SIZE_LIMIT = 1 MB,
así que 139K cayó en INLINE (no superó 1MB). Pero al escribir 139K inline en la columna
`output_ref` de `jacobs_steps`, MariaDB lo rechazó:
`pymysql.err.DataError: (1406, "Data too long for column 'output_ref'")`.
El step quedó colgado en `running` con output vacío.

DESAJUSTE: save_if_large cree que la columna aguanta 1MB; la columna aguanta mucho menos.

**Fix: alinear los dos límites.**
- A) Bajar SIZE_LIMIT a un valor seguro (ej. 60 KB) → outputs grandes van a ARTIFACT (archivo),
  y solo un ref corto se guarda en output_ref. El ensamble de 139K iría a archivo.
- B) Asegurar que output_ref sea LONGTEXT (respaldo, por si algún inline mediano se acerca).

## Paso 1 — Reconocimiento
```bash
grep -n "SIZE_LIMIT" ~/jax/jacobs/artifacts.py
sed -n '1,40p' ~/jax/jacobs/artifacts.py    # ver save_if_large completo + read_artifact
```
Reportar el valor actual de SIZE_LIMIT y cómo save_if_large guarda el artifact (ruta, formato del ref).

## Paso 2 — Fix A: bajar SIZE_LIMIT en artifacts.py
Backup: `artifacts.py.backup-sizefix-$(date +%Y%m%d-%H%M%S)`

Cambiar SIZE_LIMIT de 1 MB a 60 KB (alineado con TEXT de MySQL = 64KB, con margen):
```python
SIZE_LIMIT = 60_000   # bytes. >60KB → artifact en archivo (la columna output_ref no aguanta más inline).
```
> Verificar que el path de read_artifact siga funcionando (los refs que ya estén guardados
> no deben romperse). El ref que devuelve save_if_large debe ser corto (un path o id), no el contenido.

## Paso 3 — Fix B: agrandar output_ref a LONGTEXT (vía el servicio, con sus credenciales)
El usuario fruiz no tiene acceso MySQL directo. Ejecutar el ALTER con las credenciales del store:
```bash
cd ~/jax/las_manos
~/jax/las_manos/.venv/bin/python -c "
import asyncio, sys; sys.path.insert(0,'.')
from jacobs import store
async def go():
    conn = await store.get_conn()
    cur = await conn.cursor()
    await cur.execute(\"SHOW COLUMNS FROM jacobs_steps LIKE 'output_ref'\")
    print('ANTES:', await cur.fetchone())
    await cur.execute('ALTER TABLE jacobs_steps MODIFY COLUMN output_ref LONGTEXT NULL')
    await conn.commit()
    await cur.execute(\"SHOW COLUMNS FROM jacobs_steps LIKE 'output_ref'\")
    print('DESPUES:', await cur.fetchone())
    conn.close()
asyncio.run(go())
"
```
> Reportar el tipo ANTES y DESPUES. Si ANTES ya era LONGTEXT, el problema es solo el SIZE_LIMIT
> (Fix A) y no hace falta el ALTER — declararlo.
> NOTA: con Fix A, el ensamble de 139K va a artifact (no inline), así que output_ref solo guarda
> un ref corto. El Fix B es respaldo para inlines medianos (10-60KB) que sí van inline.

## Paso 4 — Gate (reproducir el caso que falló, barato)
```bash
sudo systemctl restart jax-las-manos && sleep 8 && systemctl is-active jax-las-manos

# Pipeline con un step que genera un output GRANDE (>60KB) para forzar el artifact.
# Usamos un assemble que concatene módulos repetidos para inflar el tamaño.
PIPE=$(curl -s --max-time 120 http://127.0.0.1:7777/jacobs/pipeline -X POST -H "Content-Type: application/json" \
  -d '{
    "name":"gate-sizefix",
    "objective":"prueba output grande",
    "invoked_by":"Fernando",
    "mode":"autonomous",
    "steps":[
      {"facet":"jax_local","capability":"design","prompt":"Escribí un párrafo de 50 palabras sobre capabilities.","depends_on":[]},
      {"facet":"ada","capability":"assemble","prompt":"manifest","depends_on":[0]}
    ]
  }' | python3 -c "import sys,json; print(json.load(sys.stdin)['pipeline_id'])")
echo "pipeline: $PIPE"
sleep 40
curl -s http://127.0.0.1:7777/jacobs/pipeline/$PIPE | python3 -c "
import sys,json; d=json.load(sys.stdin)
print('pipeline status:', d['pipeline']['status'])
for s in d.get('steps',[]):
    ref = s.get('output_ref') or 'VACIO'
    print('step', s['step_index'], s['capability'], s['status'], '— ref:', ref[:60])
"
sudo journalctl -u jax-las-manos -n 15 --no-pager | grep -iE "ensamble|artifact|error|data too long" | tail -5
```
**Criterio de aceptación:**
- Servicio 'active'.
- El pipeline llega a status 'completed' (NO se cuelga en running).
- El step assemble queda 'completed' con un output_ref (artifact:// o inline corto), NO vacío.
- NO aparece "Data too long" en los logs.

> Para una prueba MÁS fuerte del caso real: re-correr el examen completo después de este fix
> y confirmar que el step assemble (139K) se guarda como artifact y el pipeline completa.
> Eso es opcional en esta misión (tarda); el gate mínimo ya prueba el mecanismo.

## Reporte final
Valor de SIZE_LIMIT antes/después, tipo de output_ref antes/después del ALTER, resultado del
gate (¿el pipeline completó? ¿el assemble se guardó como ref?), incertidumbres,
rollback (*.backup-sizefix-*).
