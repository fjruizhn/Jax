faceta: hyde

# Jacobs Fase A — Fix: alinear nombre de key Ada + activar logs + verificar Ada REAL

> Editar el Jacobs CANÓNICO `~/jax/jacobs/`. HYDE: backup, gate, rollback si falla.

## Causa raíz (verificada)
La key de Z.ai está en `/etc/jax/.env` como `ZAI_API_KEY`, pero Jacobs la busca como
`ZHIPU_API_KEY` (nombre viejo). Por eso Ada nunca se activa y los objetivos formales
caen a qwen. El servicio SÍ carga el .env (EnvironmentFile confirmado).

## Fix 1 — Alinear el nombre en los 3 lugares
Backups: `executor.py.backup-keyfix-$(date +%Y%m%d-%H%M%S)`, `plan.py.backup-keyfix-$(date +%Y%m%d-%H%M%S)`

En `~/jax/jacobs/executor.py` (líneas ~264, ~269) y `~/jax/jacobs/plan.py` (líneas ~93, ~116):
Reemplazar `ZHIPU_API_KEY` → `ZAI_API_KEY` en TODAS las ocurrencias.
```bash
grep -rn "ZHIPU_API_KEY" ~/jax/jacobs/*.py    # localizar todas
# editar cada una: ZHIPU_API_KEY → ZAI_API_KEY
grep -rn "ZHIPU_API_KEY" ~/jax/jacobs/*.py    # debe devolver VACÍO tras el fix
grep -rn "ZAI_API_KEY" ~/jax/jacobs/*.py      # debe mostrar las 3+ ocurrencias nuevas
```
> Robustez extra (PROPUESTO): aceptar ambos nombres por compatibilidad, ej.:
> `api_key = os.environ.get("ZAI_API_KEY") or os.environ.get("ZHIPU_API_KEY", "")`
> Esto evita romper si en el futuro alguna config usa el nombre viejo. Aplicar en los
> puntos de lectura de la key (executor _invoke_ada, plan _ada_plan, y la condición de
> enrutamiento línea 93). Declarar en el reporte si se hizo así o con reemplazo directo.

## Fix 2 — Activar logs INFO de Jacobs (trazabilidad del cerebro usado)
En `~/jax/las_manos/server.py`, en el startup (cerca de donde se configura la app o el
`_jacobs_init`), agregar:
```python
import logging
logging.getLogger("jacobs").setLevel(logging.INFO)
```
> Backup de server.py antes. Esto hace que los `logger.info("Jacobs cerebro=...")` lleguen
> a journald, para ver qué cerebro (Ada/qwen) usó cada plan — la métrica de soberanía.

## Gate (obligatorio) — esta vez Ada DEBE activarse
```bash
sudo systemctl restart jax-las-manos && sleep 8 && systemctl is-active jax-las-manos

# Confirmar que el proceso ve la key ahora:
PID=$(systemctl show -p MainPID --value jax-las-manos)
sudo tr '\0' '\n' < /proc/$PID/environ | grep -i "ZAI_API_KEY" | sed 's/=.*/=<presente>/'

# Objetivo FORMAL → AHORA debe ir a Ada (no caer a qwen)
curl -s http://127.0.0.1:7777/jacobs/plan -X POST -H "Content-Type: application/json" \
  -d '{"objective":"Genera la especificación formal modular del contrato de capabilities con tipos comunes, invariantes y dependencias entre módulos","invoked_by":"fernando","mode":"dry_run"}' \
  | python3 -m json.tool

# Ver en los logs qué cerebro usó (ahora INFO visible)
sudo journalctl -u jax-las-manos -n 40 --no-pager | grep -i "cerebro\|ada\|qwen\|formal\|trivial" | tail -10
```
**Criterio de aceptación:**
- Servicio 'active'.
- El proceso ve `ZAI_API_KEY` (confirmado en /proc/PID/environ).
- El objetivo formal genera un plan Y el log dice que usó **Ada** (no qwen).
- `grep ZHIPU_API_KEY ~/jax/jacobs/*.py` → vacío (o con fallback explícito si se usó la variante robusta).
- Si Ada falla por cualquier razón, el plan igual se genera (cae a qwen) — el servicio no muere.

## Reporte final
Diffs de los 3+ cambios de nombre, el cambio de logging en server.py, salida del gate
(¿el plan formal usó Ada? pegá el log que lo confirma), el plan que generó Ada (¿más
estructurado que los 3 genéricos?), incertidumbres, rollback (*.backup-keyfix-*).
