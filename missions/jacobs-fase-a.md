faceta: hyde

# Jacobs Fase A — Cerebro Ada + enrutamiento por dificultad

> Editar el Jacobs CANÓNICO: `~/jax/jacobs/` (archivo físico).
> `las_manos/jacobs` es un symlink a este — los cambios se reflejan solos. NO tocar el symlink.
> Hipatia inicia con `/using-superpowers` y `/ruflo`.
> HYDE activo: backup antes de editar, gate obligatorio, rollback si falla, declarar incertidumbres.

---

## Contexto y objetivo

Jacobs descompone objetivos en planes con `PlanBuilder._llm_plan` usando qwen3:14b (Ollama).
Qwen es débil para trabajo formal (módulos, dependencias). Meta: que el trabajo FORMAL se
descomponga con **Ada** (glm-5.2), y el trivial siga en local (soberanía).

DOS cambios en esta fase:
- **A1:** portar a `_invoke_ada` del executor los fixes ya probados en base.py
  (max_tokens 131072 + stream True + manejo SSE). Hoy `_invoke_ada` tiene `stream:False`
  sin max_tokens — se truncaría igual que se truncaba el contrato. Sin esto, Ada planificando falla.
- **A2:** PlanBuilder enruta por dificultad: formal→Ada, trivial→qwen (local), fallback intacto.

NO tocar el prompt modular todavía (eso es Fase B). Esta fase: que Ada SEA el cerebro para lo formal.

---

## A1 — Portar fixes de Ada a `~/jax/jacobs/executor.py`

Reconocimiento primero:
```bash
grep -n "import\|httpx\|json" ~/jax/jacobs/executor.py | head
sed -n '262,300p' ~/jax/jacobs/executor.py   # _invoke_ada actual
```
Backup: `executor.py.backup-faseA-$(date +%Y%m%d-%H%M%S)`

En `_invoke_ada`, reemplazar el bloque de payload + request (stream:False, sin max_tokens)
por la versión con streaming y presupuesto completo (mismo patrón que base.py ya en producción):
```python
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "max_tokens": 131072,
    }
    texto = ""
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream("POST", url, headers=headers, json=payload) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                raise RuntimeError(f"Z.ai HTTP {resp.status_code}: {body[:200]!r}")
            partes = []
            async for linea in resp.aiter_lines():
                if not linea or not linea.startswith("data:"):
                    continue
                payload_str = linea[5:].strip()
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
```
> Asegurar `import json` arriba del archivo (probablemente ya está — verificar).
> El resto de `_invoke_ada` (chequeo de api_key, return dict) queda IGUAL.
> NOTA: razonamiento de Ada queda ENCENDIDO (sin thinking:disabled) — para planificar,
> el razonamiento ayuda. El presupuesto de 131072 lo cubre.

## A2 — Enrutamiento por dificultad en `~/jax/jacobs/plan.py`

Backup: `plan.py.backup-faseA-$(date +%Y%m%d-%H%M%S)`

### A2.1 — Función de invocación a Ada para planificar
Agregar en `plan.py` un método/función que llame a Ada para descomponer (reusa el patrón
de `_invoke_ada` del executor, con streaming + max_tokens 131072). Lee `ZHIPU_API_KEY` del env.
Devuelve el texto crudo de Ada (que `_parse_plan_json` luego procesa — el parser EXISTENTE
ya extrae el array JSON de texto con markdown, se reutiliza tal cual).

### A2.2 — Clasificador de dificultad (explícito y auditable, NO mágico)
Agregar un método `_classify_difficulty(objective: str) -> str` que devuelva "formal" | "trivial":
- Heurística simple y declarada (PROPUESTA, ajustable por Fernando):
  - "formal" si el objetivo contiene señales de trabajo estructurado: longitud > 200 chars,
    o keywords como {"contrato","módulo","módulos","invariante","esquema","arquitectura",
    "especificación","spec","tipos comunes","dependencias","formaliza","capabilities"}.
  - "trivial" en caso contrario.
- Declarar en un comentario que esta heurística es v1 y se refinará (Fase D la mejora con
  ejemplos de oro). NO inventar un clasificador LLM todavía — heurística transparente.

### A2.3 — Enrutamiento en `_from_objective` / `_llm_plan`
Modificar el flujo para que:
```
dificultad = _classify_difficulty(objective)
if dificultad == "formal" and ZHIPU_API_KEY presente:
    specs = await _ada_plan(objective, max_steps)      # Ada descompone
    if not specs:                                       # Ada falló
        specs = await _llm_plan(objective, max_steps)   # cae a qwen local
else:
    specs = await _llm_plan(objective, max_steps)       # qwen local (trivial)
if not specs:
    specs = _fallback_plan(objective)                   # red de seguridad EXISTENTE
```
> El fallback de 3 steps genéricos se MANTIENE intacto como última red.
> Loggear qué cerebro se usó (logger.info) para trazabilidad y para la métrica de soberanía futura.
> Dejar un comentario marcando el punto donde la Fase D capturará el plan de Ada como ejemplo de oro.

## Gate (obligatorio)
```bash
# El servicio debe seguir arrancando tras los cambios
sudo systemctl restart jax-las-manos && sleep 8 && systemctl is-active jax-las-manos

# 1. Objetivo TRIVIAL → debe rutear a qwen local (ver log)
curl -s http://127.0.0.1:7777/jacobs/plan -X POST -H "Content-Type: application/json" \
  -d '{"objective":"resume las noticias de IA de hoy","invoked_by":"fernando","mode":"dry_run"}' \
  | python3 -m json.tool | head -30

# 2. Objetivo FORMAL → debe rutear a Ada (ver log; plan más estructurado)
curl -s http://127.0.0.1:7777/jacobs/plan -X POST -H "Content-Type: application/json" \
  -d '{"objective":"Genera la especificación formal modular del contrato de capabilities con tipos comunes, invariantes y dependencias entre módulos","invoked_by":"fernando","mode":"dry_run"}' \
  | python3 -m json.tool | head -40

# 3. Ver en los logs qué cerebro se usó en cada caso
sudo journalctl -u jax-las-manos -n 30 --no-pager | grep -i "cerebro\|ada\|qwen\|plan\|dificultad" | tail -10
```
**Criterio de aceptación:**
- Servicio 'active' tras los cambios.
- El objetivo trivial rutea a qwen (log lo confirma), devuelve un plan.
- El objetivo formal rutea a Ada (log lo confirma), devuelve un plan (idealmente más steps/estructura).
- El fallback sigue existiendo (no se rompió).
- Si Ada falla, cae a qwen sin tumbar el pipeline.

## Reporte final
Archivos tocados + diffs (executor.py _invoke_ada, plan.py clasificador+enrutamiento),
resultado del gate (los dos planes + qué cerebro usó cada uno), incertidumbres,
rollback disponible (*.backup-faseA-*).
