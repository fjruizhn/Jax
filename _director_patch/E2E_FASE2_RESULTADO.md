# Fase 2 — Test de integración E2E del wave scheduler contra el contrato cerrado

> **Evidencia cruda, no veredicto.** El veredicto lo dan Fernando y el arquitecto.
> **Rama:** `feat/director-orquesta-waves` · **NO mergeado** (esto es validación).
> **Fecha:** 2026-06-30 · **Pipeline:** `b006be85-fd25-42d0-8e47-e2db85479b82`

---

## 0. Estado del servicio (verificado, no asumido)

```
MainPID=246948
ExecMainStartTimestamp=Tue 2026-06-30 07:02:54 CST   (posterior a los edits de Fase A: 06:46–06:56)
/health → {"service":"LAS MANOS","status":"alive","kill_switch_active":false}
```
El arranque post-edición garantiza que el catálogo nuevo (6 capabilities + ajustes) y el
`executor.py`/`plan.py` refinados están VIVOS en memoria.

## 1. Payload

```json
{
  "name": "e2e-olas-contrato",
  "invoked_by": "Fernando",
  "mode": "autonomous",
  "max_steps": 3,
  "steps": [
    {"facet": "ada",  "capability": "generate",             "depends_on": [],     "prompt": "Devolvé solo el número 1"},
    {"facet": "kimi", "capability": "generate",             "depends_on": [],     "prompt": "Devolvé solo el número 2"},
    {"facet": "thot", "capability": "validate_consistency", "depends_on": [0, 1], "prompt": "Auditá los outputs de step 0 y step 1"}
  ]
}
```
Olas esperadas: `[[0,1],[2]]`. Clean-room esperado: limpio.

## 2. Respuesta del POST /jacobs/pipeline

```json
{"pipeline_id":"b006be85-fd25-42d0-8e47-e2db85479b82","status":"running","mode":"autonomous","step_count":3,"message":"Pipeline iniciado. Consultar GET /jacobs/pipeline/{id}"}
__HTTP:200__
```

## 3. Events crudos (GET /jacobs/pipeline/{id}/events, ordenados por id, con ts)

```
id=305 ts=1782824654.282 PIPELINE_CREATED    {"name": "e2e-olas-contrato", "mode": "autonomous", "steps": 3}
id=306 ts=1782824654.284 PIPELINE_STARTED    {}
id=307 ts=1782824654.284 WAVE_STARTED        {"wave": 0, "steps": [0, 1], "parallel": 2}
id=308 ts=1782824654.287 STEP_STARTED        {"step_index": 0, "facet": "ada",  "capability": "generate"}
id=309 ts=1782824654.287 STEP_STARTED        {"step_index": 1, "facet": "kimi", "capability": "generate"}
id=310 ts=1782824659.356 STEP_COMPLETED      {"step_index": 1, "output_ref": "inline:{...kimi... job_id 225cd482 status completed capability generate caller jacobs result \"2\"}"}
id=311 ts=1782824662.814 STEP_COMPLETED      {"step_index": 0, "output_ref": "inline:{\"success\": true, \"facet\": \"ada\", \"model\": \"glm-5.2\", \"result\": \"1\"}"}
id=312 ts=1782824662.816 WAVE_COMPLETED      {"wave": 0, "steps": [0, 1]}
id=313 ts=1782824662.817 WAVE_STARTED        {"wave": 1, "steps": [2], "parallel": 1}
id=314 ts=1782824662.819 STEP_STARTED        {"step_index": 2, "facet": "thot", "capability": "validate_consistency"}
id=315 ts=1782824669.960 STEP_COMPLETED      {"step_index": 2, "output_ref": "inline:{...thot gpt-5.5 audit...}"}
id=316 ts=1782824669.963 WAVE_COMPLETED      {"wave": 1, "steps": [2]}
id=317 ts=1782824669.964 PIPELINE_COMPLETED  {}
```

### Comprobaciones pedidas (derivadas de los ts crudos)

| Comprobación | Evidencia | Resultado |
|---|---|---|
| `parallel=2` en ola 0 | event 307 `{"wave":0,"steps":[0,1],"parallel":2}` | ✅ |
| Steps 0 y 1 concurrentes | `STEP_STARTED` ambos en ts **654.287** | ✅ |
| `WAVE_COMPLETED` de cada ola | events 312 (ola 0) y 316 (ola 1) | ✅ |
| Ola [0,1] termina ANTES de arrancar ola [2] | `WAVE_COMPLETED` ola0 = **662.816** < `WAVE_STARTED` ola1 = **662.817** | ✅ thot esperó a ada+kimi |
| Sin `STEP_FAILED` / `PIPELINE_ABORTED` | no aparecen en el stream | ✅ |

## 4. Results crudos (GET /jacobs/pipeline/{id}/results)

```
pipeline status: completed | total_duration_s: 15.69
  step 0 ada/generate              status=completed dur=8.53s err=None  result='1'
  step 1 kimi/generate             status=completed dur=5.07s err=None  result='2'
  step 2 thot/validate_consistency status=completed dur=7.14s err=None  result='## Auditoría de consistencia…'
```

### Output completo de thot (step 2), crudo

```markdown
## Auditoría de consistencia — `validate_consistency`

### Evidencia cruda recibida

```text
--- Dependencia: step 0 (ada) ---
1

--- Dependencia: step 1 (kimi) ---
2
```

### Resultado

**FAIL — Inconsistencia detectada.**

Los outputs de las dos ramas del fan-out no coinciden:

- `step 0 (ada)` produjo: `1`
- `step 1 (kimi)` produjo: `2`

Bajo un contrato de `validate_consistency`, esto no puede considerarse consistente salvo que exista una regla explícita que permita diferencias entre ramas. Esa regla **no fue provista**.

### Incógnitas

- **INCÓGNITA:** cuál era el valor esperado correcto.
- **INCÓGNITA:** si `1` o `2` es válido individualmente.
- **INCÓGNITA:** si el contrato cerrado define tolerancias, equivalencias o normalización entre outputs.
- **INCÓGNITA:** si alguna dependencia falló parcialmente antes de emitir su salida.

### Veredicto

El pipeline E2E no pasa la validación de consistencia porque `ada` y `kimi` generaron salidas divergentes.
```

## 5. Clean-room (journalctl)

```
(sin líneas clean-room — plan limpio)
```
thot (auditor) ≠ ada/kimi (productores) → sin violación. ✅

## 6. kimi/generate NO rechazado — bug original muerto

```
Jun 30 07:04:14  uvicorn[246948]: "POST /motor/dispatch HTTP/1.1" 202 Accepted
Jun 30 07:04:19  uvicorn[246948]: "GET /motor/job/225cd482-… HTTP/1.1" 200 OK
```
- Dispatch a Motor Registry → **202 Accepted** (no 4xx).
- Job `status=completed`, `capability=generate`, `caller=jacobs`, `result="2"`.
- grep de `rechaz|reject|Capability desconocida` → sin líneas.

✅ El contrato cerrado de Fase A funciona en el wiring real. El `PIPELINE_ABORTED` latente está muerto.

---

## 7. Declaración honesta (no maquillada)

- **Wiring: 100% exitoso.** Dos olas topológicas, paralelismo intra-ola (`parallel=2`), orden DAG
  respetado (thot esperó a sus deps), kimi vía Motor Registry, los 3 steps `completed` sin error,
  cierre en `PIPELINE_COMPLETED`. Duración total 15.69s.

- **El "FAIL" del contenido de thot NO es un fallo del pipeline.** thot dice "FAIL — inconsistencia"
  porque ada devolvió `1` y kimi `2` — exactamente lo que el payload les pidió (números distintos).
  Es el veredicto **semántico** del LLM auditando dos valores divergentes, no un error de ejecución:
  el step terminó `status=completed`, sin `error`, y el pipeline cerró OK. thot hizo su trabajo
  (declaró incógnitas, no supuso). Para un e2e con "PASS" de contenido habría que pedir a ada y kimi
  outputs coherentes entre sí — pero eso es cosmético; el wave scheduler y el contrato ya quedaron probados.

- **NO se mergeó.** El merge lo decide Fernando tras ver que el wiring real corre.
