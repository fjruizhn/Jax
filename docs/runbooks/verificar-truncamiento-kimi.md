# Runbook — verificar si un motor de razonamiento está truncando respuestas

**Cuándo usar:** sospecha de que Kimi (o cualquier motor con `supports_reasoning=true`
en `motor_registry/catalog.py` — hoy Kimi y Ada) devolvió una respuesta cortada.

## Comando

```bash
grep '"motor": "kimi"' /home/fruiz/jax/las_manos/logs/motor_jobs.jsonl | tail -20 | python3 -m json.tool
```

(cambiar `"kimi"` por el motor que corresponda).

## Qué mirar en cada registro `"status": "completed"`

- **`_finish_reason`**: `"stop"` = terminó solo, sano. `"length"` = se quedó sin
  presupuesto y cortó — esto es el síntoma real, no un supuesto.
- **`_usage.completion_tokens_details.reasoning_tokens`** contra
  **`_usage.completion_tokens`**: si el razonamiento se come la mayoría del
  completion y `_finish_reason` es `"length"`, ese es el mecanismo. Si
  `_finish_reason` es `"stop"`, la proporción no importa — terminó bien.

Si no hay registros recientes del motor en cuestión, no hay evidencia para
concluir nada — no asumir que "no truncó" solo porque no se ve el síntoma en
otro lado (journal, frontend). Este archivo es la fuente de verdad para esto.

## Origen — por qué existe este runbook

2026-08-09/10: `kimi-k2.7-code` (modelo de razonamiento) cortaba respuestas a
488 bytes. Causa: `_call_kimi` (`las_manos/motor_registry/worker.py`) nunca
mandaba `max_tokens`, y `reasoning_content` competía por el mismo budget de
completion que `content` — confirmado en vivo contra la API real de Moonshot,
midiendo `reasoning_tokens` contra `completion_tokens` en la respuesta real,
no por hipótesis de transporte/credencial/modelo (las tres se descartaron con
pruebas, no por descarte teórico).

Corregido el mismo día (`017ba2f`): `max_tokens` viaja en el payload cuando el
motor lo declara (`MotorEntry.max_tokens` en `catalog.py`, `las_manos/config.toml`
`[motors.kimi]`/`[motors.ada]` en `8000`), y `_finish_reason`/`_usage` se
capturan y persisten en el job — sin eso, diagnosticar este mismo incidente
la próxima vez requeriría la misma reconstrucción manual (decrypt de key, curl
directo, comparar tamaños de archivo) en vez de un grep.

Verificado vigente el 2026-08-18 (ver `docs/REFORMAS-v3.1.md` §1.4) con dos
muestras separadas por 8 días, ambas `_finish_reason: "stop"`.

## Si vuelve a aparecer `_finish_reason: "length"`

No es un bug de código — el mecanismo de captura y el envío de `max_tokens`
ya existen. Subir el valor de `max_tokens` en `las_manos/config.toml` para el
motor correspondiente; `8000` fue un punto de partida generoso sobre lo
observado en las pruebas del incidente original, no un techo medido contra el
límite real de Moonshot/Z.ai.
