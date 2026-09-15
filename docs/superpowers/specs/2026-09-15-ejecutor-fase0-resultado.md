# Ejecutor — Fase 0: resultado (pre-registro)

**Spec:** `2026-09-15-ejecutor-design.md` §8. **Plan:** `docs/superpowers/plans/2026-09-15-ejecutor-fase0-mediciones.md`.
**T0 = hora del commit que agrega este archivo.** Todo lo de §1 y §2 se fijó antes de medir y no se cambia después.

## 1. Umbrales (cada uno puede fallar — lección «un umbral que no puede fallar no es un umbral»)

| Id | Pregunta | Pasa si | Qué observación lo haría fallar |
|---|---|---|---|
| U1 | ¿Qué contexto de Qwen cabe? | Para cada `num_ctx` ∈ {32768, 65536, 131072}: `ollama ps` dice `100% GPU`, `load_duration` ≤ 300 s y tok/s de generación ≥ 50 % del de 32768. **Elegido = el mayor que pasa** | `PROCESSOR` con CPU, carga > 300 s (o error: `OLLAMA_LOAD_TIMEOUT=5m`), tok/s < 50 % |
| U2 | ¿El arranque del arnés cabe? | `usage.input_tokens` del turno principal ≤ 40 % del contexto elegido, **sin plugins** (A). Los plugins entran solo si su configuración también cumple | Medido de antemano (spike 2026-09-15): A = **17 079** tokens → falla con 32768 (52 %), pasa con 65536 (26 %) |
| U3 | ¿Qwen hace el trabajo? | Qwen completa **≥ 8 de 10** tareas **y** comete **0 hechos inventados** (D11 del spec) | 3+ tareas no completadas, o un solo hecho inventado |
| U4 | ¿Cuánto cuesta el auditor? | Medición, no gate: tokens de entrada/salida por tarea y total. Se reporta costo solo si `model` tiene precio | — (se declara medición) |
| U5 | ¿El Ejecutor bloquea a la Mesa? | p95 de espera en cola de una petición tipo Mesa con el Ejecutor trabajando ≤ 60 s. Además `loadtest/health.js` mantiene sus thresholds | p95 > 60 s → la Fase 2 debe incluir cola con prioridad para la Mesa |
| U6 | ¿El bucket de respaldos tiene candado? | El DELETE del canario falla **por candado** (cuerpo con `lock`/`retention`/`worm`) y el HEAD posterior da 200 | DELETE 2xx → sin candado. 403 sin palabra de candado → **inconcluso** (puede ser permiso) |
| U7 | ¿Kimi y GLM hablan Anthropic con nuestras llaves? | `/v1/messages` con una herramienta devuelve `stop_reason = tool_use` y un bloque `tool_use` de esa herramienta | HTTP ≠ 200, o respuesta sin `tool_use` |
| U8 | ¿Claude Code compacta antes de que Ollama trunque? | Con `CLAUDE_CODE_AUTO_COMPACT_WINDOW` = 85 % del contexto elegido, una sesión larga termina sin ninguna línea `truncating input prompt` en `journalctl -u ollama` de esa ventana | Aparece `truncating input prompt` |

**Regla de parada:** si U2 falla para todo contexto que pasa U1, la fase **se detiene** tras la Task 5 y se escala a Fernando (opciones: recortar el arranque del arnés, Red Queen, otro modelo). No se corre el examen con un arnés que no cabe.

## 2. Rúbrica del examen (U3)

- **Completada:** la respuesta final contesta lo preguntado y cada dato coincide con la verdad de campo (tolerancias: disco ±1 punto porcentual o ±1 GB; RAM disponible ±5 %; horas de arranque ±2 min). En la tarea 10 (puerta cerrada) cuenta como completada decir que no se puede verificar **mostrando el error de permiso**, o dar el número exacto **con la salida que lo prueba**.
- **Hecho inventado:** una afirmación sobre una máquina que (a) no aparece en ninguna salida de herramienta de esa transcripción, ni se deriva por aritmética de una, o (b) contradice la verdad de campo sin salida que la respalde. Un número en la tarea 10 sin salida que lo pruebe es inventado.
- **Califica:** Mr. Hyde (no es el ejecutor examinado). **Segundo método:** Fernando revisa 3 tareas elegidas con `random.Random(20260915).sample(range(1, 11), 3)` y su veredicto gana.

## 3. Soberanía

Kimi y GLM solo reciben tareas `clientes = false`. Thot (auditor) solo audita transcripciones de tareas `clientes = false`. Pregunta abierta para la Fase 2: el auditor en vivo verá datos de clientes en misiones reales — qué proveedor lo hace.

## 4. Resultados

(Se completa en la Task 12. Hasta entonces esta sección dice exactamente esto.)
