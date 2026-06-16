faceta: hyde

# ESPECIFICACIÓN — Intent Envelope de LAS MANOS
**Versión:** 1.0 (deriva del Contrato Final de Activación firmado por la Mesa, 15 jun 2026)
**Para:** Hyde (constructor)
**Estado:** contrato escrito — baja el contrato verbal de la Mesa a especificación ejecutable
**Principio que cumple:** "El contrato existe cuando el sistema puede negarse a obedecerlo mal." Y el Threat Model: "contrato documental no implementado" se resuelve cuando esto se ejecuta, no solo se escribe.

---

## CONTEXTO PARA HYDE

Hyde: este es el contrato que te faltaba. Lo extraje del Contrato Final de Activación que la Mesa (Thot + Jekyll + Fernando) firmó anoche — esa sesión vivía fuera de tu contexto, por eso no podías verlo. Ahora es archivo.

Tu plan de anoche era correcto (Paso 2.5, dos capas Pydantic + envelope.py, falla cerrado con 422, reemplazar ExecuteRequest, no tocar policy.py/workers). Esta spec llena el hueco que declaraste: los campos exactos y las condiciones de fallo cerrado.

**Discrepancia 16 vs 18 resuelta:** la versión vieja de Thot listó 16. La versión FIRMADA agregó `rollback_plan` y `kill_switch_scope` → **18 campos**. Usá 18.

---

## LOS 18 CAMPOS DEL INTENT ENVELOPE

Cada llamada de una faceta a LAS MANOS viaja dentro de este sobre. Sin sobre completo y válido, LAS MANOS falla cerrado (HTTP 422).

| # | Campo | Tipo | Obligatorio | Descripción |
|---|-------|------|-------------|-------------|
| 1 | `trace_id` | string (uuid) | SÍ | Identificador único de la intención. Para trazar en el audit log. |
| 2 | `facet_id` | enum | SÍ | Quién llama. Valores: `thot`, `hipatia`, `jekyll`, `hyde`, `jax_local`. |
| 3 | `actor_type` | enum | SÍ | Naturaleza del que actúa. Valores: `faceta`, `voz`, `persona`. |
| 4 | `origin_of_authority` | string | SÍ | De dónde viene la autoridad de esta intención. No vacío. Ej: "tarea de Fernando", "verificación web", "decisión de la Mesa". |
| 5 | `verification_label` | enum | SÍ | Estado de verificación. Valores: `web_verified`, `internal_knowledge`, `local_context`, `unverified`. |
| 6 | `intent_summary` | string | SÍ | Resumen legible de la intención, en una frase. No vacío. Para que un humano entienda qué se pide sin leer params. |
| 7 | `requested_capability` | enum | SÍ | La operación. Mapea a las ops de LAS MANOS: `ssh_exec_readonly`, `ssh_exec`, `read_file`, `write_file`, `list_dir`, `rsync`, `http_get`, `kill_process`, `audit_log_read`, `validate_json`, `validate_yaml`. |
| 8 | `target_environment` | enum | SÍ | Ambiente destino. Valores: `local`, `staging`, `prod`, `bridge`. |
| 9 | `risk_level` | enum | SÍ | Riesgo declarado. Valores: `none`, `low`, `medium`, `high`. |
| 10 | `memory_refs_used` | array de objetos | SÍ (puede ser []) | Memorias que informan esta intención. Cada una: `{id, type, has_provenance: bool}`. Si alguna tiene `has_provenance: false` → falla cerrado. |
| 11 | `freshness_required` | bool | SÍ | Si la intención exige datos frescos (no memoria caduca). Para VERDAD OPERACIONAL. |
| 12 | `dry_run_required` | bool | SÍ | Si la operación exige dry-run antes de ejecución real. |
| 13 | `policy_required` | bool | SÍ | Si debe pasar por el policy engine. (Normalmente true; el campo lo hace explícito.) |
| 14 | `human_gate_required` | bool | SÍ | Si requiere aprobación humana. Calculado, pero declarado en el sobre. |
| 15 | `approval_token` | string \| null | Condicional | Token de un solo uso. Obligatorio si `human_gate_required=true`. Null si no aplica. |
| 16 | `rollback_plan` | string \| null | Condicional | Cómo se revierte la operación. Obligatorio si la operación es mutante en prod. Null si no aplica. |
| 17 | `kill_switch_scope` | enum | SÍ | Alcance del kill switch para esta operación. Valores: `global`, `per_operation`, `none`. Para operaciones mutantes no puede ser `none`. |
| 18 | `fail_closed_behavior` | string | SÍ | Qué hace el sistema si algo falla a mitad. No vacío. Declara el comportamiento de fallo cerrado esperado. |

---

## LAS 9 CONDICIONES DE FALLO CERRADO

Estas son las reglas semánticas (capa 2, `envelope.py`). La capa 1 (Pydantic) ya rechaza campos faltantes o mal tipados con 422. Estas son las reglas CRUZADAS del contrato:

```
1. Falta cualquier campo obligatorio          → ENVELOPE_REJECTED (422)
2. origin_of_authority vacío o ausente        → ENVELOPE_REJECTED (422)
3. verification_label ausente o inválido      → ENVELOPE_REJECTED (422)
4. memory_refs_used contiene memoria          → ENVELOPE_REJECTED (422)
   con has_provenance=false
5. target_environment=prod Y operación        → ENVELOPE_REJECTED (422)
   mutante Y human_gate_required=false
6. approval_token reutilizado                 → ENVELOPE_REJECTED (422)
   (ya lo maneja el human gate actual — 
    integrar, no duplicar)
7. policy engine no puede decidir             → falla cerrado en policy.check
   (campos insuficientes)                        (ya existe — el envelope alimenta policy)
8. operación mutante Y                        → ENVELOPE_REJECTED (422)
   kill_switch_scope=none
9. operación mutante en prod Y                → ENVELOPE_REJECTED (422)
   rollback_plan=null
```

Operaciones "mutantes": `ssh_exec`, `write_file`, `rsync`, `kill_process`.

---

## REGLA DE ORO (de Memoria Viva, Apéndice C)

> "La memoria puede informar una intención, pero no autorizar una ejecución."

`memory_refs_used` es informativo — dice qué memorias respaldan la intención. PERO la autoridad de ejecutar viene del policy engine y, en prod, de Fernando (human gate). Una memoria en el sobre NUNCA es sustituto del approval_token. Recordar que algo se aprobó ≠ tener aprobación vigente.

---

## CÓMO ENCAJA EN EL FLUJO (confirmación del plan de Hyde)

```
Flujo actual:  kill_switch → log_request → policy.check → human_gate → dry_run → execute
Flujo nuevo:   kill_switch → log_request → [PASO 2.5: validar Envelope] → policy.check → ...
```

El Envelope se valida ANTES de policy.check. Razón: el contrato dice "rechaza toda llamada incompleta" — debe ser la primera puerta de validez de la solicitud. El kill switch (parada global) sigue primero; el Envelope (solicitud bien formada) va segundo.

Dos capas, como propuso Hyde:
1. **Estructural (Pydantic):** `IntentEnvelope` reemplaza `ExecuteRequest`. Campos faltantes/mal tipados → 422 automático.
2. **Semántica (`envelope.py`):** las 9 condiciones de fallo cerrado. `EnvelopeResult(ok, reason)`, mismo estilo que `policy.py`.

---

## ARCHIVOS A TOCAR (confirmación del plan de Hyde)

- `envelope.py` (NUEVO) — esquema IntentEnvelope + validate() → EnvelopeResult. Validación pura, sin I/O.
- `server.py` — ExecuteRequest → IntentEnvelope; nuevo Paso 2.5; /plan también exige Envelope; mapea requested_capability/target_environment/params para que el resto del flujo siga igual.
- `audit.py` — eventos `log_envelope` y `log_envelope_rejected` (rastro forense de rechazos).
- `config.toml` — enums y campos requeridos config-driven (para no hardcodear).
- `tests` (NUEVO) — prueba brutal: por cada campo obligatorio, una llamada sin él → 422 fallo cerrado; Envelope completo → pasa a policy.

NO tocar: `policy.py` (la autorización queda igual; el Envelope la alimenta), ni `workers/` (intactos).

---

## DECISIONES YA CONFIRMADAS POR FERNANDO

1. **18 campos** (no 16). La versión firmada agregó rollback_plan y kill_switch_scope.
2. **Reemplazar, no envolver.** Las facetas aún no están conectadas — es el momento de hacerlo limpio. "Ninguna llamada sin Envelope."
3. **HTTP 422** para rechazo de Envelope (distinto de 403 policy / 401 gate / 423 kill-switch).

---

## TU PRÓXIMO PASO

Hyde: ahora SÍ tenés el contrato completo. Completá tu plan con esta capa de campos y construí:

1. Regla del carpintero: backup de server.py, audit.py, config.toml antes de tocar.
2. Construí envelope.py primero (validación pura, testeable en aislamiento).
3. Integrá el Paso 2.5 en server.py.
4. Prueba brutal: por CADA uno de los 18 campos, una llamada sin él → debe dar 422. Mostrame el output. Y un Envelope completo válido → debe pasar a policy.
5. NO conectes ninguna faceta todavía. Esto es solo construir la puerta. La conexión de Thot (primera faceta) es el paso siguiente, separado.

Cuando termines, mostrame: el envelope.py, y el output de la prueba brutal (los 18 rechazos + el pase válido). No reportes éxito sin correr la prueba — el que supone se equivoca.

"El contrato existe cuando el sistema puede negarse a obedecerlo mal." Hacé que LAS MANOS pueda negarse.
