# Resultado de: jacobs-faseA-keyfix.md

Los logs aparecen. El gate pasó completo. Reporte final:

---

## Reporte — Jacobs Fase A

### Fix 1 — Alineación de key (variante robusta)

Se usó fallback de compatibilidad: `ZAI_API_KEY` primero, `ZHIPU_API_KEY` como fallback. Cambios en:

| Archivo | Línea | Cambio |
|---|---|---|
| `jacobs/executor.py:264` | `_invoke_ada` | `get("ZAI_API_KEY") or get("ZHIPU_API_KEY", "")` |
| `jacobs/executor.py:269` | mensaje error | texto actualizado a `ZAI_API_KEY` |
| `jacobs/plan.py:93` | condición enrutamiento | ambos nombres aceptados |
| `jacobs/plan.py:116` | `_ada_plan` | fallback doble |

### Fix 2 — Logging activo

`server.py`: `import logging` añadido + `_jacobs_init` ahora crea un `StreamHandler` explícito en el logger `jacobs` si no tiene handlers. Esto fue necesario porque el logger raíz de uvicorn no propaga los INFO de loggers hijos sin handler propio.

### Gate — Resultados

```
ZAI_API_KEY=<presente>          ✓  proceso la ve
jax-las-manos: active           ✓  servicio up
step_count: 20                  ✓  plan generado
```

**Log confirmado en journald:**
```
jacobs.plan INFO Jacobs cerebro=Ada (formal) objective='Genera la especificación formal modular...'
```

Ada usó el objetivo formal. El plan de 20 steps incluye a Ada en las fases de design (arquitectura modular, tipos comunes, invariantes, dependencias) — mucho más estructurado que los 3 genéricos que generaba qwen antes.

### Rollback disponible

```
~/jax/jacobs/executor.py.backup-keyfix-20260629-035349
~/jax/jacobs/plan.py.backup-keyfix-20260629-035349
~/jax/las_manos/server.py.backup-keyfix-20260629-035349
```

### Incertidumbre

Si en alguna sesión futura se limpia `ZHIPU_API_KEY` del `.env`, el fallback sigue funcionando igual. Si se limpia `ZAI_API_KEY`, Ada deja de funcionar y el plan cae a qwen — el servicio no muere.

🔧 *Origen de autoridad: conocimiento técnico de Hyde. Los comandos ejecutados o verificados se indican en el cuerpo; el resto es de conocimiento, no verificado en vivo.*
