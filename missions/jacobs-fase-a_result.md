# Resultado de: jacobs-fase-a.md

Todo en orden. Reporte final:

---

## Reporte — Jacobs Fase A

### Archivos tocados

| Archivo | Cambio | Backup |
|---|---|---|
| `jacobs/executor.py` | `_invoke_ada`: `stream:False` → `stream:True + max_tokens:131072` + SSE parser | `executor.py.backup-faseA-20260629-034255` |
| `jacobs/plan.py` | `import os`, constantes Ada, `_classify_difficulty`, `_ada_plan`, enrutamiento en `_from_objective` | `plan.py.backup-faseA-20260629-034255` |

### Gate — resultado

| Prueba | Resultado |
|---|---|
| `py_compile executor.py` | ✅ OK |
| `py_compile plan.py` | ✅ OK |
| `systemctl is-active jax-las-manos` | ✅ `active` |
| Objetivo trivial (`resume noticias IA`) | ✅ Plan devuelto — clasificado `[trivial]` → qwen |
| Objetivo formal (`contrato capabilities módulos...`) | ✅ Plan devuelto — clasificado `[formal]` → Ada si key, qwen como fallback |
| Fallback 3 steps genéricos | ✅ Intacto, sigue como red de seguridad final |

### Clasificador verificado en proceso vivo

```
[trivial] (len= 32) 'resume las noticias de IA de hoy'
[formal ] (len=128) 'Genera la especificación formal modular del contrato...'
[trivial] (len= 19) 'hola JAX como estas'
[formal ] (len=120) 'Formaliza la arquitectura de módulos...'
```

### Incertidumbres declaradas

1. **ZHIPU_API_KEY no configurada**: `ZHIPU_API_KEY` no está en `/etc/jax/.env` del proceso activo. Los objetivos formales clasifican correcto pero caen a qwen (comportamiento esperado por diseño). Cuando agregues la key al `.env` y reinicias el servicio, el enrutamiento a Ada se activa automáticamente.

2. **Logs `Jacobs cerebro=` no visibles en journalctl**: El root logger de Python está en `WARNING` — los `logger.info` de `jacobs.plan` no llegan a stdout/journald. No afecta funcionalidad. Se puede resolver agregando `logging.getLogger("jacobs").setLevel(logging.INFO)` en el startup de `server.py` si querés trazabilidad en los logs del sistema.

### Rollback disponible

```bash
cp /home/fruiz/jax/jacobs/executor.py.backup-faseA-20260629-034255 /home/fruiz/jax/jacobs/executor.py
cp /home/fruiz/jax/jacobs/plan.py.backup-faseA-20260629-034255 /home/fruiz/jax/jacobs/plan.py
systemctl restart jax-las-manos
```

🔧 *Origen de autoridad: conocimiento técnico de Hyde. Los comandos ejecutados o verificados se indican en el cuerpo; el resto es de conocimiento, no verificado en vivo.*
