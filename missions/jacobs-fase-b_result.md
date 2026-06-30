# Resultado de: jacobs-fase-b.md

---

## Estado actual

**Aplicado en disco (sin activar — servicio no reiniciado aún):**

| Archivo | Cambio | Estado |
|---|---|---|
| `models.py` | `depends_on: list[int]` en `Step` | ✅ en disco, py_compile OK |
| `plan.py` | `_PLAN_SYSTEM_MODULAR` nuevo | ✅ en disco |
| `plan.py` | user-prompt Ada con ejemplo de 5 steps | ✅ en disco |
| `plan.py` | parser propaga `depends_on`, filtra OOB | ✅ en disco |
| `plan.py` | `_from_spec` pasa `depends_on` al `Step` | ✅ en disco |

**Pendiente:**
- `sudo systemctl restart jax-las-manos` — necesito que lo corras vos
- Gate: curl de objetivo formal → verificar patrón modular

**Rollback disponible:**
```
/home/fruiz/jax/jacobs/models.py.backup-faseB-20260629-040358
/home/fruiz/jax/jacobs/plan.py.backup-faseB-20260629-040358
```

Una vez que reinicies el servicio, corro el gate y reporto el plan crudo que devuelva Ada.

🔧 *Origen de autoridad: conocimiento técnico de Hyde. Los comandos ejecutados o verificados se indican en el cuerpo; el resto es de conocimiento, no verificado en vivo.*
