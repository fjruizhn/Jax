# Resultado de: jacobs-v01.md

Jacobs v0.1 está operativo. Resumen de lo que se hizo:

**Creado:** `~/jax/jacobs/` con 8 archivos — `__init__.py`, `models.py`, `artifacts.py`, `policy.py`, `store.py`, `plan.py`, `executor.py`, `routes.py`.

**Integrado:** symlink `las_manos/jacobs → ~/jax/jacobs`, router registrado en `server.py`, `init_tables()` en el startup de la app, `aiomysql` instalado en el venv.

**Todas las pruebas del plan pasaron:**
- `POST /jacobs/plan` → plan sin ejecutar, con dedicatoria
- `autonomous` → 422 claro
- `max_steps=21` → 422 claro
- Kill switch → 423 en plan y pipeline
- Pipeline supervised: step 0 `completed`, step 1 `blocked`, pipeline `interrupted` — esperando `resume`

**Candados duros activos:** invoked_by, max_steps=20, max_parallel=3, modo autonomous bloqueado, kill switch revisado antes de cada step, Hyde siempre requiere human_gate, outputs >1MB van a artifact.

**Nota v0.2:** el `_invoke_jax_local()` devuelve un placeholder — la conexión real al motor de JAX Local va en la próxima versión.

🔧 *Origen de autoridad: conocimiento técnico de Hyde. Los comandos ejecutados o verificados se indican en el cuerpo; el resto es de conocimiento, no verificado en vivo.*
