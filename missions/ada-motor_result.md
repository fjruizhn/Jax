# Resultado de: ada-motor.md

---

Misión completada. Ada está en línea.

**Resumen de lo hecho:**

4 archivos modificados, 4 backups creados. El camino de Ada en el sistema es:
- **REPL / modo tarea** → `config/config.toml` → `HttpMuscle` → `_call_openai` con `api_url=https://api.z.ai/api/paas/v4/chat/completions`
- **LAS MANOS / Motor Registry** → `las_manos/config.toml` → worker genérico → misma URL

**3 incertidumbres declaradas (no resueltas en esta misión):**
1. Guardado en MariaDB no verificado en vivo (pero el código es idéntico al de las otras facetas).
2. `jax-platform` (frontend web) puede tener su propio config — no verificado.
3. El clasificador LLM del router no conoce a Ada — solo invocable explícitamente por nombre por ahora.

🔧 *Origen de autoridad: conocimiento técnico de Hyde. Los comandos ejecutados o verificados se indican en el cuerpo; el resto es de conocimiento, no verificado en vivo.*
