# LAS VOCES v0.1.0

Paquete inicial del proyecto paralelo de Axioma 3.0.

- `index.html`: plan completo + dashboard.
- `project.json`: fuente estructurada.
- `PROJECT_CHARTER.md`: charter.
- `AGENT_ARIADNA.md`: PM Agent propuesto.
- `SYNC_CONTRACT.md`: sincronización Codex / Claude Code / Qwen.
- `agents/*.json`: entradas canónicas de proyección LV-002.
- `schemas/agent.schema.json`: schema estricto de agente.
- `projections/`: contrato, detector/generador y pruebas de sandbox LV-002.

El HTML funciona autónomamente con snapshot embebido. En Axioma debe servirse junto a `project.json` para que el dashboard refleje el estado actualizado por sync.
