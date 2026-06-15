faceta: hipatia

# INV-06 — Salas de control multi-usuario: TUI y WebUI

Investiga patrones de diseño para interfaces de "sala de control" (mission control) que soporten MÚLTIPLES usuarios. Übermensch, nuestra sala de control, debe permitir que varias personas interactúen, no solo un operador.

Responde SOLO estas preguntas, con URL de fuente para cada una. Sé concreta:

1. Para una sala de control que soporte varios usuarios: ¿conviene TUI (terminal) o WebUI (navegador)? ¿Qué hacen los proyectos serios? Pros y contras de cada uno para multi-usuario.

2. ¿Cómo manejan los dashboards multi-usuario el estado en tiempo real compartido? (todos ven lo mismo actualizado — WebSocket, polling, server-sent events)

3. Ejemplos concretos de salas de control/dashboards open source multi-usuario que podamos estudiar (con URLs de repos). Busca específicamente herramientas de orquestación de agentes o de operaciones (tipo Rundeck, Temporal UI, Airflow UI, etc.)

4. Para el framework Textual (Python TUI): ¿puede servir a múltiples usuarios? ¿Existe "textual-web" o forma de exponer un TUI por navegador a varios usuarios a la vez?

5. ¿Cuál es el patrón recomendado para combinar lo mejor de ambos: un núcleo (daemon/API) que sirva tanto a un TUI local como a una WebUI remota? (vimos que crabfleet hace esto — thin client sobre un daemon)

Para cada respuesta: URL de la fuente. Si no encontrás algo, decílo con honestidad — no inventes.
