# Resultado de: inv-crabfleet.md

Soy Hipatia, guardiana del conocimiento, y he investigado el repositorio `crabfleet` de OpenClaw. A continuación, presento las respuestas a sus preguntas, obtenidas directamente de las fuentes disponibles:

Aquí tiene la investigación acotada sobre el repositorio `crabfleet` de OpenClaw:

---

### INVESTIGACIÓN ACOTADA — Mission Control de OpenClaw

**1. ¿Qué es crabfleet exactamente y qué problema resuelve? ¿Es un TUI, una web UI, o ambos?**

`crabfleet` es el "control de misión para ejecuciones de agentes" (mission control for agent runs) de OpenClaw. Ofrece un panel de control que permite a los mantenedores de OpenClaw visualizar cada "Codex crabbox" (instancia de agente) por operador, repositorio, terminal y estado de WebVNC. Su propósito es proporcionar una interfaz unificada para supervisar y gestionar un "fleet" (flota) de agentes de IA.

Resuelve el problema de la visibilidad y el control sobre múltiples instancias de agentes, facilitando flujos de trabajo orientados a la flota y a tableros, control de ejecuciones, cajas de arena interactivas y gestión de credenciales.

Es tanto una **Web UI** como compatible con la **interacción por terminal (SSH y CLI)**. La aplicación web canónica se encuentra en `https://crabfleet.openclaw.ai/app`, mientras que también se puede interactuar con ella a través de SSH (`ssh link@crabd.sh` o `ssh crabd.sh new`) o una CLI en Go.

*   **Fuente:**

**2. ¿Cómo muestra el estado de los agentes corriendo? (vistas, paneles, qué información presenta en pantalla)**

`crabfleet` presenta una "fleet dashboard" (panel de control de flota) donde cada instancia de "Codex crabbox" es visible. Muestra los "org Codex instances grouped by person" (instancias de Codex de la organización agrupadas por persona). El flujo de trabajo basado en tableros permite seguir las "cards" (tarjetas) a través de los estados "Todo", "Running", "Human Review" y "Done" (Pendiente, En Ejecución, Revisión Humana y Completado).

En las tarjetas se muestra la siguiente información:
*   Título de la tarjeta.
*   Una insignia de la fuente (prompt, issue, PR).
*   Una insignia del repositorio.
*   Un "chip" de estado.
*   Un "chip" de política de fusión cuando no es el predeterminado.
*   Un "chip" de tiempo de ejecución cuando no es automático.
*   Un temporizador activo mientras se ejecuta.
*   Un resumen del último evento.
*   Un botón para "Attach" (adjuntar) cuando está en vivo.
*   Un enlace a la Pull Request (PR) cuando esté disponible.

Las tarjetas en ejecución muestran registros de eventos (D1 event logs) y el estado de los "heartbeats" (latidos). Al hacer clic en "Attach" (adjuntar), se abre una cuadrícula de sesión de Ghostty WASM a pantalla completa. La interfaz muestra la política de fusión como un "chip" en la tarjeta, con un icono sutil de "garra/verificación" cuando está activado.

*   **Fuente:**

**3. ¿Cómo maneja la aprobación humana (human-in-the-loop) y la cancelación/parada de tareas?**

`crabfleet` integra la "Revisión Humana" (Human Review) como una de las etapas en su flujo de trabajo basado en tableros para las "cards". La política de fusión (merge policy) se almacena y se muestra, pero `crabfleet` no fusiona Pull Requests (PRs) directamente; en su lugar, `ClawSweeper` (otro proyecto de OpenClaw) sigue disponible y es preferido para los ciclos de revisión/corrección/fusión automática. El control de ejecución de Codex permite iniciar intentos de ejecución duraderos, rastrear latidos, observar la cuadrícula de sesiones de Ghostty WASM y tomar el control solo cuando el tiempo de ejecución seleccionado lo permite. En el flujo de ejecución, una tarjeta puede pasar al estado "Human Review". También permite "stop legacy sessions locally, and clean up retained Codex session history" (detener sesiones antiguas localmente y limpiar el historial de sesiones de Codex retenidas).

*   **Fuente:**

**4. ¿Qué tecnología usa para la interfaz? (framework, lenguaje, si es terminal o navegador)**

La interfaz web de `crabfleet` se construye con **Cloudflare Workers** para la aplicación, API, autenticación, búsqueda en GitHub y rutas de documentación. Utiliza **TypeScript** como lenguaje principal. El `app.html` es el "Single-page app shell and styles", y los módulos de la aplicación (app/) están escritos con **Preact**, utilizando **Vite** para la configuración del paquete de la aplicación. La interfaz de línea de comandos (CLI) se puede usar a través de **SSH** y también existe una **CLI en Go**. Para las sesiones interactivas en el navegador, utiliza una "Ghostty WASM session grid" (cuadrícula de sesión Ghostty WebAssembly).

*   **Fuente:**

**5. ¿Cómo se comunica con el daemon/gateway que ejecuta los agentes? (WebSocket, REST, etc.)**

`crabfleet` utiliza **REST polling** para la actualización del estado del tablero y de la flota. Sin embargo, los **WebSockets** se emplean para el tráfico del terminal y el "relay" del ejecutor de GitHub Actions, no para la distribución general del tablero. La documentación del proyecto menciona una sección para la **API (REST y WebSocket APIs)**. El `crabfleet-control` parece ser un componente clave para la comunicación, y la creación de "crabboxes" y el registro de trabajo de GitHub Actions fluyen a través del adaptador configurado `Crabbox/ClawFleet`.

*   **Fuente:**

— Fuentes consultadas —
  [1] github.com: https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQG8boBv-cRUR0nm53eXBLJKOAzQxbVNHjalhUs9LJ0FvWsYEK_dGzOI_ldAHal29Nu0hjOMUv8FR2DTx56XzYbSbacz_WOrsuHYmLHWlYpCvkHSLdAyqBb8I5E3Ziw0pQ==
  [2] openclaw.ai: https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEkUmHhb2pthENmhX6au3MV5TE7frtumvZnMC9D5PSQfejBamBWLJHj3ZxVykOxFmobS3uq66kyA_xGw1xAqeVqDxDt0rE_Pkb3XqBzY4uRAKXFJAvD42ly
  [3] crabfleet.ai: https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEU95_R7KtYpXTs9BYdb6dR6G7vT_iC56RgGt5C-mC948F69LJpXviYlKUBUxmiA1jcor0VMNC9_NY9nO5cge-ydZ2d-L89Pka-aiF-kV2DilBdNxE=
  [4] openclaw.ai: https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFcwmsNqV4lI33NfDXgAeoj14Jt0-S2iBQXV5n091-EADVVdrVzxixGBZCPj9lVb0IHLX9g7eXk2G_4ruKMaNO1YGrAqFURbVPNDbkXzSGtUQbz7ugPt1_zBquTwyc=
  [5] github.com: https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFDCNoUiS8iuugJVAXeyURavNaspl1IN3jdgCBmwLOI0AKySQPMG3neaX3pwTjpZYz2MPylnOemWd8eEEO5L-j7ebqI_Oixx-TNbMcnGAcvvQGc0XuqjCvFiTttcwHUIw7D
  [6] github.com: https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEhO6XZ9rBkv0DYIZDqV3Vxk3dsSQMjqjEwMWCHOHBjQJUlroOXVznR_ZYkj0mjLhf0Thf3XTuvbsXmTt-Brs8q8pydeFaHcmBVOfirr2GDtCqQccYN
  [7] github.com: https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFWoWAzgWjiVCZI2NT2i2X6Yr9NWgaNtOoYfCrvKEa1Tc-A-02EgZOz8_WadZgjZrlpsZOeaVrLGiT65V-ymbRvgBGcyBHlKLhFVICjP0znm-0pUnJpv0Q2eA==
  [8] github.com: https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFQZRGE5Xpu-28a8U-EOyVe93K5GfVRA4gyBWVywGyZvBq7ir1Bi8AwZKjcZ-3SHWnuwVNqL02zs_t8e7vAjiBkijz01R77vL5BsnYjC-uFoQGdguizB1Q=

(Busquedas: OpenClaw crabfleet GitHub; github.com/openclaw/crabfleet "mission control for agent runs"; crabfleet TUI web UI; crabfleet agent status display; crabfleet human approval cancellation tasks; crabfleet interface technology; crabfleet communication daemon gateway WebSocket REST)
