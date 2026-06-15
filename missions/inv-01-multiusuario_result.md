# Resultado de: inv-01-multiusuario.md

Saludos. Como Hipatia, guardiana del conocimiento, procederé con la investigación solicitada sobre la arquitectura multiusuario del proyecto `crabfleet` de OpenClaw. La precisión y la referencia a las fuentes son fundamentales para esta labor.

A continuación, presento los hallazgos basados en la información disponible:

### Respuestas sobre la arquitectura multiusuario de crabfleet

El proyecto `crabfleet` de OpenClaw se posiciona como un "centro de control para ejecuciones de agentes" (Mission control for agent runs), diseñado para ser utilizado por mantenedores de OpenClaw en un entorno colaborativo. La información disponible indica un enfoque claro hacia el soporte de múltiples usuarios y la gestión de sus interacciones con el sistema.

#### 1. ¿Cómo maneja crabfleet la autenticación de usuarios? (login, GitHub OAuth, tokens, etc.)

`crabfleet` utiliza **GitHub OAuth para la autenticación de miembros de la organización**. Además, ofrece un **token de arranque (bootstrap token) para la configuración inicial del administrador y para casos de recuperación**. Las sesiones de GitHub OAuth son de corta duración y se verifican, con tokens OAuth cifrados almacenados únicamente en filas de sesión e inyectados en entornos de tiempo de ejecución creados por el servidor. Las claves SSH desconocidas pueden vincularse a una cuenta de GitHub a través de un enlace de OAuth de GitHub para SSH.

*   **URL de la fuente:**
    *   [https://github.com/openclaw/crabfleet](https://github.com/openclaw/crabfleet)
    *   [https://docs.crabfleet.ai/spec](https://docs.crabfleet.ai/spec)

#### 2. ¿Cómo separa o agrupa el trabajo por persona? (vimos "org Codex instances grouped by person" — cómo funciona eso exactamente)

`crabfleet` organiza las instancias de Codex Crabbox "agrupadas por persona" (org Codex instances grouped by person) en un "fleet dashboard". Esto significa que el sistema proporciona una vista consolidada donde cada Crabbox (un entorno de desarrollo desechable) es visible por operador, repositorio, terminal y estado de WebVNC. El flujo de trabajo "fleet-first" permite crear Crabboxes listos para el repositorio desde la aplicación, SSH o la CLI de Go, y ver las instancias de Codex de la organización agrupadas por persona. Esto facilita a los mantenedores de OpenClaw supervisar el trabajo en vivo.

*   **URL de la fuente:**
    *   [https://github.com/openclaw/crabfleet](https://github.com/openclaw/crabfleet)
    *   [https://crabfleet.openclaw.ai/](https://crabfleet.openclaw.ai/)

#### 3. ¿Tiene roles o niveles de permiso distintos para diferentes usuarios? (admin, operador, observador, etc.)

Sí, `crabfleet` implementa **control de acceso basado en roles (RBAC)**. Los roles definidos incluyen:

*   **Owner (Propietario):** Puede gestionar la configuración de la organización, usuarios/equipos, repositorios, límites (caps), secretos y políticas de fusión.
*   **Maintainer (Mantenedor):** Puede crear tarjetas (cards), iniciar/detener/tomar el control de ejecuciones (runs) y aprobar la fusión directa si está permitido.
*   **Viewer (Observador):** Puede ver el tablero (board), abrir registros y adjuntar en modo de solo lectura.

El acceso se gestiona a través de **listas blancas (allowlists) de usuarios y equipos**, y **listas blancas de repositorios**, configurables por el administrador.

*   **URL de la fuente:**
    *   [https://github.com/openclaw/crabfleet](https://github.com/openclaw/crabfleet)
    *   [https://docs.crabfleet.ai/spec](https://docs.crabfleet.ai/spec)

#### 4. ¿Cómo evita que un usuario vea o toque el trabajo de otro? ¿O todos ven todo?

La arquitectura de `crabfleet` agrupa las instancias de Codex por persona, lo que sugiere una clara distinción del trabajo. La visibilidad del "fleet dashboard" permite a los mantenedores ver todos los Crabbox, pero el control de acceso basado en roles (RBAC) y las listas blancas (allowlists) de usuarios y repositorios son mecanismos clave para restringir qué puede hacer cada usuario. Las operaciones que cambian el estado requieren autenticación, y las operaciones de repositorio requieren membresía en la lista blanca. La documentación menciona que los tokens de tiempo de ejecución (runtime tokens) tienen un alcance limitado y son de corta duración, y los secretos nunca se registran ni se almacenan en D1/R2, lo que contribuye a la seguridad y aislamiento.

Aunque el tablero general muestra todas las instancias agrupadas por persona, la capacidad de "ver o tocar" el trabajo de otro se rige por los roles y permisos. Un "Viewer" puede ver el tablero y abrir registros, pero no tomar el control o modificar, a menos que se conceda un "Request control" explícito por parte del propietario o mantenedor de la sesión.

*   **URL de la fuente:**
    *   [https://github.com/openclaw/crabfleet](https://github.com/openclaw/crabfleet)
    *   [https://docs.crabfleet.ai/spec](https://docs.crabfleet.ai/spec)

#### 5. ¿Cómo maneja sesiones concurrentes — varias personas usando el sistema al mismo tiempo?

`crabfleet` está diseñado para la gestión de "flotas de ejecuciones de agentes", lo que intrínsecamente implica el manejo de múltiples interacciones. El sistema permite que múltiples usuarios interactúen con diferentes instancias de Codex o con diferentes tarjetas (cards). La documentación menciona "Concurrent run caps (default: 20)" para las ejecuciones de Codex, lo que indica una capacidad para gestionar un número específico de sesiones activas simultáneamente por organización.

El sistema gestiona la persistencia de sesiones duraderamente, asegurando que un agente no pierda la continuidad de la conversación debido a problemas de infraestructura. Las sesiones pueden tener diferentes modos de "attach" (adjuntar): "Watch" (solo ver), "Take over" (tomar el control con entrada de usuario) y "Share" (compartir para lectura). También existe la posibilidad de "Request control" para que un observador pueda solicitar acceso de escritura a una sesión.

*   **URL de la fuente:**
    *   [https://github.com/openclaw/crabfleet](https://github.com/openclaw/crabfleet)
    *   [https://docs.crabfleet.ai/spec](https://docs.crabfleet.ai/spec)
    *   [https://clawfleet.com/sessions](https://clawfleet.com/sessions)

Confío en que esta investigación detallada satisfaga las necesidades de su consulta, ofreciendo claridad y precisión en cada punto, respaldada por las fuentes consultadas.

— Fuentes consultadas —
  [1] github.com: https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFuV-iKssTM2dXkOOx3zxDklM_SA-4woWnjJK5wTmFjYDYqYYSgaP_MwVUglXfzzz3EbHGE3fRWwu7G9uNvAdNBYTjcPSa_Xfw1YKvbRlI3v3SRKdkUHXXYPk2pTeGoZwU=
  [2] openclaw.ai: https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEbbbDnCkEhmc9opILtzoEMwnjLFt51GdNtU-sCtSmkFN_2m-RJlwOLTZT_2hz5iKpF0hlTEEMlHxSEY3_PKcKLIFS7vDzJ8aG3v12tOUJPtq3hpl13RexBSQ==
  [3] openclaw.ai: https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEhcMiMxucjJ-tTFzLtQsBi_MvmDv5LUCtG-Ir7I-Oduah1fMB6lMft-LT3I5Qpb5cQ0BJvgEWRbq8PjfR_IT3w1sNolHQFPVCvU5HCNjuvJKJSj0_E7lyoqiwNHvnR
  [4] crabfleet.ai: https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQECqKagmbOG7wYirRxGr75mqkK-jehthtq4FenB-e2CALAV5aApwUSNpOoIU84zoOtcNQp81MMOFshKiRvz8ptua4gB9bxREXEsaEzxgQErUjZJ6HOH
  [5] openclaw.ai: https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHnS9at2X7jptpwj7cqV69Qfmqas8qYZGfN2QrsBrj99BWAnSf9W2BpcSDaUqNoM8bLJrCgtqOTwJdBziLg4wsHQ3U4SY9yH5iWDub0bhvmxtSTg_Hr7r1rFA==
  [6] clawfleet.app: https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFdTx6EQeM_L5O4BKOZV_SXCPJ9JrrrktSOJSyYed2Jc_5d-QIj3ai2-bwTG3PybfQIJ6wj5nkbdZjKlj692pwAaZnM9VIiMzK6GWW75gw1TUgJ16gpx0clbZYcoOJWsOaMaw_5WpP2qQ==

(Busquedas: openclaw crabfleet multi-user authentication; openclaw crabfleet user roles permissions; openclaw crabfleet data isolation; openclaw crabfleet concurrent sessions; openclaw crabfleet github; openclaw crabfleet documentation multi-user)
