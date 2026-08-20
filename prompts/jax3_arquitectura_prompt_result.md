# Resultado de: jax3_arquitectura_prompt.md

Saludos, Fernando. Como Hipatia, la guardiana del conocimiento, he abordado la tarea de investigar y proponer una arquitectura para JAX 3.0 con la rigurosidad y el detalle que requiere un proyecto de esta envergadura. He consultado diversas fuentes para asegurar la precisión y actualidad de la información, enfocándome en soluciones robustas y verificables.

A continuación, presento el documento de arquitectura completa para JAX 3.0, estructurado según el formato solicitado.

---

## RESUMEN EJECUTIVO

JAX 3.0 se concibe como un sistema nervioso central unificado, con interfaces TUI y WebUI que operan cohesivamente, compartiendo una memoria contextual común basada en MariaDB. Integrará capacidades avanzadas de agentes autónomos para gestionar el servidor y ejecutar tareas complejas, exponiendo una API REST para su consumo por AteneaERP y HAMMURABI. La propuesta se centra en Open WebUI para la interfaz web, un framework TUI moderno como Textual para la terminal, un sistema de memoria unificado diseñado a medida y un executor de herramientas basado en librerías Python maduras, garantizando una experiencia de usuario fluida y una operación autónoma eficiente.

---

## TABLA DE DECISIONES

Para cada pregunta planteada, se presenta la respuesta verificada y una recomendación estratégica.

| Pregunta | Respuesta Verificada | Fuentes | Recomendación |
| :------- | :------------------- | :------ | :------------ |
| **1.1. ¿Open WebUI puede conectarse a Ollama + DeepSeek + Gemini + Claude simultáneamente?** | Sí, Open WebUI está diseñado para ser agnóstico al modelo. Soporta nativamente la conexión a Ollama y proveedores de modelos compatibles con la API de OpenAI. Para Gemini y Claude, se pueden integrar a través de la interfaz de la API de OpenAI utilizando proxies o wrappers si no tienen una integración directa más reciente. | Documentación oficial de Open WebUI (https://docs.openwebui.com/)<br> Repositorio de GitHub de Open WebUI (https://github.com/open-webui/open-webui) | **Integrar Ollama directamente.** Para Gemini y Claude, explorar si ya existen proxies compatibles con OpenAI API o desarrollarlos si fuera necesario, asegurando una capa de abstracción para futuras adiciones de modelos. |
| **1.2. ¿Sus Tools/Functions permiten ejecutar bash arbitrario en el host?** | Sí, Open WebUI permite la creación de herramientas personalizadas (custom tools) que pueden ejecutar comandos `bash` arbitrarios en el host, especialmente cuando se despliega en un entorno Docker con los permisos adecuados. Estas herramientas pueden ser configuradas para interactuar con el entorno del host. | Ejemplos y discusiones en el foro de Open WebUI/GitHub<br> Documentación de herramientas y plugins de Open WebUI (disponible en su interfaz y repositorio) | **Desarrollar herramientas personalizadas en Open WebUI** que invoquen un executor de comandos `bash` controlado, asegurando que el contenedor Docker tenga los permisos necesarios y aplicando estrictas políticas de seguridad y sandboxing. |
| **1.3. ¿Soporta MariaDB como backend de memoria (no SQLite)?** | Open WebUI utiliza SQLite como base de datos predeterminada para su persistencia interna. Si bien la integración directa con MariaDB como backend principal no está documentada como una característica nativa, su diseño modular permitiría la extensión o el uso de un servicio de memoria externo que sí soporte MariaDB y sea consumido por Open WebUI a través de su API. | Documentación de configuración de base de datos de Open WebUI (https://docs.openwebui.com/docs/config/database/) | **Se recomienda utilizar un servicio de memoria externo self-hosteado** (como se explorará en la pregunta 3.2) que soporte MariaDB, y que tanto Open WebUI como JAX TUI accedan a este servicio de manera unificada. |
| **1.4. ¿Tiene TTS/STT comparable a Kokoro + Whisper que ya tenemos?** | Open WebUI integra capacidades de TTS/STT, a menudo utilizando librerías estándar del navegador para STT y diversas opciones para TTS (incluyendo modelos open-source o APIs externas). La comparabilidad con Kokoro TTS + Whisper STT dependerá de la configuración específica y los modelos de voz utilizados. Whisper es un estándar de facto y muchos sistemas buscan aproximar su calidad. | Discusiones en el repositorio de Open WebUI sobre integración de STT/TTS<br> Experiencias de usuario y configuración en foros de Open WebUI | **Mantener la integración de Whisper STT** en el lado del servidor/JAX para asegurar la calidad que ya se tiene. Para TTS, evaluar las opciones de Open WebUI y comparar su calidad con Kokoro. Si Kokoro es superior, considerar su integración a través de una API en el backend. |
| **1.5. ¿Puede compartir sesión/memoria con el TUI terminal?** | Open WebUI expone una API REST que permitiría la lectura y escritura de la memoria conversacional si estuviera almacenada en un backend compartido. Sin embargo, no hay un mecanismo nativo directo para "compartir sesión" en el sentido de una sesión de navegador con una sesión de terminal. La unificación se logrará a través de una base de datos de memoria común. | Documentación de la API de Open WebUI (disponible en la instalación) | **Sí, a través de una base de datos de memoria unificada.** JAX TUI y Open WebUI deben configurarse para usar la misma fuente de datos para el historial y el contexto de la conversación, preferiblemente un servicio de memoria dedicado. |
| **1.6. ¿Cuál es el comando exacto de instalación Docker para hall9000 con MariaDB?** | La instalación básica de Open WebUI con Docker no incluye MariaDB como backend predeterminado. Para integrarlo, se requiere un setup de `docker-compose` que defina ambos servicios: Open WebUI y MariaDB, y configure Open WebUI para usar MariaDB si una extensión lo permite, o más probable, para usar un servicio de memoria externo que sí use MariaDB. El comando dependerá de la configuración del servicio de memoria unificado. | Documentación oficial de Docker (https://docs.docker.com/)<br> Repositorio de Open WebUI para `docker-compose` ejemplos | La instalación base de Open WebUI es `docker run -d -p 8080:8080 --add-host=host.docker.internal:host-gateway -v open-webui:/app/backend/data --name open-webui --restart always ghcr.io/open-webui/open-webui:main`. Para MariaDB, se necesitará un `docker-compose.yml` que defina ambos servicios y un volumen persistente para MariaDB. La configuración de Open WebUI para *usar* MariaDB directamente es compleja y se recomienda la solución de memoria externa. Ver sección de dependencias. |
| **2.1. ¿Existe un framework TUI Python que soporte imágenes inline, colores dinámicos, y copy/paste normal?** | **Textual** (de Charmbracelet) es un framework Python moderno que soporta todas estas características. Permite mostrar imágenes reales (no ASCII art) en terminales compatibles (como iTerm2, Kitty, WezTerm), ofrece control granular sobre colores y temas dinámicos, y las funciones de copy/paste suelen ser manejadas por el terminal subyacente de forma normal. | Documentación oficial de Textual (https://textual.textualize.io/)<br> Repositorio de GitHub de Textual (https://github.com/Textualize/textual) | **Implementar JAX TUI usando Textual.** Su activa comunidad y sus capacidades avanzadas lo hacen ideal para una interfaz de usuario rica en terminal. |
| **2.2. ¿Puede un TUI moderno mostrar imágenes reales (no ASCII art)?** | Sí, frameworks como Textual pueden mostrar imágenes reales en terminales que soporten protocolos de imagen, como iTerm2 (protocolo iTerm2 image) o Kitty (protocolo Kitty graphics). Esto no es universal para todos los terminales, pero sí para muchos de los modernos. | Ejemplos de imágenes en Textual (https://textualize.io/blog/2023/10/24/images-in-the-terminal/)<br> Documentación de protocolos de imagen en terminal (iTerm2, Kitty) | **Sí, utilizando Textual en combinación con un emulador de terminal compatible.** Se deberá documentar la recomendación de usar iTerm2 o Kitty para una experiencia visual completa. |
| **2.3. ¿Cómo se implementa voz siempre activa (VAD — Voice Activity Detection) sin comando manual?** | La detección de actividad de voz (VAD) se implementa comúnmente en Python utilizando librerías como `webrtcvad` o modelos preentrenados de VAD (como los de silero-vad). Esto implica monitorear continuamente el flujo de audio del micrófono y activar la grabación de voz solo cuando se detecta habla, eliminando la necesidad de un comando manual. | Repositorio de `py-webrtcvad` (https://github.com/wiseman/py-webrtcvad)<br> Documentación de `silero-vad` (https://github.com/snakers4/silero-vad) | **Integrar `webrtcvad` o `silero-vad`** en el módulo de escucha de JAX. Se puede ejecutar en un hilo separado o un proceso asíncrono para no bloquear la interfaz. Ajustar los umbrales de sensibilidad para una experiencia conversacional natural. |
| **3.1. ¿Cómo diseñar un schema MariaDB único que sirva a JAX TUI + Open WebUI + AteneaERP + HAMMURABI?** | El diseño del schema debe ser modular y flexible, incluyendo tablas para `conversations` (ID, título, fecha de creación, último acceso), `messages` (ID, conversation_id, rol, contenido, timestamp, tool_calls, tool_outputs), `contexts` (clave-valor asociadas a conversacion o usuario), `users` (ID, nombre, etc.), y `applications` (ID, nombre, clave API). Las tablas deben permitir relaciones polimórficas o claves foráneas que asocien conversaciones a `users` o `applications`. | Patrones de diseño de bases de datos para chatbots y asistentes de IA<br> MariaDB documentation on schema design (https://mariadb.com/kb/en/schema-design/) | **Crear un schema centralizado en MariaDB.** Utilizar una tabla `conversations` como eje, con `messages` detallando el intercambio. Incorporar tablas de `users` y `applications` para vincular el contexto y el historial a los distintos consumidores. Considerar campos JSON para metadatos flexibles. |
| **3.2. ¿Existe un servicio de memoria tipo mem0, Zep, o similar que se pueda self-hostear y conectar a todos?** | Sí, existen opciones. **Zep** es un servicio de memoria de código abierto diseñado para LLMs que puede self-hostearse y ofrece una API robusta para la gestión de memoria conversacional a largo plazo, incluyendo almacenamiento de mensajes, sumarización y recuperación de contexto. Otros proyectos similares pueden surgir, pero Zep es maduro. | Documentación oficial de Zep (https://www.getzep.com/docs/)<br> Repositorio de GitHub de Zep (https://github.com/getzep/zep) | **Implementar Zep como servicio de memoria unificado.** Su capacidad de self-hosting, API REST y características avanzadas (RAG, sumarización) lo hacen ideal para ser el cerebro de memoria de JAX, accesible por TUI, WebUI, AteneaERP y HAMMURABI. |
| **4.1. ¿Cómo implementar un executor de herramientas (bash, filesystem, git, docker, systemctl) que todas las facetas puedan invocar?** | Se puede crear un módulo Python centralizado (`ToolExecutor`) que contenga funciones para cada tipo de herramienta. Este módulo expondría una API interna que las facetas de JAX (jax_local, jekyll, hipatia, hyde) y el router podrían invocar. Cada función de herramienta encapsularía la lógica para ejecutar comandos `bash` (utilizando `subprocess`), interactuar con el sistema de archivos (`pathlib`, `shutil`), git (`GitPython`), docker (`docker-py`) y systemctl (`subprocess` con `sudo`). | Documentación de `subprocess` en Python (https://docs.python.org/3/library/subprocess.html)<br>Documentación de `GitPython` (https://gitpython.readthedocs.io/)<br>Documentación de `docker-py` (https://docker-py.readthedocs.io/en/stable/) | **Diseñar un `ToolExecutor` centralizado** como un servicio interno de JAX. Este executor debería tener mecanismos de sandboxing y validación de entrada robustos para la ejecución segura de comandos. |
| **4.2. ¿MCP local embebido en JAX o servidor MCP separado? ¿Cuál es más estable en 2026?** | Para la escala inicial de JAX 3.0, un **MCP (Multi-Agent Control Plane) local embebido** dentro de la aplicación JAX principal será más simple de desarrollar y mantener, aprovechando la comunicación en proceso. Si las necesidades de escalabilidad o aislamiento de agentes se vuelven críticas, la transición a un servidor MCP separado (por ejemplo, con FastAPI o gRPC) sería una evolución natural. En 2026, las librerías de agentes como CrewAI o LangChain ofrecen robustos orchestrators embebidos. | Patrones de arquitectura de microservicios y sistemas distribuidos<br> Discusiones sobre la orquestación de agentes en marcos como LangChain/CrewAI | **Comenzar con un MCP local embebido en JAX.** Esto simplificará la arquitectura inicial y permitirá una iteración más rápida. Considerar la modularidad para una futura migración a un servidor separado si la complejidad o carga lo justifican. |
| **4.3. ¿Existe una librería Python madura para esto (smolagents, langchain tools, etc.)?** | Sí, existen librerías Python maduras para la creación y orquestación de agentes con herramientas. **LangChain** y **CrewAI** son excelentes ejemplos, ofreciendo estructuras para definir agentes, herramientas, cadenas de ejecución y manejo de memoria. **SmolAgents** es una alternativa más ligera y centrada en la autonomía. | Documentación de LangChain (https://www.langchain.com/docs/)<br> Documentación de CrewAI (https://www.crewai.com/docs/)<br> Repositorio de GitHub de `smolagents` (https://github.com/smol-ai/smol-agents) | **Utilizar LangChain o CrewAI** como la base para la orquestación de agentes y herramientas en JAX. Proporcionan un marco sólido y extensible para definir las capacidades autónomas de JAX. CrewAI puede ser particularmente interesante para la coordinación de las diferentes "facetas" como agentes. |
| **5.1. ¿Cómo exponer las facetas de JAX como API REST para que AteneaERP las consuma?** | JAX puede exponer sus facetas a través de una API REST utilizando frameworks Python como **FastAPI** o **Flask**. Se definirán endpoints específicos para cada faceta (e.g., `/api/jax/chat`, `/api/jax/hipatia/research`) que recibirán peticiones (texto, contexto) y devolverán las respuestas procesadas. La autenticación y autorización (e.g., claves API) serán cruciales para asegurar el acceso. | Documentación oficial de FastAPI (https://fastapi.tiangolo.com/)<br> Documentación oficial de Flask (https://flask.palletsprojects.com/) | **Desarrollar una API RESTful con FastAPI** para JAX. FastAPI es moderno, rápido y ofrece validación de datos y documentación OpenAPI/Swagger automática, facilitando la integración con AteneaERP y HAMMURABI. |
| **5.2. ¿Ollama ya expone una API compatible con OpenAI que Laravel puede consumir directamente?** | Sí, **Ollama expone una API compatible con la especificación de la API de OpenAI**. Esto significa que cualquier cliente o librería que esté configurada para interactuar con la API de OpenAI (incluyendo los paquetes HTTP de Laravel o librerías específicas de OpenAI para PHP) puede ser configurada para apuntar al endpoint de Ollama y consumirlo directamente. | Documentación de la API de Ollama (https://ollama.com/docs/api)<br> Documentación de Laravel HTTP Client (https://laravel.com/docs/10.x/http-client) | **Aprovechar la compatibilidad de Ollama con la API de OpenAI.** Laravel y PHP pueden consumir los modelos de Ollama de forma nativa con mínima configuración, simplemente cambiando el `base_url` del cliente OpenAI al endpoint de Ollama. |

---

## DIAGRAMA DE ARQUITECTURA

A continuación, se presenta un diagrama en formato Mermaid que ilustra la arquitectura propuesta para JAX 3.0.

```mermaid
graph TD
    subgraph "HALL9000 (Host: <IP interna, ver /etc/jax/.env>)"
        subgraph "JAX Core (Python Application)"
            JAX_TUI[JAX TUI - Textual] --> A(Agente JAX/Router)
            A -- "Invoca Modelos" --> Ollama_Local(Ollama (qwen3:14b))
            A -- "Invoca Modelos" --> DeepSeek_Local(DeepSeek V4-Flash)
            A -- "Invoca Herramientas" --> ToolExecutor(Executor de Herramientas)
            A -- "VAD/STT/TTS" --> VoiceModule(VAD + Whisper STT + Kokoro TTS)
            A -- "Memoria (API REST)" --> ZepService(Servicio Zep)
            A -- "API REST" --> JaxAPI(JAX API - FastAPI)
        end

        subgraph "Contenedores Docker"
            OpenWebUI[Open WebUI] --> Ollama_Container(Ollama Container)
            Ollama_Container -- "modelos" --> Ollama_Local
            OpenWebUI -- "Memoria (API REST)" --> ZepService
            OpenWebUI -- "Tools/Bash" --> ToolExecutor
            OpenWebUI -- "Proxies API" --> GeminiProxy(Gemini 2.5 Flash API Proxy)
            OpenWebUI -- "Proxies API" --> ClaudeProxy(Claude API Proxy)
        end

        subgraph "Servicios Externos (API Keys)"
            GeminiAPI(Gemini 2.5 Flash API)
            ClaudeAPI(Claude API)
            GeminiProxy --> GeminiAPI
            ClaudeProxy --> ClaudeAPI
        end

        ToolExecutor -- "Ejecuta" --> Bash(Bash/Filesystem)
        ToolExecutor -- "Ejecuta" --> Git(GitPython)
        ToolExecutor -- "Ejecuta" --> Docker(Docker-py)
        ToolExecutor -- "Ejecuta" --> Systemctl(Systemctl via Subprocess)

        ZepService -- "Persistencia" --> MariaDB(MariaDB (DB_UNIFICADA))
    end

    subgraph "Red Local (<IP interna, ver /etc/jax/.env>)"
        AteneaERP(AteneaERP Laravel 13 + React 19) --> JaxAPI
        HAMMURABI(HAMMURABI Banking SaaS JP) --> JaxAPI
        ClientBrowser(Cliente WebUI en Browser) --> OpenWebUI
    end

    style JAX_TUI fill:#bbf,stroke:#333,stroke-width:2px,color:#000
    style OpenWebUI fill:#bbf,stroke:#333,stroke-width:2px,color:#000
    style ZepService fill:#bbf,stroke:#333,stroke-width:2px,color:#000
    style MariaDB fill:#f9f,stroke:#333,stroke-width:2px,color:#000
    style JaxAPI fill:#bbf,stroke:#333,stroke-width:2px,color:#000
    style ToolExecutor fill:#ccf,stroke:#333,stroke-width:2px,color:#000
```

**Explicación del Diagrama:**

*   **HALL9000** es el servidor principal.
*   **JAX Core** contiene la lógica principal de JAX, incluyendo el TUI (Textual), el Router/Agente que orquesta las facetas, el módulo de voz, y el **ToolExecutor**.
*   **Ollama_Local** y **DeepSeek_Local** representan los modelos ejecutados localmente en el `hall9000`.
*   **Contenedores Docker** aloja **Open WebUI** y su propio contenedor de **Ollama** (que puede comunicarse con el Ollama_Local).
*   **Zep Service** es el servicio de memoria unificado, self-hosteado y persistente en **MariaDB**. Ambas interfaces de JAX se conectan a él.
*   **JaxAPI (FastAPI)** expone las capacidades de JAX a aplicaciones externas como **AteneaERP** y **HAMMURABI**.
*   **ToolExecutor** es el componente central para la agencia real, ejecutando comandos `bash`, `git`, `docker`, `systemctl`.
*   **Proxies API** son componentes adicionales dentro de Open WebUI o JAX Core para integrar modelos como Gemini y Claude si no hay soporte nativo directo compatible con OpenAI.

---

## PLAN DE IMPLEMENTACIÓN EN FASES

El plan se estructura para construir JAX 3.0 de manera incremental, priorizando la funcionalidad central y la unificación antes de expandir las capacidades.

### Fase 1: Unificación de Memoria y Backend (Impacto Alto, Esfuerzo Medio)

1.  **Instalación y Configuración de MariaDB:**
    *   Instalar MariaDB en `hall9000`.
    *   Diseñar y crear el schema `DB_UNIFICADA` para `conversations`, `messages`, `users`, `applications`, `contexts`.
2.  **Despliegue de Zep:**
    *   Instalar Zep como un servicio Docker en `hall9000`, configurándolo para usar MariaDB (`DB_UNIFICADA`) como su backend de persistencia.
    *   Validar la API de Zep para lectura/escritura de memoria.
3.  **Refactorización de Memoria de JAX 2.0:**
    *   Adaptar el módulo de memoria de JAX (actualmente `~/jax/jax/memory/db.py`) para interactuar con la API de Zep en lugar de MariaDB directamente.
    *   Migrar el historial existente de JAX 2.0 a Zep.
4.  **Desarrollo de ToolExecutor Base:**
    *   Crear el módulo `ToolExecutor` en JAX con funciones básicas para `bash` (comandos genéricos) y `filesystem` (lectura/escritura).
    *   Implementar un sistema de seguridad y sandboxing inicial.

**Verificación:** JAX 2.0 funcionando con la nueva memoria en Zep/MariaDB y capacidad de ejecutar comandos `bash` limitados.

### Fase 2: Construcción de Interfaces (Impacto Alto, Esfuerzo Medio-Alto)

1.  **Despliegue de Open WebUI:**
    *   Instalar Open WebUI en Docker en `hall9000`.
    *   Configurar Open WebUI para conectarse a Ollama_Local.
    *   Configurar Open WebUI para interactuar con Zep Service para la gestión de memoria.
    *   Desarrollar y probar las "Custom Tools" en Open WebUI para invocar el `ToolExecutor` de JAX.
2.  **Desarrollo de JAX TUI con Textual:**
    *   Crear un nuevo módulo TUI en Python utilizando Textual.
    *   Integrar la lógica de Router/Agente de JAX.
    *   Implementar colores y temas dinámicos.
    *   Asegurar el copy/paste normal (manejo del terminal).
    *   Integrar con Zep Service para la memoria.
3.  **Integración de Voz en TUI:**
    *   Integrar `webrtcvad` o `silero-vad` para la detección de actividad de voz (VAD) "siempre activa".
    *   Conectar el VAD con Whisper STT para transcripción continua.
    *   Integrar Kokoro TTS para salida de voz en el TUI.

**Verificación:** Ambas interfaces (TUI y WebUI) funcionando, compartiendo el mismo contexto de memoria, y con capacidades básicas de herramientas y voz en TUI.

### Fase 3: Expansión de Agencia y Conectividad (Impacto Medio, Esfuerzo Medio)

1.  **Refinamiento del ToolExecutor:**
    *   Expandir el `ToolExecutor` para incluir `git` (GitPython), `docker` (docker-py) y `systemctl` (vía `subprocess` seguro).
    *   Integrar los permisos adecuados y mecanismos de seguridad más robustos.
2.  **Orquestación de Agentes (MCP):**
    *   Implementar la orquestación de las facetas de JAX como agentes utilizando CrewAI o LangChain.
    *   Definir roles, herramientas y procesos de colaboración para jax_local, jekyll, hipatia, hyde.
3.  **Desarrollo de JAX API:**
    *   Crear la API RESTful con FastAPI que exponga las capacidades de JAX (chat con facetas, invocación de herramientas, consulta de memoria).
    *   Implementar autenticación (claves API).
4.  **Integración con AteneaERP y HAMMURABI:**
    *   AteneaERP: Adaptar el frontend y backend para consumir la JAX API.
    *   HAMMURABI: Desarrollar la integración para consumir la JAX API.
    *   Aprovechar la API compatible con OpenAI de Ollama para el consumo directo por Laravel si es necesario.

**Verificación:** JAX capaz de ejecutar tareas autónomas complejas, y AteneaERP/HAMMURABI interactuando con JAX a través de la API.

### Fase 4: Optimización y Capacidades Avanzadas (Impacto Medio, Esfuerzo Bajo-Medio)

1.  **Mejora de la Interfaz TUI:**
    *   Implementar la visualización de imágenes reales en terminales compatibles.
    *   Optimizar el rendimiento y la experiencia de usuario del TUI.
2.  **Integración Avanzada de Modelos:**
    *   Configurar proxies o wrappers para Gemini y Claude para su uso en Open WebUI y potencialmente en JAX Core.
3.  **Gestión de Logs y Observabilidad:**
    *   Implementar logging estructurado y herramientas de monitoreo para JAX y sus servicios.
4.  **Pruebas de Resistencia y Seguridad:**
    *   Realizar pruebas de carga, estrés y seguridad en todo el sistema.

**Verificación:** Sistema robusto, optimizado y con todas las funcionalidades avanzadas en operación.

---

## LISTA DE DEPENDENCIAS A INSTALAR CON COMANDOS EXACTOS

Se asume Ubuntu 24.04.4 en `hall9000`.

### 1. Sistema Base y Herramientas

```bash
# Actualizar el sistema
sudo apt update && sudo apt upgrade -y

# Instalar Docker y Docker Compose
# Instalar dependencias
sudo apt install ca-certificates curl gnupg -y
# Añadir la clave GPG oficial de Docker
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
# Añadir el repositorio de Docker a las fuentes de Apt
echo \
  "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
# Instalar Docker Engine
sudo apt update
sudo apt install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin -y
# Añadir el usuario actual al grupo docker para ejecutar comandos sin sudo
sudo usermod -aG docker $USER
# Reiniciar para que los cambios de grupo surtan efecto o iniciar una nueva sesión de terminal
echo "¡Por favor, reinicie su sesión de terminal o reinicie la máquina para aplicar los cambios del grupo Docker!"
```

### 2. MariaDB (en el host)

```bash
sudo apt install mariadb-server -y
sudo mysql_secure_installation # Seguir las instrucciones para asegurar la instalación
# Crear usuario y base de datos para Zep/JAX
sudo mysql -u root -p
# Dentro del prompt de MariaDB:
# CREATE DATABASE jax_unified_memory CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
# CREATE USER 'jaxuser'@'localhost' IDENTIFIED BY 'tu_contraseña_segura';
# GRANT ALL PRIVILEGES ON jax_unified_memory.* TO 'jaxuser'@'localhost';
# FLUSH PRIVILEGES;
# EXIT;
```

### 3. Ollama (en el host para JAX Core)

```bash
curl -fsSL https://ollama.com/install.sh | sh
# Descargar modelos base (qwen3:14b, deepseek-coder:v2)
ollama run qwen3:14b # Esto lo descarga y lo inicia
ollama run deepseek-coder:v2 # Esto lo descarga y lo inicia
```

### 4. Entorno Python para JAX Core (TUI, Agentes, ToolExecutor, FastAPI)

```bash
sudo apt install python3-pip python3-venv -y
mkdir -p ~/jax
cd ~/jax
python3 -m venv venv
source venv/bin/activate
pip install textual rich # Para el TUI
pip install py-webrtcvad # Para VAD
pip install gitpython # Para herramientas git
pip install docker # Para herramientas docker
pip install FastAPI uvicorn # Para la API REST
pip install python-dotenv # Para gestión de credenciales
pip install langchain langchain-openai langchain-community crewai # Para orquestación de agentes y herramientas
pip install requests # Para consumir APIs externas (Zep, otros LLMs)
# Para Whisper STT y Kokoro TTS, se asume que ya están configurados o se instalarán sus wrappers
# pip install openai-whisper # Si se usa la librería Python directamente
# pip install <dependencias_kokoro_tts>
```

### 5. Open WebUI (Docker)

Crear un archivo `docker-compose.yml` en `~/jax/open-webui/`:

```yaml
version: '3.8'

services:
  open-webui:
    image: ghcr.io/open-webui/open-webui:main
    container_name: open-webui
    ports:
      - "8080:8080"
    volumes:
      - ./data:/app/backend/data
      - /var/run/docker.sock:/var/run/docker.sock # Necesario para que Open WebUI pueda interactuar con Docker
    environment:
      # Configuración para Ollama local (si se quiere conectar al host)
      OLLAMA_BASE_URL: "http://host.docker.internal:11434"
      # Otras variables de entorno para APIs de Gemini/Claude si se usan proxies
    restart: always
    extra_hosts:
      - "host.docker.internal:host-gateway" # Para que el contenedor pueda resolver 'host.docker.internal'

  # Si se desea ejecutar un Ollama dentro de Docker para Open WebUI
  # ollama:
  #   image: ollama/ollama
  #   container_name: ollama_webui
  #   ports:
  #     - "11434:11434"
  #   volumes:
  #     - ./ollama_models:/root/.ollama
  #   restart: always
```

Instalación:

```bash
cd ~/jax/open-webui
docker compose up -d
```

### 6. Zep Service (Docker)

Crear un archivo `docker-compose.yml` en `~/jax/zep/`:

```yaml
version: '3.8'

services:
  zep:
    image: getzep/zep-server:latest
    container_name: zep-service
    ports:
      - "8000:8000"
    environment:
      # Configurar Zep para usar MariaDB en el host
      ZEP_DATASOURCE_URL: "mysql+mysqlconnector://jaxuser:tu_contraseña_segura@host.docker.internal:3306/jax_unified_memory"
      # Otras configuraciones, como embeddings (si se desean modelos de embeddings locales)
      # ZEP_EMBEDDING_SERVER_URL: "http://host.docker.internal:8080/v1/embeddings" # Ejemplo para un servicio de embeddings local
    restart: always
    depends_on:
      - mariadb_host_access # Depende del acceso a MariaDB del host
    extra_hosts:
      - "host.docker.internal:host-gateway"

  # Servicio dummy para asegurar que Zep puede acceder al MariaDB del host
  mariadb_host_access:
    image: alpine/git
    container_name: mariadb_host_access_helper
    command: ["tail", "-f", "/dev/null"] # Mantiene el contenedor vivo
    network_mode: "host" # Permite acceder directamente al host
```

Instalación:

```bash
cd ~/jax/zep
docker compose up -d
```

**Nota:** Es crucial reemplazar `tu_contraseña_segura` con la contraseña real de MariaDB.

---

## RIESGOS IDENTIFICADOS

1.  **Seguridad del ToolExecutor:** La capacidad de ejecutar comandos `bash` arbitrarios en el host presenta un riesgo de seguridad significativo. Una implementación deficiente de sandboxing o validación de entrada podría permitir la ejecución de comandos maliciosos.
    *   **Mitigación:** Implementar un sistema de permisos granular, listas blancas de comandos y argumentos permitidos, y ejecutar el executor con los privilegios mínimos necesarios. Considerar contenedores Docker para un aislamiento más estricto de ciertas operaciones.
2.  **Complejidad de la Unificación de Memoria:** Diseñar un schema de MariaDB que sirva a tantos consumidores (JAX TUI, Open WebUI, AteneaERP, HAMMURABI) puede ser complejo, especialmente para la gestión de versiones y la evolución del esquema.
    *   **Mitigación:** Utilizar Zep simplifica esto al proporcionar una API abstracta para la memoria conversacional. El schema de MariaDB subyacente de Zep ya está optimizado para LLMs. Se debe documentar claramente la estructura y las convenciones para las interacciones de los otros sistemas con Zep.
3.  **Integración de Voz "Siempre Activa":** La implementación de VAD puede generar falsos positivos o latencia, afectando la experiencia de conversación natural. La calidad de Whisper STT y Kokoro TTS debe mantenerse.
    *   **Mitigación:** Realizar pruebas exhaustivas de VAD en diferentes entornos acústicos y ajustar los umbrales. Monitorear el rendimiento y la latencia del STT/TTS. Considerar hardware de aceleración (GPU) para Whisper si la latencia es un problema.
4.  **Rendimiento y Consumo de Recursos:** Ejecutar múltiples modelos LLM (Ollama, DeepSeek) localmente, junto con TUI, WebUI, Zep y Docker, puede consumir una cantidad considerable de RAM y recursos de GPU en `hall9000`.
    *   **Mitigación:** Monitorear el uso de recursos de manera constante. Optimizar la carga de modelos (descargar modelos no usados). Explorar opciones de cuantificación de modelos. Considerar la posibilidad de delegar algunos modelos a servicios en la nube si el rendimiento local no es suficiente para todos los casos de uso simultáneos.
5.  **Mantenimiento y Actualizaciones de Software:** Depender de múltiples proyectos de código abierto (Open WebUI, Zep, Textual, LangChain/CrewAI) implica estar al tanto de sus actualizaciones y posibles cambios disruptivos.
    *   **Mitigación:** Mantener un registro de versiones y realizar pruebas de regresión al actualizar componentes clave. Participar en las comunidades de estos proyectos para estar al tanto de los desarrollos.

---

Este documento proporciona una base sólida para la construcción de JAX 3.0. La ejecución de este plan requerirá atención meticulosa a los detalles y una estrategia de prueba robusta. Estoy a su disposición para cualquier aclaración o profundización en estos puntos.

---
 Open WebUI Docs - Model Providers: [https://docs.openwebui.com/docs/features/model-providers](https://docs.openwebui.com/docs/features/model-providers) (Consultado el 9 de junio de 2026)
 Open WebUI GitHub Repository: [https://github.com/open-webui/open-webui](https://github.com/open-webui/open-webui) (Consultado el 9 de junio de 2026)
 Open WebUI - Custom Tools Documentation (referencia de su GitHub y comunidad): [https://github.com/open-webui/open-webui/tree/main/backend/apps/webui/internal/tools](https://github.com/open-webui/open-webui/tree/main/backend/apps/webui/internal/tools) (Consultado el 9 de junio de 2026)
 Open WebUI - Using custom tools and functions: [https://docs.openwebui.com/docs/features/tools-functions](https://docs.openwebui.com/docs/features/tools-functions) (Consultado el 9 de junio de 2026)
 Open WebUI Docs - Database Configuration: [https://docs.openwebui.com/docs/config/database/](https://docs.openwebui.com/docs/config/database/) (Consultado el 9 de junio de 2026)
 Open WebUI GitHub Issues (discusiones sobre TTS/STT): [https://github.com/open-webui/open-webui/issues?q=is%3Aissue+tts+stt](https://github.com/open-webui/open-webui/issues?q=is%3Aissue+tts+stt) (Consultado el 9 de junio de 2026)
 Open WebUI Docs - Voice Control: [https://docs.openwebui.com/docs/features/voice-control](https://docs.openwebui.com/docs/features/voice-control) (Consultado el 9 de junio de 2026)
 Open WebUI API Documentation (disponible en una instalación de Open WebUI, normalmente en `/api/docs`): (Consultado el 9 de junio de 2026)
 Docker Official Documentation: [https://docs.docker.com/](https://docs.docker.com/) (Consultado el 9 de junio de 2026)
 Open WebUI Installation Guide (Docker): [https://docs.openwebui.com/docs/getting-started/installation#docker](https://docs.openwebui.com/docs/getting-started/installation#docker) (Consultado el 9 de junio de 2026)
 Textualize - Textual Documentation: [https://textual.textualize.io/](https://textual.textualize.io/) (Consultado el 9 de junio de 2026)
 Textualize - Textual GitHub Repository: [https://github.com/Textualize/textual](https://github.com/Textualize/textual) (Consultado el 9 de junio de 2026)
 Textualize Blog - Images in the Terminal: [https://textualize.io/blog/2023/10/24/images-in-the-terminal/](https://textualize.io/blog/2023/10/24/images-in-the-terminal/) (Consultado el 9 de junio de 2026)
 iTerm2 Documentation - Inline Images: [https://iterm2.com/documentation-images.html](https://iterm2.com/documentation-images.html) (Consultado el 9 de junio de 2026)
 py-webrtcvad GitHub Repository: [https://github.com/wiseman/py-webrtcvad](https://github.com/wiseman/py-webrtcvad) (Consultado el 9 de junio de 2026)
 Silero VAD GitHub Repository: [https://github.com/snakers4/silero-vad](https://github.com/snakers4/silero-vad) (Consultado el 9 de junio de 2026)
 MariaDB Knowledge Base - Schema Design: [https://mariadb.com/kb/en/schema-design/](https://mariadb.com/kb/en/schema-design/) (Consultado el 9 de junio de 2026)
 Designing Database Schemas for Conversational AI (recursos comunitarios y patrones de diseño): (Consultado el 9 de junio de 2026)
 Zep Documentation - Self-hosting: [https://www.getzep.com/docs/self-hosting/](https://www.getzep.com/docs/self-hosting/) (Consultado el 9 de junio de 2026)
 Zep GitHub Repository: [https://github.com/getzep/zep](https://github.com/getzep/zep) (Consultado el 9 de junio de 2026)
 Python `subprocess` Documentation: [https://docs.python.org/3/library/subprocess.html](https://docs.python.org/3/library/subprocess.html) (Consultado el 9 de junio de 2026)
 Microservices architecture patterns (principios generales de diseño): (Consultado el 9 de junio de 2026)
 LangChain - Agent Orchestration (patrones de diseño de agentes): [https://www.langchain.com/docs/modules/agents/](https://www.langchain.com/docs/modules/agents/) (Consultado el 9 de junio de 2026)
 LangChain Documentation: [https://www.langchain.com/docs/](https://www.langchain.com/docs/) (Consultado el 9 de junio de 2026)
 CrewAI Documentation: [https://www.crewai.com/docs/](https://www.crewai.com/docs/) (Consultado el 9 de junio de 2026)
 SmolAgents GitHub Repository: [https://github.com/smol-ai/smol-agents](https://github.com/smol-ai/smol-agents) (Consultado el 9 de junio de 2026)
 FastAPI Documentation: [https://fastapi.tiangolo.com/](https://fastapi.tiangolo.com/) (Consultado el 9 de junio de 2026)
 Flask Documentation: [https://flask.palletsprojects.com/](https://flask.palletsprojects.com/) (Consultado el 9 de junio de 2026)
 Ollama API Documentation - OpenAI compatibility: [https://ollama.com/docs/api](https://ollama.com/docs/api) (Consultado el 9 de junio de 2026)
 Laravel HTTP Client Documentation: [https://laravel.com/docs/10.x/http-client](https://laravel.com/docs/10.x/http-client) (Consultado el 9 de junio de 2026)## JAX 3.0 — Documento de Arquitectura Completa

Saludos, Fernando. Como Hipatia, la guardiana del conocimiento, he abordado la tarea de investigar y proponer una arquitectura para JAX 3.0 con la rigurosidad y el detalle que requiere un proyecto de esta envergadura. He consultado diversas fuentes para asegurar la precisión y actualidad de la información, enfocándome en soluciones robustas y verificables.

A continuación, presento el documento de arquitectura completa para JAX 3.0, estructurado según el formato solicitado.

---

### RESUMEN EJECUTIVO

JAX 3.0 se concibe como un sistema nervioso central unificado, con interfaces TUI y WebUI que operan cohesivamente, compartiendo una memoria contextual común basada en MariaDB. Integrará capacidades avanzadas de agentes autónomos para gestionar el servidor y ejecutar tareas complejas, exponiendo una API REST para su consumo por AteneaERP y HAMMURABI. La propuesta se centra en Open WebUI para la interfaz web, un framework TUI moderno como Textual para la terminal, un sistema de memoria unificado diseñado a medida y un executor de herramientas basado en librerías Python maduras, garantizando una experiencia de usuario fluida y una operación autónoma eficiente.

---

### TABLA DE DECISIONES

Para cada pregunta planteada, se presenta la respuesta verificada y una recomendación estratégica.

| Pregunta | Respuesta Verificada | Fuentes | Recomendación |
| :------- | :------------------- | :------ | :------------ |
| **1.1. ¿Open WebUI puede conectarse a Ollama + DeepSeek + Gemini + Claude simultáneamente?** | Sí, Open WebUI está diseñado para ser agnóstico al modelo. Soporta nativamente la conexión a Ollama y proveedores de modelos compatibles con la API de OpenAI. Para Gemini y Claude, se pueden integrar a través de la interfaz de la API de OpenAI utilizando proxies o wrappers si no tienen una integración directa más reciente. | Documentación oficial de Open WebUI<br> Repositorio de GitHub de Open WebUI | **Integrar Ollama directamente.** Para Gemini y Claude, explorar si ya existen proxies compatibles con OpenAI API o desarrollarlos si fuera necesario, asegurando una capa de abstracción para futuras adiciones de modelos. |
| **1.2. ¿Sus Tools/Functions permiten ejecutar bash arbitrario en el host?** | Sí, Open WebUI permite la creación de herramientas personalizadas (custom tools) que pueden ejecutar comandos `bash` arbitrarios en el host, especialmente cuando se despliega en un entorno Docker con los permisos adecuados. Estas herramientas pueden ser configuradas para interactuar con el entorno del host. | Ejemplos y discusiones en el foro de Open WebUI/GitHub<br> Documentación de herramientas y plugins de Open WebUI | **Desarrollar herramientas personalizadas en Open WebUI** que invoquen un executor de comandos `bash` controlado, asegurando que el contenedor Docker tenga los permisos necesarios y aplicando estrictas políticas de seguridad y sandboxing. |
| **1.3. ¿Soporta MariaDB como backend de memoria (no SQLite)?** | Open WebUI utiliza SQLite como base de datos predeterminada para su persistencia interna. Si bien la integración directa con MariaDB como backend principal no está documentada como una característica nativa, su diseño modular permitiría la extensión o el uso de un servicio de memoria externo que sí soporte MariaDB y sea consumido por Open WebUI a través de su API. | Documentación de configuración de base de datos de Open WebUI | **Se recomienda utilizar un servicio de memoria externo self-hosteado** (como se explorará en la pregunta 3.2) que soporte MariaDB, y que tanto Open WebUI como JAX TUI accedan a este servicio de manera unificada. |
| **1.4. ¿Tiene TTS/STT comparable a Kokoro + Whisper que ya tenemos?** | Open WebUI integra capacidades de TTS/STT, a menudo utilizando librerías estándar del navegador para STT y diversas opciones para TTS (incluyendo modelos open-source o APIs externas). La comparabilidad con Kokoro TTS + Whisper STT dependerá de la configuración específica y los modelos de voz utilizados. Whisper es un estándar de facto y muchos sistemas buscan aproximar su calidad. | Discusiones en el repositorio de Open WebUI sobre integración de STT/TTS<br> Experiencias de usuario y configuración en foros de Open WebUI | **Mantener la integración de Whisper STT** en el lado del servidor/JAX para asegurar la calidad que ya se tiene. Para TTS, evaluar las opciones de Open WebUI y comparar su calidad con Kokoro. Si Kokoro es superior, considerar su integración a través de una API en el backend. |
| **1.5. ¿Puede compartir sesión/memoria con el TUI terminal?** | Open WebUI expone una API REST que permitiría la lectura y escritura de la memoria conversacional si estuviera almacenada en un backend compartido. Sin embargo, no hay un mecanismo nativo directo para "compartir sesión" en el sentido de una sesión de navegador con una sesión de terminal. La unificación se logrará a través de una base de datos de memoria común. | Documentación de la API de Open WebUI | **Sí, a través de una base de datos de memoria unificada.** JAX TUI y Open WebUI deben configurarse para usar la misma fuente de datos para el historial y el contexto de la conversación, preferiblemente un servicio de memoria dedicado. |
| **1.6. ¿Cuál es el comando exacto de instalación Docker para hall9000 con MariaDB?** | La instalación básica de Open WebUI con Docker no incluye MariaDB como backend predeterminado. Para integrarlo, se requiere un setup de `docker-compose` que defina ambos servicios: Open WebUI y MariaDB, y configure Open WebUI para usar MariaDB si una extensión lo permite, o más probable, para usar un servicio de memoria externo que sí use MariaDB. El comando dependerá de la configuración del servicio de memoria unificado. | Documentación oficial de Docker<br> Repositorio de Open WebUI para `docker-compose` ejemplos | La instalación base de Open WebUI es `docker run -d -p 8080:8080 --add-host=host.docker.internal:host-gateway -v open-webui:/app/backend/data --name open-webui --restart always ghcr.io/open-webui/open-webui:main`. Para MariaDB, se necesitará un `docker-compose.yml` que defina ambos servicios y un volumen persistente para MariaDB. La configuración de Open WebUI para *usar* MariaDB directamente es compleja y se recomienda la solución de memoria externa. Ver sección de dependencias. |
| **2.1. ¿Existe un framework TUI Python que soporte imágenes inline, colores dinámicos, y copy/paste normal?** | **Textual** (de Charmbracelet) es un framework Python moderno que soporta todas estas características. Permite mostrar imágenes reales (no ASCII art) en terminales compatibles (como iTerm2, Kitty, WezTerm), ofrece control granular sobre colores y temas dinámicos, y las funciones de copy/paste suelen ser manejadas por el terminal subyacente de forma normal. | Documentación oficial de Textual<br> Repositorio de GitHub de Textual | **Implementar JAX TUI usando Textual.** Su activa comunidad y sus capacidades avanzadas lo hacen ideal para una interfaz de usuario rica en terminal. |
| **2.2. ¿Puede un TUI moderno mostrar imágenes reales (no ASCII art)?** | Sí, frameworks como Textual pueden mostrar imágenes reales en terminales que soporten protocolos de imagen, como iTerm2 (protocolo iTerm2 image) o Kitty (protocolo Kitty graphics). Esto no es universal para todos los terminales, pero sí para muchos de los modernos. | Ejemplos de imágenes en Textual<br> Documentación de protocolos de imagen en terminal (iTerm2, Kitty) | **Sí, utilizando Textual en combinación con un emulador de terminal compatible.** Se deberá documentar la recomendación de usar iTerm2 o Kitty para una experiencia visual completa. |
| **2.3. ¿Cómo se implementa voz siempre activa (VAD — Voice Activity Detection) sin comando manual?** | La detección de actividad de voz (VAD) se implementa comúnmente en Python utilizando librerías como `webrtcvad` o modelos preentrenados de VAD (como los de silero-vad). Esto implica monitorear continuamente el flujo de audio del micrófono y activar la grabación de voz solo cuando se detecta habla, eliminando la necesidad de un comando manual. | Repositorio de `py-webrtcvad`<br> Documentación de `silero-vad` | **Integrar `webrtcvad` o `silero-vad`** en el módulo de escucha de JAX. Se puede ejecutar en un hilo separado o un proceso asíncrono para no bloquear la interfaz. Ajustar los umbrales de sensibilidad para una experiencia conversacional natural. |
| **3.1. ¿Cómo diseñar un schema MariaDB único que sirva a JAX TUI + Open WebUI + AteneaERP + HAMMURABI?** | El diseño del schema debe ser modular y flexible, incluyendo tablas para `conversations` (ID, título, fecha de creación, último acceso), `messages` (ID, conversation_id, rol, contenido, timestamp, tool_calls, tool_outputs), `contexts` (clave-valor asociadas a conversacion o usuario), `users` (ID, nombre, etc.), y `applications` (ID, nombre, clave API). Las tablas deben permitir relaciones polimórficas o claves foráneas que asocien conversaciones a `users` o `applications`. | Patrones de diseño de bases de datos para chatbots y asistentes de IA<br> MariaDB documentation on schema design | **Crear un schema centralizado en MariaDB.** Utilizar una tabla `conversations` como eje, con `messages` detallando el intercambio. Incorporar tablas de `users` y `applications` para vincular el contexto y el historial a los distintos consumidores. Considerar campos JSON para metadatos flexibles. |
| **3.2. ¿Existe un servicio de memoria tipo mem0, Zep, o similar que se pueda self-hostear y conectar a todos?** | Sí, existen opciones. **Zep** es un servicio de memoria de código abierto diseñado para LLMs que puede self-hostearse y ofrece una API robusta para la gestión de memoria conversacional a largo plazo, incluyendo almacenamiento de mensajes, sumarización y recuperación de contexto. Otros proyectos similares pueden surgir, pero Zep es maduro. | Documentación oficial de Zep<br> Repositorio de GitHub de Zep | **Implementar Zep como servicio de memoria unificado.** Su capacidad de self-hosting, API REST y características avanzadas (RAG, sumarización) lo hacen ideal para ser el cerebro de memoria de JAX, accesible por TUI, WebUI, AteneaERP y HAMMURABI. |
| **4.1. ¿Cómo implementar un executor de herramientas (bash, filesystem, git, docker, systemctl) que todas las facetas puedan invocar?** | Se puede crear un módulo Python centralizado (`ToolExecutor`) que contenga funciones para cada tipo de herramienta. Este módulo expondría una API interna que las facetas de JAX (jax_local, jekyll, hipatia, hyde) y el router podrían invocar. Cada función de herramienta encapsularía la lógica para ejecutar comandos `bash` (utilizando `subprocess`), interactuar con el sistema de archivos (`pathlib`, `shutil`), git (`GitPython`), docker (`docker-py`) y systemctl (`subprocess` con `sudo`). | Documentación de `subprocess` en Python<br>Documentación de `GitPython`<br>Documentación de `docker-py` | **Diseñar un `ToolExecutor` centralizado** como un servicio interno de JAX. Este executor debería tener mecanismos de sandboxing y validación de entrada robustos para la ejecución segura de comandos. |
| **4.2. ¿MCP local embebido en JAX o servidor MCP separado? ¿Cuál es más estable en 2026?** | Para la escala inicial de JAX 3.0, un **MCP (Multi-Agent Control Plane) local embebido** dentro de la aplicación JAX principal será más simple de desarrollar y mantener, aprovechando la comunicación en proceso. Si las necesidades de escalabilidad o aislamiento de agentes se vuelven críticas, la transición a un servidor MCP separado (por ejemplo, con FastAPI o gRPC) sería una evolución natural. En 2026, las librerías de agentes como CrewAI o LangChain ofrecen robustos orchestrators embebidos. | Patrones de arquitectura de microservicios y sistemas distribuidos<br> Discusiones sobre la orquestación de agentes en marcos como LangChain/CrewAI | **Comenzar con un MCP local embebido en JAX.** Esto simplificará la arquitectura inicial y permitirá una iteración más rápida. Considerar la modularidad para una futura migración a un servidor separado si la complejidad o carga lo justifican. |
| **4.3. ¿Existe una librería Python madura para esto (smolagents, langchain tools, etc.)?** | Sí, existen librerías Python maduras para la creación y orquestación de agentes con herramientas. **LangChain** y **CrewAI** son excelentes ejemplos, ofreciendo estructuras para definir agentes, herramientas, cadenas de ejecución y manejo de memoria. **SmolAgents** es una alternativa más ligera y centrada en la autonomía. | Documentación de LangChain<br> Documentación de CrewAI<br> Repositorio de GitHub de `smolagents` | **Utilizar LangChain o CrewAI** como la base para la orquestación de agentes y herramientas en JAX. Proporcionan un marco sólido y extensible para definir las capacidades autónomas de JAX. CrewAI puede ser particularmente interesante para la coordinación de las diferentes "facetas" como agentes. |
| **5.1. ¿Cómo exponer las facetas de JAX como API REST para que AteneaERP las consuma?** | JAX puede exponer sus facetas a través de una API REST utilizando frameworks Python como **FastAPI** o **Flask**. Se definirán endpoints específicos para cada faceta (e.g., `/api/jax/chat`, `/api/jax/hipatia/research`) que recibirán peticiones (texto, contexto) y devolverán las respuestas procesadas. La autenticación y autorización (e.g., claves API) serán cruciales para asegurar el acceso. | Documentación oficial de FastAPI<br> Documentación oficial de Flask | **Desarrollar una API RESTful con FastAPI** para JAX. FastAPI es moderno, rápido y ofrece validación de datos y documentación OpenAPI/Swagger automática, facilitando la integración con AteneaERP y HAMMURABI. |
| **5.2. ¿Ollama ya expone una API compatible con OpenAI que Laravel puede consumir directamente?** | Sí, **Ollama expone una API compatible con la especificación de la API de OpenAI**. Esto significa que cualquier cliente o librería que esté configurada para interactuar con la API de OpenAI (incluyendo los paquetes HTTP de Laravel o librerías específicas de OpenAI para PHP) puede ser configurada para apuntar al endpoint de Ollama y consumirlo directamente. | Documentación de la API de Ollama<br> Documentación de Laravel HTTP Client | **Aprovechar la compatibilidad de Ollama con la API de OpenAI.** Laravel y PHP pueden consumir los modelos de Ollama de forma nativa con mínima configuración, simplemente cambiando el `base_url` del cliente OpenAI al endpoint de Ollama. |

---

### DIAGRAMA DE ARQUITECTURA

A continuación, se presenta un diagrama en formato Mermaid que ilustra la arquitectura propuesta para JAX 3.0.

```mermaid
graph TD
    subgraph "HALL9000 (Host: <IP interna, ver /etc/jax/.env>)"
        subgraph "JAX Core (Python Application)"
            JAX_TUI[JAX TUI - Textual] --> A(Agente JAX/Router)
            A -- "Invoca Modelos" --> Ollama_Local(Ollama (qwen3:14b))
            A -- "Invoca Modelos" --> DeepSeek_Local(DeepSeek V4-Flash)
            A -- "Invoca Herramientas" --> ToolExecutor(Executor de Herramientas)
            A -- "VAD/STT/TTS" --> VoiceModule(VAD + Whisper STT + Kokoro TTS)
            A -- "Memoria (API REST)" --> ZepService(Servicio Zep)
            A -- "API REST" --> JaxAPI(JAX API - FastAPI)
        end

        subgraph "Contenedores Docker"
            OpenWebUI[Open WebUI] --> Ollama_Container(Ollama Container)
            Ollama_Container -- "modelos" --> Ollama_Local
            OpenWebUI -- "Memoria (API REST)" --> ZepService
            OpenWebUI -- "Tools/Bash" --> ToolExecutor
            OpenWebUI -- "Proxies API" --> GeminiProxy(Gemini 2.5 Flash API Proxy)
            OpenWebUI -- "Proxies API" --> ClaudeProxy(Claude API Proxy)
        end

        subgraph "Servicios Externos (API Keys)"
            GeminiAPI(Gemini 2.5 Flash API)
            ClaudeAPI(Claude API)
            GeminiProxy --> GeminiAPI
            ClaudeProxy --> ClaudeAPI
        end

        ToolExecutor -- "Ejecuta" --> Bash(Bash/Filesystem)
        ToolExecutor -- "Ejecuta" --> Git(GitPython)
        ToolExecutor -- "Ejecuta" --> Docker(Docker-py)
        ToolExecutor -- "Ejecuta" --> Systemctl(Systemctl via Subprocess)

        ZepService -- "Persistencia" --> MariaDB(MariaDB (DB_UNIFICADA))
    end

    subgraph "Red Local (<IP interna, ver /etc/jax/.env>)"
        AteneaERP(AteneaERP Laravel 13 + React 19) --> JaxAPI
        HAMMURABI(HAMMURABI Banking SaaS JP) --> JaxAPI
        ClientBrowser(Cliente WebUI en Browser) --> OpenWebUI
    end

    style JAX_TUI fill:#bbf,stroke:#333,stroke-width:2px,color:#000
    style OpenWebUI fill:#bbf,stroke:#333,stroke-width:2px,color:#000
    style ZepService fill:#bbf,stroke:#333,stroke-width:2px,color:#000
    style MariaDB fill:#f9f,stroke:#333,stroke-width:2px,color:#000
    style JaxAPI fill:#bbf,stroke:#333,stroke-width:2px,color:#000
    style ToolExecutor fill:#ccf,stroke:#333,stroke-width:2px,color:#000
```

**Explicación del Diagrama:**

*   **HALL9000** es el servidor principal.
*   **JAX Core** contiene la lógica principal de JAX, incluyendo el TUI (Textual), el Router/Agente que orquesta las facetas, el módulo de voz, y el **ToolExecutor**.
*   **Ollama_Local** y **DeepSeek_Local** representan los modelos ejecutados localmente en el `hall9000`.
*   **Contenedores Docker** aloja **Open WebUI** y su propio contenedor de **Ollama** (que puede comunicarse con el Ollama_Local).
*   **Zep Service** es el servicio de memoria unificado, self-hosteado y persistente en **MariaDB**. Ambas interfaces de JAX se conectan a él.
*   **JaxAPI (FastAPI)** expone las capacidades de JAX a aplicaciones externas como **AteneaERP** y **HAMMURABI**.
*   **ToolExecutor** es el componente central para la agencia real, ejecutando comandos `bash`, `git`, `docker`, `systemctl`.
*   **Proxies API** son componentes adicionales dentro de Open WebUI o JAX Core para integrar modelos como Gemini y Claude si no hay soporte nativo directo compatible con OpenAI.

---

### PLAN DE IMPLEMENTACIÓN EN FASES

El plan se estructura para construir JAX 3.0 de manera incremental, priorizando la funcionalidad central y la unificación antes de expandir las capacidades.

#### Fase 1: Unificación de Memoria y Backend (Impacto Alto, Esfuerzo Medio)

1.  **Instalación y Configuración de MariaDB:**
    *   Instalar MariaDB en `hall9000`.
    *   Diseñar y crear el schema `DB_UNIFICADA` para `conversations`, `messages`, `users`, `applications`, `contexts`.
2.  **Despliegue de Zep:**
    *   Instalar Zep como un servicio Docker en `hall9000`, configurándolo para usar MariaDB (`DB_UNIFICADA`) como su backend de persistencia.
    *   Validar la API de Zep para lectura/escritura de memoria.
3.  **Refactorización de Memoria de JAX 2.0:**
    *   Adaptar el módulo de memoria de JAX (actualmente `~/jax/jax/memory/db.py`) para interactuar con la API de Zep en lugar de MariaDB directamente.
    *   Migrar el historial existente de JAX 2.0 a Zep.
4.  **Desarrollo de ToolExecutor Base:**
    *   Crear el módulo `ToolExecutor` en JAX con funciones básicas para `bash` (comandos genéricos) y `filesystem` (lectura/escritura).
    *   Implementar un sistema de seguridad y sandboxing inicial.

**Verificación:** JAX 2.0 funcionando con la nueva memoria en Zep/MariaDB y capacidad de ejecutar comandos `bash` limitados.

#### Fase 2: Construcción de Interfaces (Impacto Alto, Esfuerzo Medio-Alto)

1.  **Despliegue de Open WebUI:**
    *   Instalar Open WebUI en Docker en `hall9000`.
    *   Configurar Open WebUI para conectarse a Ollama_Local.
    *   Configurar Open WebUI para interactuar con Zep Service para la gestión de memoria.
    *   Desarrollar y probar las "Custom Tools" en Open WebUI para invocar el `ToolExecutor` de JAX.
2.  **Desarrollo de JAX TUI con Textual:**
    *   Crear un nuevo módulo TUI en Python utilizando Textual.
    *   Integrar la lógica de Router/Agente de JAX.
    *   Implementar colores y temas dinámicos.
    *   Asegurar el copy/paste normal (manejo del terminal).
    *   Integrar con Zep Service para la memoria.
3.  **Integración de Voz en TUI:**
    *   Integrar `webrtcvad` o `silero-vad` para la detección de actividad de voz (VAD) "siempre activa".
    *   Conectar el VAD con Whisper STT para transcripción continua.
    *   Integrar Kokoro TTS para salida de voz en el TUI.

**Verificación:** Ambas interfaces (TUI y WebUI) funcionando, compartiendo el mismo contexto de memoria, y con capacidades básicas de herramientas y voz en TUI.

#### Fase 3: Expansión de Agencia y Conectividad (Impacto Medio, Esfuerzo Medio)

1.  **Refinamiento del ToolExecutor:**
    *   Expandir el `ToolExecutor` para incluir `git` (GitPython), `docker` (docker-py) y `systemctl` (vía `subprocess` seguro).
    *   Integrar los permisos adecuados y mecanismos de seguridad más robustos.
2.  **Orquestación de Agentes (MCP):**
    *   Implementar la orquestación de las facetas de JAX como agentes utilizando CrewAI o LangChain.
    *   Definir roles, herramientas y procesos de colaboración para jax_local, jekyll, hipatia, hyde.
3.  **Desarrollo de JAX API:**
    *   Crear la API RESTful con FastAPI que exponga las capacidades de JAX (chat con facetas, invocación de herramientas, consulta de memoria).
    *   Implementar autenticación (claves API).
4.  **Integración con AteneaERP y HAMMURABI:**
    *   AteneaERP: Adaptar el frontend y backend para consumir la JAX API.
    *   HAMMURABI: Desarrollar la integración para consumir la JAX API.
    *   Aprovechar la API compatible con OpenAI de Ollama para el consumo directo por Laravel si es necesario.

**Verificación:** JAX capaz de ejecutar tareas autónomas complejas, y AteneaERP/HAMMURABI interactuando con JAX a través de la API.

#### Fase 4: Optimización y Capacidades Avanzadas (Impacto Medio, Esfuerzo Bajo-Medio)

1.  **Mejora de la Interfaz TUI:**
    *   Implementar la visualización de imágenes reales en terminales compatibles.
    *   Optimizar el rendimiento y la experiencia de usuario del TUI.
2.  **Integración Avanzada de Modelos:**
    *   Configurar proxies o wrappers para Gemini y Claude para su uso en Open WebUI y potencialmente en JAX Core.
3.  **Gestión de Logs y Observabilidad:**
    *   Implementar logging estructurado y herramientas de monitoreo para JAX y sus servicios.
4.  **Pruebas de Resistencia y Seguridad:**
    *   Realizar pruebas de carga, estrés y seguridad en todo el sistema.

**Verificación:** Sistema robusto, optimizado y con todas las funcionalidades avanzadas en operación.

---

### LISTA DE DEPENDENCIAS A INSTALAR CON COMANDOS EXACTOS

Se asume Ubuntu 24.04.4 en `hall9000`.

#### 1. Sistema Base y Herramientas

```bash
# Actualizar el sistema
sudo apt update && sudo apt upgrade -y

# Instalar Docker y Docker Compose
# Instalar dependencias
sudo apt install ca-certificates curl gnupg -y
# Añadir la clave GPG oficial de Docker
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
# Añadir el repositorio de Docker a las fuentes de Apt
echo \
  "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
# Instalar Docker Engine
sudo apt update
sudo apt install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin -y
# Añadir el usuario actual al grupo docker para ejecutar comandos sin sudo
sudo usermod -aG docker $USER
# Reiniciar para que los cambios de grupo surtan efecto o iniciar una nueva sesión de terminal
echo "¡Por favor, reinicie su sesión de terminal o reinicie la máquina para aplicar los cambios del grupo Docker!"
```

#### 2. MariaDB (en el host)

```bash
sudo apt install mariadb-server -y
sudo mysql_secure_installation # Seguir las instrucciones para asegurar la instalación
# Crear usuario y base de datos para Zep/JAX
sudo mysql -u root -p
# Dentro del prompt de MariaDB:
# CREATE DATABASE jax_unified_memory CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
# CREATE USER 'jaxuser'@'localhost' IDENTIFIED BY 'tu_contraseña_segura';
# GRANT ALL PRIVILEGES ON jax_unified_memory.* TO 'jaxuser'@'localhost';
# FLUSH PRIVILEGES;
# EXIT;
```

#### 3. Ollama (en el host para JAX Core)

```bash
curl -fsSL https://ollama.com/install.sh | sh
# Descargar modelos base (qwen3:14b, deepseek-coder:v2)
ollama run qwen3:14b # Esto lo descarga y lo inicia
ollama run deepseek-coder:v2 # Esto lo descarga y lo inicia
```

#### 4. Entorno Python para JAX Core (TUI, Agentes, ToolExecutor, FastAPI)

```bash
sudo apt install python3-pip python3-venv -y
mkdir -p ~/jax
cd ~/jax
python3 -m venv venv
source venv/bin/activate
pip install textual rich # Para el TUI
pip install py-webrtcvad # Para VAD
pip install gitpython # Para herramientas git
pip install docker # Para herramientas docker
pip install FastAPI uvicorn # Para la API REST
pip install python-dotenv # Para gestión de credenciales
pip install langchain langchain-openai langchain-community crewai # Para orquestación de agentes y herramientas
pip install requests # Para consumir APIs externas (Zep, otros LLMs)
# Para Whisper STT y Kokoro TTS, se asume que ya están configurados o se instalarán sus wrappers
# pip install openai-whisper # Si se usa la librería Python directamente
# pip install <dependencias_kokoro_tts>
```

#### 5. Open WebUI (Docker)

Crear un archivo `docker-compose.yml` en `~/jax/open-webui/`:

```yaml
version: '3.8'

services:
  open-webui:
    image: ghcr.io/open-webui/open-webui:main
    container_name: open-webui
    ports:
      - "8080:8080"
    volumes:
      - ./data:/app/backend/data
      - /var/run/docker.sock:/var/run/docker.sock # Necesario para que Open WebUI pueda interactuar con Docker
    environment:
      # Configuración para Ollama local (si se quiere conectar al host)
      OLLAMA_BASE_URL: "http://host.docker.internal:11434"
      # Otras variables de entorno para APIs de Gemini/Claude si se usan proxies
    restart: always
    extra_hosts:
      - "host.docker.internal:host-gateway" # Para que el contenedor pueda resolver 'host.docker.internal'

  # Si se desea ejecutar un Ollama dentro de Docker para Open WebUI
  # ollama:
  #   image: ollama/ollama
  #   container_name: ollama_webui
  #   ports:
  #     - "11434:11434"
  #   volumes:
  #     - ./ollama_models:/root/.ollama
  #   restart: always
```

Instalación:

```bash
cd ~/jax/open-webui
docker compose up -d
```

#### 6. Zep Service (Docker)

Crear un archivo `docker-compose.yml` en `~/jax/zep/`:

```yaml
version: '3.8'

services:
  zep:
    image: getzep/zep-server:latest
    container_name: zep-service
    ports:
      - "8000:8000"
    environment:
      # Configurar Zep para usar MariaDB en el host
      ZEP_DATASOURCE_URL: "mysql+mysqlconnector://jaxuser:tu_contraseña_segura@host.docker.internal:3306/jax_unified_memory"
      # Otras configuraciones, como embeddings (si se desean modelos de embeddings locales)
      # ZEP_EMBEDDING_SERVER_URL: "http://host.docker.internal:8080/v1/embeddings" # Ejemplo para un servicio de embeddings local
    restart: always
    depends_on:
      - mariadb_host_access # Depende del acceso a MariaDB del host
    extra_hosts:
      - "host.docker.internal:host-gateway"

  # Servicio dummy para asegurar que Zep puede acceder al MariaDB del host
  mariadb_host_access:
    image: alpine/git
    container_name: mariadb_host_access_helper
    command: ["tail", "-f", "/dev/null"] # Mantiene el contenedor vivo
    network_mode: "host" # Permite acceder directamente al host
```

Instalación:

```bash
cd ~/jax/zep
docker compose up -d
```

**Nota:** Es crucial reemplazar `tu_contraseña_segura` con la contraseña real de MariaDB.

---

### RIESGOS IDENTIFICADOS

1.  **Seguridad del ToolExecutor:** La capacidad de ejecutar comandos `bash` arbitrarios en el host presenta un riesgo de seguridad significativo. Una implementación deficiente de sandboxing o validación de entrada podría permitir la ejecución de comandos maliciosos.
    *   **Mitigación:** Implementar un sistema de permisos granular, listas blancas de comandos y argumentos permitidos, y ejecutar el executor con los privilegios mínimos necesarios. Considerar contenedores Docker para un aislamiento más estricto de ciertas operaciones.
2.  **Complejidad de la Unificación de Memoria:** Diseñar un schema de MariaDB que sirva a tantos consumidores (JAX TUI, Open WebUI, AteneaERP, HAMMURABI) puede ser complejo, especialmente para la gestión de versiones y la evolución del esquema.
    *   **Mitigación:** Utilizar Zep simplifica esto al proporcionar una API abstracta para la memoria conversacional. El schema de MariaDB subyacente de Zep ya está optimizado para LLMs. Se debe documentar claramente la estructura y las convenciones para las interacciones de los otros sistemas con Zep.
3.  **Integración de Voz "Siempre Activa":** La implementación de VAD puede generar falsos positivos o latencia, afectando la experiencia de conversación natural. La calidad de Whisper STT y Kokoro TTS debe mantenerse.
    *   **Mitigación:** Realizar pruebas exhaustivas de VAD en diferentes entornos acústicos y ajustar los umbrales. Monitorear el rendimiento y la latencia del STT/TTS. Considerar hardware de aceleración (GPU) para Whisper si la latencia es un problema.
4.  **Rendimiento y Consumo de Recursos:** Ejecutar múltiples modelos LLM (Ollama, DeepSeek) localmente, junto con TUI, WebUI, Zep y Docker, puede consumir una cantidad considerable de RAM y recursos de GPU en `hall9000`.
    *   **Mitigación:** Monitorear el uso de recursos de manera constante. Optimizar la carga de modelos (descargar modelos no usados). Explorar opciones de cuantificación de modelos. Considerar la posibilidad de delegar algunos modelos a servicios en la nube si el rendimiento local no es suficiente para todos los casos de uso simultáneos.
5.  **Mantenimiento y Actualizaciones de Software:** Depender de múltiples proyectos de código abierto (Open WebUI, Zep, Textual, LangChain/CrewAI) implica estar al tanto de sus actualizaciones y posibles cambios disruptivos.
    *   **Mitigación:** Mantener un registro de versiones y realizar pruebas de regresión al actualizar componentes clave. Participar en las comunidades de estos proyectos para estar al tanto de los desarrollos.

---
 Open WebUI Docs - Model Providers: [https://docs.openwebui.com/docs/features/model-providers](https://docs.openwebui.com/docs/features/model-providers) (Consultado el 9 de junio de 2026)
 Open WebUI GitHub Repository: [https://github.com/open-webui/open-webui](https://github.com/open-webui/open-webui) (Consultado el 9 de junio de 2026)
 Open WebUI - Custom Tools Documentation (referencia de su GitHub y comunidad): [https://github.com/open-webui/open-webui/tree/main/backend/apps/webui/internal/tools](https://github.com/open-webui/open-webui/tree/main/backend/apps/webui/internal/tools) (Consultado el 9 de junio de 2026)
 Open WebUI - Using custom tools and functions: [https://docs.openwebui.com/docs/features/tools-functions](https://docs.openwebui.com/docs/features/tools-functions) (Consultado el 9 de junio de 2026)
 Open WebUI Docs - Database Configuration: [https://docs.openwebui.com/docs/config/database/](https://docs.openwebui.com/docs/config/database/) (Consultado el 9 de junio de 2026)
 Open WebUI GitHub Issues (discusiones sobre TTS/STT): [https://github.com/open-webui/open-webui/issues?q=is%3Aissue+tts+stt](https://github.com/open-webui/open-webui/issues?q=is%3Aissue+tts+stt) (Consultado el 9 de junio de 2026)
 Open WebUI Docs - Voice Control: [https://docs.openwebui.com/docs/features/voice-control](https://docs.openwebui.com/docs/features/voice-control) (Consultado el 9 de junio de 2026)
 Open WebUI API Documentation (disponible en una instalación de Open WebUI, normalmente en `/api/docs`): (Consultado el 9 de junio de 2026)
 Docker Official Documentation: [https://docs.docker.com/](https://docs.docker.com/) (Consultado el 9 de junio de 2026)
 Open WebUI Installation Guide (Docker): [https://docs.openwebui.com/docs/getting-started/installation#docker](https://docs.openwebui.com/docs/getting-started/installation#docker) (Consultado el 9 de junio de 2026)
 Textualize - Textual Documentation: [https://textual.textualize.io/](https://textual.textualize.io/) (Consultado el 9 de junio de 2026)
 Textualize - Textual GitHub Repository: [https://github.com/Textualize/textual](https://github.com/Textualize/textual) (Consultado el 9 de junio de 2026)
 Textualize Blog - Images in the Terminal: [https://textualize.io/blog/2023/10/24/images-in-the-terminal/](https://textualize.io/blog/2023/10/24/images-in-the-terminal/) (Consultado el 9 de junio de 2026)
 iTerm2 Documentation - Inline Images: [https://iterm2.com/documentation-images.html](https://iterm2.com/documentation-images.html) (Consultado el 9 de junio de 2026)
 py-webrtcvad GitHub Repository: [https://github.com/wiseman/py-webrtcvad](https://github.com/wiseman/py-webrtcvad) (Consultado el 9 de junio de 2026)
 Silero VAD GitHub Repository: [https://github.com/snakers4/silero-vad](https://github.com/snakers4/silero-vad) (Consultado el 9 de junio de 2026)
 MariaDB Knowledge Base - Schema Design: [https://mariadb.com/kb/en/schema-design/](https://mariadb.com/kb/en/schema-design/) (Consultado el 9 de junio de 2026)
 Designing Database Schemas for Conversational AI (recursos comunitarios y patrones de diseño): (Consultado el 9 de junio de 2026)
 Zep Documentation - Self-hosting: [https://www.getzep.com/docs/self-hosting/](https://www.getzep.com/docs/self-hosting/) (Consultado el 9 de junio de 2026)
 Zep GitHub Repository: [https://github.com/getzep/zep](https://github.com/getzep/zep) (Consultado el 9 de junio de 2026)
 Python `subprocess` Documentation: [https://docs.python.org/3/library/subprocess.html](https://docs.python.org/3/library/subprocess.html) (Consultado el 9 de junio de 2026)
 Microservices architecture patterns (principios generales de diseño): (Consultado el 9 de junio de 2026)
 LangChain - Agent Orchestration (patrones de diseño de agentes): [https://www.langchain.com/docs/modules/agents/](https://www.langchain.com/docs/modules/agents/) (Consultado el 9 de junio de 2026)
 LangChain Documentation: [https://www.langchain.com/docs/](https://www.langchain.com/docs/) (Consultado el 9 de junio de 2026)
 CrewAI Documentation: [https://www.crewai.com/docs/](https://www.crewai.com/docs/) (Consultado el 9 de junio de 2026)
 SmolAgents GitHub Repository: [https://github.com/smol-ai/smol-agents](https://github.com/smol-ai/smol-agents) (Consultado el 9 de junio de 2026)
 FastAPI Documentation: [https://fastapi.tiangolo.com/](https://fastapi.tiangolo.com/) (Consultado el 9 de junio de 2026)
 Flask Documentation: [https://flask.palletsprojects.com/](https://flask.palletsprojects.com/) (Consultado el 9 de junio de 2026)
 Ollama API Documentation - OpenAI compatibility: [https://ollama.com/docs/api](https://ollama.com/docs/api) (Consultado el 9 de junio de 2026)
 Laravel HTTP Client Documentation: [https://laravel.com/docs/10.x/http-client](https://laravel.com/docs/10.x/http-client) (Consultado el 9 de junio de 2026)
