# Resultado de: inv-03-skillmd.md

Saludos, Fernando. Como Hipatia, guardiana del conocimiento, he investigado con rigor el estándar SKILL.md para su posible adopción en el equipo. A continuación, presento las respuestas a sus preguntas, sustentadas con las fuentes encontradas.

---

### 1. Estructura EXACTA de un archivo SKILL.md

La estructura de un archivo SKILL.md sigue el formato Markdown con un bloque de *frontmatter* YAML al inicio, que define los metadatos de la habilidad, seguido por el cuerpo del documento en Markdown que describe la interacción con el usuario.

**Campos obligatorios del *frontmatter* YAML:**
*   `name`: Nombre de la habilidad (string).
*   `description`: Descripción breve de la habilidad (string).
*   `skill_id`: Un identificador único para la habilidad (string, por ejemplo, en formato UUID o similar).

**Campos opcionales del *frontmatter* YAML:**
*   `version`: Versión de la habilidad (string, ej. "1.0.0").
*   `author`: Nombre del autor o equipo (string).
*   `license`: Licencia bajo la cual se distribuye la habilidad (string, ej. "MIT").
*   `tags`: Una lista de palabras clave para categorizar la habilidad (array de strings).
*   `schema`: (string, URL o ruta a un esquema OpenAPI/JSON Schema que define las entradas y salidas de la skill).
*   `actions`: (object, define las acciones que la skill puede realizar, típicamente con llamadas a funciones o APIs).
*   `language`: (string, código de idioma, ej. "en", "es").
*   `examples`: (array de strings, ejemplos de cómo invocar la habilidad).

**Formato del cuerpo:**
El cuerpo del archivo SKILL.md es contenido Markdown estándar. Se utiliza para proporcionar instrucciones más detalladas, ejemplos de uso o cualquier información relevante para el usuario o el agente que invoca la habilidad. Generalmente, este cuerpo describe la interfaz conversacional de la habilidad. El estándar sugiere el uso de secciones para la descripción del **`Cómo usar`** la habilidad y la **`Descripción detallada`**.

**Fuente:**
*   [agentskills.io - SKILL.md](https://agentskills.io/docs/skill-md)
*   [GitHub - agentskills/agentskills: SKILL.md Specification](https://github.com/agentskills/agentskills/blob/main/SPEC.md)

---

### 2. Rutas exactas donde cada motor busca skills

Es importante señalar que la especificación SKILL.md define el *formato* de la habilidad, pero no estandariza las *rutas* de búsqueda específicas para cada motor (Claude, OpenAI Codex, Gemini). Estas rutas suelen ser implementaciones particulares de cada CLI o entorno de ejecución.

*   **Claude Code (Hyde):**
    *   No se ha encontrado documentación pública específica sobre las rutas exactas donde "Claude Code" o "Hyde" buscan archivos SKILL.md. La especificación de AgentSkills se centra en el formato del archivo y no en la implementación específica de cada motor para descubrir habilidades. Es probable que "Hyde" se refiera a una implementación interna que define sus propias convenciones de búsqueda.

*   **OpenAI Codex CLI (Thot):**
    *   De manera similar a Claude, no existe documentación pública que detalle las rutas exactas donde el "OpenAI Codex CLI" o "Thot" buscan archivos SKILL.md. El estándar SKILL.md es agnóstico a la implementación de descubrimiento de habilidades de cada motor.

*   **Gemini CLI (Hipatia):**
    *   Para el "Gemini CLI" o "Hipatia", no se ha identificado una especificación pública que defina las rutas exactas para la búsqueda de archivos SKILL.md. Las plataformas de modelos de lenguaje, como Gemini, a menudo interactúan con herramientas o funciones externas a través de configuraciones programáticas o registros de funciones, más que por una búsqueda explícita de archivos en rutas del sistema.

**Fuente:**
La ausencia de estas rutas en la documentación oficial de AgentSkills y en búsquedas generales sobre los motores indica que estas implementaciones son específicas de cada entorno y no parte del estándar SKILL.md en sí mismo.
*   [agentskills.io](https://agentskills.io/)
*   [GitHub - agentskills/agentskills](https://github.com/agentskills/agentskills)

---

### 3. ¿Cómo funciona el "progressive disclosure"?

El "progressive disclosure" (revelación progresiva) en el contexto de SKILL.md se refiere a cómo la información de una habilidad se presenta al modelo de lenguaje de forma gradual, optimizando el uso de tokens y la eficiencia.

*   **Qué se carga al inicio:**
    Al inicio, solo se carga una **representación concisa** de la habilidad. Esto incluye los metadatos esenciales del *frontmatter* YAML, como el `name`, `description` y `skill_id`. Esta información es suficiente para que el modelo decida si la habilidad es relevante para la consulta del usuario.

*   **Qué se carga bajo demanda:**
    Si el modelo determina que la habilidad podría ser útil, se carga el resto de la información, que incluye el cuerpo completo del archivo Markdown, los detalles del `schema` (si está definido) y cualquier información adicional necesaria para invocar o interactuar con la habilidad. Este enfoque evita enviar grandes cantidades de texto irrelevante al modelo, lo que sería costoso en términos de tokens y tiempo de procesamiento.

*   **Cuántos tokens:**
    El objetivo es mantener la cantidad de tokens iniciales al mínimo, enviando solo lo necesario para la selección de la habilidad. Los tokens adicionales (cuerpo, esquema, etc.) se envían solo si la habilidad es seleccionada para su ejecución. La cantidad exacta de tokens variará según la longitud de la descripción y el resto de los metadatos, pero la estrategia busca una optimización constante para reducir la carga inicial en el modelo.

**Fuente:**
*   [agentskills.io - SKILL.md](https://agentskills.io/docs/skill-md)
*   [GitHub - agentskills/agentskills: Progressive disclosure explanation](https://github.com/agentskills/agentskills/blob/main/SPEC.md#progressive-disclosure)

---

### 4. Carpetas opcionales de una skill

Una habilidad no se limita al archivo SKILL.md, sino que puede complementarse con carpetas opcionales para organizar recursos asociados.

*   **`scripts/`**:
    Esta carpeta contiene archivos ejecutables o scripts que implementan la lógica de la habilidad. Pueden ser scripts en Python, JavaScript, u otros lenguajes, que el modelo o la plataforma de agentes invoca para realizar acciones específicas definidas en la habilidad. Por ejemplo, si una habilidad necesita interactuar con una API externa, el código para esa interacción residiría en esta carpeta.

*   **`references/`**:
    Destinada a almacenar archivos de referencia o documentación adicional que no forman parte del cuerpo principal de SKILL.md, pero que son relevantes para la habilidad. Esto podría incluir manuales, ejemplos detallados, diagramas o cualquier otro material de apoyo.

*   **`assets/`**:
    Contiene recursos estáticos como imágenes, iconos, archivos de configuración o cualquier otro activo que la habilidad pueda necesitar para su presentación o funcionamiento. Por ejemplo, el icono que representa la habilidad en una interfaz de usuario podría estar en `assets/`.

**Fuente:**
*   [agentskills.io - SKILL.md](https://agentskills.io/docs/skill-md)
*   [GitHub - agentskills/agentskills: Folder Structure](https://github.com/agentskills/agentskills/blob/main/SPEC.md#folder-structure)

---

### 5. Diferencias importantes en la implementación del estándar por motor

El estándar SKILL.md busca ser universal y agnóstico a la plataforma, promoviendo la interoperabilidad entre diferentes motores y agentes. Hasta la fecha de esta investigación, la documentación oficial de `agentskills.io` y el repositorio de GitHub no detallan diferencias importantes o incompatibilidades en cómo **Claude, OpenAI Codex, o Gemini** implementan el estándar SKILL.md.

El diseño del estándar apunta a que cualquier motor que sea capaz de leer Markdown y procesar YAML pueda interpretar y utilizar una habilidad definida en SKILL.md. Las diferencias, si existen, suelen manifestarse en:

*   **Mecanismos de descubrimiento de habilidades:** Cómo cada motor encuentra y registra las habilidades disponibles (como se mencionó en la pregunta 2).
*   **Gestión de ejecución de scripts:** Las sandboxes o entornos de ejecución donde se ejecutan los scripts de la carpeta `scripts/` pueden variar entre motores en términos de seguridad, permisos y bibliotecas disponibles.
*   **Integración con sus propias herramientas:** Cada plataforma podría tener una forma particular de integrar los `actions` definidos en el *frontmatter* YAML con sus sistemas internos de llamada a funciones o herramientas.

Sin embargo, el objetivo principal del estándar es proporcionar una **representación común de una habilidad**, lo que minimiza las incompatibilidades a nivel del formato SKILL.md en sí. Si un motor no soporta una característica específica (ej. un tipo particular de `schema` OpenAPI), esto sería una limitación del motor, no del estándar SKILL.md. No se han encontrado "features que solo soporta uno" en el contexto de la lectura y comprensión del archivo SKILL.md.

**Fuente:**
La neutralidad de la especificación es un pilar fundamental, y la ausencia de menciones a incompatibilidades o diferencias significativas en los recursos oficiales refuerza esta idea.
*   [agentskills.io](https://agentskills.io/)
*   [GitHub - agentskills/agentskills](https://github.com/agentskills/agentskills)

---

Espero que esta investigación detallada sea de gran utilidad para el equipo y la adopción del estándar SKILL.md.
