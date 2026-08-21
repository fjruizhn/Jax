# JAX

JAX es un ecosistema de orquestación multi-modelo: en vez de hablar con un
solo LLM, coordina varias **facetas** (personalidades respaldadas por
modelos distintos, cada una con un dominio y un proveedor propio) y un
**Director de pipelines** (Jacobs) que descompone un objetivo en pasos,
verifica que cada paso sea ejecutable antes de correrlo, y despacha el
trabajo con reglas de autoridad explícitas sobre quién puede hacer qué.

En memoria de Jairo Urbina, pionero del software libre en Honduras.
En honor al Prof. Raúl Jacobs.

## Qué hace

- **Facetas** — cada una es una personalidad + un modelo/proveedor
  concreto (local vía Ollama, o remoto vía API). El router decide qué
  faceta responde según el contenido del mensaje.
- **Motor Registry (LAS MANOS)** — despacha trabajo a los motores que
  tienen `tool-calling` real (leer/escribir archivos, ejecutar pasos de un
  plan), con un catálogo de *capabilities* en base de datos que define,
  por motor: qué puede hacer, con qué autoridad, y con qué cotas
  (timeouts, profundidad de recursión, rutas prohibidas).
- **Gate de autoridad para tool-calling** — antes de que un motor ejecute
  una tool, se resuelve contra el catálogo si esa combinación
  motor/capability está permitida; si no lo está, se rechaza con el
  motivo explícito, nunca en silencio.
- **Jacobs (pipelines)** — arma un plan (lista de steps con dependencias)
  desde un objetivo en texto, lo valida completo ANTES de persistirlo (un
  plan que no se puede ejecutar no se guarda ni se corre a medias), y lo
  ejecuta por olas topológicas, con puntos de aprobación humana explícitos
  para los pasos de mayor autoridad.
- **Memoria semántica** — hechos, decisiones y conversaciones se destilan
  y se guardan con embeddings, para que las facetas puedan recuperar
  contexto relevante en vez de depender solo de la ventana de la
  conversación actual.

## Arquitectura, a alto nivel

```
jax/               este repo
  jax/             REPL y núcleo de facetas (router, músculos por transporte)
  jacobs/          Director de pipelines (planificador + validador + executor)
  las_manos/       Motor Registry -- gate de autoridad, catálogo de capabilities
  policy/          reglas de método del proyecto + el scanner que las hace cumplir en CI

jax-platform/      repo hermano -- backend/frontend web (Axioma)
  backend/         API que expone pipelines, chat y administración
  frontend/        interfaz web (Mesa de chat, panel de administración)
```

Los dos repos comparten la misma base de datos y las mismas credenciales
de proveedor -- `jax` es el motor, `jax-platform` es la cara web de ese
motor.

## Instalación

Requisitos verificados contra el entorno real de desarrollo:

- Python 3.12+ (probado en producción con 3.12 y 3.14)
- MariaDB 11+ con soporte de tipo `VECTOR` (para embeddings)
- [Ollama](https://ollama.com) si vas a correr un modelo local
- Una o más API keys de proveedor (OpenAI, Gemini, DeepSeek, Moonshot/Kimi,
  Z.ai/GLM), según qué facetas quieras activar -- ninguna es obligatoria
  para levantar el sistema, pero sin al menos una el sistema no tiene con
  qué responder

```bash
git clone <url-del-repo>
cd jax
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# crear tu propio archivo de entorno (credenciales de DB + de proveedor) --
# no hay un .env.example todavía; ver las_manos/config.toml y
# las_manos/motor_registry/catalog.py para la lista completa de variables
# que el sistema espera
# aplicar el esquema de MariaDB (ver las_manos/db/ para las migraciones)
```

Levantar el servicio real (Motor Registry) es:

```bash
cd las_manos
uvicorn server:app --host 127.0.0.1 --port <puerto>
```

## Estado honesto

Este es un proyecto de una sola persona, con deuda técnica documentada y
activa -- no un producto terminado.

**Funciona en producción hoy:**
- El Motor Registry con gate de autoridad para tool-calling (lectura y
  escritura de archivos gobernadas, con cotas y rutas prohibidas).
- Jacobs: planificación con validación pre-persistencia, ejecución por
  olas, puntos de aprobación humana.
- Memoria semántica con destilación periódica de conversaciones.
- Un scanner de política en CI que impide fusionar código que "falla
  abierto" silenciosamente (ver `policy/`).

**Es experimental o incompleto:**
- Algunas facetas despachan por un camino HTTP directo que todavía no
  pasa por la misma gobernanza que el Motor Registry -- es una limitación
  conocida, no un descuido, y está documentada como tal.
- El confinamiento de la faceta que ejecuta comandos de shell reales usa
  un sandbox de sistema operativo (namespaces de montaje), no el gate de
  tool-calling del Motor Registry -- son dos mecanismos de gobernanza
  distintos, cada uno cubriendo una parte distinta del sistema.
- Hay columnas de base de datos sin lector, deuda de esquema conocida, y
  varias piezas marcadas explícitamente como "deuda declarada, no
  resuelta" en el historial de decisiones del proyecto.

Si algo de esto te importa para tu caso de uso, preguntá antes de asumir
que está resuelto.

## Licencia

AGPL-3.0. Ver [LICENSE](LICENSE). Si corrés una versión modificada de
este código como servicio de red, la licencia te exige ofrecer el código
fuente de esa versión a quienes lo usan -- es una condición central del
proyecto, no un detalle legal secundario.

## Contribuir

Ver [CONTRIBUTING.md](CONTRIBUTING.md).
