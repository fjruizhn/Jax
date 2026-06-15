# Resultado de: thot-fix-grounding.md

La opción correcta no es A, B ni C tal como están formuladas.

La solución robusta es una **cuarta opción**: convertir el `grounding` de Hipatia en una **política explícita de evidencia por tarea**, no en un booleano fijo de la faceta.

En términos prácticos:

```text
grounding_policy:
  - OFF                # no buscar; tarea local o creativa
  - AUTO               # puede buscar, pero la respuesta debe declararse no verificada si no buscó
  - REQUIRED_WEB       # debe buscar; si no hay groundingMetadata/citas, falla cerrado
  - LOCAL_CONTEXT_ONLY # debe responder solo con el texto provisto; no web
```

Para preguntas como:

> “¿Qué noticias de tecnología ocurrieron esta semana?”

la política correcta es `REQUIRED_WEB`. Si no hay fuentes, no hay respuesta. Punto.

Para:

> “Resumime este texto que te paso”

la política correcta es `LOCAL_CONTEXT_ONLY`. No debe buscar. Pero tampoco debe presentar el resumen como “verificado en web”; debe decir que se basa únicamente en el texto entregado.

---

## 1. Supuestos ocultos

### Supuesto 1: “Activar `google_search` equivale a verificar”

Falso. Como ya comprobaron, `google_search` habilita una capacidad, no impone su uso.

La presencia de:

```python
payload["tools"] = [{"google_search": {}}]
```

solo significa:

> “El modelo puede buscar si decide hacerlo.”

No significa:

> “El modelo buscó.”

Mucho menos significa:

> “Cada afirmación está respaldada por una fuente.”

---

### Supuesto 2: “Si hay `groundingMetadata`, entonces la respuesta está verificada”

También falso.

`groundingMetadata` prueba que hubo grounding, pero no necesariamente que:

1. todas las afirmaciones importantes estén respaldadas;
2. las fuentes sean relevantes;
3. el modelo no haya mezclado fuentes reales con inferencias inventadas;
4. la respuesta no haya agregado detalles no presentes en las fuentes.

Esto es crítico: **ausencia de grounding es fallo seguro**, pero **presencia de grounding no es garantía completa**.

Como mínimo, deben distinguir entre:

```text
Respuesta con búsqueda
```

y:

```text
Respuesta cuyas afirmaciones están citadas y soportadas
```

No son lo mismo.

---

### Supuesto 3: “Hipatia es investigadora, entonces siempre debería buscar”

No necesariamente.

Hay tareas investigativas que requieren web:

- noticias recientes;
- datos actuales;
- estado de un repositorio;
- precios;
- versiones;
- eventos;
- declaraciones públicas;
- documentación cambiante;
- comparación de productos actuales.

Pero hay tareas donde buscar es innecesario o incluso dañino:

- resumir un texto proporcionado;
- extraer entidades de un documento;
- traducir;
- ordenar ideas;
- analizar una conversación interna;
- revisar código pegado por el usuario;
- generar preguntas sobre un texto local.

El error conceptual es que están usando `grounding=True` como propiedad de la faceta. Debería ser una propiedad de la **tarea**.

---

### Supuesto 4: “El modelo puede declarar honestamente si buscó o no”

Puede, pero no deben confiar en eso como mecanismo de seguridad.

La declaración debe salir del sistema, no del modelo.

Si el modelo dice:

> “Verifiqué esto en fuentes web”

pero `groundingMetadata` está vacío, el sistema debe corregirlo o bloquearlo.

La verdad operacional es:

```text
Si no hay metadata/citas observables, el sistema no puede afirmar que hubo verificación web.
```

---

## 2. Riesgos no considerados por opción

---

## Opción A — Forzar grounding con `tool_config mode: ANY`

### Veredicto

No la usaría como única solución. Puede ser útil como intento adicional, pero no como garantía.

### Riesgos no considerados

#### Riesgo 1: `mode: ANY` puede no aplicar a `google_search`

Ustedes ya lo sospechan. En varias APIs, `function_calling_config` controla llamadas a funciones declaradas por el usuario, no necesariamente herramientas nativas como Google Search.

Si implementan esto creyendo que fuerza búsqueda y en realidad no lo hace, habrán creado una falsa sensación de seguridad.

Peor: el bug seguiría existiendo, pero ahora el equipo pensaría que está corregido.

---

#### Riesgo 2: El modelo puede usar la búsqueda superficialmente

Aunque `mode: ANY` funcione, eso no garantiza que use bien la búsqueda.

Puede buscar una vez, obtener una fuente parcialmente relacionada y luego inventar el resto.

Ejemplo peligroso:

> Pregunta: “¿Qué noticias de tecnología ocurrieron esta semana?”  
> El modelo busca “technology news this week”, obtiene una fuente real, y luego genera cinco noticias, tres de las cuales no están en la fuente.

Habría `groundingMetadata`, pero seguiría habiendo alucinación.

---

#### Riesgo 3: Costo y latencia

Forzar búsqueda para todo encarece la operación y degrada velocidad. Eso puede parecer aceptable al inicio, pero en producción genera incentivos para desactivar la protección.

Una protección cara y molesta suele terminar siendo eliminada.

---

#### Riesgo 4: Búsqueda innecesaria puede contaminar tareas locales

Para tareas como “resumí este texto”, una búsqueda externa puede introducir información que no estaba en el documento original.

Eso es malo si se espera fidelidad al input.

---

### Conclusión sobre A

A es un parche incompleto. Puede formar parte de una estrategia, pero nunca debe ser la única barrera.

---

## Opción B — Rechazar respuestas sin `groundingMetadata`

### Veredicto

Es correcta para tareas que requieren verificación web. Es incorrecta como regla global.

### Riesgos no considerados

#### Riesgo 1: Confunde “no hubo búsqueda” con “la tarea no requería búsqueda”

Este es el problema que ustedes ya detectaron.

Para una tarea local, no tener `groundingMetadata` no es fallo. Es comportamiento correcto.

Ejemplo:

> “Resumime este párrafo.”

Si Hipatia falla porque no hay fuentes web, el sistema se vuelve torpe.

---

#### Riesgo 2: Puede bloquear respuestas correctas pero no verificadas en web

Ejemplo:

> “Explicame qué es una mónada en programación funcional.”

Puede responder correctamente sin buscar. Pero si la política exige grounding, debería rechazar. Eso puede ser deseado o no, según el contrato de Hipatia.

La pregunta clave es:

> ¿Hipatia promete conocimiento general o promete investigación verificable?

No mezclen ambas promesas.

---

#### Riesgo 3: Metadata vacía puede deberse a fallos técnicos

Si no hay metadata, puede ser porque:

- el modelo decidió no buscar;
- la API no devolvió grounding;
- hubo cambio de formato;
- se perdió metadata en el parsing;
- el candidato seleccionado no fue el que tenía grounding;
- hubo una respuesta bloqueada, truncada o reintentada sin tools.

Todos esos casos deben tratarse como **no verificados**, pero conviene registrar la causa.

---

#### Riesgo 4: Rechazar sin retry puede ser demasiado agresivo

Para `REQUIRED_WEB`, una primera respuesta sin grounding debería generar al menos un retry controlado:

1. Primer intento con herramienta.
2. Si no hay grounding, segundo intento con instrucción explícita:
   > “Debes usar búsqueda web. Si no puedes usarla, responde exactamente: NO_VERIFICADO.”
3. Si sigue sin grounding, fallar cerrado.

No para darle más libertad al modelo, sino para reducir falsos negativos.

---

### Conclusión sobre B

B es la base correcta para modo `REQUIRED_WEB`, pero no debe aplicarse universalmente.

---

## Opción C — `mode: ANY` + rechazar si no hay metadata

### Veredicto

Es la mejor de las tres, pero sigue mal formulada si se aplica a toda Hipatia.

### Riesgos no considerados

#### Riesgo 1: Defensa en profundidad mal ubicada

La defensa correcta no es:

```text
Hipatia siempre debe buscar y fallar si no busca.
```

La defensa correcta es:

```text
Cuando la tarea requiere evidencia web, Hipatia debe buscar y fallar si no puede demostrarlo.
```

---

#### Riesgo 2: Complejidad sin contrato claro

Si implementan A + B sin definir políticas, van a tener bugs semánticos:

- ¿Por qué falló un resumen local?
- ¿Por qué buscó para una traducción?
- ¿Por qué agregó fuentes a una explicación conceptual?
- ¿Por qué rechazó una tarea válida sin necesidad de web?

La complejidad aceptable es la que codifica un contrato claro. La complejidad que intenta compensar ambigüedad conceptual es deuda técnica.

---

#### Riesgo 3: Metadata como única validación sigue siendo débil

Aun con C, necesitan revisar no solo si existe `groundingMetadata`, sino si hay citas utilizables.

Mínimo deberían validar:

```python
has_grounding_chunks = bool(meta.get("groundingChunks"))
has_grounding_supports = bool(meta.get("groundingSupports"))
```

Idealmente, deberían exigir que las afirmaciones principales tengan soporte.

---

### Conclusión sobre C

C es aceptable solamente dentro de una política `REQUIRED_WEB`. No como comportamiento universal de Hipatia.

---

# 3. La cuarta opción: política explícita de evidencia

La arquitectura correcta debería separar estas cosas:

```text
Faceta: Hipatia
Rol: investigación / análisis
Política de evidencia: depende de la tarea
```

No:

```text
Hipatia.grounding = True
```

sino:

```python
GroundingPolicy.OFF
GroundingPolicy.AUTO
GroundingPolicy.REQUIRED_WEB
GroundingPolicy.LOCAL_CONTEXT_ONLY
```

---

## Propuesta concreta

### Modo 1: `REQUIRED_WEB`

Para tareas donde la respuesta depende de hechos externos, actuales o verificables.

Ejemplos:

- “¿Qué pasó esta semana?”
- “Buscá información sobre X.”
- “¿Cuál es la versión actual de Y?”
- “¿Qué dice la documentación vigente?”
- “Compará precios actuales.”
- “Investigá este repositorio.”
- “Dame fuentes.”
- “Verificá si esto es cierto.”

Contrato:

```text
Debe usar búsqueda web.
Debe devolver fuentes.
Si no puede demostrar búsqueda, debe fallar.
```

Validación:

```python
if policy == REQUIRED_WEB:
    if not has_grounding_metadata(candidate):
        raise MuscleInvocationError("Respuesta no verificada: no hubo grounding web.")
```

Mejor aún:

```python
if policy == REQUIRED_WEB:
    if not has_grounding_chunks(candidate):
        raise MuscleInvocationError("No se obtuvieron fuentes web.")
    if not has_grounding_supports(candidate):
        mark_as_weakly_grounded_or_fail()
```

Respuesta visible:

```text
Estado de verificación: verificado con búsqueda web.
Fuentes consultadas:
- ...
```

Si falla:

```text
No pude verificar esta respuesta con búsqueda web. No voy a inventar una respuesta.
```

---

### Modo 2: `LOCAL_CONTEXT_ONLY`

Para tareas basadas exclusivamente en contenido entregado por el usuario.

Ejemplos:

- resumir un texto;
- extraer puntos clave;
- traducir;
- analizar código pegado;
- convertir formato;
- clasificar un documento local.

Contrato:

```text
No debe usar web.
Debe basarse solo en el material proporcionado.
Si falta información, debe decirlo.
```

Respuesta visible:

```text
Estado de verificación: basado únicamente en el material proporcionado por el usuario.
```

Esto resuelve el problema de Opción B.

No hay `groundingMetadata`, pero no es un fallo, porque la fuente es el input local.

---

### Modo 3: `AUTO`

Para uso general donde buscar puede ayudar, pero no es obligatorio.

Ejemplos:

- “Explicame qué es X.”
- “Dame contexto sobre Y.”
- “Ayudame a entender este concepto.”

Contrato:

```text
Puede buscar.
Si busca, debe mostrar fuentes.
Si no busca, debe declarar que la respuesta no fue verificada en web.
```

Respuesta visible si no buscó:

```text
Estado de verificación: no verificado en web; respuesta basada en conocimiento general del modelo.
```

Respuesta visible si buscó:

```text
Estado de verificación: verificado parcialmente con búsqueda web.
Fuentes consultadas:
- ...
```

---

### Modo 4: `OFF`

Para creatividad, redacción, brainstorming o tareas donde la verdad factual no es el eje.

Ejemplos:

- “Escribí un poema.”
- “Generá nombres para un producto.”
- “Reescribí este texto con tono formal.”
- “Proponé ideas.”

Contrato:

```text
No hay verificación factual.
No debe fingir que la hay.
```

---

# 4. Cómo distinguir “no buscó porque inventó” de “no buscó porque no hacía falta”

No se distingue mirando solo la respuesta. Se distingue por la **política asignada antes de llamar al modelo**.

Ese es el punto crítico.

El flujo correcto es:

```text
Usuario pide algo
↓
Orquestador clasifica la necesidad de evidencia
↓
Se llama a Hipatia con una política explícita
↓
Se valida la respuesta contra esa política
↓
Se entrega, etiqueta o rechaza
```

No:

```text
Modelo responde
↓
Vemos si hubo grounding
↓
Intentamos inferir si hacía falta
```

Eso es tarde y frágil.

---

## Reglas prácticas de clasificación

Una tarea debería ir a `REQUIRED_WEB` si contiene cualquiera de estas señales:

### Recencia

- “hoy”
- “esta semana”
- “este mes”
- “actual”
- “reciente”
- “último”
- “nuevo”
- “vigente”
- “2025”, “2026”, etc. cuando la fecha importa

Ejemplo:

> “¿Qué noticias de tecnología ocurrieron esta semana?”

Debe ser `REQUIRED_WEB`.

---

### Solicitud explícita de investigación

- “investigá”
- “buscá”
- “verificá”
- “confirmá”
- “comprobá”
- “dame fuentes”
- “con referencias”
- “según la documentación”
- “qué dice X sitio”

Debe ser `REQUIRED_WEB`.

---

### Datos externos variables

- precios;
- versiones;
- disponibilidad;
- APIs;
- leyes vigentes;
- clima;
- eventos;
- noticias;
- mercado;
- estado de repositorios;
- publicaciones recientes;
- vulnerabilidades;
- benchmarks actuales.

Debe ser `REQUIRED_WEB`.

---

### Tarea local pura

Si el usuario entrega el material completo y pide operar sobre él:

- resumir;
- traducir;
- corregir;
- clasificar;
- extraer;
- transformar;
- analizar código pegado.

Debe ser `LOCAL_CONTEXT_ONLY`.

---

### Conceptos estables

Preguntas conceptuales generales:

- “¿Qué es TCP?”
- “¿Cómo funciona OAuth?”
- “Explicame backpropagation.”

Pueden ser `AUTO` o `OFF`, dependiendo del estándar deseado.

Si Hipatia promete ser investigadora estricta, usen `AUTO` con declaración visible. Si JAX permite conocimiento general, `OFF` también es aceptable.

---

# 5. Cómo hacer que Hipatia declare explícitamente si verificó o no

Sí, pero la declaración no debe depender de que el modelo sea honesto. Debe generarla el wrapper de JAX con base en evidencia observable.

No basta con prompt:

> “Decí si buscaste o no.”

Eso ayuda, pero no es control de seguridad.

La etiqueta debe venir del sistema.

---

## Formato recomendado de respuesta

Toda respuesta de Hipatia debería incluir una cabecera o metadata interna:

```text
Estado de verificación: VERIFICADO_WEB
```

o:

```text
Estado de verificación: SIN_VERIFICACIÓN_WEB
```

o:

```text
Estado de verificación: BASADO_EN_CONTEXTO_LOCAL
```

o:

```text
Estado de verificación: NO_VERIFICADO_POR_FALLO_DE_GROUNDING
```

Preferiblemente como estructura interna:

```json
{
  "answer": "...",
  "verification": {
    "policy": "REQUIRED_WEB",
    "status": "VERIFIED_WEB",
    "grounding_chunks_count": 4,
    "grounding_supports_count": 7,
    "sources": [
      {
        "title": "...",
        "uri": "..."
      }
    ]
  }
}
```

Y luego renderizan eso para humanos.

---

## Regla de oro

La declaración debe basarse en esto:

```python
if policy == REQUIRED_WEB and has_grounding_metadata:
    status = "VERIFIED_WEB"
elif policy == REQUIRED_WEB and not has_grounding_metadata:
    raise MuscleInvocationError(...)
elif policy == LOCAL_CONTEXT_ONLY:
    status = "BASED_ON_LOCAL_CONTEXT"
elif policy == AUTO and has_grounding_metadata:
    status = "WEB_GROUNDED"
elif policy == AUTO and not has_grounding_metadata:
    status = "UNVERIFIED_MODEL_KNOWLEDGE"
```

No en esto:

```text
El modelo dijo que buscó.
```

---

# 6. Recomendación concreta de implementación

## Paso 1: Reemplazar `self.grounding: bool`

En vez de:

```python
self.grounding = True
```

usar algo como:

```python
class GroundingPolicy(Enum):
    OFF = "off"
    AUTO = "auto"
    REQUIRED_WEB = "required_web"
    LOCAL_CONTEXT_ONLY = "local_context_only"
```

---

## Paso 2: Construir payload según política

Ejemplo:

```python
if policy in {GroundingPolicy.AUTO, GroundingPolicy.REQUIRED_WEB}:
    payload["tools"] = [{"google_search": {}}]
```

Para `REQUIRED_WEB`, pueden intentar `tool_config`, pero como optimización, no como garantía:

```python
if policy == GroundingPolicy.REQUIRED_WEB:
    payload["tool_config"] = {
        "function_calling_config": {
            "mode": "ANY"
        }
    }
```

Pero deben tratar esto como experimental hasta validarlo.

---

## Paso 3: Prompt específico según política

Para `REQUIRED_WEB`:

```text
Debes usar búsqueda web para responder.
No respondas con conocimiento general si no puedes verificar.
Incluye solo afirmaciones respaldadas por fuentes.
Si no puedes acceder a fuentes, responde que no pudiste verificar.
```

Para `LOCAL_CONTEXT_ONLY`:

```text
Responde únicamente usando el material proporcionado por el usuario.
No uses conocimiento externo salvo para lenguaje general.
Si el material no contiene la información necesaria, dilo.
```

Para `AUTO`:

```text
Puedes usar búsqueda web si lo necesitas.
Si no usas búsqueda web, declara que la respuesta no fue verificada en web.
```

De nuevo: esto es ayuda, no garantía.

---

## Paso 4: Validación posterior obligatoria

Para `REQUIRED_WEB`:

```python
meta = candidate.get("groundingMetadata") or {}
chunks = meta.get("groundingChunks") or []
supports = meta.get("groundingSupports") or []

if not chunks:
    raise MuscleInvocationError(
        "Grounding requerido, pero Gemini no devolvió fuentes."
    )
```

Opcionalmente:

```python
if not supports:
    raise MuscleInvocationError(
        "Grounding requerido, pero Gemini no devolvió soportes de citas."
    )
```

Depende de qué tan estrictos quieran ser.

Mi recomendación: para noticias, datos actuales y verificación factual, exigir `chunks` como mínimo y registrar ausencia de `supports` como advertencia severa. Si quieren máxima robustez, exijan ambos.

---

## Paso 5: Retry controlado

Para `REQUIRED_WEB`:

```text
Intento 1: tools + instrucción normal.
Si no hay grounding:
Intento 2: instrucción más estricta.
Si no hay grounding:
fallar cerrado.
```

No más de un retry. Si no, se vuelve caro y opaco.

---

## Paso 6: Renderizar siempre el estado de verificación

No solo agregar fuentes cuando existen. Eso es precisamente parte del bug silencioso.

Actualmente hacen:

```text
Si hay fuentes, agrego “Fuentes consultadas”.
Si no hay fuentes, no digo nada.
```

Eso permite el fallo silencioso.

Debe cambiar a:

```text
Siempre declaro el estado de verificación.
```

Ejemplos:

```text
Estado de verificación: verificado con búsqueda web.
Fuentes consultadas:
- ...
```

o:

```text
Estado de verificación: no verificado en web; respuesta basada en conocimiento general del modelo.
```

o:

```text
Estado de verificación: basado únicamente en el texto proporcionado por el usuario.
```

---

# 7. Sobre el caso específico de “noticias de tecnología esta semana”

Esto debe ser tratado como fallo crítico.

Clasificación correcta:

```text
policy = REQUIRED_WEB
```

Resultado aceptable:

```text
Respuesta con fuentes reales y fechas verificables.
```

Resultado no aceptable:

```text
Respuesta sin groundingMetadata.
```

Acción correcta:

```text
MuscleInvocationError
```

Mensaje visible:

```text
No pude verificar noticias recientes mediante búsqueda web. No voy a generar una lista sin fuentes.
```

La respuesta que inventó “Tech Insights Global” y “Digital Future Magazine” no es un fallo menor de calidad. Es una violación del contrato de Hipatia como investigadora.

---

# 8. Cuidado con una trampa: fuentes al final no bastan

Agregar:

```text
Fuentes consultadas:
- ...
```

ayuda, pero no resuelve todo.

Una respuesta puede tener fuentes reales al final y aun así contener afirmaciones no soportadas.

Lo ideal es producir citas por afirmación o por párrafo:

```text
OpenAI anunció X el lunes... [1]
Google publicó Y... [2]
Nvidia presentó Z... [3]
```

Y después:

```text
[1] Fuente...
[2] Fuente...
[3] Fuente...
```

Si Gemini entrega `groundingSupports`, deberían aprovecharlos para mapear segmentos a fuentes.

Si solo tienen `groundingChunks`, al menos no permitan que el usuario confunda “consulté fuentes” con “cada frase está demostrada”.

Una etiqueta honesta sería:

```text
Estado de verificación: búsqueda web usada; revise las fuentes asociadas. 
```

Pero para tareas críticas, deberían intentar citas granulares.

---

# 9. Qué haría yo

Implementaría esto:

## Decisión

Adoptar una variante de C, pero solo dentro de una política nueva:

```text
Opción D: GroundingPolicy por tarea + fail closed cuando REQUIRED_WEB.
```

Concretamente:

1. Reemplazar `grounding: bool` por `grounding_policy`.
2. Para `REQUIRED_WEB`:
   - activar `google_search`;
   - probar `tool_config mode: ANY` si la API lo soporta;
   - instrucción explícita de no responder sin búsqueda;
   - validar `groundingMetadata`;
   - fallar si no hay fuentes;
   - renderizar estado de verificación y fuentes.
3. Para `LOCAL_CONTEXT_ONLY`:
   - no activar web;
   - declarar que se basa en el input local.
4. Para `AUTO`:
   - permitir web;
   - si no hay metadata, etiquetar como no verificado.
5. Nunca entregar una respuesta sin etiqueta de verificación.

---

# 10. Respuestas directas a tus preguntas

## 1. ¿Cuál opción es la correcta?

La correcta es una cuarta:

```text
Grounding por política de tarea, no por faceta.
```

Dentro de esa cuarta opción, para tareas que requieren web, usen C:

```text
intentar forzar búsqueda + rechazar si no hay grounding.
```

Pero no apliquen C globalmente.

---

## 2. ¿Qué riesgo de cada opción no están considerando?

Resumen:

- **A:** puede no funcionar con `google_search`; puede dar falsa seguridad; búsqueda no equivale a verificación completa.
- **B:** rompe tareas locales; confunde ausencia de web con fallo; puede ser demasiado agresiva sin clasificación previa.
- **C:** es robusta solo si se aplica a tareas correctas; si se aplica globalmente, vuelve Hipatia rígida, cara y propensa a fallos innecesarios.

---

## 3. ¿Cómo distinguir no buscó porque inventó vs no buscó porque no hacía falta?

No se distingue después. Se define antes.

La tarea debe llegar a Hipatia con una política:

```text
REQUIRED_WEB → si no buscó, fallo.
LOCAL_CONTEXT_ONLY → si no buscó, correcto.
AUTO → si no buscó, entregar como no verificado.
OFF → no aplica verificación.
```

Sí: el flag `grounding` debe ser por tarea, no fijo en la faceta.

---

## 4. ¿Hay forma de hacer que Hipatia declare “esto lo sé de memoria, no lo verifiqué”?

Sí, pero no confíen en que lo declare por voluntad propia.

El sistema debe imponerlo.

Si no hay `groundingMetadata`, el wrapper debe añadir:

```text
Estado de verificación: no verificado en web; respuesta basada en conocimiento general del modelo.
```

Si la política era `REQUIRED_WEB`, no debe añadir esa etiqueta y entregar la respuesta; debe rechazarla.

---

# Veredicto final

El bug no es solo técnico. Es contractual.

Hipatia actualmente aparenta ser una investigadora con grounding, pero en realidad opera como:

```text
investigadora cuando Gemini decide buscar;
modelo de memoria cuando Gemini decide no buscar;
sin aviso claro al usuario.
```

Eso es inaceptable para una faceta investigadora.

La corrección mínima segura es:

```text
Si la tarea requiere web y no hay groundingMetadata, no se entrega respuesta.
```

La corrección adecuada es:

```text
Política de evidencia por tarea + validación posterior + declaración visible de estado de verificación.
```

No permitan respuestas factual-actuales sin fuentes. Un “no pude verificarlo” es infinitamente mejor que una invención elegante.
