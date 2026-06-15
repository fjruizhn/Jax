faceta: thot

# DECISIÓN A VALIDAR — Fix del grounding silenciosamente roto de Hipatia

Thot, encontramos un bug serio en JAX gracias al protocolo de verificación que vos mismo nos enseñaste. Necesito tu análisis crítico antes de tocar código.

## EL BUG

Hipatia es nuestra faceta investigadora: un HttpMuscle que llama a Gemini (gemini-2.5-flash) con grounding activado (búsqueda web de Google). El grounding se pide así en el código:

```python
if self.grounding:
    payload["tools"] = [{"google_search": {}}]
```

Luego el código intenta extraer las fuentes consultadas:

```python
meta = candidate.get("groundingMetadata", {})
chunks = meta.get("groundingChunks", [])
# ... si hay chunks, agrega sección "Fuentes consultadas" al final
```

## EL PROBLEMA RAÍZ

`google_search` en la API de Gemini es una CAPACIDAD, no una OBLIGACIÓN. Le decimos a Gemini "podés buscar si querés", y Gemini DECIDE si busca.

- Cuando Gemini cree que NO sabe → busca → devuelve groundingMetadata → el código muestra las fuentes ✅
- Cuando Gemini cree que SÍ sabe → NO busca → responde de memoria → groundingMetadata viene vacío → el código no agrega fuentes → PERO la respuesta de memoria se entrega igual, como si fuera válida ❌

## LA EVIDENCIA (prueba de fuego)

Le preguntamos a Hipatia "¿qué noticias de tecnología ocurrieron esta semana?" — algo IMPOSIBLE de saber de memoria. Resultado: inventó dos noticias completas con medios falsos ("Tech Insights Global", "Digital Future Magazine") y fechas inventadas. CERO búsqueda. Y las entregó con total confianza, sin avisar que no buscó.

En investigaciones previas (sobre el repo crabfleet) SÍ buscó y trajo fuentes reales (enlaces vertexaisearch de Google). El comportamiento es inconsistente: busca cuando "no sabe", inventa cuando "cree saber".

## EL RIESGO

Hipatia inventa con la misma elegancia y confianza cuando busca que cuando alucina. La ÚNICA señal de que alucinó es la ausencia de la sección "Fuentes consultadas" al final. Un humano que no revisa esa señal toma datos inventados como verificados. Esto es exactamente el tipo de fallo silencioso del que vos nos advertiste con LAS MANOS.

## LAS TRES OPCIONES DE FIX

**Opción A — Forzar grounding con tool_config mode:ANY**
La API de Gemini permite `tool_config` con `function_calling_config: {mode: "ANY"}` para forzar que el tool se use SIEMPRE. En vez de "buscá si querés", es "buscá siempre".
- Pro: Hipatia siempre busca.
- Contra: busca hasta para preguntas que no lo necesitan; gasta más cuota; y NO está 100% confirmado que mode:ANY funcione para google_search (puede ser solo para function calling clásico).

**Opción B — Detectar y rechazar respuestas sin grounding**
Si `grounding=true` pero `groundingMetadata` viene vacío, lanzar MuscleInvocationError en vez de entregar la respuesta. Hipatia diría "no pude verificar esto" en lugar de inventar.
- Pro: nunca entrega una respuesta no verificada disfrazada de verificada.
- Contra: fallaría cuando la pregunta legítimamente no necesita búsqueda (ej. "resumime este texto que te paso").

**Opción C — Las dos juntas (defensa en profundidad)**
Forzar grounding con mode:ANY Y detectar si aun así no hubo metadata.
- Pro: cinturón y tirantes.
- Contra: complejidad, y arrastra los contras de ambas.

## TU TRABAJO

1. ¿Cuál opción es la correcta? ¿O hay una cuarta que no estamos viendo?
2. ¿Qué riesgo de cada opción no estamos considerando?
3. La Opción B tiene un problema: a veces Hipatia legítimamente NO necesita buscar (resumir un texto dado). ¿Cómo distinguimos "no buscó porque inventó" de "no buscó porque no hacía falta"? ¿Debería el flag grounding ser por-tarea en vez de fijo en la faceta?
4. ¿Hay una forma de hacer que Hipatia DECLARE explícitamente "esto lo sé de memoria, no lo verifiqué" en vez de fingir que es un hecho?

Sé brutal. Un dato inventado disfrazado de verificado es peor que un "no sé" honesto.
