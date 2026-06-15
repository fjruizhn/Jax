faceta: jekyll

# VALIDACIÓN DE DISEÑO — Política de evidencia para Hipatia

Jekyll, necesito tu mirada sobre un diseño que ya pasó por Thot. Él aportó la disciplina; busco tu ojo para la elegancia y la coherencia. No repitas lo que Thot ya dijo — buscá lo que él no vio.

## EL CONTEXTO

Hipatia es nuestra faceta investigadora: llama a Gemini (gemini-2.5-flash) con grounding (búsqueda web). Descubrimos un bug grave: `google_search` en Gemini es una CAPACIDAD, no una OBLIGACIÓN. Gemini busca cuando "cree que no sabe" e inventa con total confianza cuando "cree que sabe", sin avisar. Le preguntamos por noticias de "esta semana" e inventó dos medios falsos con fechas inventadas, sin buscar.

## EL VEREDICTO DE THOT (ya aceptado)

El error de raíz: `grounding` es propiedad de la FACETA cuando debería ser propiedad de la TAREA. Thot propuso cuatro políticas de evidencia por tarea:

```
OFF                → no buscar (tarea local o creativa)
AUTO               → puede buscar; si NO buscó, el sistema DECLARA "no verificado en web"
REQUIRED_WEB       → debe buscar; si no hay groundingMetadata, FALLA cerrado (MuscleInvocationError)
LOCAL_CONTEXT_ONLY → responde solo con el texto provisto; declara que se basa en el input
```

Reglas firmes de Thot:
- El SISTEMA declara el estado de verificación, no el modelo (no confiar en que Hipatia lo diga por voluntad).
- NUNCA entregar una respuesta sin etiqueta de estado de verificación.
- Presencia de grounding no es garantía total; validar groundingChunks Y groundingSupports.
- Para REQUIRED_WEB: un retry controlado con instrucción más estricta antes de fallar.

## EL DISEÑO DE IMPLEMENTACIÓN PROPUESTO

En JAX, las tareas se lanzan con archivos .md. Ya existe un header `faceta: <nombre>` en la primera línea que run_task lee. La propuesta es agregar un segundo header opcional:

```
faceta: hipatia
grounding: required_web
```

- Si no se especifica `grounding:` → AUTO por defecto (nunca falla silencioso).
- En base.py: reemplazar `grounding: bool` por `grounding_policy: str`.
- Siempre renderizar al final una línea "Estado de verificación: ..." según la política y el resultado.

## TU TRABAJO (lo que Thot NO cubrió)

1. **Coherencia del sistema:** Hipatia tiene grounding, pero Jekyll (vos) y Thot NO. ¿Esta política de evidencia debería aplicar solo a Hipatia, o todas las facetas deberían declarar su "estado de verificación" de algún modo? ¿Hay una asimetría peligrosa en que solo una faceta declare honestidad sobre sus fuentes?

2. **La experiencia humana:** Fernando lee estos resultados. ¿Cómo debería verse la "etiqueta de estado de verificación" para que sea clara sin ser ruidosa? Thot dio ejemplos funcionales; vos tenés mejor ojo para la forma. ¿Una línea? ¿Un encabezado? ¿Un símbolo?

3. **El default AUTO:** Thot dice que si no se especifica política, sea AUTO. Pero AUTO permite que Hipatia responda de memoria (etiquetado). ¿Es AUTO el default correcto para una faceta cuyo nombre y contrato es "la investigadora"? ¿O el default de Hipatia debería ser REQUIRED_WEB, y AUTO algo que se pide explícitamente?

4. **Filosofía:** El nombre Hipatia honra a la guardiana del conocimiento de Alejandría. Una guardiana del conocimiento que inventa fuentes traiciona su nombre. ¿Hay algo en el contrato mismo de Hipatia —su system_prompt, su identidad— que debería cambiar a la luz de esto, más allá del fix técnico?

Pensá en la coherencia y la belleza del sistema, no solo en la corrección. Thot ya cubrió la corrección.
