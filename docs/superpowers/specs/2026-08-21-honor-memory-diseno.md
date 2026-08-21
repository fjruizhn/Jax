# Diseño: `people.honor_memory`

Bloque 2 (2026-08-21). Semántica definida por Fernando, cierra la pregunta
que dejó bloqueada esta columna desde ronda 7 (2026-08-20): **cosmético,
personalidad conversacional — NO es una faceta, NO despacha, NO ejecuta.**

No implementado a propósito esta sesión: es la única pieza de todo el
Bloque 2 que cambia cómo *suena* una respuesta en producción (todo lo
demás era infraestructura, verificable con tests/logs). Fernando quiere
leer el resultado antes de que sea comportamiento real.

## Disparador

Cuando el nombre o apodo de una persona con `honor_memory=TRUE` aparece
literal en el mensaje del turno actual. Mismo criterio de matching que ya
usa `touch_person_mentions()` (`jax/memory/db.py`) — pero evaluado en
**tiempo real**, antes de despachar a la faceta, no en el worker de
destilación (`jax/memory/worker.py`, cada 20 min): el worker es post-hoc,
demasiado tarde para afectar el tono de la respuesta que se está por
generar.

## Función nueva: `get_honored_mentions(text: str) -> list[dict]`

Vive en `jax/memory/db.py`, mismo archivo y mismo patrón de query que
`touch_person_mentions`:

```sql
SELECT name, nickname FROM people
WHERE honor_memory = TRUE AND (name IN (...) OR nickname IN (...))
```

**Debe ser de solo lectura — sin efectos secundarios.** No actualiza
`last_mentioned`, no escribe nada. Ese `UPDATE` sigue viviendo donde ya
está, en el worker de 20 min (`touch_person_mentions`, hoy sin conectar —
deuda aparte, ver `DEUDA.md`). Si el día que se implemente esto hace falta
ese update desde el mismo lugar, se llama a `touch_person_mentions` por
separado — `get_honored_mentions` no lo hace por su cuenta.

## Efecto

Si la lista devuelta no está vacía, se antepone una línea al
prompt/persona de esa llamada puntual. **El texto de esa línea es
EJEMPLO, no definitivo** — es lo único que el sistema va a decir sobre
esto, Fernando quiere ajustar la redacción antes de implementar:

```
Si mencionás a {name} en tu respuesta, hacelo con el mismo respeto que le
tiene esta memoria.
```

Sin frase canónica fija guardada en la DB — cada faceta la expresa con su
propia voz. No cambia routing, no cambia qué faceta se despacha, no
ejecuta nada.

## Dónde engancha

REPL y Jacobs/Mesa web tienen dos arquitecturas de prompt distintas, ya
documentado en CONTEXT.md §T3 (ronda 7): REPL usa `Muscle` propio
(`jax/muscles/*.py`), Jacobs/Mesa usa `facet_resolver`/personas de
`jacobs/executor.py`. Este diseño no las unifica — eso es un cambio mucho
más grande, fuera de este alcance. Cada lado llama a la misma
`get_honored_mentions()` compartida antes de armar su prompt:

- REPL: en el punto donde se compone el mensaje antes de pasarlo al
  `Muscle` activo (`jax/core/main.py`).
- Jacobs/Mesa: en la composición de persona por step, junto a donde
  `_FACET_PERSONAS` se resuelve (`jacobs/executor.py`).

## Esfuerzo

Bajo — una función nueva calcada de `touch_person_mentions`, más un call
site en cada uno de los dos puntos de composición de prompt. Bloqueado
solo en que Fernando revise la redacción de la línea de ejemplo antes de
que se vuelva texto real de producción.

En memoria de Jairo Urbina.
