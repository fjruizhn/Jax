# Contribuir a JAX

Gracias por el interés. Este es un proyecto de una sola persona hasta
ahora -- las reglas de abajo son las que ya se aplican al propio trabajo
del mantenedor, no un estándar nuevo inventado para terceros.

## Antes de escribir código

- Si vas a proponer un cambio de arquitectura o algo que toca gobernanza
  (el gate de autoridad de tool-calling, el catálogo de capabilities, la
  validación de planes de Jacobs), abrí un issue primero. Cambios chicos
  y acotados pueden ir directo a PR.
- Leé `policy/` -- son las reglas de método del proyecto, con su
  justificación y, cuando aplica, el mecanismo de CI que las hace cumplir.

## Reglas de la casa

**No suponer, verificar contra el sistema real.** Si un PR afirma que algo
"debería funcionar" o "según la documentación", el reviewer va a pedir
evidencia de que efectivamente funciona contra el código/sistema real, no
contra lo que un comentario o un doc dice que hace. Esto aplica en ambas
direcciones: si encontrás una discrepancia entre este repo y su propia
documentación, la documentación está mal hasta que se demuestre lo
contrario, no el código.

**P10 -- ningún camino de error termina en éxito reportado.** Un
`except` que atrapa un error y sigue como si nada pasó (sin loguear, sin
propagar, sin que quien llama se entere) es una violación, incluso si la
intención era "no romper el flujo". El CI corre un scanner AST
(`policy/tests/test_no_fail_open_except.py`) que busca bloques
`except: pass` en todo el árbol. Si tu `except: pass` es legítimo (un
best-effort real, donde de verdad no importa si falló), marcalo en la
misma línea con un comentario `# fail-soft: <razón concreta>` -- el
scanner lo acepta, pero exige que la razón esté escrita, no implícita.
Sin esa marca, el CI falla el PR.

**Tests obligatorios.** Un cambio de comportamiento sin un test que lo
cubra no se considera terminado. Preferimos tests que corran contra el
sistema real (una base de datos de prueba, no todo mockeado) cuando el
comportamiento que se prueba depende de esa integración -- un mock que
siempre dice "sí" no prueba nada del camino real.

**Backups antes de modificar, y verificación antes de reportar éxito.**
Antes de tocar un archivo existente de forma no trivial, dejá un respaldo
local (no versionado). Antes de decir "listo" en un PR o una conversación,
corré lo que haga falta para confirmarlo con evidencia real -- un
`py_compile` limpio, una suite verde, una llamada real que devuelve lo
esperado. "Debería andar" no es un estado final.

## Qué esperamos de un PR

- Rama dedicada, con nombre que describa el cambio.
- Commits que expliquen el *por qué*, no solo el *qué* -- el diff ya dice
  qué cambió.
- CI verde (incluye el scanner de P10 de arriba) antes de pedir review.
- Si el cambio toca una de las cuatro fuentes de verdad sobre
  capabilities (la base de datos, el vocabulario cerrado del planificador,
  el catálogo de motores, o los alias semánticos del executor), decilo
  explícitamente en la descripción del PR -- es la parte del sistema que
  más fácil se desincroniza si se toca una fuente sin las demás.

## Qué NO esperamos

No hace falta que el PR sea perfecto ni que cubra todos los casos límite
imaginables -- si algo queda pendiente, decilo en la descripción en vez de
prometer que está completo. Preferimos deuda declarada a deuda escondida.
