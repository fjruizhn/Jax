# F2-D: render reutilizado con revalidación de autoridad — 2026-10-03

## Cambio medido

JAX master antes del cambio: `7024529e1ede4b3a3de92eefe4f60e3d2f567f01`.
Platform usado para ambas mediciones: `1e2888c32d7a6221a4b4395d2d2018845e31468f`.
Python 3.14.4, hall9000. La medición usa 16.000 caracteres con el mismo texto de carga
del arnés `carga/chat-f2d` (`125201728700ad6a713eeb6b56c85d5bfa63eea2`). El proveedor se
simuló con una espera fija de 1 ms; no se llamó a Ollama, GPU ni proveedor externo.

El tramo ejercitado fue el productor real `project_provider_contract` → F2-C →
`prepare_governed_chat_response` → respuesta ASGI `PreparedGovernedChatResponse` → F2-D.
Se usó el repositorio determinista de `backend/tests/test_output_lifecycle_transport.py`,
por lo que esta medida cubre la frontera de salida y sus transiciones, no la persistencia
MariaDB ni el servidor HTTP/socket completo.

## Resultado de carga de la frontera ASGI

Dos corridas secuenciales de 50 solicitudes por nivel, en bucle cerrado, una salida JSON de
16 KB por solicitud. La tabla muestra la mediana entre corridas:

| Concurrencia | Antes req/s | Después req/s | Antes p95 ms | Después p95 ms | p95 > 500 ms |
|---:|---:|---:|---:|---:|---|
| 1 | 28,34 | 59,30 | 36,86 | 19,61 | no / no |
| 5 | 29,45 | 66,32 | 173,92 | 78,10 | no / no |
| 10 | 28,61 | 61,94 | 354,14 | 189,87 | no / no |
| 25 | 28,71 | 65,10 | 879,30 | 387,38 | sí / no |

La frontera deja de escalar aproximadamente desde c=1 antes del cambio; después mantiene
su rendimiento hasta c=25. El rendimiento mejora entre 2,1x y 2,3x en los cuatro niveles.
Con el criterio medido de p95 mayor que 500 ms, la degradación aparece en c=25 antes del
cambio y no aparece hasta c=25 después. No se extrapola a más concurrencia ni a la aplicación
completa.

## Micro-medición de revalidación

El harness de `carga/chat-f2d` también midió 200 llamadas a `revalidate_for_transport` sobre
la misma unidad de 16 KB, fuera del perfilador:

| | Antes | Después |
|---|---:|---:|
| Mediana por llamada | 7,05 ms | 0,09 ms |
| p95 por llamada | 11,16 ms | 0,09 ms |
| Cinco revalidaciones por turno | ~35,2 ms | ~0,5 ms |

Las revalidaciones dejan de recorrer payload y gramáticas para renderizar otra vez. Cada una
verifica el envelope y el digest de la proyección efectiva, y vuelve a validar scope,
referencias y todos los claims visibles; para `CURRENT_OBSERVATION`, vuelve a verificar el
recibo F2-B con la hora de validación entregada por F2-D. La salida reutilizada es el mismo
`RenderedText` inmutable ligado al unit opaco. El unit vive solo en la solicitud; descartarlo
invalida la reutilización. No se introdujo caché global, TTL ni invalidación distribuida.

La segunda lectura/validación de recibo sigue inmediatamente antes de cada límite de envío
ASGI, incluidos los puntos posteriores a `TRANSPORT_COMMITTING` y al inicio de respuesta.
No se modificaron estados F2-D ni la afirmación de entrega.

## Límites de la medición

El arnés completo `chat_f2d_orquestar.py` no se ejecutó. Este lee `/etc/jax/.env` y utiliza
credenciales del entorno productivo contra la base compartida `jax_memory_test`; aquí no hay
una cuenta de prueba aislada y el daemon Docker no es accesible al usuario de la sesión.
No se usaron esas credenciales ni se escribió en ninguna base. Por tanto, los valores de esta
página son una carga real de la frontera F2-C/F2-D en ASGI con repositorio determinista, no
un reemplazo de una carga HTTP+MariaDB de la plataforma. La medición histórica del 2026-10-02
no se presenta como prueba de los masters actuales.

La suite completa de JAX tampoco pudo recolectarse: 15 módulos de Faro/LAS VOCES requieren
el paquete `mcp`, ausente en este entorno. Los grupos enfocados F2-C/F2-D/runtime-status sí
se ejecutaron por separado.
