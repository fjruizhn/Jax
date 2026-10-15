# Rotación de la credencial Max del catálogo y Hyde

HISTORIA · Codex, hall9000 · 15-oct-2026 (America/Tegucigalpa).

## Decisión y motivo

Fernando pidió adelantar la renovación del pendiente fechado para el
6-sep-2027. El token anterior se había creado el 27-sep-2026. La decisión de
usar `claude setup-token` de la cuenta Max, sin API key, viene de Fernando
(27-sep-2026; ver `2026-09-27-catalogo-modelos-y-credencial-max.md`).

## Ejecución y evidencia

- Antes del cambio, `claude auth status` informó una sesión `claude.ai` de
  tipo `max`; `jax-platform`, `jax-las-manos` y el timer del catálogo estaban
  activos. El journal del 15-oct mostró sincronizaciones `ok=True`, sin
  proveedores fallidos.
- `claude setup-token` exigió autenticación interactiva en navegador. Fernando
  terminó el flujo en su terminal y guardó el token dos veces mediante
  `/srv/jax-prod/jax-platform/ops/guardar-token-claude.sh` a las 12:55 CST.
  El script produjo `/etc/jax/.env.bak-20261015-125537` antes de sustituir
  el archivo. No se registró el valor del token.
- Después del cambio, `/etc/jax/.env` conservó `root:jaxsvc` y modo `640`.
  La copia fechada conservó modo `640` y se restauró a un archivo temporal en
  `/dev/shm`; su SHA-256 coincidió con el origen. Esta fue una prueba de
  restauración de archivo, sin sobrescribir la configuración en uso.
- Con la credencial guardada, `GET https://api.anthropic.com/v1/models?limit=1`
  respondió HTTP 200 y devolvió un modelo. La variable única en el archivo
  tenía la forma esperada y 108 caracteres. Una captura incompleta previa de
  79 caracteres devolvió HTTP 401 y se descartó sin tocar producción.
- `systemctl restart jax-platform jax-las-manos` terminó en 0 a las 12:56 CST.
  Ambos servicios quedaron `active/running`; la variable cargada en cada
  proceso coincidió exactamente con el archivo, sin imprimirla. El journal
  de ambas unidades no mostró advertencias tras el reinicio. El timer del
  catálogo siguió activo.

Fuente de estas observaciones: `stat`, `systemctl show`, `/proc/<pid>/environ`
(comparación en memoria), `journalctl`, restauración temporal y respuesta HTTP
real consultados el 15-oct-2026 por Codex. Son historia, no prueba del estado
vigente en una sesión posterior.

## Lecciones y siguiente vencimiento

La interfaz de `setup-token` puede insertar formato de terminal en la salida;
extraer una coincidencia parcial no basta. Validar la credencial completa con
la API antes de instalarla evitó afectar producción. El flujo fiable en este
caso fue la terminal interactiva de Fernando y el script existente.

El CLI declara una validez de un año para este tipo de token. Si el plazo se
cuenta desde hoy, la caducidad sería aproximadamente el 15-oct-2027; es una
estimación, no una fecha de expiración consultada al emisor. El nuevo aviso
queda en `claude-skills/PENDIENTES.md` para el 24-sep-2027.

Alternativas descartadas: almacenar la captura parcial de la terminal
(HTTP 401) y leer el token en voz alta o pegarlo en el chat. No hay cambio de
código ni de esquema pendiente de esta rotación.
