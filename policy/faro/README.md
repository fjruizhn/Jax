# policy/faro — reglas del Faro

Este directorio contendrá las reglas individuales que el **Rule Authority Kernel**
(Faro F1.1) evalúa: una regla solo produce `PERMIT` si sus bytes exactos coinciden
con una ratificación individual firmada por `human:fernando` en Block 4 y el
snapshot confiable la carga desde un commit Git exacto.

**HOY NO HAY NINGUNA REGLA ACTIVA.** Ningún archivo aquí está ratificado ni
vigente. Los ejemplos ratificables existen únicamente como fixtures de prueba en
`tests/policy/fixtures/faro_rules/`. Sin ratificación individual, nada de esto
autoriza acto alguno.

## Qué vive aquí

- `<nombre>.yaml` — reglas Faro (cuando existan): solo blobs `100644` con nombre
  canónico `[a-z][a-z0-9-]{0,63}.yaml` en ruta directa. Sin symlinks, sin
  submodules, sin bit de ejecución, sin subdirectorios con YAML.
- `schemas/rule-v1.schema.json` — espejo legible por máquina del shape cerrado.
  El validador real es `policy/rule_authority/schema.py`; una prueba los mantiene
  sincronizados.
- `README.md` — este archivo. Ni el README ni los esquemas son reglas: el loader
  los ignora al enumerar.

## El contrato de bytes (F1.1 §5)

`rule_content_hash` es `sha256` sobre los **bytes crudos del blob**, calculado
antes de parsear YAML. Espacios, comentarios, newlines, CRLF/LF, NFC/NFD y el
reordenamiento de claves cambian el hash — y un cambio de un byte deja la
ratificación sin coincidencia. La ratificación es una vez **por contenido**: la
misma ruta puede volver a coincidir más abajo en la historia si los bytes
regresan; retirar autoridad para siempre exige revocación explícita.

## El tope declara su clase (R-4)

Antes, si un recurso aceptaba tope se deducía de su **nombre** — y un nombre
en otro idioma se escapaba (`conexion` bloqueado, `connections` colando). Ahora
la regla **declara** la clase del recurso (`tope.resource_class`) con vocabulario
cerrado: `connections, concurrencia, workers, procesos_hijos, hilos, tasks,
llamadas_paralelas`. Las clases de agentes, enjambres y conexiones de D-4 no
existen en el vocabulario: un tope sobre ellas no se puede ni expresar. El
guardia semántico D-4 de `jax/faro/topes.py` sigue vigente en el runtime.

## Cómo se carga

`policy/rule_authority/snapshot.py::load_trusted_policy_snapshot(repo, pin)` —
nunca el árbol de trabajo: objetos Git del commit exacto del pin, con las
defensas de `jax/faro/git_objetos.py` (entorno limpio, hooks y fsmonitor
apagados, replace objects desactivado). Una regla inválida, un `rule_id`
duplicado o una entrada no canónica rechazan el snapshot **completo**.

Diseño completo: `docs/superpowers/specs/2026-10-05-faro-f1-1-rule-authority-kernel-design.md`.
`policy/**` se integra solo por Fernando.
