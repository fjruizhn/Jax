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

## El tope declara su clase (R-4 — decisión de Fernando, 2026-10-06)

Antes, si un recurso aceptaba tope se deducía de su **nombre** — y un nombre
en otro idioma se escapaba (`conexion` bloqueado, `connections` colando). Ahora
la regla **declara** la clase del recurso (`tope.resource_class`) con vocabulario
cerrado, y ese vocabulario son **solo las clases de actos y dinero**:
`monto_dinero` (monto de dinero), `actos_externos` (cantidad de actos externos:
mensajes, compras, publicaciones), `frecuencia`, `duracion` y `tokens_costo`.

**NUNCA llevan tope** — no existen en el vocabulario y además su nombre se
rechaza aunque la clase declarada sea legítima: conexiones, concurrencia,
workers, hilos, procesos y agentes (D-4, en castellano y en inglés).

`jax/faro/topes.py` sigue con su guardia semántica D-4 para los recursos que
llegan por el canal de control; consumir la clase DECLARADA de la regla en vez
de deducir por nombre le corresponde al kernel de evaluación (paso 5-7 de
F1.1, rama `feat/faro-f1.1-kernel`): este schema ya no deja ninguna vía a un
tope prohibido.

## Cómo se carga

`policy/rule_authority/snapshot.py::load_trusted_policy_snapshot(repo, pin)` —
nunca el árbol de trabajo: objetos Git del commit exacto del pin, con las
defensas de `jax/faro/git_objetos.py` (entorno limpio, hooks y fsmonitor
apagados, replace objects desactivado). Una regla inválida, un `rule_id`
duplicado o una entrada no canónica rechazan el snapshot **completo**.

Diseño completo: `docs/superpowers/specs/2026-10-05-faro-f1-1-rule-authority-kernel-design.md`.
`policy/**` se integra solo por Fernando.
