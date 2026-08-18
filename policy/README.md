# jax/policy — Corpus normativo de JAX/Axioma

## Qué es esto

El soporte donde viven las reglas de comportamiento de JAX, escritas como
datos (YAML), no dispersas en prompts, `CLAUDE.md` y notas sueltas. Nace de
la Fase 0.5 de REFORMAS-v3.md (`/opt/jax/docs/FASE-0-VERIFICACION.md`,
hallazgo F0.3): "Los Seis Imposibles" no es un documento normativo, y las
normas de comportamiento reales nunca se habían escrito como normas.

Este directorio **no implementa ninguna reforma** de REFORMAS-v3.md. Es
exclusivamente el soporte — el lugar donde una regla puede vivir con id,
estado, origen y (cuando existe) mecanismo de cumplimiento y test. Ese es
el alcance completo de esta fase.

## Los cuatro estados posibles

Definidos con rigor, no por optimismo — ver `tools/validate_corpus.py` para
la aplicación mecánica de estas definiciones:

- **NORMATIVA** — existe test escrito HOY que falla cuando la regla se
  viola. Requiere `enforcement.test` no nulo.
- **NORMATIVA_PENDIENTE** — el test es escribible pero todavía no existe.
  Hay (o es razonable que haya) un mecanismo de código que podría
  implementarlo; nadie lo escribió aún.
- **CULTURAL** — no admite test automatizado. Es legítima — muchas de las
  reglas más importantes de este ecosistema lo son — pero no se disfraza de
  norma verificable.
- **HISTORICA** — existió, ya no rige, o nunca pasó de borrador/versión
  provisional sin ratificación confirmada.

## Cómo se agrega una regla

1. Un archivo YAML por regla en `rules/`, nombrado `{ID}-{slug}.yaml`
   (ver `rules/` para el esquema completo, documentado también en
   `/opt/jax/docs/FASE-0.5-CORPUS.md`).
2. El estado se asigna con el criterio de arriba — nunca por aspiración.
3. `enforcement.mechanism` y `enforcement.test` van `null` si no existen
   todavía. Ninguna regla se marca NORMATIVA sin un test real y existente.
4. Corré `tools/validate_corpus.py` — falla si falta un campo obligatorio,
   el `status` no es uno de los cuatro, `status=NORMATIVA` con `test=null`,
   o hay ids duplicados.
5. Corré `tools/corpus_hash.py` para actualizar el hash en `VERSION`.
6. Corré `tools/generate_corpus.py` para regenerar `generated/CORPUS.md`.

**Regla absoluta de origen:** ninguna regla se agrega por inferencia o
porque "tendría sentido que existiera". Solo se trasladan reglas que ya
existen por escrito en alguna fuente citable (`origin`). Si falta una, va a
una sección de PROPUESTAS en el informe de fase — nunca al corpus
directamente.

## `generated/` NUNCA se edita a mano

`generated/CORPUS.md` se regenera desde los YAML de `rules/` con
`tools/generate_corpus.py` — una plantilla fija, sin lógica interpretativa
(mismo principio que el renderer de REFORMAS-v3.md §3.1.6: el generador no
decide nada, solo vierte datos en slots). Editarlo a mano es un error que
**el validador no puede detectar** — el próximo `generate_corpus.py` lo
sobreescribe en silencio y el cambio manual se pierde sin aviso. Si algo del
corpus está mal, se corrige en el YAML de origen, nunca en `generated/`.

## Estructura

```
policy/
  VERSION              semver + sha256 del corpus (concatenación ordenada por id de rules/*.yaml)
  README.md            este archivo
  rules/                una regla por archivo, {ID}-{slug}.yaml
  vocabulary/
    closed_vocabulary.yaml   vocabulario cerrado real (§3.1.5) — poblado desde el sistema
    predicates.yaml          los ocho predicados cerrados (§3.1.3)
  templates/
    render_templates.yaml    plantillas de renderer por predicado (§3.1.6)
  generated/
    CORPUS.md            NUNCA editar a mano — ver arriba
  tools/
    corpus_hash.py        calcula el sha256 del corpus
    validate_corpus.py    valida el esquema de rules/*.yaml
    generate_corpus.py    genera generated/CORPUS.md
```
