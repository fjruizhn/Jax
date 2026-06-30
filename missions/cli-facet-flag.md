# Misión Hyde — Agregar bandera `--facet` al CLI (`jax --task`)

> Ejecutar con: `jax --task ~/jax/missions/cli-facet-flag.md`
> Hipatia inicia con `/using-superpowers` y `/ruflo`.

---

## 1. Objetivo

Agregar una bandera opcional `--facet <nombre>` a la CLI de `jax`, que **fuerza**
la faceta en modo `--task`, **saltando el router**. Causa raíz: el router enruta
por contenido, y un `.md` cuyo texto habla mucho de un dominio (ej. un documento
que menciona "auditoría", "riesgo", "adversarial") es mal enrutado (→ thot) cuando
en realidad la tarea es para otra faceta (→ ada). Forzar faceta resuelve esto de
raíz y es reutilizable.

Comportamiento deseado:
- `jax --task archivo.md --facet ada` → ejecuta la tarea con Ada, sin pasar por el router.
- `jax --task archivo.md` (sin `--facet`) → comportamiento actual (router decide).
- `--facet <inexistente>` → error claro listando las facetas válidas, sin ejecutar.
- `--facet` sin `--task` → error claro (la bandera solo aplica en modo tarea).

## 2. Principios (HYDE activo)
- No suponer; leer antes de editar. Backup antes de modificar.
- Ningún cambio sin prueba (gate §6). Declarar incertidumbres.

## 3. Reconocimiento (read-only) — NO EDITAR TODAVÍA
```bash
# a) Cómo se define el parser de argumentos hoy
grep -n "argparse\|add_argument\|ArgumentParser\|parse_args\|args\.task\|--task" ~/jax/jax/core/main.py

# b) Cómo se elige la faceta en modo --task hoy (dónde llama al router)
grep -n "task\|router\|route\|_keyword_route\|_classify\|faceta\|facet\|elegida por router" ~/jax/jax/core/main.py | head -40

# c) La lista de facetas válidas (para validar el input de --facet)
grep -n "VALID_FACETAS\|AUTO_FACETAS\|LABELS\|ALIASES" ~/jax/jax/core/router.py

# d) Cómo se invoca la faceta una vez elegida (la firma a respetar)
grep -n "def main\|muscle.invoke\|invoke(\|model_override\|history_for_invocation" ~/jax/jax/core/main.py
```
**Reportá** lo encontrado antes de editar. En especial: (1) el nombre exacto de la
variable de args de la tarea, (2) la línea donde, en modo `--task`, se decide la
faceta vía router, (3) el nombre exacto de la lista de facetas válidas y dónde
importarla.

## 4. Cambios — `~/jax/jax/core/main.py`
- Backup `main.py.backup-facetflag-$TS`.

**4a. Agregar el argumento al parser.** Junto a la definición de `--task`:
```python
parser.add_argument(
    "--facet",
    metavar="NOMBRE",
    default=None,
    help="Fuerza la faceta en modo --task, saltando el router "
         "(ej: --facet ada). Sin esta bandera, el router decide.",
)
```

**4b. Validación temprana** (después de `parse_args()`, antes de ejecutar):
```python
# --facet solo aplica en modo --task
if args.facet and not args.task:
    parser.error("--facet solo puede usarse junto con --task")

# Validar que la faceta exista (importar la lista canónica del router)
if args.facet:
    from jax.core.router import VALID_FACETAS   # ajustar al nombre real (§3c)
    if args.facet not in VALID_FACETAS:
        parser.error(
            f"faceta '{args.facet}' no válida. "
            f"Válidas: {', '.join(sorted(VALID_FACETAS))}"
        )
```
> NOTA: usar `VALID_FACETAS` (incluye hyde) y NO `AUTO_FACETAS`, porque forzar es
> invocación explícita — debe poder forzarse cualquier faceta, incluida hyde.

**4c. Saltar el router en modo tarea cuando hay `--facet`.** En el bloque que hoy
elige la faceta para `--task` (identificado en §3b), envolver la llamada al router:
```python
# ANTES (conceptual): faceta = router.route(task_text)  / _keyword_route(...) etc.
# DESPUÉS:
if args.facet:
    faceta = args.facet
    print(f"[tarea] Faceta: {faceta} (FORZADA por --facet, router omitido)")
else:
    faceta = <llamada-al-router-existente>   # sin cambios respecto a hoy
    print(f"[tarea] Faceta: {faceta} (elegida por router)")
```
> Respetar la firma y el flujo existentes: lo único que cambia es **de dónde sale
> `faceta`**. El resto (invoke, history, model_override, guardado de `_result.md`)
> queda igual. Si hoy el print de "elegida por router" ya existe, reusar su formato.

## 5. Incertidumbres a declarar (no suponer)
- Si el parser NO es argparse (otra lib), adaptar 4a/4b a esa lib y reportarlo.
- Si la faceta en `--task` no se elige con una sola llamada sino disperso, mapear
  el punto exacto y reportar antes de editar.
- Si `VALID_FACETAS` no es importable sin ciclo de imports, declarar y proponer
  alternativa (ej. lista local mínima) en el reporte.

## 6. Gate de prueba (obligatorio)
```bash
# 1) Forzar Ada en un archivo cuyo CONTENIDO enrutaría a otra faceta:
cat > ~/facet-gate.md <<'EOF'
ada: auditoría adversarial de riesgo, vulnerabilidades, amenazas, deny, ataque.
Confirmá en una línea que te procesó la faceta Ada, citando: GATE-OK-ADA.
EOF
jax --task ~/facet-gate.md --facet ada
cat ~/facet-gate_result.md
#   ESPERADO: línea "[tarea] Faceta: ada (FORZADA por --facet, router omitido)"
#             y el _result.md responde con identidad de Ada + "GATE-OK-ADA".

# 2) Sin --facet, el mismo archivo va por router (debería caer en thot por contenido):
jax --task ~/facet-gate.md
#   ESPERADO: "[tarea] Faceta: thot (elegida por router)" — confirma que el router
#             sigue intacto y que --facet es lo único que lo override-a.

# 3) Faceta inválida → error claro, NO ejecuta:
jax --task ~/facet-gate.md --facet nopexiste
#   ESPERADO: error listando las facetas válidas. Sin _result.md nuevo.

# 4) --facet sin --task → error claro:
jax --facet ada
#   ESPERADO: error "--facet solo puede usarse junto con --task".
```

**Criterio de aceptación:** los 4 casos se comportan como el ESPERADO. En especial
el caso 1 (fuerza Ada pese al contenido) y el caso 2 (router intacto sin la bandera).

## 7. Reporte final
Archivo tocado + diff, resultado de los 4 casos del gate, incertidumbres declaradas,
rollback (`main.py.backup-facetflag-$TS`).
