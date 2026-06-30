faceta: hyde

# Jacobs Fase B — Planificación modular (patrón LLM-as-Compiler)

> Editar el Jacobs CANÓNICO `~/jax/jacobs/`. HYDE: backup, gate, rollback si falla.
> Hipatia inicia con `/using-superpowers` y `/ruflo`.

## Objetivo
Hoy Ada descompone con el prompt genérico de v0.2 → listas planas. Esta fase le enseña a
planificar como COMPILADOR: tipos comunes primero, dependencias declaradas, orden por
dependencia, validación y ensamble al final. Patrón confirmado por revisión externa
(GPT + DeepSeek convergieron): manifest → common_types → módulos en orden de dependencia
(cada uno importa los previos) → validación de consistencia → ensamble.

DOS cambios:
- **B1:** agregar `depends_on` al modelo `Step` (dato estructural, no texto).
- **B2:** reescribir el prompt de planificación de Ada para producir el patrón modular.

NO tocar el executor todavía (pasar el contexto completo según depends_on es Fase C).

---

## B1 — `~/jax/jacobs/models.py`: campo `depends_on` en Step
Backup: `models.py.backup-faseB-$(date +%Y%m%d-%H%M%S)`

En la clase `Step` (línea ~43), agregar un campo opcional. Referencia por step_index (int),
más estable que step_id (UUID). Insertar junto a los otros campos:
```python
    depends_on:       list[int] = Field(default_factory=list)  # step_index de dependencias
```
> Default lista vacía → planes existentes y triviales no se afectan (sin dependencias).
> Verificar que el `_from_spec` de plan.py propague este campo si viene en el spec
> (ver B2.3). Verificar que store.py serialice el Step completo (usa model_dump() → lo incluye solo).

## B2 — `~/jax/jacobs/plan.py`: prompt modular para Ada
Backup: `plan.py.backup-faseB-$(date +%Y%m%d-%H%M%S)`

### B2.1 — Nuevo system prompt de planificación modular
Reescribir `_PLAN_SYSTEM` (o crear `_PLAN_SYSTEM_MODULAR` usado solo en la ruta Ada/formal)
para instruir el patrón compilador. Contenido (adaptar redacción, preservar el principio
de evidencia que ya existe):
```
Eres Jacobs, el Director, planificando trabajo FORMAL COMPLEJO. Generás un plan de
ejecución como JSON (array de objetos), SOLO JSON, sin markdown ni explicaciones.

Patrón OBLIGATORIO para trabajo formal (compilador de especificaciones):
1. El PRIMER step SIEMPRE produce "common_types": define UNA vez todos los tipos, enums
   e identificadores compartidos. Todos los demás módulos los referencian, ninguno los redefine.
2. Luego los módulos en ORDEN DE DEPENDENCIA: cada módulo declara de qué steps anteriores
   depende (campo depends_on: lista de step_index). Un módulo va DESPUÉS de aquellos que necesita.
3. Las piezas que referencian a todo (invariantes, validaciones globales) van AL FINAL.
4. El PENÚLTIMO step es "validación de consistencia": revisa que no haya nombres huérfanos,
   tipos no definidos ni referencias rotas entre los módulos previos. Devuelve solo discrepancias.
5. El ÚLTIMO step es "ensamble": integra los módulos en el documento/paquete final.

Cada step: {"facet","capability","prompt","depends_on":[indices]}.
- facet para diseño formal/tipos/arquitectura: "ada". Para crítica/auditoría: "thot".
  Para investigación: "hipatia". Para código: "kimi".
- depends_on lista los step_index (0-based) de los steps cuyos OUTPUTS este step necesita.
- El prompt de cada step debe ser autocontenido y referir explícitamente a sus dependencias
  ("usando los tipos comunes del step 0 y las capabilities del step 1, definí...").

PRINCIPIO DE EVIDENCIA (innegociable): no asumas hechos no verificados; si un dato es
incógnita, incluí un step que lo verifique. "El que supone se equivoca."

Salida: SOLO el array JSON.
```

### B2.2 — Actualizar el user-prompt de `_ada_plan`
El prompt que arma `_ada_plan` debe pedir explícitamente el patrón modular y dar un ejemplo
con depends_on. Ejemplo a incluir en el prompt:
```
Ejemplo de forma esperada (no de contenido):
[
  {"facet":"ada","capability":"design","prompt":"Definí los tipos comunes: enums, identificadores, estructuras base compartidas.","depends_on":[]},
  {"facet":"ada","capability":"design","prompt":"Usando los tipos del step 0, definí el módulo de capabilities.","depends_on":[0]},
  {"facet":"ada","capability":"design","prompt":"Usando tipos (0) y capabilities (1), definí las invariantes.","depends_on":[0,1]},
  {"facet":"thot","capability":"critique","prompt":"Validá consistencia: nombres huérfanos, tipos no definidos, referencias rotas entre steps 0-2.","depends_on":[0,1,2]},
  {"facet":"ada","capability":"design","prompt":"Ensamblá los módulos 0-2 en el documento final, aplicando las correcciones del step 3.","depends_on":[0,1,2,3]}
]
```

### B2.3 — Propagar `depends_on` en el parser
En `_parse_plan_json` (donde arma cada dict válido) y en `_from_spec` (donde crea Step),
incluir `depends_on`:
```python
# en _parse_plan_json, dentro del dict válido:
"depends_on": [int(x) for x in item.get("depends_on", []) if isinstance(x, (int, str)) and str(x).isdigit()],
# en _from_spec, al crear Step:
depends_on=spec.get("depends_on", []),
```
> Robustez: si Ada devuelve depends_on con índices fuera de rango, filtrarlos (un índice
> >= step actual o < 0 se descarta). Declarar si se implementó ese filtro.

## Gate (obligatorio)
```bash
sudo systemctl restart jax-las-manos && sleep 8 && systemctl is-active jax-las-manos

# Objetivo formal → el plan debe tener forma MODULAR
curl -s http://127.0.0.1:7777/jacobs/plan -X POST -H "Content-Type: application/json" \
  -d '{"objective":"Genera el paquete modular del contrato de capabilities: tipos comunes, registro de capabilities, invariantes formales y validación de consistencia entre módulos","invoked_by":"fernando","mode":"dry_run"}' \
  | python3 -m json.tool
```
**Verificar en el plan devuelto (criterio de aceptación):**
- El step 0 produce los TIPOS COMUNES (common_types).
- Al menos algunos steps tienen `depends_on` POBLADO (no todos vacíos).
- Hay un step de VALIDACIÓN de consistencia cerca del final.
- Hay un step de ENSAMBLE al final.
- El orden respeta dependencias (un step no depende de uno posterior).
- Servicio 'active'. Objetivo trivial sigue yendo a qwen sin depends_on (no se rompió).

> Si Ada NO respeta el patrón (ej. todos los depends_on vacíos, o sin common_types primero):
> NO dar por bueno. Reportar el plan crudo que devolvió y ajustar el prompt. Es esperable
> necesitar 1-2 iteraciones de prompt hasta que el modelo siga la estructura.

## Reporte final
Diff de models.py (depends_on), diff de plan.py (system prompt + user prompt + parser),
el PLAN COMPLETO que generó Ada para el objetivo formal (pegarlo), análisis de si cumplió
el patrón (common_types primero / depends_on poblado / validación / ensamble),
incertidumbres, rollback (*.backup-faseB-*).
