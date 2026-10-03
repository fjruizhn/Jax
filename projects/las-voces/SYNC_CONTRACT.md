# LAS VOCES — Sync Contract v0.1

## Canonical source
`projects/las-voces/`

**LV-003 documentation correction (2026-09-28):** the original LV-000 import
used `axioma/projects/las-voces/` as its intended repository-relative target.
The LV-000 commit `9a63721` created the actual canonical tree at
`projects/las-voces/`, and LV-002's `scripts/axioma_sync.py` reads that same
path. This corrects documentation drift only; it does not rewrite historical
events or alter canonical content.

Canónico:
- project.json
- PROJECT_CHARTER.md
- decisions/
- risks.json
- activity.ndjson
- agents/
- skills/

## Proyecciones
Codex:
- `projects/las-voces/AGENTS.md` (nested instructions discovered by Codex from
  the project working directory). This environment has no repository-local
  Codex skill path; no `.codex/skills/` projection is invented.

Claude Code:
- `projects/las-voces/CLAUDE.md`

Qwen Code:
- `projects/las-voces/QWEN.md`
- `projects/las-voces/.qwen/skills/`
- `projects/las-voces/.qwen/agents/`

## Regla
Canonical first. Las copias específicas de harness son generadas. `axioma sync las-voces` debe detectar drift por hash y negarse a sobrescribir silenciosamente modificaciones manuales.

## Reconciliation
`python3 scripts/axioma_sync.py las-voces --check` is non-mutating and fails
closed on stale or manually edited projections. `python3 scripts/axioma_sync.py
las-voces` is the explicit reconciliation mode; it stages all files, atomically
replaces them, restores prior files if a replacement fails, writes
`sync/manifest.json`, and verifies the result.
When a canonical skill id changes, reconciliation removes the old skill file
only if the previous manifest identifies it and its bytes still match the
recorded generated hash. An unlisted or manually edited skill is preserved and
the command fails closed; replacement and removal share one rollback path.
Before staging and immediately before each replacement/removal, the generator
rejects symlink components along every output path, including expected files.
The check also rejects symlink components in every canonical input, expected
output and manifest, even when the linked file has identical bytes.

## 2026-10-03 · revisión de la proyección Qwen (Jax#320)

**DECISIÓN — Fernando, comentario del PR #320:** el constructor Qwen declara
alcance completo de herramientas en `project.json` (`tools: ["*"]`,
`disallowedTools: []`); el generador falla si faltan esos campos. El `*` es el
comodín explícito que Qwen Code 0.24.7 expande al catálogo de herramientas
disponible, incluidos los MCP registrados. La proyección usa `approvalMode:
bubble`, que el parser de esa versión acepta.

**HECHO — Codex, bundle instalado de Qwen Code 0.24.7:**
`chunks/chunk-AN36BHDM.js:562` expande `tools: ["*"]`; `:949` acepta
`bubble`; `:954` lee `tools`, `disallowedTools` y `approvalMode`; `:780`
resuelve `bubble` según el modo de la sesión que invoca:

| Padre | Subagente con `bubble` |
| --- | --- |
| `default` | `default` |
| `auto-edit` | `auto_edit` |
| `auto` | `auto` |
| `yolo` | `yolo` |
| `plan` | `default` |

**DECISIÓN — Fernando, 2026-10-03, vía jax-14:** conservar `bubble` y declarar
la única excepción, `plan → default`. El objetivo es evitar una escalada de
autonomía sin humano; en `default`, cada edición requiere confirmación humana.
La prueba con el parser y resolvedor reales de Qwen Code 0.24.7 fija esta
tabla para detectar cualquier cambio de comportamiento tras actualizar Qwen.

**HECHO — Codex:** `--check` enumera los archivos de `.qwen/agents/` y
`.qwen/skills/`, compara el manifiesto completo y exige un SHA de commit real
y ancestro. La CI ejecuta el check sobre el checkout versionado y usa el parser
real de Qwen 0.24.7. Una edición manual de `approvalMode` a `yolo` produjo
rc=1 y `DRIFT DETECTED` localmente. Las pruebas adversariales cubren
separadores, controles, sustitutos sueltos y nombres de skill inseguros.

**HISTORIA — Codex, 2026-10-03:** la primera CI del cierre detectó que un
test nuevo combinaba `subprocess.run` y `CLAUDE.md`, y activó el escáner de
CLIs. Un primer helper `run_local(argv, **kwargs)` pasó el escáner, pero
jax-14 lo rechazó porque escondía futuros comandos arbitrarios. Se sustituyó
por funciones de propósito fijo con argumentos cerrados; una prueba comprueba
que la API del helper no vuelva a aceptar un argv libre. Lección: un verde por
separar nombres de archivos no vale si abre una ruta genérica alrededor del
freno.

**HISTORIA — auditoría independiente, 2026-10-03:** la primera prueba de
renombre cambiaba el id antes de generar por primera vez. El auditor reprodujo
el caso real, con una proyección anterior ya presente: quedaban dos skills
cargables. Se corrigió la transición y se añadieron pruebas para el renombre,
un archivo ajeno, modificaciones manuales, symlinks y fallo de eliminación.

**HISTORIA — auditoría independiente, 2026-10-03:** el primer cerco de
symlinks protegía solo el archivo obsoleto. El auditor demostró que un
directorio esperado con symlink podía sobrescribir un archivo externo antes de
que `--check` fallara. Se añadió un rechazo previo de todos los componentes
de cada destino esperado y una segunda comprobación justo antes de reemplazar.
Una sustitución concurrente del directorio entre esa comprobación y la llamada
al sistema sigue siendo un riesgo de carreras entre procesos del mismo usuario;
la marca de propiedad del worktree coordina las escrituras normales, pero no es
un bloqueo del sistema de archivos.

**HISTORIA — auditoría independiente, 2026-10-03:** el auditor mostró que
`--check` aún aceptaba un archivo generado o canónico enlazado a otro lugar
con bytes idénticos. Eso permitía que el filtro de rutas de CI no viera una
edición posterior del destino real. El check ahora rechaza cualquier componente
symlink de las entradas canónicas, proyecciones esperadas y manifiesto; las
pruebas fijan los tres casos.

**Alternativa descartada — Codex:** enumerar nombres de herramientas del bundle
congelaría el catálogo y omitiría herramientas MCP registradas después. El
comodín explícito conserva el alcance completo decidido por Fernando.

**Alternativas descartadas — Fernando, 2026-10-03:** fijar `plan` dejaría al
constructor sin poder construir; bloquear la invocación desde `plan` no es
posible desde esta proyección.

## Agent Bus
Toda conversación/handoff relevante entre agentes debe producir un envelope auditable:
message_id, project_id, task_id, sender_agent, recipient_agent, intent, evidence_refs, authority_context, correlation_id, created_at, status.

Ariadna consume eventos verificados y actualiza `project.json`; nunca inventa estado.
