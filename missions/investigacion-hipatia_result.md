# Resultado de: investigacion-hipatia.md

Todos adentro. Armando el reporte consolidado ahora.

---

# INFORME DE INVESTIGACIÓN — HIPATIA
## Misión: Sala de Control, Skills, Hubermech
**Fecha:** 14 junio 2026 | **Agentes:** 5 en paralelo | **Fuentes verificadas**

---

## HALLAZGO CRÍTICO #1 — Ya existe el estándar

El 18 de diciembre de 2025, Anthropic publicó `SKILL.md` como **open standard** bajo la Linux Foundation's Agentic AI Foundation:

- **Spec:** [agentskills.io](https://agentskills.io) / [github.com/agentskills/agentskills](https://github.com/agentskills/agentskills)
- **Adoptado por:** Claude Code, Gemini CLI, OpenAI Codex CLI, Microsoft Agent Framework, GitHub Copilot, Cursor, Windsurf, Amp, Goose y 16+ plataformas más
- **Un skill escrito hoy para Hyde funciona mañana en Thot y en Hipatia sin cambios**

Formato canónico:

```yaml
---
name: skill-identifier
description: >
  Use when the user asks to "X", "Y", or "Z". Include trigger phrases.
license: Apache-2.0
compatibility: Requires python3
allowed-tools: bash python
---

# Instrucciones en markdown para el agente
```

Ubicación:
```
.claude/skills/skill-name/     # proyecto (compartido por equipo)
~/.claude/skills/skill-name/   # usuario (personal, todos los proyectos)
```

**Progressive disclosure:** Solo `name` + `description` se cargan al arrancar la sesión (~100 tokens por skill). El cuerpo completo se carga on demand cuando el agente decide que el skill es relevante.

---

## HALLAZGO CRÍTICO #2 — OpenClaw es la referencia de arquitectura

No es un juego. Es un agente IA autónomo local con 379k estrellas en GitHub, MIT, TypeScript + Node 24, gobernado por la OpenClaw Foundation.

**Repo:** [github.com/openclaw/openclaw](https://github.com/openclaw/openclaw)

Su arquitectura ES lo que estamos construyendo — ya lo resolvieron:

| Componente JAX | Equivalente OpenClaw | Cómo lo resuelven |
|---|---|---|
| hall9000 API | Gateway WebSocket `:18789` | Daemon persistente, runs 24/7 |
| TUI sala de control | TUI cliente WebSocket | Thin client, el daemon hace el trabajo |
| Human gate | `require approval` hooks | Pausa ejecución, espera respuesta por chat |
| Kill switch | Task cancellation API | `sticky cancel semantics` |
| Skills por faceta | ClawHub marketplace | 5,700+ skills, tres niveles de precedencia |
| Audit | Task Flows con `revision history` | Estado persistente entre reinicios |

**Tres niveles de precedencia de skills (copiar exacto):**
```
workspace skills   (más específico, gana)
    ↑ override
~/.openclaw/skills/   (usuario)
    ↑ override
bundled skills    (menos específico, base)
```

**`agentgate` skill:** API gateway para datos personales con aprobación humana en escrituras. Ya lo construyeron.

---

## INVESTIGACIÓN 1 — Sala de Control TUI

### Stack recomendado: Textual (Python)

**Documentación oficial:** [textual.textualize.io](https://textual.textualize.io)

### Proyectos de referencia reales

| Proyecto | Stars | Relevancia |
|---|---|---|
| [hermes-hud](https://github.com/joeynyc/hermes-hud) | 847 | **"TUI consciousness monitor para actividad de agentes IA"** — el más relevante |
| [dolphie](https://github.com/charles-001/dolphie) | 1.2k | Dashboard de monitoreo MySQL en tiempo real, tabs, 1-second polling |
| [cronboard](https://github.com/antoniorodr/cronboard) | 1.4k | Gestión de tareas con pause/resume/delete + confirmaciones |
| [isd](https://github.com/kainctl/isd) | 2.1k | systemd TUI, stop/start con permisos elevados |
| [terraform-tui](https://github.com/idoavrah/terraform-tui) | 750 | Modales de confirmación para operaciones destructivas |
| [toolong](https://github.com/Textualize/toolong) | 3.9k | Log viewer en streaming — referencia canónica |

Lista completa: [github.com/matan-h/written-in-textual](https://github.com/matan-h/written-in-textual)

### Widgets clave y sus APIs

**Cola de tareas pendientes → `DataTable`**
```python
# Actualización in-place sin mover el cursor
table.update_cell(row_key, "status_col", Text("RUNNING", style="green bold"))
# cursor_type="row" permite seleccionar una tarea
```
Docs: [textual.textualize.io/widgets/data_table/](https://textual.textualize.io/widgets/data_table/)

**Audit timeline streaming → `RichLog`**
```python
rich_log.write(
    Text(f"[{timestamp}] ", style="dim") + Text(message, style=severity_style)
)
```
Docs: [textual.textualize.io/widgets/rich_log/](https://textual.textualize.io/widgets/rich_log/)

> **Distinción importante:** `RichLog` regresa al fondo en cada nuevo mensaje (bueno para audit live). `Log` mantiene la posición del usuario (bueno si el operador está revisando historia). Elegir según el panel.

**Kill switch siempre visible — dos mecanismos combinados:**
```python
class KillSwitchBar(Widget):
    DEFAULT_CSS = """
    KillSwitchBar {
        dock: bottom;    # NUNCA se scrollea fuera de vista
        height: 1;
        background: $error;
    }
    """

# En la clase App:
BINDINGS = [
    Binding("ctrl+k", "emergency_stop", "KILL ALL", priority=True)
    # priority=True dispara incluso dentro de modales
]
```
Docs: [textual.textualize.io/api/binding/](https://textual.textualize.io/api/binding/)

**Human gate — el patrón correcto:**
```python
class ApprovalScreen(ModalScreen[bool]):
    def compose(self):
        yield Grid(
            Label(f"Aprobar: {self.action_description}?"),
            Button("Aprobar", variant="success", id="approve"),
            Button("Denegar", variant="error", id="deny"),
        )

    def on_button_pressed(self, event):
        self.dismiss(event.button.id == "approve")

@work  # OBLIGATORIO — push_screen_wait solo funciona dentro de un worker
async def execute_with_approval(self, action):
    approved = await self.app.push_screen_wait(ApprovalScreen(action))
    if approved:
        await self.run_action(action)
```

`push_screen_wait()` bloquea el worker pero deja la app responsive. El input-blocking (que ningún otro keybinding dispare mientras el modal está activo) es automático con `ModalScreen`.

Docs: [textual.textualize.io/guide/screens/](https://textual.textualize.io/guide/screens/)

### Lecciones de lazygit, k9s, btop

**lazygit** ([github.com/jesseduffield/lazygit](https://github.com/jesseduffield/lazygit)):
- Sistema `NoPopupPanel` bloquea TODOS los keybindings cuando hay un modal — no solo visualmente, a nivel de input
- PR [#4284](https://github.com/jesseduffield/lazygit/pull/4284): bug donde se podía saltar de panel durante confirmación → fix importante

**k9s** ([github.com/derailed/k9s](https://github.com/derailed/k9s)):
- Dos niveles de "kill": `Ctrl+D` = delete con confirmación; `Ctrl+K` = kill inmediato sin diálogo
- Log viewer: buffer circular de 5000 líneas, flush cada 50ms, toggle de autoscroll con `s`
- Refs: [dialog/confirm.go](https://pkg.go.dev/github.com/derailed/k9s@v0.19.5/internal/ui/dialog)

**btop** ([github.com/aristocratos/btop](https://github.com/aristocratos/btop)):
- Anti-flicker: construye el string completo con `out.reserve(width * height)` ANTES de escribir al terminal
- ANSI escapes para posicionamiento de cursor: actualiza solo lo que cambió, sin borrar pantalla
- Thread model: main thread para input, `Runner` thread para datos y rendering

---

## INVESTIGACIÓN 3 — Skills Persistentes (resumen ejecutivo)

| Framework | Formato | File-based | Adoptó AgentSkills standard |
|---|---|---|---|
| Claude Code | `SKILL.md` | Sí | Sí (lo creó) |
| Gemini CLI | `SKILL.md` | Sí | Sí |
| OpenAI Codex CLI | `SKILL.md` (en plugins) + `plugin.json` | Sí | Sí |
| Microsoft Agent Framework | `SKILL.md` | Sí | Sí (mayo 2026) |
| LangChain | Decoradores Python | No | No |
| CrewAI | Decoradores Python | No | No |
| AutoGPT | Módulos Python (deprecated) | No | No |

**GPT Actions** usan OpenAPI 3.1 YAML — formato diferente, caso de uso diferente (no son skills persistentes en el mismo sentido).

---

## INVESTIGACIÓN 4 — Ecosistemas de Plugins

### Mapa de madurez

| Motor | Sistema | Tamaño | Veredito |
|---|---|---|---|
| Claude Code | MCP + Plugin layer | 20,000+ MCP servers | **Muy maduro — beber de ahí** |
| Gemini CLI | Extensions (`gemini-extension.json`) | 1,052 extensiones | Creciendo, en transición a "Antigravity CLI" |
| OpenAI Codex | Plugin system (`/plugins`) | 100+ plugins | Temprano |
| Ollama | **Ninguno** | 0 | **Construir el puente nosotros** |

### Claude Code / MCP — el estándar ganador

**Directorios de MCP servers:**

| Directorio | URL | Cantidad |
|---|---|---|
| Glama | [glama.ai/mcp/servers](https://glama.ai/mcp/servers) | 21,000+ |
| mcp.so | [mcp.so](https://mcp.so) | 19,700+ |
| PulseMCP | [pulsemcp.com](https://pulsemcp.com) | 11,840+ |
| Smithery | [smithery.ai](https://smithery.ai) | 7,300+ |
| Official Registry | [registry.modelcontextprotocol.io](https://registry.modelcontextprotocol.io) | ~2,000 curados |
| Anthropic Directory | [claude.ai/directory](https://claude.ai/directory) | 439 verificados |
| Awesome MCP | [github.com/punkpeye/awesome-mcp-servers](https://github.com/punkpeye/awesome-mcp-servers) | 89k stars |

**Cómo agregar un MCP server:**
```bash
# Remote HTTP (recomendado)
claude mcp add --transport http notion https://mcp.notion.com/mcp

# Local stdio
claude mcp add --env API_KEY=xxx --transport stdio nombre -- npx -y package-name

# Scopes: local (default), project (.mcp.json), user
```

**`.mcp.json` en el repo (scope proyecto, se commitea):**
```json
{
  "mcpServers": {
    "shared-server": {
      "command": "/path/to/server",
      "args": [],
      "env": {}
    }
  }
}
```

### Gemini CLI — real pero en transición

- 105,000 estrellas en GitHub
- Formato: `gemini-extension.json` que internamente envuelve MCP servers
- Transición a "Antigravity CLI" en junio 2026 — incertidumbre de marca, pero el código open source persiste
- Ref: [The Register](https://www.theregister.com/ai-ml/2026/05/20/bye-bye-gemini-cli-google-nudges-devs-toward-antigravity/5243605)

### Ollama — el hueco real

- Tool calling: JSON Schema por request, sin registry
- No hay comunidad de tools, solo comunidad de modelos
- Modelos con soporte de tools: `llama3.1:8b`, `Llama3-Groq-Tool-Use`, `FunctionGemma`
- Limitación práctica: degradación por encima de 6-8 tools simultáneos, 78-99% de adherencia al schema vs ~100% de Claude/GPT-4
- Ref: [deepwiki.com/ollama/ollama/7.2-tool-calling-and-function-execution](https://deepwiki.com/ollama/ollama/7.2-tool-calling-and-function-execution)

**Insight de arquitectura:** Claude Code plugins, Gemini extensions, y Codex plugins todos convergen internamente en envolver MCP servers. Apuntar a MCP = llegar a los tres en un tiro. Ollama es el único que necesita un puente propio.

---

## INVESTIGACIÓN 5 — Organización Limpia

### Microsoft APM — el patrón más completo

**Docs:** [microsoft.github.io/apm](https://microsoft.github.io/apm/) | **Repo:** [github.com/microsoft/apm](https://github.com/microsoft/apm)

```yaml
# apm.yml — el manifiesto
name: jax-hyde-toolkit
version: 1.0.0
dependencies:
  apm:
    - github/awesome-copilot/plugins/azure-cloud-development#main
    - ./packages/mis-skills-propios        # ruta local
    - git: https://gitlab.com/org/skills.git
      path: packages/security
      ref: v2.0
  mcp:
    - io.github.github/github-mcp-server
targets:
  - claude
  - codex
```

```yaml
# apm.lock.yaml — el lockfile (commitear al repo)
dependencies:
  - name: plugin-name
    commit: abc1234567890...40chars   # SHA completo, byte-for-byte reproducible
```

**Un solo `apm install --mcp` conecta el MCP server a Claude, Codex, Cursor, Gemini y OpenCode simultáneamente.**

### skills-supply — alternativa en TOML

**Repo:** [github.com/803/skills-supply](https://github.com/803/skills-supply)

```toml
[agents]
claude-code = true
codex = true
gemini-cli = false    # desactivado para esta faceta

[dependencies]
superpowers = { gh = "superpowers-marketplace/superpowers" }
mis-skills  = { path = "../mis-skills-propios" }
infra-tools = { git = "git@gitlab.com:org/skills.git", rev = "abc123" }
```

Usa prefix namespacing: `superpowers-debugging`, `superpowers-code-review` — nunca colisiones.

### Tabla de formatos comparados

| Herramienta | Manifiesto | Lockfile | Granularidad del pin |
|---|---|---|---|
| **APM** | `apm.yml` | `apm.lock.yaml` | SHA de 40 chars |
| skills-supply | `agents.toml` | `.sk-state.json` | git rev o tag |
| lazy.nvim | Lua specs | `lazy-lock.json` | commit hash |
| uv (Python) | `pyproject.toml` | `uv.lock` | versión exacta + hash |

### Seguridad — el problema real

**341 skills maliciosos encontrados en febrero 2026.** El vector de ataque no son binarios — es prompt injection dentro del `SKILL.md`. Un skill que parece benign puede redirigir el comportamiento del agente, exfiltrar datos, o cambiar permisos de tools.

**Mitigación:**
- Pinear siempre a commit SHAs, nunca a `main`
- **SkillGate** en CI: [github.com/charliechenye/SkillGate](https://github.com/charliechenye/SkillGate) — escaneo estático de capabilities (shell exec, network egress, acceso a secretos, filesystem writes, prompt override language) antes de instalar

### Anti-patrones documentados

1. **WordPress/Composer**: dos plugins con versiones distintas de la misma librería → `Cannot redeclare`. Solución: namespace isolation con php-scoper. Lección: necesitan resolver compartido o namespacing de dependencias.

2. **AutoGPT**: deprecó su sistema de plugins completo porque "old-style plugins no longer work with recent versions". Sin lockfile, sin matriz de compatibilidad, el sistema se pudrió. Lección: lockfile desde el día uno.

3. **npm sin lockfile**: sin `package-lock.json`, `npm install` produce distinto árbol en cada máquina. Promedio: 4.3 horas/semana perdidas en conflictos de dependencias.

---

## DECISIONES DE ARQUITECTURA — Lo que los datos dicen

### Qué adoptar directamente (no reinventar)

| Decisión | Adoptar |
|---|---|
| Formato de skills | `SKILL.md` — open standard, 16+ plataformas |
| Formato de manifiesto de dependencias | `apm.yml` + `apm.lock.yaml` (Microsoft APM) |
| Capa de integración de tools | MCP — llega a Claude, Gemini, Codex en un tiro |
| Seguridad de skills | SkillGate en CI, SHAs no branches |
| Namespacing | Prefix por fuente: `hyde-`, `thot-`, `hipatia-` |
| Referencia de arquitectura TUI | hermes-hud + OpenClaw TUI |
| Human gate en Textual | `push_screen_wait()` dentro de `@work` |
| Kill switch | `dock: bottom` + `Binding(..., priority=True)` en App |

### Qué necesitamos construir nosotros

1. **Facet manifest**: combinar APM-style dependency declaration con la topología del equipo JAX — qué faceta recibe qué plugins, cómo comparten tools entre facetas, reglas de resolución de conflictos entre facetas. APM existe pero no modela equipos multi-faceta.

2. **Tool/function registry para Ollama**: cero ecosistema existe. Si JAX local usa Ollama, hay que construir el bridge desde cero (o envolver MCP servers hacia Ollama's JSON Schema API).

3. **`agents.toml` del equipo JAX**: usando el modelo de skills-supply, definir qué skills activa cada faceta, con lockfile por faceta.

---

*Investigación compilada por Hyde (faceta técnica de JAX) a partir de 5 agentes de investigación paralelos. Todas las fuentes verificadas en búsqueda web activa, 14 junio 2026.*

---

Eso es todo lo que los agentes trajeron, Fernando. ¿Querés que guarde el reporte en memoria del proyecto, o arrancamos directamente a diseñar la sala de control con estos datos como base?
