# El Ejecutor — Qwen con manos (diseño, spec 1 de 2)

**Fecha:** 2026-09-15 · **Aprobado por:** Fernando (diseño en chat, secciones 1–4, 2026-09-15) ·
**Redactó:** Mr. Hyde · **Repos:** `jax` (arnés, jaula, Jacobs, LAS MANOS) y `jax-platform`
(modo Ejecutor, `/api/command`, migraciones, i18n).
**Spec 2 (después):** gestión de proyectos. Este spec deja al Ejecutor listo para vivir dentro de un
proyecto (`project_id` en toda misión), pero no diseña la gestión.

## 0. Historial de este documento

| Versión | Commit | Qué cambió |
|---|---|---|
| v1 | (este commit) | Diseño original, aprobado por secciones en chat. |

## 1. Por qué — la razón de ser de Axioma

> «Quiero que entiendas que la razón de ser de Axioma era esto.» — Fernando, 2026-09-15.
> «Yo no quiero una IA para platicar, quiero que Axioma se encargue de administrar mis servidores,
> VMs, webs, programación.»

La visión: **Qwen (local, sin costo por token) hace la mayor carga del trabajo**. Las facetas de
frontera hacen esquema, bosquejo, planificación y **auditoría**. Kimi K3 y GLM 5.3 ejecutan lo más
complejo. Qwen «aprende» por la memoria, igual que aprende Mr. Hyde entre sesiones: no por
reentrenamiento, sino porque cada sesión arranca con la constitución, la Biblioteca y las lecciones.

**Lo que hay hoy contradice la visión — medido 2026-09-15, no supuesto:**

| Hecho | Evidencia |
|---|---|
| El rol de `jax_local` (Qwen) es «conversación cotidiana»; el de Hyde (Claude) es «ejecución completa» | `CONTEXT.md` §3; `AXIOMA-ECOSISTEMA-TECNICO.md` §4.3 |
| El modo Comando de la Mesa ejecuta **siempre Hyde**: `faceta: hyde` escrito a mano | `jax-platform/backend/api/command.py` (`mission_file.write_text(f"---\nfaceta: hyde\n---…")`); placeholder i18n «Describe la tarea para Hyde… (autónomo)» |
| LAS MANOS ya da tool-calling a Qwen, pero solo `read_file`/`write_file` en el workspace | `las_manos/motor_registry/tool_authority.py` (`EXECUTABLE_TOOLS`), `worker.py` (transporte `ollama`) |
| El hilo de JAX guarda 10 turnos en RAM: no hay sesión persistente con contexto | `CONTEXT.md` §5 (`MAX_TURNS=10`) |
| Ollama **0.31.1** habla la API de Anthropic (`/v1/messages`, desde 0.14) | `ollama --version` en hall9000; docs.ollama.com/api/anthropic-compatibility |
| `qwen3.6:35b-a3b-q4_K_M` devolvió un `tool_use` correcto (`df -h`) por esa API en **1,6 s** | `curl` a `localhost:11434/v1/messages` con una herramienta ficticia, 2026-09-15 |
| GPU: Radeon AI PRO R9700, 32 GB; Qwen ocupa 23 GB con `num_ctx=32768` | `rocm-smi`, `ollama ps` |
| Moonshot y Z.ai documentan endpoint compatible con Anthropic (`api.moonshot.ai/anthropic`, `api.z.ai/api/anthropic`) | documentación pública — **no probado con nuestras llaves** (Fase 0) |

## 2. Decisiones de Fernando que gobiernan este diseño (2026-09-15)

| # | Decisión | Nota |
|---|---|---|
| D1 | **Destino = arnés propio de Axioma** con skills y agentes **nuestros** (camino B). **Ahora = camino A**: Claude Code como arnés, cerebro local. B se construye después, pieza a pieza | «Opción 3: A ahora, B después» |
| D2 | **Cerebro elegible**: Qwen por defecto; Kimi K3 y GLM 5.3 para lo complejo | Kimi/GLM no quedan fuera |
| D3 | **Autonomía = «autónomo con auditoría»**. El GO de Fernando es **por misión**, no por paso | Elegida con el riesgo escrito en la opción |
| D4 | **ENMIENDA — autonomía total.** Para las misiones del Ejecutor se **revocan**: (a) T6 de GAP2 Fase 4 (2026-08-19: DNS/mail/fuera del jail mantienen gate humano); (b) el nivel 🔴 de `docs/AUTONOMIA_ANTIERROR.md` §6.1 (VM de producción, mail, DNS interrumpen + Telegram) | Se le presentó la síntesis «rojo pre-aprobado por misión» y la descartó. **Fuera de las misiones del Ejecutor, T6 y AUTONOMIA siguen vigentes** |
| D5 | **Identidad = usuario propio `axioma`** con llave SSH exclusiva y sudo en cada servidor | Crear el usuario = GO una vez por máquina |
| D6 | **Alcance = todo el ecosistema**: `.20`, `.10`, `.11`, hall9000, y toda máquina nueva que se registre | hall9000 incluido por decisión de Fernando (Hyde recomendó dejarlo para después) |
| D7 | **Sésamo = «todo menos destruir»**, solo por API | §7 |
| D8 | **Lanzamiento**: la pestaña **Comando pasa a llamarse «Ejecutor»**; el Ejecutor también es un **paso de Pipeline** (Jacobs) | |
| D9 | **Toda misión lleva `project_id`** desde el día 1 | Para el spec 2 |
| D10 | **El Ejecutor sigue TODA la constitución**: no inventar, no dejar nada para después, sin hardcoding, etc. — y el auditor lo hace cumplir | §6.3 |
| D11 | **Criterio de la Fase 0**: Qwen es cerebro por defecto si completa **≥ 8 de 10** tareas y **no inventa ningún hecho** | Fijado antes de medir |
| D12 | **Prueba de fuego = cutover de `.20` → prod** (ficha #16 del tablero). **El cutover espera** a que el Ejecutor pueda hacerlo | Riesgo exim4 de `.20` extendido, aceptado |

## 3. Arquitectura (camino A)

### 3.1 Componentes

- **Faceta `ejecutor`** — fila en `facet` y `facet_binding` (el cerebro se elige con el flujo de
  aprobación que ya existe: `model_binding_proposal` → aprobación de superadmin). Nunca un modelo
  escrito en código.
- **Transporte `harness`** (nuevo): lanza Claude Code (`claude -p`) como arnés dentro de la jaula,
  con `ANTHROPIC_BASE_URL` apuntando al proveedor del cerebro resuelto:
  Ollama local (Qwen) · Moonshot (Kimi) · Z.ai (GLM). La URL sale de `provider` (DB) y la llave de
  `credential_resolver` (Ollama: sin llave). **El Ejecutor no usa Anthropic ni la cuenta de Claude de
  Fernando.**
- **Salida `--output-format stream-json`**: cada paso es un evento. Hoy Hyde usa `text` y por eso no
  tiene `resolved_version` ni tokens (`CONTEXT.md`, 2026-08-09); el Ejecutor sí los tendrá.

### 3.2 La jaula — perfil propio en `hyde_sandbox.py`

Mismo módulo, mismo punto único de entrada (`run_sandboxed_claude()`; el job de CI
`no-naked-claude-subprocess` sigue fallando ante cualquier otro call site). Perfil `ejecutor`:

| Monta | Modo | Por qué |
|---|---|---|
| Copia **fijada** de nuestros plugins/skills (`--plugin-dir`, versión pineada) | ro | Hereda superpowers, etc., sin el `~/.claude` real de Fernando |
| `CLAUDE.md` **generado** para la sesión | ro | §6 |
| Llave SSH de `axioma` | ro | §3.4 |
| Carpeta de sesiones **persistente por proyecto** | rw | §5 |
| Workspace | rw | igual que Hyde |
| Credenciales de Anthropic | **no se montan** | el cerebro no es Anthropic |

`--clearenv` + `--setenv` puntual (`HOME`, `ANTHROPIC_BASE_URL`, `ANTHROPIC_AUTH_TOKEN` del cerebro).
La llave del cerebro queda visible para el propio Ejecutor: es la suya, acotada a su proveedor.

### 3.3 Caminos de despacho — enumerados por transporte, no por los archivos que uno recuerda

Lección de `CONTEXT.md` §7 (la ronda de `_HTTP_FACETS` creyó que eran dos y eran tres).

| Camino | Hoy | Con el Ejecutor |
|---|---|---|
| Modo Comando/Ejecutor de la Mesa → `POST /api/command` | archivo de misión con `faceta: hyde` fijo | la faceta sale de la DB (clave en `axioma_config`, tabla que ya existe en `jax_memory`); lleva `project_id` |
| Paso de Pipeline → `jacobs/executor.py` | `_HTTP_FACETS` / `_MOTOR_FACETS` / `_invoke_hyde` | nuevo despacho por transporte `harness` |
| Chat de la Mesa (`api/chat.py`) | intercepta `hyde` por nombre y responde enlatado | intercepta `ejecutor` igual: el Ejecutor no conversa por el chat, trabaja por su modo |
| REPL (`jax/core/main.py`) | músculos propios | **fuera de alcance** (declarado): tráfico del REPL medido en cero los últimos 14 días al 2026-09-03 |

### 3.4 Acceso a las máquinas

- Inventario de máquinas en la DB (tabla nueva `ejecutor_host`: nombre, IP, puerto, rol, `sudo`,
  `api_only`), nunca en código. Registrar una máquina = crear el usuario `axioma` + fila (GO una vez).
- SSH como `axioma` con sudo a `.20`, `.10`, `.11` y hall9000.
- **Trampa conocida, contrato del inventario:** `179.49.115.33:58291` va a **prod**, no al bridge.
  Antes de operar sobre una IP con NAT, el Ejecutor verifica `machine-id` + `ip -4 addr`, nunca el
  hostname.
- Sésamo: solo API (§7).

## 4. Los contratos — antes de habilitar la capacidad (Principio IX)

Cada contrato existe **y se vio fallar** (Principio VII; lección de `CONTEXT.md` «un guard que nunca se
probó con un caso que DEBERÍA bloquear no es un guard») antes de que el Ejecutor toque un servidor.

| # | Contrato | Mecanismo | Prueba (el caso que debe bloquear) |
|---|---|---|---|
| C1 | **Prohibiciones duras** | Lista en DB; hook `PreToolUse` del propio Ejecutor; **fail-closed**: sin lista legible, bloquea todo | cada prohibido se intenta y sale bloqueado; lista ilegible → todo bloqueado |
| C2 | **Respaldo antes de destruir** | paso destructivo (borrar datos, `DROP`, DNS/MX, parar servicio, quitar paquetes) exige un punto de restauración **verificado** de ese servidor; si no, el hook bloquea | destructivo sin respaldo → bloqueado |
| C3 | **Registro intocable** | cada tool call del Ejecutor → API de LAS MANOS → `audit.jsonl` (fuera de la jaula); más log de `sudo` de `axioma` en cada servidor | desde la jaula, escribir el registro falla |
| C4 | **Kill switch en vuelo** | el Ejecutor corre en su propio scope de cgroup; `/etc/jax/PAUSE` (watcher de 250 ms que ya existe) mata la jaula entera y los procesos remotos (`ssh -tt` → SIGHUP) | comando remoto largo + PAUSE → muerto, **verificado dos veces con segundos de por medio** |
| C5 | **Auditor en vivo** | faceta de frontera (Thot por defecto, configurable en DB, **nunca el mismo proveedor que el cerebro**: quien produce no aprueba) lee los pasos por lotes; si ve algo fuera de misión, prohibido, o un hecho sin salida de comando, escribe `PAUSE`. Informe final: afirmación ↔ evidencia | misión trampa que se sale del plan → detenida |
| C6 | **Revocación de un golpe** | un script quita la llave de `axioma` en todas las máquinas del inventario | revocar y comprobar que no entra a ninguna |

**Canario (lección de `CONTEXT.md`: todo verificador necesita una entrada que DEBE dispararlo):** C1 y
C5 llevan una entrada canario permanente; si no se dispara, el Ejecutor no arranca.

**Semilla de C1**, derivada de la Biblioteca (lecciones pagadas), no inventada:
`apt full-upgrade` en `.20` · `migrate:fresh` en producción · `sed -i` sobre un `.env` · parar,
deshabilitar o enmascarar `pure-ftpd` en `.11` · borrar respaldos o acortar retenciones (§7).

**Verdad incómoda, escrita para no engañarnos:** C1 y C2 operan por patrones de texto. **Atajan errores
honestos, no a un agente decidido a esquivarlos** (`hyde_sandbox.py` documenta que `python3 -c` esquiva
los filtros por patrón). La defensa de fondo son C3, C4 y C5. En Sésamo (§7) la prohibición sí es real.

**Gate verificable por un tercero (lección 29 de `CONTEXT.md`: el ejecutor no puede ser juez de su
propio gate):** el cierre de cada contrato lo declara una prueba que cualquiera puede volver a romper
(CI o script documentado), no el implementador.

## 5. Sesiones con contexto

- Cada conversación en modo Ejecutor = una sesión de Claude Code con id generado por Axioma
  (`--session-id <uuid>`); cada mensaje siguiente la retoma (`--resume`). Ambas banderas verificadas en
  la CLI 2.1.272 de hall9000. El id vive en `conversations` (columna nueva; ya tiene `project_id`).
- El historial de la sesión vive en la carpeta persistente del proyecto, respaldada.
- `stream-json` alimenta tres destinos: la pantalla, el auditor (C5) y LAS MANOS (C3).
- **Límite a medir (Fase 0):** Qwen corre con 32k de contexto; Claude Code compacta solo, pero hay que
  medir cuánto contexto cabe en la R9700 y cuánto se come el arranque.

## 6. La memoria que enseña

### 6.1 `CLAUDE.md` generado en cada arranque, nunca escrito a mano

(Lección del blueprint de Ricardo: una lista escrita a mano se desincroniza.) Contiene, en orden:
1. La **constitución completa** (núcleo común de `claude-skills`: principios, Regla Absoluta, políticas
   absolutas, Cuatro del Rendimiento, Memoria Viva).
2. **Reglas del proyecto**: inyectadas siempre, con tope; superarlo **rechaza** la regla nueva, nunca
   trunca en silencio.
3. **Hechos verificados y decisiones** del proyecto (`facts.is_verified`, `decisions.project_id`).
4. Inventario y prohibiciones de las máquinas.

### 6.2 Bitácora con aprobación humana

Al cerrar una misión el Ejecutor propone una entrada (qué probó, qué falló, cómo lo verificó, con el
número). Queda como **borrador**, excluido de toda búsqueda, y entra a la memoria solo con aprobación de
Fernando. Principio de Ricardo: que el modelo diga que funcionó no prueba que funcione.

### 6.3 La constitución se hace cumplir, no solo se lee

El checklist del auditor (C5) se deriva de la constitución: detiene o marca un hecho sin evidencia
(Principios I y VIII), un parche o solución temporal, código comentado o «después lo arreglo» (Regla
Absoluta, Principio II), hardcoding (IV), un cierre sin verificación independiente. El informe final
declara el origen de autoridad de cada afirmación.

## 7. Sésamo — «todo menos destruir», por API

- TrueNAS **25.10.4** (medido). API **JSON-RPC 2.0 por WebSocket** (la REST desaparece en TrueNAS 26).
- Usuario `axioma` con **privilegio a medida**: roles `*_READ` y `*_WRITE`, **sin** `DATASET_DELETE`,
  `SNAPSHOT_DELETE`, `ZFS_RESOURCE_DELETE`. **Nunca shell** (con shell de root el RBAC no vale nada).
- La herramienta del Ejecutor bloquea además **acortar una retención** (tareas de snapshot/replicación):
  sería un borrado indirecto.
- Borrar = GO de Fernando.
- **Por qué (hechos):** ADATA (`SU630`, `/srv/backup-adata`) está dentro de hall9000 y prod `.10` solo
  tiene ADATA + Sésamo (sin R2 hasta el cutover). Con `FULL_ADMIN`, un solo error del Ejecutor
  alcanzaría prod y todas sus copias.

## 8. Fases

Cada fase se verifica con evidencia antes de la siguiente.

**Fase 0 — Medir antes de construir.** GO una vez: recarga Qwen y corta `jax_local` unos minutos.
1. Contexto de Qwen en la R9700: 32k → 64k → 128k; ¿sigue 100% GPU?, tiempo de carga, tokens/s.
2. Tokens del arranque de Claude Code + nuestros plugins.
3. Examen de calidad: 10 tareas reales de solo lectura en nuestros servidores (incluye inventario de
   `.20`), con Qwen, Kimi y GLM; cada afirmación verificada contra la máquina. Criterio D11.
4. Coste del auditor por misión (tokens reales).
5. Carga (Cuatro del Rendimiento): p95 del chat de la Mesa mientras el Ejecutor trabaja; límite de
   misiones simultáneas (configurable, arranca en 1: Qwen atiende una inferencia a la vez).
6. **¿R2 tiene Object Lock?** (no verificado). Decide si existe una segunda copia inalcanzable.
7. Endpoints Anthropic de Moonshot y Z.ai probados con nuestras llaves.

**Fase 1 — Los seis contratos con sus pruebas** (TDD; no toca servidores).

**Fase 2 — El Ejecutor**: faceta, transporte `harness`, perfil de jaula, modo «Ejecutor» (i18n es/en,
claro/oscuro), paso de Pipeline, sesiones, `CLAUDE.md` generado, bitácora. Probado contra una **VM
desechable en hall9000** (libvirt, 131 GB libres en `/var`).

**Fase 3 — Usuario `axioma`** en cada máquina (GO por máquina); Sésamo por API; prueba de revocación.

**Fase 4 — Ensayo general.** El Ejecutor restaura el respaldo de `.20` en una VM desechable (ese
respaldo **nunca se restauró**: solo `restic check --read-data-subset=1%`) y migra esa copia a aaPanel
en otra VM desechable. Cutover ensayado con datos reales, sin riesgo para clientes.

**Fase 5 — Prueba de fuego.** Fernando lanza desde el modo Ejecutor la misión «cutover de `.20`».
Frontera planifica, el Ejecutor ejecuta, el auditor vigila.

## 9. Riesgos aceptados (firmados, no ocultos)

- **Autonomía total en DNS, correo y producción** para las misiones del Ejecutor (D4).
- **Red sin acotar** de la jaula (`--share-net`). El riesgo aceptado del 2026-09-01 para Hyde tenía
  como condición de revisión «gana acceso a otras máquinas»: el Ejecutor la cumple por diseño y
  **se cierra con D3–D6**.
- **C1/C2 por patrón**: atajan errores honestos (§4).
- **exim4 4.95 sin parche en `.20`** hasta la Fase 5 (D12).

## 10. Fuera de alcance

- Gestión de proyectos (spec 2).
- El arnés propio (camino B).
- El REPL.
- Mover otras facetas o cambiar el transporte de Hyde.

## 11. A verificar contra la DB viva antes de planificar

(Leído del archivo de esquema, no de la base; lección del blueprint de Ricardo y de `CONTEXT.md` §5.)
- `facts.fact_type` no tiene categoría para «regla»: añadirla o tabla propia.
- `conversations` necesita la columna del id de sesión.
- Si un subagente de Claude Code corre dentro del mismo proceso de la jaula.

*En memoria de Jairo Urbina. En honor al Prof. Raúl Jacobs.*
