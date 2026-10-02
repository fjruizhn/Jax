# Facetas Thot y Kimi por suscripción — diseño (escalón 3)

> Fecha: 2026-10-01 · Autor: arquitecto-adversarial (escalón 3, solo lectura), encargado por Hyde en hall9000.
> Pedido de Fernando (2026-10-01): pasar Thot (OpenAI) y Kimi (Moonshot) de API a suscripción, como Hyde.
> Estado: DISEÑO. D-1 a D-5 y D-7 CERRADAS por Fernando el 2026-10-01 (ver la adenda al final). D-6 CERRADA: (a)+(b). Fase 1 pasos 0-3 en feat/suscripcion-fase1.

# Diseño: Thot y Kimi por suscripción (escalón 3, solo lectura)

**Resumen.** El diseño es posible, pero la premisa "igual que ya funciona Hyde" no se sostiene. Hoy el transporte `subprocess` solo corre en el REPL viejo; en el chat web, en Jacobs y en el Motor Registry no hay un camino que funcione. Basta cambiar `facet.transport` o `provider.auth_type` para que pasen tres cosas: el chat de Thot queda sin su compuerta de autorización, sus pipelines se rompen y el motor Kimi se rompe. Abajo están los hechos, el diseño, el plan y lo que no pude verificar.

## 0 · Hechos verificados que cambian el diseño

**Dónde corre Hyde de verdad:**
- **Chat web:** no se despacha. `jax-platform/backend/api/chat.py:1289` devuelve la respuesta fija `hyde_usa_modo_comando`.
- **Jacobs:** `jax/jacobs/executor.py:643` lanza `GOVERNED_EXECUTION_REQUIRED` antes de cualquier código (Block 6).
- **Motor Registry:** `worker.py::_TRANSPORT_DISPATCH` solo tiene `http_openai_compat` y `ollama`.
- **Único camino vivo:** el REPL. `jax/core/main.py:83` → `SubprocessMuscle` → `run_sandboxed_claude`.

**Cómo se decide el transporte:**
- Lo decide `facet.transport`, no `provider.auth_type` (`jax-platform/backend/facet_resolver.py:293-311`). `auth_type` no participa del ruteo.
- El PUT de bindings (`api/admin/facet_bindings.py:131-193`) cambia solo `facet_binding`. Con transporte por faceta, volver a la API exigiría tocar `facet` con SQL a mano.

**Lo que se rompe si se pone `transport='subprocess'` en Thot o Kimi sin más cambios:**

| Dónde | Archivo | Qué pasa |
|---|---|---|
| Chat de Thot | `chat.py:86,988` | La compuerta `authorize-facet` está atada a `{"http_gemini","http_openai_compat"}`. Con `subprocess` **se salta**: se ignora `allowed_callers` (falla abierta). Después cae en `transporte_no_soportado` (`:1071`). |
| Pipelines de Thot | `executor.py:1190` | Todo `subprocess` va a `_invoke_hyde`, que lanza `GOVERNED_EXECUTION_REQUIRED`. Thot tuvo 11 llamadas en pipelines en los últimos 30 días. |
| Motor `kimi` (y `thot`) | vista `motor_resolved` | Toma `model_ref` del binding `primary` de la faceta del mismo nombre. Cambiar el binding de kimi a otro proveedor cambia el modelo del motor, que sigue como `http_openai_compat` y pide una credencial que no existe. Kimi tuvo 7 llamadas como motor en 30 días. |

**Credenciales.** Los servicios corren como `jaxsvc` (`systemctl show`). Las credenciales de Fernando son `600 fruiz` y `jaxsvc` no las puede leer: comprobé `~/.codex/auth.json` y `~/.kimi-code/credentials/kimi-code.json`. Además `~/.codex` trae un AGENTS.md de 71 KB (la constitución de Hyde), MCP ruflo, skills, plugins, un `history.jsonl` de 2,7 MB y una base de 228 MB. **No se monta nunca.** Hace falta un login propio de `jaxsvc` en un directorio dedicado.

**Modelos:**
- **Codex:** la caché local lista `gpt-6-sol` (el mismo slug de hoy). Que funcione con la cuenta ChatGPT no está probado (ver §8).
- **Kimi:** el modelo por defecto de Kimi Code, `kimi-code/kimi-for-coding`, se presenta como **"K2.8 Preview", no K3**. El equivalente del binding actual (`kimi-k3`) es **`kimi-code/k3`**. Hay que decidirlo, porque si no la faceta cambia de modelo sin que nadie lo note.

**Binarios:**
- Codex 0.159.0 es un binario musl estático y trae su propio `bwrap` en `codex-resources/`.
- Kimi 2.1.1 (Bun, 190 MB) **se actualizó solo hoy a las 09:19**.

**Anidar sandboxes.** El host tiene `kernel.apparmor_restrict_unprivileged_userns=1`. Dentro de nuestro bwrap rige el perfil `unpriv_bwrap`, que dice `audit deny capability`. Por eso el sandbox propio de Codex (su bwrap) casi seguro **no puede funcionar anidado**. No lo probé.

**Volumen.** Thot y Kimi juntos costaron unos US$2,66 en 30 días (`axioma_usage`), con un solo usuario. Hay 3 cuentas activas en el tenant 1 (1 superadmin y 2 operadores).

## Hallazgo sobre el código actual

**MAJOR · `jax/jax/muscles/subprocess_muscle.py:107` (con `_sanitize` en :83-86).** `safe_prompt = self._sanitize(contexto + prompt)` corta por la **cola**.
- **Escenario:** el REPL con Hyde y 10 turnos (`MAX_TURNS`) de respuestas técnicas largas pasa de 32.000 caracteres de historial. El mensaje actual de Fernando queda total o parcialmente cortado.
- **Efecto:** el modelo responde al contexto viejo, sin ningún error.
- **Para el diseño nuevo:** el runner nuevo no puede heredar esto.

**MINOR · `subprocess_muscle.py:12`.** El docstring dice "Prompt: por argumento", pero el prompt viaja por stdin (`hyde_sandbox.py:516`).

**Advertencia de reutilización (no es defecto hoy).** No hay que reutilizar `_check_error` (:88-99, que marca error si stderr contiene "error"/"failed"). `codex exec` sin `--json` manda el transcript entero a stderr, y cualquier conversación que mencione "error" fallaría.

## 1 · Arquitectura

**Un solo módulo, en el repo jax.**
- Nuevo `jax/cli_sandbox.py` en la raíz, con symlink `las_manos/cli_sandbox.py` (el mismo patrón que `hyde_sandbox.py`).
- Tiene un núcleo común: argv base de bwrap, env mínimo, flock y `run()`. Los perfiles están definidos **en código**: `PERFILES = {"claude": ..., "codex": ..., "kimi": ...}`.
- La base solo elige una clave de perfil (ENUM). Nunca guarda una ruta de binario ni flags: la confianza sigue a quien escribió el valor.
- `hyde_sandbox.wrap_hyde_command` pasa a usar ese núcleo. La condición es una prueba *golden* que muestre que el argv y el env de Hyde quedan **idénticos byte a byte**. Si no se logra, Hyde no se toca en la fase 1.

**jax-platform no lanza procesos.** Importa `cli_sandbox` desde `JAX_REPO`, igual que ya lo hace `shadow_validation.py:84-88`. Así el único `create_subprocess_exec` queda en el repo que el CI sí escanea.
- Alternativa descartada: un endpoint en las_manos. Agrega un salto de red y una superficie de autenticación nueva sin cerrar nada extra.

**Interfaz:** `run_cli(perfil, system_prompt, historial, mensaje, modelo, timeout, *, correlation_id, entry_point) -> ResultadoCLI(texto, tokens_in, tokens_out, version_cli, clase_error)`.

**Montajes comunes (codex y kimi):**
- `--unshare-all --share-net --die-with-parent --new-session`
- `--proc /proc`, `--dev /dev`, `--tmpfs /tmp`
- `/usr`, `/lib`, `/lib64`, `/bin`, `/sbin` en solo lectura, más `_ETC_RO_PATHS`
- `--tmpfs /home/cli-sandbox`
- **Sin repos, sin `~/.nvm`, sin nada de `/home/fruiz`.** Thot y Kimi de chat no necesitan código.
- Por llamada, `/run/jax-cli/<uuid>/` (RAM, `0700 jaxsvc`) se monta en solo lectura como `/work` con `--chdir /work`. Ahí van el system prompt, la memoria y el archivo de agente. **Nunca en el argv**, que se lee por `/proc/<pid>/cmdline`. El directorio se borra en un `finally`.

**Perfil codex:**
- Binario fijado en `/opt/jax-cli/codex/0.159.0/`, `root:root` en solo lectura, con su SHA256 constante en el perfil. Se verifica al cargar y se cachea por `(inode, mtime, size)`; esa es su invalidación.
- Credencial: bind de lectura y escritura del **directorio** `/srv/jax-data/cli-suscripcion/codex/` (`jaxsvc 0700`, login propio) como `CODEX_HOME`. Es el directorio y no el archivo, para que el refresh pueda escribir con rename.
- Env: `HOME`, `PATH`, `LANG`, `CODEX_HOME`, `CODEX_SQLITE_HOME=/tmp/codex-sqlite`.
- Comando: `codex exec - --json --ephemeral --skip-git-repo-check --ignore-user-config --ignore-rules -s read-only -m <modelo> -c model_reasoning_effort=... -c project_doc_max_bytes=...`.
- Desactivar herramientas: `shell_tool`, `unified_exec`, `apps`, `browser_use*`, `computer_use`, `image_generation`, `plugins`, `multi_agent`, `memories`, `hooks` y `web_search` (`--disable` / `-c`).
- **No usar `--dangerously-bypass-approvals-and-sandbox`.** Sin herramientas, el sandbox de Codex nunca se invoca y el problema de anidar desaparece. Con ese flag, si una herramienta llegara a correr, lo haría sin el freno de Codex.
- System prompt: `-c model_instructions_file=/work/sistema.md` si la clave existe en 0.159 (no verificado). Si no, AGENTS.md en `/work`, sabiendo que queda debajo de las instrucciones base de Codex (riesgo de que hable como agente de código).

**Perfil kimi:**
- Binario fijado en `/opt/jax-cli/kimi/2.1.1/kimi`, con SHA256.
- `KIMI_CODE_HOME` = bind de lectura y escritura de `/srv/jax-data/cli-suscripcion/kimi/` (login propio).
- Env: `KIMI_CODE_NO_AUTO_UPDATE=1`, `KIMI_CLI_NO_AUTO_UPDATE=1`, `KIMI_DISABLE_TELEMETRY=1`, `KIMI_DISABLE_CRON=1`. **Nunca `KIMI_CODE_INFINITE_RETRY`.**
- Comando: `kimi --agent-file /work/faceta.md -m kimi-code/k3 --output-format stream-json -p ...`.
- El archivo de agente lleva `tools: []` en el frontmatter y el system prompt en el cuerpo. El binario muestra que `possesses = profile.tools.includes(name)`; falta probarlo en ejecución.
- **No usar `--yolo` ni `--auto`.**
- Después de cada llamada, con el lock tomado, se purgan `sessions/`, `user-history/`, `logs/` y `telemetry/` del home dedicado (Kimi no tiene `--ephemeral`).
- Si `-p` no lee el prompt de stdin, se usa `kimi acp` (JSON-RPC por stdio). Lo que **no** se acepta es poner la conversación en el argv.

**Concurrencia.** Un flock por perfil con N ranuras (`/tmp/jax-cli-locks/<perfil>.<i>.lock`; N=2 configurable en `axioma_config`). No se reutiliza el lock de Hyde, que serializaría a Thot detrás de Hyde.
- La espera del lock para el chat es corta (unos 10 s). No es igual a `timeout` como en Hyde, que llega a unas 2×.

## 2 · Esquema y datos (recomendación: proveedores nuevos)

**Proveedores nuevos `openai_sub` y `moonshot_sub`** con `auth_type='subprocess'`, más una columna nueva `provider.cli_profile ENUM('claude','codex','kimi') NULL` (`anthropic` → `'claude'`). `openai` y `moonshot` quedan intactos.

No cambiar `auth_type` de los proveedores existentes, porque:
- Rompería el sync del catálogo de `openai` y `moonshot`.
- Rompería la gestión de llaves y el panel (`dashboard.py:30` cuenta los `api_key`).
- Haría imposible el respaldo sin tocar código.

**Transporte efectivo derivado del proveedor**, en los **tres** espejos de `facet_resolver` (jax/core, las_manos, jax-platform; mantener sincronizados con `scripts/check_facet_resolver_sync.py`):
- Si `p.auth_type='subprocess'`, el transporte es `'subprocess'` y `ResolvedFacet` gana el campo `cli_profile`.
- Si no, `f.transport`.

Con eso, **cambiar a la API y volver es un solo `PUT /api/admin/facet-bindings/{thot,kimi}`**: superadmin, auditado, con `probe_after_rebind`, sin SQL y sin esperar una aprobación nueva.

**Filas de `model`** (`source='manual'`):
- `openai_sub/gpt-6-sol` y `moonshot_sub/kimi-code/k3`.
- `input_modalities='text'`: las imágenes se rechazan limpio con 422.
- Precios en `0.000000` explícito, para que `cost_usd=0` y no NULL.

**Respaldo apagado.** Dejar los bindings `role='fallback_1'` → `openai/gpt-6-sol` y `moonshot/kimi-k3`. Hoy ningún resolvedor lee los fallback, así que quedan como dato documentado.

**Vista `motor_resolved`.** Que solo herede `fb.model_ref` cuando el proveedor del binding no es `subprocess`; si lo es, cae a `mo.model_ref`, que hoy es NULL en todos los motores y habría que sembrarlo con el modelo de la API. Otra opción es implementar el despacho por CLI en el worker (ver D-3).

**`axioma_config`:**
- `suscripcion_usuarios_permitidos` (user_ids de Fernando más la sonda canary)
- `suscripcion_<perfil>_timeout_s`
- `suscripcion_<perfil>_ranuras`
- `suscripcion_<perfil>_tope_diario`

## 3 · Modelo, historial, system prompt, límites

**Alias:** Thot → `gpt-6-sol`. Kimi → `kimi-code/k3`, salvo que Fernando quiera K2.8 de forma explícita.

**Historial:**
- Se serializa como en Hyde, pero con dos cambios:
  - Los delimitadores llevan un **nonce aleatorio por llamada**, para que un mensaje viejo no pueda imitar el `[Fin del contexto…]`.
  - Las etiquetas son neutras ("Usuario"), sin "Fernando" fijo.
- Se recorta **desde el turno más viejo**. Si el mensaje actual solo ya pasa el tope, el error es explícito.

**System prompt:** `personality.system_prompt`, más `_CONTRACT_PROMPT_SUFFIX`, más el snapshot de gobernanza y la memoria con su clasificación de confianza. Va a `/work/…`, nunca al argv.
- La memoria releída sigue sin ser confiable aunque venga de la tabla de la casa. Por eso las herramientas apagadas son obligatorias.

**Salida:** se parsea el JSONL estructurado y solo se toma el mensaje final. El razonamiento de Kimi (`always_thinking`) se descarta. Se piden los tokens de los eventos de uso.

**Timeouts:** de `axioma_config` (sugerido 180 s para el chat; hoy HTTP usa 120 s en `chat.py:863`). Recalcular `CANARY_SWEEP_TIMEOUT_SECONDS` (`facet_canary.py:66`, hoy 1180) con el peor caso nuevo.

## 4 · Errores: cuota y sesión

**Clasificar por evento estructurado y código de salida**, no por palabras en stderr. Las clases:

| Clase | Mensaje / efecto |
|---|---|
| `CuotaAgotada` | "Thot (Codex): cuota de la suscripción ChatGPT agotada. Para seguir por API: Admin → Bindings → thot → `openai/gpt-6-sol`" (i18n, `AvisoDeChat` y outcome `provider_error`) |
| `SesionVencida` | Se repite con `kimi login` / `codex login --device-auth` como `jaxsvc` |
| `ErrorProtocolo` | — |
| `Timeout` / `LockTimeout` | — |
| `SandboxUnavailable` | Falla cerrado (P10) |
| `BinarioAlterado` | El SHA no coincide |

- **Sin conmutación automática a la API:** Fernando pidió el respaldo apagado. Volver es el PUT de §2.
- Cuota agotada y sesión vencida disparan **una alerta de Telegram por ventana**. Hay que probarla una vez a propósito con un CLI falso que emita el error.
- Todo log lleva `correlation_id`, `entry_point` (chat|canary|jacobs|repl), `facet`, `provider_id`, `cli_profile`, `version_cli`, clase y latencia.
- Prueba de que alcanza: provocar `SesionVencida` y localizarla solo con la telemetría.

## 5 · Política y pruebas a extender

**`policy/tests/test_claude_subprocess_solo_via_sandbox.py`:**
- Agregar `cli_sandbox.py` y `_cli_sandbox_test.py` a `ALLOWED_FILENAMES` (:115).
- Ampliar la detección (:255-274). Hoy un `subprocess.run(["codex","exec"])` o `["kimi","-p"]` pasa limpio porque solo busca "claude".
- **No basta con buscar la palabra "kimi" en literales:** es el nombre de la faceta y del motor en todo el código, y `worker.py` usa `subprocess.run` (:313,:365). Daría falsos positivos.
- Detectar en cambio:
  - un literal que sea **argv[0]** de una llamada a subproceso igual a `claude|codex|kimi` o que termine en `/codex` o `/kimi`;
  - cualquier literal con `/opt/jax-cli`, `.kimi-code` o `.codex/packages`.
- Autopruebas positivas: run, Popen, execvp y una constante con la ruta de Kimi.
- Autoprueba negativa: `subprocess.run(["git","log"])` con `FACET="kimi"` no debe ser violación.

**`_cli_sandbox_test.py`:**
- El argv **no** contiene un centinela del prompt, del system prompt ni de la memoria.
- El `env=` es exactamente el devuelto, sin fusionarse con `os.environ`.
- No hay bind de `/home/fruiz/.codex`, `/home/fruiz/.kimi-code` ni de los repos.
- Golden de las banderas de "herramientas apagadas".
- El flock por perfil no bloquea a Hyde.
- El recorte nunca corta el mensaje actual.

**Contención real (como `_hyde_containment_test.py`):** dentro del perfil, `/etc/jax/.env`, `/home/fruiz/.ssh` y `/home/fruiz/.codex/auth.json` no existen, y escribir fuera de `/tmp` falla.

**Compuerta de chat:** `_GOVERNED_TRANSPORTS` pasa a gobernar también `subprocess` cuando `cli_profile ∈ {codex,kimi}`. Prueba: un binding `openai_sub` con `allowed_callers` sin `jax_platform_chat` termina en `faceta_no_autorizada`.

**Titular:** un usuario que no está en `suscripcion_usuarios_permitidos` recibe `suscripcion_solo_titular` y la CLI nunca se lanza (falla cerrado).

**Herramientas apagadas en ejecución (control del control):** una prueba de integración, marcada manual, pide "ejecutá `id` y leé /etc/passwd" y exige que no haya ningún evento de herramienta en el JSONL.

## 6 · Plan para el escalón 2

1. **jax · `cli_sandbox.py`** (núcleo y perfiles) más `_cli_sandbox_test.py`. Hyde migra solo si la prueba golden pasa.
2. **jax · política:** extender el scanner y sus autopruebas.
3. **jax-platform · `db/migrations.py`:** columna `cli_profile`, proveedores `*_sub`, filas de `model`, claves de `axioma_config`, vista `motor_resolved` corregida. Idempotente (`INSERT IGNORE`).
4. **Tres espejos de `facet_resolver`:** transporte efectivo y `cli_profile`, con el checker de sincronía en verde.
5. **jax-platform · `api/chat.py`:** rama `subprocess`+codex/kimi que llama a `run_cli`, compuerta de gobierno, compuerta de titular, `UsageInfo` con tokens y costo 0. Textos i18n nuevos.
6. **`jacobs/executor.py:1190`:** rutear por `cli_profile`. `claude` sigue cerrado; codex y kimi según D-3. `worker.py` según D-3.
7. **REPL:** `SubprocessMuscle` recibe `cli_profile`; `registro_facetas` acepta el mapeo.
8. **Operación (con el GO de Fernando):**
   - Instalar los binarios en `/opt/jax-cli` con su SHA.
   - Crear `/srv/jax-data/cli-suscripcion/{codex,kimi}` como `jaxsvc 0700`.
   - Login por device-code como `jaxsvc`.
   - Excluir esos directorios del restic (se regeneran con un login; un refresh token viejo en un respaldo solo es riesgo).
9. **Prueba de carga** (regla 4) en hall9000: p95 y la concurrencia a la que se degrada con N ranuras. Se registra en la Biblioteca.
10. **Corte:** PUT de los bindings, sonda y alerta disparada a propósito.

## 7 · Riesgos

- **Términos de uso:** solo uso propio de Fernando. La compuerta de titular es un contrato que se implementa **antes** de habilitar (Principio IX): hay 2 operadores más en el tenant. Nunca para BinB.
- **Cuota compartida:**
  - Codex compite con las sesiones de Codex del propio Fernando.
  - Kimi compite con el trabajo de BinB.
  - El tope diario por perfil no es una aprobación; falla cerrado con un mensaje claro.
- **Credencial dentro del sandbox:** la CLI puede leer su propio refresh token, el mismo riesgo residual B-4 de Hyde. Con las herramientas apagadas, el modelo no tiene cómo leerlo. La red queda completa (`--share-net`) y no hay proxy de salida.
- **Las herramientas apagadas dependen de flags de versión:** por eso los binarios fijados y las pruebas golden. El bwrap sin repos es la segunda capa.
- **Latencia:** arranque de la CLI más razonamiento, sin medir.
- **Persona:** las instrucciones base de Codex y Kimi Code son de agente de código y pueden contaminar la voz de Thot y Kimi.
- **Sesión compartida:** usar el login de Fernando en vez de uno propio causaría una carrera de refresh (rotación de refresh tokens). Por eso, logins dedicados.

## 8 · Sin verificar

- Si `kimi -p` acepta el prompt por stdin.
- Que `tools: []` deje a Kimi sin herramientas en ejecución.
- El esquema de `--output-format stream-json`.
- La clave `model_instructions_file` en Codex 0.159.
- El esquema exacto de los eventos de `--json` y de los errores de cuota y sesión (texto y código de salida) en las dos CLIs.
- Que `gpt-6-sol` responda con la cuenta ChatGPT. La caché lo lista con `supported_in_api: True`, pero no sé si eso cubre ChatGPT.
- Cómo escribe Codex `auth.json` al hacer refresh.
- El fallo del sandbox anidado de Codex: lo deduzco del perfil AppArmor, no lo probé.
- Si Kimi exige `device_id` en el home dedicado.
- Los términos de servicio de OpenAI y Moonshot.
- Si los 2 operadores son personas distintas de Fernando.

No gasté cuota: solo corrí `--help`, `features list`, `strings` sobre los binarios y SELECT.

## Decisiones para Fernando

- **D-1.** ¿Kimi en `kimi-code/k3` (mismo modelo que hoy) o en `kimi-for-coding` (K2.8 Preview)?
- **D-2.** ¿Quién va en `suscripcion_usuarios_permitidos`?
- **D-3.** Jacobs y el motor Kimi: ¿suscripción también (un camino por CLI sin herramientas en `_despachar_transporte_directo` y en el worker; el Block 6 se escribió pensando en Hyde con herramientas) o se quedan en API con `motor.model_ref` sembrado?
- **D-4.** ¿Hyde migra al núcleo común en esta fase?

No ratifico nada: el GO es de Fernando.

---

## Adenda 2026-10-01: decisiones de Fernando y diseño complementario (escalón 3)

> Autor: arquitecto-adversarial (escalón 3, solo lectura), por encargo de Hyde en hall9000.
> Fuentes leídas: jax `master` 183cefd; jax-platform `origin/master` 0b5cab7, leído con `git show` porque el checkout local está en `fix/tenant-user-mutex`, 47 archivos detrás; la base `jax_memory` en 3308, solo SELECT y SHOW; `~/.codex` y `~/.kimi-code`, solo metadatos.
> **Bloqueo de hook (texto exacto):** `PreToolUse:Bash hook error: Bloqueado: sub-agente (agent_id=a388a422cb78152ed) intento launchctl/systemctl sobre el servicio del sync (publicacion indirecta). Solo la sesion principal puede escribir a git compartido -- ver deuda 'sub-agentes sin gobernanza' en CONTEXT.md.` Lo disparó un `systemctl show -p User`. No lo esquivé. Saqué el usuario de los servicios de `ps` y de los drop-ins.

## A · Decisiones cerradas

| | Decisión | Fecha · autor |
|---|---|---|
| **D-1** | Kimi usa `kimi-code/k3` como modelo base. Fernando pide además un router de modelo por clase de tarea (diseño en §C). | 2026-10-01 · Fernando |
| **D-2** | Usan la suscripción **solo user_id 1 y 8**. Fernando declara que las dos cuentas son suyas. La base muestra a 8 como `operator`, `admin@axioma-ia.io`, y a 1 como `superadmin`; quién es el dueño no se puede verificar en la base. El resto de los usuarios usa su propia suscripción o su propia API key. | 2026-10-01 · Fernando |
| **D-3** | Jacobs y el motor Kimi también van por suscripción. Motivo: costo. | 2026-10-01 · Fernando |
| **D-4** | Hyde migra al núcleo común solo si la prueba golden pasa: argv y env idénticos byte a byte. | 2026-10-01 · Fernando |

## B · Hechos nuevos que corrigen el spec

1. **El checker no se llama así.** Hoy es `scripts/check_mirror_sync.py` (`:13-14`, renombrado el 2026-09-01) y corre en `policy.yml:1385`. Además, "tres espejos" son **dos archivos reales y un symlink**: `las_manos/facet_resolver.py -> ../jax/core/facet_resolver.py`. Corregir el §2 y el paso 4.
2. **Codex se actualizó solo hoy a la 0.160.0.** El binario de `~/.codex/packages/.../0.160.0-.../codex` es del 1-oct a las 11:33 y `codex --version` da 0.160.0. El spec fija la 0.159.0. Esto confirma que hace falta fijar la versión en `/opt` con su SHA. El paso 8 tiene que fijar la versión que se pruebe, no la 0.159.
3. **El transporte efectivo tiene más lectores que los resolvedores.** `contrato_dispatch.py:260` (plataforma) y `jacobs/prevuelo_catalogo.py:76` leen `facet.transport` directo.
4. **El pre-vuelo no controla nada en `subprocess`.** `jacobs/prevuelo_reglas.py:250-256` devuelve cero violaciones y costo 0: no mira titular, ni tope, ni si la capability necesita herramientas.
5. **El motor `kimi` tiene herramientas.** `motor.has_tool_access=1` para kimi y ada (el comentario de `worker.py:612-620` dice "solo jax_local", y está desactualizado). `capability_motor` manda `implementation`, `refactor`, `code_swarm` y `bug_hunt` solo a kimi, y `file_write` a kimi con prioridad 1. El worker arma el bucle de herramientas (`worker.py:620, 696-743`), y `jacobs_events` registra `TOOL_CALL_*` y `TOOL_WRITE_REVERTED`. No verifiqué cuántos de esos eventos son de kimi.
6. **Hoy no hay respaldo.** Thot y kimi tienen solo la fila `primary`. El §2 dice "dejar" los bindings `fallback_1`, pero hay que **crearlos**.
7. **El REPL no puede leer el login dedicado.** Los servicios corren como `jaxsvc` (`ps`, y el drop-in `cuenta-de-servicio.conf` de las_manos), pero el REPL (`~/.local/bin/jax`) corre como `fruiz`. Un `/srv/jax-data/cli-suscripcion/*` con `jaxsvc 0700` le queda fuera. El paso 7 del spec no se puede hacer tal como está escrito.
8. **`user_api_keys` no son llaves por usuario.**
   - Las 5 filas son `user_id=1`, una por proveedor: es la copia legada de las llaves de la plataforma.
   - Su único lector es `api/admin/keys.py`, siempre con `USUARIO_LLAVES_LEGADO=1` (`:37`, `:133-165`), más la migración de una sola vez (`migrations.py:2052`).
   - `user_id` tiene `DEFAULT 1`.
   - **Ningún resolvedor resuelve por usuario.** `credential_resolver.resolve_credential(provider_id)` (`:98-125`) busca por proveedor, la caché usa solo `provider_id` como clave, y `credential` no tiene dueño.
   - `facet_resolver` cachea `ResolvedFacet`, **con la credencial adentro**, usando solo `facet_key` como clave (`:106, :331-345`).
9. **`axioma_usage` no alcanza para cobrar por usuario.**
   - No tiene `provider_id` ni columna de origen de la credencial.
   - `user_id` tiene `DEFAULT 1`.
   - No hay índice por `user_id` (solo `created_at, facet...` y `pipeline_id`).
   - Hay 15 filas de `kimi`/`motor` con `user_id NULL`, entre el 21 y el 24 de agosto.
10. **F2-D impide hoy mostrar el modelo.**
    - `ChatResponse` (`chat.py:592-612`) no trae el modelo.
    - `BottomBar.jsx:241-245` prohíbe reconstruir del lado del cliente quién respondió (faceta, modelo o binding).
    - `_validate_projection` ata solo `facet`, `timestamp` y `contract_degraded` (`webchat_f2d/transport.py:54-56`).
    - Si alguien pregunta "qué modelo sos", recibe `estado_actual_no_disponible` (`chat.py:1034`).
11. **El PUT genérico de configuración escribe cualquier clave.** `PUT /api/admin/config` (`config_admin.py:128`, solo superadmin) acepta cualquier clave que no empiece con `smtp.`. Por eso la lista `suscripcion_usuarios_permitidos` del §2 se podría editar desde la pantalla genérica.
12. **Dos huecos del PUT de bindings.**
    - `role` es un `str` libre (`facet_bindings.py:128`): un valor fuera del ENUM, con `STRICT_TRANS_TABLES`, termina en 500, no en 400.
    - `probe_after_rebind` solo sondea el `primary` (`facet_canary.py:169-215`).
13. **El uso real no coincide con el binding.** En 30 días, Thot facturó con `gpt-5.6-terra` y `gpt-6-astra`; el binding de hoy es `gpt-6-sol`. Nadie fuera del user 1 (o sin usuario) usó la plataforma en 90 días.

## C · Router de modelo por clase de tarea

**Dónde vive: una tabla nueva, sin tocar `facet_binding`.**
```
facet_task_route(id PK, facet_key FK facet, task_class ENUM('exploracion','codigo','auditoria','apelacion') NOT NULL,
  provider_id FK, model_ref FK model NOT NULL, approved_by FK jax_users NOT NULL, approved_at DATETIME NOT NULL,
  UNIQUE(facet_key, task_class))
```
Opciones descartadas:
- **Ampliar el ENUM `role`.** Mezcla el orden de respaldo con la clase, y `uk_facet_role` impide tener un respaldo por clase.
- **Agregar una columna `task_class` a `facet_binding`.** Hay unos 20 lectores con `role='primary'` sin más filtro: los dos resolvedores, `motor_resolved`, `keys.py:123`, `model_catalog`, `adjuntos/politica.py`, `ejecutor/misiones.py`, `prevuelo_catalogo`, `registro_facetas` y `gobernanza_semilla`. Con dos filas `primary` por faceta, `fetchone()` devolvería una cualquiera (escenario de fallo en §H-1).

**Quién clasifica: nunca un LLM, nunca el texto del usuario.**
- **Jacobs:** usa `step.capability`, que se valida contra el catálogo, más una columna nueva `capability.task_class ENUM NULL` que se siembra por migración. NULL quiere decir "sin clase".
- **Chat:** usa el primario, salvo que el usuario elija la clase de forma explícita (un campo `clase_tarea` validado contra el ENUM). Clasificar por palabras del mensaje queda fuera: con eso, un tercero que entra por la memoria o por un adjunto decide el modelo y el gasto.
- **`apelacion`:** nunca se asigna sola. Solo un titular la pide, de forma explícita, en el chat.

**Resolución, en los dos archivos reales y en la familia de `check_mirror_sync.py`:**
- La firma pasa a ser `resolve_facet(facet_key, task_class=None)`.
- **Si hay ruta,** `provider_id` y `model_ref` salen **los dos de la ruta**. Nunca un `COALESCE` columna por columna, que podría juntar el proveedor del primario con el modelo de la ruta.
- **Si no hay clase,** o la clase no tiene ruta, sale el primario y se marca `motivo_ruta ∈ {sin_clase, clase_sin_ruta}`.
- La clave de caché pasa a ser `(facet_key, task_class)`. El sello actual ya invalida todas las entradas.
- `ResolvedFacet` gana tres campos: `task_class_aplicada`, `motivo_ruta` y `cli_profile`.
- El transporte efectivo sale de **una sola función** espejada, `transporte_efectivo(facet.transport, provider.auth_type)`, y la usan resolvedores, `contrato_dispatch`, `prevuelo_catalogo` y el catálogo de motores.

**Cómo se cambia:**
- `PUT /api/admin/facet-bindings/{key}/rutas/{task_class}` y su `DELETE`, solo superadmin.
- Usan el mismo `detalle_si_rompe_el_contrato`, ya con el transporte efectivo.
- Se audita en `model_catalog_audit`.
- Después del commit se encola `probe_after_rebind(facet_key, task_class)`: la sonda gana el parámetro y `facet_health_event` gana la columna `task_class`.
- `role` pasa a ser `Literal[...]`, para que un valor inválido sea 422 y no 500.
- Nunca un UPDATE a mano.

**Las compuertas se evalúan sobre la ruta resuelta, no sobre el primario.** La compuerta de autorización (`authorize-facet`) y la de titular corren después de resolver, como hoy en `chat.py:1047`. Si se evaluaran sobre el primario, una ruta `codigo` → `moonshot_sub` saltaría las dos.

**Cómo se ve en el chat.** Se agrega `ejecucion = {provider_id, model_id, task_class_aplicada, motivo_ruta, via: suscripcion|api|local}`, que el servidor deriva del `ResolvedFacet` que de verdad se usó. Se suma a `trusted_metadata` y a `_validate_projection`, y también se persiste en la memoria y en `axioma_usage`. **Esto cambia el contrato F2-D** (B-10). Lo tiene que decidir el dueño de F2 o Fernando (D-7) antes de habilitar el router.

**No degradar en silencio.**
- Si la clase no se pudo determinar: se usa el primario y se dice (`motivo_ruta`).
- Si la ruta existe pero falla (cuota, sesión, timeout o modelo `gone`): error tipado, **sin** caer al primario ni a otra clase.
- Nunca se conmuta solo a la API.

**Clases y modelos propuestos:**

| Clase | Capabilities | Kimi (suscripción) | Thot (Codex con cuenta ChatGPT) |
|---|---|---|---|
| exploracion (Luna) | research, analysis, file_read, pipeline_analysis | `kimi-code/kimi-for-coding-highspeed` | `gpt-6-luna` (el router de Codex usa `gpt-5.6-luna`) |
| codigo (Terra) | implementation, refactor, code_swarm, generate, file_write | `kimi-code/kimi-for-coding` (K2.8) | `gpt-5.6-terra` |
| auditoria (Sol) | critique, review, validate_consistency, architecture_review, bug_hunt, reason, design, reconcile | `kimi-code/k3` (= primario, D-1) | `gpt-6-sol` (= primario) |
| apelacion (Astra) | solo pedido explícito | `kimi-code/k3-256k` | `gpt-6-astra` |

Los slugs salen de `~/.kimi-code/config.toml [secondary_model]` y de `~/.codex/models_cache.json`, y los mapeos Codex, de `~/.codex/agents/*.toml`. La caché también lista `gpt-6.1-sol`, que es nueva y tiene prioridad 1. **No está verificado** que esos slugs respondan con la cuenta ChatGPT. Tampoco que K2.8 gaste menos cuota que K3: en una suscripción, "más barato" es cuota, no dólares.

**Recomendación de fase: el router va en una fase 2 separada, después de que la fase 1 esté estable.**
- La fase 1 ya trae un transporte nuevo, custodia de credenciales y la compuerta de titular.
- El router multiplica bindings, sondas y compuertas por cuatro clases y cambia el contrato F2-D.
- No difiere ningún contrato (Principio IX): la capacidad (rutas) no se habilita en fase 1, y sus contratos (compuerta por ruta, sonda por ruta, `ejecucion` atada en F2-D) llegan **junto con** ella.
- Lo que sí entra en fase 1, porque D-2 lo necesita ya: las columnas de origen de la credencial en `axioma_usage` (§D).

## D · Credenciales por usuario (D-2)

**La compuerta de titular.**
- **Dónde se guarda la lista.** Se recomienda `JAX_SUSCRIPCION_TITULARES=1,8` en `/etc/jax/.env` (`root:jaxsvc 640`), **no** en `axioma_config`. La confianza sigue a quien escribió el valor: una fila de `axioma_config` la puede escribir cualquier proceso con la credencial de la base y cualquier superadmin desde el PUT genérico (B-11). Esa sería una autoridad capaz de agregarse titulares a sí misma. El archivo solo lo escribe root, cambiarlo exige reiniciar y, al arrancar, se avisa a Telegram con la lista.
- **Cómo se verifica.** Un único `exigir_titular(user_id, tenant_id, entry_point) -> Titular` en `cli_sandbox.py`. Exige que el `user_id` sea entero y esté en la lista, y que en `jax_users` la cuenta esté `status='active'`, con `deleted_at IS NULL` y el `tenant_id` que corresponde. Esa consulta se hace sin caché, o con un TTL ≤ 30 s declarado. Si `user_id` es `None`, se niega.
- **Un solo punto de paso.** `run_cli` exige el objeto `Titular`, que solo construye `exigir_titular`, así que ningún llamador puede saltarlo.
- **La sonda.** `entry_point='canary'` lo fija solo el código de la sonda, dentro del proceso. Nunca viene de un request.

**El resto de los usuarios.**
- **Fase 1:** con la ruta en suscripción y sin ser titular, la respuesta es `suscripcion_solo_titular` (i18n). La CLI no se lanza y **no** se cae a la API con la llave de Fernando.
- **Pregunta nueva (D-5).** Hoy el user 4 en Thot por API usa `credential(openai)`, que es la llave de Fernando, porque `credential` no tiene dueño. El texto de D-2 ("su propia API key") implica cortar también eso. Si Fernando lo confirma, la compuerta se aplica a todo proveedor pago que no sea local.

**Llave propia por usuario: una tabla nueva, en fase 1b.**
- La tabla es `user_credential`: `tenant_id` y `user_id` NOT NULL sin default, `state`, auditoría propia y UNIQUE activo por (user, provider).
- **No se reutiliza `user_api_keys`:** sus filas de user 1 son las llaves de la plataforma, y su `DEFAULT 1` le atribuye a Fernando cualquier INSERT que omita `user_id`.
- **No se agrega un dueño a `credential`:** la consulta del resolvedor (`credential_resolver.py:84-88`, `ORDER BY activated_at DESC LIMIT 1` por proveedor) devolvería la llave de un usuario como la de la plataforma.
- `resolve_facet` deja de cargar la credencial. Se resuelve en cada llamada con `resolve_credential_for(tenant, user, provider)`, con caché que incluye tenant y usuario.

**Suscripción por usuario: se difiere.** Implica:
- custodiar refresh tokens de terceros;
- todos los logins bajo el mismo uid `jaxsvc` (el proceso de la plataforma los lee todos; bwrap aísla por llamada, no por usuario);
- un login por device-code que hay que pasar por la web;
- ranuras y topes por usuario;
- términos de uso de planes de consumo usados desde un servidor ajeno, sin verificar;
- un `device_id` de Kimi, sin verificar.

Diferirla es seguro porque la compuerta falla cerrado.

**Auditoría y costos.**
- `axioma_usage` gana `provider_id`, `origen_credencial ENUM('suscripcion_titular','llave_plataforma','llave_usuario','local')`, `entry_point` y, en fase 2, `task_class`.
- Se agrega un índice `(user_id, created_at)` y se verifica con `EXPLAIN` sobre el reporte real.
- El default de `user_id` pasa a NULL.
- En suscripción, `cost_usd=0`, pero se registran los tokens para medir la cuota.

## E · D-3 concretado

- **`jacobs/executor.py:1188-1196`:** antes de la rama de Hyde va `if f.transport=="subprocess" and f.cli_profile in ("codex","kimi"): return await _despachar_cli(...)`. Esa función llama a `exigir_titular(pipeline.user_id, pipeline.tenant_id, "jacobs")`, después a `run_cli`, y registra con `record_direct_usage` con origen, en éxito y en truncado. Con `claude` sigue `_invoke_hyde`, que sigue prohibido.
- **`executor.py:1049-1050`:** el `else: _invoke_ollama` pasa a ser `elif f.transport=="ollama"` más un `raise` (§H-5).
- **Block 6:** `policy/**` no menciona `subprocess` (solo lo menciona el código de `executor.py:643`). Que una CLI sin herramientas sea "llamada a modelo" y no "ejecución" tiene que quedar declarado en la política, con el control de herramientas apagadas del §5 como evidencia. **NO VERIFICADO** que los dueños de Block 6 lo acepten.
- **Pre-vuelo (`prevuelo_reglas.py:250`):** con `cli_profile` exige titular y tope diario, y una capability que pide herramientas es una violación.
- **`worker.py:214`:** `_TRANSPORT_DISPATCH["subprocess"]` existe **solo si `has_tool_access=0`**. Con 1, el job queda FAILED con "motor con herramientas no va por CLI" (falla cerrado). Usa el `user_id` del job, y un job sin identidad se niega.
- **El motor `kimi` tiene `has_tool_access=1` (B-5), así que D-3 para el motor no se puede cumplir sin perder herramientas.** **D-6 para Fernando:** (a) el motor kimi sigue por API y D-3 cubre solo la faceta Thot en Jacobs; (b) `kimi_sub` como motor nuevo con `has_tool_access=0`, solo para capabilities sin herramientas; o (c) investigar `kimi acp`, donde el cliente atiende las operaciones de archivos, para pasarlas por `tool_authority`. La (c) no está verificada y sería una fase 3.
- **`motor_resolved` / `catalog.py:248-256`:** el catálogo calcula el transporte con `transporte_efectivo(...)` en vez de leer `m.transport`. La vista no cambia.

## F · D-4: la prueba golden

- La fixture se genera con el `wrap_hyde_command` **actual**, en un commit propio y anterior al refactor, con el env controlado (`HYDE_OAUTH_TOKEN_ENV` como centinela, `_BWRAP_BIN` fijo y un workspace fijo).
- Un test exige que el blob de la fixture no haya cambiado desde ese commit. Si se regenerara con el código nuevo, pasaría por la razón equivocada.
- Hay que agregarla a `policy/tests/test_archivos_de_test_wireados_en_ci.py`.

## G · Plan actualizado para el escalón 2

| # | Repo · paso | Prueba |
|---|---|---|
| 0 | Commit de la fixture golden de Hyde | La fixture coincide con el código actual |
| 1 | jax · `cli_sandbox.py`: núcleo, perfiles, `exigir_titular`, `Titular`, `transporte_efectivo` | `_cli_sandbox_test.py` (del §5), más: sin `Titular` no lanza; `None` se niega; un usuario borrado se niega |
| 2 | jax · Hyde usa el núcleo | La golden del paso 0 queda idéntica; si no, se revierte |
| 3 | jax · scanner de política (§5) | Autopruebas positivas y negativas |
| 4 | plataforma · migración: `provider.cli_profile`, `*_sub`, filas de `model`, **creación** de los `fallback_1`, columnas e índice de `axioma_usage` | Idempotente; `EXPLAIN` del reporte por usuario |
| 5 | `transporte_efectivo` en los resolvedores, `contrato_dispatch` (las dos copias), `prevuelo_catalogo` y `catalog.py`; familia en `check_mirror_sync.py` | Un PUT hacia `moonshot_sub/kimi-code/k3` **no** da 409; el checker queda en verde |
| 6 | plataforma · `chat.py`: rama CLI, `_GOVERNED_TRANSPORTS` según `cli_profile`, compuerta de titular, `UsageInfo` con origen; i18n | Un no titular recibe `suscripcion_solo_titular` y la CLI nunca se lanza |
| 7 | jax · executor (§E) y `else` explícito | Un `subprocess`+codex nunca llega a `_invoke_ollama`; un pipeline sin `user_id` se niega |
| 8 | jax · pre-vuelo | Una capability con herramientas sobre CLI es una violación |
| 9 | jax · worker, según D-6 | `has_tool_access=1` + `subprocess` → FAILED |
| 10 | REPL: **fuera de la fase 1**, o con un login propio de `fruiz` en `~/.local/share/jax-cli/` | — |
| 11 | Operación con GO: binarios con SHA (la versión probada), logins de `jaxsvc`, `JAX_SUSCRIPCION_TITULARES`, exclusión del restic | Telegram con la lista de titulares |
| 12 | Prueba de carga y corte (§6, pasos 9 y 10) | p95 y concurrencia a la que se degrada, anotados en la Biblioteca |
| F2 | Router (§C) más el contrato F2-D de `ejecucion` (D-7) | Sonda por ruta; una ruta caída no cae al primario |
| 1b | `user_credential`, la llave propia (§D) | Caché con clave (tenant, user, provider); un usuario sin llave no hereda la de Fernando |

## H · Hallazgos

- **MAJOR · H-1 · solo si se elige la columna `task_class` en `facet_binding`.** Lectores: `jax/core/facet_resolver.py:253` y `motor_resolved`. Si se inserta una fila `primary` de clase `codigo` antes de desplegar los lectores filtrados, `_query_facet` devuelve una fila arbitraria (`fetchone` sin ORDER) y el chat de Kimi responde con K2.8 sin que nadie lo sepa. Además, `motor_resolved` duplica la fila del motor. Por eso se eligió la tabla aparte.
- **MAJOR · H-2 · `contrato_dispatch.py:260` y `prevuelo_catalogo.py:76`.** El `PUT facet-bindings/kimi` hacia una fila `moonshot_sub` con `max_tokens_param` NULL lee `facet.transport='http_openai_compat'` y devuelve **409 `modelo_sin_contrato_de_dispatch`**. Si se sortea sembrando ese campo, el pre-vuelo busca una credencial de `moonshot_sub` que no existe y bloquea todos los pipelines. Hoy el corte del spec no se puede hacer como está descrito.
- **MAJOR · H-3 · `credential_resolver.py:84-88`, si se agrega un dueño a `credential`.** Si un usuario carga su llave de OpenAI (con `activated_at` más nuevo), el `ORDER BY activated_at DESC LIMIT 1` por proveedor se la sirve a todos, Fernando incluido. Es una fuga de credencial entre usuarios.
- **MAJOR · H-4 · `axioma_config` como lista de titulares (spec §2).** Un segundo superadmin, o cualquier proceso con la credencial de la base, agrega su `user_id` a `suscripcion_usuarios_permitidos` desde `PUT /api/admin/config` y empieza a consumir la suscripción de Fernando, con el riesgo de términos de uso que eso trae. La única marca sería una auditoría de configuración genérica.
- **MINOR · H-5 · `executor.py:1049-1050`.** Si el escalón 2 agrega `"subprocess"` a la tupla de `:1188`, un paso de Thot por suscripción se manda a `_invoke_ollama` con el modelo `gpt-6-sol`. Falla con un error de Ollama que tapa la causa.
- **MINOR · H-6 · `facet_bindings.py:128`.** `role="foo"` termina en 500 (`DataError`), no en 400, y un `role='fallback_1'` dispara la sonda del primario.
- **MINOR · H-7 · `axioma_usage.user_id DEFAULT 1` y `user_api_keys.user_id DEFAULT 1`.** Un escritor nuevo que omita `user_id` le atribuye el gasto a Fernando. Hoy los tres escritores lo pasan explícito.
- **MINOR · H-8 · `facet_resolver.py:121` (plataforma).** Dice "User=fruiz en los dos units"; los procesos corren como `jaxsvc`.

## I · Riesgos nuevos y lo que queda sin verificar

**Riesgos:**
- Las dos CLIs se actualizaron solas el mismo día; fijar la versión es obligatorio, no prudencia.
- En Jacobs y en el motor, el `user_id` llega en el body (frontera de confianza declarada en `motor_registry/models.py:49`): la compuerta de titular confía en que el servicio que llama esté autenticado.
- Si D-6 no se resuelve, D-3 para el motor kimi queda sin cumplir. Prometerlo sería decir algo que no es cierto.

**Sin verificar** (además de lo del §8 original):
- que los slugs de Codex respondan con la cuenta ChatGPT;
- cuánta cuota gasta cada modelo de Kimi Code;
- que `kimi acp` delegue las operaciones de archivos al cliente;
- la postura de Block 6 sobre una CLI sin herramientas;
- cuántos `TOOL_CALL_*` son de kimi;
- que user 8 sea Fernando (está declarado, no verificado).

**Preguntas nuevas para Fernando:** D-5 (¿la compuerta corta también las llaves API de la plataforma para otros usuarios?), D-6 (¿qué hacer con las herramientas del motor kimi?) y D-7 (¿se cambia el contrato F2-D para mostrar el modelo?).

No ratifico nada: el GO es de Fernando.

### Decisiones D-5 a D-7 (Fernando, 2026-10-01)

| | Decisión |
|---|---|
| **D-5** | **Cortar ya.** La compuerta de titular se aplica a todo proveedor pago que no sea local. Quien no es titular (user 1 u 8) y no tiene llave propia recibe un error tipado (i18n) y nunca usa la credencial de la plataforma. Entra en la fase 1 (paso 6), no se difiere a la 1b. |
| **D-6** | **Investigar primero la opción (c), `kimi acp`**: que Kimi pida las operaciones de archivos al cliente para que pasen por `tool_authority`. Solo si no sirve se decide entre (a) y (b). El paso 9 (worker) espera ese resultado; los pasos 0 a 8 no dependen de él. |
| **D-7** | **Sí.** El servidor declara `ejecucion = {provider_id, model_id, task_class_aplicada, motivo_ruta, via}` en el contrato F2-D, atada en `_validate_projection`. Entra con el router, en la fase 2. |

### D-6 cerrada (Fernando, 2026-10-02): (a) + (b)

La investigación de escalón 3 descartó la opción (c), `kimi acp` con kimi 2.1.1, con experimento y lectura del binario:
- `Grep`/`Glob` leen el disco local sin pasar por el cliente ni pedir permiso;
- `Write` crea directorios aunque el cliente niegue la escritura;
- Kimi lanza los servidores MCP de su `mcp.json` aunque `session/new` pida `mcpServers: []`.

ACP es un canal de delegación opcional, no un confinamiento. Decisión:
- **(a)** el motor `kimi` con herramientas (`has_tool_access=1`: implementation, refactor, code_swarm, bug_hunt, file_write) **sigue por API**;
- **(b)** motor nuevo **`kimi_sub`**, por suscripción, con `has_tool_access=0`, para las capabilities sin herramientas (análisis, revisión, auditoría), con compuerta de titular. Paso 9 del plan.

Se reconsidera si Kimi permite desactivar herramientas en modo ACP.

### Paso de host de locks (hecho 2026-10-02, GO de Fernando)

- grupo `jax-cli-lock` (jaxsvc, fruiz);
- `/etc/tmpfiles.d/jax-locks.conf` (`/run/jax-locks/hyde` `0750 root:jax-cli-lock` con `3a8b11a89cad98b1.lock` `0640`, y `/run/jax-locks/cli` `0700 jaxsvc`);
- drop-in `jax-las-manos.service.d/locks.conf` (`SupplementaryGroups=jax-cli-lock`), con `daemon-reload` sin reiniciar.

Verificado: exclusión cruzada fruiz↔jaxsvc con `flock` sobre `O_RDONLY`, y `axioma` recibe EACCES. Pendiente: que fruiz vuelva a iniciar sesión, la integración y el reinicio de las_manos al desplegar.
