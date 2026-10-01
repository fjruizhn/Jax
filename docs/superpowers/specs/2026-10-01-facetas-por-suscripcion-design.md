# Facetas Thot y Kimi por suscripción — diseño (escalón 3)

> Fecha: 2026-10-01 · Autor: arquitecto-adversarial (escalón 3, solo lectura), encargado por Hyde en hall9000.
> Pedido de Fernando (2026-10-01): pasar Thot (OpenAI) y Kimi (Moonshot) de API a suscripción, como Hyde.
> Estado: DISEÑO, pendiente de las decisiones D-1 a D-4 de Fernando. Nada implementado.

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