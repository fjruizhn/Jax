# Ejecutor · Fase 1 (SP1) · los seis contratos — índice y decisiones

> **Para agentes ejecutores:** este documento NO tiene tareas. Es el mapa de los seis planes de SP1,
> las decisiones de diseño con su porqué y lo que el spec dice mal. Cada plan numerado es
> autocontenido y se ejecuta con `superpowers:subagent-driven-development` (recomendado) o
> `superpowers:executing-plans`.

**Objetivo:** que C1–C6 existan, se hayan **visto fallar** con una prueba que un tercero puede volver a
romper, y que el Ejecutor **no arranque** si alguno no está vivo (canarios de C1 y C5).

**Specs:** `docs/superpowers/specs/2026-09-15-ejecutor-design.md` (§3.2, §3.4, §4, §7, §8) y
`docs/superpowers/specs/2026-09-16-ejecutor-fase2-design.md` (§2.0, §2.2 avisos, §3.4 bis, §6.2, §7).
**Alcance y fronteras:** `~/ejecutor-producto/LEDGER.md` (SP1; hallazgos del mapa de cableado).

## Los planes, en orden de ejecución

| # | Plan | Repos | Depende de |
|---|---|---|---|
| 1 | `2026-09-17-ejecutor-contratos-1-c1-c2-prohibiciones-y-respaldo.md` | jax-platform (tablas) + jax | nada |
| 2 | `2026-09-17-ejecutor-contratos-2-c3-registro-y-cerco.md` | jax | plan 1 (inventario y `cuenta_axioma`) |
| 3 | `2026-09-17-ejecutor-contratos-3-c4-freno-en-vuelo.md` | jax | **merge del frente B** (`feat/kill-switch-real`) + planes 1 y 2 |
| 4 | `2026-09-17-ejecutor-contratos-4-c5-auditor.md` | jax-platform (config) + jax | plan 2 (registro) + frente B (`escribir_pausa`) |
| 5 | `2026-09-17-ejecutor-contratos-5-c6-revocacion.md` | jax | plan 1 (inventario); Parte B toca .10/.11/.20 **sin destruir** |
| 6 | `2026-09-17-ejecutor-contratos-6-arranque.md` | jax | planes 1–5 |

Un PR por plan (en jax-platform, un PR aparte para las tablas del plan 1 y la config del plan 4, que se
mergean **antes** que el PR de jax del mismo plan: el job `jacobs-gobernanza-db` clona jax-platform
`master`).

## Decisiones de diseño (leí el código; cada una con su porqué)

### D-SP1-1 · C1 y C2 viven en un gancho `PreToolUse` instalado en `/opt/ejecutor/lib`, activado por un `managed-settings.json` que SÓLO ve la jaula

- **Por qué un gancho y no el proxy ni LAS MANOS:** el gancho es lo único que ve la llamada **antes** de
  que corra y puede negarla (exit 2). El proxy ve el `tool_use` cuando el modelo lo pide, pero bloquearlo
  ahí rompe la conversación en vez de devolverle al modelo un «no» legible. LAS MANOS no ve nada: Claude
  Code ejecuta sus herramientas él mismo.
- **Por qué NO `/etc/claude-code/managed-settings.json` a secas:** ese archivo es de TODO el sistema.
  Aplicaría el gancho a las sesiones de Claude Code de Fernando en hall9000, y `allowManagedHooksOnly`
  le apagaría sus propios ganchos (`block-subagent-git-write.sh`). Se monta **sólo dentro de la jaula**
  con `bwrap --tmpfs /etc/claude-code --ro-bind <archivo> /etc/claude-code/managed-settings.json`.
  **Medido el 2026-09-17 como `axioma`:** `bwrap --dev-bind / / --tmpfs <dir existente> --ro-bind <archivo> <dir>/x`
  funciona (rc=0). En el host se crea `/etc/claude-code` vacío (sin archivo: no afecta a nadie).
- **Fail-closed de verdad (documentación oficial de Claude Code, verificada 2026-09-17):** un gancho que
  sale con 1, con 127, se cae o vence su timeout **NO bloquea**: la herramienta sigue. Por eso `gancho.sh`
  convierte todo lo que no sea un 0 explícito en 2, se corta a sí mismo con `timeout -s KILL` antes que el
  timeout del gancho, y la política ilegible (ausente, corrupta, con sha256 que no cuadra, escribible por
  la cuenta, sin canario) bloquea todo.
- **La lista está en la DB** (`ejecutor_regla`), pero la jaula no tiene credenciales de la DB ni red hacia
  ella (C3, cerco). Un exportador que corre como `fruiz` la vuelca a `/etc/jax-ejecutor/politica.json`
  (directorio `fruiz:axioma 2750`, archivo `0640`): `axioma` la lee y no la puede tocar.
- **Cada regla trae sus ejemplos** (`ejemplos_coincide`, `ejemplos_no_coincide`) y al arrancar se corre la
  autoprueba de TODAS como `axioma` con el gancho instalado. Es literalmente la prueba del spec («cada
  prohibido se intenta y sale bloqueado»), repetida en cada arranque y no una vez.
- **Canario C1 por Claude Code real, sin modelo:** un upstream Anthropic falso (`canario_upstream.py`)
  devuelve un `tool_use` de Bash con el comando canario y después uno de control. Se lanza el `claude` real,
  como `axioma`, dentro de la jaula. Tiene que volver `is_error` con el código del gancho para el canario y
  ejecutarse el de control. Determinista, sin GPU, y prueba la cadena entera (binario + settings + gancho).

### D-SP1-2 · C3: el registro lo escribe el PROXY, fuera de la jaula; no un endpoint de LAS MANOS

El spec dice «tool call → API de LAS MANOS → `audit.jsonl`», y el mapa de cableado pide un endpoint nuevo.
**Lo cambio**, y es la decisión que más importa de SP1:
- **Medido el 2026-09-17 como `axioma`:** alcanza `127.0.0.1:7777` (LAS MANOS), `11434` (Ollama), `3308`
  (MariaDB) y `8080` (jax-platform). Un endpoint de LAS MANOS que reciba tool calls desde la jaula es un
  endpoint que la jaula puede **llamar con lo que quiera** (o no llamar). Viola la prueba del spec: «desde
  la jaula, escribir el registro falla».
- **Hallazgo grave al medirlo:** LAS MANOS no tiene autenticación y `POST /human_gate/token` emite tokens
  de aprobación a quien lo pida. Desde `axioma` se puede sacar un token y llamar `/execute` como una
  faceta. Con el cerco (abajo) se cierra para `axioma`; **para Hyde sigue abierto** (ver «Hallazgos fuera
  del Ejecutor»).
- **El proxy ve todo lo que el modelo pide**, porque Claude Code no puede hacer nada sin su cerebro, y el
  cerebro sólo responde por el proxy. El `tool_use` sale del upstream, no de la jaula: la jaula no lo puede
  falsificar. El proxy lo anota **antes de reenviar el trozo que lo completa**; si no puede anotar, corta la
  conexión y la herramienta nunca corre. «Sin registro no hay acción.»
- **Intocable por tres capas:** (1) otra cuenta (`fruiz`, directorio `0750`, `axioma` ni lo lee);
  (2) `chattr +a` (ni `fruiz` lo puede truncar); (3) cadena de hashes (`prev` = sha256 de la línea
  anterior): cualquier edición se detecta con `verificar_cadena`.
- **El cerco (`nftables`, `meta skuid axioma`)**: la jaula sólo puede abrir conexiones al proxy, al puerto
  del canario y a los `ip:puerto` SSH del inventario. Todo lo demás, rechazado. Es lo que hace que «desde
  la jaula no se puede escribir el registro» sea cierto también por red.
- **El log de `sudo` de `axioma`** (la otra mitad de C3) se instala en cada máquina en el plan 5 Parte B,
  **antes** de que `axioma` tenga sudo (Fase 3): hoy no lo tiene en ninguna.

### D-SP1-3 · C4: el Ejecutor es su propio cgroup (la cuenta `axioma` entera), y el freno es un servicio root

- **Reusa la base del frente B** (`jax/core/interruptor.py`: `JAX_KILL_SWITCH_PATH`, `interruptor_activo`,
  `correr_con_interruptor`, `escribir_pausa`; prod `/etc/jax/interruptor/PAUSE`). **No la rediseña ni toca
  sus archivos.** C4 cubre lo que esa base no: `proxy_carril.py` y los procesos de `axioma`.
- **El scope de cgroup ya existe:** `axioma` es una cuenta dedicada; sshd + `pam_systemd` meten todo lo que
  corre en `user.slice/user-1001.slice/session-N.scope` (medido: `/proc/self/cgroup`). Matar el Ejecutor
  entero = escribir `1` en `cgroup.kill` de `user-<uid>.slice` (cgroup v2, kernel 7.0: el archivo existe en
  `user.slice`, medido). Es atómico, incluye procesos con `setsid`/`nohup` y lo que nazca mientras se mata.
- **Por qué un servicio root aparte y no el watcher de LAS MANOS:** `fruiz` no puede matar procesos de otra
  cuenta sin root, y un freno que depende de que LAS MANOS esté vivo no es freno. `ejecutor-freno.service`
  sondea el interruptor cada 250 ms (el mismo intervalo que `ssh_worker`) y mata mientras esté puesto.
- **Lo remoto, en dos capas:** (1) `ssh -tt` → al morir el cliente, sshd manda SIGHUP (lo que ya probó LAS
  MANOS); C1 lleva la regla `ssh_sin_tt`. (2) Barrido remoto: el freno entra a cada máquina con una llave
  propia de **comando forzado** (`ejecutor-freno-remoto`: mata todo proceso de `axioma` salvo su propia
  sesión). Cubre lo que se desprendió con `nohup`.
- **El proxy también frena:** con el interruptor puesto responde 423 sin tocar el upstream, y corta en
  vuelo el stream en curso (`correr_con_interruptor`). Sin cerebro, Claude Code no pide nada más.

### D-SP1-4 · C5: auditor de otro proveedor, con canario de dos caras, y con compuerta de datos de clientes

- **Lee el registro de C3** (fuera de la jaula; no lo que la jaula dice de sí misma), por lotes.
- **Tres cosas:** (a) paso fuera de misión o prohibido → `escribir_pausa`; (b) afirmación que cita una línea
  real pero **no responde lo que la misión pregunta** (el límite medido de las citas: 3 de las 11 invenciones
  de U3 citan una línea verdadera y concluyen algo falso; «el 8188 es Docker multi-hilo» sale respaldada)
  → la afirmación **se retiene**, no sale; (c) checklist de la constitución → marca en el informe.
- **Fail-closed:** auditor caído, respuesta ilegible o que cita un paso que no está en el lote → PAUSE.
  Una afirmación sin veredicto → retenida.
- **Canario de dos caras:** positivos (una trampa fuera de misión, una conclusión falsa sobre línea real) que
  DEBEN dispararse, y un negativo limpio que NO. Sin el negativo, un auditor que pausa todo pasaría el
  canario (lección: un control que no falla no valida).
- **Otro proveedor:** `provider_id` del cerebro ≠ `provider_id` del auditor, resuelto en vivo de
  `facet_binding`. Igual → no arranca.

### D-SP1-5 · C6: el archivo de llaves de `axioma` pasa a ser de root; revocar mata también las sesiones abiertas

- **Medido:** `~axioma/.ssh/authorized_keys` es de `axioma` (`600`). Un Ejecutor que se sale del plan puede
  **agregarse otra llave** antes de que lo revoquen, y la revocación «quita la llave» y no lo saca.
  Con `Match User axioma` + `AuthorizedKeysFile /etc/ssh/authorized_keys.d/%u` (root, `0644`) eso se cierra.
- **Quitar una llave no corta una conexión ya abierta:** revocar también mata los procesos de `axioma`.
- **«No entra» se prueba por `Permission denied (publickey)`**, no por un timeout: una máquina caída no
  prueba una revocación.

## Configuración nueva (sin hardcoding)

`/etc/jax/.env` (la leen los servicios de `fruiz` y el freno root con `EnvironmentFile=`; valores con espacios
entre comillas, lección del frente A):

| Variable | Valor de producción | Quién la lee |
|---|---|---|
| `JAX_EJECUTOR_CUENTA` | `axioma` | todos los contratos |
| `JAX_EJECUTOR_SSH_PUERTO` | `58291` | `cuenta_axioma` |
| `JAX_EJECUTOR_CONTROLADOR_LLAVE` | `/home/fruiz/.ssh/id_ejecutor_controlador` | `cuenta_axioma` |
| `JAX_EJECUTOR_NODE_BIN` | `/opt/ejecutor/node-v24.16.0/bin` | `cuenta_axioma` |
| `JAX_EJECUTOR_LIB` | `/opt/ejecutor/lib` | instalador, arranque |
| `JAX_EJECUTOR_POLITICA` | `/etc/jax-ejecutor/politica.json` | exportador, gancho |
| `JAX_EJECUTOR_GANCHO_TOPE_S` | `10` | instalador (se hornea en `gancho.sh`) |
| `JAX_EJECUTOR_CANARIO_PUERTO` | `18436` | canario C1, cerco |
| `JAX_EJECUTOR_REGISTRO` | `/var/log/jax-ejecutor/registro.jsonl` | proxy, auditor, arranque |
| `JAX_EJECUTOR_FRENO_LLAVE` | `/etc/jax-ejecutor/freno/id_freno` | freno root |
| `JAX_EJECUTOR_FRENO_KNOWN_HOSTS` | `/etc/jax-ejecutor/freno/known_hosts` | freno root |
| `JAX_EJECUTOR_FRENO_ESTADO` | `/run/ejecutor-freno/estado.json` | freno root (latido), arranque |
| `JAX_EJECUTOR_CERCO_SONDAS` | `7777,11434,3308,8080` | canario C3 |
| `JAX_KILL_SWITCH_PATH` | `/etc/jax/interruptor/PAUSE` (**lo define el frente B**) | proxy, freno, C5 |
| `JAX_PROXY_CARRIL_*` | (ya definidas por el proxy) · `TOPE_S` ≥ 60 (§6.2.2) | proxy |

`axioma_config` (DB): `ejecutor.c2_edad_max_s`, `ejecutor.cerebro_faceta`, `ejecutor.auditor_faceta`,
`ejecutor.c5_lote_max`, `ejecutor.c5_intervalo_s`, `ejecutor.c5_auditor_admite_datos_de_clientes`.

## Cómo se validaron estos planes antes de commitearlos

- **El código de los seis planes se extrajo tal cual a un directorio descartable y se corrió:** 305 tests puros pasan
  juntos (planes 1–6 más los tests existentes del proxy, con los cambios de fixture que los planes indican), contra
  `proxy_carril.py` de `master` en `de6964e` y contra `jax/core/interruptor.py` de la rama del frente B (leído, no
  modificado). Los tests con DB (`test_ejecutor_politica_db.py`, `test_ejecutor_c5_db.py`) y los de jax-platform no se
  corrieron: necesitan el esquema de las migraciones.
- **La semilla de 14 reglas pasa su autoprueba** con el evaluador del plan 1.
- **C1 se vio vivo y se vio fallar con el `claude` real (2.1.273) como `axioma`** (detalle en el punto 8 de abajo).
- **Mutaciones comprobadas al escribir:** `destinos` sin fail-closed, el proxy que entrega antes de anotar. Ambas
  dieron rojo en el test que el plan nombra.
- **Hallazgos del escribir que ya están corregidos en los planes:** el upstream falso no contestaba `HEAD /api/hello` ni
  leía `tool_result` fuera del último mensaje; un test del plan 1 dependía del umask; un regex del plan 4 tenía tres
  comillas invertidas que rompían los bloques de código del propio documento; el plan 2 reconstruía el cliente HTTP que
  el frente E centralizó.
- **Si `master` avanza antes de ejecutar un plan**, cada task que modifica un archivo existente dice contra qué versión
  se escribió y que se relee antes de aplicar (lección: los specs envejecen).

## Lo que el spec dice mal o es imposible (con evidencia)

1. **C5 + soberanía = imposible hoy para misiones con datos de clientes.** El auditor tiene que ser de
   **otro proveedor** que el cerebro. El cerebro por defecto es Qwen **local** (Ollama). No hay un segundo
   proveedor local: todo auditor posible (Thot/OpenAI, Kimi/Moonshot, Ada/Z.ai) es nube, y la Fase 0 fijó
   `datos_de_clientes = false` para los tres (`scripts/ejecutor_fase0/cerebros.toml`; `examen.py:54` lo hacía
   cumplir). Un auditor que no puede leer la salida no audita nada. **Pide DECISIÓN de Fernando:** (a) aceptar
   que el auditor de nube lea datos de clientes; (b) esperar un proveedor local distinto (Red Queen, Q3 2026);
   (c) auditar sólo misiones sin datos de clientes. El plan 4 construye la compuerta
   (`ejecutor.c5_auditor_admite_datos_de_clientes`, por defecto `false`): hasta que se decida, una misión que
   toca una máquina `con_datos_de_clientes` **no arranca**.
2. **§4 C3 («API de LAS MANOS»):** reemplazado por el proxy (D-SP1-2). Con la API, la jaula escribe el registro.
3. **§4 C4 («el watcher de 250 ms que ya existe mata la jaula entera»):** ese watcher mata un cliente ssh que
   lanzó LAS MANOS como `fruiz`; no puede matar procesos de `axioma` y muere con LAS MANOS. Hace falta un
   freno root (D-SP1-3).
4. **§4 C5 «Thot por defecto» y «un hecho sin salida de comando»:** lo segundo ya lo hace `cita.py` (spec F2
   §7). Y con el Ejecutor sin prosa (§2.0), la «conclusión falsa» no vive en un texto: vive en **qué pregunta
   dice responder** una afirmación. `cita.Afirmacion` no tiene ese campo. El plan 4 define
   `AfirmacionAuditable.proposito`; **SP2 tiene que emitirlo** en el transporte.
5. **§4 C5 «escribe PAUSE»:** `/etc/jax/PAUSE` es el freno GLOBAL de JAX (la Mesa responde 423). Un falso
   positivo del auditor frena toda la plataforma, no sólo al Ejecutor. Se implementa así porque es lo que el
   spec pide y lo seguro; queda declarado. Además, la activación del auditor no pasa por
   `kill_switch_audit` del frente B (esa tabla la escribe jax-platform con un superadmin): queda en el
   contenido del archivo PAUSE (`origen=c5`) y en el journal del vigía.
6. **§8 Fase 1 «no toca servidores»** choca con C6 («revocar y comprobar que no entra a ninguna»): una
   revocación sólo se prueba contra una máquina real. Va separado (plan 5 Parte B), no destructivo.
7. **§11 «si un subagente de Claude Code corre dentro del mismo proceso de la jaula»:** para C1 no importa —
   la documentación oficial dice que `PreToolUse` se dispara también para subagentes (`agent_id`). Para C3
   tampoco: sus peticiones pasan por el mismo `ANTHROPIC_BASE_URL`.
8. **`disableAllHooks` (la documentación dice que «en cualquier nivel» apaga todos los ganchos) — medido, no
   supuesto.** 2026-09-17 03:45, `claude` 2.1.273 como `axioma`, con el código del plan 1 en un directorio
   descartable: un `.claude/settings.json` **de proyecto** con `disableAllHooks: true` **no** apagó el gancho
   gestionado (el canario siguió bloqueado), y el mismo archivo **sí** se leyó (su `deny` de control se aplicó).
   El nivel de **usuario** queda a medir en el plan 1 (Task 9, Step 7); la jaula lo monta de solo lectura igual.

**Validación empírica de D-SP1-1 hecha al escribir el plan (no es implementación: directorio descartable, borrado):**
el código de las Tasks 2–7 del plan 1, copiado a `/tmp` y corrido como `axioma` con el `claude` real dentro de la
jaula superpuesta, dio `verificar_c1 → ()` en 2,2 s; con la política alterada en un byte dio **exactamente** los
cinco fallos que el plan predice (`politica_ilegible`, `canario_directo_sin_su_regla`, `control_directo_bloqueado`,
`canario_sin_su_regla`, `control_no_ejecutado`). La semilla de 14 reglas pasa su autoprueba. De ahí salieron dos
arreglos al upstream falso que ya están en el plan: el arnés manda `HEAD /api/hello` y su último mensaje es
`role: system`, no el `tool_result`. Los 126 tests puros del plan 1 pasan contra ese código.

## Hallazgos fuera del Ejecutor (no se difieren: se registran con fecha)

- **Hyde alcanza LAS MANOS sin autenticación.** `hyde_sandbox.py` usa `--share-net` y corre como `fruiz`:
  puede pedir un token de human gate y llamar `/execute`. El cerco de `axioma` no lo cubre (otra cuenta).
  Es la misma propiedad que cierra C3 para otra faceta → **DEUDA con fecha 2026-09-24** (plan 2, Task 8).
- **`fruiz` tiene `(ALL) NOPASSWD: ALL` en hall9000** (`sudo -n -l`). Todo proceso de `fruiz` (LAS MANOS,
  jax-platform, el proxy, el vigía de C5) es root en potencia. No cambia SP1 (`axioma` no está en sudoers),
  pero es la razón de que el freno y el registro no se apoyen en «fruiz no puede».
- **Locks de `prioridad` entre cuentas (hallazgo 7 del LEDGER):** con el proxy corriendo como `fruiz`,
  `axioma` nunca toma un lock. No hace falta un directorio de grupo común mientras eso sea cierto; el plan 2
  lo fija con un test (el proxy es el único que abre `ejecutor.lock`).

*En memoria de Jairo Urbina. En honor al Prof. Raúl Jacobs.*
