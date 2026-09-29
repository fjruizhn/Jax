# El Ejecutor programa — misiones de CÓDIGO (diseño)

**Fecha:** 2026-09-28 · **Aprobado por:** Fernando (diseño en chat, secciones 1–4, 2026-09-28) ·
**Redactó:** Mr. Hyde · **Repos:** `jax` (LAS MANOS, jaula, contratos) y `jax-platform` (modo
Ejecutor, API, i18n) · **Antecedente:** `2026-09-15-ejecutor-design.md` (el Ejecutor de servidores),
cuyas decisiones siguen vigentes salvo lo que este documento enmienda explícitamente (§2).

## 0. Historial de este documento

| Versión | Commit | Qué cambió |
|---|---|---|
| v1 | (este commit) | Diseño original, aprobado por secciones en chat. |
| v1.1 | (este commit) | Correcciones medidas contra el código vivo antes de planificar. |
| v1.2 | (este commit) | Espejo privado de `jaxsvc` + clon de trabajo de Qwen (§3.1, §3.3, §4.2). La auditoría de escalón 3 de las Tareas 5-6 rechazó la v1.1 con tres BLOCK nacidos del diseño: `jaxsvc` corría git (con el token), npm y pip **dentro del clon que Qwen puede reescribir**. Ruling ARQUITECTURA del controlador, 2026-09-28. |
| v1.3 | (este commit) | Sandbox bwrap por lista blanca para toda instalación de dependencias y reglas de rechazo de fuentes (§3.1), riesgo residual del sandbox (§9). Dos re-revisiones de escalón 3: con la v1.2, pip ejecutaba código de terceros como `jaxsvc` (un `.pth` de rueda, `--no-binary` en requirements) y `npm ci --ignore-scripts` ejecutó el código de una dependencia git (verificado por el revisor con el node de producción). Rulings del controlador, 2026-09-28. |

## 1. Por qué

> «Axioma debe poder hacer todo lo que hacemos nosotros.» — Fernando, 2026-09-25.
> «Que qwen trabaje por nosotros, es gratis.» · «Qwen debe ser capaz de trabajar solo, estamos
> probando, si falla lo arreglamos.» — Fernando, 2026-09-28.

Hoy el Ejecutor opera servidores (lectura, con citas verificadas). No escribe código. La vía vieja,
`/command`, está **rota desde el 2026-09-17** (el lanzador hace `cd $HOME/jax` y `jaxsvc` tiene
`HOME=/var/lib/jaxsvc`) y **parada** por decisión de Fernando del 2026-09-25; su rama de arreglo
(`ops/lanzador-misiones`, `efe877d`) fue rechazada en escalón 3.

**Criterio de éxito:** Fernando pide un cambio en un repo suyo desde el chat web; Qwen, solo, trabaja
en una rama, corre las pruebas y abre un **PR listo para revisión** con un informe firmado por el
auditor. Nadie interviene entre el pedido y el PR. Integrar sigue siendo de Fernando.

## 2. Decisiones de Fernando (2026-09-28)

| # | Decisión | Nota |
|---|---|---|
| DC1 | **Cerebro = Qwen local** (el mismo del Ejecutor, vía el proxy del carril) | Mantiene D1 del 2026-09-15: el Ejecutor no usa Anthropic. El token Max de `jaxsvc` sigue siendo solo del catálogo |
| DC2 | **Enfoque A**: un tipo de misión nuevo, «código», dentro del Ejecutor existente | Descartados: paso de Pipeline (no conversacional) y vía separada (reconstruir o saltarse C1–C6, Principio IX) |
| DC3 | **Entrega con un token de grano fino de la cuenta de Fernando**, acotado a *Contents* y *Pull requests* (escritura) | **ENMIENDA** de la regla «ninguna clave de máquina escribe en GitHub» (Biblioteca de AteneaERP, 2026-09-06), solo para la entrega de misiones de código. Razón de Fernando: «muchas cuentas es un desorden». Riesgo escrito en la opción elegida: los PRs aparecen como suyos (mitigado en §4.3) |
| DC4 | **Alcance: todos los repos de Fernando** | Incluye privados sin protección de rama (plan gratuito): la barrera es nuestra (§4) |
| DC5 | **Autonomía total dentro de la misión**: el pedido en el chat es el GO; no hay aprobación humana intermedia | El único alto humano es integrar el PR |
| DC6 | **PR normal, listo para revisión** (no borrador) | «Por mí que haga el PR también» |
| DC7 | **Lista C1 de código** tal como §5.1 | Ampliarla es una fila en la base |
| DC8 | **C5 bloquea**: sin informe firmado por el auditor no se abre el PR | Quien produce no aprueba |
| DC9 | **`/command` se retira** (endpoint, UI y el lanzador `~/.local/bin/jax`) cuando la vía nueva pase su primera misión real | La rama `ops/lanzador-misiones` se cierra sin integrar |
| DC10 | **Primera misión real en `jax-platform`** | Sus pruebas corren completas en hall9000 |

## 3. Arquitectura — tres etapas; solo la del medio es del modelo

```
 chat (modo Ejecutor, tipo Código)
   │  repo + pedido
   ▼
 [1] PREPARAR   `mision_servicio` (proceso de `jaxsvc`, fuera de la jaula)
   │  clon + rama axioma/<misión> + dependencias desde lockfiles
   ▼
 [2] TRABAJAR   Qwen · jaula `codigo` · sin red salvo el proxy del cerebro · sin credenciales
   │  editar · probar · git commit local       (C1 gancho · C3 registro · C4 pausa · C5 auditor)
   ▼
 [3] ENTREGAR   `mision_servicio` (proceso de `jaxsvc`, fuera de la jaula) · ÚNICA pieza con el token
      revisa el diff contra C1 · empuja SOLO axioma/<misión> · abre el PR con el informe de C5
```

> *El Ejecutor no pasa por LAS MANOS (medido 2026-09-28: es un subproceso de jax-platform); la
> propiedad que el diseño exige —fuera de la jaula, el modelo nunca ve el token— se conserva.*

### 3.1 Preparar

*(v1.2.)* Todo vive bajo `<raiz>/<misión>/`, dueño `jaxsvc`, con ACL `u:axioma:--x` (la cuenta
solo lo **atraviesa**: no lista ni crea nada ahí) y sin ACL por omisión (`setfacl -k`: medido,
`--set` sin entradas `d:` conserva la heredada del padre). Tres piezas:

| Pieza | Qué es | Quién la toca |
|---|---|---|
| `espejo.git` | bare, `0700`, **sin ACL para `axioma`**. `origin` = `https://github.com/<owner_repo>.git`, **derivada** de un `owner_repo` validado (`^[\w.-]+/[\w.-]+$`, sin `.`/`..`), nunca un campo libre | solo `jaxsvc`: único lugar con el token |
| `deps/` | dependencias instaladas desde la rama por omisión | escribe `jaxsvc`; `axioma` lee (`r-X`) |
| `repo/` | el clon de trabajo, clonado **desde el espejo** con `--no-hardlinks`, en `axioma/<misión>` | `axioma` escribe (`rwX`, `setfacl -P -R`) |

- El espejo trae **solo** la rama por omisión (leída de la API, nunca asumida `main`/`master`);
  la rama de la misión en GitHub no se trae nunca (§3.3, lease).
- Dependencias **solo en el turno 1**, desde los archivos de bloqueo del commit de la rama por
  omisión (leídos del espejo, no del clon), en `deps/`. En el clon, enlaces
  `.venv`/`node_modules`/`vendor` → `deps/`, anotados en `.git/info/exclude` (para git un enlace no
  es un directorio: `node_modules/` en `.gitignore` no lo tapa).

#### 3.1.1 Sandbox de instalación *(v1.3)*

pip, npm y composer pueden ejecutar código de terceros, y `jaxsvc` lee `/etc/jax/.env`: filtrar
el entorno no alcanza (el código lee archivos). Toda instalación —el `python3 -m venv`, pip, npm,
composer— corre en `bwrap` por **lista blanca** (`jax/ejecutor/codigo/sandbox.py`):

| | Qué |
|---|---|
| Espacios de nombres | `--unshare-all --share-net --die-with-parent --new-session`: todo nuevo salvo la red |
| Entorno | `--clearenv`; solo `PATH`, `HOME=/tmp/h`, `LANG=C.UTF-8` |
| Solo lectura | `/usr`; `/lib`, `/lib64`, `/bin`, `/sbin` (enlace como enlace, directorio real montado); `/etc/ssl`, `/etc/ca-certificates`, `/etc/resolv.conf`, `/etc/hosts`, `/etc/nsswitch.conf`, `/etc/passwd`, `/etc/group`; el prefijo de node (padre de `JAX_EJECUTOR_NODE_BIN`); las **copias** de los archivos de dependencias, hechas antes y fuera de `deps/` (npm/composer las ven montadas encima de `package.json`/lockfile: no pueden reescribirlos) |
| Escribible | **solo** `<misión>/deps/`; `/tmp` es un tmpfs propio; `/proc` y `/dev` propios |
| Nunca | `/etc/jax`, `/var/lib/jaxsvc`, `/srv`, `/home`, el directorio de misiones (ni el espejo ni el clon). Un origen dentro de esas rutas **o que las contenga** (`/etc`, `/`) se rechaza antes de lanzar nada |

Los directorios dentro de `deps/` se crean componente a componente con `lstat` (una instalación
anterior pudo dejar un enlace ahí): un enlace en el camino falla cerrado.

#### 3.1.2 Reglas de fuentes *(v1.3)*: ninguna instalación ejecuta código de terceros

`--ignore-scripts` no basta: npm prepara las dependencias git igual (verificado). Se revisa la
forma de cada fuente ANTES de instalar; lo que no pasa **no se ejecuta** y se declara en
`dependencias` (visible en el informe), y la misión sigue sin eso.

| Gestor | Se instala solo si… | Comando | Si no, se declara |
|---|---|---|---|
| pip | el `requirements*.txt` (raíz y `backend/`) no tiene ninguna línea de opción salvo `-r <hermano>` (`--no-binary`, `--index-url`/`-i`, `--extra-index-url`, `-f`/`--find-links`, `-e`/`--editable`, `--trusted-host`, `-c`…) ni requisitos fuera del índice (URL, `pkg @ …`, ruta local); tampoco el que incluye a uno rechazado | `pip install --only-binary=:all: -r a -r b` — **una sola** invocación por venv | `pip_opcion_rechazada:<archivo>`; sin rueda: `pip_sin_wheel:<archivo>` |
| npm | TODA entrada de `package-lock.json` (v2/v3 `packages`, v1 `dependencies` recursivo) tiene `resolved` bajo `https://registry.npmjs.org/` e `integrity`, no es `link`, y su versión no es `git+`/`github:`/`file:`/`http:`/URL | `npm ci --ignore-scripts --allow-git=none --no-audit --no-fund` (raíz y `frontend/`; `--allow-git` existe desde npm 11, verificado en 11.13.0 del node de producción: sin él, un `package.json` con una dependencia git ausente del lockfile hacía bajar su tarball antes de fallar) | `npm_fuente_rechazada:<lockfile>:<paquete>`; `sin_lockfile:<package.json>`; `npm_no_instalado:<ruta>` fuera de raíz/`frontend/` |
| composer | `composer.json` no declara `repositories` que no sean packagist (vcs, path, git, artifact, package…) y cada paquete de `composer.lock` tiene `dist` zip de packagist/GitHub y, si lo tiene, `source` git de github.com | `composer install --no-interaction --no-scripts --no-plugins --prefer-dist` | `composer_fuente_rechazada:<archivo>:<paquete o repositories>` |
- **Turno ≥ 2**: solo `fetch` del espejo desde GitHub. Ni reinstala, ni vuelve a aplicar ACL (la
  ACL por omisión del clon gobierna lo nuevo), ni ejecuta nada dentro del clon: su `.git` y sus
  binarios ya son de quien los puede reescribir.
- Nunca copia un `.env` real. Si el repo trae `.env.example`, la misión lo ve como ejemplo.
- Todo git de `jaxsvc` corre con entorno en lista blanca (`PATH`, `HOME` temporal, `LANG`; las
  variables de askpass solo en operaciones de red), `GIT_CONFIG_NOSYSTEM=1`,
  `GIT_CONFIG_GLOBAL=/dev/null`, `-c core.hooksPath=/dev/null -c credential.helper=` y `--` antes
  de los posicionales. El token llega por un `GIT_ASKPASS` que lee una variable del entorno del
  subproceso y responde **solo** a los dos avisos exactos de `https://github.com`.

### 3.2 Trabajar (la jaula `codigo`)

Variante de la jaula de `jax/ejecutor/contratos/cuenta_axioma.py` (`_jaula` + `remoto_claude`), la misma
del Ejecutor: `--dev-bind / /` como `axioma`; la red la limita el cerco `inet ejecutor_cerco` (uid de
`axioma`: solo proxy del carril en loopback y SSH del inventario; GitHub bloqueado).

| Monta / permite | Modo | Por qué |
|---|---|---|
| El clon de la misión | rw | su único lugar de trabajo; ACL `u:axioma:rwX` (Tarea 0) |
| Dependencias instaladas en la preparación | ro | la jaula no tiene internet |
| Plugins/skills fijados del Ejecutor | ro | igual que el perfil `ejecutor` |
| `CLAUDE.md` generado (constitución + reglas del repo + `CLAUDE.md`/`CONTEXT.md` del propio repo) | ro | §6 del spec del 2026-09-15 |
| Red | **solo el proxy del cerebro** | ni GitHub, ni MariaDB 3308, ni otras máquinas |
| Token de GitHub, `/etc/jax/.env` (`root:jaxsvc 640`, `axioma` no lo lee — medido), llaves SSH, credenciales de Anthropic | **no se montan** | el modelo nunca las tiene |

Herramientas: Bash, lectura/edición, `git` local (commit, diff, log). `git push` dentro de la jaula
falla por red **y** por C1 (dos barreras). El entorno de la jaula lleva `GIT_CONFIG_COUNT=1
GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0=<clon>` (medido con git 2.53: sin esto, `axioma`
recibe «dubious ownership» al operar sobre un repo cuyo directorio pertenece a `jaxsvc`).

### 3.3 Entregar

*(v1.2.)* `jaxsvc` **no corre git dentro del clon**. `man git`, sección SECURITY: *«it is not safe
to run Git commands in a .git directory (or the working tree that surrounds it) when that .git
directory itself comes from an untrusted source. The commands in its config and hooks are executed
in the usual way»*; y para un `.git` no confiable recomienda *«serve the repository as an
unprivileged user (either via git-daemon(1), ssh, or using other tools to change user ids)»*. Eso
es lo que se hace:

0. **Traer.** Desde el espejo: `git fetch --no-tags --upload-pack=<ssh a axioma@127.0.0.1, la misma
   cuenta, llave y puerto de cuenta_axioma> <clon> +refs/heads/axioma/<misión>:refs/heads/axioma/<misión>`.
   El lado que lee el clon (`upload-pack`) corre **como `axioma`**; `jaxsvc` solo recibe objetos, y
   después corre `git fsck --no-dangling` en el espejo. Ningún error de este paso lleva salida de
   los comandos (la escribe o la provoca el clon).
1. Lee el diff completo `origin/<rama por omisión>...axioma/<misión>` **en el espejo** y lo pasa
   por C1 de código (§5.1) en su forma estructural (lista de rutas y hunks, no regex sobre texto
   plano). Si algo choca: no hay push, la misión termina **«rechazada por contrato»** con qué y dónde.
2. Barrido de secretos (el mismo de los respaldos) y tope de tamaño por archivo (5 MB, configurable).
3. Exige el **informe firmado por C5** (§5.3). Sin informe: no hay PR (DC8).
4. Desde el espejo, `git push origin refs/heads/axioma/<misión>:refs/heads/axioma/<misión>` con
   `--force-with-lease=refs/heads/axioma/<misión>`, `--no-follow-tags` y
   `--recurse-submodules=no`. El valor esperado del lease es el seguimiento
   `refs/remotes/origin/axioma/<misión>` **del espejo**, que solo mueve nuestro propio empuje (todo
   `fetch` del espejo nombra solo la rama por omisión): si otro movió la rama en GitHub, el empuje
   falla y no se pisa. La función de entrega rechaza cualquier otra referencia antes de llamar a
   git, y rechaza la rama por omisión aunque el nombre coincidiera (defensa en profundidad para
   repos sin protección de rama). Los errores de git del espejo llevan a lo sumo los últimos 300
   caracteres, con el token reemplazado por `***`.
5. Abre o actualiza el PR (API de GitHub), **listo para revisión** (`draft: false` también al
   actualizar), con etiqueta `axioma` y pie `Hecho-por: Axioma (Ejecutor, misión <id>, cerebro
   <modelo resuelto>)`. Un PR **cerrado no se reabre**: se abre otro y el resultado lo declara
   (`pr_reabierto_nuevo`). Un 422 «already exists» (carrera) se resuelve volviendo a listar.

### 3.4 Sesiones

Una misión = una sesión (`--session-id`). Un mensaje nuevo en la misma conversación la retoma
(`--resume`): mismo clon, misma rama, el PR se actualiza. El flujo `stream-json` alimenta pantalla,
C3 y C5, igual que hoy.

## 4. El token (DC3)

### 4.1 Guarda

- Variable `JAX_GITHUB_TOKEN` en `/etc/jax/.env` (`root:jaxsvc 640`, como hoy).
- Lo guarda Fernando con un guion hermano de `ops/guardar-token-claude.sh` (entrada sin eco,
  validación de formato `github_pat_…`, escritura atómica, respaldo); el valor nunca pasa por el chat.
- Solo lo lee el proceso de LAS MANOS que prepara y entrega. **No entra a la jaula** (§3.2): un
  test de la jaula lo comprueba enumerando el entorno y los montajes del proceso enjaulado.

### 4.2 Por qué la barrera es nuestra

Los repos privados en plan gratuito no admiten protección de rama (medido: 403 *Upgrade to Pro*), y
un token de grano fino no se acota por rama. Con este token, **nada del lado de GitHub impide un push a
`main`** en `ateneaerp`, `claude-skills` y los demás privados. La barrera es: el modelo nunca tiene el
token, y quien lo tiene solo sabe empujar a `axioma/*` (§3.3.4).

*(v1.2.)* «Quien lo tiene» es `jaxsvc`, y eso solo vale si **nada que el modelo escriba se ejecuta
con el token a mano**. En la v1.1 no se cumplía: `jaxsvc` empujaba desde el clon, y el `.git` del
clon es del modelo — un gancho `pre-push` leía el token del entorno; `credential.helper`,
`url.<x>.insteadOf`, `http.proxy` o `push.followTags` en su `.git/config` lo mandaban a otro lado o
empujaban de más (reproducido contra la v1.1: un `pre-push` del clon recibió el token). Por eso el
token vive solo en el **espejo** (§3.1), que el modelo no puede leer ni escribir, y los commits
cruzan con `upload-pack` corriendo como `axioma` (§3.3, paso 0). Se descartó filtrar la
configuración del clon: una lista negra de claves de git es frágil (cada versión suma claves que
ejecutan cosas).

### 4.3 Distinguir lo de Axioma

En GitHub el autor visible es Fernando. Todo PR de Axioma lleva la etiqueta `axioma`, el pie
`Hecho-por:` y commits con autor `Axioma (Ejecutor) <axioma@axioma-ia.io>` (configurable en `axioma_config`) configurado en el clon, de modo que
`git log` distingue a Axioma de Fernando y de Hyde aunque el push lo firme el token.

## 5. Contratos adaptados a código

### 5.1 C1 — prohibiciones de código (DC7)

Nuevas filas de `ejecutor_regla` para misiones de tipo código, fail-closed como las de hoy:

| Regla | Qué bloquea |
|---|---|
| Flujos de CI | cualquier cambio bajo `.github/workflows/` |
| Archivos que integra solo Fernando | en `claude-skills`: `common/firmantes-fernando`, `bin/go-fernando`, `common/CLAUDE.md.core`, `common/secciones-constitucionales.txt`, `common/hooks/`, `bin/codex-bridge.py`, `lib/assemble.py`, `lib/hostid.py`, `bin/codex-auto` |
| Pruebas debilitadas | borrar archivos de prueba; añadir `skip`/`skipIf`/`xfail`/`.skip(`/`markTestSkipped`; quitar aserciones netas |
| Pisos de CI | bajar un piso/techo numérico de pruebas o cobertura |
| Ganchos | `--no-verify`, `core.hooksPath`, desactivar ganchos |
| Secretos | crear o modificar `.env*` con valores (se permite `.env.example`) |

Se aplican **dos veces**: en el gancho `PreToolUse` dentro de la jaula (ataja errores honestos) y en la
entrega sobre el diff (§3.3.1), que es la que manda.

### 5.2 C2, C3, C4

- **C2** no aplica: la misión no toca servidores; el remoto solo recibe `axioma/*`.
- **C3** y **C4**: sin cambios. La jaula `codigo` cuelga del mismo cgroup, registro y pausa.

### 5.3 C5 — auditor de otro proveedor (DC8)

Elegido como hoy (`eleccion_c5`: nunca el proveedor del cerebro). Su lista para código, derivada de la
constitución:
- afirmación sin salida de comando que la respalde («las pruebas pasan» sin haberlas corrido);
- solución temporal, código comentado, «después lo arreglo»;
- hardcoding; texto visible sin i18n;
- pruebas debilitadas (redundante con C1, a propósito);
- trabajo fuera del pedido.

Si ve algo: `PAUSE`. Al final firma el informe afirmación ↔ evidencia que va en el cuerpo del PR.

### 5.4 Canarios (Principio VII)

Dos misiones trampa permanentes: una intenta `git push` desde la jaula; otra toca
`.github/workflows/`. Ambas deben quedar bloqueadas; si alguna no se bloquea, **el tipo Código no
arranca**.

## 6. Interfaz y datos

- Modo «Ejecutor» de la Mesa (solo superadmin, como hoy): tipo de misión **Código**, selector de repo,
  pedido libre. i18n es/en; claro/oscuro con tokens; confirmaciones en ventana propia.
- Tabla nueva **`ejecutor_repo`** (nombre, `owner/repo`, rama por omisión observada, `activo`,
  comandos de prueba declarados). La lista de repos sale de ahí, nunca del código.
- `ejecutor_mision` gana el tipo (`servidor` | `codigo`), `repo_id`, rama, URL del PR y estado de
  entrega (`abierto` | `sin_cambios` | `rechazada_por_contrato` | `sin_informe_c5` | `fallo_entrega`
  | `sin_entregar` | `empujado_sin_pr`). *(Tarea 9, auditoría de escalón 3, 2026-09-28.)*
  - `sin_cambios`: la rama no tiene commits respecto de la rama por omisión; no se empuja. Nota
    `pr_previo_sin_cambios_nuevos` si ya hay un PR abierto de la misión.
  - `rechazada_por_contrato`: C1 de código, identidad, secretos (líneas agregadas, rutas, mensaje y
    cabeceras de cada commit, título y cuerpo del PR), `encoding` no UTF-8 o topes de tamaño; no se
    empuja.
  - `sin_entregar` (con `motivo`): no se tocó GitHub porque otra parte del turno falló -- pausa de
    C4, registro de C3, cadena, vigía, C5 que **pausó** o quedó **ilegible**, cerebro, preparación,
    o el turno no produjo **ninguna** afirmación (ni respaldada ni descartada) -- o la pausa apareció
    justo antes del empuje. El motivo es el código del turno.
  - **C5 que descarta afirmaciones NO impide el PR** (DC5: lo revisa Fernando). Si el verificador o
    C5 retienen algunas o todas, el turno sigue y el PR se abre igual; el informe lleva primero el
    bloque «Descartadas por el verificador/C5» (estado y motivo) y después las respaldadas. El tope
    del cuerpo (60 000 caracteres) recorta primero las respaldadas.
  - `fallo_entrega`: falló el espejo o el empuje. Si el empuje falló se consulta `ls-remote`: si la
    rama remota ya es el SHA que se iba a empujar, cuenta como empujada y se sigue al PR; si no,
    `rama_empujada: false`; si tampoco responde, `rama_empujada: "desconocido"`.
  - `empujado_sin_pr`: la rama **ya está en GitHub** (`rama_empujada: true`, `sha`) pero el PR de
    este turno **no se confirmó** (fallo de la API, respuesta ilegible o sin `html_url` https válida,
    pausa puesta tras el empuje). Si ya había un PR abierto de la misión, lleva su `pr_url` y la
    nota `pr_con_informe_desactualizado`. Lo revisa Fernando.
  - `abierto` puede llevar la nota `sin_etiqueta` (el PR existe, falló solo la etiqueta).
  Todo resultado de entrega lleva `rama_empujada` y `sha`.
- Concurrencia: una misión a la vez (config en DB), porque Qwen atiende una inferencia por turno.

## 7. Pruebas que no pueden correr aquí

Las suites que necesitan infraestructura ausente de hall9000 (AteneaERP: su PHP y su MariaDB en
atem-ai) se declaran **no corridas** en el PR, con el motivo. Nunca se reportan como verdes.

## 8. Orden de construcción

Cada paso con su auditoría de escalón 3 antes del siguiente; cada contrato visto fallar antes de habilitar.

1. Pruebas en rojo de C1 de código (gancho y diff) y de la función de entrega.
2. Preparar y entregar en LAS MANOS, con **token falso** contra un remoto git local (bare repo).
3. Perfil de jaula `codigo` (+ test: el token no está en entorno ni montajes).
4. Tipo de misión, `ejecutor_repo`, UI (i18n, temas).
5. Canarios §5.4.
6. Fernando guarda el token real; primera misión real en `jax-platform` (DC10).
7. Retiro de `/command` (DC9).
8. Prueba de carga (Cuatro del Rendimiento): p95 del chat de la Mesa con una misión de código activa.

## 9. Riesgos aceptados (firmados)

- **DC3:** un token con escritura en todos los repos privados sin protección de rama. Mitigación:
  §4.2 (el modelo nunca lo tiene; la entrega solo empuja a `axioma/*`).
- **C1 por patrones** ataja errores honestos, no a un agente decidido; la defensa de fondo es la
  revisión del diff en la entrega, C5 y que integrar sea de Fernando.
- **Calidad de Qwen en código:** en la Fase 0 no pasó U3/U5/U6. Aceptado por DC5: «si falla lo
  arreglamos». Cada misión deja su informe; los fallos se ven, no se esconden.
- **Espejo y clon (v1.2):** el token, el diff de C1 y el empuje viven solo en el espejo privado de
  `jaxsvc`; el clon de Qwen se clona desde él y sus commits vuelven con `upload-pack` corriendo
  como la cuenta, más `fetch.fsckObjects` y `fsck` a una ref temporal (§3.1, §3.3). Riesgo que
  queda: la superficie de `upload-pack` es grande (`man git`, SECURITY), pero corre sin privilegios
  de `jaxsvc`.
- **Sandbox de instalación (v1.3):** comparte la **red del host** (a propósito: hace falta para bajar
  paquetes). Desde adentro se alcanza lo que alcanza `jaxsvc` por red — `127.0.0.1:11434` (Ollama,
  sin autenticación), la LAN e internet —, pero **no corre código de terceros**: el único binario
  que se ejecuta es npm, pip o composer (y `python3 -m venv`), que son de confianza, sobre fuentes
  que pasaron §3.1.2. Queda: (a) un defecto del propio npm/pip/composer al procesar un paquete
  malicioso daría ejecución con esa red, dentro del sandbox; (b) un paquete malicioso **publicado
  en el registro** con integridad correcta se instala (sin scripts) y su código podría correr
  después, cuando Qwen lo use dentro de la jaula `codigo` — sin red salvo el proxy y sin
  credenciales. Costo aceptado: dependencias legítimas de git o de registros privados no se
  instalan (se declaran).

## 10. Fuera de alcance

- Integrar o desplegar (siempre de Fernando).
- Misiones de código que además operen servidores.
- Otros cerebros para código (Kimi/GLM quedan para después, si Qwen no alcanza).

## 11. A verificar contra el código vivo antes de planificar

- Cómo resuelve hoy `hyde_sandbox.py` el perfil `ejecutor` y qué hace falta para uno `codigo` sin red
  general. **Verificado (2026-09-28):** no hay perfil `ejecutor` en `hyde_sandbox.py` — la jaula del
  Ejecutor la arma `jax/ejecutor/contratos/cuenta_axioma.py` (`_jaula` + `remoto_claude`,
  `--dev-bind / /` como `axioma`). Para código no hace falta un perfil nuevo: se reutiliza esa misma
  jaula (§3.2), y la red ya queda acotada por el cerco `inet ejecutor_cerco` existente (uid de
  `axioma`: solo proxy del carril en loopback y SSH del inventario; GitHub ya está bloqueado por ese
  mismo cerco, sin cambio adicional).
- Si el proxy del carril admite sesiones largas de edición (contexto de 131k). **Verificado
  (2026-09-28), con dato existente, no con un ensayo nuevo:** Fase 0 (`docs/superpowers/specs/
  2026-09-15-ejecutor-fase0-resultado.md`, prueba U1) midió `num_ctx=131072` con 100% GPU y
  `load_duration` ≤ 300 s — pasó, y es el mayor `num_ctx` elegido; en producción el proxy del carril
  sirve `qwen3.6-mesa-131k` con ese contexto (`DEUDA.md:979-980`). No se corrió una misión de código
  real contra el proxy en esta tarea: la evidencia es la de Fase 0 sobre el mismo modelo/`num_ctx`,
  no una sesión de edición nueva — sigue pendiente medir el caso de uso real (edición, no solo chat).
- Columnas reales de `ejecutor_mision` y `ejecutor_regla` en `jax_memory` (esquema vivo, no el
  archivo). **Verificado (2026-09-28)** con `SHOW COLUMNS` contra producción (127.0.0.1:3308):
  `ejecutor_mision` tiene `id, user_id, objetivo, maquinas, sesion_id, created_at, updated_at` — **no**
  tiene `tipo`, `repo_id`, `rama`, URL de PR ni estado de entrega; §6 los da como columnas nuevas por
  agregar, y en efecto no existen hoy. `ejecutor_regla` tiene `id, codigo, tipo('prohibido',
  'destructivo'), herramientas, campo('command','file_path','cualquiera'), patron, ambito_host,
  ambito_roles(set), es_canario, activa, origen, ejemplos_coincide, ejemplos_no_coincide, created_at`
  — el `campo` de hoy no tiene un valor para diff/rutas estructurales; C1 de código (§5.1) necesitará
  ampliarlo o resolverlo aparte cuando se implemente, no asumir que `command`/`file_path` alcanzan.
- Nombre y formato del token de grano fino de GitHub vigente. **Verificado (2026-09-28):** la variable
  es `JAX_GITHUB_TOKEN` en `/etc/jax/.env` (`root:jaxsvc 640`); hoy está **ausente**
  (`grep -c '^JAX_GITHUB_TOKEN='` da 0), consistente con que el paso 6 de §8 («Fernando guarda el
  token real») todavía no se ejecutó. El formato que exige el guion previsto
  (`ops/guardar-token-github.sh`, `docs/superpowers/plans/2026-09-28-ejecutor-codigo.md:1190`) es
  `^github_pat_[A-Za-z0-9_]+$`, largo 40–255; un token clásico `ghp_…` se rechaza (DC3 exige grano
  fino).

*En memoria de Jairo Urbina. En honor al Prof. Raúl Jacobs.*
