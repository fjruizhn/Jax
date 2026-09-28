# El Ejecutor programa — misiones de CÓDIGO (diseño)

**Fecha:** 2026-09-28 · **Aprobado por:** Fernando (diseño en chat, secciones 1–4, 2026-09-28) ·
**Redactó:** Mr. Hyde · **Repos:** `jax` (LAS MANOS, jaula, contratos) y `jax-platform` (modo
Ejecutor, API, i18n) · **Antecedente:** `2026-09-15-ejecutor-design.md` (el Ejecutor de servidores),
cuyas decisiones siguen vigentes salvo lo que este documento enmienda explícitamente (§2).

## 0. Historial de este documento

| Versión | Commit | Qué cambió |
|---|---|---|
| v1 | (este commit) | Diseño original, aprobado por secciones en chat. |

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
 [1] PREPARAR   LAS MANOS · jaxsvc · fuera de la jaula
   │  clon + rama axioma/<misión> + dependencias desde lockfiles
   ▼
 [2] TRABAJAR   Qwen · jaula `codigo` · sin red salvo el proxy del cerebro · sin credenciales
   │  editar · probar · git commit local       (C1 gancho · C3 registro · C4 pausa · C5 auditor)
   ▼
 [3] ENTREGAR   LAS MANOS · jaxsvc · fuera de la jaula · ÚNICA pieza con el token
      revisa el diff contra C1 · empuja SOLO axioma/<misión> · abre el PR con el informe de C5
```

### 3.1 Preparar

- Clona o actualiza el repo en `$JAX_EJECUTOR_CODIGO_DIR/<repo>/<misión>` (ruta en config; valor
  propuesto `/srv/jax-data/ejecutor/codigo`), crea `axioma/<misión>` desde la rama por omisión del
  remoto (leída de la API, nunca asumida `main`/`master`).
- Instala dependencias **solo desde archivos de bloqueo** del repo (`requirements*.txt`,
  `package-lock.json`, `composer.lock`); sin lockfile, no instala y lo declara en el informe.
- Nunca copia un `.env` real. Si el repo trae `.env.example`, la misión lo ve como ejemplo.
- La clonación usa el token por un *credential helper* de un solo uso, nunca en la URL ni en
  `.git/config` del clon (el clon entra a la jaula).

### 3.2 Trabajar (la jaula `codigo`)

Perfil nuevo en `hyde_sandbox.py`, mismo punto único de entrada (`run_sandboxed_claude()`; el control
de CI `no-naked-claude-subprocess` sigue vigente).

| Monta / permite | Modo | Por qué |
|---|---|---|
| El clon de la misión | rw | su único lugar de trabajo |
| Dependencias instaladas en la preparación | ro | la jaula no tiene internet |
| Plugins/skills fijados del Ejecutor | ro | igual que el perfil `ejecutor` |
| `CLAUDE.md` generado (constitución + reglas del repo + `CLAUDE.md`/`CONTEXT.md` del propio repo) | ro | §6 del spec del 2026-09-15 |
| Red | **solo el proxy del cerebro** | ni GitHub, ni MariaDB 3308, ni otras máquinas |
| Token de GitHub, `/etc/jax/.env`, llaves SSH, credenciales de Anthropic | **no se montan** | el modelo nunca las tiene |

Herramientas: Bash, lectura/edición, `git` local (commit, diff, log). `git push` dentro de la jaula
falla por red **y** por C1 (dos barreras).

### 3.3 Entregar

1. Lee el diff completo `origin/<rama por omisión>...axioma/<misión>` **fuera de la jaula** y lo pasa
   por C1 de código (§5.1) en su forma estructural (lista de rutas y hunks, no regex sobre texto
   plano). Si algo choca: no hay push, la misión termina **«rechazada por contrato»** con qué y dónde.
2. Barrido de secretos (el mismo de los respaldos) y tope de tamaño por archivo (5 MB, configurable).
3. Exige el **informe firmado por C5** (§5.3). Sin informe: no hay PR (DC8).
4. `git push --force-with-lease` **solo** a `refs/heads/axioma/<misión>`. La función de entrega
   rechaza cualquier otra referencia antes de llamar a git, y rechaza la rama por omisión aunque el
   nombre coincidiera (defensa en profundidad para repos sin protección de rama).
5. Abre o actualiza el PR (API de GitHub), **listo para revisión**, con etiqueta `axioma` y pie
   `Hecho-por: Axioma (Ejecutor, misión <id>, cerebro <modelo resuelto>)`.

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
  entrega (`abierto` | `rechazada_por_contrato` | `sin_informe_c5` | `fallo_entrega`).
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

## 10. Fuera de alcance

- Integrar o desplegar (siempre de Fernando).
- Misiones de código que además operen servidores.
- Otros cerebros para código (Kimi/GLM quedan para después, si Qwen no alcanza).

## 11. A verificar contra el código vivo antes de planificar

- Cómo resuelve hoy `hyde_sandbox.py` el perfil `ejecutor` y qué hace falta para uno `codigo` sin red
  general.
- Si el proxy del carril admite sesiones largas de edición (contexto de 131k).
- Columnas reales de `ejecutor_mision` y `ejecutor_regla` en `jax_memory` (esquema vivo, no el archivo).
- Nombre y formato del token de grano fino de GitHub vigente.

*En memoria de Jairo Urbina. En honor al Prof. Raúl Jacobs.*
