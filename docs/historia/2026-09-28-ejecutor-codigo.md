# 2026-09-28 · El Ejecutor programa (modo Código) — implementación

**Quién:** Mr. Hyde (hall9000), con subagentes implementadores (sonnet/opus) y auditores de escalón 3.
**Decisiones:** Fernando, 2026-09-28 (DC1–DC10 en la spec). **PRs:** Jax#292, jax-platform#169. **No desplegado.**

## Qué se hizo
- Spec v1.3 y plan (17 tareas) en `docs/superpowers/`. Implementadas T0–T14; T15 (primera misión real) y
  T16 (retiro de `/command`) esperan despliegue y token.
- **T0, arreglo de producción:** desde el 2026-09-20 el Ejecutor no puede arrancar como `jaxsvc`
  (`preparar_directorio_projects` usaba `sudo install`; `jaxsvc` no tiene sudo). Último turno completado:
  2026-09-20 17:22. Arreglo: ACL fija (`setfacl --set`), lstat + dueño, sin herencia de `fruiz`.

## Por qué así (lo que las auditorías cambiaron)
- El plan original dejaba a `jaxsvc` (dueño del token) correr git/npm/pip DENTRO del clon que Qwen
  escribe: hooks, `credential.helper`, `insteadOf`, `.npmrc`, `.pth` y `--no-binary` podían sacar el token
  (3 BLOCK, escalón 3). Quedó: **espejo privado** de jaxsvc (único lugar con token), commits traídos por
  `upload-pack` corriendo como `axioma`, git blindado (sin config global/sistema, hooks a /dev/null) y
  **sandbox bwrap por lista blanca** para instalar, sin ejecutar código de terceros (npm solo registro con
  integridad y `--allow-git=none`; pip `--only-binary`; composer sin scripts ni plugins).
- La entrega revisa C1 sobre el diff Y cada commit: mensajes (`cat-file --batch`), rutas, `--text`,
  tamaños por `cat-file` de todos los blobs, identidad del autor; entrega SOLO si C3/C4/cadena/vigía/C5
  pasaron, relee la pausa antes de empujar y de abrir el PR, y los estados tras el push son honestos
  (`empujado_sin_pr`, `rama_empujada: "desconocido"`, verificación con `ls-remote`).

## Evidencia con usuarios reales (controlador)
- ACL: jaxsvc crea, axioma escribe, fruiz no lista ni lee. Ciclo jaxsvc clona → axioma commitea con
  `safe.directory` → jaxsvc lee y empuja (git 2.53).
- Espejo: axioma no lo lee; `pre-push` y `core.fsmonitor` plantados por Qwen NO se ejecutan al traer.
- Sandbox: `npm ci` con el node de producción rc 0; desde adentro no se ven `/etc/jax/.env` ni `/home`.
- Cerco: la llave de axioma no entra a axioma@hall9000; axioma → 127.0.0.1:58291 rechazado.
- Política exportada (17 reglas): push/--no-verify/workflows PROHIBIDO; git status, editar src, vitest
  PERMITIDO.

## Lecciones
- Un plan que no mira quién ejecuta qué con qué credencial es un plan con fuga: la auditoría de escalón 3
  encontró en el plan lo que ninguna prueba unitaria habría visto.
- Un auditor «de solo lectura» con sudo NOPASSWD ejecutó un bloque del runbook de Reverb en atem-ai
  (sin daño). Los briefs de auditoría ahora prohíben sudo y ejecutar bloques de runbooks.
- Un subagente con `git stash` o `npm` en el directorio equivocado ensucia un worktree compartido:
  verificar `git status` y la pila de stash tras cada entrega.

## Alternativas descartadas
- Perfil nuevo en `hyde_sandbox.py` (el Ejecutor no pasa por ahí; usa `cuenta_axioma`).
- Filtrar claves peligrosas de `.git/config` (lista negra frágil) → espejo privado.
- Aislar la red del sandbox sin root (slirp4netns no bloquea la LAN) → no ejecutar código de terceros.
- Usuario nuevo `axioma-deps` (requiere crear cuenta por máquina, GO de Fernando).

## Pendiente (con fecha, en claude-skills/PENDIENTES.md)
Despliegue en orden (jax primero + INSTALABLES; token; jax-platform), primera misión real, retiro de
`/command`, prueba de carga.
