# Traspaso — El Faro 0.6 / R-2

- Fecha: 2026-10-09
- Responsable: Codex, hall9000
- Rama de trabajo: `codex/faro-r2`
- Base de origen: `origin/master@709237f901e8ec8840f6b2b39a1a2ce88682674a`
- Estado actual: análisis confirmado; plan escrito; baseline Faro de control/lanzador 247 passed con Python 3.14.4 y el venv existente de `jax-faro-f11-rule-authority`.
- Trabajo en paralelo que no se debe tocar: cadena F1.1 de c2, incluidas #373/#371/#375/#378/#379/#381. c2 recibió aviso de que esta rama cubre solo R-2/0.6.
- Evidencia de sistema local: `systemd 259`; prueba temporal de `.scope` y `.service` del user manager aceptó MemoryMax/CPUQuota/TasksMax y `systemctl kill --kill-whom=all` vació el cgroup. No existe usuario `faro` en hall9000; nada fue instalado.
- Decisión técnica propuesta: unidad `.service` transitoria por ejecución/uid para que el manager cree el proceso dentro del cgroup. No solicitar system manager/broker privilegiado. La precisión se documenta como desviación de la palabra `.scope`; pendiente de auditar.
- Bloqueo externo no relacionado: Fernando aún no respondió la elección de red para 0.5; no afecta este tramo.
- Siguiente paso: implementar TDD según `docs/superpowers/plans/2026-10-09-faro-scope-r2.md` en este worktree; no editar el checkout raíz `/home/fruiz/jax` ni ramas F1.1.
- No hacer: desplegar, crear la cuenta/user manager `faro`, cambiar `policy/**`, integrar PR ajeno o afirmar que 0.10/0.9 están cerrados.
