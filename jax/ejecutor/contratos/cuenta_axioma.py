# jax/ejecutor/contratos/cuenta_axioma.py
"""Entrar a la cuenta del Ejecutor desde fuera (C1, C3, C4, C6).

`fruiz` entra a `axioma@127.0.0.1` con la llave del controlador: el kernel acota
las dos puntas (Fase 0, decisión de Fernando 2026-09-15). Todo sale de
JAX_EJECUTOR_*; sin una variable, no se entra.

El `claude` del Ejecutor se lanza SIEMPRE dentro de una jaula bwrap SUPERPUESTA
(`--dev-bind / /`): el mismo sistema que ve `axioma`, más montajes de solo lectura
que la cuenta no puede quitar:
- /etc/claude-code/managed-settings.json → el gancho de C1/C2. En el host ese
  directorio está vacío: el gancho no alcanza a nadie más (las sesiones de Fernando
  en hall9000 no lo ven).
- ~/.claude/settings.json y settings.local.json → `{}`: la cuenta no puede apagar
  los ganchos con `disableAllHooks` (documentación oficial: en cualquier nivel).
- /etc/ssh/ssh_config.d → vacío. En el espacio de usuarios de bwrap los archivos de root
  se ven de 65534 y ssh rechaza un Include del sistema que no es de root («Bad owner or
  permissions»): sin esto el Ejecutor no entra por ssh a ninguna máquina (visto 2026-09-17,
  misión de humo contra la VM desechable). Lo que se tapa (en hall9000,
  20-systemd-ssh-proxy.conf: `.host`, `unix/*`, `vsock/*`, `machine/*`) no es ninguna
  máquina del inventario; `destinos` las rechazaría igual.
SP2 reemplaza esta jaula por el perfil `ejecutor` de hyde_sandbox y conserva estos
montajes.

La llave del cerebro viaja por stdin (`read -r K`), nunca en argv: `ps` la vería.

M3 (auditoría adversarial, 2026-09-22): además de `$HOME/.claude/CLAUDE.md`, Claude Code
carga el CLAUDE.md DE PROYECTO en el directorio actual -- con `cd ~`, eso es
`$HOME/CLAUDE.md` -- y, aparte, guarda memoria automática entre sesiones en
`$HOME/.claude/projects/<cwd saneado>/memory/` (verificado contra el binario pinneado
2.1.273: `CLAUDE_CODE_DISABLE_AUTO_MEMORY`/`autoMemoryEnabled` y la ayuda de la opción de
directorio de memoria, que documenta ese default -- no se asume). Las dos rutas quedaban
sin tapar: `axioma` podía escribirse ahí una identidad o una "memoria" propia que
sobreviviera entre misiones, con la jaula cargándola igual que el CLAUDE.md real. Se
tapan con el mismo criterio que el resto de la jaula -- de sólo lectura, sin depender de
que exista nada del lado del host: un archivo vacío instalado (`$HOME/CLAUDE.md`) y un
`--tmpfs` + `--remount-ro` (`$HOME/.claude/.../memory`, que bwrap crea solo aunque el
host no tenga esa carpeta -- probado empíricamente).
"""
from __future__ import annotations

import asyncio
import os
import re
import shlex
import uuid
from dataclasses import dataclass
from pathlib import Path

from jax.ejecutor.contratos import contexto

_NOMBRE = re.compile(r"^[a-z_][a-z0-9_-]{0,31}$")


class CuentaSinConfigurar(RuntimeError):
    """`args[0]` es la variable que falta o no vale."""


@dataclass(frozen=True)
class Cuenta:
    nombre: str
    puerto: int
    llave: Path
    node_bin: Path
    lib: Path
    politica: Path


def cuenta_desde_entorno(env=None) -> Cuenta:
    env = os.environ if env is None else env

    def obligatoria(nombre):
        valor = env.get(nombre, "").strip()
        if not valor:
            raise CuentaSinConfigurar(nombre)
        return valor

    def ruta(nombre):
        valor = obligatoria(nombre)
        if not valor.startswith("/"):
            raise CuentaSinConfigurar(nombre)
        return Path(valor)

    nombre = obligatoria("JAX_EJECUTOR_CUENTA")
    if not _NOMBRE.match(nombre):
        raise CuentaSinConfigurar("JAX_EJECUTOR_CUENTA")
    puerto = obligatoria("JAX_EJECUTOR_SSH_PUERTO")
    if not puerto.isdigit() or not 0 < int(puerto) < 65536:
        raise CuentaSinConfigurar("JAX_EJECUTOR_SSH_PUERTO")
    return Cuenta(nombre, int(puerto), ruta("JAX_EJECUTOR_CONTROLADOR_LLAVE"), ruta("JAX_EJECUTOR_NODE_BIN"),
                  ruta("JAX_EJECUTOR_LIB"), ruta("JAX_EJECUTOR_POLITICA"))


def ssh_a_la_cuenta(c: Cuenta, remoto: str) -> list[str]:
    return ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", "-p", str(c.puerto),
            "-i", str(c.llave), f"{c.nombre}@127.0.0.1", remoto]


def _jaula(c: Cuenta) -> str:
    q = shlex.quote
    return " ".join([
        "bwrap", "--dev-bind", "/", "/", "--die-with-parent",
        "--tmpfs", "/etc/claude-code",
        "--tmpfs", "/etc/ssh/ssh_config.d",
        "--ro-bind", q(str(c.lib / "managed-settings.json")), "/etc/claude-code/managed-settings.json",
        "--ro-bind", q(str(c.lib / "settings-usuario.json")), '"$HOME/.claude/settings.json"',
        "--ro-bind", q(str(c.lib / "settings-usuario.json")), '"$HOME/.claude/settings.local.json"',
        # El CLAUDE.md GENERADO (spec §6.1: nunca a mano) tapa el de axioma -- no puede
        # reescribir su propia constitución dentro de la jaula, la única forma en que
        # corre `claude`. Las skills (cerebros.toml `skills`), igual: solo lectura.
        "--ro-bind", q(str(c.lib / contexto.CLAUDE_MD_REL)), '"$HOME/.claude/CLAUDE.md"',
        "--ro-bind", q(str(c.lib / contexto.SKILLS_REL)), '"$HOME/.claude/skills"',
        # M3: el CLAUDE.md DE PROYECTO (distinto del de arriba) y la memoria automática --
        # ver el docstring del módulo. `-home-<cuenta>` es el saneado de "$HOME" con "cd ~"
        # como cwd (reemplaza cada "/" por "-"), la misma convención que documenta la ayuda
        # de Claude Code para el directorio de memoria por omisión.
        "--ro-bind", q(str(c.lib / contexto.CLAUDE_MD_HOME_VACIO_REL)), '"$HOME/CLAUDE.md"',
        "--tmpfs", f'"$HOME/.claude/projects/-home-{c.nombre}/memory"',
        "--remount-ro", f'"$HOME/.claude/projects/-home-{c.nombre}/memory"',
        "--",
    ])


def _sesion_valida(sesion: str) -> bool:
    try:
        return str(uuid.UUID(sesion)) == sesion
    except (ValueError, TypeError, AttributeError):
        return False


def remoto_claude(c: Cuenta, *, base_url: str, modelo: str, prompt: str, herramientas: str = "Bash,Read",
                  max_salida_tokens: int | None = None, sesion: str | None = None, reanudar: bool = False) -> str:
    """`max_salida_tokens`: el tope que el proxy exige (JAX_PROXY_CARRIL_MAX_SALIDA_TOKENS) cuando
    `base_url` es un proxy con carril; sin él el arnés pide su propio `max_tokens` y el proxy da 403.

    `sesion` (SP2, spec 2026-09-15 §5): el id que genera Axioma. `reanudar=False` crea la sesión
    con ese id (`--session-id`); `reanudar=True` la retoma (`--resume`). Sólo un UUID canónico en
    minúsculas: el id termina en una ruta del HOME de la cuenta y en argv."""
    if sesion is not None and not _sesion_valida(sesion):
        raise ValueError("sesion_invalida")
    if reanudar and sesion is None:
        raise ValueError("reanudar_sin_sesion")
    q = shlex.quote
    variables = [
        f"ANTHROPIC_BASE_URL={q(base_url)}", 'ANTHROPIC_AUTH_TOKEN="$K"', f"PATH={q(str(c.node_bin))}:/usr/bin:/bin",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1", "DISABLE_AUTOUPDATER=1", "DISABLE_TELEMETRY=1",
        "DISABLE_ERROR_REPORTING=1",
    ]
    if max_salida_tokens is not None:
        variables.append(f"CLAUDE_CODE_MAX_OUTPUT_TOKENS={int(max_salida_tokens)}")
    entorno = " ".join(variables)
    claude = " ".join(q(a) for a in [str(c.node_bin / "claude"), "-p", prompt, "--output-format", "stream-json",
                                      "--verbose", "--model", modelo, "--allowedTools", herramientas]
                      + ([] if sesion is None else ["--resume" if reanudar else "--session-id", sesion]))
    return f"read -r K; cd ~ && env {entorno} {_jaula(c)} {claude}"


async def correr_en_la_cuenta(c: Cuenta, remoto: str, *, entrada: bytes = b"", tope_s: float):
    proc = await asyncio.create_subprocess_exec(
        *ssh_a_la_cuenta(c, remoto), stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE, start_new_session=True)
    try:
        salida, errores = await asyncio.wait_for(proc.communicate(entrada), tope_s)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        proc.kill()
        await proc.wait()
        raise
    return proc.returncode, salida, errores
