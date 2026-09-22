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

B-1/M-4 (auditoría adversarial, ronda 3, 2026-09-22 -- reemplaza el parche puntual de M3
de la ronda 2, que sólo tapaba `$HOME/CLAUDE.md` y el directorio de memoria uno por uno y
dejaba TODO lo demás bajo `$HOME` -- `commands/`, `rules/`, `agents/`, `~/.claude.json`
con `mcpServers`, sesiones viejas -- como el `$HOME` REAL de `axioma`, escribible y
persistente entre misiones. Verificado en producción, 2026-09-22 (lectura de sólo
inspección): `/home/axioma/.claude/` tenía `plugins/`, `projects/` con subcarpetas de
sesiones anteriores, `session-env/` con 226 entradas -- exactamente lo que un parche
archivo-por-archivo no podía cerrar).

`$HOME` pasa a ser un `--tmpfs` PROPIO de esta invocación: nace vacío y muere con el
proceso. Encima se montan, de sólo lectura, las piezas REALES que `axioma` necesita para
operar (probado con bwrap real, binario pinneado 2.1.273, arrancando en un `$HOME` vacío:
no hace falta nada más para que `claude` arranque):
- `~/.ssh` (la llave PROPIA de axioma, su `config` y su `known_hosts` -- la que usa para
  entrar por ssh a las máquinas remotas; NO es la llave del controlador, que es de fruiz
  hacia axioma, ni la del freno).
- `~/.claude/settings.json` / `settings.local.json` → `{}` (no puede apagar los ganchos).
- `~/.claude/CLAUDE.md` (el generado, spec §6.1) y `~/.claude/skills` (las tres skills).

`~/.claude/projects` es la ÚNICA pieza que se monta en lectura y ESCRITURA cuando hay una
misión (`directorio_projects`, no `None`): ahí vive la sesión que hace posible `--resume`
entre turnos de la MISMA misión, y con ella la memoria automática de Claude Code
(`~/.claude/projects/<cwd>/memory/`). LÍMITE, documentado a propósito, no un descuido:
DENTRO de la misma misión esa carpeta es escribible por el propio agente. Lo que se cierra
es que sobreviva a la misión SIGUIENTE -- cada misión recibe un directorio nuevo
(`mision_servicio.py` lo crea, dueño `axioma`, antes del primer turno) y nada de otra
misión es visible ahí. Sin misión (canario, smoke test de humo), `directorio_projects` es
`None` y esa carpeta es un `--tmpfs` más: ni siquiera dentro de esa corrida sobrevive nada.
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
    home: Path


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
                  ruta("JAX_EJECUTOR_LIB"), ruta("JAX_EJECUTOR_POLITICA"), ruta("JAX_EJECUTOR_CUENTA_HOME"))


def ssh_a_la_cuenta(c: Cuenta, remoto: str) -> list[str]:
    return ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", "-p", str(c.puerto),
            "-i", str(c.llave), f"{c.nombre}@127.0.0.1", remoto]


def _jaula(c: Cuenta, *, directorio_projects: Path | None) -> str:
    """`directorio_projects=None`: sin misión (canario, humo) -- ni "--resume" ni memoria
    sobreviven ni siquiera dentro de la corrida. Con una ruta: esa carpeta, EN LECTURA Y
    ESCRITURA -- la prepara `mision_servicio.py` antes del primer turno, dueño `axioma`."""
    q = shlex.quote
    argv = [
        "bwrap", "--dev-bind", "/", "/", "--die-with-parent",
        "--tmpfs", "/etc/claude-code",
        "--tmpfs", "/etc/ssh/ssh_config.d",
        "--ro-bind", q(str(c.lib / "managed-settings.json")), "/etc/claude-code/managed-settings.json",
        # B-1/M-4: "$HOME" nace vacío en CADA invocación -- ver el docstring del módulo.
        "--tmpfs", '"$HOME"',
        # La llave PROPIA de axioma hacia las remotas (no la del controlador ni la del
        # freno), su `config` y su `known_hosts`. De sólo lectura: la cuenta no puede
        # rotar su propia llave ni tocar el known_hosts sin que se note (sería tapar C6 a
        # mano -- ya prohibido en NUCLEO_REGLAS).
        "--ro-bind", q(str(c.home / ".ssh")), '"$HOME/.ssh"',
        "--ro-bind", q(str(c.lib / "settings-usuario.json")), '"$HOME/.claude/settings.json"',
        "--ro-bind", q(str(c.lib / "settings-usuario.json")), '"$HOME/.claude/settings.local.json"',
        # El CLAUDE.md GENERADO (spec §6.1: nunca a mano) tapa el de axioma -- no puede
        # reescribir su propia constitución dentro de la jaula, la única forma en que
        # corre `claude`. Las skills (cerebros.toml `skills`), igual: solo lectura.
        "--ro-bind", q(str(c.lib / contexto.CLAUDE_MD_REL)), '"$HOME/.claude/CLAUDE.md"',
        "--ro-bind", q(str(c.lib / contexto.SKILLS_REL)), '"$HOME/.claude/skills"',
    ]
    if directorio_projects is None:
        argv += ["--tmpfs", '"$HOME/.claude/projects"']
    else:
        argv += ["--bind", q(str(directorio_projects)), '"$HOME/.claude/projects"']
    argv.append("--")
    return " ".join(argv)


def _sesion_valida(sesion: str) -> bool:
    try:
        return str(uuid.UUID(sesion)) == sesion
    except (ValueError, TypeError, AttributeError):
        return False


def remoto_claude(c: Cuenta, *, base_url: str, modelo: str, prompt: str, herramientas: str = "Bash,Read",
                  max_salida_tokens: int | None = None, sesion: str | None = None, reanudar: bool = False,
                  directorio_projects: Path | None = None) -> str:
    """`max_salida_tokens`: el tope que el proxy exige (JAX_PROXY_CARRIL_MAX_SALIDA_TOKENS) cuando
    `base_url` es un proxy con carril; sin él el arnés pide su propio `max_tokens` y el proxy da 403.

    `sesion` (SP2, spec 2026-09-15 §5): el id que genera Axioma. `reanudar=False` crea la sesión
    con ese id (`--session-id`); `reanudar=True` la retoma (`--resume`). Sólo un UUID canónico en
    minúsculas: el id termina en una ruta del HOME de la cuenta y en argv.

    `directorio_projects` (B-1/M-4, ronda 3): la carpeta que la jaula monta en lectura y
    escritura sobre "$HOME/.claude/projects" -- ver `ruta_projects_de_la_mision` y el
    docstring del módulo. `None` (canario, humo, sin misión): ni siquiera dentro de esta
    corrida sobrevive nada ahí."""
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
    return f"read -r K; cd ~ && env {entorno} {_jaula(c, directorio_projects=directorio_projects)} {claude}"


_MISION_ID_VALIDA = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


class MisionIdInvalido(ValueError):
    pass


def ruta_projects_de_la_mision(misiones: Path, mision_id: str) -> Path:
    """El directorio POR MISIÓN (B-1/M-4) que se monta en lectura y escritura sobre
    "$HOME/.claude/projects". `misiones` es JAX_EJECUTOR_MISIONES, el mismo directorio
    donde ya vive `<id_vigia>.json` (mision_servicio.py::abrir_vigia) -- éste va en un
    subdirectorio nuevo, `<mision_id>/claude-projects`, para no mezclarse con esos
    archivos. Sólo un UUID canónico en minúsculas: termina en una ruta de sistema de
    archivos y en el argv de `sudo install` (`preparar_directorio_projects`)."""
    if not _MISION_ID_VALIDA.match(mision_id):
        raise MisionIdInvalido(mision_id)
    return misiones / mision_id / "claude-projects"


async def preparar_directorio_projects(c: Cuenta, ruta: Path, *, correr=None) -> None:
    """Crea `ruta`, dueño `c.nombre` (axioma), 0700 -- root:root o jaxsvc:jaxsvc no
    alcanza: `axioma` tiene que poder escribir ahí desde DENTRO de la jaula. Requiere
    NOPASSWD sudo para `install` en el controlador (fruiz en hall9000, ya lo tiene para
    el resto de `ops/ejecutor/instalar_*.sh`); si falla, se propaga (fail-closed: sin
    directorio propio no hay `--resume` posible, y el llamador no debe seguir)."""
    correr = correr or asyncio.create_subprocess_exec
    proc = await correr("sudo", "install", "-d", "-o", c.nombre, "-g", c.nombre, "-m", "0700", str(ruta),
                        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    _, errores = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"preparar_directorio_projects_fallo: {errores.decode(errors='replace')}")


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
