#!/usr/bin/env python3
"""Gobernanza de sub-agentes de Claude Code (DEUDA.md, "Sub-agentes de
Claude Code sin gobernanza real"): el UNICO punto de entrada aprobado
para lanzar `claude` como subproceso es
hyde_sandbox.py::run_sandboxed_claude() -- aplica el sandbox de bwrap y
el lock cross-proceso (flock(2)) que serializa las invocaciones entre el
proceso de las_manos y el del REPL (ver Task 1/2 del plan que agregó este
scanner).

Enforcement mecanico y acotado (mismo espiritu que
test_no_fail_open_except.py): un archivo que (a) lanza un subproceso por
CUALQUIERA de las formas conocidas (asyncio.create_subprocess_exec/_shell,
loop.subprocess_exec, subprocess.run/Popen/call/check_call/check_output,
os.system, os.exec*), Y (b) menciona "claude" en un STRING LITERAL del AST
(no en un comentario `#`), DEBE ser uno de los dos archivos aprobados del
root del repo -- cualquier otro archivo con esa combinacion esta lanzando
`claude` (o algo que lo referencia) por fuera del sandbox compartido.

Los dos archivos exentos, ambos por ruta resuelta en el ROOT del repo (no
por basename en cualquier lado del arbol, ver ALLOWED_FILENAMES):
  - hyde_sandbox.py -- el modulo aprobado, tambien alcanzable via el
    symlink las_manos/hyde_sandbox.py (es el mismo archivo).
  - _hyde_sandbox_test.py -- su archivo de tests dedicado; su
    subprocess.Popen lanza sys.executable para probar el flock(2) entre
    dos procesos de SO reales, nunca `claude`.

El criterio (b) mira string literals del AST y NO comentarios: los
comentarios de Python se descartan antes de parsear y nunca llegan al AST.
Un archivo que apenas COMENTA sobre claude/Hyde mientras lanza `git` por
subprocess no es una violacion -- caso real que este scanner marcaba mal
(las_manos/motor_registry/tool_authority.py), y que se iba a repetir
porque Hyde se discute por todo el codebase. Los docstrings SI cuentan.

La lista de formas se amplio 2026-08-25 (review final de rama): antes solo
cubria las dos async, asi que un `subprocess.run(["claude", ...])` pasaba
limpio. El item de DEUDA.md que este scanner cierra habla de "cualquier
OTRO músculo/automatización que dispare `claude`", y este repo ya usa la
API sincrona en 9+ archivos para otras cosas (git, etc.) -- es un idioma
vivo aca, no una hipotesis.

No detecta un lanzamiento de `claude` disfrazado (sin la palabra literal
"claude" en el archivo, ej. leida de una variable de entorno con otro
nombre) -- eso queda como residuo conocido, mismo criterio que P10 con
las formas mas sutiles del patron fail-open.

Imports con alias (ampliado 2026-10-02, MINOR-6): `import subprocess as sp`,
`from subprocess import run [as r]`, `import os as o`, `from os import system` y
sus equivalentes de pty/asyncio SE RESUELVEN (`_Alias`): un nombre solo cuenta si
viene de esos modulos. Las formas dinamicas se cubren desde la ronda 2 (ver abajo).

ALCANCE REAL EN CI (misma limitacion honesta que
test_no_fail_open_except.py, ver tambien el header de
.github/workflows/policy.yml): _repo_roots() incluye un fallback a
jax-platform, pero en el runner de GitHub Actions ese directorio NO existe
(repo privado separado, sin checkout cruzado configurado -- requeriria un
PAT/secret nuevo), asi que `if not root.exists(): continue` hace que en CI
solo se escanee el arbol de jax. La verificacion manual 2026-08-25 contra
los dos repos (cero falsos positivos: ningun otro call site de subprocess
-- las_manos/workers/ssh_worker.py, jax/voice/tts.py, jax/voice/ears.py,
jax-platform/backend/api/command.py -- menciona "claude") fue eso, manual;
no es una garantia que este job sostenga corrida a corrida para
jax-platform.

EXTENSION 2026-10-01 (facetas Thot y Kimi por suscripcion, spec §5): el mismo
control cubre ahora a `codex` y `kimi`, los otros dos CLIs que el nucleo
cli_sandbox.py lanza dentro de bwrap. Antes un `subprocess.run(["codex","exec"])`
o `["kimi","-p"]` pasaba limpio porque solo se buscaba "claude". Dos criterios
nuevos, ademas del de arriba:
  (c) un subproceso cuyo argv[0] es un LITERAL igual a claude|codex|kimi, o que
      termina en /claude, /codex o /kimi. Se resuelve el argv[0] de una lista
      literal, de una cadena (os.system), de `*lista` y de un nombre asignado a
      una lista literal en el mismo archivo.
  (d) cualquier string literal con la ruta de los binarios fijados o de las
      credenciales de esos CLIs (`_RUTAS_DE_CLI`): esa ruta no se escribe fuera
      del modulo aprobado, lance o no lance algo en ese archivo. Las rutas se
      arman por partes en este archivo A PROPOSITO: escritas enteras, el
      scanner se marcaria a si mismo.
EXTENSION 2026-10-02 (auditoria del paso 3, MINOR-6): el argv[0] se resuelve ahora
tambien a traves de `izq + der` (por la izquierda), `shutil.which("x")`, una
asignacion anotada (`cmd: list[str] = [...]`), `env [VAR=valor] cmd` y
`bash|sh -c "cmd ..."` (se toma el primer token de cmd, hasta 4 niveles); y los
lanzadores suman os.posix_spawn*, os.spawn*, os.popen y pty.spawn.

EXTENSION 2026-10-02 (ronda 2, MINOR-16): `getattr(subprocess, 'run')(...)` (y
`ejecutar = getattr(subprocess, 'run')`), `importlib.import_module('subprocess')` /
`__import__('subprocess')` (directo o asignado a un nombre) y la concatenacion de
literales (`['co' + 'dex']`, `'/opt/' + 'jax-cli'`, tambien a traves de un nombre
asignado) se resuelven. Se quito el pre-filtro de texto del recorrido: buscaba el nombre
entero y `'co' + 'dex'` no lo contiene. Residuo conocido: f-strings y cualquier nombre armado con algo mas que `+` de literales
o de nombres asignados a literales.

EXTENSION 2026-10-02 (reauditoria, MINOR-28): el criterio (a) TAMBIEN pliega las
concatenaciones de literales y `"".join([...])` de literales. scripts/axioma_sync.py, que
parte "CLAUDE.md" en dos literales y solo lanza `git`, pasa a una LISTA EXPLICITA de
exenciones (`_EXENTOS_POR_MENCION_PARTIDA`) con su justificacion, que ademas exige demostrar
que el archivo lanza solo `git`. Los criterios (c) y (d) no pliegan `.join`: este archivo
arma las rutas de los CLIs con `"".join(...)` a proposito (ver `_OPT_CLI`).

NO se busca la palabra "kimi" a secas: es el nombre de la faceta y del motor en
todo el codigo, y worker.py usa subprocess.run (para git). Daria falsos
positivos; el argv[0] y las rutas no.

Corre con:
  python3 policy/tests/test_claude_subprocess_solo_via_sandbox.py

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import ast
import os
import shlex
import sys
from pathlib import Path

_THIS_REPO_ROOT = Path(__file__).resolve().parents[2]


def _repo_roots() -> list[Path]:
    roots = [_THIS_REPO_ROOT]
    env_root = os.environ.get("JAX_PLATFORM_REPO_ROOT")
    roots.append(Path(env_root) if env_root else _THIS_REPO_ROOT.parent / "jax-platform")
    return roots


REPO_ROOTS = _repo_roots()

EXCLUDE_DIR_NAMES = {
    ".venv", "venv", "node_modules", ".git", ".worktrees", "worktrees",
    "__pycache__", "dist", "build",
}

# Exencion por RUTA RELATIVA, no por nombre pelado: tienen que ser los
# archivos que viven en el root de uno de los repos escaneados. Con
# `path.name in ALLOWED_FILENAMES` un hipotetico tools/hyde_sandbox.py
# quedaba exento tambien, sin ser el modulo real y aprobado.
#
# Se compara sobre la ruta RESUELTA (readlink) a proposito: el modulo real
# vive en el root del repo y se alcanza tambien como
# las_manos/hyde_sandbox.py, que es un SYMLINK a ../hyde_sandbox.py (mismo
# patron que facet_resolver.py / credential_resolver.py). Es el mismo
# archivo, no una copia -- exentarlo no afloja nada: un tools/hyde_sandbox.py
# que fuera un archivo DISTINTO resuelve a tools/hyde_sandbox.py y sigue
# siendo violacion.
#
# _hyde_sandbox_test.py es el archivo de tests DEDICADO del modulo aprobado.
# Su subprocess.Popen (ClaudeSubprocessLockRealCrossProcessTest) lanza
# sys.executable + un script worker para probar que el flock(2) serializa
# entre procesos de SO REALES -- nunca lanza `claude`. Es infraestructura de
# test necesaria del modulo aprobado, misma categoria que el modulo mismo.
# La exencion es de ESE nombre exacto en el root, no de "cualquier test":
# un *_test.py generico que lance `claude` sigue siendo violacion.
#
# cli_sandbox.py / _cli_sandbox_test.py (facetas-por-suscripcion, 2026-10-01): el
# nucleo comun del confinamiento (bwrap + flock + env minimo) del que hyde_sandbox
# pasa a depender, y su test dedicado. Es el UNICO otro lugar con un
# create_subprocess_exec: lo lanza siempre dentro de bwrap, con `env=` explicito.
ALLOWED_FILENAMES = frozenset({
    "hyde_sandbox.py", "_hyde_sandbox_test.py", "cli_sandbox.py", "_cli_sandbox_test.py",
})

# AISLAMIENTO POR CUENTA DE USUARIO (2026-09-16). El sandbox de bwrap no es el
# unico aislamiento valido: lo que la politica persigue es que ningun `claude`
# corra con el $HOME y los secretos de Fernando. El arnes de la Fase 0 del
# Ejecutor lo consigue por otra via —— lanza `ssh` a `axioma@127.0.0.1`, una
# cuenta SIN sudo y con su propio HOME (decision de Fernando, verificada en las
# cuatro maquinas del parque) —— y ahi el kernel acota las dos puntas.
#
# NO es una allowlist: cada archivo declarado tiene que DEMOSTRARLO. El
# detector comprueba que su subproceso arranca con `ssh` y lleva un destino
# `usuario@host`; si alguien quita el ssh y deja el `claude` desnudo, el
# archivo vuelve a ser violacion aunque siga declarado aqui. Lo comprueba
# test_un_archivo_declarado_que_pierde_el_ssh_vuelve_a_ser_violacion.
_AISLADO_POR_CUENTA = {
    "jax/ejecutor/contratos/cuenta_axioma.py":
        "entra por ssh a la cuenta axioma (sin sudo, HOME propio) y lanza el claude del "
        "Ejecutor dentro de una jaula bwrap superpuesta con el gancho de C1/C2 "
        "(Ejecutor SP1 plan 1, 2026-09-17)",
}


def _lanza_via_otra_cuenta(tree: ast.AST) -> bool:
    """¿El subproceso sale por `ssh` hacia un `usuario@host`?"""
    literales = [n.value for n in ast.walk(tree)
                 if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    return any(l == "ssh" for l in literales) and any("@" in l for l in literales)


# AISLAMIENTO POR bwrap DIRECTO, sin ssh (ronda 3 del contexto del Ejecutor, auditoría
# adversarial 2026-09-22). `_AISLADO_POR_CUENTA` sólo reconoce ssh-a-otra-cuenta porque
# es la ÚNICA vía que usa `cuenta_axioma.py` en PRODUCCIÓN. Pero su archivo de test
# (`tests/test_ejecutor_contratos_cuenta_axioma.py`) agregó pruebas que corren `bwrap`
# de VERDAD -- pedido explícito del coordinador ("un test tiene que ejecutar bwrap de
# verdad, no comparar el texto del comando") -- para probar el "$HOME" efímero (B-1/M-4)
# con un kernel real, sin necesitar un servidor ssh de axioma@127.0.0.1 en el runner de
# CI. Esas pruebas NUNCA lanzan `/opt/ejecutor/node-*/bin/claude`: el payload dentro de
# la jaula es `bash -c <script de prueba>` -- no hay ningún `claude` real corriendo, así
# que el riesgo que esta política persigue (un `claude` sin sandbox, con el HOME y los
# secretos de Fernando) no existe ahí. Mismo criterio que `_AISLADO_POR_CUENTA`: NO es
# una allowlist -- `_lanza_via_bwrap_directo` exige que el AST tenga, de verdad, un
# `bwrap` como primer argumento de un subproceso Y un `--tmpfs` sobre `$HOME` (la pieza
# central de B-1/M-4); si el archivo deja de invocar bwrap así, vuelve a ser violación
# (test_un_archivo_declarado_por_bwrap_que_pierde_bwrap_vuelve_a_ser_violacion).
_AISLADO_POR_BWRAP_DIRECTO = {
    "tests/test_ejecutor_contratos_cuenta_axioma.py":
        "corre `bwrap` de verdad (pedido del coordinador, ronda 3) para probar "
        "cuenta_axioma._jaula() con un kernel real -- el payload es `bash -c <script "
        "de prueba>`, nunca el binario `claude`; no hay riesgo de un claude sin sandbox",
}


def _lanza_via_bwrap_directo(tree: ast.AST) -> bool:
    """¿El subproceso arranca con `bwrap` y monta "$HOME" con `--tmpfs`? Las dos cosas
    juntas: `bwrap` solo no alcanza (podría no tocar $HOME en absoluto), y `--tmpfs`
    solo tampoco (podría ser sobre cualquier otra ruta)."""
    literales = [n.value for n in ast.walk(tree)
                 if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    return (any(l == "bwrap" or l.split()[:1] == ["bwrap"] for l in literales)
            and any("--tmpfs" in l for l in literales) and any("$HOME" in l for l in literales))


# EXENCIONES POR MENCION PARTIDA (MINOR-28, reauditoria 2026-10-02). El criterio (a) pliega
# las concatenaciones y los `"".join([...])` de literales, y por eso marca los archivos que
# parten el nombre "claude" en pedazos junto a un subproceso. Estos son los unicos que se
# aceptan, por ruta relativa al root de un repo escaneado y con su justificacion. NO es una
# allowlist ciega: `_exento_por_mencion_partida` exige ademas que el archivo lance SOLO `git`,
# que no lance ningun CLI de suscripcion (criterio c) ni escriba la ruta de sus binarios
# (criterio d); si deja de cumplirlo, vuelve a ser violacion. Y (MAJOR-30, auditoria del SHA
# 174da8c) el conjunto de cadenas plegadas que contienen "claude" (sin distinguir mayusculas)
# tiene que ser EXACTAMENTE el declarado en `_CADENAS_CLAUDE_EXENTAS`: sin esa condicion, un
# `['git', '-c', 'alias.x=!claude ...', 'x']`, un `core.pager=claude ...` o un
# `env={'GIT_SSH_COMMAND': 'claude -p x'}` dentro del archivo exento lanzaban claude via git y
# la exencion los dejaba pasar.
_EXENTOS_POR_MENCION_PARTIDA = {
    "scripts/axioma_sync.py":
        "lanza solo `git` (subprocess.run([\"git\", \"-C\", ...]) para leer el SHA de origen); el "
        "literal partido es \"CLAUDE.md\" dentro de una ruta de archivo, no el comando "
        "(`_CLAUDE_FILE = \"C\" + \"LAUDE.md\"`); las unicas cadenas con \"claude\" que se "
        "aceptan son tres, las que parte en sus lineas 27-29: \"CLAUDE.md\", \"claude-code\" "
        "y \"Claude Code\" -- cualquier otra constante con \"claude\" (argumento de git, valor "
        "de env) hace caer la exencion",
}
# El conjunto exacto de cadenas plegadas con "claude" que el archivo exento puede tener.
_CADENAS_CLAUDE_EXENTAS = frozenset({"CLAUDE.md", "claude-code", "Claude Code"})

# LV-001, 2026-10-03: dos excepciones POR RUTA para el paquete Faro permanente.
# El publicador root invoca exclusivamente Git y gh para fijar el SHA oficial;
# su _run(argv) tiene argv libre por la composición de opciones/refspec de Git.
# La lista exacta de literales y la forma de TODOS sus call sites de _run se
# comprueban abajo: si alguien añade un CLI de suscripción, una credencial,
# otro literal "claude" o un argv libre procedente de fuera, vuelve a rojo.
# El test del lanzador crea un qwen-auto temporal y SOLO ejecuta esa ruta fija;
# jamás lanza el binario real de Qwen/Claude. No son excepciones generales del
# scanner ni autorizan lanzar `claude` fuera de cli_sandbox.
_EXENTOS_FARO = {
    "ops/las-voces/faro_paquete_permanente.py": frozenset({
        "/srv/faro/claude-skills.git", "fjruizhn/claude-skills",
        "git@github.com:fjruizhn/claude-skills.git",
    }),
    "tests/test_las_voces_qwen_auto.py": frozenset({
        "JAX_FARO_REPO=/srv/faro/claude-skills.git\n",
        "JAX_FARO_REPO=/srv/faro/claude-skills.git\nJAX_FARO_SHA=",
    }),
}

# Congela los OCHO argv que llegan a `_run` en el publicador root. `ast.unparse`
# normaliza formato pero conserva estructura; un nuevo call site, incluso
# `_run([programa])` sin literal "claude", deja de estar exento. Cambiar esta
# lista exige revisión explícita de la operación root y de este escáner.
_FARO_RUN_ARGV_APROBADOS = frozenset({
    "['gh', 'api', f'repos/{REPO_OFICIAL}', '--jq', '.default_branch']",
    "['gh', 'api', f'repos/{REPO_OFICIAL}/git/ref/heads/{rama}', '--jq', '.object.sha']",
    "['/usr/bin/git', '--no-replace-objects', '-c', 'core.hooksPath=/dev/null', '-c', 'core.fsmonitor=false', 'clone', '--mirror', '--config', 'core.sshCommand=ssh -oBatchMode=yes', 'git@github.com:fjruizhn/claude-skills.git', str(MIRROR)]",
    "[*git, 'rev-parse', '--is-bare-repository']",
    "[*git, 'replace', '-l']",
    "[*git, 'config', '--local', '--get-all', 'safe.directory']",
    "[*git, '-c', 'core.sshCommand=ssh -oBatchMode=yes', 'fetch', '--prune', 'git@github.com:fjruizhn/claude-skills.git', '+refs/heads/*:refs/heads/*']",
    "[*git, 'rev-parse', REF_FRESCURA]",
})


def _cadenas_con_claude(tree: ast.AST) -> frozenset[str]:
    return frozenset(t for t in _cadenas_plegadas(tree) if "claude" in t.lower())


def _exento_por_mencion_partida(root: Path, path: Path, tree: ast.AST) -> bool:
    try:
        rel = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return False
    return (
        rel in _EXENTOS_POR_MENCION_PARTIDA
        and _cadenas_con_claude(tree) == _CADENAS_CLAUDE_EXENTAS
        and _lanza_solo_git(tree)
        and not _lanza_cli_de_suscripcion(tree)
        and not _menciona_ruta_de_cli(tree)
    )


def _exento_faro(root: Path, path: Path, tree: ast.AST) -> bool:
    """Exención acotada al publicador Git/gh y al test de su lanzador temporal."""
    try:
        rel = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return False
    if rel not in _EXENTOS_FARO or _cadenas_con_claude(tree) != _EXENTOS_FARO[rel]:
        return False
    if _lanza_cli_de_suscripcion(tree) or _menciona_ruta_de_cli(tree):
        return False
    alias = _Alias(tree)
    lanzamientos = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and _nombre_lanzador(n, alias)]
    if rel == "ops/las-voces/faro_paquete_permanente.py":
        # Un solo subprocess.run dentro de _run; todos sus call sites pasan
        # una lista escrita en este archivo, nunca `argv` que entre de fuera.
        if len(lanzamientos) != 1 or not isinstance(lanzamientos[0].func, ast.Attribute):
            return False
        if not (isinstance(lanzamientos[0].args[0], ast.Name)
                and lanzamientos[0].args[0].id == "argv"):
            return False
        llamadas = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Name) and n.func.id == "_run"]
        return (len(llamadas) == len(_FARO_RUN_ARGV_APROBADOS)
                and all(n.args and isinstance(n.args[0], ast.List) for n in llamadas)
                and {ast.unparse(n.args[0]) for n in llamadas} == _FARO_RUN_ARGV_APROBADOS)
    # El test usa subprocess.run únicamente con el script fixture creado por
    # _script_con_env_de_prueba; ninguna función acepta un argv de fuera.
    return bool(lanzamientos) and all(
        n.args and isinstance(n.args[0], ast.List)
        and len(n.args[0].elts) in (1, 2)
        and isinstance(n.args[0].elts[0], ast.Call)
        and isinstance(n.args[0].elts[0].func, ast.Name)
        and n.args[0].elts[0].func.id == "str"
        and len(n.args[0].elts[0].args) == 1
        and isinstance(n.args[0].elts[0].args[0], ast.Name)
        and n.args[0].elts[0].args[0].id in {"script", "unsafe"}
        and (len(n.args[0].elts) == 1 or (
            isinstance(n.args[0].elts[1], ast.Constant)
            and n.args[0].elts[1].value == "--help"))
        for n in lanzamientos
    )


def _declarado_aislado_por_cuenta(root: Path, path: Path, tree: ast.AST) -> bool:
    try:
        rel = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return False
    if rel in _AISLADO_POR_CUENTA and _lanza_via_otra_cuenta(tree):
        return True
    if rel in _AISLADO_POR_BWRAP_DIRECTO and _lanza_via_bwrap_directo(tree):
        return True
    return False

# Todas las formas de lanzar un subproceso que este repo podria usar. Las
# dos async eran las unicas cubiertas hasta la review final de rama
# (2026-08-25) -- un subprocess.run(["claude", ...]) pasaba limpio.
# os.exec* se matchea por prefijo (execl/execle/execlp/execlpe/execv/
# execve/execvp/execvpe). loop.subprocess_exec se matchea solo por nombre
# de atributo, sin verificar que el objeto sea un event loop: imprecision
# aceptada a proposito, igual que el resto de este scanner de nivel AST.
_SUBPROCESS_CALL_NAMES = {
    "create_subprocess_exec", "create_subprocess_shell", "subprocess_exec",
    "run", "Popen", "call", "check_call", "check_output",
    "system",
    # Ampliacion 2026-10-02 (MINOR-6): os.popen y pty.spawn.
    "popen", "spawn",
}

# os.exec*, os.spawn* (spawnl/spawnv/...) y os.posix_spawn* se matchean por prefijo.
_SUBPROCESS_CALL_PREFIXES = ("exec", "spawn", "posix_spawn")

# Nombres genericos que SOLO cuentan como lanzamiento de subproceso si se
# llaman como atributo de `subprocess`/`os`/`pty` (subprocess.run(...),
# os.system(...), pty.spawn(...)). Sin esto, cualquier `x.run(...)`,
# `self.call(...)` o `sock.spawn(...)` del codebase daria un falso positivo masivo.
_QUALIFIED_ONLY_NAMES = {"run", "call", "check_call", "check_output", "system", "popen", "spawn"}
_SUBPROCESS_MODULES = {"subprocess", "os", "pty"}
# Modulos cuyos alias (`import X as Y`, `from X import f [as g]`) se resuelven.
_MODULOS_CON_ALIAS = _SUBPROCESS_MODULES | {"asyncio", "shutil", "importlib"}


class _Alias:
    """Que nombres locales son, de verdad, los modulos subprocess/os/pty/asyncio/
    shutil/importlib o funciones importadas de ellos. Resuelve `import subprocess as
    sp` y `from subprocess import run [as r]` (el scanner anterior los daba por residuo
    aceptado: auditoria 2026-10-02, MINOR-6) y, desde la ronda 2 (MINOR-16), tambien
    `sp = importlib.import_module('subprocess')`, `sp = __import__('subprocess')` y
    `ejecutar = getattr(subprocess, 'run')`. Un nombre que NO viene de esos modulos no
    se toca: `from mimodulo import run` no es un lanzador."""

    def __init__(self, tree: ast.AST | None = None) -> None:
        self.modulos: dict[str, str] = {}
        self.funciones: dict[str, tuple[str, str]] = {}
        if tree is None:
            return
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                for a in n.names:
                    if a.name in _MODULOS_CON_ALIAS:
                        self.modulos[a.asname or a.name] = a.name
            elif isinstance(n, ast.ImportFrom) and n.module in _MODULOS_CON_ALIAS and not n.level:
                for a in n.names:
                    self.funciones[a.asname or a.name] = (n.module, a.name)
        # Asignaciones: en una segunda pasada, porque dependen de los imports de arriba.
        for n in ast.walk(tree):
            if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
                destino, valor = n.targets[0].id, n.value
            elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.value is not None:
                destino, valor = n.target.id, n.value
            else:
                continue
            modulo = _modulo_de_expr(valor, self)
            if modulo in _MODULOS_CON_ALIAS and isinstance(valor, ast.Call):
                self.modulos[destino] = modulo
                continue
            if isinstance(valor, ast.Call):
                par = _getattr_de_modulo(valor, self)
                if par is not None:
                    self.funciones[destino] = par


def _plegar(
    nodo: ast.AST, asignaciones: dict[str, ast.AST] | None = None, _prof: int = 0, *, con_join: bool = False,
) -> str | None:
    """La cadena a la que se pliega una expresion hecha SOLO de literales unidos con
    `+` (`'co' + 'dex'`), de nombres asignados a una (`A = 'co'; A + 'dex'`) o de
    un literal; None si no se puede resolver entera. Con `con_join=True` (solo el criterio
    (a), MINOR-28) tambien pliega `<literal>.join([literales])` / `.join((literales))`, cada
    elemento plegado a su vez. Los criterios (c) y (d) NO lo piden: este mismo archivo arma
    las rutas de los CLIs con `"".join(...)` a proposito, para no marcarse a si mismo."""
    if _prof > 6:
        return None
    if isinstance(nodo, ast.Constant) and isinstance(nodo.value, str):
        return nodo.value
    if isinstance(nodo, ast.BinOp) and isinstance(nodo.op, ast.Add):
        izq = _plegar(nodo.left, asignaciones, _prof + 1, con_join=con_join)
        der = _plegar(nodo.right, asignaciones, _prof + 1, con_join=con_join) if izq is not None else None
        return None if izq is None or der is None else izq + der
    if isinstance(nodo, ast.Name) and asignaciones and nodo.id in asignaciones:
        return _plegar(asignaciones[nodo.id], asignaciones, _prof + 1, con_join=con_join)
    if (
        con_join and isinstance(nodo, ast.Call) and not nodo.keywords and len(nodo.args) == 1
        and isinstance(nodo.func, ast.Attribute) and nodo.func.attr == "join"
        and isinstance(nodo.func.value, ast.Constant) and isinstance(nodo.func.value.value, str)
        and isinstance(nodo.args[0], (ast.List, ast.Tuple))
    ):
        partes = [_plegar(e, asignaciones, _prof + 1, con_join=True) for e in nodo.args[0].elts]
        if all(x is not None for x in partes):
            return nodo.func.value.value.join(partes)
    return None


def _es_llamada_de_importacion(call: ast.Call, alias: "_Alias") -> bool:
    """`importlib.import_module(...)`, `import_module(...)` (importado de importlib, con o
    sin alias) o `__import__(...)`."""
    f = call.func
    if isinstance(f, ast.Name):
        if f.id == "__import__":
            return True
        return alias.funciones.get(f.id) == ("importlib", "import_module")
    if isinstance(f, ast.Attribute) and f.attr == "import_module" and isinstance(f.value, ast.Name):
        return alias.modulos.get(f.value.id, f.value.id) == "importlib"
    return False


def _modulo_de_expr(expr: ast.AST, alias: "_Alias") -> str | None:
    """El modulo que representa `expr`: un nombre (`sp` -> subprocess si es un alias) o
    una importacion dinamica con nombre literal (`importlib.import_module('subprocess')`)."""
    if isinstance(expr, ast.Name):
        return alias.modulos.get(expr.id, expr.id)
    if isinstance(expr, ast.Call) and _es_llamada_de_importacion(expr, alias) and expr.args:
        return _plegar(expr.args[0])
    return None


def _getattr_de_modulo(call: ast.Call, alias: "_Alias") -> tuple[str, str] | None:
    """`getattr(<modulo>, 'nombre')` -> (modulo, nombre); el nombre puede ser una
    concatenacion de literales."""
    if _call_name(call.func) != "getattr" or len(call.args) < 2:
        return None
    modulo = _modulo_de_expr(call.args[0], alias)
    nombre = _plegar(call.args[1])
    if modulo is None or nombre is None:
        return None
    return modulo, nombre


_SIN_ALIAS = _Alias()


def _nombre_lanzador(node: ast.Call, alias: _Alias = _SIN_ALIAS) -> str | None:
    """Nombre canonico del lanzador de subprocesos que es `node` (`run`, `Popen`,
    `system`, `spawnlp`...), o None si no lo es."""
    func = node.func
    modulo: str | None = None
    if isinstance(func, ast.Call):
        par = _getattr_de_modulo(func, alias)  # getattr(subprocess, 'run')(...)
        if par is None:
            return None
        modulo, name = par
    elif isinstance(func, ast.Name) and func.id in alias.funciones:
        modulo, name = alias.funciones[func.id]
    else:
        name = _call_name(func)
        if isinstance(func, ast.Attribute):
            modulo = _modulo_de_expr(func.value, alias)
    if name is None:
        return None
    if name not in _SUBPROCESS_CALL_NAMES and not name.startswith(_SUBPROCESS_CALL_PREFIXES):
        return None
    if name in _QUALIFIED_ONLY_NAMES or name.startswith(_SUBPROCESS_CALL_PREFIXES):
        # Solo cuenta como subprocess.run / os.system / os.execv / pty.spawn, etc.
        if modulo not in _SUBPROCESS_MODULES:
            return None
    return name


def _iter_python_files():
    """Rinde pares (root, path) -- el root hace falta para poder exentar
    por ruta relativa, no por nombre de archivo pelado."""
    for root in REPO_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            rel_parts = path.relative_to(root).parts
            if any(part in EXCLUDE_DIR_NAMES for part in rel_parts):
                continue
            yield root, path


def _call_name(func: ast.expr) -> str | None:
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def _is_subprocess_launch(node: ast.Call, alias: _Alias = _SIN_ALIAS) -> bool:
    return _nombre_lanzador(node, alias) is not None


def _calls_create_subprocess(tree: ast.Module) -> bool:
    alias = _Alias(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _is_subprocess_launch(node, alias):
            return True
    return False


def _literales(tree: ast.AST):
    """Cada string literal del AST y, ademas, cada cadena que resulta de plegar una
    concatenacion de literales (`'/opt/' + 'jax-cli'`, tambien a traves de un nombre
    asignado): partir la ruta en dos literales no la esconde. Lo usa el criterio (d)."""
    asignaciones = _asignaciones_a_listas(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node.value
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            plegada = _plegar(node, asignaciones)
            if plegada is not None:
                yield plegada


def _cadenas_plegadas(tree: ast.AST):
    """Cada string literal del AST y cada cadena que resulta de plegar una concatenacion o un
    `"".join([...])` de literales (el plegado del criterio (a), con `con_join=True`)."""
    asignaciones = _asignaciones_a_listas(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node.value
        elif isinstance(node, (ast.BinOp, ast.Call)):
            texto = _plegar(node, asignaciones, con_join=True)
            if texto is not None:
                yield texto


def _references_claude_literal(tree: ast.Module) -> bool:
    """True si el archivo menciona "claude" en un STRING LITERAL del AST
    (incluidos docstrings), no en un comentario `#`.

    PLIEGA las concatenaciones de literales (`'cla' + 'ude'`, tambien a traves de un nombre
    asignado) y `"".join([...])` de literales (MINOR-28, reauditoria 2026-10-02): partir el
    nombre en dos literales no lo esconde. Marca asi a scripts/axioma_sync.py, que parte
    "CLAUDE.md" y solo lanza `git`: ese archivo esta en la lista explicita
    `_EXENTOS_POR_MENCION_PARTIDA`, con su justificacion, y no por un hueco del detector.

    Los comentarios de Python se descartan antes de parsear y NUNCA forman
    parte del AST -- por eso este chequeo los excluye de raiz, que es
    exactamente el alcance correcto: un comentario no puede ejecutar nada.
    El chequeo de texto crudo (`"claude" in source.lower()`) daba falso
    positivo con cualquier archivo que apenas COMENTARA sobre claude/Hyde
    -- caso real: las_manos/motor_registry/tool_authority.py, cuyos
    subprocess.run son todos `git` y cuya unica mencion de "claude" es un
    comentario que dice que Hyde corre claude FUERA de ese modulo. Hyde se
    discute por todo este codebase, asi que sin este fix el falso positivo
    se iba a repetir. Los docstrings SI cuentan (son ast.Constant): una
    referencia real, aunque inusual, podria esconderse ahi.

    Residuo conocido: f-strings, `%`/`.format` y un join con algo que no sea literal."""
    return any("claude" in t.lower() for t in _cadenas_plegadas(tree))


# CLIs de suscripcion que solo el nucleo cli_sandbox puede lanzar (spec §5).
_CLIS_DE_SUSCRIPCION = ("claude", "codex", "kimi")
# Armadas con "".join(...) y no con `+` a proposito: el criterio (d) pliega las
# concatenaciones de literales, y estas rutas escritas enteras o con `+` marcarian a
# este archivo. `.join` NO se pliega (residuo conocido, ver el header).
_OPT_CLI = "".join(("/opt/", "jax-cli"))
_HOME_KIMI = "".join((".", "kimi-code"))
_PAQUETES_CODEX = "".join((".", "codex/packages"))
_RUTAS_DE_CLI = (_OPT_CLI, _HOME_KIMI, _PAQUETES_CODEX)


def _es_argv0_de_cli(valor: str) -> bool:
    return any(valor == n or valor.endswith("/" + n) for n in _CLIS_DE_SUSCRIPCION)


_TIPOS_ASIGNABLES = (ast.List, ast.Tuple, ast.Constant, ast.BinOp, ast.Call)


def _asignaciones_a_listas(tree: ast.AST) -> dict[str, ast.AST]:
    """`cmd = ["kimi", "-p"]` o `cmd: list[str] = [...]` -> {"cmd": <List>}. Solo
    asignaciones simples de un nombre a una lista/tupla/cadena literal, a una
    concatenacion (`["kimi"] + x`) o a una llamada (`shutil.which("kimi")`); si el
    nombre se reasigna, gana la ultima (imprecision aceptada: el scanner es de
    nivel AST)."""
    out: dict[str, ast.AST] = {}
    for n in ast.walk(tree):
        if (isinstance(n, ast.Assign) and len(n.targets) == 1
                and isinstance(n.targets[0], ast.Name)
                and isinstance(n.value, _TIPOS_ASIGNABLES)):
            out[n.targets[0].id] = n.value
        elif (isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)
                and n.value is not None and isinstance(n.value, _TIPOS_ASIGNABLES)):
            out[n.target.id] = n.value
    return out


def _tokens_de_cadena(valor: str) -> list[str | None]:
    try:
        return list(shlex.split(valor))
    except ValueError:  # comillas sin cerrar: lo mejor que se puede hacer
        return valor.split()


def _tokens(nodo: ast.AST, asignaciones: dict[str, ast.AST], _prof: int = 0) -> list[str | None] | None:
    """Los tokens del argv que se le pasa a un lanzador, en la medida en que son
    literales (`None` = un elemento que no se puede resolver). Resuelve: cadena
    literal (se parte como un shell), lista/tupla, `*lista`, un nombre asignado,
    `izq + der` (por la izquierda: lo que haya a la derecha, si no es literal, queda
    como `None`) y `shutil.which("x")` / `which("x")` (el binario que se busca)."""
    if _prof > 4:
        return None
    if isinstance(nodo, ast.Constant) and isinstance(nodo.value, str):
        return _tokens_de_cadena(nodo.value)
    if isinstance(nodo, (ast.List, ast.Tuple)):
        out: list[str | None] = []
        for i, elt in enumerate(nodo.elts):
            if isinstance(elt, ast.Starred):
                sub = _tokens(elt.value, asignaciones, _prof + 1)
                out += sub if sub is not None else [None]
            elif (plegada := _plegar(elt, asignaciones)) is not None and not isinstance(elt, ast.Name):
                # el argv0 con espacios ("codex exec") cuenta por su primera palabra;
                # una concatenacion de literales (`'co' + 'dex'`) cuenta como su resultado
                out.append(plegada.split()[0] if i == 0 and plegada.split() else plegada)
            else:
                # un nombre asignado a una cadena/`which(...)` o un `which(...)` directo
                sub = _tokens(elt, asignaciones, _prof + 1) if isinstance(elt, (ast.Call, ast.Name)) else None
                out.append(sub[0] if sub and len(sub) == 1 else None)
        return out
    if isinstance(nodo, ast.Starred):
        return _tokens(nodo.value, asignaciones, _prof + 1)
    if isinstance(nodo, ast.Name) and nodo.id in asignaciones:
        return _tokens(asignaciones[nodo.id], asignaciones, _prof + 1)
    if isinstance(nodo, ast.BinOp) and isinstance(nodo.op, ast.Add):
        plegada = _plegar(nodo, asignaciones)
        if plegada is not None:  # solo literales: `'co' + 'dex'` es "codex"
            return _tokens_de_cadena(plegada)
        izq = _tokens(nodo.left, asignaciones, _prof + 1)
        if izq is None:
            return None
        return izq + [None]
    if isinstance(nodo, ast.Call) and _call_name(nodo.func) == "which" and nodo.args:
        arg = nodo.args[0]
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            return [arg.value]
    return None


def _es_shell_con_c(tokens: list[str | None], i: int) -> int | None:
    """Indice de la opcion `-c` (o `-lc`, `-ec`...) de un shell, si hay."""
    for j in range(i + 1, len(tokens)):
        t = tokens[j]
        if t is None:
            return None
        if t.startswith("-") and not t.startswith("--") and "c" in t[1:]:
            return j
        if not t.startswith("-"):
            return None
    return None


_SHELLS = ("bash", "sh", "dash", "zsh")


def _argv0_efectivo(tokens: list[str | None] | None) -> str | None:
    """El programa que de verdad se ejecuta: salta `env` (y sus opciones y
    `VAR=valor`) y entra en `bash -c '<cmd>'` / `sh -c '<cmd>'` tomando el primer
    token de <cmd>. Hasta 4 niveles (`env bash -c 'env A=1 codex'`)."""
    toks = tokens
    for _ in range(4):
        if not toks or toks[0] is None:
            return None
        a0 = toks[0]
        base = a0.rsplit("/", 1)[-1]
        if base == "env":
            k = 1
            while k < len(toks) and toks[k] is not None and (toks[k].startswith("-") or "=" in toks[k]):
                k += 1
            toks = toks[k:]
            continue
        if base in _SHELLS:
            j = _es_shell_con_c(toks, 0)
            if j is None or j + 1 >= len(toks) or toks[j + 1] is None:
                return a0
            toks = _tokens_de_cadena(toks[j + 1])
            continue
        return a0
    return None


def _primer_literal(nodo: ast.AST, asignaciones: dict[str, ast.AST], _prof: int = 0) -> str | None:
    """argv[0] literal (efectivo: tras `env` y dentro de `bash -c`) de lo que se le
    pasa a un lanzador de subprocesos."""
    return _argv0_efectivo(_tokens(nodo, asignaciones, _prof))


def _argv0s_de_lanzamientos(tree: ast.AST) -> list[str | None]:
    """El argv[0] efectivo (literal o None si no se resuelve) de CADA lanzamiento de
    subproceso del arbol, en el orden del recorrido. Un lanzamiento sin argumentos resolubles
    aporta None."""
    asignaciones = _asignaciones_a_listas(tree)
    alias = _Alias(tree)
    out: list[str | None] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        nombre = _nombre_lanzador(node, alias)
        if nombre is None:
            continue
        candidatos = list(node.args)
        candidatos += [k.value for k in node.keywords if k.arg == "args"]
        if nombre == "subprocess_exec" or (
            nombre.startswith("spawn") and nombre != "spawn"
        ):
            candidatos = candidatos[1:]  # loop.subprocess_exec(protocolo, argv0, ...) / os.spawn*(modo, ruta, ...)
        if not candidatos:
            out.append(None)
            continue
        if nombre.startswith("spawn") and nombre != "spawn" and len(candidatos) > 1:
            # os.spawnl*(modo, ruta, arg0, ...): el programa es `ruta`
            candidatos = candidatos[:1]
        out.append(_primer_literal(candidatos[0], asignaciones))
    return out


def _lanza_cli_de_suscripcion(tree: ast.AST) -> bool:
    """Criterio (c): un lanzamiento de subproceso cuyo argv[0] es claude|codex|kimi
    (o una ruta que termina en ellos). NO mira la palabra suelta."""
    return any(a is not None and _es_argv0_de_cli(a) for a in _argv0s_de_lanzamientos(tree))


def _lanza_solo_git(tree: ast.AST) -> bool:
    """Todos los lanzamientos del arbol son de `git` (argv[0] literal, resuelto), y hay al
    menos uno. Un argv[0] que no se resuelve cuenta como NO git: la exencion exige demostrarlo."""
    argv0s = _argv0s_de_lanzamientos(tree)
    return bool(argv0s) and all(a is not None and a.rsplit("/", 1)[-1] == "git" for a in argv0s)


def _menciona_ruta_de_cli(tree: ast.AST) -> bool:
    """Criterio (d): un literal (o una concatenacion de literales) con la ruta de los
    binarios fijados o de las credenciales de un CLI de suscripcion."""
    return any(r in lit for lit in _literales(tree) for r in _RUTAS_DE_CLI)


def _viola_la_politica(tree: ast.Module) -> bool:
    return (
        (_references_claude_literal(tree) and _calls_create_subprocess(tree))
        or _lanza_cli_de_suscripcion(tree)
        or _menciona_ruta_de_cli(tree)
    )


def _is_approved_sandbox_file(root: Path, path: Path) -> bool:
    """Los archivos aprobados son el hyde_sandbox.py del ROOT de un repo
    escaneado y su archivo de tests dedicado -- no cualquier archivo que se
    llame asi. Se resuelve el symlink primero
    (las_manos/hyde_sandbox.py -> ../hyde_sandbox.py es el mismo archivo)."""
    resolved = path.resolve()
    try:
        rel = resolved.relative_to(root.resolve())
    except ValueError:
        return False
    return len(rel.parts) == 1 and rel.name in ALLOWED_FILENAMES


def find_naked_claude_subprocess_files() -> list[str]:
    violations = []
    for root, path in _iter_python_files():
        if _is_approved_sandbox_file(root, path):
            continue
        try:
            source = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        # SIN pre-filtro de texto (ronda 2, MINOR-16): buscaba "codex"/"kimi"/"claude" en el
        # texto crudo y `'co' + 'dex'` no lo contiene, asi que el archivo se saltaba
        # entero. Parsear todo el arbol cuesta ~0,5 s (medido: 716 archivos).
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError:
            continue
        if _viola_la_politica(tree):
            if _declarado_aislado_por_cuenta(root, path, tree):
                continue
            if _exento_por_mencion_partida(root, path, tree):
                continue
            if _exento_faro(root, path, tree):
                continue
            violations.append(str(path))
    return violations


# --------------------------------------------------------------------------
# Auto-tests del scanner. Viven ACA y no en un archivo aparte a proposito:
# este archivo ya corre en CI (job no-naked-claude-subprocess), y operan
# sobre SNIPPETS de codigo fuente (strings), no sobre archivos plantados en
# el arbol -- asi ningun fixture puede caer dentro de un root escaneado ni
# enredarse con la logica de exencion de _hyde_sandbox_test.py. Nota de
# consistencia: los `subprocess.run(...)` de los snippets de abajo son
# STRING LITERALS, no llamadas reales, asi que este archivo no se convierte
# en violacion de su propia regla (verificado: el scanner lo da OK).
# --------------------------------------------------------------------------

def _detects(source: str) -> bool:
    """Aplica los criterios del scanner a un snippet, igual que
    find_naked_claude_subprocess_files() a un archivo real."""
    return _viola_la_politica(ast.parse(source))


def test_detecta_subprocess_run_sincrono() -> None:
    assert _detects(
        "import subprocess\n"
        "subprocess.run(['claude', '--print'], capture_output=True)\n"
    ), "subprocess.run(['claude', ...]) debe ser violacion (era invisible hasta 2026-08-25)"


def test_detecta_os_system() -> None:
    assert _detects("import os\nos.system('claude --print hola')\n")


def test_detecta_popen_y_exec_y_loop_subprocess_exec() -> None:
    assert _detects("from subprocess import Popen\nPopen(['claude'])\n")
    assert _detects("import os\nos.execvp('claude', ['claude'])\n")
    assert _detects("loop.subprocess_exec(proto, 'claude', '--print')\n")


def test_detecta_las_formas_async_originales() -> None:
    assert _detects("asyncio.create_subprocess_exec('claude', '--print')\n")
    assert _detects("asyncio.create_subprocess_shell('claude --print')\n")


def test_claude_solo_en_comentario_no_es_violacion() -> None:
    """Regresion del falso positivo real de
    las_manos/motor_registry/tool_authority.py: lanza `git` por subprocess y
    su unica mencion de claude es un comentario `#`. Los comentarios no
    llegan al AST -- no son señal de que el archivo referencie claude."""
    assert not _detects(
        "import subprocess\n"
        "# Este archivo NO lanza claude -- Hyde corre `claude -p` en otro modulo.\n"
        "subprocess.run(['git', 'status'])  # nada de claude aca\n"
    )


def test_claude_en_docstring_si_cuenta() -> None:
    """Los docstrings SI son ast.Constant -- una referencia real, aunque
    inusual, podria esconderse ahi. Solo se excluyen los comentarios `#`."""
    assert _detects(
        "import subprocess\n"
        "def go():\n"
        "    '''lanza claude'''\n"
        "    return subprocess.run(['x'])\n"
    )


def test_subprocess_sin_mencionar_claude_no_es_violacion() -> None:
    assert not _detects("import subprocess\nsubprocess.run(['git', 'status'])\n")


def test_menciona_claude_pero_no_lanza_subprocess_no_es_violacion() -> None:
    assert not _detects("CLI = 'claude'\nprint(CLI)\n")


def test_run_generico_no_calificado_no_cuenta() -> None:
    """`run`/`call`/`system` solo cuentan sobre subprocess/os -- si no,
    cualquier self.run(...) del codebase seria falso positivo."""
    assert not _detects("self.run(['claude'])\n")
    assert not _detects("runner.call('claude')\n")


def test_exencion_es_solo_para_el_root_del_repo() -> None:
    """La exencion es por ruta resuelta en el ROOT, no por basename: un
    tools/hyde_sandbox.py que fuera un archivo DISTINTO sigue siendo
    violacion."""
    root = _THIS_REPO_ROOT
    assert _is_approved_sandbox_file(root, root / "hyde_sandbox.py")
    assert _is_approved_sandbox_file(root, root / "_hyde_sandbox_test.py")
    assert _is_approved_sandbox_file(root, root / "cli_sandbox.py")
    assert _is_approved_sandbox_file(root, root / "_cli_sandbox_test.py")
    assert not _is_approved_sandbox_file(root, root / "tools" / "cli_sandbox.py")
    assert not _is_approved_sandbox_file(root, root / "tools" / "hyde_sandbox.py")
    assert not _is_approved_sandbox_file(root, root / "tools" / "_hyde_sandbox_test.py")


def test_symlink_del_modulo_aprobado_sigue_exento() -> None:
    """las_manos/hyde_sandbox.py es un symlink a ../hyde_sandbox.py -- el
    mismo archivo, alcanzado por el path por el que lo importa las_manos."""
    symlink = _THIS_REPO_ROOT / "las_manos" / "hyde_sandbox.py"
    assert symlink.is_symlink(), "symlink las_manos/hyde_sandbox.py debe existir"
    assert _is_approved_sandbox_file(_THIS_REPO_ROOT, symlink)


def test_symlink_de_cli_sandbox_sigue_exento() -> None:
    symlink = _THIS_REPO_ROOT / "las_manos" / "cli_sandbox.py"
    assert symlink.is_symlink(), "symlink las_manos/cli_sandbox.py debe existir"
    assert _is_approved_sandbox_file(_THIS_REPO_ROOT, symlink)


# --- Extension codex/kimi (spec §5): autopruebas positivas y negativas -------
# Cada autoprueba positiva es un snippet que el scanner VIEJO (solo "claude")
# dejaba pasar; si el criterio nuevo se rompe, se pone rojo (Principio VII).

def test_detecta_subprocess_run_de_codex_y_kimi() -> None:
    assert _detects("import subprocess\nsubprocess.run(['codex', 'exec', '-'])\n")
    assert _detects("import subprocess\nsubprocess.run(['kimi', '-p', 'hola'])\n")


def test_detecta_popen_y_execvp_de_codex_y_kimi() -> None:
    assert _detects("import subprocess\nsubprocess.Popen(['codex', 'exec'])\n")
    assert _detects("import subprocess\nsubprocess.Popen(('kimi',))\n")
    assert _detects("import os\nos.execvp('kimi', ['kimi', '-p', 'x'])\n")
    assert _detects("import os\nos.system('codex exec - < in.txt')\n")


def test_detecta_las_formas_async_de_codex_y_kimi() -> None:
    assert _detects("import asyncio\nasyncio.create_subprocess_exec('codex', 'exec')\n")
    assert _detects("import asyncio\nasyncio.create_subprocess_exec(*['kimi', '-p', 'x'])\n")
    assert _detects("import asyncio\nasyncio.create_subprocess_shell('kimi -p x')\n")
    assert _detects("loop.subprocess_exec(proto, 'codex', 'exec')\n")


def test_detecta_el_binario_por_ruta_absoluta() -> None:
    assert _detects(f"import subprocess\nsubprocess.run(['{_OPT_CLI}/codex/0.160.0/codex', 'exec'])\n")
    assert _detects("import subprocess\nsubprocess.run(['/usr/local/bin/kimi', '-p', 'x'])\n")
    assert _detects("import subprocess\nsubprocess.run(['/algun/lado/codex'])\n")


def test_detecta_el_argv_asignado_a_una_variable_de_la_misma_lista() -> None:
    assert _detects("import subprocess\ncmd = ['kimi', '-p', 'x']\nsubprocess.run(cmd)\n")
    assert _detects(
        "import asyncio\ncmd = ['codex', 'exec']\nasyncio.create_subprocess_exec(*cmd)\n")


def test_detecta_una_constante_con_la_ruta_de_kimi_aunque_no_lance_nada() -> None:
    assert _detects(f"KIMI = '{_OPT_CLI}/kimi/2.1.1/kimi'\n")
    assert _detects(f"HOME_KIMI = '/home/fruiz/{_HOME_KIMI}/credentials'\n")
    assert _detects(f"P = '/home/fruiz/{_PAQUETES_CODEX}/standalone/releases/x/codex'\n")


def test_la_faceta_kimi_junto_a_un_subprocess_de_git_no_es_violacion() -> None:
    """Autoprueba NEGATIVA del spec: `kimi` es el nombre de la faceta y del motor
    en todo el codigo, y worker.py usa subprocess.run para git. Buscar la palabra
    suelta daria un falso positivo en cada uno de esos archivos."""
    assert not _detects(
        "import subprocess\n"
        "FACET = 'kimi'\n"
        "subprocess.run(['git', 'log'])\n")
    assert not _detects(
        "import subprocess\n"
        "FACETAS = ['kimi', 'thot', 'jekyll']\n"
        "MOTOR = 'codex'\n"
        "subprocess.run(['git', 'status'], capture_output=True)\n")
    assert not _detects("import subprocess\nsubprocess.run(['git', 'commit', '-m', 'kimi codex'])\n")


def test_codex_o_kimi_sin_lanzar_nada_no_es_violacion() -> None:
    assert not _detects("FACET = 'kimi'\nprint(FACET)\n")
    assert not _detects("MOTORES = ('kimi', 'ada', 'codex')\n")


def test_run_generico_no_calificado_de_codex_o_kimi_no_cuenta() -> None:
    assert not _detects("self.run(['codex', 'exec'])\n")
    assert not _detects("runner.call('kimi')\n")


def test_un_comando_que_apenas_termina_parecido_no_es_argv0_de_cli() -> None:
    assert not _detects("import subprocess\nsubprocess.run(['/usr/bin/xcodex'])\n")
    assert not _detects("import subprocess\nsubprocess.run(['kimi-helper'])\n")


def test_un_archivo_que_no_es_el_aprobado_con_la_ruta_de_cli_sigue_siendo_violacion(tmp_path) -> None:
    """Control del control sobre el recorrido de archivos real: un archivo plantado
    en un root escaneado, que NO es uno de los aprobados, aparece en la lista."""
    raiz = tmp_path / "repo"
    (raiz / "tools").mkdir(parents=True)
    plantado = f"RUTA = '{_OPT_CLI}/codex/x/codex'\n"
    (raiz / "tools" / "cli_sandbox.py").write_text(plantado)
    (raiz / "cli_sandbox.py").write_text(plantado)
    global REPO_ROOTS
    anteriores = REPO_ROOTS
    REPO_ROOTS = [raiz]
    try:
        encontrados = {Path(v).relative_to(raiz).as_posix() for v in find_naked_claude_subprocess_files()}
    finally:
        REPO_ROOTS = anteriores
    assert encontrados == {"tools/cli_sandbox.py"}


def test_un_archivo_declarado_que_pierde_el_ssh_vuelve_a_ser_violacion(tmp_path) -> None:
    """La declaracion NO es un salvoconducto: si el archivo deja de lanzar por
    ssh hacia otra cuenta, el aislamiento que declaraba ya no existe y vuelve a
    contar como violacion. Sin esto, _AISLADO_POR_CUENTA seria una allowlist."""
    arbol_con_ssh = ast.parse(
        'import subprocess\n'
        'CLAUDE = "/opt/ejecutor/bin/claude"\n'
        'subprocess.run(["ssh", "axioma@127.0.0.1", CLAUDE])\n')
    arbol_sin_ssh = ast.parse(
        'import subprocess\n'
        'CLAUDE = "/opt/ejecutor/bin/claude"\n'
        'subprocess.run([CLAUDE, "-p", "hola"])\n')
    assert _lanza_via_otra_cuenta(arbol_con_ssh)
    assert not _lanza_via_otra_cuenta(arbol_sin_ssh)


def test_el_harness_declarado_sigue_lanzando_por_ssh() -> None:
    """Control del control sobre el archivo REAL: si el harness de la Fase 0
    deja de usar ssh, esto se pone rojo antes que nadie lo note."""
    for rel in _AISLADO_POR_CUENTA:
        for root, path in _iter_python_files():
            try:
                if path.resolve().relative_to(root.resolve()).as_posix() != rel:
                    continue
            except ValueError:
                continue
            arbol = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            assert _lanza_via_otra_cuenta(arbol), (
                f"{rel} esta declarado como aislado por cuenta pero ya no lanza "
                f"por ssh: o se le devuelve el ssh, o se quita de _AISLADO_POR_CUENTA")


def test_un_archivo_declarado_por_bwrap_que_pierde_bwrap_vuelve_a_ser_violacion() -> None:
    """Mismo criterio que `test_un_archivo_declarado_que_pierde_el_ssh_vuelve_a_ser_
    violacion`, para el mecanismo nuevo: la declaración NO es un salvoconducto."""
    con_bwrap = ast.parse(
        'import subprocess\n'
        'subprocess.run(["bwrap", "--dev-bind", "/", "/", "--tmpfs", "$HOME", "--", "bash"])\n')
    sin_tmpfs_home = ast.parse(
        'import subprocess\n'
        'subprocess.run(["bwrap", "--dev-bind", "/", "/", "--", "bash"])\n')
    sin_bwrap = ast.parse(
        'import subprocess\n'
        'subprocess.run(["/opt/ejecutor/node/bin/claude", "-p", "hola"])\n')
    assert _lanza_via_bwrap_directo(con_bwrap)
    assert not _lanza_via_bwrap_directo(sin_tmpfs_home)
    assert not _lanza_via_bwrap_directo(sin_bwrap)


def test_el_archivo_de_test_declarado_por_bwrap_sigue_corriendo_bwrap_de_verdad() -> None:
    """Control del control sobre el archivo REAL: si
    tests/test_ejecutor_contratos_cuenta_axioma.py deja de invocar bwrap con
    "--tmpfs $HOME", esto se pone rojo antes que nadie lo note."""
    for rel in _AISLADO_POR_BWRAP_DIRECTO:
        for root, path in _iter_python_files():
            try:
                if path.resolve().relative_to(root.resolve()).as_posix() != rel:
                    continue
            except ValueError:
                continue
            arbol = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            assert _lanza_via_bwrap_directo(arbol), (
                f"{rel} esta declarado como aislado por bwrap directo pero ya no lo "
                f"hace: o se le devuelve el bwrap con --tmpfs $HOME, o se quita de "
                f"_AISLADO_POR_BWRAP_DIRECTO")


# --- Formas que el scanner no veia (auditoria 2026-10-02, MINOR-6) ----------
# Cada autoprueba positiva es un snippet que el scanner anterior dejaba pasar.

def test_detecta_la_concatenacion_por_la_izquierda() -> None:
    assert _detects("import subprocess\nsubprocess.run(['codex'] + extra)\n")
    assert _detects("import subprocess\nsubprocess.run(['kimi', '-p'] + [x])\n")
    assert _detects("import os\nos.system('codex exec - ' + entrada)\n")
    assert _detects("import subprocess\ncmd = ['kimi'] + args\nsubprocess.run(cmd)\n")
    assert _detects("import subprocess\nsubprocess.run(cmd)\ncmd = ['codex', 'exec'] + extra\n")
    assert not _detects("import subprocess\nsubprocess.run(['git'] + ['codex'])\n")


def test_detecta_shutil_which() -> None:
    assert _detects("import subprocess, shutil\nsubprocess.run([shutil.which('codex'), 'exec'])\n")
    assert _detects("import subprocess\nfrom shutil import which\nsubprocess.run([which('kimi')])\n")
    assert _detects("import subprocess, shutil\nbinario = shutil.which('kimi')\nsubprocess.run([binario, '-p'])\n")
    assert not _detects("import subprocess, shutil\nsubprocess.run([shutil.which('git'), 'log'])\n")


def test_detecta_la_asignacion_anotada() -> None:
    assert _detects("import subprocess\ncmd: list[str] = ['kimi', '-p', 'x']\nsubprocess.run(cmd)\n")
    assert _detects("import asyncio\ncmd: tuple = ('codex', 'exec')\nasyncio.create_subprocess_exec(*cmd)\n")
    assert not _detects("import subprocess\ncmd: list[str] = ['git', 'log']\nsubprocess.run(cmd)\n")


def test_detecta_subprocess_con_alias_o_importado_directo() -> None:
    assert _detects("import subprocess as sp\nsp.run(['codex', 'exec'])\n")
    assert _detects("import subprocess as sp\nsp.Popen(['kimi'])\n")
    assert _detects("from subprocess import run\nrun(['kimi', '-p', 'x'])\n")
    assert _detects("from subprocess import run as correr\ncorrer(['codex'])\n")
    assert _detects("from subprocess import check_output\ncheck_output(['codex', 'exec'])\n")
    assert _detects("from os import system\nsystem('codex exec -')\n")
    assert _detects("import os as sistema\nsistema.system('kimi -p x')\n")
    assert _detects("from os import execvp\nexecvp('kimi', ['kimi'])\n")
    assert _detects("from subprocess import run\nrun(['claude', '--print'])\n")
    assert _detects("import subprocess as sp\nsp.run(['claude', '--print'])\n")


def test_el_alias_no_inventa_violaciones() -> None:
    assert not _detects("import subprocess as sp\nsp.run(['git', 'status'])\n")
    assert not _detects("from subprocess import run\nrun(['git', 'log'])\n")
    assert not _detects("def run(x): pass\nrun(['codex'])\n")          # `run` local, no es subprocess
    assert not _detects("from mimodulo import run\nrun(['kimi'])\n")  # `run` de otro modulo
    assert not _detects("import foo as sp\nsp.run(['codex'])\n")      # `sp` no es subprocess


def test_detecta_el_comando_dentro_de_un_shell() -> None:
    assert _detects("import subprocess\nsubprocess.run(['bash', '-c', 'codex exec - < in.txt'])\n")
    assert _detects("import subprocess\nsubprocess.run(['sh', '-c', 'kimi -p x'])\n")
    assert _detects("import subprocess\nsubprocess.run(['/bin/bash', '-c', 'codex'])\n")
    assert _detects("import subprocess\nsubprocess.run(['/bin/sh', '-lc', 'kimi -p x'])\n")
    assert _detects("import os\nos.system(\"bash -c 'kimi -p x'\")\n")
    assert _detects("import subprocess\nsubprocess.run(['bash', '-c', 'env A=1 codex exec'])\n")
    assert not _detects("import subprocess\nsubprocess.run(['bash', '-c', 'git status'])\n")
    assert not _detects("import subprocess\nsubprocess.run(['bash', '-c', 'echo codex'])\n")
    assert not _detects("import subprocess\nsubprocess.run(['bash', 'script-de-codex.sh'])\n")


def test_detecta_el_comando_tras_env() -> None:
    assert _detects("import subprocess\nsubprocess.run(['env', 'FOO=1', 'codex', 'exec'])\n")
    assert _detects("import subprocess\nsubprocess.run(['/usr/bin/env', 'A=1', 'B=2', 'kimi'])\n")
    assert _detects("import subprocess\nsubprocess.run(['env', '-i', 'A=1', 'kimi'])\n")
    assert _detects("import os\nos.system('env FOO=1 codex exec -')\n")
    assert _detects("import subprocess\nsubprocess.run(['env', 'bash', '-c', 'kimi -p x'])\n")
    assert not _detects("import subprocess\nsubprocess.run(['env', 'FOO=1', 'git', 'log'])\n")
    assert not _detects("import subprocess\nsubprocess.run(['env'])\n")


def test_detecta_los_otros_lanzadores_de_procesos() -> None:
    assert _detects("import os\nos.posix_spawn('/usr/local/bin/codex', ['codex'], {})\n")
    assert _detects("import os\nos.posix_spawnp('kimi', ['kimi', '-p'], {})\n")
    assert _detects("import os\nos.spawnlp(os.P_WAIT, 'codex', 'codex', 'exec')\n")
    assert _detects("import os\nos.spawnv(os.P_NOWAIT, '/usr/local/bin/kimi', ['kimi'])\n")
    assert _detects("import os\nos.spawnvp(os.P_WAIT, 'codex', ['codex'])\n")
    assert _detects("import os\nos.popen('codex exec -')\n")
    assert _detects("import os\nos.popen('kimi -p x').read()\n")
    assert _detects("import pty\npty.spawn(['kimi', '-p', 'x'])\n")
    assert _detects("import pty\npty.spawn('codex')\n")
    assert _detects("import pty as terminal\nterminal.spawn(['codex'])\n")
    assert _detects("import os\nos.popen('claude --print')\n")


def test_los_otros_lanzadores_no_inventan_violaciones() -> None:
    assert not _detects("import os\nos.popen('git log')\n")
    assert not _detects("import os\nos.posix_spawn('/usr/bin/git', ['git'], {})\n")
    assert not _detects("import pty\npty.spawn(['bash'])\n")
    assert not _detects("import os\nos.spawnlp(os.P_WAIT, 'git', 'git')\n")
    assert not _detects("sock.spawn(['codex'])\n")      # `spawn` de otro objeto
    assert not _detects("pool.popen('kimi')\n")         # `popen` de otro objeto


# --- Formas dinamicas (auditoria 2026-10-02, ronda 2, MINOR-16) -------------
# `getattr(subprocess, 'run')(...)`, `importlib.import_module('subprocess')` y la
# concatenacion de literales (`'co' + 'dex'`) dejaron de ser residuo aceptado.

def test_detecta_getattr_de_un_lanzador() -> None:
    assert _detects("import subprocess\ngetattr(subprocess, 'run')(['codex', 'exec'])\n")
    assert _detects("import subprocess as sp\ngetattr(sp, 'Popen')(['kimi', '-p'])\n")
    assert _detects("import os\ngetattr(os, 'system')('codex exec -')\n")
    assert _detects("import asyncio\ngetattr(asyncio, 'create_subprocess_exec')('codex', 'exec')\n")
    assert _detects("import subprocess\ngetattr(subprocess, 'ru' + 'n')(['kimi'])\n")
    assert _detects("import subprocess\ngetattr(subprocess, 'run')(['claude', '--print'])\n")
    assert _detects("import subprocess\nejecutar = getattr(subprocess, 'run')\nejecutar(['codex'])\n")


def test_getattr_no_inventa_violaciones() -> None:
    assert not _detects("import subprocess\ngetattr(subprocess, 'run')(['git', 'status'])\n")
    assert not _detects("getattr(obj, 'run')(['codex'])\n")                       # `obj` no es un modulo de procesos
    assert not _detects("import subprocess\ngetattr(subprocess, 'PIPE')\n")      # no lo llama
    assert not _detects("import subprocess\ngetattr(subprocess, 'list2cmdline')(['codex'])\n")


def test_detecta_importlib_y_dunder_import() -> None:
    assert _detects("import importlib\nsp = importlib.import_module('subprocess')\nsp.run(['codex', 'exec'])\n")
    assert _detects("import importlib\nimportlib.import_module('subprocess').run(['kimi'])\n")
    assert _detects("import importlib\nimportlib.import_module('os').system('codex exec -')\n")
    assert _detects("from importlib import import_module\nimport_module('subprocess').Popen(['codex'])\n")
    assert _detects("from importlib import import_module as im\nim('subprocess').run(['kimi'])\n")
    assert _detects("__import__('subprocess').run(['codex'])\n")
    assert _detects("sp = __import__('subprocess')\nsp.check_output(['kimi', '-p', 'x'])\n")
    assert _detects("import importlib\nsp = importlib.import_module('subprocess')\ngetattr(sp, 'run')(['codex'])\n")
    assert _detects("import importlib\nimportlib.import_module('subprocess').run(['claude'])\n")


def test_importlib_no_inventa_violaciones() -> None:
    assert not _detects("import importlib\nsp = importlib.import_module('subprocess')\nsp.run(['git', 'log'])\n")
    assert not _detects("import importlib\nm = importlib.import_module('json')\nm.run(['codex'])\n")
    assert not _detects("import importlib\nimportlib.import_module('mimodulo').run(['kimi'])\n")
    assert not _detects("import importlib\nimportlib.import_module('subprocess')\n")  # lo importa, no lanza nada


def test_detecta_la_concatenacion_de_literales() -> None:
    assert _detects("import subprocess\nsubprocess.run(['co' + 'dex', 'exec'])\n")
    assert _detects("import subprocess\nsubprocess.run(['/usr/local/bin/' + 'kimi', '-p'])\n")
    assert _detects("import subprocess\nsubprocess.run(['c' + 'o' + 'de' + 'x'])\n")
    assert _detects("import subprocess\nsubprocess.run('co' + 'dex')\n")
    assert _detects("import os\nos.system('co' + 'dex exec -')\n")
    assert _detects("import os\nos.system('co' + 'dex exec - ' + entrada)\n")
    assert _detects("import subprocess\nBIN = 'ki' + 'mi'\nsubprocess.run([BIN, '-p'])\n")
    assert _detects("import subprocess\nA = 'co'\nB = A + 'dex'\nsubprocess.run([B, 'exec'])\n")
    assert _detects("import subprocess\nsubprocess.run(['cla' + 'ude', '--print'])\n")
    assert _detects("import subprocess\nsubprocess.run(['bash', '-c', 'co' + 'dex exec -'])\n")


def test_la_concatenacion_tambien_cubre_las_rutas_de_los_binarios() -> None:
    assert _detects(f"RUTA = '/opt/' + '{_OPT_CLI.rsplit('/', 1)[-1]}' + '/codex'\n")
    assert _detects("HOME = '/home/fruiz/.ki' + 'mi-code/credentials'\n")
    assert _detects("P = '/home/fruiz/.co' + 'dex/packages/x'\n")


def test_la_concatenacion_no_inventa_violaciones() -> None:
    assert not _detects("import subprocess\nsubprocess.run(['gi' + 't', 'log'])\n")
    assert not _detects("import subprocess\nsubprocess.run(['co' + 'dex-helper'])\n")
    assert not _detects("import subprocess\nsubprocess.run(['x' + 'co' + 'dex'])\n")
    assert not _detects("import subprocess\nsubprocess.run(['git', 'commit', '-m', 'co' + 'dex'])\n")
    assert not _detects("MOTOR = 'co' + 'dex'\nprint(MOTOR)\n")                  # nombra, no lanza
    assert not _detects("import subprocess\nsubprocess.run(['git'] + ['co' + 'dex'])\n")
    # el caso real de scripts/axioma_sync.py parte "CLAUDE.md" en dos literales y solo lanza
    # `git`: desde MINOR-28 el criterio (a) SI pliega la mencion partida, y ese archivo queda
    # exento por la lista explicita `_EXENTOS_POR_MENCION_PARTIDA` (ver las autopruebas de abajo),
    # no por un hueco del detector.


def test_la_mencion_partida_de_claude_junto_a_un_subproceso_es_violacion() -> None:
    """MINOR-28 (reauditoria 2026-10-02): el criterio (a) no plegaba, asi que un archivo que
    lanzaba un subproceso y armaba "claude" a pedazos (`'cla' + 'ude'`, `"".join([...])`) para
    pasarlo por una variable, un f-string o un shell quedaba limpio. Ahora la mencion partida
    cuenta igual que la entera."""
    # concatenacion
    assert _detects("import subprocess\nN = 'cla' + 'ude'\nsubprocess.run(['sh', '-c', N])\n")
    assert _detects("import subprocess\nsubprocess.run(['git', 'log'])\nNOMBRE = 'C' + 'LAUDE.md'\n")
    assert _detects("import subprocess\nA = 'cla'\nB = A + 'ude'\nsubprocess.run(['sh', '-c', B])\n")
    # "".join([...]) de literales, con lista, con tupla y con separador
    assert _detects("import subprocess\nN = ''.join(['cla', 'ude'])\nsubprocess.run(['sh', '-c', N])\n")
    assert _detects("import subprocess\nN = ''.join(('cla', 'ude'))\nsubprocess.run(['sh', '-c', N])\n")
    assert _detects("import subprocess\nN = '-'.join(['x', 'cla' + 'ude'])\nsubprocess.run(['sh', '-c', N])\n")
    assert _detects("import os\nos.system(''.join(['cl', 'au', 'de']) + ' --print')\n")


def test_la_mencion_partida_no_inventa_violaciones() -> None:
    # sin subproceso, o con un subproceso que no tiene nada que ver, no hay violacion
    assert not _detects("N = ''.join(['cla', 'ude'])\nprint(N)\n")
    assert not _detects("N = 'cla' + 'ude'\nprint(N)\n")
    assert not _detects("import subprocess\nN = ''.join(['gi', 't'])\nsubprocess.run([N, 'log'])\n")
    assert not _detects("import subprocess\nN = '-'.join(['a', 'b'])\nsubprocess.run(['git', N])\n")
    # un join con algo que no es literal no se pliega (residuo conocido): no se inventa un nombre
    assert not _detects("import subprocess\nN = ''.join([x, 'ude'])\nsubprocess.run(['sh', '-c', N])\n")
    assert not _detects("import subprocess\nN = ''.join(partes)\nsubprocess.run(['git', N])\n")


def test_axioma_sync_esta_en_la_lista_explicita_de_exenciones_con_su_justificacion() -> None:
    rel = "scripts/axioma_sync.py"
    assert rel in _EXENTOS_POR_MENCION_PARTIDA
    just = _EXENTOS_POR_MENCION_PARTIDA[rel]
    assert "git" in just and all(c in just for c in _CADENAS_CLAUDE_EXENTAS), "la justificacion nombra las tres"
    # el archivo real: el detector lo marca (si no, la exencion seria letra muerta) y la
    # exencion lo demuestra (solo lanza `git`, y el nombre partido es "CLAUDE.md")
    arbol = ast.parse((_THIS_REPO_ROOT / rel).read_text(encoding="utf-8"))
    assert _viola_la_politica(arbol), "el detector ya no ve la mencion partida de axioma_sync.py"
    assert _exento_por_mencion_partida(_THIS_REPO_ROOT, _THIS_REPO_ROOT / rel, arbol)


def test_exenciones_faro_exigen_literales_y_lanzamientos_exactos() -> None:
    for rel in _EXENTOS_FARO:
        ruta = _THIS_REPO_ROOT / rel
        fuente = ruta.read_text(encoding="utf-8")
        arbol = ast.parse(fuente)
        assert _viola_la_politica(arbol), f"exención muerta: {rel}"
        assert _exento_faro(_THIS_REPO_ROOT, ruta, arbol), rel
        # Una mención nueva del CLI en un argumento Git o fixture no se
        # convierte en permiso por estar en un archivo exento.
        mutado = ast.parse(fuente + "\nMARCA = 'claude --danger'\n")
        assert not _exento_faro(_THIS_REPO_ROOT, ruta, mutado), rel


def test_exenciones_faro_niegan_argv_libre_nuevo() -> None:
    publicador = _THIS_REPO_ROOT / "ops/las-voces/faro_paquete_permanente.py"
    fuente = publicador.read_text(encoding="utf-8")
    for llamada in (
        "_run(argv_externo)",
        "_run([programa])",
        "_run([programa, '--print'])",
        "_run(['/usr/bin/git', *argv_externo])",
    ):
        mutado = ast.parse(fuente + f"\n{llamada}\n")
        assert not _exento_faro(_THIS_REPO_ROOT, publicador, mutado), llamada
    prueba = _THIS_REPO_ROOT / "tests/test_las_voces_qwen_auto.py"
    fuente_prueba = prueba.read_text(encoding="utf-8")
    mutado_prueba = ast.parse(fuente_prueba + "\nsubprocess.run(argv_externo)\n")
    assert not _exento_faro(_THIS_REPO_ROOT, prueba, mutado_prueba)


def _plantar(tmp_path, archivos: dict[str, str]) -> set[str]:
    raiz = tmp_path / "repo"
    for rel, contenido in archivos.items():
        (raiz / rel).parent.mkdir(parents=True, exist_ok=True)
        (raiz / rel).write_text(contenido)
    global REPO_ROOTS
    anteriores = REPO_ROOTS
    REPO_ROOTS = [raiz]
    try:
        return {Path(v).relative_to(raiz).as_posix() for v in find_naked_claude_subprocess_files()}
    finally:
        REPO_ROOTS = anteriores


_SOLO_GIT_CON_CLAUDE_MD = (
    "import subprocess\nNOMBRE = 'C' + 'LAUDE.md'\n"
    "HARNESS = 'cla' + 'ude-code'\nTITULO = 'Cla' + 'ude Code'\n"
    "subprocess.run(['git', 'rev-parse', 'HEAD'])\n"
)


def test_la_exencion_vale_solo_para_la_ruta_declarada_y_solo_si_lanza_git(tmp_path) -> None:
    """Negativa de la exencion: el contenido que pasa en scripts/axioma_sync.py se marca en
    cualquier otro lado, y en esa misma ruta se marca en cuanto lanza algo que no es `git`."""
    encontrados = _plantar(tmp_path, {
        "scripts/axioma_sync.py": _SOLO_GIT_CON_CLAUDE_MD,                    # exento
        "tools/axioma_sync.py": _SOLO_GIT_CON_CLAUDE_MD,                      # otra ruta, mismo nombre
        "scripts/otro.py": _SOLO_GIT_CON_CLAUDE_MD,                           # otro nombre
    })
    assert encontrados == {"tools/axioma_sync.py", "scripts/otro.py"}
    encontrados = _plantar(tmp_path / "b", {
        "scripts/axioma_sync.py": _SOLO_GIT_CON_CLAUDE_MD + "subprocess.run(['ls'])\n",
    })
    assert encontrados == {"scripts/axioma_sync.py"}, "lanza algo que no es git: la exencion cae"
    encontrados = _plantar(tmp_path / "c", {
        "scripts/axioma_sync.py": _SOLO_GIT_CON_CLAUDE_MD + "subprocess.run(['co' + 'dex'])\n",
    })
    assert encontrados == {"scripts/axioma_sync.py"}, "lanza un CLI de suscripcion: la exencion cae"
    encontrados = _plantar(tmp_path / "c2", {
        "scripts/axioma_sync.py": _SOLO_GIT_CON_CLAUDE_MD + "subprocess.run([''.join(['gi', 't'])])\n",
    })
    assert encontrados == {"scripts/axioma_sync.py"}, "un argv[0] que no se resuelve no demuestra que sea git"
    encontrados = _plantar(tmp_path / "d", {
        "scripts/axioma_sync.py": _SOLO_GIT_CON_CLAUDE_MD + f"RUTA = '{_OPT_CLI}/codex'\n",
    })
    assert encontrados == {"scripts/axioma_sync.py"}, "escribe la ruta de un CLI: la exencion cae"


def test_la_exencion_exige_el_conjunto_exacto_de_cadenas_con_claude(tmp_path) -> None:
    """MAJOR-30: en scripts/axioma_sync.py, cualquier cadena con "claude" fuera de las tres
    declaradas (argumento de git, valor de env) hace caer la exencion; las tres solas, no."""
    trampolines = {
        "alias": "subprocess.run(['git', '-c', 'alias.x=!claude --dangerously-skip-permissions -p x', 'x'])\n",
        "core.pager": "subprocess.run(['git', '-c', 'core.pager=claude -p hola', 'log'])\n",
        "GIT_SSH_COMMAND": "subprocess.run(['git', 'fetch'], env={'GIT_SSH_COMMAND': 'claude -p x'})\n",
        "plegado": "subprocess.run(['git', '-c', 'alias.x=!' + 'cla' + 'ude -p x', 'x'])\n",
        "mayusculas": "subprocess.run(['git', '-c', 'core.pager=CLAUDE -p x', 'log'])\n",
    }
    for nombre, extra in trampolines.items():
        encontrados = _plantar(tmp_path / nombre, {"scripts/axioma_sync.py": _SOLO_GIT_CON_CLAUDE_MD + extra})
        assert encontrados == {"scripts/axioma_sync.py"}, f"trampolin {nombre}: la exencion no debe cubrirlo"
    # falta una de las tres: tampoco es el archivo declarado
    sin_titulo = _SOLO_GIT_CON_CLAUDE_MD.replace("TITULO = 'Cla' + 'ude Code'\n", "")
    assert _plantar(tmp_path / "falta", {"scripts/axioma_sync.py": sin_titulo}) == {"scripts/axioma_sync.py"}
    # las tres exactas: exento
    assert _plantar(tmp_path / "ok", {"scripts/axioma_sync.py": _SOLO_GIT_CON_CLAUDE_MD}) == set()


def test_el_archivo_real_axioma_sync_tiene_exactamente_las_tres_cadenas_con_claude() -> None:
    arbol = ast.parse((_THIS_REPO_ROOT / "scripts/axioma_sync.py").read_text(encoding="utf-8"))
    assert _cadenas_con_claude(arbol) == _CADENAS_CLAUDE_EXENTAS == {"CLAUDE.md", "claude-code", "Claude Code"}


def test_un_archivo_con_el_nombre_partido_no_se_salta_el_prefiltro(tmp_path) -> None:
    """El pre-filtro del recorrido buscaba "codex" en el texto crudo: `'co' + 'dex'`
    no lo contiene y el archivo se saltaba entero. Se prueba sobre el recorrido real."""
    raiz = tmp_path / "repo"
    (raiz / "tools").mkdir(parents=True)
    (raiz / "tools" / "lanza.py").write_text("import subprocess\nsubprocess.run(['co' + 'dex', 'exec'])\n")
    global REPO_ROOTS
    anteriores = REPO_ROOTS
    REPO_ROOTS = [raiz]
    try:
        encontrados = {Path(v).relative_to(raiz).as_posix() for v in find_naked_claude_subprocess_files()}
    finally:
        REPO_ROOTS = anteriores
    assert encontrados == {"tools/lanza.py"}


def test_no_naked_claude_subprocess() -> None:
    violations = find_naked_claude_subprocess_files()
    assert not violations, (
        f"{len(violations)} archivo(s) lanzan un subproceso y "
        "mencionan 'claude', o lanzan `codex`/`kimi`, o escriben la ruta de sus binarios "
        "o credenciales, fuera de hyde_sandbox.py::run_sandboxed_claude() / cli_sandbox.py "
        "-- cualquier invocacion de esos CLIs como subproceso debe pasar por "
        "esos modulos (sandbox de bwrap + lock cross-proceso via flock):\n"
        + "\n".join(violations)
    )


def main() -> int:
    violations = find_naked_claude_subprocess_files()
    if violations:
        print(f"FAIL — {len(violations)} archivo(s):")
        for v in violations:
            print(f"  {v}")
        return 1
    print(
        "OK — ningun subprocess de claude/codex/kimi fuera de hyde_sandbox.py y "
        "cli_sandbox.py (+ sus tests dedicados)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
