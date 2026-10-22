# tests/test_ejecutor_contratos_cuenta_axioma.py
"""Cómo se entra a la cuenta del Ejecutor: todo desde el entorno, sin defaults; la
llave del cerebro nunca en argv; el lanzamiento siempre dentro de la jaula
superpuesta con los settings de solo lectura."""
import asyncio
import getpass
import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

from jax.ejecutor.contratos import cuenta_axioma as CA

ENV = {
    "JAX_EJECUTOR_CUENTA": "axioma", "JAX_EJECUTOR_SSH_PUERTO": "58291",
    "JAX_EJECUTOR_CONTROLADOR_LLAVE": "/home/fruiz/.ssh/id_ejecutor_controlador",
    "JAX_EJECUTOR_NODE_BIN": "/opt/ejecutor/node-v24.16.0/bin", "JAX_EJECUTOR_LIB": "/opt/ejecutor/lib",
    "JAX_EJECUTOR_POLITICA": "/etc/jax-ejecutor/politica.json", "JAX_EJECUTOR_CUENTA_HOME": "/home/axioma",
}


def test_cuenta_desde_el_entorno():
    c = CA.cuenta_desde_entorno(ENV)
    assert c == CA.Cuenta("axioma", 58291, Path("/home/fruiz/.ssh/id_ejecutor_controlador"),
                          Path("/opt/ejecutor/node-v24.16.0/bin"), Path("/opt/ejecutor/lib"),
                          Path("/etc/jax-ejecutor/politica.json"), Path("/home/axioma"))


@pytest.mark.parametrize("variable", sorted(ENV))
def test_sin_una_variable_no_se_entra(variable):
    with pytest.raises(CA.CuentaSinConfigurar):
        CA.cuenta_desde_entorno({k: v for k, v in ENV.items() if k != variable})


@pytest.mark.parametrize("variable, valor", [("JAX_EJECUTOR_SSH_PUERTO", "x"), ("JAX_EJECUTOR_LIB", "relativa"),
                                             ("JAX_EJECUTOR_CUENTA", "root; rm")])
def test_valores_invalidos_no_entran(variable, valor):
    with pytest.raises(CA.CuentaSinConfigurar):
        CA.cuenta_desde_entorno({**ENV, variable: valor})


def test_ssh_a_la_cuenta():
    argv = CA.ssh_a_la_cuenta(CA.cuenta_desde_entorno(ENV), "uptime")
    assert argv == ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", "-p", "58291",
                    "-i", "/home/fruiz/.ssh/id_ejecutor_controlador", "axioma@127.0.0.1", "uptime"]


def test_remoto_claude_va_en_la_jaula_y_sin_la_llave_en_argv():
    remoto = CA.remoto_claude(CA.cuenta_desde_entorno(ENV), base_url="http://127.0.0.1:18436", modelo="canario",
                              prompt="hola 'mundo'")
    assert remoto.startswith("read -r K; cd ~ && env ")
    assert 'ANTHROPIC_AUTH_TOKEN="$K"' in remoto
    palabras = shlex.split(remoto.split(" && ", 1)[1].replace('"$K"', "K"))
    i = palabras.index("bwrap")
    jaula = palabras[i:palabras.index("--", i)]
    assert ["--tmpfs", "/etc/claude-code"] == jaula[jaula.index("--tmpfs"):jaula.index("--tmpfs") + 2]
    assert ["--ro-bind", "/opt/ejecutor/lib/managed-settings.json", "/etc/claude-code/managed-settings.json"] in \
        [jaula[k:k + 3] for k in range(len(jaula))]
    assert "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1" in palabras
    assert palabras[palabras.index("--", i) + 1] == "/opt/ejecutor/node-v24.16.0/bin/claude"
    assert "hola 'mundo'" in palabras


def test_la_jaula_monta_el_claude_md_generado_y_tapa_el_de_axioma():
    """C1/C6 del contexto (2026-09-22): el CLAUDE.md instalado (generado, nunca a
    mano) se monta ro sobre "$HOME/.claude/CLAUDE.md" -- el viejo, propiedad de
    axioma, queda TAPADO: axioma no puede reescribir su propia constitución
    dentro de la jaula (la única forma en que corre `claude`)."""
    remoto = CA.remoto_claude(CA.cuenta_desde_entorno(ENV), base_url="http://127.0.0.1:18436", modelo="canario",
                              prompt="x")
    palabras = shlex.split(remoto.split(" && ", 1)[1].replace('"$K"', "K"))
    i = palabras.index("bwrap")
    jaula = palabras[i:palabras.index("--", i)]
    assert ["--ro-bind", "/opt/ejecutor/lib/contexto/CLAUDE.md", "$HOME/.claude/CLAUDE.md"] in \
        [jaula[k:k + 3] for k in range(len(jaula))]


def test_la_jaula_monta_las_skills():
    remoto = CA.remoto_claude(CA.cuenta_desde_entorno(ENV), base_url="http://127.0.0.1:18436", modelo="canario",
                              prompt="x")
    palabras = shlex.split(remoto.split(" && ", 1)[1].replace('"$K"', "K"))
    i = palabras.index("bwrap")
    jaula = palabras[i:palabras.index("--", i)]
    assert ["--ro-bind", "/opt/ejecutor/lib/contexto/skills", "$HOME/.claude/skills"] in \
        [jaula[k:k + 3] for k in range(len(jaula))]


# --- HOME efímero por misión (B-1/M-4, ronda 3, 2026-09-22) -------------------------
#
# Reemplaza el parche puntual de M3 (ronda 2): "$HOME" entero pasa a ser un tmpfs propio
# de la invocación, y encima se montan sólo las piezas reales que hacen falta.

def _jaula_de(**kw):
    remoto = CA.remoto_claude(CA.cuenta_desde_entorno(ENV), base_url="http://127.0.0.1:18436", modelo="canario",
                              prompt="x", **kw)
    palabras = shlex.split(remoto.split(" && ", 1)[1].replace('"$K"', "K"))
    i = palabras.index("bwrap")
    return palabras[i:palabras.index("--", i)]


def test_la_jaula_monta_home_como_tmpfs_propio():
    jaula = _jaula_de()
    duplas = [jaula[k:k + 2] for k in range(len(jaula))]
    assert ["--tmpfs", "$HOME"] in duplas
    # ANTES de cualquier bind que caiga DENTRO de "$HOME" -- si no, el punto de montaje
    # no existe todavía cuando bwrap intenta el --ro-bind/--bind siguiente.
    idx_tmpfs_home = next(k for k in range(len(jaula)) if jaula[k:k + 2] == ["--tmpfs", "$HOME"])
    for k in range(len(jaula) - 2):
        if jaula[k] in ("--ro-bind", "--bind") and jaula[k + 2].startswith("$HOME"):
            assert k > idx_tmpfs_home, (jaula[k:k + 3], idx_tmpfs_home)


def test_la_jaula_monta_la_llave_propia_de_axioma_hacia_las_remotas():
    """La llave PROPIA de axioma (para entrar por ssh a las máquinas remotas), no la
    del controlador (fruiz -> axioma) ni la del freno -- de sólo lectura."""
    tripletes = [_jaula_de()[k:k + 3] for k in range(len(_jaula_de()))]
    assert ["--ro-bind", "/home/axioma/.ssh", "$HOME/.ssh"] in tripletes


def test_sin_mision_projects_es_un_tmpfs_efimero():
    """Canario, humo: sin `directorio_projects`, ni "--resume" ni memoria sobreviven ni
    siquiera dentro de la corrida."""
    jaula = _jaula_de()
    duplas = [jaula[k:k + 2] for k in range(len(jaula))]
    assert ["--tmpfs", "$HOME/.claude/projects"] in duplas
    assert not any(jaula[k] == "--bind" for k in range(len(jaula)))


def test_con_mision_projects_es_de_lectura_y_escritura():
    jaula = _jaula_de(directorio_projects=Path("/var/lib/jax-ejecutor-misiones/m1/claude-projects"))
    tripletes = [jaula[k:k + 3] for k in range(len(jaula))]
    assert ["--bind", "/var/lib/jax-ejecutor-misiones/m1/claude-projects", "$HOME/.claude/projects"] in tripletes
    assert not any(jaula[k:k + 2] == ["--tmpfs", "$HOME/.claude/projects"] for k in range(len(jaula)))


def test_ruta_projects_de_la_mision():
    ruta = CA.ruta_projects_de_la_mision(Path("/var/lib/jax-ejecutor-misiones"),
                                         "0b4e7a52-3c1d-4f7e-9a51-6f2d8e4c1a90")
    assert ruta == Path("/var/lib/jax-ejecutor-misiones/0b4e7a52-3c1d-4f7e-9a51-6f2d8e4c1a90/claude-projects")


@pytest.mark.parametrize("mision_id", ["../../etc", "m1; rm -rf ~", "0B4E7A52-3C1D-4F7E-9A51-6F2D8E4C1A90", ""])
def test_ruta_projects_de_la_mision_rechaza_un_id_que_no_es_uuid(mision_id):
    with pytest.raises(CA.MisionIdInvalido):
        CA.ruta_projects_de_la_mision(Path("/var/lib/jax-ejecutor-misiones"), mision_id)


# --- directorio de la cuenta por ACL, sin sudo (2026-09-28, Task 0) ------------------
#
# `jaxsvc` no tiene sudo: la versión con `sudo install -o axioma` dejó al Ejecutor sin
# arrancar desde el 2026-09-20 (`mision_servicio.py` llama esto antes de CADA turno). El
# dueño pasa a ser el proceso (jaxsvc); `axioma` entra por ACL, también en lo que se cree
# adentro (ACL por omisión).
#
# Fix round 1 (2026-09-28, hallazgos I-1/I-2 de la auditoría del escalón 3):
# - I-1: `setfacl --set` (no `-m`) FIJA la ACL entera de la hoja -- si sólo se agrega,
#   la hoja hereda `default:user:fruiz:rwx` de `/var/lib/jax-ejecutor-misiones`, y fruiz
#   (o cualquier proceso con su UID) puede leer y ESCRIBIR la sesión que `--resume`
#   reanuda: canal de inyección sobre un agente con SSH.
# - I-2: `mkdir(exist_ok=True)` acepta un symlink a directorio (y `setfacl` seguiría el
#   enlace) y también aceptaría un directorio YA EXISTENTE de otro dueño (m-1) sin que
#   nadie se entere. `os.lstat` (no sigue symlinks) después del `mkdir` exige directorio
#   real y dueño == `os.geteuid()`; cualquiera de las dos falla cerrado.

def _cuenta(tmp_path):
    return CA.Cuenta("axioma", 22, tmp_path / "k", tmp_path / "n", tmp_path / "l", tmp_path / "p", tmp_path / "h")


def _acl_esperada(nombre):
    return (f"u::rwx,g::---,o::---,m::rwx,u:{nombre}:rwx,"
            f"d:u::rwx,d:g::---,d:o::---,d:m::rwx,d:u:{nombre}:rwx")


def test_preparar_directorio_no_usa_sudo_y_pone_acl(tmp_path):
    llamadas = []

    class _Proc:
        returncode = 0

        async def communicate(self):
            return b"", b""

    async def correr(*argv, **kw):
        llamadas.append(argv)
        return _Proc()

    ruta = tmp_path / "m" / "claude-projects"
    asyncio.run(CA.preparar_directorio_de_la_cuenta(_cuenta(tmp_path), ruta, correr=correr))
    assert ruta.is_dir()
    assert all(a[0] != "sudo" for a in llamadas)
    assert ("setfacl", "--set", _acl_esperada("axioma"), str(ruta)) in llamadas


def test_preparar_directorio_falla_cerrado_si_setfacl_falla(tmp_path):
    class _Proc:
        returncode = 1

        async def communicate(self):
            return b"", b"setfacl: Operation not permitted"

    async def correr(*argv, **kw):
        return _Proc()

    with pytest.raises(RuntimeError, match="preparar_directorio_fallo"):
        asyncio.run(CA.preparar_directorio_de_la_cuenta(_cuenta(tmp_path), tmp_path / "x", correr=correr))


def test_preparar_directorio_rechaza_un_symlink_en_vez_de_directorio_real(tmp_path):
    """I-2: un symlink a directorio pasa `mkdir(exist_ok=True)` sin problema (Path.is_dir()
    sigue el enlace) y `setfacl` seguiría el enlace también (con `-P` incluso da rc 0 sin
    aplicar nada -- fallo abierto). `os.lstat` no sigue symlinks: lo detecta y falla
    cerrado ANTES de llamar a `setfacl`."""
    destino = tmp_path / "destino-real"
    destino.mkdir()
    ruta = tmp_path / "m" / "claude-projects"
    ruta.parent.mkdir()
    ruta.symlink_to(destino, target_is_directory=True)

    async def correr(*argv, **kw):
        raise AssertionError("no debe llegar a setfacl si la ruta es un symlink")

    with pytest.raises(RuntimeError, match="preparar_directorio_fallo"):
        asyncio.run(CA.preparar_directorio_de_la_cuenta(_cuenta(tmp_path), ruta, correr=correr))


def test_preparar_directorio_rechaza_una_ruta_ocupada_por_otra_cosa(tmp_path):
    """m-1 (vía I-2): un `claude-projects` previo de OTRO dueño no se puede fabricar sin
    sudo dentro de un test -- se simula con un archivo regular en la misma ruta. El punto
    es el mismo: lo que ya está ahí no es un directorio propio de este proceso, y el
    código tiene que fallar cerrado en vez de dejar que `mkdir`/`setfacl` sigan de largo."""
    ruta = tmp_path / "m" / "claude-projects"
    ruta.parent.mkdir()
    ruta.write_text("no soy un directorio")

    async def correr(*argv, **kw):
        raise AssertionError("no debe llegar a setfacl si la ruta no es un directorio propio")

    with pytest.raises(RuntimeError, match="preparar_directorio_fallo"):
        asyncio.run(CA.preparar_directorio_de_la_cuenta(_cuenta(tmp_path), ruta, correr=correr))


requiere_acl = pytest.mark.skipif(shutil.which("setfacl") is None or shutil.which("getfacl") is None,
                                  reason="setfacl/getfacl no están instalados en este runner")


@requiere_acl
def test_preparar_directorio_pone_la_acl_real_sin_heredar_del_padre(tmp_path):
    """I-1 con `setfacl`/`getfacl` REALES (no mockeados). El padre lleva una ACL por
    omisión ajena (UID 99999 -- no resuelve a ningún usuario del sistema, así `getfacl`
    no puede disfrazarlo con un nombre real y esconder el chequeo) -- lo que I-1 dice
    que la hoja NO debe heredar, porque el problema real era justo ese: heredar
    `default:user:fruiz:rwx` del padre (`/var/lib/jax-ejecutor-misiones`). `setfacl
    --set` tiene que reemplazar TODO, no sólo agregar."""
    nombre = getpass.getuser()
    padre = tmp_path / "padre"
    padre.mkdir()
    subprocess.run(["setfacl", "-d", "-m", "u:99999:rwx", str(padre)], check=True)
    ruta = padre / "claude-projects"

    c = CA.Cuenta(nombre, 22, tmp_path / "k", tmp_path / "n", tmp_path / "l", tmp_path / "p", tmp_path / "h")
    asyncio.run(CA.preparar_directorio_de_la_cuenta(c, ruta))

    salida = subprocess.run(["getfacl", "-p", str(ruta)], capture_output=True, text=True, check=True).stdout
    lineas = {l.strip() for l in salida.splitlines() if l.strip() and not l.startswith("#")}
    assert "user:99999:rwx" not in lineas and "default:user:99999:rwx" not in lineas
    assert f"user:{nombre}:rwx" in lineas
    assert f"default:user:{nombre}:rwx" in lineas
    assert "mask::rwx" in lineas
    assert "default:mask::rwx" in lineas


def test_preparar_directorio_projects_llama_a_setfacl_con_el_dueno_correcto(tmp_path):
    """Adaptado de `test_preparar_directorio_projects_llama_a_sudo_install_con_el_dueno_correcto`
    (2026-09-28): `preparar_directorio_projects` pasa a ser un alias de
    `preparar_directorio_de_la_cuenta` -- mismo contrato para los llamadores
    (`mision_servicio.py`), sin `sudo`. La ruta ya no puede ser un literal fuera de
    `tmp_path`: antes `sudo install` la creaba; ahora el propio proceso hace `mkdir`."""
    llamadas = []

    async def correr_falso(*argv, **kw):
        llamadas.append(argv)

        class ProcFalso:
            returncode = 0

            async def communicate(self):
                return b"", b""
        return ProcFalso()

    c = CA.cuenta_desde_entorno(ENV)
    ruta = tmp_path / "m1" / "claude-projects"
    asyncio.run(CA.preparar_directorio_projects(c, ruta, correr=correr_falso))
    assert ruta.is_dir()
    assert ("setfacl", "--set", _acl_esperada("axioma"), str(ruta)) in llamadas


def test_preparar_directorio_projects_propaga_el_fallo(tmp_path):
    async def correr_falso(*argv, **kw):
        class ProcFalso:
            returncode = 1

            async def communicate(self):
                return b"", b"permission denied"
        return ProcFalso()

    c = CA.cuenta_desde_entorno(ENV)
    with pytest.raises(RuntimeError):
        asyncio.run(CA.preparar_directorio_projects(c, tmp_path / "m1" / "claude-projects",
                                                     correr=correr_falso))


def test_la_jaula_tapa_los_includes_del_ssh_del_sistema():
    """Dentro del espacio de usuarios de bwrap los archivos de root se ven de 65534, y ssh rechaza
    un Include del sistema que no es de root («Bad owner or permissions»): sin tapar
    /etc/ssh/ssh_config.d el Ejecutor no puede entrar por ssh a NINGUNA máquina. Visto 2026-09-17
    en la misión de humo contra la VM desechable (hall9000 trae 20-systemd-ssh-proxy.conf)."""
    remoto = CA.remoto_claude(CA.cuenta_desde_entorno(ENV), base_url="http://127.0.0.1:18436", modelo="canario",
                              prompt="x")
    palabras = shlex.split(remoto.split(" && ", 1)[1].replace('"$K"', "K"))
    i = palabras.index("bwrap")
    jaula = palabras[i:palabras.index("--", i)]
    assert ["--tmpfs", "/etc/ssh/ssh_config.d"] in [jaula[k:k + 2] for k in range(len(jaula))]


def test_remoto_claude_con_tope_de_salida_lo_pasa_al_arnes():
    # SP3: el proxy rechaza `max_tokens` por encima de JAX_PROXY_CARRIL_MAX_SALIDA_TOKENS; un
    # arnés que pasa por el proxy tiene que pedir ese tope, o todas sus peticiones dan 403.
    c = CA.cuenta_desde_entorno(ENV)
    palabras = shlex.split(CA.remoto_claude(c, base_url="http://127.0.0.1:18436", modelo="canario", prompt="x",
                                            max_salida_tokens=1024).replace('"$K"', "K"))
    assert "CLAUDE_CODE_MAX_OUTPUT_TOKENS=1024" in palabras
    sin = CA.remoto_claude(c, base_url="http://127.0.0.1:18436", modelo="canario", prompt="x")
    assert "CLAUDE_CODE_MAX_OUTPUT_TOKENS" not in sin


def test_remoto_claude_crea_la_sesion_con_su_id_y_la_retoma_por_id():
    """SP2 (spec 2026-09-15 §5): el primer turno crea la sesión con el id que genera Axioma y los
    siguientes la retoman con ese mismo id. Sin sesión, ninguna de las dos banderas."""
    c = CA.cuenta_desde_entorno(ENV)
    sesion = "0b4e7a52-3c1d-4f7e-9a51-6f2d8e4c1a90"

    def palabras(**kw):
        return shlex.split(CA.remoto_claude(c, base_url="http://127.0.0.1:18436", modelo="m", prompt="x",
                                            **kw).replace('"$K"', "K"))
    nueva = palabras(sesion=sesion)
    assert nueva[nueva.index("--session-id") + 1] == sesion and "--resume" not in nueva
    retomada = palabras(sesion=sesion, reanudar=True)
    assert retomada[retomada.index("--resume") + 1] == sesion and "--session-id" not in retomada
    sin = palabras()
    assert "--session-id" not in sin and "--resume" not in sin


@pytest.mark.parametrize("sesion", ["x; rm -rf ~", "../../etc/passwd", "0B4E7A52-3C1D-4F7E-9A51-6F2D8E4C1A90", ""])
def test_remoto_claude_rechaza_una_sesion_que_no_es_un_uuid_canonico(sesion):
    with pytest.raises(ValueError):
        CA.remoto_claude(CA.cuenta_desde_entorno(ENV), base_url="http://127.0.0.1:18436", modelo="m", prompt="x",
                         sesion=sesion)


def test_remoto_claude_con_directorio_de_trabajo_entra_ahi_y_marca_el_clon_como_seguro():
    """Misión de código (spec 2026-09-28 v1.3 §3.2): el cerebro trabaja EN el clon, no en `~`.
    Medido 2026-09-28 con git 2.53 y los usuarios reales: sin `safe.directory`, `axioma` recibe
    «dubious ownership» sobre un clon cuyo directorio es de `jaxsvc`. Va por GIT_CONFIG_* en el
    entorno de la jaula (no en un archivo que la cuenta pueda reescribir)."""
    c = CA.cuenta_desde_entorno(ENV)
    remoto = CA.remoto_claude(c, base_url="http://127.0.0.1:1", modelo="m", prompt="p",
                              directorio_trabajo=Path("/var/lib/x/con espacio/repo"))
    assert remoto.startswith("read -r K; cd '/var/lib/x/con espacio/repo' && env ") and "cd ~" not in remoto
    palabras = shlex.split(remoto.split(" && ", 1)[1].replace('"$K"', "K"))
    entorno = palabras[1:palabras.index("bwrap")]
    assert entorno[-3:] == ["GIT_CONFIG_COUNT=1", "GIT_CONFIG_KEY_0=safe.directory",
                            "GIT_CONFIG_VALUE_0=/var/lib/x/con espacio/repo"]


def test_remoto_claude_sin_directorio_de_trabajo_no_toca_git():
    remoto = CA.remoto_claude(CA.cuenta_desde_entorno(ENV), base_url="http://127.0.0.1:1", modelo="m", prompt="p")
    assert remoto.startswith("read -r K; cd ~ && env ")
    assert "GIT_CONFIG" not in remoto and "safe.directory" not in remoto


def test_remoto_claude_rechaza_un_directorio_de_trabajo_relativo():
    with pytest.raises(ValueError, match="directorio_trabajo"):
        CA.remoto_claude(CA.cuenta_desde_entorno(ENV), base_url="http://127.0.0.1:1", modelo="m", prompt="p",
                         directorio_trabajo=Path("repo"))


def test_reanudar_sin_sesion_no_vale():
    with pytest.raises(ValueError):
        CA.remoto_claude(CA.cuenta_desde_entorno(ENV), base_url="http://127.0.0.1:18436", modelo="m", prompt="x",
                         reanudar=True)


# --- bwrap REAL (B-1/M-4, ronda 3, 2026-09-22) ---------------------------------------
#
# "Un test tiene que ejecutar bwrap de verdad, no comparar el texto del comando"
# (decisión del coordinador). Lo de arriba compara argv -- rápido, sin dependencias, pero
# no prueba que el kernel haga lo que el argv dice. Esto sí: arma una `Cuenta` de mentira
# con fuentes reales en `tmp_path` (nunca toca `/home/axioma` ni ninguna ruta fuera de
# este worktree) y corre `bwrap` de verdad, dos veces, para comprobar que lo escrito en
# la primera invocación NO existe en la segunda.

requiere_bwrap = pytest.mark.skipif(shutil.which("bwrap") is None, reason="bwrap no está instalado en este runner")


def _cuenta_de_prueba(tmp_path: Path) -> CA.Cuenta:
    home = tmp_path / "home-axioma"
    (home / ".ssh").mkdir(parents=True)
    (home / ".ssh" / "id_ed25519").write_text("llave de prueba, no una real")
    (home / ".ssh" / "known_hosts").write_text("")
    lib = tmp_path / "lib"
    (lib / "contexto" / "skills" / "una-skill").mkdir(parents=True)
    (lib / "contexto" / "skills" / "una-skill" / "SKILL.md").write_text("skill de prueba")
    (lib / "contexto" / "CLAUDE.md").write_text("identidad de prueba")
    (lib / "settings-usuario.json").write_text("{}\n")
    (lib / "managed-settings.json").write_text("{}\n")
    return CA.Cuenta("axioma", 58291, Path("/k"), Path("/n"), lib, tmp_path / "politica.json", home)


def _correr_en_la_jaula(c: CA.Cuenta, home_externo: Path, script: str, *, directorio_projects=None) -> str:
    jaula = CA._jaula(c, directorio_projects=directorio_projects)
    cmd = f"{jaula} bash -c {shlex.quote(script)}"
    r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=20,
                       env={**os.environ, "HOME": str(home_externo)})
    assert r.returncode == 0, (cmd, r.stdout, r.stderr)
    return r.stdout


@requiere_bwrap
def test_bwrap_real_nada_de_home_sobrevive_a_la_siguiente_invocacion(tmp_path):
    """El caso que el coordinador pidió literal: escribe ~/CLAUDE.local.md y
    ~/.claude/commands/x.md en una primera jaula; en la segunda, no existen."""
    c = _cuenta_de_prueba(tmp_path)
    home_externo = tmp_path / "home-externo"
    home_externo.mkdir()

    _correr_en_la_jaula(c, home_externo,
                        'echo basura > "$HOME/CLAUDE.local.md" && '
                        'mkdir -p "$HOME/.claude/commands" "$HOME/.claude/rules" "$HOME/.claude/agents" && '
                        'echo basura > "$HOME/.claude/commands/x.md" && '
                        'echo basura > "$HOME/.claude/rules/x.md" && '
                        'echo basura > "$HOME/.claude/agents/x.md" && '
                        'echo \'{"mcpServers": {"x": {}}}\' > "$HOME/.claude.json"')

    salida = _correr_en_la_jaula(c, home_externo,
                                 'for f in "$HOME/CLAUDE.local.md" "$HOME/.claude/commands/x.md" '
                                 '"$HOME/.claude/rules/x.md" "$HOME/.claude/agents/x.md" "$HOME/.claude.json"; '
                                 'do test -e "$f" && echo "SOBREVIVIO:$f" || echo "limpio:$f"; done')
    assert "SOBREVIVIO" not in salida, salida
    assert salida.count("limpio:") == 5, salida
    # Y en el HOME EXTERNO (fuera de la jaula) tampoco quedó nada: no se filtró para
    # afuera, se perdió adentro del tmpfs de la primera invocación.
    assert not (home_externo / "CLAUDE.local.md").exists()
    assert not (home_externo / ".claude").exists()


@requiere_bwrap
def test_bwrap_real_las_piezas_declaradas_si_estan_y_son_de_solo_lectura(tmp_path):
    c = _cuenta_de_prueba(tmp_path)
    home_externo = tmp_path / "home-externo2"
    home_externo.mkdir()
    salida = _correr_en_la_jaula(
        c, home_externo,
        'cat "$HOME/.claude/CLAUDE.md"; echo ---; cat "$HOME/.ssh/id_ed25519"; echo ---; '
        'ls "$HOME/.claude/skills"; echo ---; '
        'echo intento > "$HOME/.claude/CLAUDE.md" 2>/dev/null || echo "no_se_pudo_escribir"; '
        'touch "$HOME/.ssh/nueva" 2>/dev/null || echo "ssh_no_se_pudo_escribir"')
    assert "identidad de prueba" in salida
    assert "llave de prueba, no una real" in salida
    assert "una-skill" in salida
    assert "no_se_pudo_escribir" in salida
    assert "ssh_no_se_pudo_escribir" in salida


@requiere_bwrap
def test_bwrap_real_projects_persiste_dentro_de_la_misma_mision(tmp_path):
    """LÍMITE documentado: DENTRO de la misma misión, "$HOME/.claude/projects" SÍ
    sobrevive entre invocaciones (para que "--resume" funcione) -- lo único que se
    monta en lectura y escritura."""
    c = _cuenta_de_prueba(tmp_path)
    home_externo = tmp_path / "home-externo3"
    home_externo.mkdir()
    directorio_mision = tmp_path / "projects-de-la-mision"
    directorio_mision.mkdir()

    _correr_en_la_jaula(c, home_externo, 'echo sesion > "$HOME/.claude/projects/sesion-1.json"',
                        directorio_projects=directorio_mision)
    salida = _correr_en_la_jaula(c, home_externo, 'cat "$HOME/.claude/projects/sesion-1.json"',
                                 directorio_projects=directorio_mision)
    assert salida.strip() == "sesion"
    assert (directorio_mision / "sesion-1.json").read_text().strip() == "sesion"


@requiere_bwrap
def test_bwrap_real_sin_mision_projects_tampoco_sobrevive(tmp_path):
    c = _cuenta_de_prueba(tmp_path)
    home_externo = tmp_path / "home-externo4"
    home_externo.mkdir()
    _correr_en_la_jaula(c, home_externo, 'echo sesion > "$HOME/.claude/projects/sesion-1.json"')
    salida = _correr_en_la_jaula(c, home_externo,
                                 'test -e "$HOME/.claude/projects/sesion-1.json" && echo SOBREVIVIO || echo limpio')
    assert salida.strip() == "limpio"


# --- dar_acceso_recursivo: mismo patrón endurecido de Task 0, pero RECURSIVO --------
#
# (Ruling del controlador, plan "El Ejecutor programa", Tarea 6, 2026-09-28.) El clon de
# la misión de código (`preparar.preparar`, Tarea 6) lo crea `git clone`/`git checkout` --
# dueño el PROCESO (jaxsvc), no `axioma` -- y hay que dar acceso RECURSIVO a lo que ya
# existe adentro (a diferencia de `preparar_directorio_de_la_cuenta`, que sólo prepara un
# directorio vacío antes de que nada se escriba ahí). Mismo patrón: `setfacl` con `--set`
# fijo (no `-m`, no hereda del padre), `os.lstat` ANTES para rechazar symlink/dueño ajeno.
#
# Se agrega, además de `u:<cuenta>:rwX` y su `default` correspondiente, un
# `default:u:<usuario del proceso>:rwX` -- medido 2026-09-28 con los usuarios reales: sin
# esa entrada por omisión, `jaxsvc` deja de poder leer lo que `axioma` cree DESPUÉS
# (commits de Qwen) y la entrega (Tarea 5, corre como jaxsvc) no vería nada.

def _acl_recursiva_esperada(nombre_cuenta, usuario_proceso):
    return (f"u::rwX,g::---,o::---,m::rwX,u:{nombre_cuenta}:rwX,"
           f"d:u::rwX,d:g::---,d:o::---,d:m::rwX,d:u:{nombre_cuenta}:rwX,d:u:{usuario_proceso}:rwX")


def test_dar_acceso_recursivo_llama_setfacl_r_set_con_el_dueno_correcto(tmp_path):
    llamadas = []

    class _Proc:
        returncode = 0

        async def communicate(self):
            return b"", b""

    async def correr(*argv, **kw):
        llamadas.append(argv)
        return _Proc()

    ruta = tmp_path / "m1" / "repo"
    ruta.mkdir(parents=True)
    c = _cuenta(tmp_path)
    asyncio.run(CA.dar_acceso_recursivo(c, ruta, correr=correr))
    usuario_proceso = getpass.getuser()
    assert ("setfacl", "-P", "-R", "--set", _acl_recursiva_esperada("axioma", usuario_proceso), str(ruta)) in llamadas


def test_dar_acceso_recursivo_rechaza_un_symlink(tmp_path):
    destino = tmp_path / "destino-real"
    destino.mkdir()
    ruta = tmp_path / "m1" / "repo"
    ruta.parent.mkdir()
    ruta.symlink_to(destino, target_is_directory=True)

    async def correr(*argv, **kw):
        raise AssertionError("no debe llegar a setfacl si la ruta es un symlink")

    with pytest.raises(RuntimeError, match="preparar_directorio_fallo"):
        asyncio.run(CA.dar_acceso_recursivo(_cuenta(tmp_path), ruta, correr=correr))


requiere_acl_recursiva = pytest.mark.skipif(shutil.which("setfacl") is None or shutil.which("getfacl") is None,
                                            reason="setfacl/getfacl no están instalados en este runner")


@requiere_acl_recursiva
def test_dar_acceso_recursivo_pone_la_acl_real_recursiva_sin_heredar_del_padre(tmp_path):
    """`setfacl`/`getfacl` REALES. El padre lleva una ACL por omisión ajena (UID 99999,
    no resuelve a ningún usuario real) -- la hoja NO debe heredarla. Se crea un
    subdirectorio y un archivo DENTRO de `ruta` ANTES de llamar (simula lo que `git
    clone` deja) para comprobar que `-R` de verdad alcanza lo existente, y que la ACL
    por omisión de `ruta` (no la del padre) es la que gobierna lo que se cree DESPUÉS
    dentro del subdirectorio -- y que el usuario del PROCESO tiene su propia entrada por
    omisión."""
    nombre = getpass.getuser()
    padre = tmp_path / "padre"
    padre.mkdir()
    subprocess.run(["setfacl", "-d", "-m", "u:99999:rwx", str(padre)], check=True)
    ruta = padre / "repo"
    ruta.mkdir()
    (ruta / "archivo-ya-existente").write_text("de git clone")
    sub = ruta / "sub"
    sub.mkdir()

    c = CA.Cuenta(nombre, 22, tmp_path / "k", tmp_path / "n", tmp_path / "l", tmp_path / "p", tmp_path / "h")
    asyncio.run(CA.dar_acceso_recursivo(c, ruta))

    salida_ruta = subprocess.run(["getfacl", "-p", str(ruta)], capture_output=True, text=True,
                                 check=True).stdout
    lineas_ruta = {l.strip() for l in salida_ruta.splitlines() if l.strip() and not l.startswith("#")}
    assert "user:99999:rwx" not in lineas_ruta and "default:user:99999:rwx" not in lineas_ruta
    assert f"user:{nombre}:rwx" in lineas_ruta
    assert f"default:user:{nombre}:rwx" in lineas_ruta

    salida_sub = subprocess.run(["getfacl", "-p", str(sub)], capture_output=True, text=True, check=True).stdout
    lineas_sub = {l.strip() for l in salida_sub.splitlines() if l.strip() and not l.startswith("#")}
    assert "user:99999:rwx" not in lineas_sub and "default:user:99999:rwx" not in lineas_sub
    assert f"user:{nombre}:rwx" in lineas_sub, "la ACL -R no alcanzó lo que ya existía adentro"

    (sub / "nuevo-dentro").write_text("creado por axioma, simulado")
    salida_nuevo = subprocess.run(["getfacl", "-p", str(sub / "nuevo-dentro")], capture_output=True, text=True,
                                  check=True).stdout
    # El nuevo archivo hereda la entrada NAMED USER del default ACL de `sub/` verbatim
    # (getfacl la muestra como "user:<nombre>:rwx", con un comentario aparte
    # "#effective:rw-" cuando la máscara del propio archivo la recorta) -- se compara
    # sólo la parte antes del tabulador, sin el comentario de efectivo.
    lineas_nuevo = {l.split("\t", 1)[0].strip() for l in salida_nuevo.splitlines()
                    if l.strip() and not l.startswith("#")}
    assert f"user:{nombre}:rwx" in lineas_nuevo, (
        "lo creado DESPUÉS dentro de sub/ no heredó la entrada del propio proceso "
        f"({nombre}) del default ACL -- sin esto jaxsvc no podría leer lo que axioma escriba: "
        f"{lineas_nuevo}"
    )


# --- Retrabajo por auditoría (espejo privado, 2026-09-28) -------------------------------
#
# `setfacl -P` (nunca sigue un enlace dentro del árbol: el clon lleva `.venv` y
# `node_modules` enlazados a `deps/`), `permiso="r-X"` para `deps/` (la cuenta lee, no
# escribe), y `dar_paso` para `<raiz>/<misión>`: la cuenta solo atraviesa, sin ACL por
# omisión -- medido 2026-09-28: `setfacl --set` SIN entradas `d:` deja intacta la ACL por
# omisión heredada del padre; hace falta `-k`.

class _ProcOk:
    returncode = 0

    async def communicate(self):
        return b"", b""


def _correr_que_registra(llamadas):
    async def correr(*argv, **kw):
        llamadas.append(argv)
        return _ProcOk()
    return correr


def test_dar_acceso_recursivo_de_solo_lectura(tmp_path):
    llamadas = []
    ruta = tmp_path / "deps"
    ruta.mkdir()
    asyncio.run(CA.dar_acceso_recursivo(_cuenta(tmp_path), ruta, permiso="r-X", correr=_correr_que_registra(llamadas)))
    assert llamadas == [("setfacl", "-P", "-R", "--set",
                         "u::rwX,g::---,o::---,m::rwX,u:axioma:r-X,d:u::rwX,d:g::---,d:o::---,d:m::rwX,d:u:axioma:r-X",
                         str(ruta))]


def test_dar_acceso_recursivo_rechaza_un_permiso_desconocido(tmp_path):
    ruta = tmp_path / "x"
    ruta.mkdir()
    with pytest.raises(ValueError, match="permiso_invalido"):
        asyncio.run(CA.dar_acceso_recursivo(_cuenta(tmp_path), ruta, permiso="rwx,u:root:rwx",
                                            correr=_correr_que_registra([])))


def test_dar_paso_solo_atravesar_y_sin_acl_por_omision(tmp_path):
    llamadas = []
    ruta = tmp_path / "mision"
    ruta.mkdir()
    asyncio.run(CA.dar_paso(_cuenta(tmp_path), ruta, correr=_correr_que_registra(llamadas)))
    assert llamadas == [("setfacl", "-P", "-k", "--set", "u::rwx,g::---,o::---,m::--x,u:axioma:--x", str(ruta))]


def test_dar_paso_rechaza_un_symlink(tmp_path):
    destino = tmp_path / "destino"
    destino.mkdir()
    ruta = tmp_path / "mision"
    ruta.symlink_to(destino, target_is_directory=True)

    async def correr(*argv, **kw):
        raise AssertionError("no debe llegar a setfacl si la ruta es un symlink")
    with pytest.raises(RuntimeError, match="preparar_directorio_fallo"):
        asyncio.run(CA.dar_paso(_cuenta(tmp_path), ruta, correr=correr))


@requiere_acl_recursiva
def test_dar_paso_real_quita_la_acl_por_omision_heredada(tmp_path):
    nombre = getpass.getuser()
    padre = tmp_path / "padre"
    padre.mkdir()
    subprocess.run(["setfacl", "-d", "-m", "u:99999:rwx", str(padre)], check=True)
    ruta = padre / "mision"
    ruta.mkdir()
    c = CA.Cuenta(nombre, 22, tmp_path / "k", tmp_path / "n", tmp_path / "l", tmp_path / "p", tmp_path / "h")
    asyncio.run(CA.dar_paso(c, ruta))
    salida = subprocess.run(["getfacl", "-p", str(ruta)], capture_output=True, text=True, check=True).stdout
    lineas = {l.split("\t", 1)[0].strip() for l in salida.splitlines() if l.strip() and not l.startswith("#")}
    assert f"user:{nombre}:--x" in lineas and "other::---" in lineas
    assert not any(l.startswith("default:") for l in lineas), lineas
    (ruta / "nuevo").mkdir()
    salida = subprocess.run(["getfacl", "-p", str(ruta / "nuevo")], capture_output=True, text=True, check=True).stdout
    assert "99999" not in salida and "default:" not in salida
