# tests/test_ejecutor_contratos_cuenta_axioma.py
"""Cómo se entra a la cuenta del Ejecutor: todo desde el entorno, sin defaults; la
llave del cerebro nunca en argv; el lanzamiento siempre dentro de la jaula
superpuesta con los settings de solo lectura."""
import asyncio
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


def test_preparar_directorio_projects_llama_a_sudo_install_con_el_dueno_correcto():
    vistos = {}

    async def correr_falso(*argv, **kw):
        vistos["argv"] = argv

        class ProcFalso:
            returncode = 0

            async def communicate(self):
                return b"", b""
        return ProcFalso()

    c = CA.cuenta_desde_entorno(ENV)
    ruta = Path("/var/lib/jax-ejecutor-misiones/m1/claude-projects")
    asyncio.run(CA.preparar_directorio_projects(c, ruta, correr=correr_falso))
    assert vistos["argv"] == ("sudo", "install", "-d", "-o", "axioma", "-g", "axioma", "-m", "0700", str(ruta))


def test_preparar_directorio_projects_propaga_el_fallo():
    async def correr_falso(*argv, **kw):
        class ProcFalso:
            returncode = 1

            async def communicate(self):
                return b"", b"permission denied"
        return ProcFalso()

    c = CA.cuenta_desde_entorno(ENV)
    with pytest.raises(RuntimeError):
        asyncio.run(CA.preparar_directorio_projects(c, Path("/var/lib/jax-ejecutor-misiones/m1/claude-projects"),
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
