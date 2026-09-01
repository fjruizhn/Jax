"""Tests del tripwire de OLLAMA_NUM_PARALLEL -- probado ROMPIENDOLO.

El punto de estos tests no es que el script "corra". Es que **el tripwire se
pone rojo cuando debe**. Un tripwire que nunca se vio fallar no es
distinguible de uno que no existe (primera leccion de metodo de CONTEXT.md:
el CI en verde no es evidencia de que el CI funcione).

Que verifica CI, y que NO
-------------------------
En `ubuntu-latest` no hay ningun Ollama, y el Ollama de hall9000 corre como
usuario `ollama` con /proc/<pid>/environ en 0400 -- leerlo exige root. Asi
que CI **no puede** verificar el invariante en produccion, y no se finge que
si. Lo que CI verifica es que el instrumento funciona, y lo hace sobre
**procesos vivos de verdad**: se lanza un proceso real con
OLLAMA_NUM_PARALLEL=2 y se le apunta el checker a su /proc/<pid>/environ
real. Es el mismo camino de lectura, el mismo parser y el mismo veredicto
que corre contra Ollama -- lo unico distinto es cual es el proceso.

Deliberadamente NO se mockea `Path.read_bytes` ni se fabrica un environ en
memoria para el caso principal: el arnes controlaria exactamente el termino
bajo prueba, y el resultado dejaria de ser evidencia sobre el ataque
(vigesima primera leccion de metodo).
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_ollama_num_parallel import (  # noqa: E402
    EXPECTED,
    VAR,
    TripwireError,
    check_environ,
    is_ollama_serve,
    main,
    parse_environ,
    read_environ,
)

SCRIPT = Path(__file__).resolve().parent / "check_ollama_num_parallel.py"


# --------------------------------------------------------------------------
# Procesos VIVOS de verdad. Este es el bloque que importa: ejercita el camino
# real de lectura de /proc, no una simulacion de el.
# --------------------------------------------------------------------------

def _spawn_with(value: str | None) -> subprocess.Popen:
    env = dict(os.environ)
    env.pop(VAR, None)
    if value is not None:
        env[VAR] = value
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # Esperar a que el environ sea legible antes de medir: un proceso recien
    # forkeado puede no tenerlo poblado todavia.
    deadline = time.time() + 5
    while time.time() < deadline:
        try:
            if Path(f"/proc/{proc.pid}/environ").read_bytes():
                return proc
        except OSError:
            pass
        time.sleep(0.02)
    proc.kill()
    raise AssertionError("el proceso de prueba no expuso su environ en 5s")


@pytest.fixture
def live_proc():
    procs: list[subprocess.Popen] = []

    def _make(value: str | None):
        proc = _spawn_with(value)
        procs.append(proc)
        return proc

    yield _make
    for proc in procs:
        proc.kill()
        proc.wait(timeout=10)


def test_proceso_vivo_con_1_da_VERDE(live_proc):
    """El caso sano, sobre un proceso real: exit 0."""
    proc = live_proc("1")
    rc = main(["--environ-file", f"/proc/{proc.pid}/environ"])
    assert rc == 0


def test_proceso_vivo_con_2_da_ROJO(live_proc):
    """LA MUTACION. Mismo camino, mismo parser, valor roto: exit 1."""
    proc = live_proc("2")
    rc = main(["--environ-file", f"/proc/{proc.pid}/environ"])
    assert rc == 1


def test_proceso_vivo_sin_la_variable_da_ROJO(live_proc):
    """Ausente no es 1. Sin el valor, Ollama usa su default, que no es 1."""
    proc = live_proc(None)
    rc = main(["--environ-file", f"/proc/{proc.pid}/environ"])
    assert rc == 1


def test_el_script_como_subproceso_propaga_el_exit_code(live_proc):
    """Lo que CI observa es el exit code del script, no el return de main()."""
    bad = live_proc("2")
    good = live_proc("1")
    rojo = subprocess.run(
        [sys.executable, str(SCRIPT), "--environ-file", f"/proc/{bad.pid}/environ"],
        capture_output=True, text=True,
    )
    verde = subprocess.run(
        [sys.executable, str(SCRIPT), "--environ-file", f"/proc/{good.pid}/environ"],
        capture_output=True, text=True,
    )
    assert rojo.returncode == 1
    assert "ROJO" in rojo.stderr
    assert verde.returncode == 0
    assert "VERDE" in verde.stdout


def test_proceso_muerto_da_ROJO(live_proc):
    """El /proc de un proceso muerto desaparece. Eso es rojo, no 'no aplica'."""
    proc = live_proc("1")
    path = f"/proc/{proc.pid}/environ"
    proc.kill()
    proc.wait(timeout=10)
    deadline = time.time() + 5
    while Path(path).exists() and time.time() < deadline:
        time.sleep(0.02)
    assert main(["--environ-file", path]) == 1


# --------------------------------------------------------------------------
# Fail-closed: todo camino que NO pudo medir tiene que ser rojo.
# --------------------------------------------------------------------------

def test_archivo_inexistente_es_ROJO(tmp_path):
    with pytest.raises(TripwireError, match="no existe"):
        read_environ(tmp_path / "no-esta")


def test_environ_vacio_es_ROJO(tmp_path):
    vacio = tmp_path / "environ"
    vacio.write_bytes(b"")
    with pytest.raises(TripwireError, match="vacio"):
        read_environ(vacio)


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignora los permisos de archivo")
def test_sin_permiso_es_ROJO(tmp_path):
    """El caso de hall9000: Ollama corre como otro usuario."""
    prohibido = tmp_path / "environ"
    prohibido.write_bytes(f"{VAR}=1\0".encode())
    prohibido.chmod(0o000)
    try:
        with pytest.raises(TripwireError, match="sin permiso"):
            read_environ(prohibido)
    finally:
        prohibido.chmod(0o600)


# --------------------------------------------------------------------------
# El invariante, aislado.
# --------------------------------------------------------------------------

def test_exactamente_1_es_lo_unico_verde():
    check_environ({VAR: EXPECTED})  # no lanza


@pytest.mark.parametrize("valor", ["2", "0", "", " 1", "1 ", "01", "4", "-1", "uno"])
def test_cualquier_otro_valor_es_ROJO(valor):
    """Incluye los que 'se parecen' a 1: '01' y ' 1' NO son 1 para Ollama."""
    with pytest.raises(TripwireError, match="se esperaba"):
        check_environ({VAR: valor})


def test_variable_ausente_es_ROJO():
    with pytest.raises(TripwireError, match="no esta en el entorno"):
        check_environ({"OLLAMA_MODELS": "/srv/jax-data/ollama-models"})


def test_no_se_confunde_con_una_variable_de_nombre_parecido():
    with pytest.raises(TripwireError, match="no esta en el entorno"):
        check_environ({f"{VAR}_OLD": "1", f"X_{VAR}": "1"})


# --------------------------------------------------------------------------
# Parser del formato de /proc.
# --------------------------------------------------------------------------

def test_parser_respeta_los_iguales_dentro_del_valor():
    """Un environ real los tiene (LS_COLORS, DBUS_*). split('=') los rompe."""
    env = parse_environ(b"A=b=c\0LS_COLORS=rs=0:di=01;34\0")
    assert env["A"] == "b=c"
    assert env["LS_COLORS"] == "rs=0:di=01;34"


def test_parser_descarta_entradas_sin_igual_y_nulos_de_sobra():
    env = parse_environ(b"\0SIN_IGUAL\0" + f"{VAR}=1".encode() + b"\0\0")
    assert env == {VAR: "1"}


def test_parser_sobre_un_environ_real_de_este_mismo_proceso():
    """Contra el formato de verdad, no contra la idea que tengo de el."""
    env = parse_environ(Path(f"/proc/{os.getpid()}/environ").read_bytes())
    assert env.get("PATH") == os.environ["PATH"]


# --------------------------------------------------------------------------
# Descubrimiento del proceso. NO se puede ejercitar entero en CI -- en
# ubuntu-latest no hay ningun Ollama. Lo que si se puede, y es donde vive el
# riesgo real, es el predicado que decide QUE proceso es el servidor: un
# falso positivo aca no da un error ruidoso, da un veredicto sobre el proceso
# equivocado. Las cmdlines de abajo son reales, tomadas de hall9000.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("cmdline", [
    "/usr/local/bin/ollama serve",
    "ollama serve",
])
def test_reconoce_al_servidor_de_ollama(cmdline):
    assert is_ollama_serve(cmdline) is True


@pytest.mark.parametrize("cmdline", [
    # El falso positivo OBSERVADO al medir este item: pgrep -f se engancho a
    # la propia linea de bash que lo estaba invocando.
    "/bin/bash -c pgrep -a -f \"ollama serve\"",
    # Otros clientes de ollama que NO son el servidor.
    "/usr/local/bin/ollama run qwen3-coder",
    "/usr/local/bin/ollama ps",
    "python3 scripts/check_ollama_num_parallel.py",
    "/usr/local/bin/ollama",
    "",
    "   ",
    # Un ejecutable cuyo nombre TERMINA en ollama pero no lo es.
    "/opt/fake/notollama serve",
])
def test_no_confunde_a_otros_procesos_con_el_servidor(cmdline):
    assert is_ollama_serve(cmdline) is False
