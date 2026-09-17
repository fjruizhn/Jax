# tests/test_ejecutor_contratos_cuenta_axioma.py
"""Cómo se entra a la cuenta del Ejecutor: todo desde el entorno, sin defaults; la
llave del cerebro nunca en argv; el lanzamiento siempre dentro de la jaula
superpuesta con los settings de solo lectura."""
import shlex
from pathlib import Path

import pytest

from jax.ejecutor.contratos import cuenta_axioma as CA

ENV = {
    "JAX_EJECUTOR_CUENTA": "axioma", "JAX_EJECUTOR_SSH_PUERTO": "58291",
    "JAX_EJECUTOR_CONTROLADOR_LLAVE": "/home/fruiz/.ssh/id_ejecutor_controlador",
    "JAX_EJECUTOR_NODE_BIN": "/opt/ejecutor/node-v24.16.0/bin", "JAX_EJECUTOR_LIB": "/opt/ejecutor/lib",
    "JAX_EJECUTOR_POLITICA": "/etc/jax-ejecutor/politica.json",
}


def test_cuenta_desde_el_entorno():
    c = CA.cuenta_desde_entorno(ENV)
    assert c == CA.Cuenta("axioma", 58291, Path("/home/fruiz/.ssh/id_ejecutor_controlador"),
                          Path("/opt/ejecutor/node-v24.16.0/bin"), Path("/opt/ejecutor/lib"),
                          Path("/etc/jax-ejecutor/politica.json"))


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
