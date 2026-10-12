from __future__ import annotations

from jax.faro.puente_http import construir_argv_puente
from jax.faro.jaula import perfil_motor_proxy


def test_puente_no_recibe_credencial_real_ni_directorio_de_socket():
    argv = construir_argv_puente("/faro/proxy.sock", "/faro/proxy-token", "claude")
    assert argv[:3] == ["/usr/bin/python3", "-m", "jax.faro.puente_http"]
    assert "/faro/proxy.sock" in argv and "/faro/proxy-token" in argv
    assert "clave-real" not in " ".join(argv)


def test_perfil_motor_proxy_monta_solo_los_dos_inodos_del_proxy():
    argv = perfil_motor_proxy(
        "/run/faro/run.sock", "/run/faro/run.token",
        "/run/faro/run.proxy.sock", "/run/faro/run.proxy.token",
        workspace="/srv/work", paquete="/srv/paquete", red="aislada",
    )
    assert "--unshare-net" in argv
    assert "/run/faro" not in argv
    assert "/run/faro/run.proxy.sock" in argv
    assert "/run/faro/run.proxy.token" in argv

