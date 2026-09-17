# tests/test_ejecutor_contratos_destinos.py
"""¿A qué máquinas del inventario toca un comando? (C1/C2, plan 1).

Honesto a propósito: lee ssh/scp/rsync/sftp escritos a la vista. Lo que tiene
que garantizar es fallar CERRADO: destino fuera del inventario o comando que no
se puede partir -> excepción, nunca "no toca nada"."""
import pytest

from jax.ejecutor.contratos import formato
from jax.ejecutor.contratos.destinos import ComandoIlegible, Host, HostDesconocido, destinos

HOSTS = (
    Host("hall9000", "192.0.2.5", 58291, "hypervisor", True),
    Host("atemai", "192.0.2.11", 58291, "desarrollo", False),
    Host("prod", "192.0.2.10", 58291, "produccion", False),
    Host("bridge", "192.0.2.20", 58291, "clientes", False),
)


@pytest.mark.parametrize("comando, esperado", [
    ("uptime", {"hall9000"}),
    ("", {"hall9000"}),
    ("ssh -tt -p 58291 axioma@192.0.2.20 uptime", {"bridge"}),
    ("ssh -tt axioma@bridge 'sudo apt list'", {"bridge"}),
    ("ssh -tt -l axioma prod df -h", {"prod"}),
    ("ssh -tt -p58291 -o BatchMode=yes axioma@atemai", {"atemai"}),
    ("uptime && ssh -tt axioma@prod df", {"hall9000", "prod"}),
    ("ssh -tt axioma@bridge 'ssh -tt axioma@prod uptime'", {"bridge", "prod"}),
    ("sudo -u root ssh -tt axioma@prod uptime", {"prod"}),
    ("FOO=1 timeout 5 ssh -tt axioma@prod uptime", {"prod"}),
    ("/usr/bin/ssh -tt axioma@prod uptime", {"prod"}),
    ("ssh -tt ssh://axioma@prod:58291 uptime", {"prod"}),
    ("scp -P 58291 informe.txt axioma@bridge:/tmp/", {"hall9000", "bridge"}),
    ("rsync -a -e 'ssh -p 58291' axioma@192.0.2.11:/srv/x/ ./x/", {"hall9000", "atemai"}),
    ("grep ssh /var/log/auth.log", {"hall9000"}),
    ("echo a:b", {"hall9000"}),
])
def test_destinos(comando, esperado):
    assert destinos(comando, HOSTS) == frozenset(esperado)


@pytest.mark.parametrize("comando", [
    "ssh -tt axioma@192.0.2.99 uptime",
    "ssh -tt axioma@desconocido uptime",
    "ssh -tt localhost uptime",
    "scp x axioma@otra:/tmp/",
    "ssh -tt axioma@bridge 'ssh -tt axioma@192.0.2.77 uptime'",
])
def test_destino_fuera_del_inventario_falla_cerrado(comando):
    with pytest.raises(HostDesconocido):
        destinos(comando, HOSTS)


@pytest.mark.parametrize("comando", ["echo 'sin cerrar", "ssh -tt", "ssh -p 58291"])
def test_comando_que_no_se_puede_partir_falla_cerrado(comando):
    with pytest.raises(ComandoIlegible):
        destinos(comando, HOSTS)


def test_anidamiento_excesivo_falla_cerrado():
    comando = "uptime"
    for _ in range(5):
        comando = f"ssh -tt axioma@prod {formato.valor(comando)}"
    with pytest.raises(ComandoIlegible):
        destinos(comando, HOSTS)


def test_inventario_sin_una_sola_local_falla_cerrado():
    sin_local = tuple(Host(h.nombre, h.ip, h.puerto, h.rol, False) for h in HOSTS)
    with pytest.raises(HostDesconocido):
        destinos("uptime", sin_local)
    with pytest.raises(HostDesconocido):
        destinos("uptime", HOSTS + (Host("otra", "192.0.2.6", 22, "hypervisor", True),))


def test_formato_neutro():
    assert formato.campos((("codigo", "prohibido"), ("hosts", ("bridge",)), ("regla", None), ("n", 3), ("ok", True))) \
        == 'codigo="prohibido" hosts=["bridge"] regla=null n=3 ok=true'
    assert formato.campos((("x", "ñ \"y\""),)) == 'x="\\u00f1 \\"y\\""'
