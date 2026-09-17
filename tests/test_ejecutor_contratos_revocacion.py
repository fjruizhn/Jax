# tests/test_ejecutor_contratos_revocacion.py
"""Orquestación de C6: el orden, la clasificación y que «no pude comprobar» nunca
cuente como «revocada»."""
import asyncio

import pytest

from jax.ejecutor.contratos import revocacion as R
from jax.ejecutor.contratos.destinos import Host

HOSTS = (Host("hall9000", "127.0.0.1", 58291, "hypervisor", True), Host("atemai", "192.0.2.11", 58291, "desarrollo", False),
         Host("bridge", "192.0.2.20", 58291, "clientes", False))


@pytest.mark.parametrize("rc, stderr, esperado", [
    (0, b"", R.ENTRA),
    (255, b"axioma@192.0.2.11: Permission denied (publickey).\r\n", R.DENEGADO),
    (255, b"ssh: connect to host 192.0.2.11 port 58291: Connection timed out\n", R.INALCANZABLE),
    (255, b"ssh: connect to host 192.0.2.11 port 58291: Connection refused\n", R.INALCANZABLE),
    (1, b"", R.INALCANZABLE),
])
def test_clasificar(rc, stderr, esperado):
    assert R.clasificar_intento(rc, stderr) == esperado


def test_orden_remotas_primero_y_local_al_final():
    orden = []

    async def revocar_en(h):
        orden.append(("revocar", h.nombre))
        return 0, b"revocar=ok quitadas=1 quedan=0\n", b""

    async def probar(h):
        orden.append(("probar", h.nombre))
        return 255, b"", b"Permission denied (publickey)."

    res = asyncio.run(R.revocar_todas(HOSTS, revocar_en=revocar_en, probar_entrada=probar))
    assert [r.estado for r in res] == [R.REVOCADA] * 3
    locales = [i for i, (_, n) in enumerate(orden) if n == "hall9000"]
    assert min(locales) > max(i for i, (_, n) in enumerate(orden) if n != "hall9000")
    assert [r.host for r in res] == ["atemai", "bridge", "hall9000"]


def test_inalcanzable_no_cuenta_como_revocada_y_sigue_entrando_se_dice():
    async def revocar_en(h):
        return 0, b"revocar=ok quitadas=1 quedan=0\n", b""

    async def probar(h):
        return {"atemai": (255, b"", b"Connection timed out"), "bridge": (0, b"", b""),
                "hall9000": (255, b"", b"Permission denied (publickey).")}[h.nombre]

    res = {r.host: r.estado for r in asyncio.run(R.revocar_todas(HOSTS, revocar_en=revocar_en, probar_entrada=probar))}
    assert res == {"atemai": R.NO_VERIFICABLE, "bridge": R.SIGUE_ENTRANDO, "hall9000": R.REVOCADA}


def test_error_al_revocar_no_se_da_por_revocada():
    async def revocar_en(h):
        return (2, b"revocar=error codigo=sin_archivo\n", b"") if h.nombre == "bridge" else (0, b"revocar=ok quitadas=1 quedan=0\n", b"")

    async def probar(h):
        return 255, b"", b"Permission denied (publickey)."

    res = {r.host: r for r in asyncio.run(R.revocar_todas(HOSTS, revocar_en=revocar_en, probar_entrada=probar))}
    assert res["bridge"].estado == R.ERROR_AL_REVOCAR
    assert res["bridge"].detalle == (("salida", "revocar=error codigo=sin_archivo"),)


def test_una_excepcion_al_revocar_es_error_no_silencio():
    async def revocar_en(h):
        if h.nombre == "atemai":
            raise OSError("ssh no está")
        return 0, b"revocar=ok quitadas=1 quedan=0\n", b""

    async def probar(h):
        return 255, b"", b"Permission denied (publickey)."

    res = {r.host: r.estado for r in asyncio.run(R.revocar_todas(HOSTS, revocar_en=revocar_en, probar_entrada=probar))}
    assert res["atemai"] == R.ERROR_AL_REVOCAR


def test_comandos():
    h = HOSTS[2]
    assert R.remoto_probar_entrada(h, "axioma") == (
        "LC_ALL=C ssh -o BatchMode=yes -o PreferredAuthentications=publickey -o ConnectTimeout=5 "
        "-o StrictHostKeyChecking=yes -p 58291 axioma@192.0.2.20 true")
    assert R.argv_admin(h, "admin", "sudo -n true") == [
        "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=yes",
        "-p", "58291", "admin@192.0.2.20", "sudo -n true"]
