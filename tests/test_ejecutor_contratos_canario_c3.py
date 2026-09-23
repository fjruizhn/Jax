# tests/test_ejecutor_contratos_canario_c3.py
"""verificar_c3 con la cuenta falsa: cada forma en que C3 puede estar roto da su código."""
import asyncio
from pathlib import Path

import pytest

from jax.ejecutor.contratos import canario_c3
from jax.ejecutor.contratos import registro as R
from jax.ejecutor.contratos.cuenta_axioma import Cuenta
from jax.ejecutor.contratos.fallo import Fallo

C = Cuenta("axioma", 58291, Path("/k"), Path("/n"), Path("/l"), Path("/p"), Path("/home/axioma"))


def _registro(tmp_path):
    ruta = tmp_path / "registro.jsonl"
    reg = R.Registro(ruta)
    reg.anotar({"evento": "x"})
    reg.cerrar()
    return ruta


def _cuenta(abiertas=(), proxy=True, escribe=False, rc=0):
    async def correr(cuenta, remoto, *, entrada=b"", tope_s):
        if rc:
            return rc, b"", b"ssh: connect refused"
        lineas = [f"sonda={p} {'abierta' if p in abiertas else 'cerrada'}" for p in (7777, 11434)]
        lineas.append(f"proxy={'abierto' if proxy else 'cerrado'}")
        lineas.append(f"registro_escrito={'si' if escribe else 'no'}")
        return 0, ("\n".join(lineas) + "\n").encode(), b""
    return correr


def _verificar(tmp_path, correr, banderas=canario_c3.FS_APPEND_FL):
    return asyncio.run(canario_c3.verificar_c3(C, registro=_registro(tmp_path), puerto_proxy=18435,
                                               sondas=(7777, 11434), correr=correr, leer_banderas=lambda r: banderas))


def test_todo_vivo(tmp_path):
    assert _verificar(tmp_path, _cuenta()) == ()


@pytest.mark.parametrize("correr, banderas, esperado", [
    (_cuenta(abiertas=(7777,)), canario_c3.FS_APPEND_FL, Fallo("c3", "cerco_abierto", (("puerto", 7777),))),
    (_cuenta(proxy=False), canario_c3.FS_APPEND_FL, Fallo("c3", "proxy_inalcanzable")),
    (_cuenta(escribe=True), canario_c3.FS_APPEND_FL, Fallo("c3", "registro_escribible_desde_la_jaula")),
    (_cuenta(), 0, Fallo("c3", "registro_sin_append_only")),
])
def test_cada_rotura(tmp_path, correr, banderas, esperado):
    assert esperado in _verificar(tmp_path, correr, banderas)


def test_cuenta_inalcanzable(tmp_path):
    assert _verificar(tmp_path, _cuenta(rc=255)) == (Fallo("c3", "cuenta_inalcanzable", (("rc", 255),)),)


def test_registro_que_cambia_durante_la_sonda(tmp_path):
    ruta = _registro(tmp_path)

    async def correr(cuenta, remoto, *, entrada=b"", tope_s):
        with open(ruta, "ab") as f:
            f.write(b"x")
        return 0, b"sonda=7777 cerrada\nsonda=11434 cerrada\nproxy=abierto\nregistro_escrito=no\n", b""

    fallos = asyncio.run(canario_c3.verificar_c3(C, registro=ruta, puerto_proxy=18435, sondas=(7777, 11434),
                                                 correr=correr, leer_banderas=lambda r: canario_c3.FS_APPEND_FL))
    assert Fallo("c3", "registro_cambio_desde_la_jaula") in fallos
    assert any(f.codigo == "cadena_rota" for f in fallos)


def test_banderas_lee_un_archivo_real(tmp_path):
    ruta = tmp_path / "f"
    ruta.write_text("x")
    assert canario_c3.banderas(ruta) & canario_c3.FS_APPEND_FL == 0
