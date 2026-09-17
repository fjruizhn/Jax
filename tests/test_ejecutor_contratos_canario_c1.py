# tests/test_ejecutor_contratos_canario_c1.py
"""verificar_c1 con la cuenta y el arnés falsos: qué observación da cada Fallo.
Un control que no falla no valida: cada código tiene su caso."""
import asyncio
import json
from pathlib import Path

import pytest

from jax.ejecutor.contratos import canario_c1
from jax.ejecutor.contratos.canario_upstream import Resultado
from jax.ejecutor.contratos.cuenta_axioma import Cuenta
from jax.ejecutor.contratos.fallo import Fallo

C = Cuenta("axioma", 58291, Path("/k"), Path("/opt/node/bin"), Path("/opt/ejecutor/lib"), Path("/etc/p.json"))
NONCE = "abc123"
BLOQUEO = b'contrato="c1" codigo="prohibido" regla="canario_c1" hosts=["hall9000"]\n'


class Arnes:
    """Cuenta falsa. `rotura` elige qué observación sale mal."""

    def __init__(self, rotura=None):
        self.rotura = rotura
        self.upstream = None

    async def correr(self, cuenta, remoto, *, entrada=b"", tope_s):
        if remoto.endswith("autoprueba"):
            if self.rotura == "autoprueba":
                return 2, b'[{"regla":"x","indice":0,"esperado":"coincide","obtenido":[]}]', b""
            if self.rotura == "ilegible":
                return 2, b"", b'contrato="c1" codigo="politica_ilegible" motivo="sha256_no_cuadra"\n'
            return 0, b"[]", b""
        if "gancho.sh" in remoto:
            canario = b"ejecutor-canario-c1" in entrada
            if canario:
                if self.rotura == "directo_sin_regla":
                    return 2, b"", b'contrato="c1" codigo="politica_ilegible" motivo="sha256_no_cuadra"\n'
                return (0, b"", b"") if self.rotura == "directo" else (2, b"", BLOQUEO)
            return (2, b"", BLOQUEO) if self.rotura == "control_directo" else (0, b"", b"")
        if remoto.startswith("test -e"):
            return (0, b"", b"") if self.rotura == "ejecutado" else (1, b"", b"")
        # lanzamiento del arnés: simula lo que el upstream anotaría
        r = self.upstream.resultados
        if self.rotura != "no_llego":
            r[f"toolu_canario_{NONCE}"] = Resultado(
                f"toolu_canario_{NONCE}", self.rotura != "no_bloqueado",
                {"no_bloqueado": "", "sin_regla": "Error: permiso denegado"}.get(self.rotura, BLOQUEO.decode()))
            ok = self.rotura != "control"
            r[f"toolu_control_{NONCE}"] = Resultado(f"toolu_control_{NONCE}", not ok,
                                                    f"control-c1-{NONCE}" if ok else BLOQUEO.decode())
        return 0, b"", b""

    def upstream_falso(self, guion, host, puerto):
        arnes = self

        class _Up:
            async def __aenter__(self_inner):
                self_inner.resultados, self_inner.peticiones, self_inner.puerto = {}, [], puerto
                arnes.upstream = self_inner
                return self_inner

            async def __aexit__(self_inner, *exc):
                return False
        return _Up()


def _verificar(arnes):
    return asyncio.run(canario_c1.verificar_c1(C, puerto_canario=18436, correr=arnes.correr,
                                               upstream=arnes.upstream_falso, nonce=NONCE))


def test_todo_vivo_no_da_fallos():
    assert _verificar(Arnes()) == ()


@pytest.mark.parametrize("rotura, codigo", [
    ("autoprueba", "autoprueba_fallida"), ("ilegible", "politica_ilegible"),
    ("directo", "canario_directo_no_bloqueado"), ("directo_sin_regla", "canario_directo_sin_su_regla"),
    ("control_directo", "control_directo_bloqueado"),
    ("no_llego", "claude_no_llego_al_canario"), ("no_bloqueado", "canario_no_bloqueado"),
    ("sin_regla", "canario_sin_su_regla"),
    ("control", "control_no_ejecutado"), ("ejecutado", "canario_ejecutado"),
])
def test_cada_rotura_da_su_codigo(rotura, codigo):
    fallos = _verificar(Arnes(rotura))
    assert codigo in [f.codigo for f in fallos], fallos
    assert all(f.contrato == "c1" for f in fallos)


def test_cuenta_inalcanzable():
    async def caida(*a, **k):
        return 255, b"", b"ssh: connect to host 127.0.0.1 port 58291: Connection refused\n"
    fallos = asyncio.run(canario_c1.verificar_c1(C, puerto_canario=18436, correr=caida,
                                                 upstream=Arnes().upstream_falso, nonce=NONCE))
    assert fallos == (Fallo("c1", "cuenta_inalcanzable", (("rc", 255),)),)
