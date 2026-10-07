"""El catalogo sellado de un PIN DE PRUEBA (r7, MAJOR-1).

Desde la r7, ``CatalogoTopes`` solo lo emite el snapshot del pin: ninguna
prueba puede fabricarlo de bytes sueltos. Este helper monta un repo git minimo
con ``policy/faro/catalogo-topes.json`` (por defecto el del ARBOL HEAD del
repositorio — nunca el disco suelto), evalua ``load_trusted_policy_snapshot``
y devuelve su catalogo sellado. No es un archivo de test (no empieza por
``test_``): es utileria compartida, como ``tests/_faro_utils.py``.
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from policy.rule_authority.snapshot import TrustedPolicyPin, load_trusted_policy_snapshot

RAIZ = Path(__file__).resolve().parents[2]


def catalogo_del_pin(bytes_catalogo: bytes | None = None):
    """El CatalogoTopes sellado de un pin de prueba. Con ``bytes_catalogo``
    custom (p. ej. la decision de otra prueba); sin el, el catalogo del arbol
    HEAD del repositorio — el que la CI protege como literal exacto."""
    if bytes_catalogo is None:
        bytes_catalogo = subprocess.run(
            ["git", "show", "HEAD:policy/faro/catalogo-topes.json"],
            capture_output=True, check=True, cwd=RAIZ).stdout

    def g(*args: str) -> str:
        r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True)
        assert r.returncode == 0, r.stderr.decode()
        return r.stdout.decode().strip()

    repo = Path(tempfile.mkdtemp(prefix="catalogo-pin-"))
    g("init", "-q", "-b", "main")
    (repo / "policy" / "faro").mkdir(parents=True)
    (repo / "policy" / "faro" / "catalogo-topes.json").write_bytes(bytes_catalogo)
    g("add", "-A")
    g("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "catalogo de prueba")
    pin = TrustedPolicyPin("prueba", g("rev-parse", "HEAD"),
                           g("rev-parse", "HEAD:policy"), "prueba:pin")
    return load_trusted_policy_snapshot(repo, pin).catalogo
