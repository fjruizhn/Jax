"""El Faro: las dependencias del Puerto se fijan CON HASH (plan 0.2; Principio de cadena de
suministro de `endureciendo`). `requirements-faro.txt` es la fuente y el CI instala con
`--require-hashes`: un paquete cuyo contenido cambie en PyPI no se instala."""
from __future__ import annotations

import re
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
ARCHIVO = RAIZ / "requirements-faro.txt"


def _bloques() -> dict[str, list[str]]:
    """{nombre: [hashes]} de cada requisito, con la linea del requisito `nombre==version \\`."""
    fijados: dict[str, list[str]] = {}
    actual = None
    for linea in ARCHIVO.read_text(encoding="utf-8").splitlines():
        if not linea.strip() or linea.lstrip().startswith("#"):
            continue
        m = re.match(r"^([A-Za-z0-9_.\-]+)==([^\s\\]+)\s*\\?$", linea)
        if m:
            actual = f"{m.group(1).lower()}=={m.group(2)}"
            fijados[actual] = []
        elif linea.strip().startswith("--hash=sha256:") and actual:
            fijados[actual].append(linea.strip().split(":", 1)[1].rstrip(" \\"))
        else:
            raise AssertionError(f"linea que no es ni requisito fijado ni hash: {linea!r}")
    return fijados


def test_mcp_esta_fijado_a_la_version_verificada():
    assert "mcp==2.2.0" in _bloques()


def test_todo_requisito_esta_fijado_con_version_exacta_y_con_hashes():
    fijados = _bloques()
    assert len(fijados) >= 10
    sin_hash = [n for n, hs in fijados.items() if not hs]
    assert sin_hash == []
    assert all(re.fullmatch(r"[0-9a-f]{64}", h) for hs in fijados.values() for h in hs)


def test_el_ci_instala_el_puerto_con_require_hashes():
    texto = (RAIZ / ".github" / "workflows" / "policy.yml").read_text(encoding="utf-8")
    lineas = texto.splitlines()
    inicio = lineas.index("  faro-fase0:")
    fin = next((i for i in range(inicio + 1, len(lineas)) if re.match(r"^  [a-z0-9-]+:\s*$", lineas[i])), len(lineas))
    job = "\n".join(lineas[inicio:fin])
    assert "pip install --require-hashes -r requirements-faro.txt" in job
