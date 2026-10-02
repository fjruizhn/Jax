"""El Faro: las dependencias del Puerto y las de CI se fijan CON HASH (plan 0.2; cadena de suministro, skill
`endureciendo`). `requirements-faro.txt` (mcp) y `requirements-faro-ci.txt` (pytest, cliente de MariaDB) son la
fuente y el CI instala con `--require-hashes`: un paquete cuyo contenido cambie en PyPI no se instala."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
ARCHIVOS = ("requirements-faro.txt", "requirements-faro-ci.txt")


def _bloques(nombre: str) -> dict[str, list[str]]:
    """{nombre==version: [hashes]} de cada requisito."""
    fijados: dict[str, list[str]] = {}
    actual = None
    for linea in (RAIZ / nombre).read_text(encoding="utf-8").splitlines():
        if not linea.strip() or linea.lstrip().startswith("#"):
            continue
        m = re.match(r"^([A-Za-z0-9_.\-]+)==([^\s\\]+)\s*\\?$", linea)
        if m:
            actual = f"{m.group(1).lower()}=={m.group(2)}"
            fijados[actual] = []
        elif linea.strip().startswith("--hash=sha256:") and actual:
            fijados[actual].append(linea.strip().split(":", 1)[1].rstrip(" \\"))
        else:
            raise AssertionError(f"{nombre}: linea que no es ni requisito fijado ni hash: {linea!r}")
    return fijados


def _job(nombre: str) -> str:
    lineas = (RAIZ / ".github" / "workflows" / "policy.yml").read_text(encoding="utf-8").splitlines()
    inicio = lineas.index(f"  {nombre}:")
    fin = next((i for i in range(inicio + 1, len(lineas)) if re.match(r"^  [a-z0-9-]+:\s*$", lineas[i])), len(lineas))
    return "\n".join(lineas[inicio:fin])


def test_mcp_esta_fijado_a_la_version_verificada():
    assert "mcp==2.2.0" in _bloques("requirements-faro.txt")


def test_pytest_y_el_cliente_de_base_estan_fijados_en_el_archivo_de_ci():
    fijados = _bloques("requirements-faro-ci.txt")
    assert {n.split("==")[0] for n in fijados} >= {"pytest", "aiomysql", "pymysql", "pluggy", "iniconfig", "packaging"}


@pytest.mark.parametrize("archivo", ARCHIVOS)
def test_todo_requisito_esta_fijado_con_version_exacta_y_con_hashes(archivo):
    fijados = _bloques(archivo)
    assert len(fijados) >= 6
    assert [n for n, hs in fijados.items() if not hs] == []
    assert all(re.fullmatch(r"[0-9a-f]{64}", h) for hs in fijados.values() for h in hs)


@pytest.mark.parametrize("job", ["faro-fase0", "faro-bitacora-db"])
def test_el_ci_instala_todo_con_require_hashes_y_sin_pip_suelto(job):
    texto = _job(job)
    assert "pip install --require-hashes -r requirements-faro.txt" in texto
    assert "pip install --require-hashes -r requirements-faro-ci.txt" in texto
    sueltos = [l.strip() for l in texto.splitlines() if "pip install" in l and "--require-hashes" not in l]
    assert sueltos == [], f"pip install sin hashes: {sueltos}"
