"""C1 de código aplicado al DIFF en la entrega (spec 2026-09-28 §5.1, DC7). Es la que manda:
el gancho de la jaula ataja errores honestos; esto revisa lo que de verdad saldría."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath

from jax.ejecutor.codigo.diff import Cambio

ARCHIVOS_DE_FERNANDO = {
    "fjruizhn/claude-skills": ("common/firmantes-fernando", "bin/go-fernando", "common/CLAUDE.md.core",
                               "common/secciones-constitucionales.txt", "common/hooks/", "bin/codex-bridge.py",
                               "lib/assemble.py", "lib/hostid.py", "bin/codex-auto"),
}
_SKIP = re.compile(r"(@pytest\.mark\.(skip|xfail)|@unittest\.skip|\bskipTest\(|\bmarkTestSkipped\(|"
                   r"\b(it|test|describe)\.(skip|todo)\(|\bxit\(|\bxdescribe\()")
_ASERCION = re.compile(r"^\s*(assert\b|self\.assert|expect\(|\$this->assert|assert[A-Z]\w*\()")
_GANCHOS = re.compile(r"(--no-verify\b|core\.hooksPath|HUSKY=0|SKIP_HOOKS)")


@dataclass(frozen=True)
class Violacion:
    regla: str
    ruta: str
    detalle: str


def _es_prueba(ruta: str | None) -> bool:
    if not ruta:
        return False
    p = PurePosixPath(ruta)
    return ("tests" in p.parts or "test" in p.parts or p.name.startswith("test_")
            or re.search(r"\.(test|spec)\.[jt]sx?$", p.name) is not None or p.name.endswith("Test.php"))


def _es_env_con_valores(c: Cambio) -> bool:
    nombre = PurePosixPath(c.ruta).name
    if not nombre.startswith(".env") or nombre == ".env.example":
        return False
    return any("=" in l and l.split("=", 1)[1].strip() for l in c.agregadas)


def revisar(cambios: tuple[Cambio, ...], repo: str) -> tuple[Violacion, ...]:
    v: list[Violacion] = []
    for c in cambios:
        if c.ruta.startswith(".github/workflows/") or (c.ruta_anterior or "").startswith(".github/workflows/"):
            v.append(Violacion("flujos_ci", c.ruta, "cambio bajo .github/workflows/"))
        for prot in ARCHIVOS_DE_FERNANDO.get(repo, ()):
            if c.ruta == prot or (prot.endswith("/") and c.ruta.startswith(prot)):
                v.append(Violacion("archivo_de_fernando", c.ruta, prot))
        if _es_prueba(c.ruta_anterior or c.ruta) and (c.estado == "D" or (c.estado == "R" and not _es_prueba(c.ruta))):
            v.append(Violacion("prueba_debilitada", c.ruta, "prueba borrada o sacada de las pruebas"))
        for linea in c.agregadas:
            if _SKIP.search(linea):
                v.append(Violacion("prueba_debilitada", c.ruta, linea.strip()[:200]))
            if _GANCHOS.search(linea):
                v.append(Violacion("ganchos", c.ruta, linea.strip()[:200]))
        if _es_prueba(c.ruta):
            netas = sum(bool(_ASERCION.match(l)) for l in c.quitadas) - sum(bool(_ASERCION.match(l)) for l in c.agregadas)
            if netas > 0:
                v.append(Violacion("prueba_debilitada", c.ruta, f"{netas} aserciones netas menos"))
        if _es_env_con_valores(c):
            v.append(Violacion("secretos", c.ruta, ".env con valores"))
    return tuple(v)
