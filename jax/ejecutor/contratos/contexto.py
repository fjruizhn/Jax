# jax/ejecutor/contratos/contexto.py
"""El CLAUDE.md y las skills de axioma: SIEMPRE generados, nunca escritos a mano
(spec 2026-09-15-ejecutor-design.md §6.1). Punto único de verdad para que el
instalador (`ops/ejecutor/instalar_contexto.sh`) y el arranque de cada misión
(`arranque.py`, dentro de `verificar_instalacion`) comparen EXACTAMENTE lo mismo:
lo que el generador produce HOY contra lo instalado en `JAX_EJECUTOR_LIB`.

El generador real vive en `scripts/ejecutor_fase0/generar_claude_md.py`; `scripts/`
no es un paquete Python, así que se carga por ruta (mismo patrón que
`tests/test_ejecutor_fase0.py`). Ese script (y `cerebros.toml`) son host-bound: leen
`/home/fruiz/claude-skills/common/...` de esta máquina, no una ruta relativa al repo
-- es la misma constitución que usa Mr. Hyde, y vive en OTRO repo (Principio IV: la
fuente de la constitución no se copia a mano en `jax`, se lee de un único lugar)."""
from __future__ import annotations

import hashlib
import importlib.util
import sys
import tomllib
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[3]
_GENERADOR = RAIZ / "scripts" / "ejecutor_fase0" / "generar_claude_md.py"
_CEREBROS = RAIZ / "scripts" / "ejecutor_fase0" / "cerebros.toml"

#: Rutas, DENTRO de JAX_EJECUTOR_LIB, del CLAUDE.md instalado, su sha256 de
#: acompañamiento (sólo informativo/de auditoría -- la comparación real siempre
#: rehace el hash de los bytes instalados) y las skills.
CLAUDE_MD_REL = "contexto/CLAUDE.md"
CLAUDE_MD_SHA256_REL = "contexto/CLAUDE.md.sha256"
SKILLS_REL = "contexto/skills"


class SkillFaltante(RuntimeError):
    """`args[0]` es el nombre de la skill declarada que no está (o está vacía) en
    `skills_fuente()`. No se instala una selección parcial de skills."""


def _cargar_generador():
    spec = importlib.util.spec_from_file_location("ejecutor_generar_claude_md", _GENERADOR)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _constitucion() -> dict:
    return tomllib.loads(_CEREBROS.read_text(encoding="utf-8"))["constitucion"]


def constitucion_disponible() -> bool:
    """¿Existe, en ESTA máquina, el archivo del que `claude_md()` lee la
    constitución? Host-bound (Fase 0): los tests que llaman a `claude_md()` se
    saltan con esto donde no exista, en vez de fallar contra un runner que nunca
    la va a tener."""
    return Path(_constitucion()["fuente"]).is_file()


def claude_md() -> bytes:
    """El CLAUDE.md que el generador produce AHORA MISMO, desde CLAUDE.md.core."""
    return _cargar_generador().generar().encode("utf-8")


def sha256_claude_md() -> str:
    return hashlib.sha256(claude_md()).hexdigest()


def skills_declaradas() -> tuple[str, ...]:
    """Los nombres de skills de `cerebros.toml`, en el orden declarado ahí."""
    return tuple(_constitucion()["skills"])


def skills_fuente() -> Path:
    """De dónde se copian las skills (carpeta completa, con subcarpetas). Función,
    no constante: así un test la monkeypatchea sin escribir en la ruta real."""
    return Path(_constitucion()["skills_fuente"])


def archivos_de_skills() -> dict[str, bytes]:
    """`{"<skill>/<ruta relativa>": contenido}` de TODAS las skills declaradas,
    carpeta completa. Una carpeta ausente o vacía es `SkillFaltante(nombre)`: no se
    instala una selección parcial ni una skill a medias."""
    fuente = skills_fuente()
    archivos: dict[str, bytes] = {}
    for nombre in skills_declaradas():
        base = fuente / nombre
        if not base.is_dir():
            raise SkillFaltante(nombre)
        vistos = sorted(p for p in base.rglob("*") if p.is_file())
        if not vistos:
            raise SkillFaltante(nombre)
        for p in vistos:
            archivos[f"{nombre}/{p.relative_to(base).as_posix()}"] = p.read_bytes()
    return archivos


def renderizar_etapa(etapa: Path) -> None:
    """Escribe en `etapa` (directorio de staging, aún no instalado) exactamente lo
    que `ops/ejecutor/instalar_contexto.sh` va a copiar con `sudo install`:
    `CLAUDE.md`, `CLAUDE.md.sha256` (sólo de auditoría -- ver el docstring del
    módulo) y `skills/<nombre>/...` con la carpeta COMPLETA de cada skill
    declarada. Una skill faltante en la fuente hace que esto reviente con
    `SkillFaltante`: no se instala una etapa a medias."""
    etapa.mkdir(parents=True, exist_ok=True)
    doc = claude_md()
    (etapa / "CLAUDE.md").write_bytes(doc)
    (etapa / "CLAUDE.md.sha256").write_text(hashlib.sha256(doc).hexdigest())
    for rel, datos in archivos_de_skills().items():
        destino = etapa / "skills" / rel
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_bytes(datos)


def principal(argv: list[str]) -> int:
    """`python -m jax.ejecutor.contratos.contexto <etapa>` -- lo que
    `ops/ejecutor/instalar_contexto.sh` invoca para rellenar el staging antes del
    `sudo install`, mismo patrón que `jax.ejecutor.contratos.instalacion`."""
    renderizar_etapa(Path(argv[0]))
    return 0


if __name__ == "__main__":
    sys.exit(principal(sys.argv[1:]))
