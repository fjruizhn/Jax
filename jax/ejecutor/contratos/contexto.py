# jax/ejecutor/contratos/contexto.py
"""El CLAUDE.md y las skills de axioma: SIEMPRE generados, nunca escritos a mano
(spec 2026-09-15-ejecutor-design.md §6.1). Punto único de verdad para que el
instalador (`ops/ejecutor/instalar_contexto.sh`) escriba lo generado Y un MANIFIESTO
firmado por esa instalación (root, sólo lectura para axioma).

M-2 (ronda 6, auditoría adversarial 2026-09-22): «el contexto instalado es la
autoridad». El arranque de cada misión (`arranque.py::verificar_contexto`) YA NO
regenera desde `/home/fruiz/claude-skills` en cada misión -- compara los bytes
instalados contra el MANIFIESTO (`MANIFIESTO_REL`) que se escribió AL INSTALAR, nunca
contra lo que el generador produciría si corriera ahora mismo. Integridad ("¿lo
instalado es lo que se instaló?") y frescura ("¿lo instalado sigue siendo lo que la
constitución de HOY generaría?") son preguntas DISTINTAS: la primera la contesta el
arranque, siempre, bloqueando la misión si falla; la segunda la contesta
`--comprobar-frescura` (más abajo, y en `instalar_contexto.sh`), aparte, informando sin
bloquear nada -- un cambio en la constitución de Fernando DESPUÉS de instalar no tiene
por qué frenar una misión en curso.

El generador real vive en `scripts/ejecutor_fase0/generar_claude_md.py`; `scripts/`
no es un paquete Python, así que se carga por ruta (mismo patrón que
`tests/test_ejecutor_fase0.py`). Ese script (y `cerebros.toml`) son host-bound: leen
`/home/fruiz/claude-skills/common/...` de esta máquina, no una ruta relativa al repo
-- es la misma constitución que usa Mr. Hyde, y vive en OTRO repo (Principio IV: la
fuente de la constitución no se copia a mano en `jax`, se lee de un único lugar)."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tomllib
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[3]
_GENERADOR = RAIZ / "scripts" / "ejecutor_fase0" / "generar_claude_md.py"
_CEREBROS = RAIZ / "scripts" / "ejecutor_fase0" / "cerebros.toml"

#: Rutas, DENTRO de JAX_EJECUTOR_LIB, del CLAUDE.md instalado, su sha256 de
#: acompañamiento (sólo informativo/de auditoría humana) y las skills.
CLAUDE_MD_REL = "contexto/CLAUDE.md"
CLAUDE_MD_SHA256_REL = "contexto/CLAUDE.md.sha256"
SKILLS_REL = "contexto/skills"
#: El REGISTRO que el arranque compara (M-2, ronda 6): {"CLAUDE.md": sha256,
#: "skills/<...>": sha256, ...}, escrito por `instalar_contexto.sh` (root, 0644 --
#: axioma lo lee, no lo escribe). Es la ÚNICA fuente de "qué se instaló"; el arranque
#: NO vuelve a leer `/home/fruiz/claude-skills`.
MANIFIESTO_REL = "contexto/MANIFIESTO.sha256.json"
# NOTA (B-1/M-4, ronda 3, 2026-09-22): el archivo `contexto/CLAUDE.md.home.vacio` que la
# ronda 2 (M3) instalaba para tapar "$HOME/CLAUDE.md" a mano queda RETIRADO -- ver
# cuenta_axioma.py. Con "$HOME" como `--tmpfs` propio de cada invocación, ese archivo (y
# el directorio de memoria) ya no existen ahí salvo que la jaula los monte, así que un
# bind puntual sobre uno solo era un parche sobre el síntoma, no la causa.


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


def constitucion_fuente() -> Path:
    """De dónde sale la constitución real (cerebros.toml `constitucion.fuente`).
    Función, no constante: igual que `skills_fuente()`, así un test la
    monkeypatchea con un doble hermético (B3/M1, auditoría adversarial 2026-09-22)
    sin depender de que ESTA máquina tenga /home/fruiz/claude-skills."""
    return Path(_constitucion()["fuente"])


def constitucion_disponible() -> bool:
    """¿Existe, en ESTA máquina, el archivo real del que `claude_md()` leería la
    constitución si no se le da un doble? Host-bound (Fase 0): sigue existiendo
    para las pruebas que quieren la constitución REAL (no un doble) y se saltan
    donde no exista -- la mayoría de las afirmaciones, sin embargo, ya no lo
    necesitan: corren igual con `monkeypatch.setattr(contexto, "constitucion_fuente", ...)`."""
    return constitucion_fuente().is_file()


def claude_md() -> bytes:
    """El CLAUDE.md que el generador produce AHORA MISMO, desde `constitucion_fuente()`."""
    return _cargar_generador().generar(fuente_constitucion=constitucion_fuente()).encode("utf-8")


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


def manifiesto() -> dict:
    """{"CLAUDE.md": sha256, "skills/<skill>/<ruta>": sha256, ...} de lo que el
    generador produciría AHORA MISMO -- se usa para escribir la etapa
    (`renderizar_etapa`) y para el chequeo de frescura (`principal`,
    `--comprobar-frescura`). El arranque de una misión NUNCA llama a esto: lee el
    manifiesto YA INSTALADO (`MANIFIESTO_REL`), no lo recalcula (M-2, ronda 6)."""
    m = {"CLAUDE.md": hashlib.sha256(claude_md()).hexdigest()}
    for rel, datos in archivos_de_skills().items():
        m[f"skills/{rel}"] = hashlib.sha256(datos).hexdigest()
    return m


def renderizar_etapa(etapa: Path) -> None:
    """Escribe en `etapa` (directorio de staging, aún no instalado) exactamente lo
    que `ops/ejecutor/instalar_contexto.sh` va a copiar con `sudo install`:
    `CLAUDE.md`, `CLAUDE.md.sha256` (sólo de auditoría humana),
    `skills/<nombre>/...` con la carpeta COMPLETA de cada skill declarada, y
    `MANIFIESTO.sha256.json` (M-2, ronda 6 -- lo que el arranque de cada misión va a
    comparar, root, de sólo lectura para axioma una vez instalado). Una skill
    faltante en la fuente hace que esto reviente con `SkillFaltante`: no se instala
    una etapa a medias."""
    etapa.mkdir(parents=True, exist_ok=True)
    doc = claude_md()
    archivos_skills = archivos_de_skills()
    (etapa / "CLAUDE.md").write_bytes(doc)
    (etapa / "CLAUDE.md.sha256").write_text(hashlib.sha256(doc).hexdigest())
    m = {"CLAUDE.md": hashlib.sha256(doc).hexdigest()}
    for rel, datos in archivos_skills.items():
        destino = etapa / "skills" / rel
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_bytes(datos)
        m[f"skills/{rel}"] = hashlib.sha256(datos).hexdigest()
    (etapa / "MANIFIESTO.sha256.json").write_text(json.dumps(m, sort_keys=True), encoding="utf-8")


def principal(argv: list[str]) -> int:
    """`python -m jax.ejecutor.contratos.contexto <etapa>` -- lo que
    `ops/ejecutor/instalar_contexto.sh` invoca para rellenar el staging antes del
    `sudo install`, mismo patrón que `jax.ejecutor.contratos.instalacion`.

    `--manifiesto` (M-2, ronda 6): imprime el manifiesto que el generador produciría
    AHORA MISMO, en JSON, a stdout -- lo usa `instalar_contexto.sh --comprobar-frescura`
    para comparar contra el manifiesto YA INSTALADO sin bloquear nada (ver el docstring
    del módulo)."""
    if argv and argv[0] == "--manifiesto":
        print(json.dumps(manifiesto(), sort_keys=True))
        return 0
    renderizar_etapa(Path(argv[0]))
    return 0


if __name__ == "__main__":
    sys.exit(principal(sys.argv[1:]))
