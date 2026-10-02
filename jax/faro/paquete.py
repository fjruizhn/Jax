"""El paquete fijado del ecosistema: constitucion, skills, agentes y (opcional) plugins,
armados desde UN SHA de `origin/main` de `claude-skills` y verificables byte a byte.

Spec 2026-10-02-faro-pasarela-ecosistema-design.md §3 («Paquete fijado») y §5 («En cada
ejecucion se comprueba la integridad contra el manifiesto. La frescura se comprueba
aparte, sin bloquear»). Plan: docs/superpowers/plans/2026-10-02-faro-fase-0.md, paso 0.1.

DE DONDE SALE. Los bytes salen de los OBJETOS git del SHA pedido (`git ls-tree` /
`git cat-file`), nunca del arbol de trabajo de `~/claude-skills`: ese checkout lo mueve
`claude-skills-sync pull` y cualquier sesion puede dejarle cambios sin commitear. El SHA
tiene que ser ancestro de `origin/main` (de otro modo un commit de una rama cualquiera
podria fijarse como «el ecosistema»).

LA CONSTITUCION es el NUCLEO COMUN (`common/CLAUDE.md.core`) con el sello
`<!-- claude-skills: SHA ... -->` que pone ESTE constructor (hallazgo F-4 del spec: no se
deduce la version de un archivo ensamblado). No lleva bloque de host: el bloque de host es
propio de una maquina y el paquete sirve a facetas de varios motores; ver la seccion
«Desviaciones» del plan.

INTEGRIDAD Y FRESCURA SON PREGUNTAS DISTINTAS (patron de
`jax/ejecutor/contratos/arranque.py::verificar_contexto`, lineas 154-200):
- `verificar_integridad` / `exigir_integridad`: ¿lo que hay en disco es lo que se
  construyo? Compara el CONJUNTO completo (un archivo de mas, uno de menos, un byte
  cambiado, un modo cambiado, un symlink) contra `MANIFIESTO.json`, y el SHA del
  manifiesto contra el pedido. BLOQUEA.
- `frescura`: ¿hay un SHA mas nuevo en `origin/main`? SOLO AVISA. Un cambio posterior en
  la constitucion no tiene por que frenar nada en curso.

EL MANIFIESTO se firma con su propio hash (`sha256_manifiesto`, sobre su cuerpo canonico):
detecta un manifiesto editado a medias, NO es una firma de autoria. Quien puede reescribir
el paquete entero y recalcular el hash no se detecta con esto; por eso en produccion el
paquete es de root y de solo lectura para quien lo carga (spec §5), igual que
`contexto/MANIFIESTO.sha256.json` del Ejecutor.

Este modulo es SINCRONO a proposito (herramienta de construccion y de arranque, con
subprocesos `git` y E/S de disco). Quien lo llame desde `async def` lo envuelve en
`asyncio.to_thread`.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import secrets
import shutil
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType

from .config import ConfigFaro, PluginFuente, sha_valido
from .git_objetos import FuenteInvalida, exigir_sha_ancestro_de, leer_blobs, listar, resolver_ref

logger = logging.getLogger(__name__)

MANIFIESTO = "MANIFIESTO.json"
ESQUEMA = 1

_FUENTE_CONSTITUCION = "common/CLAUDE.md.core"
_FUENTE_SKILLS = "common/skills"
_FUENTE_AGENTES = "common/agents"

_MODOS = {"0644": 0o644, "0755": 0o755}
_RE_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class PaqueteNoVerifica(RuntimeError):
    """El paquete en disco no coincide con su manifiesto o con el SHA pedido: no se carga."""

    def __init__(self, fallos: tuple["Fallo", ...]):
        self.fallos = fallos
        super().__init__("el paquete del ecosistema no verifica: " + "; ".join(
            f"{f.codigo}" + (f" ({f.ruta})" if f.ruta else "") for f in fallos[:10]))


@dataclass(frozen=True)
class Fallo:
    codigo: str
    ruta: str = ""


@dataclass(frozen=True)
class Frescura:
    estado: str          # "al_dia" | "atrasado" | "desconocida"
    sha_paquete: str
    sha_actual: str | None = None


# --------------------------------------------------------------------------- #
# armado                                                                      #
# --------------------------------------------------------------------------- #

def _archivos_desde_git(cfg: ConfigFaro) -> dict[str, tuple[int, bytes]]:
    """{ruta en el paquete: (modo, bytes)} de lo que el spec pide, leido del SHA."""
    entradas = listar(cfg.repo, cfg.sha, _FUENTE_CONSTITUCION, _FUENTE_SKILLS, _FUENTE_AGENTES)
    for e in entradas:
        if e.modo == "120000":
            raise FuenteInvalida(f"{e.ruta} es un symlink en {cfg.sha[:12]}: no se sigue ni se copia")
        if e.modo not in ("100644", "100755"):
            raise FuenteInvalida(f"{e.ruta} tiene un modo git no soportado ({e.modo})")
    elegidas: dict[str, _EntradaGit] = {}
    for e in entradas:
        if e.ruta == _FUENTE_CONSTITUCION:
            elegidas["constitucion/CLAUDE.md"] = e
        elif e.ruta.startswith(_FUENTE_SKILLS + "/"):
            elegidas["skills/" + e.ruta[len(_FUENTE_SKILLS) + 1:]] = e
        elif e.ruta.startswith(_FUENTE_AGENTES + "/") and e.ruta.count("/") == 2 and e.ruta.endswith(".md"):
            elegidas["agentes/" + e.ruta[len(_FUENTE_AGENTES) + 1:]] = e
    if "constitucion/CLAUDE.md" not in elegidas:
        raise FuenteInvalida(f"{_FUENTE_CONSTITUCION} no existe en {cfg.sha[:12]}")
    if not any(k.startswith("skills/") for k in elegidas):
        raise FuenteInvalida(f"{_FUENTE_SKILLS} esta vacio en {cfg.sha[:12]}")
    blobs = leer_blobs(cfg.repo, [e.oid for e in elegidas.values()])
    archivos = {}
    for destino, e in sorted(elegidas.items()):
        datos = blobs[e.oid]
        if destino == "constitucion/CLAUDE.md":
            datos = f"<!-- claude-skills: SHA {cfg.sha} -->\n".encode() + datos
        archivos[destino] = (0o755 if e.modo == "100755" else 0o644, datos)
    return archivos


def _archivos_de_plugin(p: PluginFuente) -> dict[str, tuple[int, bytes]]:
    base = f"plugins/{p.nombre}"
    if not p.ruta.is_dir() or p.ruta.is_symlink():
        raise FuenteInvalida(f"el plugin {p.nombre!r} no esta en {p.ruta} (o es un symlink)")
    archivos: dict[str, tuple[int, bytes]] = {}

    def recorrer(dir_fuente: Path, prefijo: str, solo_md: bool) -> None:
        for entrada in sorted(os.scandir(dir_fuente), key=lambda e: e.name):
            st = entrada.stat(follow_symlinks=False)
            rel = f"{prefijo}/{entrada.name}"
            if stat.S_ISLNK(st.st_mode):
                raise FuenteInvalida(f"{entrada.path} es un symlink: no se sigue ni se copia")
            if stat.S_ISDIR(st.st_mode):
                if not solo_md:
                    recorrer(Path(entrada.path), rel, solo_md)
            elif stat.S_ISREG(st.st_mode):
                if solo_md and not entrada.name.endswith(".md"):
                    continue
                modo = 0o755 if st.st_mode & 0o111 else 0o644
                archivos[rel] = (modo, Path(entrada.path).read_bytes())
            else:
                raise FuenteInvalida(f"{entrada.path} no es un archivo regular")

    for sub, solo_md in (("skills", False), ("agents", True)):
        d = p.ruta / sub
        if d.is_symlink():
            raise FuenteInvalida(f"{d} es un symlink: no se sigue ni se copia")
        if d.is_dir():
            recorrer(d, f"{base}/{sub}", solo_md)
    return archivos


def _sha256(datos: bytes) -> str:
    return hashlib.sha256(datos).hexdigest()


def hash_del_manifiesto(manifiesto: dict) -> str:
    """sha256 del cuerpo canonico del manifiesto, SIN su propio campo `sha256_manifiesto`."""
    cuerpo = {k: v for k, v in manifiesto.items() if k != "sha256_manifiesto"}
    return _sha256(json.dumps(cuerpo, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode())


def _escribir_paquete(raiz: Path, cfg: ConfigFaro, archivos: dict[str, tuple[int, bytes]]) -> None:
    raiz.mkdir(mode=0o755)
    os.chmod(raiz, 0o755)
    for rel, (modo, datos) in archivos.items():
        ruta = raiz / rel
        for padre in reversed([p for p in ruta.parents if p != raiz and raiz in p.parents]):
            if not padre.exists():
                padre.mkdir(mode=0o755)
                os.chmod(padre, 0o755)
        fd = os.open(ruta, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(datos)
        os.chmod(ruta, modo)
    manifiesto = {
        "esquema": ESQUEMA,
        "sha_origen": cfg.sha,
        "plugins": [{"nombre": p.nombre, "sha_declarado": p.sha_declarado} for p in cfg.plugins],
        "archivos": {rel: {"modo": f"{modo:04o}", "sha256": _sha256(datos)} for rel, (modo, datos) in archivos.items()},
    }
    manifiesto["sha256_manifiesto"] = hash_del_manifiesto(manifiesto)
    fd = os.open(raiz / MANIFIESTO, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(manifiesto, f, sort_keys=True, indent=1)
        f.write("\n")
    os.chmod(raiz / MANIFIESTO, 0o644)


def construir_paquete(cfg: ConfigFaro) -> Path:
    """Arma `<destino>/<SHA>/` y devuelve su ruta. Se construye en un directorio temporal
    hermano, se VERIFICA y recien entonces se publica con `rename` (atomico en el mismo
    sistema de archivos). Si ya existe un paquete de ese SHA: integro -> se devuelve tal
    cual; alterado -> `PaqueteNoVerifica` (no se pisa ni se repara en silencio)."""
    final = cfg.raiz_paquete
    if final.exists() or final.is_symlink():
        fallos = _verificar_raiz(final, cfg.sha)
        if fallos:
            raise PaqueteNoVerifica(fallos)
        return final
    exigir_sha_ancestro_de(cfg.repo, cfg.sha, cfg.ref_frescura)
    archivos = _archivos_desde_git(cfg)
    for p in cfg.plugins:
        archivos.update(_archivos_de_plugin(p))
    cfg.destino.mkdir(parents=True, exist_ok=True)
    tmp = cfg.destino / f".construyendo-{cfg.sha[:12]}-{os.getpid()}-{secrets.token_hex(4)}"
    try:
        _escribir_paquete(tmp, cfg, archivos)
        fallos = _verificar_raiz(tmp, cfg.sha)
        if fallos:
            raise PaqueteNoVerifica(fallos)
        try:
            os.rename(tmp, final)
        except OSError:
            # Otro constructor publico el mismo SHA entre medias: vale si esta integro.
            fallos = _verificar_raiz(final, cfg.sha)
            if fallos:
                raise PaqueteNoVerifica(fallos) from None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return final


# --------------------------------------------------------------------------- #
# verificacion                                                                #
# --------------------------------------------------------------------------- #

def _ruta_limpia(rel: object) -> bool:
    if not isinstance(rel, str) or not rel or "\0" in rel or "\\" in rel or rel.startswith("/"):
        return False
    partes = PurePosixPath(rel).parts
    return rel == "/".join(partes) and all(p not in ("", ".", "..") for p in partes)


def _leer_manifiesto(raiz: Path) -> dict | None:
    try:
        m = json.loads((raiz / MANIFIESTO).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(m, dict) or m.get("esquema") != ESQUEMA or not sha_valido(m.get("sha_origen")):
        return None
    if not isinstance(m.get("sha256_manifiesto"), str) or not isinstance(m.get("plugins"), list):
        return None
    archivos = m.get("archivos")
    if not isinstance(archivos, dict) or not archivos:
        return None
    for rel, meta in archivos.items():
        if (not _ruta_limpia(rel) or rel == MANIFIESTO or not isinstance(meta, dict)
                or meta.get("modo") not in _MODOS or not _RE_SHA256.fullmatch(str(meta.get("sha256", "")))):
            return None
    return m


def _recorrer_disco(raiz: Path) -> tuple[dict[str, Path], set[str], set[str]]:
    """(archivos regulares, directorios, especiales: symlinks y demas) bajo `raiz`, sin
    seguir ningun enlace."""
    regulares: dict[str, Path] = {}
    dirs: set[str] = set()
    especiales: set[str] = set()
    pila = [raiz]
    while pila:
        actual = pila.pop()
        with os.scandir(actual) as it:
            for e in it:
                rel = Path(e.path).relative_to(raiz).as_posix()
                st = e.stat(follow_symlinks=False)
                if stat.S_ISDIR(st.st_mode):
                    dirs.add(rel)
                    pila.append(Path(e.path))
                elif stat.S_ISREG(st.st_mode):
                    regulares[rel] = Path(e.path)
                else:
                    especiales.add(rel)
    return regulares, dirs, especiales


def _verificar_raiz(raiz: Path, sha_pedido: str) -> tuple[Fallo, ...]:
    if raiz.is_symlink() or not raiz.is_dir():
        return (Fallo("paquete_ausente"),)
    manifiesto = _leer_manifiesto(raiz)
    if manifiesto is None:
        return (Fallo("manifiesto_ilegible"),)
    if manifiesto["sha256_manifiesto"] != hash_del_manifiesto(manifiesto):
        return (Fallo("manifiesto_hash"),)
    if manifiesto["sha_origen"] != sha_pedido:
        return (Fallo("sha_distinto_del_pedido", manifiesto["sha_origen"]),)
    try:
        regulares, dirs, especiales = _recorrer_disco(raiz)
    except OSError:
        return (Fallo("paquete_ilegible"),)
    esperados = manifiesto["archivos"]
    fallos: list[Fallo] = []
    for rel in sorted(esperados):
        if rel in especiales:
            fallos.append(Fallo("archivo_no_regular", rel))
        elif rel not in regulares:
            fallos.append(Fallo("archivo_faltante", rel))
        else:
            ruta = regulares[rel]
            try:
                datos = ruta.read_bytes()
                modo = stat.S_IMODE(ruta.lstat().st_mode)
            except OSError:
                fallos.append(Fallo("archivo_faltante", rel))
                continue
            if _sha256(datos) != esperados[rel]["sha256"]:
                fallos.append(Fallo("archivo_alterado", rel))
            if modo != _MODOS[esperados[rel]["modo"]]:
                fallos.append(Fallo("modo_distinto", rel))
    extras = (set(regulares) | especiales) - set(esperados) - {MANIFIESTO}
    fallos.extend(Fallo("archivo_extra", rel) for rel in sorted(extras))
    dirs_esperados = {p.as_posix() for rel in esperados for p in PurePosixPath(rel).parents if p.as_posix() != "."}
    fallos.extend(Fallo("directorio_extra", rel) for rel in sorted(dirs - dirs_esperados))
    return tuple(fallos)


def verificar_integridad(cfg: ConfigFaro) -> tuple[Fallo, ...]:
    """Tupla vacia = integro. Nunca lanza: devuelve los fallos (como `arranque.py`)."""
    return _verificar_raiz(cfg.raiz_paquete, cfg.sha)


def exigir_integridad(cfg: ConfigFaro) -> Path:
    """Lo que llama el arranque: devuelve la raiz del paquete o `PaqueteNoVerifica`."""
    fallos = verificar_integridad(cfg)
    if fallos:
        raise PaqueteNoVerifica(fallos)
    return cfg.raiz_paquete


def frescura(cfg: ConfigFaro) -> Frescura:
    """¿Hay un SHA mas nuevo en `origin/main`? SOLO AVISA (log WARNING); jamas bloquea ni
    lanza. Lee la ref local: quien mantiene el checkout (`claude-skills-sync pull`, cron) es
    quien la mueve; esta funcion no hace red."""
    actual = resolver_ref(cfg.repo, cfg.ref_frescura)
    if not sha_valido(actual):
        logger.warning("frescura del paquete %s: desconocida (no se pudo leer %s)", cfg.sha[:12], cfg.ref_frescura)
        return Frescura("desconocida", cfg.sha)
    if actual != cfg.sha:
        logger.warning("el paquete del ecosistema %s esta atrasado: %s ya esta en %s (no se bloquea nada)",
                       cfg.sha, actual, cfg.ref_frescura)
        return Frescura("atrasado", cfg.sha, actual)
    return Frescura("al_dia", cfg.sha, actual)


# --------------------------------------------------------------------------- #
# carga en memoria: lo que el Puerto sirve                                    #
# --------------------------------------------------------------------------- #

class NoExiste(LookupError):
    """Lo pedido no esta en el paquete. Nunca se distingue «no existe» de «no se puede»:
    para el que pide, lo que no esta en el paquete no existe."""


def _frontmatter(datos: bytes) -> dict[str, str]:
    """Los pares `clave: valor` de una sola linea del frontmatter YAML de un `.md`. Es lo
    minimo que el catalogo necesita (nombre, descripcion, modelo); no interpreta YAML (las
    skills y agentes del ecosistema escriben escalares de una linea)."""
    texto = datos.decode("utf-8", errors="replace")
    if not texto.startswith("---\n"):
        return {}
    fin = texto.find("\n---", 4)
    if fin == -1:
        return {}
    pares = {}
    for linea in texto[4:fin].splitlines():
        clave, sep, valor = linea.partition(":")
        if sep and clave and not clave[0].isspace() and clave not in pares:
            pares[clave.strip()] = valor.strip()
    return pares


@dataclass(frozen=True)
class PaqueteCargado:
    """El paquete VERIFICADO, en memoria. El Puerto sirve estos bytes y no vuelve a leer el
    disco: un symlink o un archivo cambiado despues de la carga no se sirve, y la busqueda
    de una skill es un `dict`, de modo que `../`, rutas absolutas o enlaces no tienen donde
    engancharse (lo que no es una clave exacta no existe).

    Cache e invalidacion (LAS CUATRO DEL RENDIMIENTO, 2): el paquete es INMUTABLE por SHA. No hay
    TTL ni invalidacion parcial; cambiar de version es cargar otro `PaqueteCargado` (y otro
    Puerto) con otro SHA, y el cambio atomico lo hace quien orquesta (spec §5)."""
    sha: str
    constitucion: str
    skills: Mapping[str, Mapping[str, bytes]]     # nombre -> {ruta relativa dentro de la skill: bytes}
    agentes: Mapping[str, bytes]                  # nombre -> .md completo (el cuerpo NO sale en el catalogo)

    def descripcion_de(self, nombre: str) -> str:
        return _frontmatter(self.skills[nombre]["SKILL.md"]).get("description", "")

    def buscar(self, consulta: str, limite: int) -> list[dict]:
        terminos = consulta.casefold().split()
        encontradas = []
        for nombre in sorted(self.skills):
            descripcion = self.descripcion_de(nombre)
            cuerpo = self.skills[nombre]["SKILL.md"].decode("utf-8", errors="replace")
            pajar = f"{nombre}\n{descripcion}\n{cuerpo}".casefold()
            if all(t in pajar for t in terminos):
                en_nombre = all(t in nombre.casefold() for t in terminos) if terminos else False
                encontradas.append((0 if en_nombre else 1, nombre, descripcion))
        encontradas.sort()
        return [{"nombre": n, "descripcion": d, "uri": f"skill://{n}"} for _, n, d in encontradas[:limite]]

    def leer(self, nombre: str, archivo: str = "SKILL.md") -> str:
        try:
            datos = self.skills[nombre][archivo]
        except KeyError:
            raise NoExiste(f"la skill {nombre!r} (archivo {archivo!r}) no esta en el paquete") from None
        try:
            return datos.decode("utf-8")
        except UnicodeDecodeError:
            raise NoExiste(f"{nombre}/{archivo} no es texto UTF-8") from None

    def catalogo_agentes(self) -> list[dict]:
        salida = []
        for nombre in sorted(self.agentes):
            fm = _frontmatter(self.agentes[nombre])
            salida.append({"nombre": nombre, "descripcion": fm.get("description", ""),
                           "modelo": fm.get("model", ""), "herramientas": fm.get("tools", "")})
        return salida


def _leer_sin_seguir_enlaces(ruta: Path) -> bytes:
    fd = os.open(ruta, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as f:
        return f.read()


def cargar_paquete(cfg: ConfigFaro) -> PaqueteCargado:
    """Verifica la integridad (`exigir_integridad`: lanza `PaqueteNoVerifica`) y carga en
    memoria lo que el Puerto sirve. Cada archivo se lee SIN seguir enlaces y se vuelve a
    comparar con el manifiesto sobre los bytes que quedaron en memoria: lo que se sirve es
    exactamente lo que se verifico, sin ventana entre la comprobacion y el uso."""
    raiz = exigir_integridad(cfg)
    esperados = _leer_manifiesto(raiz)["archivos"]
    memoria: dict[str, bytes] = {}
    for rel in sorted(esperados):
        try:
            datos = _leer_sin_seguir_enlaces(raiz / rel)
        except OSError:
            raise PaqueteNoVerifica((Fallo("archivo_faltante", rel),)) from None
        if _sha256(datos) != esperados[rel]["sha256"]:
            raise PaqueteNoVerifica((Fallo("archivo_alterado", rel),))
        memoria[rel] = datos
    skills: dict[str, dict[str, bytes]] = {}
    agentes: dict[str, bytes] = {}
    for rel, datos in memoria.items():
        partes = rel.split("/")
        prefijo = ""
        if partes[0] == "plugins" and len(partes) > 3:
            prefijo, partes = f"{partes[1]}:", partes[2:]
        if partes[0] == "skills" and len(partes) >= 3:
            skills.setdefault(prefijo + partes[1], {})["/".join(partes[2:])] = datos
        elif partes[0] == "agentes" or (prefijo and partes[0] == "agents"):
            if len(partes) == 2 and partes[1].endswith(".md"):
                agentes[prefijo + partes[1][:-3]] = datos
    skills = {n: archivos for n, archivos in skills.items() if "SKILL.md" in archivos}
    return PaqueteCargado(
        sha=cfg.sha,
        constitucion=memoria["constitucion/CLAUDE.md"].decode("utf-8"),
        skills=MappingProxyType({n: MappingProxyType(a) for n, a in skills.items()}),
        agentes=MappingProxyType(agentes),
    )
