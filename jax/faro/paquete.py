"""El paquete fijado del ecosistema: constitucion, skills y agentes,
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
el paquete entero y recalcular el hash no se detecta con eso. Dos defensas, que NO dependen
del manifiesto:
1. el manifiesto guarda el oid git de cada blob, y `cargar_paquete` lo compara contra
   `git ls-tree <SHA>` (sin replace refs) y re-calcula el oid de los bytes leidos: el contenido
   queda atado al SHA, no a si mismo;
2. el dueño: `verificar_integridad` exige que la raiz, cada entrada y sus ancestros sean del
   dueño esperado (`ConfigFaro.uid_duenio`: root en produccion, inyectable en pruebas) y que
   nadie ajeno pueda escribirlos (modo sin g+w/o+w; un ancestro con escritura ajena solo
   vale si tiene sticky, como /tmp).

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

from .config import ConfigFaro, sha_valido
from .git_objetos import EntradaGit, FuenteInvalida, es_ancestro, exigir_sha_ancestro_de, leer_blobs, listar, resolver_ref

logger = logging.getLogger(__name__)

MANIFIESTO = "MANIFIESTO.json"
ESQUEMA = 2

_FUENTE_CONSTITUCION = "common/CLAUDE.md.core"
_FUENTE_SKILLS = "common/skills"
_FUENTE_AGENTES = "common/agents"

_MODOS = {"0644": 0o644, "0755": 0o755}
_RE_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RE_OID = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")


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
    estado: str          # "al_dia" | "atrasado" | "retirado" (ya no es ancestro de main) | "desconocida"
    sha_paquete: str
    sha_actual: str | None = None


# --------------------------------------------------------------------------- #
# armado                                                                      #
# --------------------------------------------------------------------------- #

_CONSTITUCION_DEST = "constitucion/CLAUDE.md"


def ruta_limpia(rel: object) -> bool:
    """Una ruta relativa POSIX sin `..`, `.`, vacios, barra inicial, `\\`, NUL ni `.git`. Se exige a
    CADA entrada que viene del arbol de git y a cada ruta del manifiesto: un `git mktree` puede
    fabricar entradas llamadas `..`."""
    if not isinstance(rel, str) or not rel or "\0" in rel or "\\" in rel or rel.startswith("/"):
        return False
    partes = PurePosixPath(rel).parts
    return rel == "/".join(partes) and all(p not in ("", ".", "..") and p.lower() != ".git" for p in partes)


def _sello(sha: str) -> bytes:
    return f"<!-- claude-skills: SHA {sha} -->\n".encode()


def _seleccionar(cfg: ConfigFaro) -> dict[str, "EntradaGit"]:
    """{ruta en el paquete: entrada de git} de lo que el spec pide, leido del SHA. TODA entrada
    del arbol que pase por las rutas de origen se valida (ruta limpia, sin symlinks ni
    submodulos) y, si alguna falla, se rechaza la fuente entera."""
    entradas = listar(cfg.repo, cfg.sha, _FUENTE_CONSTITUCION, _FUENTE_SKILLS, _FUENTE_AGENTES)
    for e in entradas:
        if not ruta_limpia(e.ruta):
            raise FuenteInvalida(f"ruta no permitida en el arbol de {cfg.sha[:12]}: {e.ruta!r}")
        if e.modo == "120000":
            raise FuenteInvalida(f"{e.ruta} es un symlink en {cfg.sha[:12]}: no se sigue ni se copia")
        if e.modo not in ("100644", "100755"):
            raise FuenteInvalida(f"{e.ruta} tiene un modo git no soportado ({e.modo})")
    elegidas: dict[str, EntradaGit] = {}
    for e in entradas:
        if e.ruta == _FUENTE_CONSTITUCION:
            elegidas[_CONSTITUCION_DEST] = e
        elif e.ruta.startswith(_FUENTE_SKILLS + "/"):
            elegidas["skills/" + e.ruta[len(_FUENTE_SKILLS) + 1:]] = e
        elif e.ruta.startswith(_FUENTE_AGENTES + "/") and e.ruta.count("/") == 2 and e.ruta.endswith(".md"):
            elegidas["agentes/" + e.ruta[len(_FUENTE_AGENTES) + 1:]] = e
    if _CONSTITUCION_DEST not in elegidas:
        raise FuenteInvalida(f"{_FUENTE_CONSTITUCION} no existe en {cfg.sha[:12]}")
    if not any(k.startswith("skills/") for k in elegidas):
        raise FuenteInvalida(f"{_FUENTE_SKILLS} esta vacio en {cfg.sha[:12]}")
    for destino in elegidas:
        if not ruta_limpia(destino):
            raise FuenteInvalida(f"ruta de destino no permitida: {destino!r}")
    return elegidas


def _modo_de(e: "EntradaGit") -> int:
    return 0o755 if e.modo == "100755" else 0o644


def _archivos_desde_git(cfg: ConfigFaro) -> tuple[dict[str, tuple[int, bytes]], dict[str, str]]:
    """({ruta: (modo, bytes)}, {ruta: oid git del blob de origen})."""
    elegidas = _seleccionar(cfg)
    blobs = leer_blobs(cfg.repo, [e.oid for e in elegidas.values()])
    archivos, oids = {}, {}
    for destino, e in sorted(elegidas.items()):
        datos = blobs[e.oid]
        if destino == _CONSTITUCION_DEST:
            datos = _sello(cfg.sha) + datos
        archivos[destino] = (_modo_de(e), datos)
        oids[destino] = e.oid
    return archivos, oids


def _sha256(datos: bytes) -> str:
    return hashlib.sha256(datos).hexdigest()


def _oid_de_bytes(datos: bytes, largo_oid: int) -> str:
    """El oid que git le daria a un blob con estos bytes (sha1 o sha256 segun el formato del repo)."""
    h = hashlib.sha1 if largo_oid == 40 else hashlib.sha256
    return h(b"blob %d\0" % len(datos) + datos).hexdigest()


def hash_del_manifiesto(manifiesto: dict) -> str:
    """sha256 del cuerpo canonico del manifiesto, SIN su propio campo `sha256_manifiesto`."""
    cuerpo = {k: v for k, v in manifiesto.items() if k != "sha256_manifiesto"}
    return _sha256(json.dumps(cuerpo, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode())


_O_BASE = os.O_CLOEXEC


def _escribir_archivos(raiz: Path, archivos: dict[str, tuple[int, bytes]]) -> None:
    """Escribe `archivos` bajo `raiz` (que ya existe) CONFINADO a ella: cada directorio se abre
    con `openat` relativo al anterior, con O_NOFOLLOW|O_DIRECTORY, y cada archivo con
    O_CREAT|O_EXCL|O_NOFOLLOW. Un enlace en el camino (o una ruta con `..`) no desvia la escritura."""
    try:
        raiz_fd = os.open(raiz, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | _O_BASE)
    except OSError as exc:
        raise FuenteInvalida(f"la raiz {raiz} no se puede abrir sin seguir enlaces: {exc.strerror}") from exc
    abiertos: dict[tuple[str, ...], int] = {(): raiz_fd}
    try:
        for rel, (modo, datos) in archivos.items():
            if not ruta_limpia(rel):
                raise FuenteInvalida(f"ruta no permitida: {rel!r}")
            *dirs, nombre = rel.split("/")
            actual = ()
            for parte in dirs:
                previo, actual = actual, actual + (parte,)
                if actual not in abiertos:
                    try:
                        try:
                            os.mkdir(parte, 0o755, dir_fd=abiertos[previo])
                        except FileExistsError:
                            pass
                        fd = os.open(parte, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | _O_BASE, dir_fd=abiertos[previo])
                    except OSError as exc:
                        raise FuenteInvalida(f"{'/'.join(actual)} no es un directorio propio (enlace o fuera de la raiz): {exc.strerror}") from exc
                    os.fchmod(fd, 0o755)
                    abiertos[actual] = fd
            try:
                fd = os.open(nombre, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | _O_BASE, 0o600, dir_fd=abiertos[actual])
            except OSError as exc:
                raise FuenteInvalida(f"{rel} no se puede crear sin seguir enlaces: {exc.strerror}") from exc
            with os.fdopen(fd, "wb") as f:
                f.write(datos)
                os.fchmod(f.fileno(), modo)
    finally:
        for fd in abiertos.values():
            os.close(fd)


def _escribir_paquete(raiz: Path, cfg: ConfigFaro, archivos: dict[str, tuple[int, bytes]], oids: dict[str, str]) -> None:
    raiz.mkdir(mode=0o755)
    os.chmod(raiz, 0o755)
    _escribir_archivos(raiz, archivos)
    manifiesto = {
        "esquema": ESQUEMA,
        "sha_origen": cfg.sha,
        "plugins": [],
        "archivos": {rel: {"modo": f"{modo:04o}", "sha256": _sha256(datos), "oid_git": oids[rel]}
                     for rel, (modo, datos) in archivos.items()},
    }
    manifiesto["sha256_manifiesto"] = hash_del_manifiesto(manifiesto)
    cuerpo = (json.dumps(manifiesto, sort_keys=True, indent=1) + "\n").encode()
    _escribir_archivos(raiz, {MANIFIESTO: (0o644, cuerpo)})


def _crear_destino(destino: Path) -> None:
    """Crea los directorios que falten con 0755 explicito (sin depender del umask: un umask 002
    dejaria g+w y la verificacion de ancestros lo rechazaria)."""
    faltan = [p for p in (destino, *destino.parents) if not p.exists()]
    for p in reversed(faltan):
        p.mkdir(mode=0o755, exist_ok=True)
        os.chmod(p, 0o755)


def construir_paquete(cfg: ConfigFaro) -> Path:
    """Arma `<destino>/<SHA>/` y devuelve su ruta. Se construye en un directorio temporal
    hermano, se VERIFICA y recien entonces se publica con `rename` (atomico en el mismo
    sistema de archivos). Si ya existe un paquete de ese SHA: integro -> se devuelve tal
    cual; alterado -> `PaqueteNoVerifica` (no se pisa ni se repara en silencio)."""
    final = cfg.raiz_paquete
    if final.exists() or final.is_symlink():
        fallos, _ = _verificar_raiz(final, cfg.sha, cfg.uid_duenio)
        if fallos:
            raise PaqueteNoVerifica(fallos)
        return final
    exigir_sha_ancestro_de(cfg.repo, cfg.sha, cfg.ref_frescura)
    archivos, oids = _archivos_desde_git(cfg)
    _crear_destino(cfg.destino)
    tmp = cfg.destino / f".construyendo-{cfg.sha[:12]}-{os.getpid()}-{secrets.token_hex(4)}"
    try:
        _escribir_paquete(tmp, cfg, archivos, oids)
        fallos, _ = _verificar_raiz(tmp, cfg.sha, cfg.uid_duenio)
        if fallos:
            raise PaqueteNoVerifica(fallos)
        try:
            os.rename(tmp, final)
        except OSError:
            # Otro constructor publico el mismo SHA entre medias: vale si esta integro.
            fallos, _ = _verificar_raiz(final, cfg.sha, cfg.uid_duenio)
            if fallos:
                raise PaqueteNoVerifica(fallos) from None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return final


# --------------------------------------------------------------------------- #
# verificacion                                                                #
# --------------------------------------------------------------------------- #

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
        if (not ruta_limpia(rel) or rel == MANIFIESTO or not isinstance(meta, dict)
                or meta.get("modo") not in _MODOS or not _RE_SHA256.fullmatch(str(meta.get("sha256", "")))
                or not _RE_OID.fullmatch(str(meta.get("oid_git", "")))):
            return None
    return m


def _recorrer_disco(raiz: Path) -> tuple[dict[str, Path], set[str], set[str], dict[str, os.stat_result]]:
    """(archivos regulares, directorios, especiales: symlinks y demas, lstat de cada uno) bajo
    `raiz`, sin seguir ningun enlace."""
    regulares: dict[str, Path] = {}
    dirs: set[str] = set()
    especiales: set[str] = set()
    stats: dict[str, os.stat_result] = {}
    pila = [raiz]
    while pila:
        actual = pila.pop()
        with os.scandir(actual) as it:
            for e in it:
                rel = Path(e.path).relative_to(raiz).as_posix()
                st = e.stat(follow_symlinks=False)
                stats[rel] = st
                if stat.S_ISDIR(st.st_mode):
                    dirs.add(rel)
                    pila.append(Path(e.path))
                elif stat.S_ISREG(st.st_mode):
                    regulares[rel] = Path(e.path)
                else:
                    especiales.add(rel)
    return regulares, dirs, especiales, stats


def _fallos_de_ancestros(raiz: Path, uid_duenio: int) -> list[Fallo]:
    """Cada ancestro de la raiz tiene que ser del dueño esperado o de root y no admitir
    escritura ajena (salvo directorio con sticky, como /tmp)."""
    fallos = []
    for ancestro in Path(os.path.realpath(raiz)).parents:
        try:
            st = os.lstat(ancestro)
        except OSError:
            fallos.append(Fallo("ancestro_inseguro", str(ancestro)))
            continue
        con_sticky = stat.S_ISDIR(st.st_mode) and bool(st.st_mode & stat.S_ISVTX)
        if st.st_uid not in (uid_duenio, 0) or (st.st_mode & 0o022 and not con_sticky):
            fallos.append(Fallo("ancestro_inseguro", str(ancestro)))
    return fallos


def _verificar_raiz(raiz: Path, sha_pedido: str, uid_duenio: int) -> tuple[tuple[Fallo, ...], dict | None]:
    """(fallos, manifiesto verificado). Con fallos, el manifiesto puede ser None."""
    if raiz.is_symlink() or not raiz.is_dir():
        return (Fallo("paquete_ausente"),), None
    manifiesto = _leer_manifiesto(raiz)
    if manifiesto is None:
        return (Fallo("manifiesto_ilegible"),), None
    if manifiesto["sha256_manifiesto"] != hash_del_manifiesto(manifiesto):
        return (Fallo("manifiesto_hash"),), None
    if manifiesto["sha_origen"] != sha_pedido:
        return (Fallo("sha_distinto_del_pedido", manifiesto["sha_origen"]),), None
    try:
        regulares, dirs, especiales, stats = _recorrer_disco(raiz)
        st_raiz = os.lstat(raiz)
    except OSError:
        return (Fallo("paquete_ilegible"),), None
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
    # El dueño: la raiz, cada entrada y los ancestros. Un paquete que otro puede reescribir no vale.
    for rel, st in [("", st_raiz), *sorted(stats.items())]:
        if st.st_uid != uid_duenio:
            fallos.append(Fallo("duenio_distinto", rel))
        if st.st_mode & 0o022 and not stat.S_ISLNK(st.st_mode):
            fallos.append(Fallo("escritura_ajena", rel))
    fallos.extend(_fallos_de_ancestros(raiz, uid_duenio))
    return tuple(fallos), manifiesto


def verificar_integridad(cfg: ConfigFaro) -> tuple[Fallo, ...]:
    """Tupla vacia = integro. Nunca lanza: devuelve los fallos (como `arranque.py`)."""
    return _verificar_raiz(cfg.raiz_paquete, cfg.sha, cfg.uid_duenio)[0]


def exigir_integridad(cfg: ConfigFaro) -> Path:
    """Devuelve la raiz del paquete o `PaqueteNoVerifica`. Es la comprobacion CONTRA EL
    MANIFIESTO y el dueño; atar el contenido al SHA es de `cargar_paquete`."""
    fallos = verificar_integridad(cfg)
    if fallos:
        raise PaqueteNoVerifica(fallos)
    return cfg.raiz_paquete


def frescura(cfg: ConfigFaro) -> Frescura:
    """¿Hay un SHA mas nuevo en `origin/main`? SOLO AVISA (log WARNING); jamas bloquea ni
    lanza. Distingue `atrasado` (el paquete sigue siendo ancestro de main) de `retirado` (ya no
    lo es: la historia se reescribio o la rama se cambio). Lee la ref local: quien mantiene el
    checkout es quien la mueve; esta funcion no hace red."""
    actual = resolver_ref(cfg.repo, cfg.ref_frescura)
    if not sha_valido(actual):
        logger.warning("frescura del paquete %s: desconocida (no se pudo leer %s)", cfg.sha[:12], cfg.ref_frescura)
        return Frescura("desconocida", cfg.sha)
    if actual == cfg.sha:
        return Frescura("al_dia", cfg.sha, actual)
    if es_ancestro(cfg.repo, cfg.sha, cfg.ref_frescura) is False:
        logger.warning("el paquete del ecosistema %s fue RETIRADO de %s (ya no es ancestro de %s): se sigue sirviendo, "
                       "no se bloquea nada", cfg.sha, cfg.ref_frescura, actual)
        return Frescura("retirado", cfg.sha, actual)
    logger.warning("el paquete del ecosistema %s esta atrasado: %s ya esta en %s (no se bloquea nada)",
                   cfg.sha, actual, cfg.ref_frescura)
    return Frescura("atrasado", cfg.sha, actual)


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
    """Verifica el paquete y lo ATA AL SHA, y carga en memoria lo que el Puerto sirve.

    1. integridad contra el manifiesto y dueño (`_verificar_raiz`; devuelve el manifiesto ya
       verificado, que no se vuelve a leer);
    2. el conjunto de rutas, modos y oids del manifiesto tiene que ser EXACTAMENTE el del arbol
       de git del SHA (`git --no-replace-objects ls-tree`): un manifiesto coherente pero forjado
       no pasa. Esto necesita leer el repo (el espejo): si git no puede leerlo, no se carga;
    3. cada archivo se lee SIN seguir enlaces y sus bytes se vuelven a comparar con sha256 y con el
       oid git del arbol: lo que se sirve es exactamente lo verificado, sin ventana entre la
       comprobacion y el uso."""
    fallos, manifiesto = _verificar_raiz(cfg.raiz_paquete, cfg.sha, cfg.uid_duenio)
    if fallos:
        raise PaqueteNoVerifica(fallos)
    raiz = cfg.raiz_paquete
    esperados = manifiesto["archivos"]
    arbol = _seleccionar(cfg)
    fallos_arbol: list[Fallo] = []
    for rel in sorted(set(arbol) | set(esperados)):
        if rel not in arbol:
            fallos_arbol.append(Fallo("archivo_fuera_del_arbol", rel))
        elif rel not in esperados:
            fallos_arbol.append(Fallo("archivo_del_arbol_ausente", rel))
        else:
            if esperados[rel]["oid_git"] != arbol[rel].oid:
                fallos_arbol.append(Fallo("oid_distinto_del_arbol", rel))
            if _MODOS[esperados[rel]["modo"]] != _modo_de(arbol[rel]):
                fallos_arbol.append(Fallo("modo_distinto_del_arbol", rel))
    if fallos_arbol:
        raise PaqueteNoVerifica(tuple(fallos_arbol))
    memoria: dict[str, bytes] = {}
    for rel in sorted(esperados):
        try:
            datos = _leer_sin_seguir_enlaces(raiz / rel)
        except OSError:
            raise PaqueteNoVerifica((Fallo("archivo_faltante", rel),)) from None
        if _sha256(datos) != esperados[rel]["sha256"]:
            raise PaqueteNoVerifica((Fallo("archivo_alterado", rel),))
        origen = datos
        if rel == _CONSTITUCION_DEST:
            sello = _sello(cfg.sha)
            if not datos.startswith(sello):
                raise PaqueteNoVerifica((Fallo("sello_distinto", rel),))
            origen = datos[len(sello):]
        if _oid_de_bytes(origen, len(arbol[rel].oid)) != arbol[rel].oid:
            raise PaqueteNoVerifica((Fallo("oid_distinto_del_arbol", rel),))
        memoria[rel] = datos
    skills: dict[str, dict[str, bytes]] = {}
    agentes: dict[str, bytes] = {}
    for rel, datos in memoria.items():
        partes = rel.split("/")
        if partes[0] == "skills" and len(partes) >= 3:
            skills.setdefault(partes[1], {})["/".join(partes[2:])] = datos
        elif partes[0] == "agentes" and len(partes) == 2 and partes[1].endswith(".md"):
            agentes[partes[1][:-3]] = datos
    skills = {n: archivos for n, archivos in skills.items() if "SKILL.md" in archivos}
    return PaqueteCargado(
        sha=cfg.sha,
        constitucion=memoria[_CONSTITUCION_DEST].decode("utf-8"),
        skills=MappingProxyType({n: MappingProxyType(a) for n, a in skills.items()}),
        agentes=MappingProxyType(agentes),
    )
