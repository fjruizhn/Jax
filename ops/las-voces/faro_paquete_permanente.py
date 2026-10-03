#!/usr/bin/env python3
"""Publica el paquete Faro permanente de LAS VOCES, sólo como root.

El operador instala este archivo desde un snapshot de JAX propiedad de root. Nunca se
ejecuta desde un checkout de un usuario: antes de modificar el espejo o el ambiente se
comprueba la cadena de archivos Python que se importó.
"""
from __future__ import annotations

import argparse
import os
import re
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_OFICIAL = "fjruizhn/claude-skills"
MIRROR = Path("/srv/faro/claude-skills.git")
DESTINO = Path("/srv/faro/ecosistema")
ENV_FILE = Path("/etc/jax/las-voces-faro.env")
REF_FRESCURA = "refs/heads/main"
_RE_SHA = re.compile(r"^[0-9a-f]{40}$")
_NOMBRES_ENV = (
    "JAX_FARO_REPO",
    "JAX_FARO_SHA",
    "JAX_FARO_ECOSISTEMA_DIR",
    "JAX_FARO_REF_FRESCURA",
)


class PublicacionRechazada(RuntimeError):
    """La precondición de confianza falló; no se publica ni se reemplaza nada."""


ConfigFaro = None
cargar_paquete = None
construir_paquete = None
verificar_contra_arbol = None


def _run(argv: list[str], *, acepta_ausencia: bool = False) -> str:
    entorno = {"PATH": "/usr/bin:/bin", "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null",
               "GIT_TERMINAL_PROMPT": "0"}
    try:
        resultado = subprocess.run(argv, check=not acepta_ausencia, text=True, capture_output=True, env=entorno)
        if acepta_ausencia and resultado.returncode not in (0, 1):
            raise subprocess.CalledProcessError(resultado.returncode, argv, stderr=resultado.stderr)
        return resultado.stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        detalle = getattr(exc, "stderr", "") or str(exc)
        raise PublicacionRechazada(f"comando rechazado: {' '.join(argv[:3])}: {detalle.strip()}") from None


def _componentes(ruta: Path) -> tuple[Path, ...]:
    absoluta = Path(os.path.abspath(os.fspath(ruta)))
    partes: list[Path] = [Path("/")]
    actual = Path("/")
    for parte in absoluta.parts[1:]:
        actual /= parte
        partes.append(actual)
    return tuple(partes)


def _ruta_root_segura(ruta: Path, *, directorio: bool | None = None) -> None:
    """Exige dueño root, sin enlaces ni escritura de grupo/otros, incluidos padres."""
    try:
        datos_final = os.lstat(ruta)
        if stat.S_ISLNK(datos_final.st_mode):
            raise PublicacionRechazada(f"enlace no permitido: {ruta}")
        if directorio is True and not stat.S_ISDIR(datos_final.st_mode):
            raise PublicacionRechazada(f"se esperaba directorio: {ruta}")
        if directorio is False and not stat.S_ISREG(datos_final.st_mode):
            raise PublicacionRechazada(f"se esperaba archivo regular: {ruta}")
        for componente in _componentes(ruta):
            datos = os.lstat(componente)
            if stat.S_ISLNK(datos.st_mode):
                raise PublicacionRechazada(f"enlace no permitido: {componente}")
            if datos.st_uid != 0 or datos.st_gid != 0:
                raise PublicacionRechazada(f"no es root:root: {componente}")
            if stat.S_IMODE(datos.st_mode) & 0o022:
                raise PublicacionRechazada(f"es escribible por grupo/otros: {componente}")
    except OSError as exc:
        raise PublicacionRechazada(f"ruta no verificable: {ruta}: {exc}") from None


def _cargar_modulos_confiables() -> None:
    """Valida primero el origen y sólo entonces importa Faro desde ese árbol root-owned."""
    global ConfigFaro, cargar_paquete, construir_paquete, verificar_contra_arbol
    propio = Path(os.path.abspath(__file__))
    _ruta_root_segura(propio, directorio=False)
    raiz = next((c for c in propio.parents if (c / "jax/faro/config.py").is_file()), None)
    if raiz is None:
        raise PublicacionRechazada("no se encontró raíz JAX junto al publicador")
    # Python ejecuta __init__.py y dependencias transitivas durante el import. El
    # snapshot completo de jax tiene que ser root-owned antes del PRIMER import.
    fuentes = tuple((raiz / "jax").rglob("*.py"))
    if not fuentes:
        raise PublicacionRechazada("snapshot JAX sin fuentes Python")
    for fuente in fuentes:
        _ruta_root_segura(fuente, directorio=False)
    sys.path[:] = [str(raiz)] + [p for p in sys.path if p != str(raiz)]
    import jax
    import jax.faro.config as config
    import jax.faro.paquete as paquete
    for modulo in (jax, config, paquete):
        archivo = getattr(modulo, "__file__", None)
        if not archivo:
            raise PublicacionRechazada(f"módulo sin archivo verificable: {modulo.__name__}")
        ruta_modulo = Path(archivo)
        try:
            ruta_modulo.relative_to(raiz)
        except ValueError:
            raise PublicacionRechazada(f"módulo fuera del snapshot confiable: {ruta_modulo}") from None
        _ruta_root_segura(ruta_modulo, directorio=False)
    ConfigFaro = config.ConfigFaro
    cargar_paquete = paquete.cargar_paquete
    construir_paquete = paquete.construir_paquete
    verificar_contra_arbol = paquete.verificar_contra_arbol


def _exigir_root_y_codigo() -> None:
    if os.geteuid() != 0:
        raise PublicacionRechazada("este publicador requiere root")
    if not sys.flags.isolated:
        raise PublicacionRechazada("invocar con python3 -I para aislar PYTHONPATH y site")
    _cargar_modulos_confiables()


def _oid_oficial() -> str:
    rama = _run(["gh", "api", f"repos/{REPO_OFICIAL}", "--jq", ".default_branch"]).strip()
    if rama != "main":
        raise PublicacionRechazada(f"default branch inesperado: {rama!r}")
    oid = _run(["gh", "api", f"repos/{REPO_OFICIAL}/git/ref/heads/{rama}", "--jq", ".object.sha"]).strip()
    if not _RE_SHA.fullmatch(oid):
        raise PublicacionRechazada("SHA oficial inválido")
    return oid


def _crear_directorio_root(ruta: Path) -> None:
    padre = ruta.parent
    _ruta_root_segura(padre, directorio=True)
    ruta.mkdir(mode=0o755, exist_ok=True)
    _ruta_root_segura(ruta, directorio=True)


def _validar_mirror() -> None:
    _ruta_root_segura(MIRROR, directorio=True)
    for obligatorio in (MIRROR / "config", MIRROR / "HEAD"):
        _ruta_root_segura(obligatorio, directorio=False)
    for opcional in (MIRROR / "packed-refs", MIRROR / "objects/info/alternates"):
        if opcional.exists() or opcional.is_symlink():
            _ruta_root_segura(opcional, directorio=False)
    alternates = MIRROR / "objects/info/alternates"
    if alternates.exists() and alternates.read_bytes().strip():
        raise PublicacionRechazada("el espejo declara object alternates")
    git = ["/usr/bin/git", "--no-replace-objects", "-c", "core.hooksPath=/dev/null", "-c", "core.fsmonitor=false", "-C", str(MIRROR)]
    if _run([*git, "rev-parse", "--is-bare-repository"]).strip() != "true":
        raise PublicacionRechazada("el espejo no es bare")
    if _run([*git, "replace", "-l"]).strip():
        raise PublicacionRechazada("el espejo tiene refs/replace")
    if _run([*git, "config", "--local", "--get-all", "safe.directory"], acepta_ausencia=True).strip():
        raise PublicacionRechazada("el espejo declara safe.directory")


def _preparar_mirror(oid_oficial: str) -> None:
    _crear_directorio_root(MIRROR.parent)
    if not MIRROR.exists():
        _run(["/usr/bin/git", "--no-replace-objects", "-c", "core.hooksPath=/dev/null", "-c", "core.fsmonitor=false",
              "clone", "--mirror", "--config", "core.sshCommand=ssh -oBatchMode=yes",
              "git@github.com:fjruizhn/claude-skills.git", str(MIRROR)])
    _validar_mirror()
    git = ["/usr/bin/git", "--no-replace-objects", "-c", "core.hooksPath=/dev/null", "-c", "core.fsmonitor=false", "-C", str(MIRROR)]
    _run([*git, "-c", "core.sshCommand=ssh -oBatchMode=yes", "fetch", "--prune",
          "git@github.com:fjruizhn/claude-skills.git", "+refs/heads/*:refs/heads/*"])
    _validar_mirror()
    oid_mirror = _run([*git, "rev-parse", REF_FRESCURA]).strip()
    if oid_mirror != oid_oficial:
        raise PublicacionRechazada("el mirror no coincide con el SHA oficial consultado")


def _env_bytes(sha: str) -> bytes:
    if not _RE_SHA.fullmatch(sha):
        raise PublicacionRechazada("SHA a publicar inválido")
    return (
        f"JAX_FARO_REPO={MIRROR}\n"
        f"JAX_FARO_SHA={sha}\n"
        f"JAX_FARO_ECOSISTEMA_DIR={DESTINO}\n"
        f"JAX_FARO_REF_FRESCURA={REF_FRESCURA}\n"
    ).encode("ascii")


def _publicar_env(sha: str) -> None:
    _crear_directorio_root(ENV_FILE.parent)
    fd, nombre = tempfile.mkstemp(prefix=f".{ENV_FILE.name}.", dir=ENV_FILE.parent)
    temporal = Path(nombre)
    try:
        os.fchmod(fd, 0o644)
        os.fchown(fd, 0, 0)
        with os.fdopen(fd, "wb") as archivo:
            archivo.write(_env_bytes(sha))
            archivo.flush()
            os.fsync(archivo.fileno())
        os.replace(temporal, ENV_FILE)
        descriptor_dir = os.open(ENV_FILE.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor_dir)
        finally:
            os.close(descriptor_dir)
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        temporal.unlink(missing_ok=True)
        raise
    _ruta_root_segura(ENV_FILE, directorio=False)
    if stat.S_IMODE(ENV_FILE.stat().st_mode) != 0o644:
        raise PublicacionRechazada("modo final del env no es 0644")


def _leer_env_exacta() -> dict[str, str]:
    _ruta_root_segura(ENV_FILE, directorio=False)
    if stat.S_IMODE(ENV_FILE.stat().st_mode) != 0o644:
        raise PublicacionRechazada("el env no tiene modo 0644")
    try:
        lineas = ENV_FILE.read_text(encoding="ascii").splitlines()
    except UnicodeError as exc:
        raise PublicacionRechazada("el env no es ASCII") from exc
    valores: dict[str, str] = {}
    for linea in lineas:
        nombre, separador, valor = linea.partition("=")
        if not separador or nombre not in _NOMBRES_ENV or nombre in valores or not valor:
            raise PublicacionRechazada("el env no contiene exactamente las cuatro variables")
        valores[nombre] = valor
    if tuple(valores) != _NOMBRES_ENV:
        raise PublicacionRechazada("el env no contiene exactamente las cuatro variables")
    if (valores["JAX_FARO_REPO"] != str(MIRROR)
            or valores["JAX_FARO_ECOSISTEMA_DIR"] != str(DESTINO)
            or valores["JAX_FARO_REF_FRESCURA"] != REF_FRESCURA
            or not _RE_SHA.fullmatch(valores["JAX_FARO_SHA"])):
        raise PublicacionRechazada("el env contiene valores fuera del contrato")
    return valores


def comprobar_publicacion() -> None:
    valores = _leer_env_exacta()
    cfg = ConfigFaro(repo=MIRROR, sha=valores["JAX_FARO_SHA"], destino=DESTINO,
                     ref_frescura=REF_FRESCURA, uid_duenio=0)
    cargar_paquete(cfg)


def publicar() -> str:
    _exigir_root_y_codigo()
    oid = _oid_oficial()
    _preparar_mirror(oid)
    cfg = ConfigFaro(repo=MIRROR, sha=oid, destino=DESTINO,
                     ref_frescura=REF_FRESCURA, uid_duenio=0)
    construir_paquete(cfg)
    fallos = verificar_contra_arbol(cfg)
    if fallos:
        raise PublicacionRechazada("verificar_contra_arbol falló: " + ", ".join(x.codigo for x in fallos))
    cargar_paquete(cfg)
    _publicar_env(oid)
    comprobar_publicacion()
    return oid


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verifica el env y paquete ya publicados; no muta")
    argumentos = parser.parse_args(argv)
    try:
        _exigir_root_y_codigo()
        if argumentos.check:
            comprobar_publicacion()
            print("paquete Faro permanente verificado")
        else:
            print(publicar())
    except PublicacionRechazada as exc:
        print(f"falla cerrada: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
