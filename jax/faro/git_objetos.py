"""Lectura de OBJETOS git del repo de skills del ecosistema: el unico lugar del Faro que lanza `git`.

Solo lectura y siempre contra un SHA (`ls-tree`, `cat-file`, `merge-base`, `rev-parse`): jamas el
arbol de trabajo del checkout. Va aparte de `paquete.py` para que el lanzamiento de subprocesos
quede en un modulo chico y revisable.

Mismo modelo de amenaza que el ensamblado de `lib/assemble.py` del repo de skills: ninguna
invocacion de git ejecuta hooks ni fsmonitor del repo que se esta leyendo.
"""
from __future__ import annotations

import os
import re
import select
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

_GIT_SIN_HOOKS = ("-c", "core.hooksPath=/dev/null", "-c", "core.fsmonitor=false")

# Tope de tamano de blob para el snapshot de reglas (F1.1 §6): una regla es un
# YAML chico; algo de megas no es una regla, es un ataque de memoria. NO es un
# default de leer_blobs: el paquete del Faro lee blobs de ~800 KB legitimamente,
# asi que el tope es OPT-IN del snapshot (M-6, jax#370 r3).
MAX_BLOB_BYTES = 1024 * 1024

# El lector del snapshot nunca deja que un arbol fijado convierta una lectura
# chica de reglas en una enumeracion ilimitada. Estos limites cubren commit,
# arboles y blobs juntos; una regla individual conserva su limite de 1 MiB.
MAX_SNAPSHOT_OBJECTS = 1024
MAX_SNAPSHOT_TREE_DEPTH = 16
MAX_SNAPSHOT_BYTES = 8 * 1024 * 1024
MAX_SNAPSHOT_TREE_BYTES = 512 * 1024
MAX_SNAPSHOT_COMMIT_BYTES = 1024 * 1024
_RE_OID = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")

# `git replace` (refs/replace/) hace que un SHA muestre OTRO contenido sin cambiar el SHA: un
# paquete «fijado por SHA» dejaria de serlo. Se apaga con la bandera Y con la variable.
_SIN_REPLACE = ("--no-replace-objects",)


def _entorno_limpio() -> dict[str, str]:
    """El entorno de git es ESTE y no el heredado: ninguna `GIT_*` del proceso (GIT_DIR,
    GIT_WORK_TREE, GIT_OBJECT_DIRECTORY, GIT_CONFIG_PARAMETERS, GIT_CONFIG_COUNT...) puede
    desviar la lectura. Solo `PATH` (para encontrar `git`); configuracion global y de sistema
    apagadas; sin `safe.directory` (el repo tiene que ser del usuario que lo lee)."""
    return {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "LC_ALL": "C",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_NO_LAZY_FETCH": "1",
        "GIT_TERMINAL_PROMPT": "0",
    }


class FuenteInvalida(RuntimeError):
    """Lo que se quiere empaquetar no se puede empaquetar con garantias (SHA que no es de
    `origin/main`, symlink, plugin ausente...). La construccion no publica nada."""


@dataclass(frozen=True)
class ObjetoGit:
    tipo: str
    contenido: bytes


@dataclass(frozen=True)
class InventarioFaroVerificado:
    entradas: tuple["EntradaGit", ...]
    objetos_restantes: int
    bytes_restantes: int


def git(repo: Path, *args: str, entrada: bytes | None = None, aceptar: tuple[int, ...] = (0,)) -> subprocess.CompletedProcess:
    try:
        r = subprocess.run(["git", *_SIN_REPLACE, *_GIT_SIN_HOOKS, "-C", str(repo), *args], input=entrada,
                           capture_output=True, timeout=120, check=False, env=_entorno_limpio())
    except (OSError, subprocess.SubprocessError) as exc:
        raise FuenteInvalida(f"no se pudo ejecutar git sobre {repo}: {type(exc).__name__}") from exc
    if r.returncode not in aceptar:
        raise FuenteInvalida(f"git {args[0]} fallo (rc={r.returncode}): {r.stderr.decode(errors='replace')[:200].strip()}")
    return r


def exigir_sha_ancestro_de(repo: Path, sha: str, ref: str) -> None:
    """El commit existe y es ancestro de `ref` (en la practica `origin/main`). Si no se puede
    afirmar, se niega (fallo cerrado)."""
    r = git(repo, "cat-file", "-e", f"{sha}^{{commit}}", aceptar=(0, 1, 128))
    if r.returncode != 0:
        raise FuenteInvalida(f"el commit {sha} no existe en {repo}")
    r = git(repo, "merge-base", "--is-ancestor", sha, ref, aceptar=(0, 1, 128))
    if r.returncode != 0:
        # 1 = no es ancestro; 128 = la ref no existe.
        raise FuenteInvalida(f"el commit {sha} no es ancestro de {ref}: solo se fija un SHA de origin/main")


def es_ancestro(repo: Path, sha: str, ref: str) -> bool | None:
    """True/False si `sha` es o no ancestro de `ref`; None si no se puede afirmar. No lanza."""
    try:
        r = git(repo, "merge-base", "--is-ancestor", sha, ref, aceptar=(0, 1, 128))
    except FuenteInvalida:
        return None
    return {0: True, 1: False}.get(r.returncode)


def resolver_ref(repo: Path, ref: str) -> str:
    """El SHA al que apunta `ref`, o "" si no se puede leer. No lanza."""
    try:
        r = git(repo, "rev-parse", "--verify", "-q", f"{ref}^{{commit}}", aceptar=(0, 1, 128))
    except FuenteInvalida:
        return ""
    return r.stdout.decode().strip() if r.returncode == 0 else ""


def largo_oid_del_repositorio(repo: Path) -> int:
    """Formato de almacenamiento del repo, consultado antes de leer objetos."""
    r = git(repo, "rev-parse", "--show-object-format=storage")
    formato = r.stdout.decode("ascii", errors="replace").strip()
    if formato == "sha1":
        return 40
    if formato == "sha256":
        return 64
    raise FuenteInvalida(f"formato de objetos Git no admitido: {formato!r}")


def oid_de_objeto(repo: Path, tipo: str, contenido: bytes, largo_oid: int) -> str:
    """Recalcula el OID con Git, incluido su endurecimiento SHA-1 propio."""
    if largo_oid not in (40, 64):
        raise FuenteInvalida("OID con longitud no soportada")
    r = git(repo, "hash-object", "-t", tipo, "--stdin", entrada=contenido)
    oid = r.stdout.decode("ascii", errors="replace").strip()
    if len(oid) != largo_oid or not _RE_OID.fullmatch(oid):
        raise FuenteInvalida("git hash-object devolvio un OID no canonico")
    return oid


_CAT_FILE_IO_TIMEOUT_SECONDS = 120.0


def _leer_exacto(stream, longitud: int, deadline: float) -> bytes:
    partes = []
    restante = longitud
    while restante:
        espera = deadline - time.monotonic()
        if espera <= 0 or not select.select([stream.fileno()], [], [], espera)[0]:
            raise FuenteInvalida("cat-file excedió el plazo de lectura")
        parte = os.read(stream.fileno(), min(restante, 64 * 1024))
        if not parte:
            raise FuenteInvalida("cat-file devolvio contenido truncado")
        partes.append(parte)
        restante -= len(parte)
    return b"".join(partes)


def _leer_cabecera_cat_file(stream, deadline: float) -> bytes:
    cabecera = bytearray()
    while len(cabecera) < 257:
        cabecera.extend(_leer_exacto(stream, 1, deadline))
        if cabecera[-1] == 0x0A:
            return bytes(cabecera)
    raise FuenteInvalida("cat-file devolvió una cabecera demasiado larga")


def _terminar_cat_file(proceso) -> None:
    try:
        proceso.kill()
    except (OSError, ProcessLookupError, subprocess.SubprocessError):  # fail-soft: cleanup must preserve the original failure
        pass
    try:
        proceso.wait(timeout=5)
    except (OSError, subprocess.SubprocessError):  # fail-soft: cleanup must preserve the original failure
        pass


def _leer_objetos_lote_limitado(
    repo: Path,
    solicitudes: list[tuple[str, str, int]],
    *,
    max_total_bytes: int | None = None,
) -> list[ObjetoGit]:
    """Lee `cat-file --batch` en streaming; decide el límite con su cabecera real.

    No consulta tamaños en otro proceso: un objeto suelto puede cambiar entre dos
    invocaciones. El cuerpo se lee solo después de comprobar su tamaño contra el
    presupuesto individual y el agregado que queda.
    """
    if not solicitudes:
        return []
    try:
        proceso = subprocess.Popen(
            ["git", *_SIN_REPLACE, *_GIT_SIN_HOOKS, "-C", str(repo), "cat-file", "--batch"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            env=_entorno_limpio(), bufsize=0,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise FuenteInvalida(f"no se pudo iniciar cat-file en streaming: {type(exc).__name__}") from exc
    resultados: list[ObjetoGit] = []
    restante_total = max_total_bytes
    deadline = time.monotonic() + _CAT_FILE_IO_TIMEOUT_SECONDS
    try:
        if proceso.stdin is None or proceso.stdout is None:
            raise FuenteInvalida("cat-file no abrió sus pipes")
        for oid, tipo, limite_individual in solicitudes:
            if not _RE_OID.fullmatch(oid) or tipo not in {"blob", "tree", "commit"}:
                raise FuenteInvalida("solicitud de objeto Git inválida")
            limite = limite_individual
            if restante_total is not None:
                limite = min(limite, restante_total)
            consulta = (oid + "\n").encode("ascii")
            if not select.select([], [proceso.stdin.fileno()], [], max(0, deadline - time.monotonic()))[1]:
                raise FuenteInvalida("cat-file excedió el plazo de escritura")
            proceso.stdin.write(consulta)
            proceso.stdin.flush()
            cabecera = _leer_cabecera_cat_file(proceso.stdout, deadline)
            campos = cabecera[:-1].decode("ascii", errors="replace").split(" ")
            if len(campos) != 3 or campos[0] != oid or campos[1] != tipo or not campos[2].isdigit():
                raise FuenteInvalida(f"{oid}: cabecera cat-file inesperada")
            tamano = int(campos[2])
            if str(tamano) != campos[2]:
                raise FuenteInvalida(f"{oid}: tamaño cat-file no canónico")
            if tamano > limite:
                raise FuenteInvalida(f"{tipo} {oid} excede el presupuesto de {limite} bytes")
            contenido = _leer_exacto(proceso.stdout, tamano, deadline)
            if _leer_exacto(proceso.stdout, 1, deadline) != b"\n":
                raise FuenteInvalida(f"{oid}: separador cat-file inválido")
            resultados.append(ObjetoGit(tipo, contenido))
            if restante_total is not None:
                restante_total -= tamano
        proceso.stdin.close()
        proceso.wait(timeout=max(0.001, deadline - time.monotonic()))
        if proceso.stdout.read(1):
            raise FuenteInvalida("cat-file devolvió bytes sobrantes")
        codigo = proceso.returncode
        if codigo != 0:
            raise FuenteInvalida(f"cat-file terminó con rc={codigo}")
        return resultados
    except FuenteInvalida:
        _terminar_cat_file(proceso)
        raise
    except (OSError, subprocess.SubprocessError, UnicodeError, ValueError) as exc:
        _terminar_cat_file(proceso)
        raise FuenteInvalida(f"falló lectura acotada de cat-file: {type(exc).__name__}") from exc
    finally:
        for pipe in (proceso.stdin, proceso.stdout):
            if pipe is not None:
                try:
                    pipe.close()
                except OSError:  # fail-soft: pipe cleanup must preserve the original failure
                    pass


def leer_objeto_verificado(repo: Path, oid: str, *, tipo: str, max_bytes: int) -> ObjetoGit:
    """Lee un objeto y prueba su identidad desde sus bytes crudos.

    ``cat-file`` sabe descomprimir un objeto suelto aunque alguien haya sustituido
    sus bytes bajo el mismo nombre. Por eso el tipo, tamano y OID se vuelven a
    comprobar aqui, antes de que ningun parser interprete commit o arbol.
    """
    if not _RE_OID.fullmatch(oid):
        raise FuenteInvalida("OID de objeto no canonico")
    objeto = _leer_objetos_lote_limitado(repo, [(oid, tipo, max_bytes)])[0]
    contenido = objeto.contenido
    if oid_de_objeto(repo, tipo, contenido, len(oid)) != oid:
        raise FuenteInvalida(f"{tipo} {oid}: OID no corresponde a sus bytes")
    return objeto


def _entradas_arbol(contenido: bytes, largo_oid: int) -> list[tuple[str, str, bytes]]:
    """Decodifica el formato binario de un tree Git ya autenticado."""
    ancho = largo_oid // 2
    i, entradas, nombres = 0, [], set()
    while i < len(contenido):
        fin_modo = contenido.find(b" ", i)
        if fin_modo <= i:
            raise FuenteInvalida("arbol Git con modo invalido")
        modo = contenido[i:fin_modo]
        fin_nombre = contenido.find(b"\0", fin_modo + 1)
        if fin_nombre < 0:
            raise FuenteInvalida("arbol Git truncado")
        nombre = contenido[fin_modo + 1:fin_nombre]
        inicio_oid, fin_oid = fin_nombre + 1, fin_nombre + 1 + ancho
        if not nombre or b"/" in nombre or nombre in (b".", b"..") or fin_oid > len(contenido):
            raise FuenteInvalida("arbol Git con nombre invalido")
        if nombre in nombres:
            raise FuenteInvalida("arbol Git con nombre duplicado")
        if modo not in (b"40000", b"100644", b"100755", b"120000", b"160000"):
            raise FuenteInvalida("arbol Git con modo no admitido")
        oid = contenido[inicio_oid:fin_oid].hex()
        entradas.append((modo.decode("ascii"), oid, nombre))
        nombres.add(nombre)
        i = fin_oid
    return entradas


def listar_faro_desde_commit_verificado(repo: Path, commit: str, policy_oid: str,
                                        *, largo_oid: int) -> InventarioFaroVerificado:
    """Enumera ``policy/faro`` sin confiar en ``ls-tree``.

    Cada commit y tree que se interpreta se relee desde bytes y se verifica con
    el OID fijado. El resultado contiene hojas de ``faro`` relativas a policy.
    """
    if len(commit) != largo_oid or len(policy_oid) != largo_oid:
        raise FuenteInvalida("OID del pin no coincide con el formato del repositorio")
    presupuesto_objetos, presupuesto_bytes = MAX_SNAPSHOT_OBJECTS, MAX_SNAPSHOT_BYTES

    def leer_y_consumir(oid: str, tipo: str, limite_individual: int) -> ObjetoGit:
        nonlocal presupuesto_objetos, presupuesto_bytes
        if presupuesto_objetos <= 0:
            raise FuenteInvalida("snapshot excede limite de objetos")
        limite = min(limite_individual, presupuesto_bytes)
        objeto = leer_objeto_verificado(repo, oid, tipo=tipo, max_bytes=limite)
        presupuesto_objetos -= 1
        presupuesto_bytes -= len(objeto.contenido)
        if presupuesto_objetos < 0 or presupuesto_bytes < 0:
            raise FuenteInvalida("snapshot excede limite de objetos o bytes")
        return objeto

    commit_obj = leer_y_consumir(commit, "commit", MAX_SNAPSHOT_COMMIT_BYTES)
    cabeceras, _, _cuerpo = commit_obj.contenido.partition(b"\n\n")
    arboles = [linea[5:] for linea in cabeceras.split(b"\n") if linea.startswith(b"tree ")]
    if len(arboles) != 1 or not _RE_OID.fullmatch(arboles[0].decode("ascii", errors="replace")):
        raise FuenteInvalida("commit sin arbol canonico")
    raiz = arboles[0].decode("ascii")
    if len(raiz) != largo_oid:
        raise FuenteInvalida("commit apunta a arbol de otro formato")

    def leer_arbol(oid: str, profundidad: int) -> list[tuple[str, str, bytes]]:
        if profundidad > MAX_SNAPSHOT_TREE_DEPTH:
            raise FuenteInvalida("snapshot excede profundidad maxima de arboles")
        objeto = leer_y_consumir(oid, "tree", MAX_SNAPSHOT_TREE_BYTES)
        return _entradas_arbol(objeto.contenido, len(oid))

    raiz_entradas = leer_arbol(raiz, 0)
    policy = next((e for e in raiz_entradas if e[2] == b"policy"), None)
    if policy is None:
        raise FuenteInvalida("commit no contiene policy")
    if policy[0] != "40000":
        return InventarioFaroVerificado((EntradaGit(policy[0].zfill(6), policy[1], "policy"),),
                                        presupuesto_objetos, presupuesto_bytes)
    if policy[1] != policy_oid:
        raise FuenteInvalida("el arbol policy no corresponde al pin")
    policy_entradas = leer_arbol(policy_oid, 1)
    faro = next((e for e in policy_entradas if e[2] == b"faro"), None)
    if faro is None:
        return InventarioFaroVerificado((), presupuesto_objetos, presupuesto_bytes)
    if faro[0] != "40000":
        return InventarioFaroVerificado((EntradaGit(faro[0].zfill(6), faro[1], "faro"),),
                                        presupuesto_objetos, presupuesto_bytes)

    resultado: list[EntradaGit] = []
    def recorrer(oid: str, prefijo: str, profundidad: int) -> None:
        for modo, hijo, nombre_bytes in leer_arbol(oid, profundidad):
            try:
                nombre = nombre_bytes.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise FuenteInvalida("ruta de objeto no UTF-8: el snapshot niega") from exc
            ruta = prefijo + nombre
            if modo == "40000":
                recorrer(hijo, ruta + "/", profundidad + 1)
            else:
                if len(resultado) >= presupuesto_objetos:
                    raise FuenteInvalida("snapshot excede limite de objetos")
                resultado.append(EntradaGit(modo.zfill(6), hijo, ruta))
    recorrer(faro[1], "faro/", 2)
    return InventarioFaroVerificado(tuple(resultado), presupuesto_objetos, presupuesto_bytes)


def oid_subarbol(repo: Path, commit: str, ruta: str) -> str:
    """El OID del ARBOL `ruta` en `commit`, verificado como arbol (F1.1 §6: el pin
    compara el arbol esperado de `policy/` contra el real). Falla cerrado: si
    `commit` no existe o `ruta` no es un arbol, no hay snapshot.

    OJO con la sintaxis: `<commit>:<ruta>^{tree}` NO vale (todo lo que sigue al
    colon se toma como ruta); se usa `ls-tree`, que si declara el tipo."""
    r = git(repo, "ls-tree", "-z", "--full-tree", commit, "--", ruta)
    entradas = [e for e in r.stdout.split(b"\0") if e]
    if len(entradas) != 1:
        raise FuenteInvalida(f"no existe el arbol {ruta} en {commit}")
    meta, _, _ruta = entradas[0].partition(b"\t")
    modo, tipo, oid = meta.decode().split(" ")
    if tipo != "tree":
        raise FuenteInvalida(f"{ruta} no es un arbol en {commit} (tipo {tipo})")
    return oid


@dataclass(frozen=True)
class EntradaGit:
    modo: str
    oid: str
    ruta: str


def listar(repo: Path, sha: str, *rutas: str) -> list[EntradaGit]:
    r = git(repo, "ls-tree", "-r", "-z", "--full-tree", sha, "--", *rutas)
    entradas = []
    for crudo in r.stdout.split(b"\0"):
        if not crudo:
            continue
        meta, _, ruta = crudo.partition(b"\t")
        modo, _tipo, oid = meta.decode("ascii").split(" ")
        try:
            ruta_texto = ruta.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise FuenteInvalida("ruta de objeto no UTF-8: el snapshot niega") from exc
        entradas.append(EntradaGit(modo, oid, ruta_texto))
    return entradas


def leer_blobs(repo: Path, oids: list[str], *, max_bytes: int | None = None,
               max_total_bytes: int | None = None, max_objects: int | None = None) -> dict[str, bytes]:
    """Lee blobs por lote; los límites se aplican a la cabecera del mismo stream
    antes de leer cada body, sin confiar en un precheck de otra invocación Git."""
    unicos = list(dict.fromkeys(oids))
    if not unicos:
        return {}
    if max_objects is not None and len(unicos) > max_objects:
        raise FuenteInvalida(f"snapshot excede limite de objetos ({max_objects})")
    limite_individual = max_bytes
    if limite_individual is None:
        limite_individual = max_total_bytes if max_total_bytes is not None else (1 << 63) - 1
    objetos = _leer_objetos_lote_limitado(
        repo, [(oid, "blob", limite_individual) for oid in unicos],
        max_total_bytes=max_total_bytes,
    )
    blobs = {}
    for oid, objeto in zip(unicos, objetos, strict=True):
        if oid_de_objeto(repo, "blob", objeto.contenido, len(oid)) != oid:
            raise FuenteInvalida(f"blob {oid}: OID no corresponde a sus bytes")
        blobs[oid] = objeto.contenido
    return blobs
