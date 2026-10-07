"""Snapshot confiable de ``policy/faro`` desde objetos Git (diseno F1.1 §6).

`load_trusted_policy_snapshot(repo, pin)` carga las reglas Faro del commit EXACTO
que fija el pin, nunca del arbol de trabajo:

  1. el pin se resuelve UNA sola vez: ``rev-parse --verify --end-of-options
     <commit>^{commit}`` debe devolver EXACTAMENTE ``pin.commit`` (un nombre de
     rama con forma hex, o el oid de un tag anotado, no pasan);
  2. el arbol ``policy/`` de ese commit debe coincidir con el del pin, y la
     enumeracion sale de ESE arbol ya verificado (nada se resuelve dos veces);
  3. lectura por objetos Git con las defensas de ``jax/faro/git_objetos.py``
     (entorno limpio, hooks y fsmonitor apagados, replace objects desactivado);
  4. solo blobs ``100644`` con ruta directa ``faro/<nombre>.yaml`` (desde la
     raiz del arbol policy/); symlinks, submodules, bits de ejecucion, rutas
     anidadas, nombres no canonicos y EXTENSIONES que aparentan ser regla
     (``.YAML``, ``.yaml.bak``) niegan el snapshot completo; los archivos que
     no son reglas (README, ``schemas/*.json``) simplemente no se enumeran;
  5. TODOS los blobs se obtienen antes de validar uno, con tope de tamano;
  6. el OID git de cada blob se RECALCULA desde sus bytes y se compara (un
     objeto suelto adulterado no pasa), y ``rule_content_hash`` se calcula
     sobre los BYTES CRUDOS antes de parsear;
  7. YAML estricto (sin claves duplicadas, aliases ni merge keys) y shape
     cerrado rule-v1;
  8. una regla invalida o un ``rule_id`` duplicado rechazan el snapshot ENTERO;
  9. el resultado es profundamente inmutable y sellado: clases NO-dataclass
     con testigo privado, asi ``dataclasses.replace`` no puede fabricar
     objetos con el sello ajeno, y el hash del snapshot se recalcula dentro
     del constructor ante cualquier manipulacion.

El testigo impide fabricacion ACCIDENTAL por la API publica; no es aislamiento
frente a codigo arbitrario dentro del proceso (§6).
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path

from jax.faro.git_objetos import (
    MAX_BLOB_BYTES,
    FuenteInvalida,
    git,
    leer_blobs,
    listar,
    oid_subarbol,
)
from policy.canonicalization.errors import CanonicalizationError, StrictYAMLError
from policy.canonicalization.strict_yaml import load_strict_yaml

from .errors import RuleSnapshotError
from .schema import ReglaValidada, validar_regla

# Testigo privado de construccion: fuera de este modulo nadie construye objetos
# confiables (ni con dataclasses.replace, que aqui ya no aplica).
_TESTIGO = object()

_PREFIJO = "faro/"                      # rutas relativas al arbol policy/ ya verificado
_RE_NOMBRE_CANONICO = re.compile(r"[a-z][a-z0-9-]{0,63}\.yaml\Z")
_RE_OBJETO = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_DOMINIO_SNAPSHOT = "jax-faro-policy-snapshot-v1"


def _hash_dominio(dominio: str, carga: bytes) -> str:
    return "sha256:" + hashlib.sha256(dominio.encode() + b"\0" + carga).hexdigest()


def _oid_git_de(contenido: bytes, largo_oid: int) -> str:
    """El OID que git calcularia para este blob (sha1 o sha256 segun el repo)."""
    cabecera = b"blob %d\0" % len(contenido)
    digesto = hashlib.sha1(cabecera + contenido) if largo_oid == 40 else hashlib.sha256(cabecera + contenido)
    return digesto.hexdigest()


class TrustedPolicyPin:
    """El pin activo: commit exacto, arbol esperado de ``policy/`` y procedencia.
    Nada de nombres moviles: la raiz es el SHA, no una rama ni un tag. ``repositorio``
    se valida (no vacio, NFC) y se conserva para auditoria; ligarlo al origen real
    (remote/URL) es trabajo del ActivePolicyPinProvider, no del loader."""

    __slots__ = ("repositorio", "commit", "policy_tree_oid", "procedencia")

    def __init__(self, repositorio: str, commit: str, policy_tree_oid: str, procedencia: str) -> None:
        for campo, valor in (("repositorio", repositorio), ("procedencia", procedencia)):
            if not isinstance(valor, str) or not valor.strip() \
                    or unicodedata.normalize("NFC", valor) != valor:
                raise RuleSnapshotError(f"pin.{campo}: no vacio y en NFC")
        for campo, valor in (("commit", commit), ("policy_tree_oid", policy_tree_oid)):
            if not isinstance(valor, str) or not _RE_OBJETO.fullmatch(valor):
                raise RuleSnapshotError(f"pin.{campo}: debe ser un oid hex completo")
        object.__setattr__(self, "repositorio", repositorio)
        object.__setattr__(self, "commit", commit)
        object.__setattr__(self, "policy_tree_oid", policy_tree_oid)
        object.__setattr__(self, "procedencia", procedencia)

    def __setattr__(self, *_args) -> None:
        raise RuleSnapshotError("TrustedPolicyPin es inmutable")

    def __delattr__(self, _name: str) -> None:
        raise RuleSnapshotError("TrustedPolicyPin es inmutable")


class ReglaSellada:
    """Una regla del snapshot, con sus bytes identificados. Solo la construye
    este modulo (testigo privado): una ReglaValidada suelta no es autoridad."""

    __slots__ = ("ruta", "modo", "blob_oid", "content_hash", "regla")

    def __init__(self, *, ruta: str, modo: str, blob_oid: str, content_hash: str,
                 regla: ReglaValidada, _testigo: object = None) -> None:
        # El testigo se COMPARA y se descarta: nunca queda en la instancia, asi
        # nadie lo lee de un objeto legitimo para fabricar otro (r2).
        if _testigo is not _TESTIGO:
            raise RuleSnapshotError("ReglaSellada no se fabrica por la API publica")
        object.__setattr__(self, "ruta", ruta)
        object.__setattr__(self, "modo", modo)
        object.__setattr__(self, "blob_oid", blob_oid)
        object.__setattr__(self, "content_hash", content_hash)
        object.__setattr__(self, "regla", regla)

    def _clave(self) -> tuple:
        return (self.ruta, self.modo, self.blob_oid, self.content_hash, self.regla)

    def __eq__(self, otro: object) -> bool:
        return isinstance(otro, ReglaSellada) and self._clave() == otro._clave()

    def __hash__(self) -> int:
        return hash(self._clave())

    def __setattr__(self, *_args) -> None:
        raise RuleSnapshotError("ReglaSellada es inmutable")

    def __delattr__(self, _name: str) -> None:
        raise RuleSnapshotError("ReglaSellada es inmutable")

    def __reduce__(self) -> tuple:
        # ni pickle ni copy: deserializar no puede fabricar objetos sellados
        raise RuleSnapshotError("ReglaSellada no se serializa")


class TrustedPolicySnapshot:
    """El snapshot sellado. El hash se RECALCULA en la construccion: cualquier
    combinacion de reglas que no corresponda a su propio hash, niega."""

    __slots__ = ("commit", "policy_tree_oid", "reglas", "snapshot_hash",
                 "repositorio", "procedencia")

    def __init__(self, *, commit: str, policy_tree_oid: str, reglas: tuple[ReglaSellada, ...],
                 repositorio: str, procedencia: str, _testigo: object = None) -> None:
        if _testigo is not _TESTIGO:
            raise RuleSnapshotError("TrustedPolicySnapshot no se fabrica por la API publica")
        carga = "".join(f"{r.ruta}\0{r.modo}\0{r.blob_oid}\0{r.content_hash}\n"
                        for r in reglas).encode()
        object.__setattr__(self, "commit", commit)
        object.__setattr__(self, "policy_tree_oid", policy_tree_oid)
        object.__setattr__(self, "reglas", tuple(reglas))
        object.__setattr__(self, "snapshot_hash", _hash_dominio(_DOMINIO_SNAPSHOT, carga))
        object.__setattr__(self, "repositorio", repositorio)
        object.__setattr__(self, "procedencia", procedencia)

    def _clave(self) -> tuple:
        return (self.commit, self.policy_tree_oid, self.reglas, self.snapshot_hash,
                self.repositorio, self.procedencia)

    def __eq__(self, otro: object) -> bool:
        return isinstance(otro, TrustedPolicySnapshot) and self._clave() == otro._clave()

    def __hash__(self) -> int:
        return hash(self._clave())

    def __setattr__(self, *_args) -> None:
        raise RuleSnapshotError("TrustedPolicySnapshot es inmutable")

    def __delattr__(self, _name: str) -> None:
        raise RuleSnapshotError("TrustedPolicySnapshot es inmutable")

    def __reduce__(self) -> tuple:
        raise RuleSnapshotError("TrustedPolicySnapshot no se serializa")


def _clasificar(entradas) -> list:
    """Candidatos a regla del arbol policy/ ya verificado. NIEGA lo raro:
    modos que no sean blob plano, el propio faro/ que no sea arbol, y nombres
    que aparenten regla sin ser canonicos (incluida la extension en mayusculas
    o con cola .bak). Los archivos claramente ajenos a reglas (README.md,
    schemas/*.json) no se enumeran."""
    candidatos = []
    for entrada in entradas:
        if entrada.ruta == "faro":
            raise RuleSnapshotError(
                f"faro no es un arbol (modo {entrada.modo}): symlink o archivo en su lugar")
        if not entrada.ruta.startswith(_PREFIJO):
            continue                                    # fuera de faro/: nada que ver
        if entrada.modo != "100644":
            raise RuleSnapshotError(
                f"{entrada.ruta}: modo {entrada.modo} no admitido (symlink/submodulo/ejecutable)")
        nombre = entrada.ruta[len(_PREFIJO):]
        parece_regla = ".yaml" in nombre.casefold() or ".yml" in nombre.casefold()
        if parece_regla and not _RE_NOMBRE_CANONICO.fullmatch(nombre):
            raise RuleSnapshotError(
                f"{entrada.ruta}: nombre o ruta no canonica para regla (nested/mayusculas/cola)")
        if parece_regla:
            candidatos.append(entrada)
            continue
        # Lo que no es regla solo puede ser infraestructura conocida del
        # directorio: README.md o schemas/<nombre>.json|.md. Cualquier otra cosa
        # —sin extension, punto de ancho completo, cirilicos, colas— niega el
        # snapshot: aqui no se ignora nada en silencio (§6.5).
        _RE_INFRAESTRUCTURA = re.compile(r"[a-z][a-z0-9.-]{0,63}\.(json|md)\Z")
        if nombre in ("README.md", "catalogo-topes.json") or (
                nombre.startswith("schemas/")
                and nombre.count("/") == 1
                and _RE_INFRAESTRUCTURA.fullmatch(nombre[8:])):
            continue
        raise RuleSnapshotError(
            f"{entrada.ruta}: archivo no regla y no infraestructura conocida (§6.5: nada en silencio)")
    return candidatos


def load_trusted_policy_snapshot(repo: Path, pin: TrustedPolicyPin) -> TrustedPolicySnapshot:
    if not isinstance(repo, Path) or not repo.is_dir():
        raise RuleSnapshotError(f"el repositorio {repo} no es un directorio")

    try:
        # (1) UNA sola resolucion: el pin debe SER ese commit exacto. Un nombre
        # de rama con forma hex o el oid de un tag anotado no coinciden y niegan.
        r = git(repo, "rev-parse", "--verify", "--end-of-options", f"{pin.commit}^{{commit}}")
    except FuenteInvalida as exc:
        raise RuleSnapshotError(str(exc)) from exc
    resuelto = r.stdout.decode().strip()
    if resuelto != pin.commit:
        raise RuleSnapshotError(
            f"el pin no es el commit exacto: {pin.commit[:16]}… resuelve a {resuelto[:16]}…")

    try:
        # (2) arbol policy/ contra el pin; la enumeracion sale de ESE arbol.
        arbol = oid_subarbol(repo, pin.commit, "policy")
    except FuenteInvalida as exc:
        raise RuleSnapshotError(str(exc)) from exc
    if arbol != pin.policy_tree_oid:
        raise RuleSnapshotError("el arbol policy/ no coincide con el pin: no se carga nada")

    try:
        entradas = listar(repo, arbol, "faro")
    except FuenteInvalida as exc:
        raise RuleSnapshotError(str(exc)) from exc
    candidatos = _clasificar(entradas)

    # (5) TODOS los blobs antes de validar uno; el snapshot impone su tope de
    # tamano (opt-in de M-6: el paquete del Faro lee blobs grandes legitimamente).
    try:
        blobs = leer_blobs(repo, [e.oid for e in candidatos],
                           max_bytes=MAX_BLOB_BYTES)
    except FuenteInvalida as exc:
        raise RuleSnapshotError(str(exc)) from exc

    # (6) OID recalculado + hash de bytes crudos ANTES de parsear.
    selladas: list[ReglaSellada] = []
    vistos: set[str] = set()
    for entrada in sorted(candidatos, key=lambda e: e.ruta):
        crudo = blobs[entrada.oid]
        oid_real = _oid_git_de(crudo, len(entrada.oid))
        if oid_real != entrada.oid:
            raise RuleSnapshotError(
                f"{entrada.ruta}: el blob no corresponde a su OID (objeto adulterado)")
        content_hash = "sha256:" + hashlib.sha256(crudo).hexdigest()
        try:
            datos = load_strict_yaml(crudo)
        except (StrictYAMLError, CanonicalizationError) as exc:
            raise RuleSnapshotError(f"{entrada.ruta}: YAML no estricto: {exc}") from exc
        try:
            regla = validar_regla(datos)
        except Exception as exc:
            raise RuleSnapshotError(f"{entrada.ruta}: regla invalida: {exc}") from exc
        if regla.rule_id in vistos:
            raise RuleSnapshotError(f"rule_id duplicado en el snapshot: {regla.rule_id}")
        vistos.add(regla.rule_id)
        selladas.append(ReglaSellada(ruta=f"policy/{entrada.ruta}", modo=entrada.modo,
                                     blob_oid=entrada.oid, content_hash=content_hash,
                                     regla=regla, _testigo=_TESTIGO))

    return TrustedPolicySnapshot(
        commit=pin.commit, policy_tree_oid=pin.policy_tree_oid, reglas=tuple(selladas),
        repositorio=pin.repositorio, procedencia=pin.procedencia, _testigo=_TESTIGO,
    )
