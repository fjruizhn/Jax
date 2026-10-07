"""Snapshot confiable de ``policy/faro`` desde objetos Git (diseno F1.1 §6).

`load_trusted_policy_snapshot(repo, pin)` carga las reglas Faro del commit EXACTO
que fija el pin, nunca del arbol de trabajo:

  1. el commit y el arbol `policy/` esperados existen y coinciden;
  2. lectura por objetos Git con las defensas de ``jax/faro/git_objetos.py``
     (entorno limpio, hooks y fsmonitor apagados, replace objects desactivado);
  3. solo blobs ``100644`` con ruta directa ``policy/faro/<nombre>.yaml``;
     symlinks, submodules, bits de ejecucion, rutas anidadas y nombres no
     canonicos niegan el snapshot completo (los archivos que no son reglas --
     README, ``schemas/*.json`` -- simplemente no son reglas y no se enumeran);
  4. TODOS los blobs se obtienen antes de validar uno, y el hash
     ``rule_content_hash`` se calcula sobre los BYTES CRUDOS antes de parsear;
  5. YAML estricto (sin claves duplicadas, aliases ni merge keys) y shape
     cerrado rule-v1;
  6. una regla invalida o un ``rule_id`` duplicado rechazan el snapshot ENTERO;
  7. el resultado es profundamente inmutable y sellado: no se fabrica por la API
     publica, solo este modulo construye objetos confiables.

El sello impide fabricacion ACCIDENTAL; no es aislamiento frente a codigo
arbitrario dentro del proceso (§6).
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from jax.faro.git_objetos import FuenteInvalida, git, leer_blobs, listar, oid_subarbol
from policy.canonicalization.errors import StrictYAMLError
from policy.canonicalization.strict_yaml import load_strict_yaml

from .errors import RuleSnapshotError
from .schema import ReglaValidada, validar_regla

_SELLO = object()

_PREFIJO = "policy/faro/"
_RE_NOMBRE_CANONICO = re.compile(r"[a-z][a-z0-9-]{0,63}\.yaml\Z")
_RE_OBJETO = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_DOMINIO_SNAPSHOT = "jax-faro-policy-snapshot-v1"


def _hash_dominio(dominio: str, carga: bytes) -> str:
    return "sha256:" + hashlib.sha256(dominio.encode() + b"\0" + carga).hexdigest()


@dataclass(frozen=True)
class TrustedPolicyPin:
    """El pin activo: commit exacto, arbol esperado de ``policy/`` y procedencia.
    Nada de nombres moviles: la raiz es el SHA, no una rama."""
    repositorio: str
    commit: str
    policy_tree_oid: str
    procedencia: str

    def __post_init__(self) -> None:
        for campo in ("repositorio", "procedencia"):
            valor = getattr(self, campo)
            if not isinstance(valor, str) or not valor.strip() \
                    or unicodedata.normalize("NFC", valor) != valor:
                raise RuleSnapshotError(f"pin.{campo}: no vacio y en NFC")
        for campo in ("commit", "policy_tree_oid"):
            valor = getattr(self, campo)
            if not isinstance(valor, str) or not _RE_OBJETO.fullmatch(valor):
                raise RuleSnapshotError(f"pin.{campo}: debe ser un oid hex completo")


@dataclass(frozen=True)
class ReglaSellada:
    """Una regla del snapshot, con sus bytes identificados. Solo la construye
    este modulo (sello privado): una ReglaValidada suelta no es autoridad."""
    ruta: str
    blob_oid: str
    content_hash: str            # sha256 de los BYTES CRUDOS del blob (§5)
    regla: ReglaValidada
    _sello: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._sello is not _SELLO:
            raise RuleSnapshotError("ReglaSellada no se fabrica por la API publica")


@dataclass(frozen=True)
class TrustedPolicySnapshot:
    commit: str
    policy_tree_oid: str
    reglas: tuple[ReglaSellada, ...]          # ordenadas por ruta
    snapshot_hash: str                        # dominio propio, lista ordenada (§6)
    repositorio: str
    procedencia: str                          # procedencia del pin que lo cargo
    _sello: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._sello is not _SELLO:
            raise RuleSnapshotError("TrustedPolicySnapshot no se fabrica por la API publica")


def _clasificar(entradas) -> tuple[list, list]:
    """Separa candidatos a regla de archivos que no son reglas; NIEGA lo raro."""
    candidatos, ignorados = [], []
    for entrada in entradas:
        if not entrada.ruta.startswith(_PREFIJO):
            continue                                    # fuera de policy/faro: nada que ver
        nombre = entrada.ruta[len(_PREFIJO):]
        es_yaml = nombre.endswith((".yaml", ".yml"))
        if entrada.modo != "100644":
            raise RuleSnapshotError(
                f"{entrada.ruta}: modo {entrada.modo} no admitido (symlink/submodulo/ejecutable)")
        if es_yaml and not _RE_NOMBRE_CANONICO.fullmatch(nombre):
            raise RuleSnapshotError(
                f"{entrada.ruta}: nombre o ruta no canonica para regla (nested/no-canonico)")
        if es_yaml:
            candidatos.append(entrada)
        else:
            ignorados.append(entrada)                   # README, schemas/*.json: no son reglas
    return candidatos, ignorados


def load_trusted_policy_snapshot(repo: Path, pin: TrustedPolicyPin) -> TrustedPolicySnapshot:
    if not isinstance(repo, Path) or not repo.is_dir():
        raise RuleSnapshotError(f"el repositorio {repo} no es un directorio")

    try:
        # (1) el commit existe, exacto, como commit; y el arbol policy/ coincide con el pin.
        r = git(repo, "cat-file", "-e", f"{pin.commit}^{{commit}}", aceptar=(0, 1, 128))
        if r.returncode != 0:
            raise RuleSnapshotError(f"el commit {pin.commit} no existe en {repo}")
        arbol_real = oid_subarbol(repo, pin.commit, "policy")
    except FuenteInvalida as exc:
        raise RuleSnapshotError(str(exc)) from exc
    if arbol_real != pin.policy_tree_oid:
        raise RuleSnapshotError("el arbol policy/ no coincide con el pin: no se carga nada")

    # (2-3) enumerar solo policy/faro, por objetos; nada del arbol de trabajo.
    try:
        entradas = listar(repo, pin.commit, "policy/faro")
    except FuenteInvalida as exc:
        raise RuleSnapshotError(str(exc)) from exc
    candidatos, _ = _clasificar(entradas)

    # (4) TODOS los blobs antes de validar uno.
    try:
        blobs = leer_blobs(repo, [e.oid for e in candidatos])
    except FuenteInvalida as exc:
        raise RuleSnapshotError(str(exc)) from exc

    # (5-6) hash de bytes crudos ANTES de parsear; una invalida rompe el snapshot entero.
    selladas: list[ReglaSellada] = []
    ordenadas = sorted(candidatos, key=lambda e: e.ruta)
    vistos: set[str] = set()
    for entrada in ordenadas:
        crudo = blobs[entrada.oid]
        content_hash = "sha256:" + hashlib.sha256(crudo).hexdigest()
        try:
            datos = load_strict_yaml(crudo)
        except StrictYAMLError as exc:
            raise RuleSnapshotError(f"{entrada.ruta}: YAML no estricto: {exc}") from exc
        try:
            regla = validar_regla(datos)
        except Exception as exc:
            raise RuleSnapshotError(f"{entrada.ruta}: regla invalida: {exc}") from exc
        if regla.rule_id in vistos:
            raise RuleSnapshotError(f"rule_id duplicado en el snapshot: {regla.rule_id}")
        vistos.add(regla.rule_id)
        selladas.append(ReglaSellada(ruta=entrada.ruta, blob_oid=entrada.oid,
                                     content_hash=content_hash, regla=regla, _sello=_SELLO))

    # §6: hash del snapshot = dominio propio + lista ordenada de (ruta, modo, oid, hash).
    carga = "".join(
        f"{s.ruta}\0{e.modo}\0{s.blob_oid}\0{s.content_hash}\n"
        for s, e in zip(selladas, ordenadas)
    ).encode()
    return TrustedPolicySnapshot(
        commit=pin.commit, policy_tree_oid=pin.policy_tree_oid, reglas=tuple(selladas),
        snapshot_hash=_hash_dominio(_DOMINIO_SNAPSHOT, carga),
        repositorio=pin.repositorio, procedencia=pin.procedencia, _sello=_SELLO,
    )
