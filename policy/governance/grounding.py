"""
policy/governance — Grounding por snapshot inyectado (REFORMAS Fase 2 SP3).

Spec: docs/superpowers/specs/2026-09-02-reformas-fase2-sp3-grounding-design.md

Qué hace: construye, desde el MISMO ValidationContext que usa validator.py,
el conjunto de hechos que el servidor inyecta en el prompt de la Mesa web
(build_snapshot / render), y acredita un claim contra ese snapshot
(accredit) derivando authority y provenance_ref del lado del servidor.

Invariante (spec §3): todo hecho inyectado tiene quién lo re-resuelva. Por
eso el snapshot se genera desde ctx.ops (lo que resuelve
_resolve_capability_available) y no desde una lista curada.

Este módulo es PURO: sin I/O, sin red, testeable en aislamiento. La única
fuente de datos es el ValidationContext que recibe. Si construir el
snapshot falla, LANZA (P10): nunca devuelve vacío por error.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Literal, Mapping

# Sección del snapshot -> predicado que acredita. Crece SOLO cuando un
# predicado gana resolver (spec §3): no agregar entradas acá sin resolver
# en validator._RESOLVERS.
SECTION_PREDICATE: dict[str, str] = {"capabilities": "CAPABILITY_AVAILABLE"}

# Formato de evidence_pointer que el modelo puede citar. Estricto a
# propósito: "/capabilities/-1", "capabilities/10", "/capabilities/abc" y
# "" NO matchean -> PROVENANCE_MISMATCH, nunca indexación negativa ni
# excepción (spec §9.1b).
_POINTER_RE = re.compile(r"^/([a-z_]+)/(0|[1-9][0-9]*)$")


class GroundingBuildError(RuntimeError):
    """El snapshot no pudo construirse. Se lanza, no se degrada (P10)."""


@dataclass(frozen=True)
class SnapshotEntry:
    pointer: str
    predicate: str
    args: dict[str, str]


@dataclass(frozen=True)
class Snapshot:
    entries: tuple[SnapshotEntry, ...]
    canonical_json: str
    sha256: str

    def lookup(self, pointer: str) -> SnapshotEntry | None:
        for e in self.entries:
            if e.pointer == pointer:
                return e
        return None


@dataclass(frozen=True)
class SnapshotError:
    """Marca de 'el snapshot falló al construirse' que viaja al validador
    en lugar de un Snapshot. Persistida como sha256='ERROR' (spec §5.4)."""
    reason: str


@dataclass(frozen=True)
class Accreditation:
    """Resultado de acreditar un claim contra el snapshot. La AUTORIDAD la
    deriva el servidor acá (spec §2.1, P08): el modelo solo señaló una línea.

    outcome:
      ACCREDITED  -> puntero resuelve y args coinciden: authority=OBSERVADO
      NO_POINTER  -> el claim no trae evidence_pointer: authority=INFERIDO
      MISMATCH    -> puntero malformado / fuera de rango / de otro predicado /
                     args que no coinciden: authority=INFERIDO
      UNAVAILABLE -> el snapshot del turno no existe (SnapshotError)
    Qué veredicto sale de cada uno lo decide validator.validate() en el orden
    normativo del spec §4.1 -- acá no se conoce si el predicado tiene resolver.
    """
    authority: Literal["OBSERVADO", "INFERIDO"]
    provenance_ref: str
    evidence_pointer_raw: object | None
    outcome: Literal["ACCREDITED", "NO_POINTER", "MISMATCH", "UNAVAILABLE"]
    detail: str


def normalize_args(args: Mapping[str, object]) -> dict[str, str]:
    """UNA sola normalización para los dos lados (spec §5.3): la usa
    build_snapshot al producir cada entrada y accredit al comparar. Solo
    valores: las claves las juzga ARGS_MISMATCH en validator.validate()."""
    return {k: str(v).strip() for k, v in args.items()}


def _canonical(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def build_snapshot(ctx) -> Snapshot:
    """Snapshot desde ctx.ops + ctx.mutating_capabilities. Orden por name
    (spec §5.2): mismo contenido => mismo hash y mismos punteros, en
    cualquier orden de archivo."""
    try:
        ops = sorted(ctx.ops)
        mutating = ctx.mutating_capabilities
    except Exception as e:
        raise GroundingBuildError(f"ValidationContext inutilizable: {type(e).__name__}: {e}") from e

    caps = [
        normalize_args({"name": name, "mode": "mutating" if name in mutating else "read_only"})
        for name in ops
    ]
    data = {"capabilities": caps}
    canonical = _canonical(data)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    entries = tuple(
        SnapshotEntry(pointer=f"/capabilities/{i}", predicate=SECTION_PREDICATE["capabilities"], args=c)
        for i, c in enumerate(caps)
    )
    return Snapshot(entries=entries, canonical_json=canonical, sha256=digest)


def render(snapshot: Snapshot) -> str:
    """Bloque para el system prompt. NO incluye el hash (spec §5.1): el
    modelo solo cita la línea; provenance_ref lo escribe el servidor."""
    lines = [
        "HECHOS VERIFICADOS — leídos del sistema por el servidor. "
        "Para afirmar uno, poné su evidence_pointer en el claim.",
        "  capabilities:",
    ]
    for e in snapshot.entries:
        lines.append(f"    {e.pointer}: " + ", ".join(f"{k}={v}" for k, v in e.args.items()))
    return "\n".join(lines)


def accredit(raw_claim: Mapping[str, object], grounding: Snapshot | SnapshotError) -> Accreditation:
    """Nunca lanza por contenido del claim: un puntero que mata la
    background task es fail-open (spec §9.1b). Todo lo raro es MISMATCH."""
    raw_ptr = raw_claim.get("evidence_pointer")

    if isinstance(grounding, SnapshotError):
        return Accreditation(
            authority="INFERIDO", provenance_ref="none", evidence_pointer_raw=raw_ptr,
            outcome="UNAVAILABLE",
            detail=f"el snapshot de este turno no se construyó: {grounding.reason}",
        )

    if raw_ptr is None:
        return Accreditation(
            authority="INFERIDO", provenance_ref="none", evidence_pointer_raw=None,
            outcome="NO_POINTER", detail="el claim no trae evidence_pointer.",
        )

    def mismatch(why: str) -> Accreditation:
        return Accreditation(
            authority="INFERIDO", provenance_ref="none", evidence_pointer_raw=raw_ptr,
            outcome="MISMATCH", detail=why,
        )

    if not isinstance(raw_ptr, str):
        return mismatch(f"evidence_pointer no es string (es {type(raw_ptr).__name__}).")
    m = _POINTER_RE.match(raw_ptr)
    if m is None:
        shown = raw_ptr if len(raw_ptr) <= 120 else raw_ptr[:120] + "…"
        return mismatch(f"evidence_pointer malformado: {shown!r}.")
    entry = grounding.lookup(raw_ptr)
    if entry is None:
        return mismatch(f"evidence_pointer {raw_ptr!r} no existe en el snapshot del turno.")
    if entry.predicate != raw_claim.get("predicate"):
        return mismatch(
            f"{raw_ptr} es una entrada de {entry.predicate}, el claim es de {raw_claim.get('predicate')!r}."
        )
    args = raw_claim.get("args")
    if not isinstance(args, Mapping):
        return mismatch("args no es un objeto.")
    given = normalize_args(args)
    if given != entry.args:
        return mismatch(f"los args no coinciden con {raw_ptr}: snapshot={entry.args}, claim={given}.")
    return Accreditation(
        authority="OBSERVADO",
        provenance_ref=f"tool_result:sha256:{grounding.sha256}",
        evidence_pointer_raw=raw_ptr,
        outcome="ACCREDITED",
        detail=f"acreditado contra {raw_ptr}.",
    )
