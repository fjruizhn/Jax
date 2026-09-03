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
from typing import Mapping

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
