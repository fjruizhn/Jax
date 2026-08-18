"""
policy/governance — Validación semántica de claims contra fuentes reales.

Capa 2 de dos (la 1 es claims.py, estructural). Recibe un Claim ya válido
estructuralmente y lo despacha contra las fuentes de verdad reales del
sistema (config.toml de las_manos, motor_registry, filesystem).

Distinto de loaders.py: ese lee config ESTÁTICA versionada por commit (si
falla, el subsistema no arranca). Este lee ESTADO VIVO que cambia sin
commits — si un resolver individual falla, es un rechazo de ESE claim,
no una falla de arranque.

RESOLVER_NOT_IMPLEMENTED es un veredicto, no una excepción sin capturar:
P07 ("no existe bypass en producción") aplicado al tipo de retorno — no
hay forma de expresar "falló pero seguí" porque Verdict no lo permite.

Este módulo SÍ hace I/O (config.toml, filesystem) — a diferencia de
claims.py, vocab_sweep.py y renderer.py, que son puros.
"""
from __future__ import annotations

import hashlib
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from las_manos.envelope import MUTATING_CAPABILITIES  # noqa: E402
from las_manos.motor_registry.catalog import MotorCatalog  # noqa: E402

import claims  # noqa: E402


class Verdict(BaseModel):
    status: Literal[
        "VALID",
        "UNKNOWN_PREDICATE",
        "ARGS_MISMATCH",
        "RESOLVER_NOT_IMPLEMENTED",
        "FACT_MISMATCH",
        "AUTHORITY_INVALID",
        "SOURCE_CONFLICT",
        "PATH_NOT_ALLOWED",
    ]
    predicate: str
    detail: str


@dataclass(frozen=True)
class ValidationContext:
    ops: frozenset[str]
    mutating_capabilities: frozenset[str]
    catalog: MotorCatalog
    config_paths_allowlist: frozenset[str]
    repo_root: Path


def load_validation_context(
    repo_root: Path, config_paths_allowlist: frozenset[str]
) -> ValidationContext:
    config_path = repo_root / "las_manos" / "config.toml"
    with config_path.open("rb") as f:
        config = tomllib.load(f)
    return ValidationContext(
        ops=frozenset(config.get("ops", {}).keys()),
        mutating_capabilities=frozenset(MUTATING_CAPABILITIES),
        catalog=MotorCatalog(config),
        config_paths_allowlist=config_paths_allowlist,
        repo_root=repo_root,
    )


_RESOLVERS: dict[str, Callable[["claims.Claim", ValidationContext], Verdict]] = {}
_UNIMPLEMENTED_REASONS: dict[str, str] = {}


def validate(
    claim: "claims.Claim", predicates: dict, ctx: ValidationContext
) -> Verdict:
    spec = predicates.get(claim.predicate)
    if spec is None:
        return Verdict(
            status="UNKNOWN_PREDICATE",
            predicate=claim.predicate,
            detail=f"'{claim.predicate}' no está en predicates.yaml.",
        )

    if set(claim.args.keys()) != set(spec.args):
        return Verdict(
            status="ARGS_MISMATCH",
            predicate=claim.predicate,
            detail=(
                f"Args esperados {sorted(spec.args)}, "
                f"recibidos {sorted(claim.args.keys())}."
            ),
        )

    if claim.authority == "INFERIDO":
        return Verdict(
            status="AUTHORITY_INVALID",
            predicate=claim.predicate,
            detail="authority=INFERIDO prohibido en canal claim (§3.1.4).",
        )

    resolver = _RESOLVERS.get(claim.predicate)
    if resolver is not None:
        return resolver(claim, ctx)

    reason = _UNIMPLEMENTED_REASONS.get(
        claim.predicate, f"{claim.predicate}: resolver no implementado."
    )
    return Verdict(
        status="RESOLVER_NOT_IMPLEMENTED", predicate=claim.predicate, detail=reason
    )
