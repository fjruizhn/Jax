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
import os
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


def _normalize_path(path: str, repo_root: Path) -> Path:
    p = Path(path)
    p = p if p.is_absolute() else repo_root / p
    return Path(os.path.normpath(str(p)))  # lexical only: no stat/readlink, no filesystem I/O


def _path_allowed(path: str, allowlist: frozenset[str], repo_root: Path) -> bool:
    """Solo aritmética de paths — CERO llamadas a exists()/stat()/read_bytes()."""
    candidate = _normalize_path(path, repo_root)
    for entry in allowlist:
        is_dir_entry = entry.endswith("/")
        entry_path = _normalize_path(
            entry.rstrip("/") if is_dir_entry else entry, repo_root
        )
        if is_dir_entry:
            if candidate == entry_path or entry_path in candidate.parents:
                return True
            continue
        elif candidate == entry_path:
            return True
    return False


def _resolve_file_exists(claim: "claims.Claim", ctx: ValidationContext) -> Verdict:
    path = claim.args["path"]
    expected_hash = claim.args["hash"]

    if not _path_allowed(path, ctx.config_paths_allowlist, ctx.repo_root):
        return Verdict(
            status="PATH_NOT_ALLOWED",
            predicate="FILE_EXISTS",
            detail="Path fuera de la allowlist de config_paths.",
        )

    candidate = _normalize_path(path, ctx.repo_root)
    if not candidate.exists():
        return Verdict(
            status="FACT_MISMATCH", predicate="FILE_EXISTS", detail=f"'{path}' no existe."
        )

    actual_hash = hashlib.sha256(candidate.read_bytes()).hexdigest()
    if actual_hash != expected_hash:
        return Verdict(
            status="FACT_MISMATCH",
            predicate="FILE_EXISTS",
            detail=(
                f"'{path}' existe pero su hash no coincide (esperado "
                f"{expected_hash}, real {actual_hash})."
            ),
        )
    return Verdict(
        status="VALID", predicate="FILE_EXISTS", detail=f"'{path}' existe con hash verificado."
    )


def _resolve_capability_available(
    claim: "claims.Claim", ctx: ValidationContext
) -> Verdict:
    name = claim.args["name"]
    mode = claim.args["mode"]
    in_ops = name in ctx.ops
    in_catalog = ctx.catalog.get_capability(name) is not None

    if in_ops and in_catalog:
        return Verdict(
            status="SOURCE_CONFLICT",
            predicate="CAPABILITY_AVAILABLE",
            detail=(
                f"'{name}' presente en ops y en capabilities — dos "
                "fuentes de verdad para el mismo nombre (ver P04, "
                "tensión documentada en el corpus)."
            ),
        )
    if in_ops:
        derived_mode = "mutating" if name in ctx.mutating_capabilities else "read_only"
        if mode != derived_mode:
            return Verdict(
                status="FACT_MISMATCH",
                predicate="CAPABILITY_AVAILABLE",
                detail=(
                    f"'{name}' tiene mode real '{derived_mode}', el "
                    f"claim afirma '{mode}'."
                ),
            )
        return Verdict(
            status="VALID",
            predicate="CAPABILITY_AVAILABLE",
            detail=f"'{name}' verificado en ops, mode='{derived_mode}'.",
        )
    if in_catalog:
        return Verdict(
            status="VALID",
            predicate="CAPABILITY_AVAILABLE",
            detail=(
                f"'{name}' verificado en catálogo de capabilities (mode "
                "no verificable ahí, aceptado sin contradicción)."
            ),
        )
    return Verdict(
        status="FACT_MISMATCH",
        predicate="CAPABILITY_AVAILABLE",
        detail=f"'{name}' no está en ops ni en capabilities.",
    )


_RESOLVERS: dict[str, Callable[["claims.Claim", ValidationContext], Verdict]] = {
    "FILE_EXISTS": _resolve_file_exists,
    "CAPABILITY_AVAILABLE": _resolve_capability_available,
}
_UNIMPLEMENTED_REASONS: dict[str, str] = {
    "ENGINE_STATUS": (
        "ENGINE_STATUS: sin fuente de verdad en el dominio de jax. La "
        "tabla 'model' de jax-platform tiene semántica distinta "
        "(disponibilidad del provider, no salud del motor). Ver "
        "REFORMAS-v3 §3.1.3 y el spec de gobernanza, sección 'Por qué "
        "solo dos resolvers reales'."
    ),
    "FACET_EXISTS": (
        "FACET_EXISTS: sin fuente ni consumidor identificado esta "
        "ronda. Ver spec de gobernanza, 'Fuera de alcance, "
        "explícitamente'."
    ),
    "CONFIG_VALUE": (
        "CONFIG_VALUE: sin fuente ni consumidor identificado esta "
        "ronda. Ver spec de gobernanza, 'Fuera de alcance, "
        "explícitamente'."
    ),
    "AUDIT_EVENT_EXISTS": (
        "AUDIT_EVENT_EXISTS: sin fuente ni consumidor identificado esta "
        "ronda. Ver spec de gobernanza, 'Fuera de alcance, "
        "explícitamente'."
    ),
    "JOB_STATUS": (
        "JOB_STATUS: sin fuente ni consumidor identificado esta ronda. "
        "Ver spec de gobernanza, 'Fuera de alcance, explícitamente'."
    ),
    "MEMORY_ENTRY_EXISTS": (
        "MEMORY_ENTRY_EXISTS: sin fuente ni consumidor identificado "
        "esta ronda. Ver spec de gobernanza, 'Fuera de alcance, "
        "explícitamente'."
    ),
}


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
