"""Raíz de confianza compilada para JAX-POLICY-C14N/2.

``BUNDLE.sha256`` es deliberadamente ajeno a la decisión de confianza. El
valor esperado vive en este módulo y se compara contra los bytes de los cuatro
recursos bootstrap mediante un encuadre no ambiguo.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .errors import BootstrapIntegrityError

CANONICALIZER_VERSION = "2"
CANONICALIZER_IDENTITY = f"JAX-POLICY-C14N/{CANONICALIZER_VERSION}"
PINNED_UNICODE_VERSION = "16.0.0"
PINNED_BOOTSTRAP_BUNDLE_ID = (
    "sha256:a01df4bb87f628202b7ffa5b28e72f50238f5a00dc6bd29b11fbe7241eacf594"
)
BOOTSTRAP_DIR = Path(__file__).resolve().parents[1] / "bootstrap" / "v2"
BOOTSTRAP_DOMAIN = b"JAX-POLICY-BOOTSTRAP\0"
RESOURCE_NAMES = (
    "bundle.json",
    "field-classes.json",
    "legacy-rule-shadow.schema.json",
    "manifest-bootstrap.schema.json",
)


@dataclass(frozen=True)
class VerifiedBootstrap:
    bundle_id: str
    bundle: dict
    field_classes: dict
    legacy_rule_schema: dict
    manifest_schema: dict


def bootstrap_resource_bytes(directory: Path | str = BOOTSTRAP_DIR) -> dict[str, bytes]:
    root = Path(directory)
    resources = {name: (root / name).read_bytes() for name in RESOURCE_NAMES}
    artifact = root / "BUNDLE.sha256"
    if artifact.exists():
        resources["BUNDLE.sha256"] = artifact.read_bytes()
    return resources


def compute_bootstrap_bundle_id(
    resources: Mapping[str, bytes], *, canonicalizer_version: str = CANONICALIZER_VERSION
) -> str:
    missing = set(RESOURCE_NAMES) - resources.keys()
    if missing:
        raise BootstrapIntegrityError(
            f"bootstrap incompleto; faltan recursos: {sorted(missing)}"
        )
    digest = hashlib.sha256(BOOTSTRAP_DOMAIN)
    try:
        digest.update(canonicalizer_version.encode("ascii", errors="strict"))
    except UnicodeEncodeError as exc:
        raise BootstrapIntegrityError("canonicalizer_version no es ASCII") from exc
    for name in sorted(RESOURCE_NAMES):
        name_bytes = name.encode("utf-8")
        data = resources[name]
        digest.update(len(name_bytes).to_bytes(4, "big"))
        digest.update(name_bytes)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return f"sha256:{digest.hexdigest()}"


def _json_resource(resources: Mapping[str, bytes], name: str) -> dict:
    try:
        decoded = resources[name].decode("utf-8", errors="strict")
        value = json.loads(decoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BootstrapIntegrityError(f"{name}: JSON/UTF-8 inválido") from exc
    if not isinstance(value, dict):
        raise BootstrapIntegrityError(f"{name}: la raíz debe ser un objeto")
    return value


def verify_bootstrap_bundle(resources: Mapping[str, bytes]) -> VerifiedBootstrap:
    computed = compute_bootstrap_bundle_id(resources)
    if computed != PINNED_BOOTSTRAP_BUNDLE_ID:
        raise BootstrapIntegrityError(
            "bootstrap no coincide con el digest pinneado por "
            f"{CANONICALIZER_IDENTITY}: esperado {PINNED_BOOTSTRAP_BUNDLE_ID}, "
            f"calculado {computed}"
        )

    if "BUNDLE.sha256" in resources:
        try:
            artifact = resources["BUNDLE.sha256"].decode("ascii").strip()
        except UnicodeDecodeError as exc:
            raise BootstrapIntegrityError("BUNDLE.sha256 no es ASCII") from exc
        if artifact != PINNED_BOOTSTRAP_BUNDLE_ID:
            raise BootstrapIntegrityError(
                "BUNDLE.sha256 difiere del pin; el artefacto no puede autorizarse"
            )

    bundle = _json_resource(resources, "bundle.json")
    if bundle.get("canonicalizer_version") != CANONICALIZER_IDENTITY:
        raise BootstrapIntegrityError("bundle declara otro canonicalizer_version")
    if bundle.get("unicode_normalization") != "NFC":
        raise BootstrapIntegrityError("bundle no pinnea normalización NFC")
    if bundle.get("unicode_version") != PINNED_UNICODE_VERSION:
        raise BootstrapIntegrityError("bundle declara otra versión Unicode")
    if bundle.get("purpose") != "SHADOW_ONLY" or bundle.get("authorizes") is not False:
        raise BootstrapIntegrityError("bundle no está confinado a shadow")
    if bundle.get("resources") != [
        "manifest-bootstrap.schema.json",
        "legacy-rule-shadow.schema.json",
        "field-classes.json",
    ]:
        raise BootstrapIntegrityError("inventario bootstrap inesperado")

    return VerifiedBootstrap(
        bundle_id=computed,
        bundle=bundle,
        field_classes=_json_resource(resources, "field-classes.json"),
        legacy_rule_schema=_json_resource(resources, "legacy-rule-shadow.schema.json"),
        manifest_schema=_json_resource(resources, "manifest-bootstrap.schema.json"),
    )


def verified_bootstrap() -> VerifiedBootstrap:
    return verify_bootstrap_bundle(bootstrap_resource_bytes())
