"""Pinned C14N/3 bootstrap root; BUNDLE.sha256 never selects trust."""
from __future__ import annotations
import hashlib, json, os, stat
from dataclasses import dataclass
from pathlib import Path
from .errors import BootstrapIntegrityError

CANONICALIZER_IDENTITY = "JAX-POLICY-C14N/3"
UNICODE_VERSION = "16.0.0"
V3_RESOURCE_NAMES = ("bundle.json", "field-classes.json", "authority-meta-contract.schema.json", "authoritative-policy-manifest.schema.json", "normative-policy-document.schema.json")
PINNED_BOOTSTRAP_BUNDLE_ID = "sha256:23d5523dcf78a5631aba7f44e6d1c74f3394b7c2594bbb50e2d5392cd688fe99"
BOOTSTRAP_DIR = Path(__file__).resolve().parents[1] / "bootstrap" / "v3"

@dataclass(frozen=True)
class VerifiedBootstrapV3:
    bundle_id: str; bundle: dict; field_classes: dict; schemas: dict[str, dict]

def _regular(path: Path) -> bytes:
    try: mode = path.lstat().st_mode
    except FileNotFoundError as exc: raise BootstrapIntegrityError(f"recurso bootstrap ausente: {path.name}") from exc
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode): raise BootstrapIntegrityError(f"recurso bootstrap inseguro: {path.name}")
    return path.read_bytes()

def bootstrap_resource_bytes(directory: Path | str = BOOTSTRAP_DIR) -> dict[str, bytes]:
    root = Path(directory)
    out = {n: _regular(root / n) for n in V3_RESOURCE_NAMES}
    out["BUNDLE.sha256"] = _regular(root / "BUNDLE.sha256")
    return out

def compute_bootstrap_bundle_id(resources: dict[str, bytes]) -> str:
    if set(V3_RESOURCE_NAMES) - resources.keys(): raise BootstrapIntegrityError("bootstrap v3 incompleto")
    h = hashlib.sha256(b"JAX-POLICY-BOOTSTRAP\0" + CANONICALIZER_IDENTITY.encode("ascii"))
    for name in sorted(V3_RESOURCE_NAMES):
        nb, data = name.encode("utf-8"), resources[name]
        h.update(len(nb).to_bytes(8,"big")); h.update(nb); h.update(len(data).to_bytes(8,"big")); h.update(data)
    return "sha256:" + h.hexdigest()

def _obj(raw: bytes, name: str) -> dict:
    def no_duplicate_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"clave JSON duplicada: {key}")
            result[key] = value
        return result
    try: value = json.loads(raw.decode("utf-8"), object_pairs_hook=no_duplicate_keys)
    except Exception as exc: raise BootstrapIntegrityError(f"{name}: JSON inválido") from exc
    if not isinstance(value, dict): raise BootstrapIntegrityError(f"{name}: raíz no objeto")
    return value

def verify_bootstrap_bundle(resources: dict[str, bytes]) -> VerifiedBootstrapV3:
    got = compute_bootstrap_bundle_id(resources)
    if got != PINNED_BOOTSTRAP_BUNDLE_ID: raise BootstrapIntegrityError(f"bootstrap v3 pin mismatch: {got}")
    try: artifact = resources["BUNDLE.sha256"].decode("ascii")
    except Exception as exc: raise BootstrapIntegrityError("BUNDLE.sha256 inválido") from exc
    if artifact != PINNED_BOOTSTRAP_BUNDLE_ID + "\n": raise BootstrapIntegrityError("BUNDLE.sha256 difiere del pin")
    bundle, fields = _obj(resources["bundle.json"], "bundle.json"), _obj(resources["field-classes.json"], "field-classes.json")
    if bundle.get("canonicalizer_version") != CANONICALIZER_IDENTITY or bundle.get("unicode_version") != UNICODE_VERSION: raise BootstrapIntegrityError("identidad bootstrap inválida")
    if bundle.get("authorizes") is not False or bundle.get("purpose") != "CANDIDATE_AUTHORITY_CONTRACT": raise BootstrapIntegrityError("bootstrap autoriza indebidamente")
    if tuple(bundle.get("digest_resources", [])) != V3_RESOURCE_NAMES: raise BootstrapIntegrityError("digest_resources no coincide con constante compilada")
    allowed = bundle.get("allowed_document_classes")
    expected_allowed = ("CONSTITUTIONAL_CORE", "PRODUCT_POLICY", "SUBORDINATE_POLICY")
    if not isinstance(allowed, list) or any(not isinstance(item, str) for item in allowed):
        raise BootstrapIntegrityError("clases permitidas inválidas")
    # This is a declared SET_SCALAR.  Do not turn it into a set before
    # detecting duplicates: doing so would erase an invalid input.
    normalized_allowed = tuple(item for item in allowed)
    if len(normalized_allowed) != len(set(normalized_allowed)):
        raise BootstrapIntegrityError("clases permitidas duplicadas")
    if tuple(sorted(normalized_allowed)) != tuple(sorted(expected_allowed)):
        raise BootstrapIntegrityError("clases permitidas inválidas")
    schemas = {name: _obj(resources[name], name) for name in V3_RESOURCE_NAMES[2:]}
    if not isinstance(fields.get("fields"), dict) or not isinstance(fields.get("array_semantics"), dict): raise BootstrapIntegrityError("field registry inválido")
    verified = VerifiedBootstrapV3(got,bundle,fields,schemas)
    # The registry is verified from schema structure, never from an instance.
    from .schemas_v3 import verify_registry
    verify_registry(verified)
    return verified

def verified_bootstrap_v3() -> VerifiedBootstrapV3: return verify_bootstrap_bundle(bootstrap_resource_bytes())
