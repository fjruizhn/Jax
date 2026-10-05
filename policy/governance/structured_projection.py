"""Additive F2-C JSON projection for the authenticated pipeline-list DTO.

F2-C text rendering remains unchanged. The closed schema below treats every
producer field as typed untrusted data and fills only each row's ``status``
slot from a separately rendered, assertable F2-B PIPELINE_STATUS claim.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
import unicodedata
from types import MappingProxyType
from typing import Any, Mapping

from .governed_domain import (
    GOVERNED_ENVELOPE_SCHEMA_VERSIONS,
    GOVERNED_RENDERER_API_VERSION,
    GOVERNED_DOMAIN_SPEC_VERSION,
    GovernedDomainSpecification,
)
from .governed_renderer import GovernedRenderer, RenderContext
from .response import (
    ClaimDisposition,
    ContentBlockKind,
    ContractState,
    EpistemicStatus,
    GovernanceContractError,
    GovernedResponseEnvelope,
    SourceClass,
)


STRUCTURED_PROJECTION_API_VERSION = "f2-c.structured-projection.1"
PIPELINE_LIST_SCHEMA_VERSION = "jax.pipeline-list.json.1"
_MAX_CANONICAL_BYTES = 65_536
_MAX_NESTING_DEPTH = 32
_MAX_NODES = 1_024
_PROJECTION_TOKEN = object()
_PIPELINE_LIST_ACTIVE_TOP_LEVEL_FIELDS = frozenset({"pipelines", "has_more"})
_PIPELINE_LIST_DISCARDED_TOP_LEVEL_FIELDS = frozenset({"pipelines", "has_more", "cursor_siguiente"})
_PIPELINE_ACTIVE_FIELDS = frozenset({
    "pipeline_id", "name", "created_at", "updated_at", "duracion_s", "costo_usd", "causa",
})
_PIPELINE_DISCARDED_FIELDS = _PIPELINE_ACTIVE_FIELDS | frozenset({"descartado_at"})
_CAUSA_FIELDS = frozenset({"tipo", "paso", "detalle"})
_MAX_PIPELINES = 50


class StructuredProjectionError(GovernanceContractError):
    """The DTO cannot be projected under its exact F2-C contract."""


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _validate_causa(value: object) -> None:
    if value is None:
        return
    if not isinstance(value, Mapping) or not value:
        raise StructuredProjectionError("pipeline causa must be null or a closed object")
    if not set(value).issubset(_CAUSA_FIELDS) or "tipo" not in value:
        raise StructuredProjectionError("pipeline causa has an unsupported shape")
    if not isinstance(value["tipo"], str) or not value["tipo"]:
        raise StructuredProjectionError("pipeline causa tipo must be a nonempty string")
    if "paso" in value and (not isinstance(value["paso"], int) or isinstance(value["paso"], bool)):
        raise StructuredProjectionError("pipeline causa paso must be an integer")
    if "detalle" in value and (not isinstance(value["detalle"], str) or not value["detalle"]):
        raise StructuredProjectionError("pipeline causa detalle must be a nonempty string")


def _validate_pipeline_row(row: Mapping[str, object], *, discarded: bool) -> None:
    if "status" in row:
        raise StructuredProjectionError("producer may not supply an accredited status slot")
    fields = set(row)
    expected = _PIPELINE_DISCARDED_FIELDS if discarded else _PIPELINE_ACTIVE_FIELDS
    if fields != expected:
        raise StructuredProjectionError("pipeline-list entry has an unsupported closed shape")
    for field in ("pipeline_id", "name"):
        if not isinstance(row[field], str) or not row[field]:
            raise StructuredProjectionError(f"pipeline {field} must be a nonempty string")
    for field in ("created_at", "updated_at"):
        if not _is_number(row[field]):
            raise StructuredProjectionError(f"pipeline {field} must be a finite number")
    for field in ("duracion_s", "costo_usd"):
        if row[field] is not None and not _is_number(row[field]):
            raise StructuredProjectionError(f"pipeline {field} must be null or a finite number")
    if "descartado_at" in row and row["descartado_at"] is not None and not _is_number(row["descartado_at"]):
        raise StructuredProjectionError("pipeline descartado_at must be null or a finite number")
    _validate_causa(row["causa"])


@dataclass(frozen=True)
class StructuredFieldOrigin:
    origin: str
    source_ref: str

    def __post_init__(self) -> None:
        if self.origin not in {"USER", "ASSISTANT/MODEL", "TOOL", "SYSTEM", "AGENT"}:
            raise StructuredProjectionError("structured field origin is not registered")
        if not isinstance(self.source_ref, str) or not self.source_ref:
            raise StructuredProjectionError("structured field origin requires provenance")


_FIELD_ORIGIN = StructuredFieldOrigin("SYSTEM", "jax-platform:GET:/api/pipelines")


def _freeze_json(value: Any, *, path: str = "dto", depth: int = 0,
                 nodes: list[int] | None = None) -> Any:
    if nodes is None:
        nodes = [0]
    nodes[0] += 1
    if nodes[0] > _MAX_NODES or depth > _MAX_NESTING_DEPTH:
        raise StructuredProjectionError("structured DTO exceeds decoder bounds")
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise StructuredProjectionError("structured DTO numbers must be finite")
        return value
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise StructuredProjectionError(f"{path} keys must be strings")
            normalized = unicodedata.normalize("NFC", key)
            if normalized in result:
                raise StructuredProjectionError(f"{path} has duplicate keys after NFC normalization")
            result[normalized] = _freeze_json(item, path=f"{path}.{normalized}",
                depth=depth + 1, nodes=nodes)
        return MappingProxyType(dict(sorted(result.items())))
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_json(item, path=f"{path}[{index}]", depth=depth + 1, nodes=nodes)
            for index, item in enumerate(value))
    raise StructuredProjectionError(f"{path} is not a canonical JSON value")


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


_TRUSTED_FIELD_NAMES = frozenset({
    "authority", "authority_refs", "authority_origin", "source", "source_class",
    "verification", "verified", "epistemic_status", "trust", "trusted_label",
    "current_observation", "citation", "citations", "memory_label", "claim_id",
    "scope_digest", "governance_receipt", "current",
})


def _contains_trusted_metadata(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(key.lower() in _TRUSTED_FIELD_NAMES or _contains_trusted_metadata(item)
            for key, item in value.items() if isinstance(key, str))
    if isinstance(value, (list, tuple)):
        return any(_contains_trusted_metadata(item) for item in value)
    return False


def _unique_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise StructuredProjectionError("untrusted DTO contains duplicate JSON keys")
        result[key] = value
    return result


def _canonical_bytes(value: Any) -> bytes:
    try:
        encoded = json.dumps(_plain(value), sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise StructuredProjectionError("structured DTO cannot be encoded canonically") from exc
    if len(encoded) > _MAX_CANONICAL_BYTES:
        raise StructuredProjectionError("structured DTO exceeds canonical byte bound")
    return encoded


def _origin_paths(value: Any, path: str = "") -> dict[str, StructuredFieldOrigin]:
    if isinstance(value, Mapping):
        if not value:
            return {path or "/": _FIELD_ORIGIN}
        result = {}
        for key, item in value.items():
            escaped = key.replace("~", "~0").replace("/", "~1")
            result.update(_origin_paths(item, f"{path}/{escaped}"))
        return result
    if isinstance(value, tuple):
        if not value:
            return {path or "/": _FIELD_ORIGIN}
        result = {}
        for index, item in enumerate(value):
            result.update(_origin_paths(item, f"{path}/{index}"))
        return result
    return {path or "/": _FIELD_ORIGIN}


@dataclass(frozen=True, init=False)
class GovernedStructuredProjection:
    schema_version: str
    api_version: str
    payload: Mapping[str, Any]
    canonical_bytes: bytes
    bytes_digest: str
    envelope_digest: str
    claim_ids: tuple[str, ...]
    field_origins: Mapping[str, StructuredFieldOrigin]
    field_claim_ids: Mapping[str, str]
    _integrity_digest: str

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise StructuredProjectionError("projection must be minted by GovernedStructuredRenderer")

    @classmethod
    def _mint(cls, token: object, **values: object) -> "GovernedStructuredProjection":
        if token is not _PROJECTION_TOKEN:
            raise StructuredProjectionError("projection minting is server-owned")
        instance = object.__new__(cls)
        for key, value in values.items():
            object.__setattr__(instance, key, value)
        return instance

    def _compute_integrity_digest(self) -> str:
        material = b"\x00".join((
            self.schema_version.encode("utf-8"),
            self.api_version.encode("utf-8"),
            self.envelope_digest.encode("ascii"),
            self.bytes_digest.encode("ascii"),
            "\n".join(self.claim_ids).encode("utf-8"),
            "\n".join(f"{path}:{source.origin}:{source.source_ref}"
                for path, source in self.field_origins.items()).encode("utf-8"),
            "\n".join(f"{path}:{claim_id}"
                for path, claim_id in self.field_claim_ids.items()).encode("utf-8"),
        ))
        return "sha256:" + hashlib.sha256(material).hexdigest()


class GovernedStructuredRenderer:
    """Exact-version F2-C JSON renderer for registered structured schemas."""

    unavailable_payload = MappingProxyType({"code": "governed_output_unavailable"})

    def render_json(self, envelope: GovernedResponseEnvelope, context: RenderContext,
                    *, schema_version: str = PIPELINE_LIST_SCHEMA_VERSION) -> GovernedStructuredProjection:
        if schema_version != PIPELINE_LIST_SCHEMA_VERSION:
            raise StructuredProjectionError("unsupported structured schema version")
        if not isinstance(envelope, GovernedResponseEnvelope) or not isinstance(context, RenderContext):
            raise StructuredProjectionError("structured renderer requires sealed envelope and server context")
        if envelope.compute_digest() != envelope.envelope_digest:
            raise StructuredProjectionError("sealed envelope digest mismatch")
        if envelope.candidate.schema_version not in GOVERNED_ENVELOPE_SCHEMA_VERSIONS:
            raise StructuredProjectionError("unsupported sealed envelope version")
        if (GOVERNED_RENDERER_API_VERSION != "f2-c.renderer.3"
                or GOVERNED_DOMAIN_SPEC_VERSION != "f2-c.domain.6"
                or context.renderer_api_version != GOVERNED_RENDERER_API_VERSION
                or context.domain_registry.specification.version != GOVERNED_DOMAIN_SPEC_VERSION):
            raise StructuredProjectionError("unsupported F2-C renderer/domain compatibility tuple")
        if envelope.contract_state is not ContractState.VALID:
            raise StructuredProjectionError("structured projection requires a valid governed envelope")

        tool_blocks = [block for block in envelope.content_blocks
            if block.kind is ContentBlockKind.TOOL_DATA]
        if len(tool_blocks) != 1 or not isinstance(tool_blocks[0].payload, str):
            raise StructuredProjectionError("pipeline-list projection requires one encoded TOOL_DATA block")
        raw = tool_blocks[0].payload
        try:
            raw_size = len(raw.encode("utf-8"))
        except UnicodeEncodeError as exc:
            raise StructuredProjectionError("untrusted DTO contains invalid Unicode") from exc
        if raw_size > _MAX_CANONICAL_BYTES:
            raise StructuredProjectionError("untrusted DTO exceeds decoder byte bound")
        try:
            untrusted = json.loads(raw, object_pairs_hook=_unique_object_pairs)
        except StructuredProjectionError:
            raise
        except (ValueError, RecursionError) as exc:
            raise StructuredProjectionError("untrusted DTO is not valid JSON") from exc
        frozen_untrusted = _freeze_json(untrusted)
        if _contains_trusted_metadata(untrusted):
            raise StructuredProjectionError("TOOL_DATA may not introduce trusted governance metadata")
        if GovernedDomainSpecification().structured_runtime_status_predicate(untrusted) is not None:
            raise StructuredProjectionError("untrusted DTO contains an unaccredited structured runtime status")
        if not isinstance(frozen_untrusted, Mapping):
            raise StructuredProjectionError("pipeline-list DTO must be a JSON object")
        rows = frozen_untrusted.get("pipelines")
        if not isinstance(rows, tuple):
            raise StructuredProjectionError("pipeline-list DTO requires a pipelines array")
        top_level_fields = set(frozen_untrusted)
        discarded_page = "cursor_siguiente" in frozen_untrusted
        expected_top_level = (_PIPELINE_LIST_DISCARDED_TOP_LEVEL_FIELDS if discarded_page
            else _PIPELINE_LIST_ACTIVE_TOP_LEVEL_FIELDS)
        if top_level_fields != expected_top_level:
            raise StructuredProjectionError("pipeline-list DTO has an unsupported top-level shape")
        if not isinstance(frozen_untrusted.get("has_more"), bool):
            raise StructuredProjectionError("pipeline-list has_more must be boolean")
        if discarded_page:
            cursor = frozen_untrusted["cursor_siguiente"]
            if frozen_untrusted["has_more"]:
                if not isinstance(cursor, str) or not cursor:
                    raise StructuredProjectionError("discarded pipeline-list with more rows requires a cursor")
            elif cursor is not None:
                raise StructuredProjectionError("discarded pipeline-list without more rows must not carry a cursor")
        if len(rows) > _MAX_PIPELINES:
            raise StructuredProjectionError("pipeline-list exceeds the closed page bound")

        claims = {claim.claim_id: claim for claim in envelope.claims}
        visible_claim_ids = tuple(claim_id for block in envelope.content_blocks
            if block.kind is ContentBlockKind.CLAIM_REF_BLOCK for claim_id in block.claim_refs)
        references = {reference.ref_id: reference for reference in envelope.references}
        seen_pipeline_ids: set[str] = set()
        selected_claim_ids: list[str] = []
        projected_rows = []
        validation_time = context.now()
        if not isinstance(validation_time, datetime) or validation_time.tzinfo is None:
            raise StructuredProjectionError("server validation clock must be timezone-aware")

        for row in rows:
            if not isinstance(row, Mapping):
                raise StructuredProjectionError("pipeline-list entries must be objects")
            _validate_pipeline_row(row, discarded=discarded_page)
            pipeline_id = row.get("pipeline_id")
            if pipeline_id in seen_pipeline_ids:
                raise StructuredProjectionError("pipeline-list contains duplicate pipeline identities")
            seen_pipeline_ids.add(pipeline_id)
            matches = [claim for claim in envelope.claims
                if claim.predicate == "PIPELINE_STATUS"
                and dict(claim.typed_arguments).get("pipeline_id") == pipeline_id]
            if len(matches) != 1:
                raise StructuredProjectionError("each pipeline status slot requires exactly one matching claim")
            claim = matches[0]
            arguments = dict(claim.typed_arguments)
            if set(arguments) != {"pipeline_id", "status"} or not isinstance(arguments["status"], str):
                raise StructuredProjectionError("PIPELINE_STATUS claim arguments do not match the closed slot")
            if claim.claim_id not in visible_claim_ids:
                raise StructuredProjectionError("status slot claim is not visible in the sealed F2-A envelope")
            if (claim.disposition is not ClaimDisposition.ASSERTABLE
                    or claim.epistemic_status is not EpistemicStatus.CURRENT_OBSERVATION
                    or claim.source_class is not SourceClass.CURRENT_SOURCE
                    or claim.claim_scope.scope_digest != envelope.response_scope.scope_digest):
                raise StructuredProjectionError("status slot claim is not an accredited current claim")
            try:
                GovernedRenderer()._validate_claim(claim, envelope.response_scope, context,
                    validation_time, references)
            except (GovernanceContractError, KeyError, TypeError, ValueError) as exc:
                raise StructuredProjectionError("F2-B claim validation failed closed") from exc
            output_row = dict(row)
            output_row["status"] = arguments["status"]
            projected_rows.append(output_row)
            selected_claim_ids.append(claim.claim_id)

        pipeline_claims = [claim for claim in envelope.claims if claim.predicate == "PIPELINE_STATUS"]
        if len(pipeline_claims) != len(selected_claim_ids):
            raise StructuredProjectionError("unbound PIPELINE_STATUS claims are not permitted")
        projected = dict(frozen_untrusted)
        projected["pipelines"] = tuple(projected_rows)
        frozen_payload = _freeze_json(projected)
        canonical_bytes = _canonical_bytes(frozen_payload)
        bytes_digest = "sha256:" + hashlib.sha256(canonical_bytes).hexdigest()
        origins = _origin_paths(frozen_payload)
        field_claim_ids = {}
        for index, claim_id in enumerate(selected_claim_ids):
            claim = claims[claim_id]
            status_path = f"/pipelines/{index}/status"
            origins[status_path] = StructuredFieldOrigin(
                "SYSTEM", context.receipts[claim.resolution_receipt_ref].provenance_ref)
            field_claim_ids[status_path] = claim_id
        origin_map = MappingProxyType(origins)
        projection = GovernedStructuredProjection._mint(_PROJECTION_TOKEN,
            schema_version=PIPELINE_LIST_SCHEMA_VERSION,
            api_version=STRUCTURED_PROJECTION_API_VERSION,
            payload=frozen_payload,
            canonical_bytes=canonical_bytes,
            bytes_digest=bytes_digest,
            envelope_digest=envelope.envelope_digest,
            claim_ids=tuple(selected_claim_ids),
            field_origins=origin_map,
            field_claim_ids=MappingProxyType(field_claim_ids),
            _integrity_digest="",
        )
        object.__setattr__(projection, "_integrity_digest", projection._compute_integrity_digest())
        return projection

    def revalidate_claims_for_transport(self, envelope: GovernedResponseEnvelope,
            context: RenderContext, projection: GovernedStructuredProjection,
            now: datetime) -> None:
        """Recheck F2-B slots without rebuilding the already-rendered DTO."""
        if not isinstance(projection, GovernedStructuredProjection):
            raise StructuredProjectionError("transport revalidation requires F2-C projection")
        if not isinstance(now, datetime) or now.tzinfo is None:
            raise StructuredProjectionError("transport validation time must be timezone-aware")
        if (projection.envelope_digest != envelope.envelope_digest
                or projection._integrity_digest != projection._compute_integrity_digest()
                or projection.bytes_digest != "sha256:" + hashlib.sha256(projection.canonical_bytes).hexdigest()):
            raise StructuredProjectionError("structured projection integrity check failed")
        claims = {claim.claim_id: claim for claim in envelope.claims}
        references = {reference.ref_id: reference for reference in envelope.references}
        for claim_id in projection.claim_ids:
            claim = claims.get(claim_id)
            if claim is None:
                raise StructuredProjectionError("projected claim disappeared from envelope")
            try:
                GovernedRenderer()._validate_claim(claim, envelope.response_scope, context, now, references)
            except (GovernanceContractError, KeyError, TypeError, ValueError) as exc:
                raise StructuredProjectionError("F2-B transport revalidation failed closed") from exc
