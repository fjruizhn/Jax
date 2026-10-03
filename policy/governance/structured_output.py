"""F2-E structured governed-output projection.

This is intentionally a separate, exact compatibility tuple from F2-C text.
It never parses a text rendering back into JSON: a server-owned layout renders
one sealed TOOL_DATA root directly to canonical JSON bytes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Mapping

from .governed_domain import GOVERNED_DOMAIN_SPEC_VERSION, GOVERNED_ENVELOPE_SCHEMA_VERSIONS
from .governed_renderer import GovernedDomainRegistry, GovernedRenderError, GovernedRenderer, RenderContext
from .response import (
    ClaimDisposition, ContentBlockKind, ContractState, EpistemicStatus,
    GovernedResponseEnvelope, GovernanceContractError, _freeze, _plain, _text,
)


STRUCTURED_RENDERER_API_VERSION = "f2-c.structured-renderer.2"
STRUCTURED_OUTPUT_SCHEMA_VERSION = "f2-c.structured-output.2"
STRUCTURED_OUTPUT_SCHEMA_VERSIONS = frozenset({STRUCTURED_OUTPUT_SCHEMA_VERSION})
STRUCTURED_FLOAT64_CARRIER_TAG = "f2-e.float64"
STRUCTURED_COMPATIBILITY_TUPLE = (
    STRUCTURED_RENDERER_API_VERSION,
    GOVERNED_DOMAIN_SPEC_VERSION,
    STRUCTURED_OUTPUT_SCHEMA_VERSIONS,
)


class StructuredOutputError(GovernanceContractError):
    """A structured candidate cannot become an external projection."""


def validate_structured_compatibility(*, renderer_api_version: str,
                                      domain_spec_version: str,
                                      schema_versions: frozenset[str]) -> None:
    """Reject every old, future, partial, or wildcard structured tuple."""
    if (renderer_api_version, domain_spec_version, schema_versions) != STRUCTURED_COMPATIBILITY_TUPLE:
        raise StructuredOutputError("unsupported structured F2-C compatibility tuple")


class OutputOrigin(str, Enum):
    USER = "USER"
    ASSISTANT_MODEL = "ASSISTANT/MODEL"
    TOOL = "TOOL"
    SYSTEM = "SYSTEM"
    AGENT = "AGENT"


class SlotProjectionMode(str, Enum):
    EXACT = "EXACT"
    PRESENTATION_MAP = "PRESENTATION_MAP"


class StructuredJSONNumberType(str, Enum):
    FLOAT64 = "FLOAT64"


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(_plain(value), ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise StructuredOutputError("structured output must be canonical JSON") from exc


def _structured_freeze(value: object, label: str) -> object:
    """Freeze F2-E output after its explicitly bound float carriers unpack."""
    if isinstance(value, float):
        if not math.isfinite(value):
            raise StructuredOutputError(f"{label} has non-finite float")
        return value
    if isinstance(value, Mapping):
        return MappingProxyType({
            _text(key, f"{label} key"): _structured_freeze(item, label)
            for key, item in sorted(value.items())
        })
    if isinstance(value, (tuple, list)):
        return tuple(_structured_freeze(item, label) for item in value)
    # Preserve F2-A's complete scalar/type contract for every non-float leaf.
    return _freeze(value, label)


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _pointer(pointer: str) -> tuple[str, ...]:
    if not isinstance(pointer, str):
        raise StructuredOutputError("JSON pointer must be string")
    if pointer == "":
        return ()
    if not pointer.startswith("/"):
        raise StructuredOutputError("JSON pointer must start with '/'")
    result: list[str] = []
    for part in pointer[1:].split("/"):
        if "~" in part:
            part = part.replace("~1", "/").replace("~0", "~")
            if "~" in part:
                raise StructuredOutputError("JSON pointer has invalid escape")
        result.append(part)
    return tuple(result)


def _get(value: object, pointer: str) -> object:
    current = value
    for part in _pointer(pointer):
        if isinstance(current, Mapping):
            if part not in current:
                raise StructuredOutputError("structured layout pointer is absent")
            current = current[part]
        elif isinstance(current, (tuple, list)):
            if not part.isdecimal() or str(int(part)) != part or int(part) >= len(current):
                raise StructuredOutputError("structured layout array pointer is absent")
            current = current[int(part)]
        else:
            raise StructuredOutputError("structured layout pointer crosses scalar")
    return current


def _set(value: object, pointer: str, replacement: object) -> object:
    parts = _pointer(pointer)
    if not parts:
        return replacement
    if isinstance(value, Mapping):
        current: object = dict(value)
    elif isinstance(value, (tuple, list)):
        current = list(value)
    else:
        raise StructuredOutputError("structured layout root must be object or array")
    node = current
    for part in parts[:-1]:
        if isinstance(node, dict):
            if part not in node:
                raise StructuredOutputError("structured layout pointer is absent")
            child = node[part]
            if isinstance(child, Mapping):
                child = dict(child)
            elif isinstance(child, (tuple, list)):
                child = list(child)
            else:
                raise StructuredOutputError("structured layout pointer crosses scalar")
            node[part] = child
            node = child
        elif isinstance(node, list):
            if not part.isdecimal() or str(int(part)) != part or int(part) >= len(node):
                raise StructuredOutputError("structured layout array pointer is absent")
            child = node[int(part)]
            if isinstance(child, Mapping):
                child = dict(child)
            elif isinstance(child, (tuple, list)):
                child = list(child)
            else:
                raise StructuredOutputError("structured layout pointer crosses scalar")
            node[int(part)] = child
            node = child
        else:
            raise StructuredOutputError("structured layout pointer crosses scalar")
    final = parts[-1]
    if isinstance(node, dict):
        if final not in node:
            raise StructuredOutputError("structured layout pointer is absent")
        node[final] = replacement
    elif isinstance(node, list):
        if not final.isdecimal() or str(int(final)) != final or int(final) >= len(node):
            raise StructuredOutputError("structured layout array pointer is absent")
        node[int(final)] = replacement
    else:
        raise StructuredOutputError("structured layout pointer crosses scalar")
    return _structured_freeze(current, "structured projection")


def _is_prefix(left: str, right: str) -> bool:
    a, b = _pointer(left), _pointer(right)
    return len(a) <= len(b) and a == b[:len(a)]


def _pattern_parts(pointer: str) -> tuple[str, ...]:
    return _pointer(pointer)


def _pattern_matches(pattern: str, leaf: str) -> bool:
    parts, target = _pattern_parts(pattern), _pointer(leaf)
    return len(parts) <= len(target) and all(a == "*" or a == b for a, b in zip(parts, target))


def _pattern_specificity(pattern: str) -> tuple[int, int]:
    parts = _pattern_parts(pattern)
    return (sum(part != "*" for part in parts), len(parts))


def _leaves(value: object, prefix: str = "") -> tuple[str, ...]:
    if isinstance(value, Mapping):
        if not value:
            return (prefix,)
        result: list[str] = []
        for key, child in value.items():
            escaped = key.replace("~", "~0").replace("/", "~1")
            result.extend(_leaves(child, prefix + "/" + escaped))
        return tuple(result)
    if isinstance(value, tuple):
        if not value:
            return (prefix,)
        result: list[str] = []
        for index, child in enumerate(value):
            result.extend(_leaves(child, prefix + "/" + str(index)))
        return tuple(result)
    return (prefix,)


@dataclass(frozen=True)
class StructuredClaimSlot:
    """A layout-owned binding from a F2-B claim into an exact JSON subtree."""
    predicate: str
    object_pointer: str
    claim_id: str
    argument_pointers: Mapping[str, str]
    projection_mode: SlotProjectionMode = SlotProjectionMode.EXACT
    presentation_argument: str | None = None
    presentation_map_id: str | None = None
    server_constants: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "predicate", _text(self.predicate, "slot predicate"))
        object.__setattr__(self, "claim_id", _text(self.claim_id, "slot claim id"))
        _pointer(self.object_pointer)
        if not isinstance(self.argument_pointers, Mapping) or not self.argument_pointers:
            raise StructuredOutputError("structured claim slot needs argument pointers")
        pointers: dict[str, str] = {}
        for name, pointer in self.argument_pointers.items():
            name = _text(name, "slot argument name")
            _pointer(pointer)
            if not _is_prefix(self.object_pointer, pointer):
                raise StructuredOutputError("slot arguments must be below its object pointer")
            pointers[name] = pointer
        if len(pointers) != len(set(pointers.values())):
            raise StructuredOutputError("structured claim slot reuses an argument pointer")
        if not isinstance(self.server_constants, Mapping):
            raise StructuredOutputError("structured slot constants must be mapping")
        constants = {_text(name, "slot constant name"): _freeze(value, "slot constant")
            for name, value in self.server_constants.items()}
        if set(constants) & set(pointers):
            raise StructuredOutputError("structured slot constant duplicates pointer argument")
        object.__setattr__(self, "server_constants", MappingProxyType(dict(sorted(constants.items()))))
        if not isinstance(self.projection_mode, SlotProjectionMode):
            raise StructuredOutputError("structured slot projection mode must be typed")
        if self.projection_mode is SlotProjectionMode.EXACT:
            if self.presentation_argument is not None or self.presentation_map_id is not None:
                raise StructuredOutputError("exact structured slot cannot select presentation map")
        else:
            if not isinstance(self.presentation_argument, str) or self.presentation_argument not in pointers:
                raise StructuredOutputError("presentation map needs one mapped claim argument")
            object.__setattr__(self, "presentation_argument", _text(self.presentation_argument, "presentation argument"))
            object.__setattr__(self, "presentation_map_id", _text(self.presentation_map_id, "presentation map id"))
        object.__setattr__(self, "argument_pointers", MappingProxyType(dict(sorted(pointers.items()))))

    def semantic_projection(self) -> Mapping[str, object]:
        return {"predicate": self.predicate, "object_pointer": self.object_pointer,
            "claim_id": self.claim_id, "argument_pointers": dict(self.argument_pointers),
            "projection_mode": self.projection_mode.value,
            "presentation_argument": self.presentation_argument,
            "presentation_map_id": self.presentation_map_id,
            "server_constants": _plain(self.server_constants)}


@dataclass(frozen=True)
class OriginBinding:
    pointer: str
    origin: OutputOrigin
    source_ref: str | None = None

    def __post_init__(self) -> None:
        parts = _pointer(self.pointer)
        if any(part == "*" for part in parts):
            # A wildcard is one decoded JSON-pointer component, never a glob.
            if any("*" in part and part != "*" for part in parts):
                raise StructuredOutputError("origin wildcard must occupy one pointer component")
        if not isinstance(self.origin, OutputOrigin):
            raise StructuredOutputError("origin binding needs OutputOrigin")
        if self.source_ref is not None:
            object.__setattr__(self, "source_ref", _text(self.source_ref, "origin source ref"))

    def semantic_projection(self) -> Mapping[str, object]:
        return {"pointer": self.pointer, "origin": self.origin.value, "source_ref": self.source_ref}


@dataclass(frozen=True)
class StructJSONNumberBinding:
    """A server-owned location allowed to carry one finite IEEE-754 float."""
    json_pointer: str
    number_type: StructuredJSONNumberType = StructuredJSONNumberType.FLOAT64
    nullable: bool = False

    def __post_init__(self) -> None:
        _pointer(self.json_pointer)
        if not self.json_pointer:
            raise StructuredOutputError("numeric binding cannot target the root")
        if self.number_type is not StructuredJSONNumberType.FLOAT64:
            raise StructuredOutputError("unsupported structured numeric type")
        if not isinstance(self.nullable, bool):
            raise StructuredOutputError("structured numeric nullable must be bool")

    def semantic_projection(self) -> Mapping[str, object]:
        return {"json_pointer": self.json_pointer, "number_type": self.number_type.value,
            "nullable": self.nullable}


@dataclass(frozen=True)
class StructJSONNumberPattern:
    """One server-owned array-index wildcard expanded against one DTO tree."""
    json_pointer: str
    number_type: StructuredJSONNumberType = StructuredJSONNumberType.FLOAT64
    nullable: bool = False

    def __post_init__(self) -> None:
        parts = _pointer(self.json_pointer)
        if (not parts or len(parts) > 32 or parts.count("*") != 1 or parts[0] == "*"
                or any("*" in part and part != "*" for part in parts)):
            raise StructuredOutputError("structured numeric pattern needs one non-root array wildcard")
        if self.number_type is not StructuredJSONNumberType.FLOAT64 or not isinstance(self.nullable, bool):
            raise StructuredOutputError("structured numeric pattern invalid")

    def semantic_projection(self) -> Mapping[str, object]:
        return {"json_pointer": self.json_pointer, "number_type": self.number_type.value,
            "nullable": self.nullable}


@dataclass(frozen=True)
class StructuredNumberBindingContract:
    """The sole server registration for a route's fixed and array numeric fields."""
    bindings: tuple[StructJSONNumberBinding, ...] = ()
    patterns: tuple[StructJSONNumberPattern, ...] = ()

    def __post_init__(self) -> None:
        if (not isinstance(self.bindings, tuple) or not isinstance(self.patterns, tuple)
                or not all(isinstance(item, StructJSONNumberBinding) for item in self.bindings)
                or not all(isinstance(item, StructJSONNumberPattern) for item in self.patterns)):
            raise StructuredOutputError("structured numeric contract must be typed")
        if len({item.json_pointer for item in self.bindings}) != len(self.bindings):
            raise StructuredOutputError("duplicate structured numeric binding")
        if len({item.json_pointer for item in self.patterns}) != len(self.patterns):
            raise StructuredOutputError("duplicate structured numeric pattern")

    def semantic_projection(self) -> Mapping[str, object]:
        return {"bindings": [item.semantic_projection() for item in self.bindings],
            "patterns": [item.semantic_projection() for item in self.patterns]}


@dataclass(frozen=True)
class ExpandedNumberBindingManifest:
    """Exact pattern-to-concrete-pointer expansion sealed with one DTO."""
    patterns: tuple[StructJSONNumberPattern, ...]
    expanded: Mapping[str, tuple[str, ...]]

    def __post_init__(self) -> None:
        if not isinstance(self.patterns, tuple) or not all(isinstance(item, StructJSONNumberPattern) for item in self.patterns):
            raise StructuredOutputError("numeric manifest patterns invalid")
        expected = {item.json_pointer for item in self.patterns}
        if not isinstance(self.expanded, Mapping) or set(self.expanded) != expected:
            raise StructuredOutputError("numeric manifest shape invalid")
        frozen: dict[str, tuple[str, ...]] = {}
        seen: set[str] = set()
        for pattern in self.patterns:
            paths = self.expanded[pattern.json_pointer]
            if not isinstance(paths, tuple) or any(not isinstance(path, str) for path in paths):
                raise StructuredOutputError("numeric manifest paths invalid")
            if tuple(sorted(paths, key=lambda item: tuple(int(part) if part.isdecimal() else part for part in _pointer(item)))) != paths:
                raise StructuredOutputError("numeric manifest paths must be deterministic")
            if len(set(paths)) != len(paths) or seen.intersection(paths):
                raise StructuredOutputError("numeric manifest paths collide")
            seen.update(paths); frozen[pattern.json_pointer] = paths
        object.__setattr__(self, "expanded", MappingProxyType(dict(sorted(frozen.items()))))

    def semantic_projection(self) -> Mapping[str, object]:
        return {"patterns": [item.semantic_projection() for item in self.patterns],
            "expanded": {key: list(value) for key, value in self.expanded.items()}}


def _pointer_text(parts: tuple[str, ...]) -> str:
    return "/" + "/".join(part.replace("~", "~0").replace("/", "~1") for part in parts)


def expand_structured_number_patterns(value: object,
                                      patterns: tuple[StructJSONNumberPattern, ...]) -> tuple[tuple[StructJSONNumberBinding, ...], ExpandedNumberBindingManifest]:
    """Expand only existing array indices, with bounded deterministic traversal."""
    if not isinstance(patterns, tuple) or not all(isinstance(item, StructJSONNumberPattern) for item in patterns):
        raise StructuredOutputError("structured numeric patterns must be server typed")
    expanded: dict[str, tuple[str, ...]] = {}
    bindings: list[StructJSONNumberBinding] = []
    visits = 0
    for pattern in patterns:
        parts = _pointer(pattern.json_pointer); star = parts.index("*"); node = value
        for part in parts[:star]:
            visits += 1
            if visits > 1024:
                raise StructuredOutputError("structured numeric pattern expansion exceeds budget")
            if isinstance(node, Mapping):
                if part not in node: raise StructuredOutputError("structured numeric pattern leaf absent")
                node = node[part]
            elif isinstance(node, (tuple, list)) and part.isdecimal() and str(int(part)) == part and int(part) < len(node):
                node = node[int(part)]
            else:
                raise StructuredOutputError("structured numeric pattern crosses scalar")
        if not isinstance(node, (tuple, list)):
            raise StructuredOutputError("structured numeric wildcard must select array")
        paths: list[str] = []
        for index, item in enumerate(node):
            visits += 1
            if visits > 1024:
                raise StructuredOutputError("structured numeric pattern expansion exceeds budget")
            current = item
            for part in parts[star + 1:]:
                visits += 1
                if visits > 1024:
                    raise StructuredOutputError("structured numeric pattern expansion exceeds budget")
                if isinstance(current, Mapping):
                    if part not in current: raise StructuredOutputError("structured numeric pattern leaf absent")
                    current = current[part]
                elif isinstance(current, (tuple, list)) and part.isdecimal() and str(int(part)) == part and int(part) < len(current):
                    current = current[int(part)]
                else:
                    raise StructuredOutputError("structured numeric pattern leaf absent")
            path = _pointer_text(parts[:star] + (str(index),) + parts[star + 1:])
            paths.append(path)
            bindings.append(StructJSONNumberBinding(path, pattern.number_type, pattern.nullable))
        expanded[pattern.json_pointer] = tuple(paths)
    manifest = ExpandedNumberBindingManifest(patterns, expanded)
    return tuple(bindings), manifest


def _validate_numeric_binding_overlap(bindings: tuple[StructJSONNumberBinding, ...],
                                      slots: tuple[StructuredClaimSlot, ...]) -> None:
    claim_pointers = [pointer for slot in slots for pointer in slot.argument_pointers.values()]
    if any(_is_prefix(binding.json_pointer, pointer) or _is_prefix(pointer, binding.json_pointer)
           for binding in bindings for pointer in claim_pointers):
        raise StructuredOutputError("structured numeric binding overlaps claim slot")


def _walk_json(value: object, pointer: str = ""):
    yield pointer, value
    if isinstance(value, Mapping):
        for key, child in value.items():
            key = _text(key, "structured JSON key")
            yield from _walk_json(child, pointer + "/" + key.replace("~", "~0").replace("/", "~1"))
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            yield from _walk_json(child, pointer + "/" + str(index))


def pack_structured_float64_carriers(value: object, bindings: tuple[StructJSONNumberBinding, ...],
                                     slots: tuple[StructuredClaimSlot, ...] = ()) -> object:
    """Convert only layout-bound finite floats into the sealed F2-A-safe carrier."""
    if not isinstance(bindings, tuple) or not all(isinstance(item, StructJSONNumberBinding) for item in bindings):
        raise StructuredOutputError("structured numeric bindings must be server typed")
    _validate_numeric_binding_overlap(bindings, slots)
    binding_by_pointer = {binding.json_pointer: binding for binding in bindings}
    if len(binding_by_pointer) != len(bindings):
        raise StructuredOutputError("duplicate structured numeric binding")
    for pointer, item in _walk_json(value):
        if isinstance(item, Mapping) and STRUCTURED_FLOAT64_CARRIER_TAG in item:
            raise StructuredOutputError("preexisting structured float carrier")
        if isinstance(item, float):
            binding = binding_by_pointer.get(pointer)
            if binding is None or not math.isfinite(item):
                raise StructuredOutputError("unbound or non-finite structured float")
    packed = value
    for binding in bindings:
        item = _get(packed, binding.json_pointer)
        if item is None and binding.nullable:
            continue
        if isinstance(item, bool) or not isinstance(item, float) or not math.isfinite(item):
            raise StructuredOutputError("structured numeric binding requires finite float or null")
        packed = _set(packed, binding.json_pointer,
            {STRUCTURED_FLOAT64_CARRIER_TAG: item.hex()})
    return _freeze(packed, "structured float carrier")


def _materialize_structured_float64_carriers(value: object,
                                             bindings: tuple[StructJSONNumberBinding, ...]) -> object:
    binding_by_pointer = {binding.json_pointer: binding for binding in bindings}
    if len(binding_by_pointer) != len(bindings):
        raise StructuredOutputError("duplicate structured numeric binding")
    for pointer, item in _walk_json(value):
        if isinstance(item, float):
            raise StructuredOutputError("native float bypasses structured carrier")
        if isinstance(item, Mapping) and STRUCTURED_FLOAT64_CARRIER_TAG in item:
            binding = binding_by_pointer.get(pointer)
            if binding is None or set(item) != {STRUCTURED_FLOAT64_CARRIER_TAG}:
                raise StructuredOutputError("structured float carrier is misplaced")
            encoded = item[STRUCTURED_FLOAT64_CARRIER_TAG]
            if not isinstance(encoded, str):
                raise StructuredOutputError("structured float carrier lexeme invalid")
            try:
                decoded = float.fromhex(encoded)
            except ValueError as exc:
                raise StructuredOutputError("structured float carrier lexeme invalid") from exc
            if not math.isfinite(decoded) or decoded.hex() != encoded:
                raise StructuredOutputError("structured float carrier lexeme noncanonical")
    materialized = value
    for binding in bindings:
        item = _get(materialized, binding.json_pointer)
        if item is None:
            if binding.nullable:
                continue
            raise StructuredOutputError("structured numeric binding requires float")
        if not isinstance(item, Mapping) or set(item) != {STRUCTURED_FLOAT64_CARRIER_TAG}:
            raise StructuredOutputError("structured numeric binding lacks carrier")
        encoded = item[STRUCTURED_FLOAT64_CARRIER_TAG]
        try:
            decoded = float.fromhex(encoded) if isinstance(encoded, str) else None
        except ValueError as exc:
            raise StructuredOutputError("structured float carrier lexeme invalid") from exc
        if decoded is None or not math.isfinite(decoded) or decoded.hex() != encoded:
            raise StructuredOutputError("structured float carrier lexeme noncanonical")
        materialized = _set(materialized, binding.json_pointer, decoded)
    return _structured_freeze(materialized, "structured materialized output")


@dataclass(frozen=True)
class StructuredLayout:
    """Server-owned immutable layout. Request/model payloads cannot define it."""
    layout_id: str
    layout_version: str
    channel_id: str
    root_block_index: int
    slots: tuple[StructuredClaimSlot, ...] = ()
    origins: tuple[OriginBinding, ...] = ()
    presentation_maps: Mapping[str, Mapping[str, object]] = field(default_factory=dict)
    fallback: Mapping[str, object] = field(default_factory=lambda: {"error": "governed_output_unavailable"})
    number_bindings: tuple[StructJSONNumberBinding, ...] = ()
    number_patterns: tuple[StructJSONNumberPattern, ...] = ()
    number_binding_manifest: ExpandedNumberBindingManifest = field(
        default_factory=lambda: ExpandedNumberBindingManifest((), {}))

    def __post_init__(self) -> None:
        for name in ("layout_id", "layout_version", "channel_id"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        if not isinstance(self.root_block_index, int) or self.root_block_index < 0:
            raise StructuredOutputError("root_block_index must be a nonnegative int")
        if not isinstance(self.slots, tuple) or not all(isinstance(x, StructuredClaimSlot) for x in self.slots):
            raise StructuredOutputError("layout slots must be typed tuple")
        if len({x.claim_id for x in self.slots}) != len(self.slots):
            raise StructuredOutputError("each structured slot needs a distinct claim")
        if (not isinstance(self.number_bindings, tuple)
                or not all(isinstance(x, StructJSONNumberBinding) for x in self.number_bindings)):
            raise StructuredOutputError("layout numeric bindings must be typed tuple")
        number_pointers = [binding.json_pointer for binding in self.number_bindings]
        if len(number_pointers) != len(set(number_pointers)):
            raise StructuredOutputError("duplicate structured numeric binding")
        if (not isinstance(self.number_patterns, tuple)
                or not all(isinstance(x, StructJSONNumberPattern) for x in self.number_patterns)
                or not isinstance(self.number_binding_manifest, ExpandedNumberBindingManifest)
                or self.number_binding_manifest.patterns != self.number_patterns):
            raise StructuredOutputError("layout numeric pattern manifest must be exact")
        manifest_pointers = tuple(path for paths in self.number_binding_manifest.expanded.values() for path in paths)
        if any(pointer not in number_pointers for pointer in manifest_pointers):
            raise StructuredOutputError("layout numeric manifest must bind concrete paths")
        claim_pointers = [pointer for slot in self.slots for pointer in slot.argument_pointers.values()]
        if any(_is_prefix(left, right) or _is_prefix(right, left)
               for left in number_pointers for right in claim_pointers):
            raise StructuredOutputError("structured numeric binding overlaps claim slot")
        if not isinstance(self.origins, tuple) or not self.origins or not all(isinstance(x, OriginBinding) for x in self.origins):
            raise StructuredOutputError("layout origins must be nonempty typed tuple")
        pointers = [x.pointer for x in self.origins]
        if len(pointers) != len(set(pointers)):
            raise StructuredOutputError("duplicate origin binding")
        # Overlap is resolved only at render time against actual leaves. This
        # permits a root default plus narrow array patterns, while ties fail
        # closed rather than choosing an arbitrary provenance.
        if not isinstance(self.presentation_maps, Mapping):
            raise StructuredOutputError("presentation maps must be mapping")
        maps: dict[str, Mapping[str, object]] = {}
        for map_id, values in self.presentation_maps.items():
            map_id = _text(map_id, "presentation map id")
            if not isinstance(values, Mapping) or not values:
                raise StructuredOutputError("presentation map must be nonempty mapping")
            mapped: dict[str, object] = {}
            for canonical, presentation in values.items():
                canonical = _text(canonical, "presentation canonical value")
                mapped[canonical] = _freeze(presentation, "presentation value")
            maps[map_id] = MappingProxyType(dict(sorted(mapped.items())))
        for slot in self.slots:
            if slot.projection_mode is SlotProjectionMode.PRESENTATION_MAP and slot.presentation_map_id not in maps:
                raise StructuredOutputError("structured slot selects unknown presentation map")
        object.__setattr__(self, "presentation_maps", MappingProxyType(dict(sorted(maps.items()))))
        frozen_fallback = _freeze(self.fallback, "structured fallback")
        if not isinstance(frozen_fallback, Mapping):
            raise StructuredOutputError("structured fallback must be object")
        # A fallback is server-owned static content. It cannot smuggle a known
        # current runtime proposition.
        if GovernedDomainRegistry().specification.runtime_status_tool_data_predicate(_plain(frozen_fallback)) is not None:
            raise StructuredOutputError("structured fallback cannot contain runtime status")
        object.__setattr__(self, "fallback", frozen_fallback)

    @property
    def digest(self) -> str:
        return _digest(self.semantic_projection())

    def semantic_projection(self) -> Mapping[str, object]:
        return {"layout_id": self.layout_id, "layout_version": self.layout_version,
            "channel_id": self.channel_id, "root_block_index": self.root_block_index,
            "slots": [slot.semantic_projection() for slot in self.slots],
            "origins": [origin.semantic_projection() for origin in self.origins],
            "presentation_maps": {key: _plain(value) for key, value in self.presentation_maps.items()},
            "fallback": _plain(self.fallback),
            "number_bindings": [binding.semantic_projection() for binding in self.number_bindings],
            "number_patterns": [pattern.semantic_projection() for pattern in self.number_patterns],
            "number_binding_manifest": self.number_binding_manifest.semantic_projection()}


@dataclass(frozen=True)
class StructuredLayoutRegistry:
    """Trusted composition registry; only its exact layouts are renderable."""
    layouts: Mapping[str, StructuredLayout]

    def __post_init__(self) -> None:
        if not isinstance(self.layouts, Mapping) or not self.layouts:
            raise StructuredOutputError("structured layout registry must be nonempty mapping")
        result: dict[str, StructuredLayout] = {}
        for key, layout in self.layouts.items():
            if not isinstance(key, str) or not isinstance(layout, StructuredLayout) or key != layout.layout_id:
                raise StructuredOutputError("layout registry has invalid entry")
            result[key] = layout
        object.__setattr__(self, "layouts", MappingProxyType(dict(sorted(result.items()))))

    def require(self, layout: StructuredLayout) -> StructuredLayout:
        if not isinstance(layout, StructuredLayout) or self.layouts.get(layout.layout_id) != layout:
            raise StructuredOutputError("structured layout is not server registered")
        return layout


@dataclass(frozen=True)
class RenderedStructuredOutput:
    value: object
    canonical_bytes: bytes
    response_id: str
    envelope_digest: str
    source_envelope_digest: str
    contract_state: ContractState
    layout_id: str
    layout_digest: str
    channel_id: str
    claim_ids: tuple[str, ...]
    protected_subtrees_digest: str
    origin_manifest_digest: str
    output_digest: str
    renderer_api_version: str = STRUCTURED_RENDERER_API_VERSION
    domain_spec_version: str = GOVERNED_DOMAIN_SPEC_VERSION
    structured_schema_version: str = STRUCTURED_OUTPUT_SCHEMA_VERSION


class GovernedStructuredRenderer:
    unavailable_value = MappingProxyType({"error": "governed_output_unavailable"})

    def render(self, envelope: GovernedResponseEnvelope, context: RenderContext,
               layout: StructuredLayout, registry: StructuredLayoutRegistry) -> RenderedStructuredOutput:
        if not isinstance(envelope, GovernedResponseEnvelope) or envelope.compute_digest() != envelope.envelope_digest:
            raise StructuredOutputError("structured renderer requires intact sealed envelope")
        if not isinstance(context, RenderContext):
            raise StructuredOutputError("structured renderer requires trusted context")
        try:
            layout = registry.require(layout)
        except StructuredOutputError:
            return self._unregistered_fallback(envelope)
        if (envelope.candidate.schema_version not in GOVERNED_ENVELOPE_SCHEMA_VERSIONS
                or context.renderer_api_version != "f2-c.renderer.3"
                or context.domain_registry.specification.version != GOVERNED_DOMAIN_SPEC_VERSION):
            return self._fallback(envelope, layout)
        if envelope.contract_state is not ContractState.VALID:
            return self._fallback(envelope, layout)
        try:
            root = self._root(envelope, layout)
            expanded, manifest = expand_structured_number_patterns(root, layout.number_patterns)
            if (manifest != layout.number_binding_manifest
                    or tuple((*[binding for binding in layout.number_bindings
                                if binding.json_pointer not in {path for paths in manifest.expanded.values() for path in paths}], *expanded)) != layout.number_bindings):
                raise StructuredOutputError("structured numeric pattern expansion changed")
            claims = {claim.claim_id: claim for claim in envelope.claims}
            references = {reference.ref_id: reference for reference in envelope.references}
            claim_block_ids = self._claim_block(envelope, layout)
            protected_before: dict[str, object] = {}
            rendered = root
            validator = GovernedRenderer()
            for slot in layout.slots:
                claim = claims.get(slot.claim_id)
                if (claim is None or claim.claim_id not in claim_block_ids
                        or claim.predicate != slot.predicate
                        or claim.disposition is not ClaimDisposition.ASSERTABLE
                        or claim.epistemic_status is not EpistemicStatus.CURRENT_OBSERVATION
                        or set(slot.argument_pointers) | set(slot.server_constants) != set(claim.typed_arguments)):
                    raise StructuredOutputError("structured slot has no exact assertable current claim")
                validator._validate_claim(claim, envelope.response_scope, context, context.now(), references)
                for name, constant in slot.server_constants.items():
                    if claim.typed_arguments[name] != constant:
                        raise StructuredOutputError("structured server constant does not equal accredited claim")
                for name, pointer in slot.argument_pointers.items():
                    supplied = _get(root, pointer)
                    authoritative = claim.typed_arguments[name]
                    if slot.projection_mode is SlotProjectionMode.PRESENTATION_MAP and name == slot.presentation_argument:
                        mapped = layout.presentation_maps[slot.presentation_map_id or ""].get(str(authoritative))
                        if mapped is None or supplied != mapped:
                            raise StructuredOutputError("structured presentation value does not equal trusted map")
                        protected_before[pointer] = {"canonical": authoritative, "presented": supplied,
                            "mapper_id": slot.presentation_map_id}
                        rendered = _set(rendered, pointer, mapped)
                    else:
                        if supplied != authoritative:
                            raise StructuredOutputError("structured slot value does not equal accredited claim")
                        protected_before[pointer] = supplied
                        rendered = _set(rendered, pointer, authoritative)
            # F2-A seals only canonical JSON values.  F2-E unpacks the
            # server-owned, layout-bound carrier after claims are validated.
            root = _materialize_structured_float64_carriers(root, layout.number_bindings)
            rendered = _materialize_structured_float64_carriers(rendered, layout.number_bindings)
            self._no_residual_status(root, layout)
            self._no_residual_status(rendered, layout)
            protected_after = {}
            for slot in layout.slots:
                for name, pointer in slot.argument_pointers.items():
                    if slot.projection_mode is SlotProjectionMode.PRESENTATION_MAP and name == slot.presentation_argument:
                        protected_after[pointer] = {"canonical": claims[slot.claim_id].typed_arguments[name],
                            "presented": _get(rendered, pointer), "mapper_id": slot.presentation_map_id}
                    else:
                        protected_after[pointer] = _get(rendered, pointer)
            if protected_before != protected_after:
                raise StructuredOutputError("structured protected values changed after rendering")
            origins = self._cover_origins(rendered, layout)
            return self._effective(envelope, layout, rendered, tuple(slot.claim_id for slot in layout.slots), protected_before, origins)
        except (GovernanceContractError, GovernedRenderError, StructuredOutputError):
            return self._fallback(envelope, layout)

    def _root(self, envelope: GovernedResponseEnvelope, layout: StructuredLayout) -> object:
        blocks = envelope.content_blocks
        if layout.root_block_index >= len(blocks) or len(blocks) not in {1, 2}:
            raise StructuredOutputError("structured envelope must contain root and optional claim block")
        root_block = blocks[layout.root_block_index]
        if root_block.kind is not ContentBlockKind.TOOL_DATA:
            raise StructuredOutputError("structured root must be TOOL_DATA")
        if any(block.kind not in {ContentBlockKind.TOOL_DATA, ContentBlockKind.CLAIM_REF_BLOCK} for block in blocks):
            raise StructuredOutputError("structured envelope contains unsupported block")
        if sum(block.kind is ContentBlockKind.TOOL_DATA for block in blocks) != 1:
            raise StructuredOutputError("structured envelope needs exactly one TOOL_DATA root")
        return _freeze(root_block.payload, "structured root")

    def _claim_block(self, envelope: GovernedResponseEnvelope, layout: StructuredLayout) -> frozenset[str]:
        blocks = [block for block in envelope.content_blocks if block.kind is ContentBlockKind.CLAIM_REF_BLOCK]
        expected = frozenset(slot.claim_id for slot in layout.slots)
        if not expected:
            if blocks:
                raise StructuredOutputError("unneeded structured claim block")
            return frozenset()
        if len(blocks) != 1 or frozenset(blocks[0].claim_refs) != expected:
            raise StructuredOutputError("structured claim block must bind exact layout slots")
        return expected

    def _no_residual_status(self, root: object, layout: StructuredLayout) -> None:
        masked = root
        # Mask only the server-declared protected pointers. The untouched tree
        # must contain no registered runtime status, including encoded JSON.
        for slot in layout.slots:
            for pointer in slot.argument_pointers.values():
                masked = _set(masked, pointer, "__governed_structured_mask__")
        predicate = GovernedDomainRegistry().specification.runtime_status_tool_data_predicate(_plain(masked))
        if predicate is not None:
            raise StructuredOutputError("unbound or ambiguous runtime status in structured output")

    def _cover_origins(self, root: object, layout: StructuredLayout) -> Mapping[str, Mapping[str, object]]:
        bindings = layout.origins
        manifest: dict[str, Mapping[str, object]] = {}
        for leaf in _leaves(root):
            matches = [binding for binding in bindings if _pattern_matches(binding.pointer, leaf)]
            if not matches:
                raise StructuredOutputError("each structured leaf needs exactly one origin")
            highest = max(_pattern_specificity(binding.pointer) for binding in matches)
            selected = [binding for binding in matches if _pattern_specificity(binding.pointer) == highest]
            if len(selected) != 1:
                raise StructuredOutputError("equally specific structured origin patterns conflict")
            binding = selected[0]
            manifest[leaf] = {"origin": binding.origin.value, "source_ref": binding.source_ref,
                "pattern": binding.pointer}
        return MappingProxyType(dict(sorted(manifest.items())))

    def _effective(self, envelope: GovernedResponseEnvelope, layout: StructuredLayout,
                   value: object, claim_ids: tuple[str, ...], protected: Mapping[str, object],
                   origins: Mapping[str, Mapping[str, object]] | None = None,
                   *, contract_state: ContractState | None = None) -> RenderedStructuredOutput:
        frozen = _structured_freeze(value, "structured effective output")
        canonical = _canonical_bytes(frozen)
        protected_digest = _digest(dict(sorted(protected.items())))
        origins_digest = _digest({"contract": [binding.semantic_projection() for binding in layout.origins],
            "manifest": _plain(origins or {})})
        output_digest = _digest({"bytes_sha256": "sha256:" + hashlib.sha256(canonical).hexdigest(),
            "envelope_digest": envelope.envelope_digest, "layout_digest": layout.digest,
            "channel_id": layout.channel_id, "claim_ids": list(claim_ids),
            "protected_subtrees_digest": protected_digest, "origin_manifest_digest": origins_digest,
            "renderer_api_version": STRUCTURED_RENDERER_API_VERSION,
            "domain_spec_version": GOVERNED_DOMAIN_SPEC_VERSION,
            "structured_schema_version": STRUCTURED_OUTPUT_SCHEMA_VERSION})
        state = envelope.contract_state if contract_state is None else contract_state
        return RenderedStructuredOutput(frozen, canonical, envelope.response_id, output_digest,
            envelope.envelope_digest, state, layout.layout_id, layout.digest,
            layout.channel_id, claim_ids, protected_digest, origins_digest, output_digest)

    def _fallback(self, envelope: GovernedResponseEnvelope, layout: StructuredLayout) -> RenderedStructuredOutput:
        # Fallback is a server-owned layout value and contains no candidate text.
        fallback_origins = MappingProxyType({leaf: {"origin": OutputOrigin.SYSTEM.value,
            "source_ref": None, "pattern": "<server-fallback>"} for leaf in _leaves(layout.fallback)})
        return self._effective(envelope, layout, layout.fallback, (), {}, fallback_origins, contract_state=ContractState.UNAVAILABLE)

    def _unregistered_fallback(self, envelope: GovernedResponseEnvelope) -> RenderedStructuredOutput:
        """Fail closed without using an unregistered caller-selected layout."""
        layout = StructuredLayout("f2e-structured-unavailable", "1", "UNAVAILABLE", 0,
            (), (OriginBinding("", OutputOrigin.SYSTEM),))
        return self._fallback(envelope, layout)
