"""F2-C canonical, versioned projection of JAX governed vocabulary.

This is deliberately deterministic grammar data, never a probabilistic text
classifier.  Compositions extend it only with server-approved aliases from
their JAX registry projection.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import re
import unicodedata
from types import MappingProxyType
from typing import Mapping

from .response import GovernanceContractError, _text

GOVERNED_DOMAIN_SPEC_VERSION = "f2-c.domain.2"
GOVERNED_RENDERER_API_VERSION = "f2-c.renderer.2"
GOVERNED_ENVELOPE_SCHEMA_VERSIONS = frozenset({"f2-c.1"})

_CANONICAL_STATUS_ALIASES = MappingProxyType({
    "healthy": ("healthy", "up", "available", "operational", "sano", "saludable", "activo", "disponible", "funcionando"),
    "exists": ("exists", "exist", "present", "existe", "existen"),
    "completed": ("completed", "finished", "succeeded", "terminó", "termino", "finalizó", "finalizo", "correctamente"),
    "down": ("down", "unhealthy", "unavailable", "caído", "caido", "inactivo"),
})
_CANONICAL_LOCALE_ALIASES = MappingProxyType({
    "en": ("is", "are", "exists", "exist", "available", "healthy", "up", "down", "completed"),
    "es": ("es", "está", "esta", "son", "existe", "existen", "disponible", "saludable", "sano", "caído", "caido", "terminó", "termino"),
})

# Detection-only punctuation equivalence for the registered English
# contraction grammar. This is deliberately a small, versioned set: it does
# not transliterate arbitrary Unicode and never changes rendered payload.
_GOVERNED_APOSTROPHE_TRANSLATION = str.maketrans({"\u2019": "'"})


def _canonicalize_governed_detection_text(value: str) -> str:
    """Apply frozen Unicode normalization and approved punctuation for match."""
    normalized = unicodedata.normalize("NFC", value)
    return normalized.translate(_GOVERNED_APOSTROPHE_TRANSLATION).casefold()


def _canon(value: str, name: str) -> str:
    return _text(value, name).casefold()


@dataclass(frozen=True)
class GovernedDomainSpecification:
    """Registered predicates and their explicit English/Spanish grammar.

    ``entity_aliases`` and ``resource_aliases`` are a projection of canonical
    JAX governance registrations.  They are not a platform-maintained list.
    """
    version: str = GOVERNED_DOMAIN_SPEC_VERSION
    predicates: tuple[str, ...] = ()
    predicate_aliases: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    entity_aliases: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    resource_aliases: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    locale_aliases: Mapping[str, tuple[str, ...]] = field(default_factory=lambda: _CANONICAL_LOCALE_ALIASES)
    status_aliases: Mapping[str, tuple[str, ...]] = field(default_factory=lambda: _CANONICAL_STATUS_ALIASES)

    def __post_init__(self) -> None:
        object.__setattr__(self, "version", _text(self.version, "domain specification version"))
        canonical_predicates, canonical_entities, canonical_resources = _canonical_vocabulary()
        predicates = tuple(sorted({_text(p, "predicate") for p in (*canonical_predicates, *self.predicates)}))
        if not predicates:
            raise GovernanceContractError("domain specification requires predicates")
        object.__setattr__(self, "predicates", predicates)
        predicate_aliases = {p: (p.casefold(), p.casefold().replace("_", " ")) for p in canonical_predicates}
        if not isinstance(self.predicate_aliases, Mapping):
            raise GovernanceContractError("predicate_aliases must be mapping")
        for predicate, aliases in self.predicate_aliases.items():
            predicate = _text(predicate, "predicate alias owner")
            if not isinstance(aliases, tuple) or not aliases:
                raise GovernanceContractError("predicate aliases must be nonempty tuples")
            predicate_aliases[predicate] = tuple(predicate_aliases.get(predicate, ())) + aliases
        object.__setattr__(self, "predicate_aliases", MappingProxyType({
            p: tuple(sorted({_canon(x, "predicate alias") for x in aliases}))
            for p, aliases in sorted(predicate_aliases.items())
        }))
        canonical_maps = {
            "entity_aliases": canonical_entities,
            "resource_aliases": canonical_resources,
            "status_aliases": _CANONICAL_STATUS_ALIASES,
            "locale_aliases": _CANONICAL_LOCALE_ALIASES,
        }
        for field_name in ("entity_aliases", "resource_aliases", "status_aliases", "locale_aliases"):
            values = getattr(self, field_name)
            if not isinstance(values, Mapping):
                raise GovernanceContractError(f"{field_name} must be mapping")
            normalized = {}
            merged = dict(canonical_maps[field_name])
            for key, aliases in values.items():
                key = _canon(key, field_name)
                if not isinstance(aliases, tuple) or not aliases:
                    raise GovernanceContractError(f"{field_name} aliases must be nonempty tuples")
                prior = merged.get(key, ())
                merged[key] = tuple(prior) + tuple(aliases)
            for key, aliases in merged.items():
                normalized[key] = tuple(sorted({_canon(a, field_name) for a in aliases}))
            object.__setattr__(self, field_name, MappingProxyType(dict(sorted(normalized.items()))))

    def projection(self) -> Mapping[str, object]:
        return MappingProxyType({"version": self.version, "predicates": self.predicates,
            "predicate_aliases": dict(self.predicate_aliases),
            "entity_aliases": dict(self.entity_aliases), "resource_aliases": dict(self.resource_aliases),
            "status_aliases": dict(self.status_aliases), "locale_aliases": dict(self.locale_aliases)})

    def registered_proposition(self, text: str) -> str | None:
        """Return owning predicate when explicit registered grammar occurs."""
        text = _canonicalize_governed_detection_text(text)
        # Remove markup delimiters while retaining words/paths/URLs, so the
        # same registered assertion cannot hide in headings, tables or links.
        plain = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
        plain = re.sub(r"[`*_#|\[\]()<>]", " ", plain)
        aliases = tuple(a for values in self.entity_aliases.values() for a in values)
        resources = tuple(a for values in self.resource_aliases.values() for a in values)
        subjects = aliases + resources
        subject = "|".join(re.escape(x) for x in subjects) if subjects else r"[\w./:-]+"
        status = "|".join(re.escape(x) for values in self.status_aliases.values() for x in values)
        locale_words = {word for aliases in self.locale_aliases.values() for word in aliases}
        copula_words = sorted(locale_words & {"is", "are", "es", "está", "esta", "son"})
        copula = r"(?:" + "|".join(re.escape(x) for x in copula_words) + r"|was|were|isn't|isnt|is\s+not|are\s+not|no\s+está|no\s+esta|no\s+es)"
        exists = r"(?:exists|exist|does\s+not\s+exist|doesn't\s+exist|no\s+existe|no\s+existen|existe|existen)"
        patterns = (
            ("ENGINE_STATUS", rf"\b(?:{subject})\b\s+{copula}\s+(?:{status})\b"),
            ("FILE_EXISTS", rf"\b(?:the\s+)?(?:file|archivo|path|ruta)\s+(?:{subject}|/[^\s]+)\s+{exists}\b|\b(?:{subject}|/[^\s]+)\s+{exists}\b"),
            ("FACET_EXISTS", rf"\b(?:facet|faceta)\s+(?:{subject})\s+{exists}\b"),
            ("JOB_STATUS", rf"\b(?:job|trabajo)\s+[^\s]+\s+(?:(?:(?:is|was|está|esta|fue|ha)\s+)?(?:{status})|no\s+(?:{status}))\b"),
            ("CAPABILITY_AVAILABLE", rf"\b(?:capability|capacidad)\s+(?:{subject})\s+{copula}\s+(?:{status})\b"),
            ("CAPABILITY_AVAILABLE", rf"\b(?:the\s+)?(?:capability|capacidad)\s+{copula}\s+(?:{status})\b"),
            ("CONFIG_VALUE", r"\b(?:config(?:uration)?|configuración)\s+(?:value|valor)\b"),
            ("AUDIT_EVENT_EXISTS", r"\b(?:audit\s+event|evento\s+de\s+auditoría)\s+[^\s]+\s+(?:exists|exist|existe|existen)\b"),
            ("MEMORY_ENTRY_EXISTS", r"\b(?:memory\s+entry|entrada\s+de\s+memoria)\s+[^\s]+\s+(?:exists|exist|existe|existen)\b"),
        )
        enabled = set(self.predicates)
        for predicate, aliases in self.predicate_aliases.items():
            if predicate in enabled and any(re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", plain) for alias in aliases):
                return predicate
        for predicate, pattern in patterns:
            if predicate in enabled and re.search(pattern, plain, flags=re.IGNORECASE):
                return predicate
        return None


@lru_cache(maxsize=1)
def _canonical_vocabulary():
    """Project registered JAX predicates, entities and resource paths."""
    from .loaders import load_predicates, load_vocabulary

    predicates = tuple(load_predicates())
    vocabulary = load_vocabulary()
    entities: dict[str, tuple[str, ...]] = {
        term: (term,) for term, categories in vocabulary.term_categories.items()
        if categories & {"capabilities", "ops", "facets_las_manos", "facets_jax", "motors"}
    }
    # Hall9000 is a registered JAX governance entity. Explicit aliases remain
    # versioned here in core alongside policy-derived entities.
    entities["hall9000"] = ("Hall9000", "Hall 9000")
    resources = {path: (path,) for path in vocabulary.config_paths}
    resources.update({path: (path,) for path in ("/etc/passwd", "policy/", "las_manos/")})
    return predicates, entities, resources
