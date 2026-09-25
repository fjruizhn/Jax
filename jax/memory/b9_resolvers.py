"""Typed, read-only B9 current-source resolution adapters.

Memory references are locators.  This module deliberately returns an explicit
unavailable result unless the designated JAX source can be queried; it never
falls back to memory payload, a cache, or a caller supplied current value.
"""
from __future__ import annotations

import time
from typing import Callable, Mapping

from .b9 import MemoryReference, MemoryReferenceResolver, ResolutionResult, ResolutionState


class DesignatedSourceResolver:
    """Fixed-type resolver facade for B4--B8 and operational query sources.

    ``readers`` is composition-only plumbing.  Readers receive only the typed
    reference value and must return a ``ResolutionResult``.  A missing reader
    is an availability outcome, never a successful current value.
    """
    _SOURCES = {
        "policy_rule": "policy/B4",
        "authority": "B4",
        "authority_checkpoint": "B4",
        "decision": "B5",
        "execution": "B6",
        "evidence": "B7",
        "artifact": "B7",
        "assertion": "B7",
        "control": "B7",
        "runtime": "B8/jaxctl",
        "configuration": "B8/jaxctl",
        "health": "B8/jaxctl",
    }

    def __init__(self, readers: Mapping[str, Callable[[str], ResolutionResult]] | None = None):
        self._readers = dict(readers or {})

    def resolve(self, reference: MemoryReference) -> ResolutionResult:
        source = self._SOURCES.get(reference.reference_type)
        if source is None:
            return ResolutionResult(ResolutionState.UNRESOLVED, "B9", time.time())
        reader = self._readers.get(reference.reference_type)
        if reader is None:
            return ResolutionResult(ResolutionState.SOURCE_UNAVAILABLE, source, time.time())
        result = reader(reference.reference_value)
        if not isinstance(result, ResolutionResult):
            return ResolutionResult(ResolutionState.SOURCE_UNAVAILABLE, source, time.time())
        # A reader may not relabel a different source as current resolution.
        if result.state is ResolutionState.RESOLVED_CURRENT and result.source != source:
            return ResolutionResult(ResolutionState.INVALID_REFERENCE, source, time.time())
        return result

    def as_memory_resolver(self) -> MemoryReferenceResolver:
        return MemoryReferenceResolver({kind: (lambda value, k=kind: self.resolve(MemoryReference(k, value)))
                                        for kind in self._SOURCES})

