"""
policy/governance — Schema de claim.

REFORMAS-v3.md §3.1 (R1). Capa 1 de dos: estructural (este módulo,
Pydantic) y semántica (validator.py). Un claim mal tipado no llega nunca
al validador semántico — mismo principio de dos capas que
las_manos/envelope.py usa para IntentEnvelope.

Este módulo es PURO: sin I/O, sin red, testeable en aislamiento.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class Claim(BaseModel):
    predicate: str
    args: dict[str, str]
    authority: Literal["EJECUTADO", "OBSERVADO", "RECUPERADO", "INFERIDO"]
    provenance_ref: str
    evidence_pointer: str
    scope: str
