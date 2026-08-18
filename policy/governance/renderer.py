"""
policy/governance — Motor de plantillas por predicado (REFORMAS-v3.md
§3.1.6).

Puro: recibe templates ya cargados y verificados por hash (loaders.py),
nunca abre un archivo. No puede emitir texto para un predicado sin
plantilla 'definida' — §3.1.6: "El renderer no puede emitir texto para
un predicado sin plantilla registrada". Deliberadamente agnóstico de qué
predicados tienen resolver real en validator.py — esa desconexión es lo
que permite probarlo con claims sintéticos, sin esperar al resolver.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import claims
    import loaders


def render(claim: "claims.Claim", templates: dict) -> str:
    spec = templates.get(claim.predicate)
    if spec is None or spec.status != "definida" or spec.template is None:
        raise ValueError(
            f"'{claim.predicate}' no tiene plantilla 'definida' — el "
            "renderer no puede emitir texto para un predicado sin "
            "plantilla registrada (§3.1.6)."
        )
    return spec.template.format(**claim.args)
