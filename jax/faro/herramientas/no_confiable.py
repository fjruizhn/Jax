"""Envoltura endurecida para texto externo devuelto a un modelo.

La neutralización sigue el patrón probado por LAS MANOS: se rompen los
delimitadores y tokens de control ANTES de añadir el único cierre confiable.
"""
from __future__ import annotations

import hashlib
import re


_SENTINEL_INYECCION = re.compile(
    r"</?untrusted_source\b[^<>]*>"
    r"|<\|[A-Za-z0-9_.\-]{1,64}\|>"
    r"|<<SYS>>|<</SYS>>"
    r"|\[/?(?:INST|SYSTEM)\]"
    r"|###?[ \t]*(?:system|instruction)s?[ \t]*:?",
    re.IGNORECASE,
)


def envolver_payload_no_confiable(contenido: str) -> str:
    """Devuelve el texto como dato no confiable, neutralizando cierres falsos."""
    digest = hashlib.sha256(contenido.encode("utf-8")).hexdigest()
    seguro = _SENTINEL_INYECCION.sub(lambda match: "\u200b".join(match.group(0)), contenido)
    return f'<untrusted_source sha256="{digest}">\n{seguro}\n</untrusted_source>'
