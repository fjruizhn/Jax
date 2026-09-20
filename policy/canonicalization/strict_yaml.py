"""Parser YAML con semántica escalar explícita y cerrada.

PyYAML se usa sólo para componer la estructura y conservar el estilo léxico.
Ningún resolver implícito de ``SafeLoader`` participa en la semántica.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from .errors import StrictYAMLError
from .normalize import normalize_semantic

_CANONICAL_INTEGER = re.compile(r"[+-]?(?:0|[1-9][0-9]*)\Z")
_INTEGER_LIKE = (
    re.compile(r"[+-]?0[0-9_]+\Z"),
    re.compile(r"[+-]?0[bBoOxX][0-9A-Fa-f_]+\Z"),
    re.compile(r"[+-]?[0-9][0-9_]*:[0-9_:]+\Z"),
    re.compile(r"[+-]?[0-9][0-9_]*\Z"),
)
_FLOAT_LIKE = (
    re.compile(r"[+-]?(?:[0-9][0-9_]*)?\.[0-9_]*\Z"),
    re.compile(r"[+-]?[0-9][0-9_]*(?:\.[0-9_]*)?[eE][+-]?[0-9]+\Z"),
    re.compile(r"[+-]?\.(?:inf|nan)\Z", re.IGNORECASE),
)
_YAML_BOOLEAN_WORDS = frozenset({"yes", "no", "on", "off", "true", "false"})
_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1


def _plain_scalar(value: str) -> Any:
    if value == "true":
        return True
    if value == "false":
        return False
    if value == "null":
        return None

    folded = value.lower()
    if folded in _YAML_BOOLEAN_WORDS:
        raise StrictYAMLError(f"boolean YAML ambiguo no permitido: {value!r}")
    if value == "~" or folded == "null":
        raise StrictYAMLError(f"null YAML ambiguo no permitido: {value!r}")
    if value == "":
        raise StrictYAMLError("scalar plain vacío no permitido; use null o quotes")

    if _CANONICAL_INTEGER.fullmatch(value):
        integer = int(value, 10)
        if not _INT64_MIN <= integer <= _INT64_MAX:
            raise StrictYAMLError(f"integer fuera de int64: {value!r}")
        return integer
    if any(pattern.fullmatch(value) for pattern in _INTEGER_LIKE):
        raise StrictYAMLError(f"integer YAML no canónico: {value!r}")
    if any(pattern.fullmatch(value) for pattern in _FLOAT_LIKE):
        raise StrictYAMLError(f"float no permitido: {value!r}")
    return value


def _construct_node(node: yaml.Node) -> Any:
    if isinstance(node, yaml.ScalarNode):
        # Quotes y block scalars son representaciones explícitas de string.
        return node.value if node.style is not None else _plain_scalar(node.value)
    if isinstance(node, yaml.SequenceNode):
        return [_construct_node(item) for item in node.value]
    if isinstance(node, yaml.MappingNode):
        mapping: dict[str, Any] = {}
        for key_node, value_node in node.value:
            key = _construct_node(key_node)
            if not isinstance(key, str):
                raise StrictYAMLError("todas las claves YAML deben ser strings")
            if key == "<<":
                raise StrictYAMLError("merge keys no permitidas")
            if key in mapping:
                raise StrictYAMLError(f"clave duplicada: {key!r}")
            mapping[key] = _construct_node(value_node)
        return mapping
    raise StrictYAMLError(f"nodo YAML no permitido: {type(node).__name__}")


def _text_from_source(source: str | bytes | Path) -> str:
    if isinstance(source, Path):
        raw = source.read_bytes()
        label = str(source)
    elif isinstance(source, bytes):
        raw = source
        label = "fuente"
    elif isinstance(source, str):
        return source
    else:
        raise TypeError("source debe ser str, bytes o Path")
    try:
        return raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise StrictYAMLError(f"{label}: UTF-8 inválido") from exc


def load_strict_yaml(source: str | bytes | Path) -> Any:
    text = _text_from_source(source)
    try:
        for token in yaml.scan(text, Loader=yaml.BaseLoader):
            if isinstance(token, yaml.tokens.DirectiveToken):
                raise StrictYAMLError("directivas YAML no permitidas")
            if isinstance(token, yaml.tokens.AnchorToken):
                raise StrictYAMLError("anchors no permitidos")
            if isinstance(token, yaml.tokens.AliasToken):
                raise StrictYAMLError("aliases no permitidos")
            if isinstance(token, yaml.tokens.TagToken):
                raise StrictYAMLError("tags explícitos/custom no permitidos")
        node = yaml.compose(text, Loader=yaml.BaseLoader)
        if node is None:
            raise StrictYAMLError("documento YAML vacío")
        return normalize_semantic(_construct_node(node))
    except StrictYAMLError:
        raise
    except yaml.YAMLError as exc:
        raise StrictYAMLError(f"YAML estricto inválido: {exc}") from exc
