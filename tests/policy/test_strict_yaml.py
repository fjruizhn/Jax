from __future__ import annotations

import pytest

from policy.canonicalization.errors import CanonicalizationError, StrictYAMLError
from policy.canonicalization.legacy import (
    LEGACY_RULE_SHADOW,
    load_legacy_rule_shadow,
)
from policy.canonicalization.strict_yaml import load_strict_yaml


@pytest.mark.parametrize(
    "document",
    [
        "a: 1\na: 2\n",
        "a: &shared value\n",
        "a: &shared value\nb: *shared\n",
        "a: !custom value\n",
        "a: 0.1\n",
        "base: &base {a: 1}\nmerged:\n  <<: *base\n",
    ],
)
def test_duplicate_key_anchor_alias_tag_float_rechazan(document: str) -> None:
    with pytest.raises(StrictYAMLError):
        load_strict_yaml(document)


def test_utf8_estricto_y_unicode_nfc() -> None:
    with pytest.raises(StrictYAMLError):
        load_strict_yaml(b"name: " + bytes([0xFF]) + b"\n")
    assert load_strict_yaml("name: Cafe\u0301\n") == {"name": "Caf\u00e9"}


def test_legacy_version_plain_scalar_se_normaliza_a_string() -> None:
    parsed = load_legacy_rule_shadow(
        "version: 0.1\n", source_classification=LEGACY_RULE_SHADOW
    )
    assert parsed["version"] == "0.1"
    assert isinstance(parsed["version"], str)


def test_legacy_version_quoted_y_plain_son_equivalentes() -> None:
    plain = load_legacy_rule_shadow(
        "version: 0.1\n", source_classification=LEGACY_RULE_SHADOW
    )
    quoted = load_legacy_rule_shadow(
        'version: "0.1"\n', source_classification=LEGACY_RULE_SHADOW
    )
    assert plain == quoted


@pytest.mark.parametrize(
    "document",
    [
        "threshold: 0.1\n",
        "ratio: 0.5\n",
        "timeout: 1.5\n",
        "foo: 0.2\n",
    ],
)
def test_float_fuera_de_legacy_version_rechaza(document: str) -> None:
    with pytest.raises(StrictYAMLError):
        load_legacy_rule_shadow(document, source_classification=LEGACY_RULE_SHADOW)


def test_float_nested_rechaza() -> None:
    with pytest.raises(StrictYAMLError):
        load_legacy_rule_shadow(
            "config:\n  ratio: 0.1\n", source_classification=LEGACY_RULE_SHADOW
        )


def test_strict_normative_parser_sigue_rechazando_version_float() -> None:
    with pytest.raises(StrictYAMLError):
        load_strict_yaml("version: 0.1\n")


def test_legacy_compatibility_no_puede_activarse_por_el_documento() -> None:
    with pytest.raises(StrictYAMLError):
        load_legacy_rule_shadow("legacy: true\nversion: 0.1\n")


@pytest.mark.parametrize("value", ["yes", "no", "Yes", "ON", "off"])
def test_yaml11_boolean_aliases_rechazan(value: str) -> None:
    with pytest.raises(StrictYAMLError):
        load_strict_yaml(f"value: {value}\n")


def test_booleanos_solo_true_false() -> None:
    assert load_strict_yaml("enabled: true\ndisabled: false\n") == {
        "enabled": True,
        "disabled": False,
    }


def test_null_solo_lowercase_null() -> None:
    assert load_strict_yaml("value: null\n") == {"value": None}
    for value in ("Null", "~"):
        with pytest.raises(StrictYAMLError):
            load_strict_yaml(f"value: {value}\n")


@pytest.mark.parametrize(
    ("lexeme", "expected"),
    [
        ("0", 0),
        ("1", 1),
        ("-1", -1),
        ("+1", 1),
        ("9223372036854775807", 9223372036854775807),
        ("-9223372036854775808", -9223372036854775808),
    ],
)
def test_enteros_solo_decimal_canonico(lexeme: str, expected: int) -> None:
    value = load_strict_yaml(f"value: {lexeme}\n")["value"]
    assert type(value) is int
    assert value == expected


@pytest.mark.parametrize(
    "value", ["01", "001", "0x10", "0o10", "0b10", "1_000", "1:20"]
)
def test_enteros_yaml_ambiguos_rechazan(value: str) -> None:
    with pytest.raises(StrictYAMLError):
        load_strict_yaml(f"value: {value}\n")


@pytest.mark.parametrize("value", ["9223372036854775808", "-9223372036854775809"])
def test_int64_overflow_rechaza(value: str) -> None:
    with pytest.raises(StrictYAMLError):
        load_strict_yaml(f"value: {value}\n")


@pytest.mark.parametrize(
    "value", ["0.1", ".1", "1.", "1e3", "1E3", ".inf", "-.inf", ".nan"]
)
def test_float_forms_rechazan(value: str) -> None:
    with pytest.raises(StrictYAMLError):
        load_strict_yaml(f"value: {value}\n")


@pytest.mark.parametrize("value", ["2026-09-19", "2026-09-19T10:20:30Z"])
def test_timestamp_permanece_string(value: str) -> None:
    parsed = load_strict_yaml(f"value: {value}\n")["value"]
    assert parsed == value
    assert type(parsed) is str


@pytest.mark.parametrize("value", ["yes", "Null", "01", "0.1"])
def test_quoted_scalars_ambiguos_permanecen_string(value: str) -> None:
    assert load_strict_yaml(f'value: "{value}"\n') == {"value": value}


def test_multi_document_yaml_rechaza() -> None:
    with pytest.raises(StrictYAMLError):
        load_strict_yaml("a: 1\n---\nb: 2\n")


def test_yaml_directive_rechaza_o_clasifica() -> None:
    with pytest.raises(StrictYAMLError, match="directivas YAML no permitidas"):
        load_strict_yaml("%YAML 1.2\n---\na: 1\n")


def test_NULL_uppercase_rechaza() -> None:
    with pytest.raises(StrictYAMLError):
        load_strict_yaml("value: NULL\n")


def test_negative_float_rechaza() -> None:
    with pytest.raises(StrictYAMLError):
        load_strict_yaml("value: -0.1\n")


def test_exponent_float_rechaza() -> None:
    with pytest.raises(StrictYAMLError):
        load_strict_yaml("value: 1.0e3\n")


@pytest.mark.parametrize("value", ["yes", "null", "01", "0x10", "1e3"])
def test_quoted_ambiguous_scalars_son_strings(value: str) -> None:
    assert load_strict_yaml(f'value: "{value}"\n') == {"value": value}


def test_unicode_equivalent_mapping_keys_no_colisionan_silenciosamente() -> None:
    with pytest.raises(CanonicalizationError, match="duplicadas después de NFC"):
        load_strict_yaml('"Cafe\\u0301": 1\n"Caf\\u00e9": 2\n')


@pytest.mark.parametrize(
    "document",
    [
        "outer:\n  key: 1\n  key: 2\n",
        "outer:\n  source: &shared value\n  copy: *shared\n",
        "outer:\n  value: !custom tagged\n",
    ],
)
def test_nested_duplicate_keys_aliases_custom_tags_rechazan(document: str) -> None:
    with pytest.raises(StrictYAMLError):
        load_strict_yaml(document)
