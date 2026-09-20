from __future__ import annotations

from policy.canonicalization.canonical_json import canonical_json_bytes


def test_canonical_json_es_determinista() -> None:
    left = {"z": [3, {"b": False, "a": "Cafe\u0301"}], "a": None}
    right = {"a": None, "z": [3, {"a": "Caf\u00e9", "b": False}]}
    expected = b'{"a":null,"z":[3,{"a":"Caf\\u00e9","b":false}]}'
    assert canonical_json_bytes(left) == expected
    assert canonical_json_bytes(right) == expected
