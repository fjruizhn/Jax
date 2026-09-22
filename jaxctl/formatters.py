from __future__ import annotations
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum

def _value(value):
    if is_dataclass(value): return {key:_value(item) for key,item in asdict(value).items()}
    if isinstance(value, Enum): return value.value
    if isinstance(value, datetime): return value.isoformat().replace("+00:00", "Z")
    if isinstance(value, tuple): return [_value(item) for item in value]
    if isinstance(value, dict): return {key:_value(item) for key,item in value.items()}
    return value

def json_output(value): return json.dumps(_value(value), sort_keys=True, separators=(",", ":"))
def human_output(value): return json.dumps(_value(value), sort_keys=True, indent=2)
