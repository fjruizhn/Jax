"""Barrido de la entrega (spec 2026-09-28 §3.3.2): secretos en lo agregado y tope por archivo."""
from __future__ import annotations

import re

_SECRETOS = re.compile(r"(github_pat_[A-Za-z0-9_]{20,}|ghp_[A-Za-z0-9]{30,}|sk-ant-[a-z0-9]+-[A-Za-z0-9_-]{20,}|"
                       r"AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----)")
