#!/usr/bin/env python3
"""
Test: timeout por capability en jacobs.plan.PlanBuilder._from_spec.

Regresión del bug del pipeline aec827a0 (examen-v3-contrato): el step
'reconcile' moría en Timeout(300s) porque acumula el contexto de N dependencias.
El fix asigna un default mayor a 'reconcile' SIN tocar el resto.

Standalone (no requiere pytest):  python tests/test_jacobs_timeout_by_capability.py
"""
import sys
import os

sys.path.insert(0, os.path.expanduser("~/jax"))
from jacobs.plan import PlanBuilder


def main() -> int:
    pb = PlanBuilder()
    steps = pb._from_spec("pid-test", [
        {"facet": "ada",  "capability": "design",               "depends_on": []},
        {"facet": "ada",  "capability": "reconcile",            "depends_on": [0]},
        {"facet": "ada",  "capability": "assemble",             "depends_on": [0, 1]},
        {"facet": "thot", "capability": "validate_consistency", "depends_on": [0]},
        {"facet": "ada",  "capability": "reconcile", "timeout_seconds": 1200, "depends_on": [0]},
    ])
    checks = [
        ("design queda en 300",              steps[0].timeout_seconds, 300),
        ("reconcile sube a 900",             steps[1].timeout_seconds, 900),
        ("assemble (mecánico) queda en 300", steps[2].timeout_seconds, 300),
        ("validate_consistency queda en 300", steps[3].timeout_seconds, 300),
        ("timeout_seconds explícito gana",   steps[4].timeout_seconds, 1200),
    ]
    ok = True
    for name, got, want in checks:
        status = "PASS" if got == want else "FAIL"
        if got != want:
            ok = False
        print(f"  [{status}] {name}: got={got} want={want}")
    print("ALL PASS" if ok else "TESTS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
