#!/usr/bin/env python3
"""
Test: timeout por capability en jacobs.plan.PlanBuilder._from_spec.

Regresión del bug del pipeline aec827a0 (examen-v3-contrato): el step
'reconcile' moría en Timeout(300s) porque acumula el contexto de N dependencias.
El fix asigna un default mayor a 'reconcile' SIN tocar el resto.

Ronda 4 (2026-08-20, T2.a) -- dos cambios mas, con evidencia real
(jacobs_steps.started_at/finished_at, no supuesta):
  1. 'design'/'reason' suben a 900 tambien -- jacobs_steps mostro 1 step
     'design' (de 36) y 1 'reason' (de 6) fallando de verdad, exacto en el
     techo de 300s (dur=300.0s=timeout_seconds, status=failed). El comentario
     viejo de plan.py decia "completan holgados en ~50-130s" -- cierto para
     la mayoria, pero no una garantia; ya habia evidencia de lo contrario.
  2. Caso nuevo: 'timeout_seconds ausente via dict con clave=None' -- antes
     de este fix, StepSpec.timeout_seconds tenia default=300 a nivel
     Pydantic, y routes.py arma specs con StepSpec.model_dump() (que SIEMPRE
     incluye la clave). _from_spec() usaba `spec.get("timeout_seconds",
     default)`, que devuelve el valor de la clave si esta PRESENTE aunque
     sea None -- rompiendo el default por-capability en cualquier pipeline
     de steps explicitos. Confirmado en produccion: 'reconcile' aparecia con
     timeout_seconds=300 en 1 de 3 corridas reales pese a estar en el dict
     con valor 900. Este test simula exactamente ese dict (clave presente,
     valor None) para que no vuelva a pasar sin que este test lo agarre.

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
        {"facet": "ada",  "capability": "reason",                "depends_on": []},
        {"facet": "ada",  "capability": "reconcile",            "depends_on": [0]},
        {"facet": "ada",  "capability": "assemble",             "depends_on": [0, 1]},
        {"facet": "thot", "capability": "validate_consistency", "depends_on": [0]},
        {"facet": "ada",  "capability": "reconcile", "timeout_seconds": 1200, "depends_on": [0]},
        # simula StepSpec.model_dump() de un caller que nunca toco el campo
        {"facet": "ada",  "capability": "reconcile", "timeout_seconds": None, "depends_on": [0]},
    ])
    checks = [
        ("design sube a 900 (evidencia real, ronda 4)",     steps[0].timeout_seconds, 900),
        ("reason sube a 900 (evidencia real, ronda 4)",     steps[1].timeout_seconds, 900),
        ("reconcile sube a 900",                            steps[2].timeout_seconds, 900),
        ("assemble (mecánico) queda en 300",                steps[3].timeout_seconds, 300),
        ("validate_consistency queda en 300",               steps[4].timeout_seconds, 300),
        ("timeout_seconds explícito gana",                  steps[5].timeout_seconds, 1200),
        ("timeout_seconds=None (clave presente) NO pisa el default de capability",
                                                             steps[6].timeout_seconds, 900),
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
