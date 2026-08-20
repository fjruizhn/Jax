"""
PRUEBA BRUTAL — Intent Envelope de LAS MANOS.

"El contrato existe cuando el sistema puede negarse a obedecerlo mal."
"Un freno sin prueba no es freno." (Principio Operativo VII)

Baja la prueba ad-hoc de la sesión del 15-jun a un archivo REPRODUCIBLE: por
cada uno de los 18 campos del contrato, una llamada sin él → debe dar 422
(rechazo estructural). Por cada condición semántica cruzada → 422. Un sobre
completo y válido → debe PASAR la puerta (200 en /plan).

Corre standalone (reporte forense legible) o bajo pytest:
    las_manos/.venv/bin/python tests/test_envelope_brutal.py
    las_manos/.venv/bin/python -m pytest tests/test_envelope_brutal.py -q

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import sys
from pathlib import Path

# LAS MANOS importa por nombre de módulo (envelope, server, ...); su raíz es
# las_manos/. La metemos al path para poder construir el TestClient.
LAS_MANOS = Path(__file__).resolve().parent.parent / "las_manos"
sys.path.insert(0, str(LAS_MANOS))

from fastapi.testclient import TestClient  # noqa: E402

import server  # noqa: E402

client = TestClient(server.app, raise_server_exceptions=False)

# Los 18 campos del contrato (Mesa, 15-jun-2026), en orden.
CAMPOS_CONTRATO = [
    "trace_id", "facet_id", "actor_type", "origin_of_authority",
    "verification_label", "intent_summary", "requested_capability",
    "target_environment", "risk_level", "memory_refs_used",
    "freshness_required", "dry_run_required", "policy_required",
    "human_gate_required", "approval_token", "rollback_plan",
    "kill_switch_scope", "fail_closed_behavior",
]


def sobre_valido() -> dict:
    """Sobre completo y bien formado: hyde, list_dir (no mutante), local.
    Pasa las dos capas → /plan devuelve 200."""
    return {
        "trace_id": "00000000-0000-0000-0000-000000000001",
        "facet_id": "hyde",
        "actor_type": "faceta",
        "origin_of_authority": "prueba brutal",
        "verification_label": "local_context",
        "intent_summary": "listar un directorio de prueba",
        "requested_capability": "list_dir",
        "target_environment": "local",
        "risk_level": "none",
        "memory_refs_used": [],
        "freshness_required": False,
        "dry_run_required": False,
        "policy_required": True,
        "human_gate_required": False,
        "approval_token": None,
        "rollback_plan": None,
        "kill_switch_scope": "none",
        "fail_closed_behavior": "abortar sin ejecutar ni dejar rastro",
        # carga operacional (no parte de los 18)
        "target_host": "127.0.0.1",
        "params": {"path": "/tmp"},
    }


def _es_rechazo_envelope(resp) -> bool:
    """422 con detalle ENVELOPE_REJECTED (estructural o semántico)."""
    if resp.status_code != 422:
        return False
    detalle = str(resp.json().get("detail", ""))
    return "ENVELOPE_REJECTED" in detalle


# ── 1) LOS 18 RECHAZOS ESTRUCTURALES (un campo faltante cada vez) ──────────
def casos_estructurales():
    for campo in CAMPOS_CONTRATO:
        sobre = sobre_valido()
        del sobre[campo]
        yield (f"sin {campo}", sobre)


# ── 2) LAS CONDICIONES SEMÁNTICAS CRUZADAS ─────────────────────────────────
def casos_semanticos():
    # cond 2 — origin_of_authority vacío
    s = sobre_valido(); s["origin_of_authority"] = "   "
    yield ("cond 2: origin_of_authority vacío", s)

    # campo 6 — intent_summary vacío
    s = sobre_valido(); s["intent_summary"] = ""
    yield ("campo 6: intent_summary vacío", s)

    # campo 18 — fail_closed_behavior vacío
    s = sobre_valido(); s["fail_closed_behavior"] = "  "
    yield ("campo 18: fail_closed_behavior vacío", s)

    # cond 4 — memoria sin procedencia
    s = sobre_valido()
    s["memory_refs_used"] = [{"id": "m1", "type": "HECHO", "has_provenance": False}]
    yield ("cond 4: memoria sin procedencia", s)

    # cond 5 — prod + mutante + human_gate_required=false
    s = sobre_valido()
    s.update(requested_capability="ssh_exec", target_environment="prod",
             human_gate_required=False, target_host="203.0.113.10",
             kill_switch_scope="per_operation", rollback_plan="restaurar backup")
    yield ("cond 5: prod+mutante+sin human gate", s)

    # campo 15 — human_gate_required=true sin approval_token
    s = sobre_valido()
    s.update(human_gate_required=True, approval_token=None)
    yield ("campo 15: human gate sin token", s)

    # cond 8 — operación mutante con kill_switch_scope=none
    s = sobre_valido()
    s.update(requested_capability="ssh_exec", kill_switch_scope="none",
             params={"command": "echo hola"})
    yield ("cond 8: mutante con kill_switch=none", s)

    # cond 9 — mutante en prod sin rollback_plan
    s = sobre_valido()
    s.update(requested_capability="ssh_exec", target_environment="prod",
             target_host="203.0.113.10", human_gate_required=True,
             approval_token="tok-123", kill_switch_scope="per_operation",
             rollback_plan=None, params={"command": "echo hola"})
    yield ("cond 9: mutante en prod sin rollback", s)


def correr():
    rechazos_ok = 0
    fallos = []
    print("=" * 64)
    print("PRUEBA BRUTAL — INTENT ENVELOPE (LAS MANOS)")
    print("Cada llamada incompleta o incoherente DEBE dar 422.")
    print("=" * 64)

    print("\n── 18 RECHAZOS ESTRUCTURALES (capa Pydantic) ──")
    for i, (nombre, sobre) in enumerate(casos_estructurales(), 1):
        r = client.post("/plan", json=sobre)
        ok = r.status_code == 422
        rechazos_ok += ok
        fallos += [] if ok else [(nombre, r.status_code)]
        print(f"  {i:2d}. {nombre:<28} → {r.status_code} {'✓' if ok else '✗ ESPERABA 422'}")

    print("\n── RECHAZOS SEMÁNTICOS (capa envelope.validate) ──")
    for nombre, sobre in casos_semanticos():
        r = client.post("/plan", json=sobre)
        ok = _es_rechazo_envelope(r)
        rechazos_ok += ok
        fallos += [] if ok else [(nombre, r.status_code)]
        razon = r.json().get("detail", "") if r.status_code == 422 else ""
        print(f"  • {nombre:<34} → {r.status_code} {'✓' if ok else '✗'}")
        if ok:
            print(f"       {razon}")

    print("\n── PASE VÁLIDO (sobre completo y coherente) ──")
    r = client.post("/plan", json=sobre_valido())
    pase_ok = r.status_code == 200 and "plan" in r.json()
    if not pase_ok:
        fallos.append(("pase válido", r.status_code))
    print(f"  • sobre válido /plan → {r.status_code} "
          f"{'✓ pasó la puerta' if pase_ok else '✗ NO pasó'}")

    total_rechazos = len(CAMPOS_CONTRATO) + 8
    print("\n" + "=" * 64)
    print(f"RESULTADO: {rechazos_ok}/{total_rechazos} rechazos correctos "
          f"+ pase válido {'OK' if pase_ok else 'FALLÓ'}")
    if fallos:
        print("FALLOS:")
        for nombre, code in fallos:
            print(f"   ✗ {nombre} (devolvió {code})")
        print("=" * 64)
        return False
    print("LAS MANOS puede negarse a obedecer mal. El contrato existe.")
    print("=" * 64)
    return True


# ── pytest hooks ───────────────────────────────────────────────────────────
def test_18_rechazos_estructurales():
    for nombre, sobre in casos_estructurales():
        r = client.post("/plan", json=sobre)
        assert r.status_code == 422, f"{nombre} no fue rechazado: {r.status_code}"


def test_rechazos_semanticos():
    for nombre, sobre in casos_semanticos():
        r = client.post("/plan", json=sobre)
        assert _es_rechazo_envelope(r), f"{nombre} no fue rechazado: {r.status_code}"


def test_sobre_valido_pasa():
    r = client.post("/plan", json=sobre_valido())
    assert r.status_code == 200 and "plan" in r.json(), r.text


if __name__ == "__main__":
    sys.exit(0 if correr() else 1)
