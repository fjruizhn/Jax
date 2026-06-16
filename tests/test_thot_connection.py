"""
CONEXIÓN DE THOT — los 7 criterios de activación (Mesa, 15-jun-2026).

Thot es la PRIMERA faceta del orden firmado (Thot → Hipatia → Jekyll → Hyde).
Modo auditoría: lee, cuestiona, reporta. NO ejecuta, NO aprueba, NO muta.

Cada criterio se demuestra contra el servidor real (TestClient in-process,
escribe en el audit log real). No se reporta éxito por suposición.

    las_manos/.venv/bin/python tests/test_thot_connection.py
    las_manos/.venv/bin/python -m pytest tests/test_thot_connection.py -q

LÍMITE DECLARADO (Protocolo Hyde): esto conecta el CABLE de Thot a LAS MANOS y
demuestra que la puerta lo trata como auditor read-only. NO conecta el LLM
GPT-5.5 de Thot (no hay API key de OpenAI en el entorno). El criterio 6
("Thot audita y reporta") se demuestra con la revisión DETERMINISTA del cable;
el juicio del LLM queda pendiente de esa decisión/credencial.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import sys
from pathlib import Path

LAS_MANOS = Path(__file__).resolve().parent.parent / "las_manos"
sys.path.insert(0, str(LAS_MANOS))

from fastapi.testclient import TestClient  # noqa: E402

import server  # noqa: E402
from facet_client import FacetClient, FacetRefusal, review_audit  # noqa: E402

ORIGIN = "auditoría de rutina de Thot (Mesa)"
REVIEW_FILE = Path(__file__).resolve().parent.parent / "missions" / "thot-review-001.md"


def _client():
    transport = TestClient(server.app, raise_server_exceptions=False)
    return FacetClient.for_thot(transport)


def _events_for(transport, request_id: str, n: int = 200) -> list:
    """Lee el audit log y filtra los eventos de un request_id."""
    r = transport.get(f"/audit/tail?n={n}")
    eventos = r.json().get("events", [])
    return [e for e in eventos if e.get("request_id") == request_id]


# ── Criterio 1 — primera llamada exitosa ────────────────────────────────────
def criterio_1(thot):
    r = thot.read_audit_log(n=20, origin=ORIGIN)
    ok = r.ok and r.body.get("result", {}).get("success") is True
    return ok, r


# ── Criterio 2 — primera llamada rechazada correctamente (sobre incoherente) ─
def criterio_2(thot):
    # Thot pide algo que le toca (audit_log_read) pero SIN declarar autoridad:
    # sobre incoherente → la puerta lo rechaza (422 semántico).
    env = thot.build_envelope(
        capability="audit_log_read",
        origin_of_authority="   ",  # vacío → condición 2
        intent_summary="leer el log",
        params={"n": 5},
    )
    r = thot._post("/execute", env)
    return r.rejected_envelope, r


# ── Criterio 3 — dry-run en read-only (¿aplica?) ────────────────────────────
def criterio_3(eventos_lectura):
    # Las operaciones de lectura ignoran dry_run (dispatch) y requires_dryrun=
    # false para reads. Demostración HONESTA: la cadena de una lectura NO tiene
    # evento DRYRUN. No aplica por diseño — una lectura no muta, nada que simular.
    hay_dryrun = any(e.get("event") == "DRYRUN" for e in eventos_lectura)
    return (not hay_dryrun)


# ── Criterio 4 — fallo cerrado: Thot intenta MUTAR → LAS MANOS lo frena ──────
def criterio_4(thot):
    # (a) El cliente se niega solo (defensa en profundidad).
    cliente_se_nego = False
    try:
        thot.call(capability="write_file", origin_of_authority=ORIGIN,
                  intent_summary="intentar escribir", params={"path": "/tmp/x", "content": "x"})
    except FacetRefusal:
        cliente_se_nego = True

    # (b) Saltamos el guardrail del cliente y mandamos el sobre mutante crudo,
    #     para probar que la PARED es el servidor: policy.check → 403.
    env = thot.build_envelope(
        capability="write_file",
        origin_of_authority=ORIGIN,
        intent_summary="Thot intenta escribir un archivo (no le toca)",
        target_host="172.16.20.11",   # staging (Thot puede 'estar' pero no mutar)
        target_environment="staging",
        params={"path": "/tmp/thot_intruso", "content": "no debería pasar"},
    )
    r = thot._post("/execute", env)
    servidor_lo_freno = r.blocked_by_policy  # 403
    return (cliente_se_nego and servidor_lo_freno), r


# ── Criterio 5 — primer audit log completo (cadena forense de la llamada 1) ──
def criterio_5(eventos_lectura):
    tipos = {e.get("event") for e in eventos_lectura}
    completo = {"ENVELOPE_ACCEPTED", "POLICY_CHECK", "EXECUTION"} <= tipos
    return completo, tipos


# ── Criterio 6 — primera revisión por Thot (su función real) ────────────────
def criterio_6(thot):
    r = thot.read_audit_log(n=100, origin="revisión forense de Thot")
    eventos = r.body.get("result", {}).get("events", [])
    review = review_audit(eventos)
    # Una revisión válida cuenta eventos y detecta los rechazos que provocamos.
    hizo_revision = review.total_events > 0 and "ENVELOPE_REJECTED" in review.by_event
    return hizo_revision, review


# ── Criterio 7 — primera corrección documentada ─────────────────────────────
def criterio_7(review):
    # Thot encontró algo → se registra (se preserva, no se borra).
    contenido = (
        "# Revisión de Thot 001 — primera auditoría vía LAS MANOS\n\n"
        "faceta: thot · modo: auditoría (read-only) · vía: Intent Envelope\n\n"
        "LÍMITE: revisión determinista del cable. El juicio del LLM GPT-5.5 de "
        "Thot queda pendiente de API key de OpenAI.\n\n"
        "```\n" + review.render() + "\n```\n"
    )
    REVIEW_FILE.write_text(contenido, encoding="utf-8")
    return REVIEW_FILE.exists() and "REVISIÓN DE THOT" in REVIEW_FILE.read_text(encoding="utf-8")


def correr():
    thot = _client()
    transport = thot.transport
    print("=" * 66)
    print("CONEXIÓN DE THOT — 7 criterios de activación (modo auditoría)")
    print("=" * 66)
    fallos = []

    # 1
    ok1, r1 = criterio_1(thot)
    req_id = r1.body.get("request_id")
    print(f"\n1. Primera llamada exitosa (Thot lee el log vía Envelope)")
    print(f"   → HTTP {r1.status_code}, request_id={req_id} {'✓' if ok1 else '✗'}")
    fallos += [] if ok1 else ["1"]

    # 2
    ok2, r2 = criterio_2(thot)
    print(f"\n2. Primera llamada rechazada (sobre sin autoridad declarada)")
    print(f"   → HTTP {r2.status_code}: {r2.body.get('detail')} {'✓' if ok2 else '✗'}")
    fallos += [] if ok2 else ["2"]

    # Cadena forense de la llamada 1 (para criterios 3 y 5).
    eventos_1 = _events_for(transport, req_id)

    # 3
    ok3 = criterio_3(eventos_1)
    print(f"\n3. Dry-run en read-only")
    print(f"   → la lectura NO generó evento DRYRUN: {'✓' if ok3 else '✗'} "
          f"(no aplica a read-only por diseño — nada que pre-simular)")
    fallos += [] if ok3 else ["3"]

    # 4
    ok4, r4 = criterio_4(thot)
    print(f"\n4. Fallo cerrado: Thot intenta MUTAR (write_file)")
    print(f"   → cliente se niega (FacetRefusal) Y servidor responde "
          f"HTTP {r4.status_code}")
    print(f"     policy: {r4.body.get('detail')} {'✓ LAS MANOS lo frenó' if ok4 else '✗'}")
    fallos += [] if ok4 else ["4"]

    # 5
    ok5, tipos = criterio_5(eventos_1)
    print(f"\n5. Primer audit log completo (cadena de la llamada 1, req {req_id})")
    print(f"   → eventos: {sorted(tipos)} {'✓' if ok5 else '✗'}")
    fallos += [] if ok5 else ["5"]

    # 6
    ok6, review = criterio_6(thot)
    print(f"\n6. Primera revisión por Thot (función real, DETERMINISTA)")
    for ln in review.render().splitlines():
        print(f"   {ln}")
    print(f"   {'✓' if ok6 else '✗'}")
    fallos += [] if ok6 else ["6"]

    # 7
    ok7 = criterio_7(review)
    print(f"\n7. Primera corrección documentada")
    print(f"   → revisión preservada en {REVIEW_FILE.relative_to(LAS_MANOS.parent)} "
          f"{'✓' if ok7 else '✗'}")
    fallos += [] if ok7 else ["7"]

    print("\n" + "=" * 66)
    if fallos:
        print(f"FALLARON LOS CRITERIOS: {', '.join(fallos)}")
        print("=" * 66)
        return False
    print("THOT CONECTADO — 7/7 criterios demostrados, no supuestos.")
    print("Modo auditoría read-only. LAS MANOS lo deja leer y lo frena al mutar.")
    print("PENDIENTE declarado: cablear el LLM GPT-5.5 (falta API key OpenAI).")
    print("=" * 66)
    return True


# ── pytest hooks ─────────────────────────────────────────────────────────────
def test_criterio_1_llamada_exitosa():
    thot = _client()
    ok, r = criterio_1(thot)
    assert ok, r.body


def test_criterio_2_rechazo_correcto():
    thot = _client()
    ok, r = criterio_2(thot)
    assert ok, r.body


def test_criterio_4_thot_no_puede_mutar():
    thot = _client()
    ok, r = criterio_4(thot)
    assert ok, f"LAS MANOS no frenó la mutación de Thot: {r.status_code} {r.body}"


def test_criterio_5_audit_completo():
    thot = _client()
    _, r = criterio_1(thot)
    eventos = _events_for(thot.transport, r.body.get("request_id"))
    ok, tipos = criterio_5(eventos)
    assert ok, f"cadena forense incompleta: {tipos}"


if __name__ == "__main__":
    sys.exit(0 if correr() else 1)
