"""
THOT AUDIT WATCH — el log distingue tráfico real de tráfico de pruebas.

Mesa, 16-jun-2026: antes de activar a Thot como vigía del audit log, LAS MANOS
debe poder separar lo REAL de lo SINTÉTICO. Tres campos forenses en cada evento:

    environment    — INFERIDO del target_environment del sobre (fail-safe: production)
    traffic_class  — DECLARADO por la faceta (facet_client → "production"; tests, explícito)
    test_run_id    — opcional, agrupa una corrida de prueba

Esta prueba demuestra —contra el servidor real, escribiendo en el audit log
real— que un evento de prueba (traffic_class="test_structural") y un evento
real de Thot (traffic_class="production") quedan DISTINGUIBLES en el log.

    las_manos/.venv/bin/python tests/test_audit_traffic_class.py
    las_manos/.venv/bin/python -m pytest tests/test_audit_traffic_class.py -q

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import sys
from pathlib import Path

LAS_MANOS = Path(__file__).resolve().parent.parent / "las_manos"
sys.path.insert(0, str(LAS_MANOS))

from fastapi.testclient import TestClient  # noqa: E402

import server  # noqa: E402
from facet_client import FacetClient  # noqa: E402

# Un id de corrida fijo para reencontrar exactamente NUESTROS eventos en el log.
RUN_ID = "audit-watch-proof-001"


def _client() -> FacetClient:
    return FacetClient.for_thot(TestClient(server.app, raise_server_exceptions=False))


def _events_for(transport, request_id: str, n: int = 300) -> list:
    eventos = transport.get(f"/audit/tail?n={n}").json().get("events", [])
    return [e for e in eventos if e.get("request_id") == request_id]


def _emit_real(thot: FacetClient):
    """Lectura REAL de Thot por el cable: traffic_class default = 'production'."""
    return thot.read_audit_log(n=3, origin="auditoría real de Thot (proof)")


def _emit_test(thot: FacetClient):
    """Misma capacidad, pero declarada como tráfico de PRUEBA estructural,
    contra prod para probar de paso la inferencia de environment."""
    return thot.call(
        capability="audit_log_read",
        origin_of_authority="suite de pruebas (proof)",
        intent_summary="evento sintético para validar la clasificación forense",
        target_environment="prod",
        params={"n": 3},
        traffic_class="test_structural",
        test_run_id=RUN_ID,
    )


def correr() -> bool:
    print("=" * 66)
    print("THOT AUDIT WATCH — real vs prueba en el audit log")
    print("=" * 66)
    thot = _client()

    r_real = _emit_real(thot)
    r_test = _emit_test(thot)
    assert r_real.ok, f"lectura real falló: {r_real.status_code} {r_real.body}"
    assert r_test.ok, f"lectura de prueba falló: {r_test.status_code} {r_test.body}"

    ev_real = _events_for(thot.transport, r_real.body["request_id"])
    ev_test = _events_for(thot.transport, r_test.body["request_id"])
    assert ev_real, "no se encontraron eventos del request real"
    assert ev_test, "no se encontraron eventos del request de prueba"

    tc_real = {e.get("traffic_class") for e in ev_real}
    tc_test = {e.get("traffic_class") for e in ev_test}
    env_real = {e.get("environment") for e in ev_real}
    env_test = {e.get("environment") for e in ev_test}
    run_test = {e.get("test_run_id") for e in ev_test}

    print(f"  REAL  → traffic_class={tc_real} environment={env_real}")
    print(f"  TEST  → traffic_class={tc_test} environment={env_test} test_run_id={run_test}")

    fallos = []
    # 1) El tráfico real se declara production.
    if tc_real != {"production"}:
        fallos.append(f"el tráfico real no es production: {tc_real}")
    # 2) El tráfico de prueba se declara test_structural.
    if tc_test != {"test_structural"}:
        fallos.append(f"el tráfico de prueba no es test_structural: {tc_test}")
    # 3) Son DISTINGUIBLES — la condición que pidió la Mesa.
    if tc_real & tc_test:
        fallos.append("real y prueba comparten traffic_class: no se distinguen")
    # 4) environment se infiere del target: real (local)→test, prueba (prod)→production.
    if env_real != {"test"}:
        fallos.append(f"environment real (target local) no infirió test: {env_real}")
    if env_test != {"production"}:
        fallos.append(f"environment prueba (target prod) no infirió production: {env_test}")
    # 5) test_run_id agrupa solo la corrida de prueba; el real lo deja en null.
    if run_test != {RUN_ID}:
        fallos.append(f"test_run_id de la prueba no agrupó: {run_test}")
    if any(e.get("test_run_id") is not None for e in ev_real):
        fallos.append("el tráfico real trae test_run_id (debería ser null)")

    print("=" * 66)
    if fallos:
        for f in fallos:
            print(f"  ✗ {f}")
        print("FALLÓ: el log NO distingue tráfico real de prueba.")
        print("=" * 66)
        return False
    print("✓ El audit log distingue tráfico real (production) de prueba (test_structural).")
    print("✓ environment se infiere del sobre; test_run_id agrupa solo la prueba.")
    print("THOT AUDIT WATCH listo para activarse: el vigía ve lo que es real.")
    print("=" * 66)
    return True


# ── pytest hooks ─────────────────────────────────────────────────────────────
def test_real_vs_prueba_se_distinguen():
    assert correr()


if __name__ == "__main__":
    sys.exit(0 if correr() else 1)
