"""
PROBE ONLINE — Thot invoca GPT-5.5 real sobre el audit log.

NO es un test offline: hace una llamada EXTERNA a la API de OpenAI (cuesta y
sale a la red). Verifica el punto 2 del encargo de Fernando: que OPENAI_API_KEY
esté cargada y que Thot devuelva una observación REAL, no la revisión mecánica.

Arranque (igual que JAX — la key se sourcea, no se hardcodea):
    cd ~/jax && set -a; source /etc/jax/.env; set +a
    las_manos/.venv/bin/python tests/_probe_thot_llm.py

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import asyncio
import os
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))                       # para 'import jax.*'
sys.path.insert(0, str(ROOT / "las_manos"))         # para 'import server/facet_client'

from fastapi.testclient import TestClient  # noqa: E402
import server  # noqa: E402
from facet_client import FacetClient  # noqa: E402
from jax.muscles.base import HttpMuscle, MuscleError  # noqa: E402


def construir_thot() -> HttpMuscle:
    """Instancia a Thot tal como lo haría build_muscles() de main.py."""
    cfg = tomllib.loads((ROOT / "config" / "config.toml").read_text(encoding="utf-8"))
    p = cfg["personalities"]["thot"]
    return HttpMuscle(
        "thot", p["provider"], p["model_default"], p["models_allowed"],
        p["system_prompt"], timeout=90.0,
        grounding_policy=p.get("grounding_policy", "off"),
        authority_origin=p.get("authority_origin", ""),
    )


def leer_audit_real(n: int = 30) -> list:
    """Thot lee el log forense por el cable, a través de la puerta (200)."""
    thot_cable = FacetClient.for_thot(TestClient(server.app, raise_server_exceptions=False))
    r = thot_cable.read_audit_log(n=n, origin="probe: primera observación con LLM real")
    assert r.ok, f"la lectura por el cable falló: {r.status_code} {r.body}"
    return r.body.get("result", {}).get("events", [])


async def main() -> int:
    print("=" * 66)
    print("PROBE — Thot + GPT-5.5 real sobre el audit log")
    print("=" * 66)

    if not os.environ.get("OPENAI_API_KEY"):
        print("✗ OPENAI_API_KEY NO está en el entorno.")
        print("  Arrancá con: set -a; source /etc/jax/.env; set +a  (como JAX).")
        return 1
    print("✓ OPENAI_API_KEY presente en el entorno (no se imprime su valor).")

    eventos = leer_audit_real(30)
    conteo: dict = {}
    for e in eventos:
        conteo[e.get("event", "?")] = conteo.get(e.get("event", "?"), 0) + 1
    print(f"✓ Thot leyó {len(eventos)} eventos por el cable. Resumen: {conteo}")

    thot = construir_thot()
    print(f"✓ Thot instanciado: provider={thot.provider}, modelo={thot.model_default}")

    prompt = (
        "Sos Thot, auditor del ecosistema. Acabás de leer el audit log de LAS MANOS "
        "(el sistema nervioso inhibitorio de JAX). Este es el conteo de eventos "
        f"de los últimos {len(eventos)} registros:\n\n{conteo}\n\n"
        "Hay un evento ENVELOPE_REJECTED por cada llamada que la puerta rechazó por "
        "incompleta o incoherente, y POLICY_CHECK por cada decisión de autorización. "
        "Dame UNA observación de auditor, concreta y breve (2-3 frases): ¿qué te dice "
        "este patrón sobre la salud del sistema? Si algo te llama la atención, decilo."
    )

    print("\n— Invocando a GPT-5.5 real (puede tardar)… —\n")
    try:
        observacion = await thot.invoke(prompt)
    except MuscleError as e:
        print(f"✗ La invocación falló: {e}")
        return 2

    print("OBSERVACIÓN DE THOT (GPT-5.5 real, no mecánica):")
    print("-" * 66)
    print(observacion)
    print("-" * 66)
    print("\n✓ Thot invocó GPT-5.5 real y devolvió una observación.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
