"""Micro-medicion SIN perfilador del costo de `revalidate_for_transport` (2026-10-02).

El perfil cProfile de la carga del chat F2-D (docs/carga-chat-f2d-2026-10-02.md) mostro que
casi todo el CPU del backend se va en `GovernedRenderer.render_text` del nucleo JAX, que
`revalidate_for_transport` vuelve a ejecutar sobre el texto COMPLETO cada vez que se llama
(5 veces por turno en el camino F2-D). Este script lo mide aislado y sin el perfilador
(que infla el codigo Python puro), para que el numero no dependa del instrumento.

NO toca ninguna base ni ningun puerto: solo importa el nucleo JAX en solo lectura y llama
funciones puras en proceso.

USO:
    JAX_REPO_PATH=/srv/jax-prod/jax PYTHONPATH=backend python3 loadtest/chat_f2d_micro_revalidate.py [chars]
"""
from __future__ import annotations

import os
import statistics
import sys
import time
from datetime import datetime, timezone

os.environ.setdefault("JAX_REPO_PATH", "/srv/jax-prod/jax")
sys.path.insert(0, os.environ["JAX_REPO_PATH"])

from api import governed_chat  # noqa: E402  (backend/ en PYTHONPATH)
from jax.memory.b9 import ScopeContext  # noqa: E402


class _Contrato:  # lo minimo que project_provider_contract lee de ContractResult
    contract_parsed = True
    claims: list = []
    judgment = None

    def __init__(self, analysis: str):
        self.analysis = analysis


def main() -> None:
    chars = int(sys.argv[1]) if len(sys.argv) > 1 else 16_000
    base = "Respuesta de carga sobre planificacion y presupuesto del trimestre. "
    texto = (base * (chars // len(base) + 1))[:chars]
    scope = ScopeContext(actor_principal="user:1", actor_type="USER", subject_user_id="1",
                         tenant_id="1", project_id=None, calling_component="jax-platform-web-chat")
    lifecycle = governed_chat._lifecycle_core()

    t0 = time.perf_counter()
    proy = governed_chat.project_provider_contract(_Contrato(texto), memory_scope=scope, user_id="1")
    t_proy = (time.perf_counter() - t0) * 1000
    if proy.transport_unit is None:
        raise SystemExit("la proyeccion no produjo unidad de transporte (nucleo no disponible?)")

    muestras = []
    for _ in range(200):
        t0 = time.perf_counter()
        lifecycle.revalidate_for_transport(proy.transport_unit, datetime.now(timezone.utc))
        muestras.append((time.perf_counter() - t0) * 1000)
    muestras.sort()
    print(f"chars={chars}  project_provider_contract (1a vez, incluye import): {t_proy:.2f} ms")
    print(f"revalidate_for_transport x200: mediana {statistics.median(muestras):.2f} ms  "
          f"p95 {muestras[int(len(muestras) * 0.95)]:.2f} ms  min {muestras[0]:.2f} ms")
    print(f"=> 5 llamadas por turno ~ {5 * statistics.median(muestras):.1f} ms de CPU por turno")


if __name__ == "__main__":
    main()
