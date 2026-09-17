"""Frente E de la auditoría de jax (2026-09-16): lo que se RETIRA queda retirado.

Cada test falla contra el código anterior al retiro (visto en rojo antes de
borrar) y cita el id del hallazgo. Spec:
docs/superpowers/specs/2026-09-16-hallazgos-auditoria-jax-design.md §E.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import ast
import inspect
import os
import typing
from pathlib import Path

os.environ["JAX_DB_NAME"] = "jax_memory_test"  # barrera: nada acá abre conexión, y si algo se escapa no es producción

RAIZ = Path(__file__).resolve().parents[1]


def _importados(rel: str) -> set[str]:
    arbol = ast.parse((RAIZ / rel).read_text(encoding="utf-8"))
    nombres: set[str] = set()
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Import):
            nombres.update(a.asname or a.name.split(".")[0] for a in nodo.names)
        elif isinstance(nodo, ast.ImportFrom):
            nombres.update(a.asname or a.name for a in nodo.names)
    return nombres


def test_E02_stepresult_ya_no_existe():
    from jacobs import models
    assert not hasattr(models, "StepResult")


def test_E04_traffic_classes_ya_no_existe_y_el_literal_del_envelope_manda():
    import audit
    from envelope import IntentEnvelope
    assert not hasattr(audit, "TRAFFIC_CLASSES")
    valores = set(typing.get_args(IntentEnvelope.model_fields["traffic_class"].annotation))
    assert valores == {"test_structural", "test_semantic", "dry_run", "production", "adversarial_test", "unknown"}


def test_E05_enabled_motors_ya_no_existe():
    from motor_registry.catalog import MotorCatalog
    assert not hasattr(MotorCatalog, "enabled_motors")


def test_E06_las_voces_piper_ya_no_estan_en_el_arbol():
    assert sorted((RAIZ / "voices").glob("*.onnx*")) == []


_IMPORTS_RETIRADOS = {
    "jacobs/executor.py": {"Any", "resolve_credential_instrumented", "CredentialUnavailableError", "FacetUnavailableError"},
    "jax/muscles/base.py": {"os", "decrypt_secret"},
    "jax/muscles/ollama_muscle.py": {"MuscleTimeoutError"},
    "las_manos/audit.py": {"os"},
    "las_manos/policy.py": {"re"},
    "las_manos/server.py": {"uuid"},
}


def test_E09_imports_sin_uso_retirados():
    sobreviven = {rel: sorted(nombres & _importados(rel)) for rel, nombres in _IMPORTS_RETIRADOS.items()}
    assert {rel: n for rel, n in sobreviven.items() if n} == {}


def test_E20_el_gate_de_hyde_es_el_de_la_capability_y_no_una_funcion_que_devuelve_true():
    from jacobs import policy, store
    assert not hasattr(policy, "hyde_requires_human_gate")
    assert "requires_human_gate" in inspect.getsource(store.get_motor_governance)


def test_E07_cleanup_sh_retirado():
    """No tenía scheduler (crontab y systemd sin menciones, 2026-09-16) y, si
    alguien lo corría a mano, borraba *.backup* de un home que no está en restic."""
    assert not (RAIZ / "scripts" / "cleanup.sh").exists()
