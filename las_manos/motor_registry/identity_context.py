"""
REFORMAS-v3.md R3.5 — identidad inyectada. Cada motor recibe, antes de su
prompt real: quién es, qué capabilities tiene en esta tarea, qué motores
existen y qué puede cada uno, la lista de predicados emitibles, y el
protocolo de rechazo tipado (CAPABILITY_UNBOUND).

Esto es SOLO contexto/plumbing (Fase 1) — no valida ni fuerza que el motor
lo respete. Esa validación (canal claim/analysis/judgment, barrido de
vocabulario) es R1, Fase 2, fuera de alcance acá.
"""
from __future__ import annotations


def build_identity_context(
    motor_name: str,
    capabilities: list[str],
    catalog: dict,
    predicates: list[str],
    task_id: str,
) -> str:
    other_motors = "\n".join(
        f"  - {name}: {', '.join(info.get('allowed_motors_for', []))}"
        for name, info in catalog.items()
        if name != motor_name
    )
    # Con catálogo vacío/sin otras entradas, un header seguido de nada lee
    # como "no hay otros motores" — falso (ej. ada existe y está habilitado).
    # Placeholder honesto en vez de una sección vacía.
    other_motors_line = (
        f"Otros motores del ecosistema y qué pueden hacer:\n{other_motors}"
        if other_motors
        else "Otros motores del ecosistema: (catálogo no disponible en este contexto)."
    )
    return (
        f"[IDENTIDAD — REFORMAS-v3 R3.5]\n"
        f"Sos el motor '{motor_name}'. Task id: {task_id}.\n"
        f"Capabilities otorgadas para esta tarea: {', '.join(capabilities)}.\n"
        f"{other_motors_line}\n"
        f"Predicados emitibles en canal claim (lista cerrada, §3.1.3): "
        f"{', '.join(predicates)}.\n"
        f"Si necesitás una capability no otorgada, el rechazo tipado es "
        f"CAPABILITY_UNBOUND — no lo simules, no lo inventes en tu salida.\n"
    )
