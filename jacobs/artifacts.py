"""
Jacobs — Almacén de artifacts.

Outputs > 60 KB van a disco. El context del pipeline guarda solo la ref.
SIZE_LIMIT alineado con TEXT de MariaDB (~64 KB) con margen de seguridad.
En honor al Prof. Raúl Jacobs.
"""
from __future__ import annotations

import json
from pathlib import Path

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
SIZE_LIMIT = 60_000  # bytes. >60 KB → artifact en archivo (output_ref no aguanta más inline).


def save_if_large(pipeline_id: str, step_id: str, data: dict) -> tuple[str | None, dict | None]:
    """
    Si data serializado supera SIZE_LIMIT, guarda en archivo y devuelve (ref, None).
    Si cabe, devuelve (None, data) para guardarlo inline.
    """
    raw = json.dumps(data, ensure_ascii=False)
    if len(raw.encode()) <= SIZE_LIMIT:
        return None, data

    target = ARTIFACTS_DIR / pipeline_id / step_id
    target.mkdir(parents=True, exist_ok=True)
    artifact_path = target / "output.json"
    artifact_path.write_text(raw, encoding="utf-8")

    ref = f"artifact://jacobs/{pipeline_id}/{step_id}/output.json"
    return ref, None


def read_artifact(ref: str) -> dict:
    """Carga un artifact desde disco dado su ref URI."""
    # ref = artifact://jacobs/{pipeline_id}/{step_id}/output.json
    rel = ref.removeprefix("artifact://jacobs/")
    path = ARTIFACTS_DIR / rel
    return json.loads(path.read_text(encoding="utf-8"))
