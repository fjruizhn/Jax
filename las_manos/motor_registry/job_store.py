"""
LAS MANOS — Motor Registry: almacén de jobs.

JSONL append-only: cada línea es un evento de job.
Un índice en memoria mapea job_id → estado más reciente.
El estado vigente se reconstruye del último evento almacenado.

No borramos. No editamos líneas. Solo appendamos.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from motor_registry.models import JobStatus, MotorJobView

_JOB_VIEW_FIELDS = set(MotorJobView.model_fields.keys())


class JobStore:
    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._index: dict[str, dict] = {}   # job_id → latest state dict
        self._load()

    def _load(self) -> None:
        """Reconstruye el índice desde el JSONL al arrancar."""
        if not self._path.exists():
            return
        with open(self._path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                    self._index[event["job_id"]] = event
                except (json.JSONDecodeError, KeyError):  # fail-soft: linea de log JSONL corrupta se descarta; el peor caso es un job ausente del indice (falla cerrado: caller sigue esperando), no uno que aparente exito
                    pass  # línea corrupta — ignorar silenciosamente

    def _append(self, event: dict) -> None:
        with self._lock:
            line = json.dumps(event, ensure_ascii=False) + "\n"
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(line)
            self._index[event["job_id"]] = event

    def create(
        self,
        *,
        caller: str,
        capability: str,
        motor: str,
        trace_id: str,
        prompt: str,
        recursion_depth: int,
    ) -> str:
        job_id = str(uuid.uuid4())
        event: dict[str, Any] = {
            "job_id": job_id,
            "status": JobStatus.PENDING.value,
            "motor": motor,
            "capability": capability,
            "caller": caller,
            "trace_id": trace_id,
            "recursion_depth": recursion_depth,
            "prompt": prompt,
            "created_at": time.time(),
            "started_at": None,
            "finished_at": None,
            "error": None,
            "result_summary": None,
        }
        self._append(event)
        return job_id

    def update(self, job_id: str, **kwargs: Any) -> None:
        if job_id not in self._index:
            raise KeyError(f"job_id desconocido: {job_id}")
        event = {**self._index[job_id], **kwargs, "job_id": job_id}
        self._append(event)

    def get(self, job_id: str) -> MotorJobView | None:
        state = self._index.get(job_id)
        if state is None:
            return None
        return MotorJobView(**{k: v for k, v in state.items() if k in _JOB_VIEW_FIELDS})
