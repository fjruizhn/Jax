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
import math
import os
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from pydantic import ValidationError

from motor_registry.models import JobStatus, MotorJobView

_JOB_VIEW_FIELDS = set(MotorJobView.model_fields.keys())


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class AuthoritativeJobSnapshot:
    view: MotorJobView
    observed_at: datetime


class JobStore:
    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._index: dict[str, dict] = {}   # job_id → latest state dict
        self._history_integrity = True
        self._load()

    def _load(self) -> None:
        """Reconstruye el índice desde el JSONL al arrancar."""
        if not self._path.exists():
            return
        with open(self._path, "rb") as f:
            for raw_line in f:
                if not raw_line.strip():
                    continue
                if not raw_line.endswith(b"\n"):
                    # Every store-owned append is newline-terminated. A
                    # complete-looking final JSON object can still be a
                    # partial write if its record terminator never landed.
                    self._history_integrity = False
                try:
                    line = raw_line.decode("utf-8").strip()
                except UnicodeDecodeError:
                    self._history_integrity = False
                    continue
                if not line:
                    continue
                try:
                    event = json.loads(line)
                    if (not isinstance(event, dict) or not isinstance(event.get("job_id"), str)
                            or not event["job_id"] or event.get("status") not in {s.value for s in JobStatus}):
                        self._history_integrity = False
                        continue
                    try:
                        MotorJobView(**{k: v for k, v in event.items() if k in _JOB_VIEW_FIELDS})
                    except (ValidationError, TypeError, ValueError):
                        self._history_integrity = False
                        continue
                    timestamps = ("created_at", "started_at", "finished_at", "status_updated_at")
                    if any(event.get(name) is not None and (
                            isinstance(event.get(name), bool)
                            or not isinstance(event.get(name), (int, float))
                            or not math.isfinite(event[name]) or event[name] < 0
                    ) for name in timestamps):
                        self._history_integrity = False
                        continue
                    self._index[event["job_id"]] = event
                except (json.JSONDecodeError, KeyError, TypeError):
                    # Preserve best-effort operational polling, but never let
                    # an incomplete history authorize current-truth evidence.
                    self._history_integrity = False

    def source_configuration(self) -> dict[str, str]:
        """Non-secret identity of this fixed authoritative JSONL source."""
        return {
            "store_contract": "motor-job-store-v1",
            "source_id": str(self._path.resolve()),
            "event_format": "motor-job-event-v1",
            "durability": "append-flush-fsync-v1",
        }

    @property
    def authoritative_history_intact(self) -> bool:
        with self._lock:
            return self._history_integrity

    def authoritative_snapshot(self, job_id: str) -> AuthoritativeJobSnapshot | None:
        """One locked, integrity-gated status/scope observation for F2-B."""
        with self._lock:
            if not self._history_integrity:
                return None
            state = self._index.get(job_id)
            if state is None:
                return None
            snapshot = dict(state)
            observed_at = _utc_now()
        view = MotorJobView(**{k: v for k, v in snapshot.items() if k in _JOB_VIEW_FIELDS})
        return AuthoritativeJobSnapshot(view=view, observed_at=observed_at)

    def _append_locked(self, event: dict) -> None:
        line = json.dumps(event, ensure_ascii=False) + "\n"
        existed = self._path.exists()
        try:
            with open(self._path, "a", encoding="utf-8") as f:
                if f.write(line) != len(line):
                    raise OSError("short append to Motor JobStore")
                f.flush()
                os.fsync(f.fileno())
            if not existed:
                directory_fd = os.open(self._path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        except OSError:
            # A partial/uncertain append can no longer authorize a status read.
            self._history_integrity = False
            raise
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
        pipeline_id: str | None = None,
        tenant_id: str | None = None,
        user_id: str | None = None,
        project_id: str | None = None,
        job_id: str | None = None,
    ) -> str:
        # Governed dispatch reserves its identifier before the B6/B7 atomic
        # claim so the committed execution event and the subsequently-created
        # Motor job have one exact, durable binding.  Ordinary callers still
        # receive a store-generated identifier.
        job_id = job_id or str(uuid.uuid4())
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
            "status_updated_at": None,
            "error": None,
            "result_summary": None,
            # Task 7b (2026-09-18, historial-y-arreglos-de-pipeline): puesto
            # acá y no via update() posterior porque ya se conoce al
            # despachar (a diferencia de `model`, que worker.py resuelve
            # recién al validar el motor) -- update() re-esparce el estado
            # ENTERO (`{**self._index[job_id], **kwargs}`), así que este
            # valor sobrevive a cada escritura posterior sin que nadie tenga
            # que repetirlo.
            "pipeline_id": pipeline_id,
            "tenant_id": tenant_id,
            "user_id": user_id,
            "project_id": project_id,
        }
        with self._lock:
            if job_id in self._index:
                raise KeyError(f"job_id ya existe: {job_id}")
            self._append_locked(event)
        return job_id

    def update(self, job_id: str, **kwargs: Any) -> None:
        if {"tenant_id", "user_id", "project_id"} & set(kwargs):
            raise ValueError("ownership fields are immutable")
        with self._lock:
            if job_id not in self._index:
                raise KeyError(f"job_id desconocido: {job_id}")
            prior = self._index[job_id]
            event = {**prior, **kwargs, "job_id": job_id}
            if "status" in kwargs and getattr(kwargs["status"], "value", kwargs["status"]) != prior.get("status"):
                event["status_updated_at"] = time.time()
            self._append_locked(event)

    def write_result(self, job_id: str, content: str) -> str:
        """Guarda la salida completa del motor en un archivo propio, junto al
        JSONL (`motor_results/<job_id>.md`), y devuelve su ruta.

        En un archivo y no en el JSONL: cada update() re-appendea el estado
        ENTERO del job, así que un campo de decenas de KB se duplicaría en
        cada línea. Síncrono (escribe disco): el worker lo llama con
        asyncio.to_thread."""
        results_dir = self._path.parent / "motor_results"
        results_dir.mkdir(parents=True, exist_ok=True)
        path = results_dir / f"{job_id}.md"
        path.write_text(content, encoding="utf-8")
        return str(path)

    def get(self, job_id: str) -> MotorJobView | None:
        with self._lock:
            state = self._index.get(job_id)
            if state is None:
                return None
            snapshot = dict(state)
        return MotorJobView(**{k: v for k, v in snapshot.items() if k in _JOB_VIEW_FIELDS})

    def ids_en_estado(self, *estados: str) -> list[str]:
        """job_id de todo lo que esté en alguno de `estados` ahora mismo,
        según el índice reconstruido en memoria (2026-09-21, B-3 del
        endpoint de Procesamiento de Archivos). `_load()` reconstruye el
        índice desde el JSONL al arrancar pero NUNCA reconcilia estados
        "en vuelo" (`pending`/`running`) contra tareas realmente vivas --
        un consumidor que reinicia el proceso necesita poder preguntar
        "¿qué quedó a medias?" sin asomarse a `_index` directo."""
        with self._lock:
            return [job_id for job_id, evento in self._index.items() if evento.get("status") in estados]
