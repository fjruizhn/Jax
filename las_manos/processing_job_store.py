"""Authoritative, owner-bound JSONL store for Processing job status."""
from __future__ import annotations

import json
import math
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import ValidationError

from motor_registry.job_store import JobStore, _JOB_VIEW_FIELDS
from motor_registry.models import MotorJobView
from processing_ownership import OWNER_VERSION, ProcessingOwnershipContext, ProcessingOwnershipError


class ProcessingJobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    CANCELLING = "cancelling"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class ProcessingJobSnapshot:
    view: MotorJobView
    owner: ProcessingOwnershipContext
    observed_at: datetime


class ProcessingJobStore(JobStore):
    """Separate source contract; base Motor JobStore remains unmodified."""

    def __init__(self, path: str) -> None:
        self._owners: dict[str, ProcessingOwnershipContext] = {}
        self._quarantined: set[str] = set()
        super().__init__(path)

    @staticmethod
    def _owner_from_event(event: dict[str, Any]) -> ProcessingOwnershipContext:
        owner = event.get("processing_ownership")
        if not isinstance(owner, dict) or set(owner) != {"version", "tenant_id", "user_id", "project_id"}:
            raise ProcessingOwnershipError("processing event lacks canonical owner")
        return ProcessingOwnershipContext(**owner)

    def _load(self) -> None:
        if not self._path.exists():
            return
        with open(self._path, "rb") as stream:
            for raw_line in stream:
                if not raw_line.strip() or not raw_line.endswith(b"\n"):
                    self._history_integrity = False
                    continue
                try:
                    event = json.loads(raw_line.decode("utf-8"), object_pairs_hook=self._no_duplicate_keys)
                    job_id = event.get("job_id") if isinstance(event, dict) else None
                    if not isinstance(job_id, str) or not job_id or event.get("status") not in {item.value for item in ProcessingJobStatus}:
                        self._history_integrity = False
                        continue
                    MotorJobView(**{key: value for key, value in event.items() if key in _JOB_VIEW_FIELDS})
                    # Complete legacy records are operational history, not
                    # authority. They quarantine only their own job and a
                    # later owner-bearing event must not promote it.
                    if "processing_ownership" not in event:
                        self._quarantined.add(job_id)
                        continue
                    owner = self._owner_from_event(event)
                    if job_id in self._quarantined:
                        continue
                    prior = self._owners.get(job_id)
                    if prior is not None and prior != owner:
                        raise ProcessingOwnershipError("processing ownership drift")
                    timestamps = ("created_at", "started_at", "finished_at", "status_updated_at")
                    if any(event.get(name) is not None and (isinstance(event[name], bool) or not isinstance(event[name], (int, float)) or not math.isfinite(event[name]) or event[name] < 0) for name in timestamps):
                        raise ValueError("invalid timestamp")
                    self._owners[job_id] = owner
                    self._index[job_id] = event
                except (UnicodeDecodeError, json.JSONDecodeError, ProcessingOwnershipError, ValidationError, TypeError, ValueError):
                    self._history_integrity = False

    @staticmethod
    def _no_duplicate_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ProcessingOwnershipError("duplicate JSON key")
            result[key] = value
        return result

    def create(self, *, ownership: ProcessingOwnershipContext, caller: str, capability: str, motor: str,
               trace_id: str, prompt: str, recursion_depth: int, pipeline_id: str | None = None,
               job_id: str | None = None, **unused: Any) -> str:
        if not isinstance(ownership, ProcessingOwnershipContext):
            raise TypeError("processing ownership must be a ProcessingOwnershipContext")
        if unused:
            raise TypeError("processing jobs do not accept independent ownership fields")
        identifier = job_id or str(uuid.uuid4())
        event = {
            "job_id": identifier, "status": ProcessingJobStatus.PENDING.value, "motor": motor,
            "capability": capability, "caller": caller, "trace_id": trace_id, "prompt": prompt,
            "recursion_depth": recursion_depth, "pipeline_id": pipeline_id, "created_at": time.time(),
            # Keep the inherited view's scope fields coherent with the one
            # immutable nested owner record; updates reject these fields.
            "tenant_id": ownership.tenant_id, "user_id": ownership.user_id,
            "project_id": ownership.project_id,
            "started_at": None, "finished_at": None, "status_updated_at": None, "error": None,
            "result_summary": None,
            "processing_ownership": {"version": ownership.version, "tenant_id": ownership.tenant_id,
                                       "user_id": ownership.user_id, "project_id": ownership.project_id},
        }
        with self._lock:
            if identifier in self._index:
                raise KeyError(f"job_id ya existe: {identifier}")
            self._append_locked(event)
            self._owners[identifier] = ownership
        return identifier

    def update(self, job_id: str, **kwargs: Any) -> None:
        if "processing_ownership" in kwargs or {"tenant_id", "user_id", "project_id"} & set(kwargs):
            raise ValueError("processing ownership is immutable")
        with self._lock:
            prior = self._index.get(job_id)
            owner = self._owners.get(job_id)
            if prior is None or owner is None:
                raise KeyError(f"job_id desconocido: {job_id}")
            event = {**prior, **kwargs, "job_id": job_id, "processing_ownership": {
                "version": owner.version, "tenant_id": owner.tenant_id, "user_id": owner.user_id, "project_id": owner.project_id,
            }}
            status = getattr(kwargs.get("status"), "value", kwargs.get("status"))
            if "status" in kwargs and status not in {item.value for item in ProcessingJobStatus}:
                raise ValueError("processing status is outside its closed vocabulary")
            if status is not None and status != prior.get("status"):
                event["status_updated_at"] = time.time()
            self._append_locked(event)

    def authoritative_snapshot(self, job_id: str) -> ProcessingJobSnapshot | None:
        with self._lock:
            if not self._history_integrity or job_id in self._quarantined:
                return None
            state, owner = self._index.get(job_id), self._owners.get(job_id)
            if state is None or owner is None:
                return None
            view = MotorJobView(**{key: value for key, value in dict(state).items() if key in _JOB_VIEW_FIELDS})
            return ProcessingJobSnapshot(view=view, owner=owner, observed_at=datetime.now(timezone.utc))

    def source_configuration(self) -> dict[str, object]:
        return {"store_contract": "processing-job-store-v1", "source_id": str(self._path.resolve()),
                "event_format": "processing-job-event-v1", "durability": "append-flush-fsync-v1",
                "ownership_contract": "platform-authenticated-processing-owner.1",
                "source_role": "las-manos-processing-jobs",
                "allowed_statuses": [item.value for item in ProcessingJobStatus]}
