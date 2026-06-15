"""
LAS MANOS — Auditoría forense.

Todo lo que LAS MANOS ejecuta queda aquí.
Inmutable: solo append, nunca delete ni overwrite.
Formato: JSONL (una línea JSON por evento).

"Confío en LAS MANOS siempre que pueda ver sus manos sangrar en los logs."
— Jekyll

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import json
import os
import hashlib
from datetime import datetime, timezone
from pathlib import Path


class AuditLog:
    def __init__(self, log_path: str) -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def _write(self, entry: dict) -> None:
        """Append-only. Nunca sobreescribe."""
        entry["@timestamp"] = datetime.now(timezone.utc).isoformat()
        line = json.dumps(entry, ensure_ascii=False)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def log_request(
        self,
        facet: str,
        operation: str,
        target_host: str,
        payload: dict,
        job_id: str,
    ) -> str:
        """Registra una solicitud entrante. Devuelve el request_id."""
        # Hash del payload para trazabilidad sin exponer secretos
        payload_hash = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()[:16]

        request_id = f"{job_id}-{payload_hash}"

        self._write({
            "event":        "REQUEST",
            "request_id":   request_id,
            "job_id":       job_id,
            "facet":        facet,
            "operation":    operation,
            "target_host":  target_host,
            "payload_hash": payload_hash,
        })
        return request_id

    def log_policy_check(
        self,
        request_id: str,
        facet: str,
        operation: str,
        target_host: str,
        allowed: bool,
        reason: str,
    ) -> None:
        """Registra el resultado del policy check."""
        self._write({
            "event":       "POLICY_CHECK",
            "request_id":  request_id,
            "facet":       facet,
            "operation":   operation,
            "target_host": target_host,
            "allowed":     allowed,
            "reason":      reason,
        })

    def log_dryrun(
        self,
        request_id: str,
        plan: dict,
    ) -> None:
        """Registra el plan de dry-run antes de ejecutar."""
        self._write({
            "event":      "DRYRUN",
            "request_id": request_id,
            "plan":       plan,
        })

    def log_human_gate(
        self,
        request_id: str,
        approved: bool,
        token_used: str | None = None,
    ) -> None:
        """Registra la decisión del human gate."""
        self._write({
            "event":      "HUMAN_GATE",
            "request_id": request_id,
            "approved":   approved,
            "token_hash": hashlib.sha256(
                (token_used or "").encode()
            ).hexdigest()[:16] if token_used else None,
        })

    def log_execution(
        self,
        request_id: str,
        success: bool,
        stdout: str = "",
        stderr: str = "",
        exit_code: int | None = None,
        error: str | None = None,
    ) -> None:
        """Registra el resultado de la ejecución."""
        self._write({
            "event":      "EXECUTION",
            "request_id": request_id,
            "success":    success,
            "exit_code":  exit_code,
            "stdout_len": len(stdout),
            "stderr_len": len(stderr),
            "stdout_tail": stdout[-500:] if stdout else "",
            "stderr_tail": stderr[-500:] if stderr else "",
            "error":      error,
        })

    def log_kill_switch(self, triggered_by: str) -> None:
        """Kill switch activado — evento crítico."""
        self._write({
            "event":        "KILL_SWITCH",
            "triggered_by": triggered_by,
            "CRITICAL":     True,
        })

    def tail(self, n: int = 50) -> list[dict]:
        """Devuelve los últimos N eventos — para /audit_log_read."""
        if not self.log_path.exists():
            return []
        lines = self.log_path.read_text(encoding="utf-8").strip().splitlines()
        return [json.loads(l) for l in lines[-n:]]
