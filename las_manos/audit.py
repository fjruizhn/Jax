"""
LAS MANOS — Auditoría forense.

Todo lo que LAS MANOS ejecuta queda aquí.
Inmutable: solo append, nunca delete ni overwrite.
Formato: JSONL (una línea JSON por evento).

"Confío en LAS MANOS siempre que pueda ver sus manos sangrar en los logs."
— Jekyll

Thot Audit Watch (Mesa, 16-jun-2026): cada evento declara su procedencia
forense — environment, traffic_class y test_run_id — para que el auditor
distinga tráfico REAL de tráfico de PRUEBAS sin adivinar. Sin estado oculto:
cada llamada declara su contexto (parámetro explícito).

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import json
import os
import hashlib
from datetime import datetime, timezone
from pathlib import Path


# ── Thot Audit Watch — clasificación forense ────────────────────────────────
# environment se INFIERE del target_environment del Envelope; no lo declara la
# faceta. Fail-safe: ante un valor desconocido, "production" (mejor sobreestimar
# que subregistrar tráfico real como si fuera prueba).
def environment_from_target(target_environment: str | None) -> str:
    """Mapea el target_environment del Envelope al environment del log forense."""
    return {
        "local":   "test",
        "staging": "staging",
        "prod":    "production",
    }.get(target_environment or "", "production")


# Valores válidos de traffic_class (Mesa, 16-jun-2026). Documentados aquí para
# referencia; audit.py no los valida (lo hace el Envelope vía Pydantic).
TRAFFIC_CLASSES = (
    "test_structural",   # tests que ejercen la capa estructural (Pydantic)
    "test_semantic",     # tests que ejercen la capa semántica (validate_envelope)
    "dry_run",           # previsualización sin ejecutar
    "production",        # tráfico real de facetas
    "adversarial_test",  # pruebas que atacan la puerta a propósito
    "unknown",           # nadie lo declaró — default fail-safe
)


class AuditLog:
    def __init__(self, log_path: str) -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _forensics(
        environment: str,
        traffic_class: str,
        test_run_id: str | None,
    ) -> dict:
        """Los tres campos de procedencia forense que lleva cada evento."""
        return {
            "environment":   environment,
            "traffic_class": traffic_class,
            "test_run_id":   test_run_id,
        }

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
        environment: str = "production",
        traffic_class: str = "unknown",
        test_run_id: str | None = None,
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
            **self._forensics(environment, traffic_class, test_run_id),
        })
        return request_id

    def log_envelope(
        self,
        request_id: str,
        facet: str,
        capability: str,
        target_environment: str,
        risk_level: str,
        environment: str = "production",
        traffic_class: str = "unknown",
        test_run_id: str | None = None,
    ) -> None:
        """Registra un Intent Envelope ACEPTADO (pasó las dos capas)."""
        self._write({
            "event":              "ENVELOPE_ACCEPTED",
            "request_id":         request_id,
            "facet":              facet,
            "capability":         capability,
            "target_environment": target_environment,
            "risk_level":         risk_level,
            **self._forensics(environment, traffic_class, test_run_id),
        })

    def log_envelope_rejected(
        self,
        request_id: str | None,
        reason: str,
        layer: str = "semantica",
        environment: str = "production",
        traffic_class: str = "unknown",
        test_run_id: str | None = None,
    ) -> None:
        """Registra un Intent Envelope RECHAZADO — LAS MANOS se negó.
        layer: 'estructural' (Pydantic) o 'semantica' (envelope.validate)."""
        self._write({
            "event":      "ENVELOPE_REJECTED",
            "request_id": request_id,
            "layer":      layer,
            "reason":     reason,
            **self._forensics(environment, traffic_class, test_run_id),
        })

    def log_policy_check(
        self,
        request_id: str,
        facet: str,
        operation: str,
        target_host: str,
        allowed: bool,
        reason: str,
        environment: str = "production",
        traffic_class: str = "unknown",
        test_run_id: str | None = None,
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
            **self._forensics(environment, traffic_class, test_run_id),
        })

    def log_dryrun(
        self,
        request_id: str,
        plan: dict,
        environment: str = "production",
        traffic_class: str = "unknown",
        test_run_id: str | None = None,
    ) -> None:
        """Registra el plan de dry-run antes de ejecutar."""
        self._write({
            "event":      "DRYRUN",
            "request_id": request_id,
            "plan":       plan,
            **self._forensics(environment, traffic_class, test_run_id),
        })

    def log_human_gate(
        self,
        request_id: str,
        approved: bool,
        token_used: str | None = None,
        environment: str = "production",
        traffic_class: str = "unknown",
        test_run_id: str | None = None,
    ) -> None:
        """Registra la decisión del human gate."""
        self._write({
            "event":      "HUMAN_GATE",
            "request_id": request_id,
            "approved":   approved,
            "token_hash": hashlib.sha256(
                (token_used or "").encode()
            ).hexdigest()[:16] if token_used else None,
            **self._forensics(environment, traffic_class, test_run_id),
        })

    def log_execution(
        self,
        request_id: str,
        success: bool,
        stdout: str = "",
        stderr: str = "",
        exit_code: int | None = None,
        error: str | None = None,
        environment: str = "production",
        traffic_class: str = "unknown",
        test_run_id: str | None = None,
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
            **self._forensics(environment, traffic_class, test_run_id),
        })

    def log_kill_switch(
        self,
        triggered_by: str,
        environment: str = "production",
        traffic_class: str = "unknown",
        test_run_id: str | None = None,
    ) -> None:
        """Kill switch activado — evento crítico."""
        self._write({
            "event":        "KILL_SWITCH",
            "triggered_by": triggered_by,
            "CRITICAL":     True,
            **self._forensics(environment, traffic_class, test_run_id),
        })

    def tail(self, n: int = 50) -> list[dict]:
        """Devuelve los últimos N eventos — para /audit_log_read."""
        if not self.log_path.exists():
            return []
        lines = self.log_path.read_text(encoding="utf-8").strip().splitlines()
        return [json.loads(l) for l in lines[-n:]]
