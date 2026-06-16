"""
LAS MANOS — Cliente de faceta (capa de conexión).

El lado FACETA del Intent Envelope. LAS MANOS (servidor) sabe recibir sobres;
esto enseña a una faceta a CONSTRUIR uno válido y enviarlo. Es el puente que
faltaba: sin él, ninguna faceta podía hablarle a LAS MANOS.

Orden de conexión firmado por la Mesa: Thot → Hipatia → Jekyll → Hyde.
"Primero quien mira, después quien sabe, después quien construye, al final
quien ejecuta." Este módulo conecta al PRIMERO: Thot, en modo auditoría.

Defensa en profundidad: el cliente se niega a construir un sobre fuera de las
capacidades declaradas de la faceta (fail-closed del lado emisor), PERO la
autoridad real es del servidor — policy.py es la pared. El cliente que se
salte su propio guardrail igual choca contra LAS MANOS. La memoria informa,
no autoriza; el cliente declara, no decide.

NOTA: este módulo NO razona. No es el LLM GPT-5.5 de Thot — es el cable por el
que ese LLM (cuando esté conectado) hablará con LAS MANOS. Construir sobres ≠
pensar como Thot. Fingir el razonamiento de una faceta sería mentira grave.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field

# Capacidades NO mutantes (complemento de MUTATING_CAPABILITIES de envelope.py).
READONLY_CAPABILITIES = frozenset({
    "ssh_exec_readonly", "read_file", "list_dir", "http_get",
    "audit_log_read", "validate_json", "validate_yaml",
})


class FacetRefusal(Exception):
    """El cliente se negó a construir un sobre fuera de su contrato.
    Fail-closed del lado faceta — antes incluso de molestar a LAS MANOS."""


@dataclass
class FacetResponse:
    """Respuesta de LAS MANOS a una llamada de faceta, normalizada."""
    status_code: int
    body: dict

    @property
    def ok(self) -> bool:
        return self.status_code == 200

    @property
    def rejected_envelope(self) -> bool:
        return self.status_code == 422 and "ENVELOPE_REJECTED" in str(
            self.body.get("detail", "")
        )

    @property
    def blocked_by_policy(self) -> bool:
        return self.status_code == 403


@dataclass
class FacetClient:
    """Cliente de una faceta hacia LAS MANOS, atado a su identidad y permisos.

    transport: cualquier objeto con .post(path, json=...) -> resp con
               .status_code y .json() (httpx.Client o fastapi TestClient).
    capabilities: lo que ESTA faceta puede pedir (defensa en profundidad).
                  El servidor revalida; esto solo evita pedir lo obvio mal.
    """
    facet_id: str
    transport: object
    actor_type: str = "faceta"
    capabilities: frozenset = READONLY_CAPABILITIES

    @classmethod
    def for_thot(cls, transport) -> "FacetClient":
        """Thot, primera faceta — modo auditoría, SOLO lectura.
        Sus capacidades son las que la Mesa firmó en config.toml:
        consulta de logs y lectura. Ni una sola capacidad mutante."""
        return cls(
            facet_id="thot",
            transport=transport,
            actor_type="faceta",
            capabilities=frozenset({
                "audit_log_read", "read_file", "list_dir", "ssh_exec_readonly",
            }),
        )

    # ── Construcción de sobres ──────────────────────────────────────────────
    def build_envelope(
        self,
        *,
        capability: str,
        origin_of_authority: str,
        intent_summary: str,
        target_host: str = "127.0.0.1",
        target_environment: str = "local",
        params: dict | None = None,
        verification_label: str = "local_context",
        risk_level: str = "none",
        memory_refs_used: list | None = None,
        human_gate_required: bool = False,
        approval_token: str | None = None,
        rollback_plan: str | None = None,
        traffic_class: str = "production",
        test_run_id: str | None = None,
    ) -> dict:
        """Arma un Intent Envelope COMPLETO y coherente para esta faceta.

        Bajo nivel: NO aplica el guardrail de capabilities (lo usa el demo de
        fallo cerrado para probar que el SERVIDOR frena). Las llamadas normales
        pasan por call(), que sí lo aplica."""
        mutating = capability not in READONLY_CAPABILITIES
        return {
            "trace_id": str(uuid.uuid4()),
            "facet_id": self.facet_id,
            "actor_type": self.actor_type,
            "origin_of_authority": origin_of_authority,
            "verification_label": verification_label,
            "intent_summary": intent_summary,
            "requested_capability": capability,
            "target_environment": target_environment,
            "risk_level": risk_level,
            "memory_refs_used": memory_refs_used or [],
            "freshness_required": False,
            "dry_run_required": mutating,
            "policy_required": True,
            "human_gate_required": human_gate_required,
            "approval_token": approval_token,
            "rollback_plan": rollback_plan,
            # read-only puede ser "none"; mutante exige alcance real.
            "kill_switch_scope": "none" if not mutating else "per_operation",
            "fail_closed_behavior": (
                "abortar la intención sin ejecutar ni dejar rastro mutante; "
                "el rechazo queda en el audit log"
            ),
            "target_host": target_host,
            "params": params or {},
            # Procedencia forense (Thot Audit Watch): el cable declara tráfico
            # real por defecto; los tests pasan su clase explícita.
            "traffic_class": traffic_class,
            "test_run_id": test_run_id,
        }

    # ── Envío ───────────────────────────────────────────────────────────────
    def _post(self, path: str, envelope: dict) -> FacetResponse:
        resp = self.transport.post(path, json=envelope)
        try:
            body = resp.json()
        except Exception:  # noqa: BLE001
            body = {"raw": resp.text}
        return FacetResponse(status_code=resp.status_code, body=body)

    def call(self, *, capability: str, **kw) -> FacetResponse:
        """Ejecuta una capacidad vía /execute, con guardrail de faceta.
        Si la capacidad no está en el contrato de la faceta → FacetRefusal
        (fail-closed del lado emisor; ni se envía)."""
        if capability not in self.capabilities:
            raise FacetRefusal(
                f"Faceta '{self.facet_id}' no tiene la capacidad '{capability}' "
                f"en su contrato. Permitidas: {sorted(self.capabilities)}. "
                f"El cliente se niega antes de molestar a LAS MANOS."
            )
        return self._post("/execute", self.build_envelope(capability=capability, **kw))

    def plan(self, *, capability: str, **kw) -> FacetResponse:
        """Previsualiza el plan vía /plan (no ejecuta nada)."""
        return self._post("/plan", self.build_envelope(capability=capability, **kw))

    # ── Conveniencias de solo lectura (la función de Thot) ──────────────────
    def read_audit_log(self, n: int = 50, *, origin: str) -> FacetResponse:
        return self.call(
            capability="audit_log_read",
            origin_of_authority=origin,
            intent_summary=f"auditar los últimos {n} eventos del log forense",
            params={"n": n},
        )

    def list_dir(self, path: str, *, origin: str, host: str = "127.0.0.1") -> FacetResponse:
        return self.call(
            capability="list_dir",
            origin_of_authority=origin,
            intent_summary=f"listar el directorio {path}",
            target_host=host,
            params={"path": path},
        )


# ── Revisión forense (la función real de Thot, versión determinista) ────────
@dataclass
class AuditFinding:
    severity: str   # "info" | "atención" | "alerta"
    message: str


@dataclass
class AuditReview:
    """El reporte de una revisión de Thot. DETERMINISTA: cuenta y clasifica
    eventos. NO es el juicio del LLM GPT-5.5 (eso exige la API de OpenAI, hoy
    sin conectar). Es la auditoría mecánica que el cable ya permite."""
    total_events: int
    by_event: dict = field(default_factory=dict)
    findings: list = field(default_factory=list)

    def render(self) -> str:
        lines = [f"REVISIÓN DE THOT — {self.total_events} eventos auditados"]
        for ev, n in sorted(self.by_event.items()):
            lines.append(f"  · {ev}: {n}")
        if self.findings:
            lines.append("  Hallazgos:")
            for f in self.findings:
                lines.append(f"    [{f.severity}] {f.message}")
        else:
            lines.append("  Sin hallazgos.")
        return "\n".join(lines)


def review_audit(events: list) -> AuditReview:
    """Thot revisa una tanda de eventos del audit log y reporta.
    Función real de Thot: mirar y cuestionar. Aquí, de forma determinista."""
    by_event: dict = {}
    for e in events:
        ev = e.get("event", "DESCONOCIDO")
        by_event[ev] = by_event.get(ev, 0) + 1

    findings: list = []
    rechazos = by_event.get("ENVELOPE_REJECTED", 0)
    if rechazos:
        findings.append(AuditFinding(
            "info",
            f"{rechazos} sobres rechazados: la puerta se está negando a "
            f"obedecer mal, como debe.",
        ))
    # Toda ejecución fallida es algo que Thot debe señalar.
    for e in events:
        if e.get("event") == "EXECUTION" and e.get("success") is False:
            findings.append(AuditFinding(
                "atención",
                f"ejecución fallida (req {e.get('request_id')}): "
                f"{e.get('error', 'sin detalle')}",
            ))
        if e.get("event") == "KILL_SWITCH":
            findings.append(AuditFinding(
                "alerta",
                f"kill switch disparado por {e.get('triggered_by', '?')}",
            ))
    return AuditReview(total_events=len(events), by_event=by_event, findings=findings)
