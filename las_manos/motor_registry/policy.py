"""
LAS MANOS — Motor Registry: motor de políticas para despacho de motores.

Valida antes de crear un job:
  1. capability existe en el catálogo
  2. caller autorizado para esa capability
  3. human_gate_token presente si capability.requires_human_gate
  4. recursion_depth no excede el máximo de la capability
  5. context no contiene claves prohibidas (secretos)
  6. hay motor habilitado disponible para la capability
  7. ese motor es sandbox_only (v0.1 solo admite sandbox)
  8. timeout_seconds (si el caller lo pide) no excede capability.max_execution_minutes

Módulo PURO: sin I/O, sin red, testeable en aislamiento.
Falla cerrado: ante cualquier duda, rechaza.

Checks 1-5 delegados a MotorPolicy.check_capability_admission(), que no requiere
motor resuelto. MotorPolicy.check() los delega ahí, luego continúa con 6-8
(docs/superpowers/specs/2026-08-27-http-facets-motor-policy-governance-design.md).

Check 8 (ronda 4, 2026-08-20, T2.b): admisión, no circuit breaker. Valida el
PRESUPUESTO PEDIDO contra el techo declarado de la capability -- no mide
tiempo transcurrido (este modulo es sincrono/puro, sin reloj). El corte en
vivo sigue siendo el mismo de siempre: loop_deadline en motor_registry/
worker.py, alimentado por timeout_seconds (que hoy REUSA Step.timeout_seconds
de Jacobs, ver comentario en models.py de este mismo modulo) -- este check
no lo duplica, solo evita que alguien pida un presupuesto mayor al que la
capability tiene declarado. timeout_seconds=None (el caller no pidio
presupuesto) no dispara el check -- mismo criterio de compatibilidad que ya
declaraba MotorDispatchRequest.timeout_seconds ("None = sin presupuesto
extra, compat con cualquier caller que no lo mande"); no se vuelve
obligatorio ahora, eso cambiaria comportamiento para callers que hoy no lo
mandan y no fue lo que se pidio esta ronda.
"""
from __future__ import annotations

from dataclasses import dataclass

from motor_registry.catalog import CapabilityEntry, MotorCatalog

# Claves que nunca deben aparecer en el contexto de un job.
# Las manos no tocan secretos.
FORBIDDEN_CONTEXT_KEYS: frozenset[str] = frozenset({
    "api_key", "secret", "password", "token", "private_key",
    "access_key", "credentials", "auth",
})


@dataclass
class MotorPolicyResult:
    allowed: bool
    reason: str
    resolved_motor: str = ""
    requires_human_gate: bool = False


class MotorPolicy:
    def __init__(self, catalog: MotorCatalog) -> None:
        self._catalog = catalog

    def check_capability_admission(
        self,
        *,
        caller: str,
        capability: str,
        context_keys: list[str],
        recursion_depth: int,
        human_gate_token: str | None,
    ) -> MotorPolicyResult:
        """Checks 1-5 de check() -- capability existe, caller autorizado,
        human gate, recursion depth, claves prohibidas. NO incluye
        resolucion de motor (6-7) ni techo de timeout (8): ninguno aplica
        a un dispatch que no pasa por un motor resuelto (facets HTTP-
        directos, docs/superpowers/specs/2026-08-27-http-facets-motor-
        policy-governance-design.md, decision del punto 2 -- el techo NO
        se activa esta ronda). check() la llama como su primer paso;
        firma SIN motor/timeout_seconds a proposito, ningun check de acá
        los necesita."""
        cap = self._catalog.get_capability(capability)
        if cap is None:
            return MotorPolicyResult(False, f"Capability desconocida: '{capability}'")

        if caller not in cap.allowed_callers:
            return MotorPolicyResult(
                False,
                f"Caller '{caller}' no autorizado para '{capability}'. "
                f"Autorizados: {cap.allowed_callers}",
            )

        if cap.requires_human_gate and not (human_gate_token and human_gate_token.strip()):
            return MotorPolicyResult(
                False,
                f"Capability '{capability}' requiere human_gate_token y no fue provisto",
            )

        if recursion_depth > cap.max_recursion_depth:
            return MotorPolicyResult(
                False,
                f"recursion_depth={recursion_depth} excede máximo "
                f"{cap.max_recursion_depth} para '{capability}'",
            )

        bad_keys = [k for k in context_keys if k.lower() in FORBIDDEN_CONTEXT_KEYS]
        if bad_keys:
            return MotorPolicyResult(
                False,
                f"Context contiene claves prohibidas: {bad_keys}. "
                "Las manos no tocan secretos.",
            )

        return MotorPolicyResult(allowed=True, reason=f"OK: '{caller}' → '{capability}' (admisión)")

    def check(
        self,
        *,
        caller: str,
        capability: str,
        motor: str | None,
        context_keys: list[str],
        recursion_depth: int,
        human_gate_token: str | None,
        timeout_seconds: int | None = None,
    ) -> MotorPolicyResult:
        """Valida el dispatch completo. Devuelve al PRIMER fallo.
        Checks 1-5 delegados a check_capability_admission() -- ver esa
        docstring. 6-8 (resolver motor, sandbox_only, techo) sin cambios
        de esta refactorización (docs/superpowers/specs/2026-08-27-
        http-facets-motor-policy-governance-design.md, Requisito 1: cero
        cambio de comportamiento)."""
        admission = self.check_capability_admission(
            caller=caller, capability=capability, context_keys=context_keys,
            recursion_depth=recursion_depth, human_gate_token=human_gate_token,
        )
        if not admission.allowed:
            return admission

        cap = self._catalog.get_capability(capability)

        resolved = self._resolve_motor(motor, cap)
        if resolved is None:
            return MotorPolicyResult(
                False,
                f"No hay motor habilitado disponible para '{capability}'. "
                f"Motores permitidos: {cap.allowed_motors}",
            )

        motor_entry = self._catalog.get_motor(resolved)
        if motor_entry and not motor_entry.sandbox_only:
            return MotorPolicyResult(
                False,
                f"Motor '{resolved}' no es sandbox_only. "
                "Motor Registry v0.1 solo admite motores sandbox.",
            )

        if timeout_seconds is not None:
            max_seconds = cap.max_execution_minutes * 60
            if timeout_seconds > max_seconds:
                return MotorPolicyResult(
                    False,
                    f"timeout_seconds={timeout_seconds} excede el techo de "
                    f"'{capability}' ({cap.max_execution_minutes} min = {max_seconds}s)",
                )

        return MotorPolicyResult(
            allowed=True,
            reason=f"OK: '{caller}' → '{capability}' vía motor '{resolved}'",
            resolved_motor=resolved,
            requires_human_gate=cap.requires_human_gate,
        )

    def _resolve_motor(self, requested: str | None, cap: CapabilityEntry) -> str | None:
        """Motor solicitado si válido y habilitado; si no, el primero habilitado."""
        if requested is not None:
            if requested not in cap.allowed_motors:
                return None
            m = self._catalog.get_motor(requested)
            return requested if (m and m.enabled) else None

        for name in cap.allowed_motors:
            m = self._catalog.get_motor(name)
            if m and m.enabled:
                return name
        return None
