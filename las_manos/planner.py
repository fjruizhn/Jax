"""
LAS MANOS — Planner.

Convierte una intención declarativa de una faceta en un plan estructurado
y legible ANTES de ejecutar nada. El plan dice, en lenguaje claro:
qué host se toca, qué se hace, qué riesgo tiene, y cómo se revierte.

El planner no ejecuta ni autoriza: traduce. El policy engine decide,
el human gate aprueba, los workers ejecutan. Esto solo ilumina.

"Nadie debería apretar un botón sin leer antes qué botón está apretando."
— Thot

En memoria de Jairo Urbina.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict


# Operaciones que modifican estado remoto.
MUTATING_OPS = {"ssh_exec", "write_file", "rsync", "kill_process"}

# Operaciones de solo lectura.
READONLY_OPS = {
    "ssh_exec_readonly", "read_file", "list_dir",
    "http_get", "audit_log_read", "validate_json", "validate_yaml",
}

# Cómo se revierte cada operación. Si no hay rollback automático, se dice.
ROLLBACK_BY_OP = {
    "write_file": "Restaurar desde el snapshot generado: cp path.bak.TIMESTAMP path",
    "rsync": "Restaurar el destino desde su snapshot previo (requires_snapshot=true)",
    "ssh_exec": "SIN rollback automático — depende del comando. Revisar plan manual.",
    "kill_process": "SIN rollback automático — reiniciar el servicio/proceso manualmente.",
}


@dataclass
class Plan:
    facet: str
    operation: str
    target_host: str
    environment: str | None
    description: str
    mutating: bool
    risk: str                    # "ninguno" | "bajo" | "medio" | "alto"
    rollback: str
    requires_dryrun: bool
    requires_snapshot: bool
    requires_human_gate: bool
    params: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def render(self) -> str:
        """Representación legible para humanos (Fernando, Thot)."""
        reqs = [
            req for req, on in (
                ("dry-run", self.requires_dryrun),
                ("snapshot", self.requires_snapshot),
                ("human-gate", self.requires_human_gate),
            ) if on
        ]
        reqs_line = ", ".join(reqs) if reqs else "(ninguno)"
        lines = [
            "─" * 56,
            "  PLAN — LAS MANOS",
            "─" * 56,
            f"  Faceta      : {self.facet}",
            f"  Operación   : {self.operation}",
            f"  Host        : {self.target_host}  ({self.environment or 'desconocido'})",
            f"  Qué hace    : {self.description}",
            f"  Muta estado : {'SÍ' if self.mutating else 'no (solo lectura)'}",
            f"  Riesgo      : {self.risk.upper()}",
            f"  Rollback    : {self.rollback}",
            f"  Requisitos  : {reqs_line}",
        ]
        if self.warnings:
            lines.append("  Advertencias:")
            lines.extend(f"    ⚠ {w}" for w in self.warnings)
        lines.append("─" * 56)
        return "\n".join(lines)


class Planner:
    def __init__(self, config: dict) -> None:
        self.cfg = config
        self.environments = config["environments"]
        self.ops = config["ops"]

    def _resolve_env(self, host: str) -> str | None:
        for env, hosts in self.environments.items():
            if host in hosts:
                return env
        return None

    def _assess_risk(self, operation: str, env: str | None) -> str:
        """Heurística de riesgo: ambiente + naturaleza de la operación."""
        if operation in READONLY_OPS:
            return "ninguno"
        if operation == "kill_process":
            return "alto" if env == "prod" else "medio"
        if operation in ("rsync", "ssh_exec"):
            return "alto" if env == "prod" else "medio"
        if operation == "write_file":
            return "medio" if env == "prod" else "bajo"
        return "bajo"

    def plan(
        self,
        facet: str,
        operation: str,
        target_host: str,
        params: dict | None = None,
    ) -> Plan:
        """Construye el plan a partir de la intención declarativa."""
        params = params or {}
        env = self._resolve_env(target_host)
        op_cfg = self.ops.get(operation, {})

        mutating = operation in MUTATING_OPS
        risk = self._assess_risk(operation, env)

        # Descripción humana de qué hará exactamente.
        description = self._describe(operation, target_host, params, op_cfg)

        rollback = ROLLBACK_BY_OP.get(
            operation,
            "No aplica — operación de solo lectura, no deja rastro mutante.",
        )

        warnings: list[str] = []
        if env is None:
            warnings.append(
                f"Host '{target_host}' no pertenece a ningún ambiente conocido."
            )
        if env == "prod" and mutating:
            warnings.append("Operación MUTANTE sobre PRODUCCIÓN — human gate obligatorio.")
        if operation == "ssh_exec" and not params.get("command"):
            warnings.append("ssh_exec sin 'command' en params — el plan está incompleto.")

        # En prod toda mutación fuerza human gate (espejo del policy engine).
        requires_human_gate = op_cfg.get("requires_human_gate", False)
        if env == "prod" and mutating:
            requires_human_gate = True

        return Plan(
            facet=facet,
            operation=operation,
            target_host=target_host,
            environment=env,
            description=description,
            mutating=mutating,
            risk=risk,
            rollback=rollback,
            requires_dryrun=op_cfg.get("requires_dryrun", False),
            requires_snapshot=op_cfg.get("requires_snapshot", False),
            requires_human_gate=requires_human_gate,
            params=params,
            warnings=warnings,
        )

    def _describe(
        self, operation: str, host: str, params: dict, op_cfg: dict,
    ) -> str:
        """Traduce la operación a una frase clara según sus params."""
        if operation in ("ssh_exec", "ssh_exec_readonly"):
            cmd = params.get("command", "(sin comando)")
            return f"Ejecutar en {host}: `{cmd}`"
        if operation == "read_file":
            return f"Leer archivo {params.get('path', '?')} en {host}"
        if operation == "list_dir":
            return f"Listar directorio {params.get('path', '?')} en {host}"
        if operation == "write_file":
            return (
                f"Escribir archivo {params.get('path', '?')} en {host} "
                f"(con snapshot previo)"
            )
        if operation == "rsync":
            return (
                f"Transferir (pull) {params.get('origen_host', '?')}:"
                f"{params.get('origen_path', '?')} → {host}:"
                f"{params.get('destino_path', '?')}"
            )
        if operation == "kill_process":
            return f"Terminar proceso PID {params.get('pid', '?')} en {host}"
        if operation == "audit_log_read":
            return f"Leer log de auditoría de LAS MANOS (últimos {params.get('n', 50)})"
        # Por defecto: la descripción declarada en config.
        return op_cfg.get("description", f"Operación '{operation}' sobre {host}")
