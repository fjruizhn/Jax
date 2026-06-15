"""
LAS MANOS — Motor de políticas.

Antes de ejecutar CUALQUIER operación, el plan pasa por aquí.
Valida: faceta autorizada, ambiente permitido, operación habilitada,
comando no prohibido, blast radius aceptable.

"Los principios no ejecutan rollback. Los principios no detienen un rm -rf."
— Thot

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class PolicyResult:
    allowed: bool
    reason: str
    requires_dryrun: bool = False
    requires_snapshot: bool = False
    requires_human_gate: bool = False


class PolicyEngine:
    def __init__(self, config: dict) -> None:
        self.cfg = config
        self.environments = config["environments"]
        self.facets = config["facets"]
        self.ops = config["ops"]

    def _resolve_env(self, host: str) -> str | None:
        """¿A qué ambiente pertenece este host?"""
        for env, hosts in self.environments.items():
            if host in hosts:
                return env
        return None

    def check(
        self,
        facet: str,
        operation: str,
        target_host: str,
        command: str | None = None,
    ) -> PolicyResult:
        """La función central. Valida todo antes de permitir ejecución."""

        # 1) ¿La faceta existe?
        if facet not in self.facets:
            return PolicyResult(False, f"Faceta desconocida: {facet}")

        facet_cfg = self.facets[facet]

        # 2) ¿La operación existe?
        if operation not in self.ops:
            return PolicyResult(False, f"Operación desconocida: {operation}")

        # 3) ¿La faceta puede ejecutar esta operación?
        if operation not in facet_cfg["allowed_ops"]:
            return PolicyResult(
                False,
                f"Faceta '{facet}' no autorizada para operación '{operation}'. "
                f"Permitidas: {facet_cfg['allowed_ops']}"
            )

        # 4) ¿A qué ambiente pertenece el host?
        env = self._resolve_env(target_host)
        if env is None:
            return PolicyResult(
                False, f"Host '{target_host}' no pertenece a ningún ambiente conocido"
            )

        # 5) ¿La faceta puede operar en este ambiente?
        if env not in facet_cfg["allowed_envs"]:
            return PolicyResult(
                False,
                f"Faceta '{facet}' no autorizada en ambiente '{env}'. "
                f"Permitidos: {facet_cfg['allowed_envs']}"
            )

        op_cfg = self.ops[operation]

        # 6) Si hay comando, validar contra allowlist/denylist
        if command is not None:
            cmd_check = self._check_command(operation, op_cfg, command)
            if not cmd_check.allowed:
                return cmd_check

        # 7) ¿Escritura en prod sin permiso?
        is_prod = (env == "prod")
        is_mutating = operation in ("ssh_exec", "write_file", "rsync", "kill_process")
        if is_prod and is_mutating and not facet_cfg.get("can_write_prod", False):
            return PolicyResult(
                False,
                f"Faceta '{facet}' no puede mutar producción (can_write_prod=false)"
            )

        # PASA — pero con requisitos según la operación
        requires_human_gate = op_cfg.get("requires_human_gate", False)
        # En prod, toda operación mutante requiere human gate sin importar el op
        if is_prod and is_mutating:
            requires_human_gate = True

        return PolicyResult(
            allowed=True,
            reason=f"OK: {facet} → {operation} en {env} ({target_host})",
            requires_dryrun=op_cfg.get("requires_dryrun", False),
            requires_snapshot=op_cfg.get("requires_snapshot", False),
            requires_human_gate=requires_human_gate,
        )

    def _check_command(self, operation: str, op_cfg: dict, command: str) -> PolicyResult:
        """Valida un comando contra allow_cmds y deny_patterns."""
        # Patrones prohibidos primero (denylist gana siempre)
        deny = op_cfg.get("deny_patterns", [])
        for pattern in deny:
            if pattern.lower() in command.lower():
                return PolicyResult(
                    False,
                    f"Comando bloqueado: contiene patrón prohibido '{pattern}'"
                )

        # Allowlist
        allow = op_cfg.get("allow_cmds", [])
        if "*" in allow:
            return PolicyResult(True, "Comando permitido (allowlist abierta)")

        # ¿El comando empieza con algún prefijo permitido?
        cmd_stripped = command.strip()
        for allowed_cmd in allow:
            if cmd_stripped.startswith(allowed_cmd):
                return PolicyResult(True, f"Comando permitido: '{allowed_cmd}'")

        return PolicyResult(
            False,
            f"Comando no está en allowlist. Permitidos: {allow}"
        )
