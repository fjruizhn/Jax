"""
LAS MANOS — Motor Registry: catálogo de motores y capacidades.

Lee [motors.*] y [capabilities.*] de config.toml.
Construye un mapa en memoria que policy.py consulta.

Falla cerrado: si un motor está disabled o falta en config, no pasa.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MotorEntry:
    name: str
    enabled: bool
    provider: str
    api_key_env: str
    api_url: str
    model: str
    max_context_tokens: int
    sandbox_only: bool
    default_timeout_seconds: int
    supports_reasoning: bool
    reasoning_default_visibility: str = "none"


@dataclass
class CapabilityEntry:
    name: str
    allowed_motors: list[str]
    allowed_callers: list[str]
    risk_level: str
    sandbox_only: bool
    requires_human_gate: bool
    max_execution_minutes: int
    max_recursion_depth: int
    output_schema: str
    fallback_motor: str | None = None
    fallback_mode: str = "manual_only"
    forbidden_paths: list[str] = field(default_factory=list)


class MotorCatalog:
    def __init__(self, config: dict) -> None:
        self._motors: dict[str, MotorEntry] = {}
        self._capabilities: dict[str, CapabilityEntry] = {}
        self._load(config)

    def _load(self, config: dict) -> None:
        for name, cfg in config.get("motors", {}).items():
            self._motors[name] = MotorEntry(
                name=name,
                enabled=cfg.get("enabled", False),
                provider=cfg.get("provider", ""),
                api_key_env=cfg.get("api_key_env", ""),
                api_url=cfg.get("api_url", ""),
                model=cfg.get("model", ""),
                max_context_tokens=cfg.get("max_context_tokens", 0),
                sandbox_only=cfg.get("sandbox_only", True),
                default_timeout_seconds=cfg.get("default_timeout_seconds", 300),
                supports_reasoning=cfg.get("supports_reasoning", False),
                reasoning_default_visibility=cfg.get("reasoning_default_visibility", "none"),
            )
        for name, cfg in config.get("capabilities", {}).items():
            self._capabilities[name] = CapabilityEntry(
                name=name,
                allowed_motors=cfg.get("allowed_motors", []),
                allowed_callers=cfg.get("allowed_callers", []),
                risk_level=cfg.get("risk_level", "high"),
                sandbox_only=cfg.get("sandbox_only", True),
                requires_human_gate=cfg.get("requires_human_gate", True),
                max_execution_minutes=cfg.get("max_execution_minutes", 5),
                max_recursion_depth=cfg.get("max_recursion_depth", 0),
                output_schema=cfg.get("output_schema", ""),
                fallback_motor=cfg.get("fallback_motor"),
                fallback_mode=cfg.get("fallback_mode", "manual_only"),
                forbidden_paths=cfg.get("forbidden_paths", []),
            )

    def get_motor(self, name: str) -> MotorEntry | None:
        return self._motors.get(name)

    def get_capability(self, name: str) -> CapabilityEntry | None:
        return self._capabilities.get(name)

    def enabled_motors(self) -> list[str]:
        return [n for n, m in self._motors.items() if m.enabled]
