"""
LAS MANOS — Motor Registry: catálogo de motores y capacidades.

Lee [motors.*] y [capabilities.*] de config.toml.
Construye un mapa en memoria que policy.py consulta.

Falla cerrado: si un motor está disabled o falta en config, no pasa.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import aiomysql


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
    # Bug real 2026-08-10: sin este limite explicito, un motor de
    # razonamiento puede gastar todo el completion budget en
    # reasoning_content y dejar `content` cortado a mitad de palabra —
    # reproducido en vivo contra la API real de Moonshot. 0 = no mandar
    # max_tokens (compat con motores sin este campo en config.toml).
    max_tokens: int = 0
    transport: str = "http_openai_compat"
    model_ref: int = 0
    provider_id: str = ""


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
                max_tokens=cfg.get("max_tokens", 0),
                transport=cfg.get("transport", "http_openai_compat"),
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

    @classmethod
    async def from_db(cls) -> "MotorCatalog":
        """Carga motor/capability/capability_motor desde la DB compartida
        jax_memory -- mismo pool/patron de conexion que credential_resolver.py.
        Reemplaza la lectura de config.toml (TOML queda solo para
        [server]/kill_switch_path y lo que routes.py todavia usa aparte)."""
        conn = await aiomysql.connect(
            host=os.getenv("JAX_DB_HOST", "localhost"),
            port=int(os.getenv("JAX_DB_PORT", "3306")),
            user=os.getenv("JAX_DB_USER", ""),
            password=os.getenv("JAX_DB_PASSWORD", ""),
            db=os.getenv("JAX_DB_NAME", "jax_memory"),
            charset="utf8mb4",
            autocommit=True,
        )
        try:
            instance = cls.__new__(cls)
            instance._motors = {}
            instance._capabilities = {}
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT m.`key`, m.model_ref, mo.provider_id, mo.model_id, p.base_url, "
                    "       m.transport, m.max_tokens, m.default_timeout_seconds, "
                    "       m.supports_reasoning, m.reasoning_default_visibility, "
                    "       m.sandbox_only, m.status "
                    "FROM motor m "
                    "JOIN model mo ON mo.id = m.model_ref "
                    "JOIN provider p ON p.id = mo.provider_id"
                )
                # api_url viene de provider.base_url (JOIN arriba) -- sin esto
                # _call_http_openai_compat (Task 3) arma "/chat/completions" sin
                # host, porque MotorEntry.api_url nunca se pobló desde ningun lado.
                for (key, model_ref, provider_id, model_id, base_url, transport, max_tokens,
                     timeout, reasoning, visibility, sandbox, status) in await cur.fetchall():
                    instance._motors[key] = MotorEntry(
                        name=key,
                        enabled=(status == "active"),
                        provider=provider_id,
                        api_key_env="",
                        api_url=base_url or "",
                        model=model_id,
                        max_context_tokens=0,
                        sandbox_only=bool(sandbox),
                        default_timeout_seconds=timeout,
                        supports_reasoning=bool(reasoning),
                        reasoning_default_visibility=visibility,
                        max_tokens=max_tokens or 0,
                        transport=transport,
                        model_ref=model_ref,
                        provider_id=provider_id,
                    )

                await cur.execute(
                    "SELECT `key`, risk_level, sandbox_only, requires_human_gate, "
                    "       max_execution_minutes, max_recursion_depth, output_schema, "
                    "       fallback_motor, fallback_mode, allowed_callers, forbidden_paths "
                    "FROM capability"
                )
                cap_rows = await cur.fetchall()

                await cur.execute(
                    "SELECT capability_key, motor_key FROM capability_motor ORDER BY capability_key, priority ASC"
                )
                motor_rows = await cur.fetchall()

            import json as _json
            allowed_by_cap: dict[str, list[str]] = {}
            for capability_key, motor_key in motor_rows:
                allowed_by_cap.setdefault(capability_key, []).append(motor_key)

            for (key, risk_level, sandbox_only, gate, max_exec, max_rec, schema,
                 fallback_motor, fallback_mode, callers, forbidden) in cap_rows:
                instance._capabilities[key] = CapabilityEntry(
                    name=key,
                    allowed_motors=allowed_by_cap.get(key, []),
                    allowed_callers=_json.loads(callers) if callers else [],
                    risk_level=risk_level,
                    sandbox_only=bool(sandbox_only),
                    requires_human_gate=bool(gate),
                    max_execution_minutes=max_exec,
                    max_recursion_depth=max_rec,
                    output_schema=schema or "",
                    fallback_motor=fallback_motor,
                    fallback_mode=fallback_mode or "manual_only",
                    forbidden_paths=_json.loads(forbidden) if forbidden else [],
                )
            return instance
        finally:
            conn.close()
