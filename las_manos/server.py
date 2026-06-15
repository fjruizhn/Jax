"""
LAS MANOS — Servidor.

El único punto por donde las facetas de JAX tocan el mundo real.
FastAPI escuchando SOLO en 127.0.0.1:7777 (sin exposición de red en fase 1).

Flujo de cada /execute:
    1. Kill switch  — ¿existe /etc/jax/PAUSE? Si sí, nada se ejecuta.
    2. audit.log_request
    3. policy.check — faceta, ambiente, operación, comando.
    4. human gate   — si la operación lo exige, validar token de Fernando.
    5. dry-run      — si la operación lo exige, ejecutar simulación primero.
    6. ejecución    — el worker real corre lo aprobado.
    7. audit.log_execution

Las manos pueden moverse, pero siempre frente a un testigo (el log)
y siempre con un freno al alcance (el kill switch).

"Dame manos que se detengan cuando se lo pido, y confiaré en ellas."
— Fernando

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import json
import time
import uuid
import secrets
import hashlib
import tomllib
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from audit import AuditLog
from policy import PolicyEngine
from planner import Planner
from workers import ssh_worker, file_worker, rsync_worker


# ------------------------------------------------------------
#  Carga de configuración
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.toml"

with open(CONFIG_PATH, "rb") as f:
    CONFIG = tomllib.load(f)

SERVER_CFG = CONFIG["server"]
GATE_CFG = CONFIG["human_gate"]
KILL_SWITCH = Path(SERVER_CFG["kill_switch_path"])

audit = AuditLog(SERVER_CFG["audit_log"])
policy = PolicyEngine(CONFIG)
planner = Planner(CONFIG)

# El freno en la carretera: el ssh_worker vigila este path durante CADA
# ejecución y aborta en vuelo si aparece. Lo configuramos al arrancar.
ssh_worker.KILL_SWITCH_PATH = str(KILL_SWITCH)


# ------------------------------------------------------------
#  Human gate — tokens de aprobación de un solo uso
# ------------------------------------------------------------
class HumanGate:
    """Tokens efímeros que Fernando genera para aprobar operaciones."""

    def __init__(self, ttl: int, length: int, gate_log: str) -> None:
        self.ttl = ttl
        self.length = length
        self.gate_log = Path(gate_log)
        self.gate_log.parent.mkdir(parents=True, exist_ok=True)
        # token -> {"expires": epoch, "used": bool}
        self._tokens: dict[str, dict] = {}

    def _log(self, entry: dict) -> None:
        entry["@timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with open(self.gate_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def issue(self) -> dict:
        """Genera un token nuevo. TTL en segundos según config."""
        token = secrets.token_hex(self.length // 2)
        expires = time.time() + self.ttl
        self._tokens[token] = {"expires": expires, "used": False}
        self._log({
            "event": "TOKEN_ISSUED",
            "token_hash": hashlib.sha256(token.encode()).hexdigest()[:16],
            "ttl_seconds": self.ttl,
        })
        return {"token": token, "ttl_seconds": self.ttl, "expires_epoch": expires}

    def validate(self, token: str | None) -> tuple[bool, str]:
        """Valida y consume un token. Un token solo sirve una vez."""
        if not token:
            return False, "Falta human_gate_token para una operación que lo requiere"
        rec = self._tokens.get(token)
        if rec is None:
            return False, "Token desconocido o ya descartado"
        if rec["used"]:
            return False, "Token ya usado (un solo uso)"
        if time.time() > rec["expires"]:
            del self._tokens[token]
            return False, "Token expirado"
        rec["used"] = True  # consumido
        self._log({
            "event": "TOKEN_CONSUMED",
            "token_hash": hashlib.sha256(token.encode()).hexdigest()[:16],
        })
        return True, "Token válido"


gate = HumanGate(
    ttl=GATE_CFG["token_ttl_seconds"],
    length=GATE_CFG["token_length"],
    gate_log=GATE_CFG["gate_log"],
)


# ------------------------------------------------------------
#  Modelos de request
# ------------------------------------------------------------
class ExecuteRequest(BaseModel):
    facet: str
    operation: str
    target_host: str
    params: dict = Field(default_factory=dict)
    human_gate_token: str | None = None


# ------------------------------------------------------------
#  Despacho a workers — traduce (operation, params) → coroutine
# ------------------------------------------------------------
async def dispatch(operation: str, target_host: str, params: dict, dry_run: bool) -> dict:
    """Llama al worker correcto. Las operaciones de lectura ignoran dry_run."""

    if operation in ("ssh_exec_readonly", "ssh_exec"):
        return await ssh_worker.ssh_exec(
            target_host, params["command"], dry_run=dry_run,
        )

    if operation == "read_file":
        return await file_worker.read_file(target_host, params["path"])

    if operation == "list_dir":
        return await file_worker.list_dir(target_host, params["path"])

    if operation == "write_file":
        return await file_worker.write_file(
            target_host,
            params["path"],
            params["content"],
            dry_run=dry_run,
            snapshot=params.get("snapshot", True),
        )

    if operation == "rsync":
        return await rsync_worker.rsync_transfer(
            origen_host=params["origen_host"],
            origen_path=params["origen_path"],
            destino_host=target_host,
            destino_path=params["destino_path"],
            dry_run=dry_run,
        )

    if operation == "kill_process":
        return await ssh_worker.kill_process(
            target_host, int(params["pid"]), dry_run=dry_run,
        )

    if operation == "audit_log_read":
        return {"success": True, "events": audit.tail(int(params.get("n", 50)))}

    # Operaciones declaradas en config pero sin worker todavía.
    return {
        "success": False,
        "error": f"Operación '{operation}' aún no tiene worker implementado",
    }


def _extract_command(operation: str, params: dict) -> str | None:
    """El policy engine valida comando solo para operaciones ssh."""
    if operation in ("ssh_exec_readonly", "ssh_exec"):
        return params.get("command")
    return None


def _kill_switch_active() -> bool:
    return KILL_SWITCH.exists()


# ------------------------------------------------------------
#  Aplicación
# ------------------------------------------------------------
app = FastAPI(
    title="LAS MANOS",
    description="Sistema de capacidades de JAX — en memoria de Jairo Urbina.",
    version="1.0.0",
)


@app.get("/health")
async def health() -> dict:
    """Latido del servicio. Reporta también el estado del kill switch."""
    return {
        "service": "LAS MANOS",
        "status": "alive",
        "kill_switch_active": _kill_switch_active(),
    }


@app.post("/human_gate/token")
async def human_gate_token() -> dict:
    """Fernando genera un token de aprobación (un solo uso, TTL config)."""
    return gate.issue()


@app.get("/audit/tail")
async def audit_tail(n: int = 50) -> dict:
    """Thot consulta los últimos N eventos del log forense."""
    return {"events": audit.tail(n)}


@app.post("/plan")
async def preview_plan(req: ExecuteRequest) -> dict:
    """Previsualiza el plan SIN ejecutar nada. Para Fernando y Thot."""
    plan = planner.plan(req.facet, req.operation, req.target_host, req.params)
    return {"plan": plan.to_dict(), "rendered": plan.render()}


@app.post("/execute")
async def execute(req: ExecuteRequest) -> dict:
    """El corazón de LAS MANOS. Todo pasa por aquí, en orden, con testigo."""

    job_id = uuid.uuid4().hex[:12]

    # ---- 1) KILL SWITCH — antes de absolutamente todo ----
    if _kill_switch_active():
        audit.log_kill_switch(triggered_by=f"{req.facet}/{req.operation}")
        raise HTTPException(
            status_code=423,  # Locked
            detail="KILL SWITCH ACTIVO (/etc/jax/PAUSE) — LAS MANOS están detenidas",
        )

    # ---- 2) Registrar la solicitud ----
    request_id = audit.log_request(
        facet=req.facet,
        operation=req.operation,
        target_host=req.target_host,
        payload=req.params,
        job_id=job_id,
    )

    # ---- 3) Policy check ----
    command = _extract_command(req.operation, req.params)
    result = policy.check(
        facet=req.facet,
        operation=req.operation,
        target_host=req.target_host,
        command=command,
    )
    audit.log_policy_check(
        request_id=request_id,
        facet=req.facet,
        operation=req.operation,
        target_host=req.target_host,
        allowed=result.allowed,
        reason=result.reason,
    )
    if not result.allowed:
        raise HTTPException(status_code=403, detail=result.reason)

    # ---- 4) Human gate (si la política lo exige) ----
    if result.requires_human_gate:
        ok, reason = gate.validate(req.human_gate_token)
        audit.log_human_gate(
            request_id=request_id,
            approved=ok,
            token_used=req.human_gate_token,
        )
        if not ok:
            raise HTTPException(status_code=401, detail=f"Human gate: {reason}")

    # ---- 5) Dry-run (si la operación lo exige) ----
    dryrun_result = None
    if result.requires_dryrun:
        # Kill switch de nuevo: el estado pudo cambiar mientras esperábamos.
        if _kill_switch_active():
            audit.log_kill_switch(triggered_by=f"{req.facet}/{req.operation}")
            raise HTTPException(status_code=423, detail="KILL SWITCH ACTIVO")
        dryrun_result = await dispatch(
            req.operation, req.target_host, req.params, dry_run=True,
        )
        audit.log_dryrun(request_id=request_id, plan=dryrun_result)

    # ---- 6) Ejecución real ----
    # Último chequeo de kill switch: el freno está vivo hasta el último instante.
    if _kill_switch_active():
        audit.log_kill_switch(triggered_by=f"{req.facet}/{req.operation}")
        raise HTTPException(status_code=423, detail="KILL SWITCH ACTIVO")

    try:
        exec_result = await dispatch(
            req.operation, req.target_host, req.params, dry_run=False,
        )
    except KeyError as e:
        # Falta un parámetro requerido por el worker.
        audit.log_execution(
            request_id=request_id,
            success=False,
            error=f"Parámetro faltante: {e}",
        )
        raise HTTPException(status_code=400, detail=f"Parámetro faltante: {e}")
    except Exception as e:  # noqa: BLE001 — todo error queda en el log
        audit.log_execution(request_id=request_id, success=False, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

    # ---- 6b) ¿El watcher abortó en vuelo? Evento crítico. ----
    if exec_result.get("kill_switch"):
        audit.log_kill_switch(triggered_by=f"{req.facet}/{req.operation} [en-vuelo]")
        audit.log_execution(
            request_id=request_id,
            success=False,
            error="Abortado por kill switch durante la ejecución",
        )
        raise HTTPException(
            status_code=423,
            detail="KILL SWITCH activado en vuelo — operación abortada",
        )

    # ---- 7) Registrar resultado ----
    audit.log_execution(
        request_id=request_id,
        success=bool(exec_result.get("success", False)),
        stdout=exec_result.get("stdout", ""),
        stderr=exec_result.get("stderr", ""),
        exit_code=exec_result.get("exit_code"),
        error=exec_result.get("error"),
    )

    return {
        "request_id": request_id,
        "job_id": job_id,
        "policy": result.reason,
        "dry_run": dryrun_result,
        "result": exec_result,
    }


# ------------------------------------------------------------
#  Arranque directo: python server.py
#  (en producción: uvicorn server:app --host 127.0.0.1 --port 7777)
# ------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=SERVER_CFG["host"],
        port=SERVER_CFG["port"],
        log_level="info",
    )
