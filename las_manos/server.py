"""
LAS MANOS — Servidor.

El único punto por donde las facetas de JAX tocan el mundo real.
FastAPI escuchando SOLO en 127.0.0.1:7777 (sin exposición de red en fase 1).

Flujo de cada /execute:
    1. Kill switch  — ¿existe el archivo de JAX_KILL_SWITCH_PATH? Si sí, nada se ejecuta.
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

import asyncio
import json
import logging
import os
import time
import tomllib
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from crypto_secrets import decrypt_provider_keys_in_env
decrypt_provider_keys_in_env()

from audit import AuditLog, environment_from_target
from policy import PolicyEngine
from planner import Planner
from envelope import IntentEnvelope, validate as validate_envelope
from workers import ssh_worker, file_worker, rsync_worker
from interruptor import interruptor_activo, ruta_del_interruptor
import human_gate

# El freno ANTES de cualquier otra configuración (2026-09-16, frente B): sin
# JAX_KILL_SWITCH_PATH, LAS MANOS no arrancan (InterruptorSinConfigurar).
KILL_SWITCH = ruta_del_interruptor()


# ------------------------------------------------------------
#  Carga de configuración
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.toml"


def _load_environments() -> dict[str, list[str]]:
    """Topología real (qué IP pertenece a qué ambiente) vive en
    /etc/jax/.env vía JAX_ENV_<AMBIENTE>_HOSTS, no en config.toml (repo
    público, ronda 9). Sin la env var, ese ambiente queda vacío --
    PolicyEngine._resolve_env() rechaza cualquier host que caiga ahí
    (fail-closed, ver policy.py), nunca se cae a una IP conocida."""
    return {
        env: [h.strip() for h in os.getenv(f"JAX_ENV_{env.upper()}_HOSTS", "").split(",") if h.strip()]
        for env in ("staging", "prod", "bridge", "local")
    }


with open(CONFIG_PATH, "rb") as f:
    CONFIG = tomllib.load(f)
CONFIG["environments"] = _load_environments()

SERVER_CFG = CONFIG["server"]
GATE_CFG = CONFIG["human_gate"]

#: `config.toml` trae `audit_log` como ruta absoluta del home de producción
#: (`/home/fruiz/jax/las_manos/logs/audit.jsonl`) -- mismo criterio que
#: `JAX_WORKSPACE_DIR`/`WORKSPACE_ROOT` en tool_authority.py: un default de
#: producción, pero SIEMPRE overrideable, nunca el único camino. Sin esto,
#: cualquier test que importe `server` (aunque sea indirecto, vía
#: `from server import app`) dispara `AuditLog.__init__` -> `mkdir` sobre ESA
#: ruta literal, que en cualquier runner de CI o checkout que no sea
#: /home/fruiz/jax revienta con PermissionError o FileNotFoundError --
#: violación directa de "sin hardcoding" (2026-09-18, detector de cobertura:
#: las_manos/motor_registry/_authorize_facet_endpoint_test.py). `conftest.py`
#: de la raíz fija JAX_AUDIT_LOG_PATH a un temporal antes de cualquier import,
#: mismo patrón que JAX_KILL_SWITCH_PATH/JAX_FACET_SEAL_PATH/JAX_REPO_BASE.
AUDIT_LOG_PATH = os.getenv("JAX_AUDIT_LOG_PATH", SERVER_CFG["audit_log"])
audit = AuditLog(AUDIT_LOG_PATH)
policy = PolicyEngine(CONFIG)
planner = Planner(CONFIG)


# ------------------------------------------------------------
#  Human gate — tokens de aprobación de un solo uso (las_manos/human_gate.py).
#  Sin ruta HTTP de emisión: los emite las_manos/emitir_token_gate.py con la
#  credencial de la base. El TTL se valida al importar (fail-closed).
# ------------------------------------------------------------
human_gate.ttl_segundos(GATE_CFG)


# ------------------------------------------------------------
#  Modelo de request: el Intent Envelope (18 campos + carga operacional).
#  Definido en envelope.py. Ninguna llamada entra sin sobre completo.
# ------------------------------------------------------------


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
    return interruptor_activo(KILL_SWITCH)


# ------------------------------------------------------------
#  Aplicación
# ------------------------------------------------------------
from motor_registry.routes import router as motor_router
from jacobs.routes import router as jacobs_router
from jacobs import store as jacobs_store
from procesamiento_routes import router as procesamiento_router

app = FastAPI(
    title="LAS MANOS",
    description="Sistema de capacidades de JAX — en memoria de Jairo Urbina.",
    version="1.0.0",
)

# Autenticación de servicio (2026-09-17): deny by default, la identidad sale de
# la credencial y no del cuerpo. Sin las credenciales en /etc/jax/.env, LAS
# MANOS no arranca (EntornoInvalido). Ver las_manos/auth_servicio.py.
from auth_servicio import proteger  # noqa: E402
proteger(app)


@app.on_event("startup")
async def _jacobs_init() -> None:
    _jlog = logging.getLogger("jacobs")
    _jlog.setLevel(logging.INFO)
    if not _jlog.handlers:
        _h = logging.StreamHandler()
        _h.setFormatter(logging.Formatter("%(name)s %(levelname)s %(message)s"))
        _jlog.addHandler(_h)
    # Mismo tratamiento que el logger "jacobs" de arriba: sin un handler
    # propio, source=db (logger.info en credential_resolver.py) cae en
    # logging.lastResort (WARNING por default) y nunca llega a journald —
    # solo source=env_fallback y FAIL_CLOSED quedarían visibles. Necesario
    # para el criterio de 7 días de B1.4 (ver mismo fix en jax-platform).
    _credlog = logging.getLogger("credential_resolver")
    _credlog.setLevel(logging.INFO)
    if not _credlog.handlers:
        _ch = logging.StreamHandler()
        _ch.setFormatter(logging.Formatter("%(name)s %(levelname)s %(message)s"))
        _credlog.addHandler(_ch)

    # Frente F (2026-09-16): la config del contrato de sub-pipelines se valida
    # al arrancar. Un valor inválido en /etc/jax/.env tumba LAS MANOS acá
    # (fail-closed) en vez de descubrirse en el primer hijo de Ada.
    from jacobs.subpipelines import config_subpipelines
    config_subpipelines()

    # Pool de conexiones de Jacobs (2026-09-17): el tamano se valida ACA, antes
    # de conectar. JAX_JACOBS_DB_POOL_SIZE invalido tumba el arranque
    # (fail-closed). init_tables() crea el pool de este loop; se cierra en
    # _jacobs_shutdown.
    jacobs_store.tamanio_pool()

    await jacobs_store.init_tables()

    from motor_registry.routes import init_motor_catalog
    await init_motor_catalog()

    # T4 (2026-08-19): barrido al arranque (limpia lo que quedó de un
    # crash/restart previo, exactamente el caso de los zombies de Bug A)
    # + timer periódico en background para lo que se genera en caliente.
    from jacobs.reaper import reap_orphaned_pipelines, start_reaper_loop
    await reap_orphaned_pipelines()
    asyncio.create_task(start_reaper_loop())

    # El chequeo de consistencia codigo-vs-DB de timeouts se ELIMINO el
    # 2026-09-01 al deduplicar: existia para comparar `_CAPABILITY_TIMEOUT_SECONDS`
    # (jacobs/plan.py) contra `capability.max_execution_minutes` (DB). El dict
    # ya no existe -- el default sale de la DB, misma fila que el techo -- asi
    # que no hay dos fuentes que puedan divergir y no hay nada que chequear.
    # Junto con el chequeo se borro scripts/check_timeout_consistency.py.

    # se perdió DOS veces en 20h porque vivía dentro del árbol de jax/ y
    # cualquier limpieza del repo padre (filter-repo, git clean -x) se lo
    # llevaba puesto sin avisar. El fix real es la ubicación
    # (JAX_WORKSPACE_DIR fuera de ambos repos, ver DEUDA.md) -- este check
    # es la alarma si ese blindaje falla igual: sin repo git ahí,
    # write_file EJECUTA sin poder commitear (tool_authority.py ya lo
    # loguea fuerte por escritura), pero la garantía de reversibilidad
    # completa está rota desde el arranque, no solo en el primer write.
    from motor_registry.tool_authority import WORKSPACE_ROOT
    if not (WORKSPACE_ROOT / ".git").is_dir():
        _jlog.error(
            "workspace/ SIN repo git propio en %s -- write_file va a ejecutar "
            "SIN COMMITEAR (reversibilidad rota). No debería pasar con "
            "JAX_WORKSPACE_DIR blindado fuera de jax/ -- si esto se dispara, "
            "algo recreó el directorio sin su .git.",
            WORKSPACE_ROOT,
        )


@app.on_event("shutdown")
async def _cerrar_cliente_http() -> None:
    """E-24: el cliente HTTP compartido del proceso se cierra al apagar."""
    from cliente_http_compartido import cerrar_cliente_http
    await cerrar_cliente_http()


@app.on_event("shutdown")
async def _jacobs_shutdown() -> None:
    """Cierra el pool de conexiones de Jacobs: espera a que vuelvan las
    conexiones en uso y las cierra, en vez de dejar que el proceso corte los
    sockets a mitad de una consulta.

    Task 15b (2026-09-17): el pool del store de Jacobs
    (jacobs/store.py::conexion_del_pool) se crea perezosamente en el primer
    pedido; al apagar se cierra acá, en el mismo event loop, y no quedan
    conexiones abiertas contra MariaDB esperando su wait_timeout."""
    await jacobs_store.cerrar_pool()


app.include_router(motor_router)
app.include_router(jacobs_router)
app.include_router(procesamiento_router)

@app.exception_handler(RequestValidationError)
async def envelope_structural_rejection(request: Request, exc: RequestValidationError):
    """Rechazo ESTRUCTURAL del Envelope (capa Pydantic): campo faltante o mal
    tipado → 422 (condición 1 del contrato). Queda en el audit, igual que los
    rechazos semánticos: LAS MANOS se niega, y la negativa deja testigo."""
    campos = sorted({str(e["loc"][-1]) for e in exc.errors()})
    audit.log_envelope_rejected(
        request_id=None,
        reason=f"campos faltantes/mal-tipados: {campos}",
        layer="estructural",
    )
    return JSONResponse(
        status_code=422,
        content={"detail": "ENVELOPE_REJECTED (estructural)", "campos": campos},
    )


# ------------------------------------------------------------
#  Salud real (2026-09-16)
# ------------------------------------------------------------
# ANTES: /health devolvia {"status": "alive"} FIJO. Respondia «vivo» por el
# mero hecho de poder responder: un control que NO PUEDE FALLAR. Y no es un
# endpoint cualquiera —— lo miran loadtest/health.js (las pruebas de carga de
# LAS CUATRO DEL RENDIMIENTO), la Mesa para saber si LAS MANOS vive, y el
# dashboard. Con la base caida seguia diciendo alive y k6 seguia en verde: el
# p95 medido no era del servicio, era de FastAPI devolviendo un literal.
#
# La LOGICA vive en las_manos/salud.py, no aca: metida en este archivo, su test
# tenia que importar el servidor entero (FastAPI, planner, policy, workers,
# motor registry, jacobs) y en CI ese import fallaba —— los cinco tests se
# SALTABAN en silencio. El test que demuestra que /health puede ponerse rojo no
# corria justo donde importa.
from salud import Salud, comprobar_audit, comprobar_base  # noqa: E402

_salud = Salud({
    "base de datos": lambda: comprobar_base(jacobs_store.conexion),
    "log de auditoria": lambda: comprobar_audit(audit.log_path),
})


@app.get("/health")
async def health(response: Response) -> dict:
    """Latido del servicio. Comprueba las dependencias que LAS MANOS necesita
    para trabajar y devuelve 503 si alguna no responde.

    El kill switch se REPORTA pero no degrada: estar frenado a proposito es un
    estado deliberado, no una averia.
    """
    estado = await _salud.estado()
    if not estado["ok"]:
        response.status_code = 503
    return {
        "service": "LAS MANOS",
        "status": "alive" if estado["ok"] else "degraded",
        "kill_switch_active": _kill_switch_active(),
        "problemas": estado["fallos"],
        "comprobado_hace_s": _salud.comprobado_hace(),
        "cache_ttl_s": _salud._ttl,
    }


@app.get("/audit/tail")
async def audit_tail(n: int = 50) -> dict:
    """Thot consulta los últimos N eventos del log forense."""
    return {"events": audit.tail(n)}


@app.post("/plan")
async def preview_plan(req: IntentEnvelope) -> dict:
    """Previsualiza el plan SIN ejecutar nada. Para Fernando y Thot.
    También exige Envelope válido (capa semántica) antes de planear."""
    env_check = validate_envelope(req)
    if not env_check.ok:
        audit.log_envelope_rejected(
            request_id=None,
            reason=env_check.reason,
            environment=environment_from_target(req.target_environment),
            traffic_class=req.traffic_class,
            test_run_id=req.test_run_id,
        )
        raise HTTPException(status_code=422, detail=f"ENVELOPE_REJECTED: {env_check.reason}")
    plan = planner.plan(
        req.facet_id, req.requested_capability, req.target_host, req.params
    )
    return {"plan": plan.to_dict(), "rendered": plan.render()}


@app.post("/execute")
async def execute(req: IntentEnvelope) -> dict:
    """El corazón de LAS MANOS. Todo pasa por aquí, en orden, con testigo."""

    # Block 6: this legacy endpoint cannot be an alternate execution
    # authority.  A replayable decision still needs a governed authorization.
    raise HTTPException(status_code=410, detail="GOVERNED_EXECUTION_REQUIRED")

    job_id = req.trace_id  # el trace_id del sobre es el id de la intención

    # Procedencia forense (Thot Audit Watch): environment se INFIERE del sobre;
    # traffic_class y test_run_id los DECLARA la faceta. Una sola fuente, splat
    # en cada evento de esta intención.
    fx = {
        "environment":   environment_from_target(req.target_environment),
        "traffic_class": req.traffic_class,
        "test_run_id":   req.test_run_id,
    }

    # ---- 1) KILL SWITCH — antes de absolutamente todo ----
    if _kill_switch_active():
        audit.log_kill_switch(triggered_by=f"{req.facet_id}/{req.requested_capability}", **fx)
        raise HTTPException(
            status_code=423,  # Locked
            detail="KILL SWITCH ACTIVO — LAS MANOS están detenidas",
        )

    # ---- 2) Registrar la solicitud ----
    request_id = audit.log_request(
        facet=req.facet_id,
        operation=req.requested_capability,
        target_host=req.target_host,
        payload=req.params,
        job_id=job_id,
        **fx,
    )

    # ---- 2.5) INTENT ENVELOPE — capa semántica (9 condiciones de fallo cerrado).
    #          La capa estructural (Pydantic) ya rechazó campos faltantes/inválidos
    #          con 422 antes de llegar aquí. Esta valida las reglas CRUZADAS.
    #          Va ANTES de policy: rechaza toda llamada incompleta primero. ----
    env_check = validate_envelope(req)
    if not env_check.ok:
        audit.log_envelope_rejected(request_id=request_id, reason=env_check.reason, **fx)
        raise HTTPException(
            status_code=422, detail=f"ENVELOPE_REJECTED: {env_check.reason}"
        )
    audit.log_envelope(
        request_id=request_id,
        facet=req.facet_id,
        capability=req.requested_capability,
        target_environment=req.target_environment,
        risk_level=req.risk_level,
        **fx,
    )

    # ---- 3) Policy check ----
    command = _extract_command(req.requested_capability, req.params)
    result = policy.check(
        facet=req.facet_id,
        operation=req.requested_capability,
        target_host=req.target_host,
        command=command,
    )
    audit.log_policy_check(
        request_id=request_id,
        facet=req.facet_id,
        operation=req.requested_capability,
        target_host=req.target_host,
        allowed=result.ok,
        reason=result.reason,
        **fx,
    )
    if not result.ok:
        raise HTTPException(status_code=403, detail=result.reason)

    # ---- 4) Human gate (si la política lo exige) ----
    if result.requires_human_gate:
        veredicto = await human_gate.consumir_token_gate(
            req.approval_token, uso=f"execute:{request_id}",
        )
        audit.log_human_gate(
            request_id=request_id,
            approved=veredicto.aceptado,
            token_used=req.approval_token,
            **fx,
        )
        if not veredicto.aceptado:
            raise HTTPException(status_code=401, detail=f"Human gate: {veredicto.motivo.value}")

    # ---- 5) Dry-run (si la operación lo exige) ----
    dryrun_result = None
    if result.requires_dryrun:
        # Kill switch de nuevo: el estado pudo cambiar mientras esperábamos.
        if _kill_switch_active():
            audit.log_kill_switch(triggered_by=f"{req.facet_id}/{req.requested_capability}", **fx)
            raise HTTPException(status_code=423, detail="KILL SWITCH ACTIVO")
        dryrun_result = await dispatch(
            req.requested_capability, req.target_host, req.params, dry_run=True,
        )
        audit.log_dryrun(request_id=request_id, plan=dryrun_result, **fx)

    # ---- 6) Ejecución real ----
    # Último chequeo de kill switch: el freno está vivo hasta el último instante.
    if _kill_switch_active():
        audit.log_kill_switch(triggered_by=f"{req.facet_id}/{req.requested_capability}", **fx)
        raise HTTPException(status_code=423, detail="KILL SWITCH ACTIVO")

    try:
        exec_result = await dispatch(
            req.requested_capability, req.target_host, req.params, dry_run=False,
        )
    except KeyError as e:
        # Falta un parámetro requerido por el worker.
        audit.log_execution(
            request_id=request_id,
            success=False,
            error=f"Parámetro faltante: {e}",
            **fx,
        )
        raise HTTPException(status_code=400, detail=f"Parámetro faltante: {e}")
    except Exception as e:  # noqa: BLE001 — todo error queda en el log
        audit.log_execution(request_id=request_id, success=False, error=str(e), **fx)
        raise HTTPException(status_code=500, detail=str(e))

    # ---- 6b) ¿El watcher abortó en vuelo? Evento crítico. ----
    if exec_result.get("kill_switch"):
        audit.log_kill_switch(triggered_by=f"{req.facet_id}/{req.requested_capability} [en-vuelo]", **fx)
        audit.log_execution(
            request_id=request_id,
            success=False,
            error="Abortado por kill switch durante la ejecución",
            **fx,
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
        **fx,
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
