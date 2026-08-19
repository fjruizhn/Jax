"""
LAS MANOS — Motor Registry: gate de autoridad para tool_calls (GAP 2, Fase 2).

Por qué acá y no en el transporte: executor.py:731-733 documenta que
_HTTP_FACETS (ada/thot) no pasan por la gobernanza del Motor Registry
(allowed_callers/requires_human_gate/sandbox_only) -- "riesgo real pero hoy
dormido". El transporte no garantiza el gate para todos los facets, así que
el gate vive en el BUCLE de tool-calling (acá), no en el transporte. GAP2
Fase1 ya reduce la superficie real a jax_local (gate literal de nombre de
motor en worker.py) -- pero este módulo no depende de eso: resuelve
allowed_callers real contra la capability, no confía en el gate de arriba.

Invariante P10 aplicado al lugar nuevo: cualquier ambigüedad en la
resolución de autoridad es RECHAZO, nunca aprobación implícita. Un tool_name
sin mapeo, una capability sin seed, un caller no listado -- todos rechazan,
ninguno "deja pasar por si acaso".

En honor al Prof. Raúl Jacobs.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from jacobs.store import event_append
from motor_registry.catalog import MotorCatalog

logger = logging.getLogger("motor_registry.tool_authority")

# Mismo root que jacobs/executor.py::HYDE_WORKSPACE_DIR -- no una convención
# paralela. No se importa executor.py directamente (módulo pesado, trae
# dependencias de pipeline que este gate no necesita); se documenta la
# fuente de verdad acá para que un cambio futuro de uno se note revisando
# el otro.
WORKSPACE_ROOT = Path("/home/fruiz/jax/workspace").resolve()

# tool_name -> capability key. Nunca un nombre de función libre -- si un
# tool nuevo se declara en tools_catalog.py sin agregarlo acá, queda
# RECHAZADO por default (ver resolve() abajo), no ejecutado con permisos
# implícitos.
TOOL_CAPABILITY_MAP: dict[str, str] = {
    "read_file": "file_read",
    "write_file": "file_write",
}

# Fase 2: SOLO read_file ejecuta de verdad. write_file está declarada
# (tools_catalog.py) y mapeada (arriba) pero queda fuera de EXECUTABLE_TOOLS
# a propósito -- Fase 4 la habilita, no un cambio de config silencioso.
EXECUTABLE_TOOLS: frozenset[str] = frozenset({"read_file"})

# 200KB: generoso para código/config real, acota una lectura patológica
# (o un intento de exfiltrar algo grande) sin necesitar streaming para esta
# fase de un solo archivo, sin segundo turno.
MAX_READ_BYTES = 200_000


async def _reject(*, job_id: str, tool_name: str, caller: str, reason: str, capability: str | None = None) -> dict:
    logger.warning("tool_authority: RECHAZADO job=%s tool=%s caller=%s razón=%s", job_id, tool_name, caller, reason)
    try:
        await event_append(job_id, "TOOL_CALL_REJECTED", {
            "tool_name": tool_name, "caller": caller, "capability": capability, "reason": reason,
        })
    except Exception:  # fail-soft: si jacobs_events no responde, el rechazo YA se logueó a WARNING arriba -- no se convierte un rechazo en una ejecución por un fallo de auditoría
        logger.error("tool_authority: no se pudo registrar TOOL_CALL_REJECTED para job %s", job_id, exc_info=True)
    return {"tool_name": tool_name, "decision": "rejected", "reason": reason, "content": None}


async def _execution_error(*, job_id: str, tool_name: str, caller: str, reason: str) -> dict:
    """Distinto de _reject: la autoridad SÍ lo permitió, el fallo es
    operativo (archivo ausente/binario/etc), no una violación de gate. Se
    audita en un event_type separado para no mezclar "no tenías permiso"
    con "tenías permiso pero el archivo no se pudo leer"."""
    logger.info("tool_authority: error de ejecución job=%s tool=%s razón=%s", job_id, tool_name, reason)
    try:
        await event_append(job_id, "TOOL_CALL_EXECUTION_ERROR", {
            "tool_name": tool_name, "caller": caller, "reason": reason,
        })
    except Exception:  # fail-soft: mismo criterio que _reject
        logger.error("tool_authority: no se pudo registrar TOOL_CALL_EXECUTION_ERROR para job %s", job_id, exc_info=True)
    return {"tool_name": tool_name, "decision": "execution_error", "reason": reason, "content": None}


def resolve_jailed_path(path_str: str, forbidden_paths: list[str]) -> tuple[Path | None, str | None]:
    """Resuelve path_str contra WORKSPACE_ROOT y lo valida. Devuelve
    (ruta_resuelta, None) si pasa, o (None, razón) si rechaza.

    Todo el chequeo -- jail Y forbidden_paths -- opera sobre la forma
    CANÓNICA (Path.resolve(), que sigue symlinks y normaliza '..'/'.'),
    nunca sobre la string cruda. Mismo bug de familia que '..': la defensa
    tiene que operar sobre lo que el filesystem realmente resuelve, no
    sobre lo que el modelo escribió (verificado con casos adversariales:
    './secrets/x', 'sub/../.env', symlinks -- ver _tool_authority_test.py).
    """
    if not path_str or not isinstance(path_str, str):
        return None, "path vacío o de tipo inválido"

    raw = Path(path_str)
    if raw.is_absolute():
        return None, f"ruta absoluta no permitida: '{path_str}'"

    candidate = WORKSPACE_ROOT / raw
    try:
        # strict=False: el archivo final puede no existir todavía (se
        # confirma más abajo, en la lectura) -- pero cada componente que
        # SÍ existe se resuelve de verdad, symlinks incluidos.
        resolved = candidate.resolve(strict=False)
    except (OSError, RuntimeError) as exc:  # RuntimeError: symlink loop
        return None, f"no se pudo resolver la ruta: {exc}"

    # Jail: la forma resuelta debe seguir DENTRO de WORKSPACE_ROOT. Esto
    # cubre '..' Y symlinks que apunten fuera -- ambos casos colapsan a
    # "resolved no es descendiente de WORKSPACE_ROOT" después de resolve().
    if resolved != WORKSPACE_ROOT and WORKSPACE_ROOT not in resolved.parents:
        return None, f"la ruta escapa del workspace (directo o vía symlink): resuelve a '{resolved}'"

    # forbidden_paths: mismo principio, sobre la forma canónica. Un prefijo
    # de config.toml/DB como "secrets/" se resuelve UNA vez acá contra el
    # mismo WORKSPACE_ROOT, y se compara con == o ascendencia real
    # (Path.parents), no con str.startswith sobre texto crudo.
    for forbidden in forbidden_paths or []:
        forbidden_resolved = (WORKSPACE_ROOT / forbidden.rstrip("/")).resolve(strict=False)
        if resolved == forbidden_resolved or forbidden_resolved in resolved.parents:
            return None, f"ruta prohibida (forbidden_paths): '{path_str}' resuelve dentro de '{forbidden}'"

    return resolved, None


async def authorize_and_execute_tool_call(
    *, tool_name: str, arguments_json: str, caller: str, job_id: str, catalog: MotorCatalog,
) -> dict:
    """Punto de entrada único: resuelve autoridad y, si corresponde,
    ejecuta. Nunca lanza -- toda salida es un dict {tool_name, decision,
    reason, content}, decision en {"executed","execution_error","rejected"}."""
    capability_key = TOOL_CAPABILITY_MAP.get(tool_name)
    if capability_key is None:
        return await _reject(
            job_id=job_id, tool_name=tool_name, caller=caller,
            reason=f"'{tool_name}' no mapea a ninguna capability (TOOL_CAPABILITY_MAP)",
        )

    cap = catalog.get_capability(capability_key)
    if cap is None:
        # Ambigüedad real: el mapeo existe en código pero el seed de DB no
        # -- P10 dice rechazo, no "asumir permisos por default".
        return await _reject(
            job_id=job_id, tool_name=tool_name, caller=caller, capability=capability_key,
            reason=f"capability '{capability_key}' no encontrada en el catálogo (seed pendiente)",
        )

    if caller not in cap.allowed_callers:
        return await _reject(
            job_id=job_id, tool_name=tool_name, caller=caller, capability=capability_key,
            reason=f"caller '{caller}' no está en allowed_callers de '{capability_key}' ({cap.allowed_callers})",
        )

    if cap.requires_human_gate:
        # Ningún tool ejecutable hoy mapea a una capability con esto en 1
        # (file_read=0, write_file no llega acá -- ver check de abajo) pero
        # el chequeo va primero y es genérico: un tool futuro mapeado a una
        # capability gateada no debe colarse por descuido de orden.
        return await _reject(
            job_id=job_id, tool_name=tool_name, caller=caller, capability=capability_key,
            reason="requires_human_gate=1 y este flujo no tiene mecanismo de aprobación (blocked_human_gate)",
        )

    if tool_name not in EXECUTABLE_TOOLS:
        return await _reject(
            job_id=job_id, tool_name=tool_name, caller=caller, capability=capability_key,
            reason=f"'{tool_name}' está declarada (tools_catalog.py) y mapeada, pero no es ejecutable en Fase 2",
        )

    # A partir de acá: únicamente read_file, autorizado. Parseo de
    # argumentos y jail.
    try:
        args: dict[str, Any] = json.loads(arguments_json) if arguments_json else {}
    except (json.JSONDecodeError, TypeError) as exc:
        return await _reject(
            job_id=job_id, tool_name=tool_name, caller=caller, capability=capability_key,
            reason=f"arguments no es JSON válido: {exc}",
        )

    path_str = args.get("path")
    resolved, jail_reason = resolve_jailed_path(path_str, cap.forbidden_paths)
    if jail_reason is not None:
        return await _reject(
            job_id=job_id, tool_name=tool_name, caller=caller, capability=capability_key,
            reason=jail_reason,
        )

    return await _read_file(job_id=job_id, tool_name=tool_name, caller=caller, resolved=resolved)


async def _read_file(*, job_id: str, tool_name: str, caller: str, resolved: Path) -> dict:
    if not resolved.exists():
        return await _execution_error(job_id=job_id, tool_name=tool_name, caller=caller, reason="archivo no encontrado")
    if resolved.is_dir():
        return await _execution_error(job_id=job_id, tool_name=tool_name, caller=caller, reason="la ruta es un directorio, no un archivo")

    try:
        size = resolved.stat().st_size
    except OSError as exc:
        return await _execution_error(job_id=job_id, tool_name=tool_name, caller=caller, reason=f"no se pudo leer metadata del archivo: {exc}")
    if size > MAX_READ_BYTES:
        return await _execution_error(
            job_id=job_id, tool_name=tool_name, caller=caller,
            reason=f"archivo excede el límite de lectura ({size} bytes > {MAX_READ_BYTES})",
        )

    try:
        raw = resolved.read_bytes()
    except PermissionError:
        return await _execution_error(job_id=job_id, tool_name=tool_name, caller=caller, reason="sin permisos de lectura")
    except OSError as exc:
        return await _execution_error(job_id=job_id, tool_name=tool_name, caller=caller, reason=f"error de OS al leer: {exc}")

    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError:
        return await _execution_error(job_id=job_id, tool_name=tool_name, caller=caller, reason="archivo binario -- Fase 2 solo lee texto UTF-8")

    logger.info("tool_authority: read_file EJECUTADO job=%s path=%s (%d bytes)", job_id, resolved, size)
    return {"tool_name": tool_name, "decision": "executed", "reason": None, "content": content}
