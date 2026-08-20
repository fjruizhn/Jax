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
import os
import subprocess
import tempfile
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

# Fase 4 (2026-08-19): write_file se suma. Reversibilidad en vez de gate
# humano -- WORKSPACE_ROOT es un repo git propio (ver T1 de la sesión),
# cada escritura autorizada commitea, el rollback es git reset --hard.
EXECUTABLE_TOOLS: frozenset[str] = frozenset({"read_file", "write_file"})

# Identidad del autor de los commits automáticos -- distinguible de Fernando
# (commits manuales) y de Hyde (que corre `claude -p` directo, sin pasar por
# este módulo). Fijo vía -c, nunca depende de ~/.gitconfig global.
_GIT_AUTHOR_NAME = "JAX Agent (tool_authority)"
_GIT_AUTHOR_EMAIL = "jax-agent@localhost"

# 200KB: generoso para código/config real, acota una lectura patológica
# (o un intento de exfiltrar algo grande) sin necesitar streaming para esta
# fase de un solo archivo, sin segundo turno.
MAX_READ_BYTES = 200_000
# Mismo orden de magnitud que MAX_READ_BYTES -- un archivo HTML/CSS/JS
# completo real (el caso que motivó todo esto) entra cómodo; una escritura
# más grande que esto en una sola llamada es la señal atípica, no el caso
# normal.
MAX_WRITE_BYTES = 200_000


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
    tool_call_id: str = "",
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
        # T3 (Fase 4, 2026-08-19): file_write ya NO tiene esto en 1 -- la
        # protección pasó a ser jail+forbidden_paths+git+auditoría posterior
        # (ver CONTEXT.md). El chequeo se queda genérico: una capability
        # futura que SÍ lo declare (ej. algo fuera del jail) sigue cerrada
        # por default, sin mecanismo de aprobación en este flujo.
        return await _reject(
            job_id=job_id, tool_name=tool_name, caller=caller, capability=capability_key,
            reason="requires_human_gate=1 y este flujo no tiene mecanismo de aprobación (blocked_human_gate)",
        )

    if tool_name not in EXECUTABLE_TOOLS:
        return await _reject(
            job_id=job_id, tool_name=tool_name, caller=caller, capability=capability_key,
            reason=f"'{tool_name}' está declarada (tools_catalog.py) y mapeada, pero no es ejecutable todavía",
        )

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

    if tool_name == "write_file":
        content = args.get("content")
        if not isinstance(content, str):
            return await _reject(
                job_id=job_id, tool_name=tool_name, caller=caller, capability=capability_key,
                reason="falta 'content' (string) en los argumentos",
            )
        return await _write_file(
            job_id=job_id, tool_name=tool_name, caller=caller, resolved=resolved,
            content=content, tool_call_id=tool_call_id,
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


def _git_commit_write(resolved: Path, *, job_id: str, tool_call_id: str) -> tuple[bool, str | None, str | None]:
    """Commitea UN archivo (git add -- <path> puntual, nunca -A -- así un
    commit del bucle no puede capturar trabajo a medias de Hyde escribiendo
    en paralelo en otra parte del mismo workspace). Devuelve (ok, sha, error).

    Mensaje lleva job_id y tool_call_id -- traza el commit hasta la
    iteración exacta del bucle que lo produjo (T1). Autor/committer fijos
    vía -c, no ~/.gitconfig (distinguible de Fernando/Hyde).

    Nunca lanza: un fallo de git (repo bloqueado, disco lleno) no debe
    tumbar el job -- la escritura en disco YA ocurrió, fail-closed no es
    retroactivo. El caller decide qué hacer con (ok=False, error=...)."""
    rel = resolved.relative_to(WORKSPACE_ROOT)
    base_cmd = ["git", "-C", str(WORKSPACE_ROOT), "-c", f"user.name={_GIT_AUTHOR_NAME}", "-c", f"user.email={_GIT_AUTHOR_EMAIL}"]
    try:
        add = subprocess.run(base_cmd + ["add", "--", str(rel)], capture_output=True, text=True, timeout=10)
        if add.returncode != 0:
            return False, None, f"git add falló: {add.stderr.strip()}"
        commit = subprocess.run(
            base_cmd + ["commit", "-m", f"tool_authority: write_file {rel} (job={job_id} tool_call={tool_call_id})"],
            capture_output=True, text=True, timeout=10,
        )
        if commit.returncode != 0:
            # "nothing to commit" pasa si el contenido nuevo es IDÉNTICO al
            # ya commiteado (el modelo reescribe lo mismo) -- no es un
            # fallo real, el árbol ya refleja el estado deseado. Cualquier
            # otro código de salida sí es un fallo real de git.
            if "nothing to commit" in commit.stdout + commit.stderr:
                head = subprocess.run(base_cmd + ["rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
                return True, (head.stdout.strip() if head.returncode == 0 else None), None
            return False, None, f"git commit falló: {commit.stderr.strip()}"
        sha = subprocess.run(base_cmd + ["rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        return True, (sha.stdout.strip() if sha.returncode == 0 else None), None
    except (subprocess.TimeoutExpired, OSError) as exc:
        return False, None, f"{type(exc).__name__}: {exc}"


async def _write_file(*, job_id: str, tool_name: str, caller: str, resolved: Path, content: str, tool_call_id: str) -> dict:
    size = len(content.encode("utf-8"))
    if size > MAX_WRITE_BYTES:
        return await _execution_error(
            job_id=job_id, tool_name=tool_name, caller=caller,
            reason=f"contenido excede el límite de escritura ({size} bytes > {MAX_WRITE_BYTES})",
        )

    # Creación de directorios: el jail ya validó el árbol completo resuelto
    # (resolve_jailed_path corre sobre resolved, que ya incluye los
    # componentes intermedios inexistentes -- Path.resolve(strict=False) los
    # normaliza igual sin poder seguir symlinks que todavía no existen, lo
    # cual es correcto: un componente que no existe no puede ser un symlink
    # que escape el jail).
    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return await _execution_error(job_id=job_id, tool_name=tool_name, caller=caller, reason=f"no se pudo crear el directorio: {exc}")

    # Escritura atómica: temp file en el MISMO directorio (garantiza que
    # os.replace sea un rename atómico dentro del mismo filesystem, no una
    # copia cross-device) + os.replace -- un fallo a mitad de escribir dejaría
    # el .tmp huérfano, nunca el archivo final truncado. Sobrescritura
    # permitida a propósito (T2): con git detrás, el contenido previo no se
    # pierde, queda en el commit anterior.
    try:
        fd, tmp_path = tempfile.mkstemp(dir=str(resolved.parent), prefix=f".{resolved.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
            os.replace(tmp_path, resolved)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:  # fail-soft: cleanup de tmp_path tras error real ya capturado arriba; el `raise` de abajo propaga la falla real, nadie depende de que este unlink haya funcionado
                pass
            raise
    except OSError as exc:
        return await _execution_error(job_id=job_id, tool_name=tool_name, caller=caller, reason=f"error de OS al escribir: {exc}")

    committed, sha, git_error = _git_commit_write(resolved, job_id=job_id, tool_call_id=tool_call_id)
    if not committed:
        # T1: la escritura YA ocurrió (arriba) -- no se revierte de forma
        # retroactiva. Se declara el hueco explícito (sin protección de git
        # hasta el próximo commit que sí funcione) en vez de mentir con
        # "executed" silencioso o con un "execution_error" que sugeriría que
        # el archivo no cambió.
        logger.error("tool_authority: write_file EJECUTADO pero SIN COMMITEAR job=%s path=%s error=%s", job_id, resolved, git_error)
        try:
            await event_append(job_id, "TOOL_CALL_WRITE_UNCOMMITTED", {
                "tool_name": tool_name, "caller": caller, "path": str(resolved.relative_to(WORKSPACE_ROOT)), "error": git_error,
            })
        except Exception:  # fail-soft: mismo criterio que _reject/_execution_error
            logger.error("tool_authority: no se pudo registrar TOOL_CALL_WRITE_UNCOMMITTED para job %s", job_id, exc_info=True)

    logger.info("tool_authority: write_file EJECUTADO job=%s path=%s (%d bytes) sha=%s", job_id, resolved, size, sha)
    return {
        "tool_name": tool_name, "decision": "executed", "reason": None,
        "content": f"Escrito: {resolved.relative_to(WORKSPACE_ROOT)} ({size} bytes)",
        "git_committed": committed, "git_sha": sha, "git_error": git_error,
        "bytes_written": size,
    }


def get_workspace_head() -> str | None:
    """HEAD actual del repo de workspace, o None si no se pudo leer (repo
    recién creado sin commits todavía, o git no responde). Usado por
    worker.py para anclar el rollback de UN job -- se llama ANTES de la
    primera escritura de ese job."""
    try:
        r = subprocess.run(
            ["git", "-C", str(WORKSPACE_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return r.stdout.strip() if r.returncode == 0 else None
    except (subprocess.TimeoutExpired, OSError):
        return None
