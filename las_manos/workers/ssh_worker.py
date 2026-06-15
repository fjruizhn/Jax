"""
LAS MANOS — SSH Worker.

Ejecuta comandos en servidores remotos vía SSH.
Soporta dry-run: muestra qué haría sin hacerlo.

EL FRENO VIVE EN LA CARRETERA, NO SOLO EN EL PORTÓN:
mientras un comando corre, un watcher concurrente sondea el kill switch
cada POLL_INTERVAL segundos. Si /etc/jax/PAUSE aparece a mitad de la
operación, el watcher mata el cliente SSH. Con `ssh -tt` (pty forzado),
sshd propaga SIGHUP al proceso remoto: una cadena `sleep 10 && touch`
muere en el sleep y el touch NUNCA llega a ejecutarse.

Todo comando que llega aquí YA pasó por el policy engine.
Este worker no decide — ejecuta lo aprobado, y se detiene si se lo piden.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

SSH_PORT = "58291"
SSH_USER = "fruiz"
# -tt fuerza pseudo-tty → al morir el cliente, sshd manda SIGHUP al remoto.
SSH_OPTS = ["-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=accept-new"]

# Configurado por server.py al arrancar. Si está seteado, el watcher vigila.
KILL_SWITCH_PATH: str | None = None
POLL_INTERVAL = 0.25  # segundos entre sondeos del kill switch

# Sentinela: distingue "no me pasaron nada" de "me pasaron None a propósito".
_UNSET = object()


def _normalize(raw: bytes) -> str:
    """Decodifica y normaliza CRLF→LF (el pty de -tt mete \\r)."""
    return raw.decode("utf-8", errors="replace").replace("\r\n", "\n")


async def _kill_switch_watcher(proc, kill_switch_path: str, aborted: dict) -> None:
    """Sondea el kill switch mientras el proceso corre. Si aparece, mata."""
    path = Path(kill_switch_path)
    while True:
        await asyncio.sleep(POLL_INTERVAL)
        if proc.returncode is not None:
            return  # el proceso ya terminó solo
        if path.exists():
            aborted["flag"] = True
            try:
                proc.kill()  # SIGKILL al cliente ssh → SIGHUP al remoto (pty)
            except ProcessLookupError:
                pass
            return


async def ssh_exec(
    host: str,
    command: str,
    dry_run: bool = False,
    timeout: float = 120.0,
    kill_switch_path: object = _UNSET,
) -> dict:
    """Ejecuta un comando vía SSH. Devuelve dict con resultado.

    Si kill_switch_path apunta a un archivo (o se usa el configurado en
    KILL_SWITCH_PATH), un watcher concurrente aborta la operación en vuelo
    cuando ese archivo aparece.
    """
    if kill_switch_path is _UNSET:
        kill_switch_path = KILL_SWITCH_PATH

    full_cmd = ["ssh", "-tt", "-p", SSH_PORT, *SSH_OPTS, f"{SSH_USER}@{host}", command]

    if dry_run:
        return {
            "dry_run": True,
            "would_execute": " ".join(full_cmd),
            "host": host,
            "command": command,
        }

    proc = None
    try:
        proc = await asyncio.create_subprocess_exec(
            *full_cmd,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        # El freno en la carretera: watcher concurrente (si hay kill switch).
        aborted: dict = {"flag": False}
        watcher = None
        if kill_switch_path:
            watcher = asyncio.create_task(
                _kill_switch_watcher(proc, str(kill_switch_path), aborted)
            )

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        finally:
            if watcher is not None:
                watcher.cancel()

        if aborted["flag"]:
            return {
                "dry_run": False,
                "success": False,
                "aborted": True,
                "kill_switch": True,
                "exit_code": proc.returncode,
                "host": host,
                "error": "KILL SWITCH activado en vuelo — operación abortada",
            }

        return {
            "dry_run": False,
            "success": proc.returncode == 0,
            "exit_code": proc.returncode,
            "stdout": _normalize(stdout),
            "stderr": _normalize(stderr),
            "host": host,
        }
    except asyncio.TimeoutError:
        if proc:
            proc.kill()
            await proc.wait()
        return {
            "dry_run": False,
            "success": False,
            "error": f"Timeout tras {timeout}s",
            "host": host,
        }
    except Exception as e:
        return {
            "dry_run": False,
            "success": False,
            "error": str(e),
            "host": host,
        }


async def kill_process(host: str, pid: int, dry_run: bool = False) -> dict:
    """Termina un proceso remoto por PID."""
    command = f"kill -TERM {pid}"
    return await ssh_exec(host, command, dry_run=dry_run, timeout=15.0)
