"""
LAS MANOS — SSH Worker.

Ejecuta comandos en servidores remotos vía SSH.
Soporta dry-run: muestra qué haría sin hacerlo.

EL FRENO VIVE EN LA CARRETERA, NO SOLO EN EL PORTÓN:
mientras un comando corre, un watcher concurrente sondea el kill switch
cada POLL_INTERVAL segundos. Si el archivo de JAX_KILL_SWITCH_PATH aparece a mitad de la
operación, el watcher mata el cliente SSH. Con `ssh -tt` (pty forzado),
sshd propaga SIGHUP al proceso remoto: una cadena `sleep 10 && touch`
muere en el sleep y el touch NUNCA llega a ejecutarse.

Todo comando que llega aquí YA pasó por el policy engine.
Este worker no decide — ejecuta lo aprobado, y se detiene si se lo piden.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

from interruptor import interruptor_activo, ruta_del_interruptor

# Puerto/usuario reales viven en /etc/jax/.env (repo público, ronda 9) --
# sin JAX_SSH_USER seteado, el comando ssh queda mal formado ("@host") y
# falla ruidosamente en vez de conectar con un usuario adivinado.
SSH_PORT = os.getenv("JAX_SSH_PORT", "22")
SSH_USER = os.getenv("JAX_SSH_USER", "")
# -tt fuerza pseudo-tty → al morir el cliente, sshd manda SIGHUP al remoto.
SSH_OPTS = ["-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=accept-new"]

POLL_INTERVAL = 0.25  # segundos entre sondeos del kill switch


def _normalize(raw: bytes) -> str:
    """Decodifica y normaliza CRLF→LF (el pty de -tt mete \\r)."""
    return raw.decode("utf-8", errors="replace").replace("\r\n", "\n")


async def _kill_switch_watcher(proc, kill_switch_path: str, aborted: dict) -> None:
    """Sondea el kill switch mientras el proceso corre. Si aparece (o no se lo
    puede mirar: interruptor_activo falla cerrado), mata."""
    while True:
        await asyncio.sleep(POLL_INTERVAL)
        if proc.returncode is not None:
            return  # el proceso ya terminó solo
        if interruptor_activo(kill_switch_path):
            aborted["flag"] = True
            try:
                proc.kill()  # SIGKILL al cliente ssh → SIGHUP al remoto (pty)
            except ProcessLookupError:  # fail-soft: kill() sobre un proceso que ya pudo haber terminado solo entre el chequeo y el kill (TOCTOU benigno)
                pass
            return


async def ssh_exec(
    host: str,
    command: str,
    dry_run: bool = False,
    timeout: float = 120.0,
    kill_switch_path: str | Path | None = None,
) -> dict:
    """Ejecuta un comando vía SSH. Devuelve dict con resultado.

    Un watcher concurrente vigila el freno (por defecto, el de
    JAX_KILL_SWITCH_PATH) y aborta la operación en vuelo si aparece. Sin la
    variable lanza InterruptorSinConfigurar antes de ejecutar nada.
    """
    if kill_switch_path is None:
        kill_switch_path = ruta_del_interruptor()

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

        # El freno en la carretera: watcher concurrente, SIEMPRE.
        aborted: dict = {"flag": False}
        watcher = asyncio.create_task(_kill_switch_watcher(proc, str(kill_switch_path), aborted))

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        finally:
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
    except Exception as e:  # fail-soft: traduce el fallo a success=False con el error; jamas reporta un comando como ejecutado con exito (server.py y file_worker.py deciden por 'success'), la direccion del error es cerrada
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
