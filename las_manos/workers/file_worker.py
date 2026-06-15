"""
LAS MANOS — File Worker.

Lee y escribe archivos en servidores remotos vía SSH.
Toda escritura saca un snapshot previo (cp path path.bak.TIMESTAMP)
para que nada se pierda sin posibilidad de revertir.

Soporta dry-run: muestra qué haría sin tocar el disco remoto.

Todo lo que llega aquí YA pasó por el policy engine.
Este worker no decide — ejecuta lo aprobado y deja huella reversible.

"Una mano que escribe sin guardar lo que borra, no es una mano: es un cuchillo."
— Thot

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import base64
import shlex
from datetime import datetime, timezone

try:
    from .ssh_worker import ssh_exec
except ImportError:  # ejecución directa fuera del paquete
    from ssh_worker import ssh_exec


def _timestamp() -> str:
    """Sello de tiempo UTC compacto para nombres de snapshot."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


async def read_file(host: str, path: str, timeout: float = 60.0) -> dict:
    """Lee el contenido de un archivo remoto vía `cat`. Solo lectura."""
    command = f"cat -- {shlex.quote(path)}"
    result = await ssh_exec(host, command, dry_run=False, timeout=timeout)
    # Enriquecemos el resultado con metadatos de la operación
    result["operation"] = "read_file"
    result["path"] = path
    if result.get("success"):
        result["content"] = result.get("stdout", "")
    return result


async def list_dir(host: str, path: str, timeout: float = 30.0) -> dict:
    """Lista un directorio remoto con `ls -la`. Solo lectura."""
    command = f"ls -la -- {shlex.quote(path)}"
    result = await ssh_exec(host, command, dry_run=False, timeout=timeout)
    result["operation"] = "list_dir"
    result["path"] = path
    return result


async def write_file(
    host: str,
    path: str,
    content: str,
    dry_run: bool = False,
    snapshot: bool = True,
    timeout: float = 120.0,
) -> dict:
    """
    Escribe `content` en `path` del host remoto.

    Si snapshot=True y el archivo existe, primero hace
    `cp -p path path.bak.TIMESTAMP` para poder revertir.

    Escritura ATÓMICA (defensa en profundidad, pedido por Thot):
    el contenido se decodifica a `path.tmp` y luego `mv path.tmp path`.
    El mv en el mismo filesystem es atómico a nivel kernel: queda el
    archivo viejo o el nuevo completo, nunca uno truncado.
        - atomicidad  → previene el truncado a media escritura.
        - snapshot .bak → permite revertir un cambio bueno que salió malo.

    El contenido viaja codificado en base64 y se decodifica en el
    remoto, evitando problemas de quoting con caracteres especiales.
    """
    qpath = shlex.quote(path)
    tmp_path = f"{path}.tmp.{_timestamp()}"
    qtmp = shlex.quote(tmp_path)
    snapshot_path = f"{path}.bak.{_timestamp()}"
    qsnap = shlex.quote(snapshot_path)

    # Snapshot condicional: solo si el archivo ya existe.
    snapshot_cmd = ""
    if snapshot:
        snapshot_cmd = f"if [ -e {qpath} ]; then cp -p -- {qpath} {qsnap}; fi; "

    # Contenido codificado para transporte seguro.
    # Escribimos a .tmp y solo si tiene éxito hacemos el mv atómico (&&).
    b64 = base64.b64encode(content.encode("utf-8")).decode("ascii")
    write_cmd = (
        f"printf %s {shlex.quote(b64)} | base64 -d > {qtmp} "
        f"&& mv -- {qtmp} {qpath}"
    )

    full_command = snapshot_cmd + write_cmd

    if dry_run:
        # No tocamos nada: describimos el plan completo.
        preview = await ssh_exec(host, full_command, dry_run=True, timeout=timeout)
        preview["operation"] = "write_file"
        preview["path"] = path
        preview["snapshot"] = snapshot
        preview["snapshot_path"] = snapshot_path if snapshot else None
        preview["tmp_path"] = tmp_path
        preview["atomic"] = True
        preview["bytes_to_write"] = len(content.encode("utf-8"))
        return preview

    result = await ssh_exec(host, full_command, dry_run=False, timeout=timeout)
    result["operation"] = "write_file"
    result["path"] = path
    result["snapshot"] = snapshot
    result["snapshot_path"] = snapshot_path if snapshot else None
    result["tmp_path"] = tmp_path
    result["atomic"] = True
    result["bytes_written"] = len(content.encode("utf-8")) if result.get("success") else 0
    return result
