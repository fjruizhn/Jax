"""
LAS MANOS — Rsync Worker.

Transfiere archivos entre servidores remotos.

Decisión de diseño (Thot): SIEMPRE se hace *pull* desde el destino.
El host de destino es quien jala los datos del origen. Es más seguro:
ningún servidor empuja a otro: cada uno solo recibe lo que pide, y el
comando se ejecuta en el lado que sufre el cambio.

Soporta dry-run nativo de rsync (--dry-run): rsync calcula y muestra
exactamente qué transferiría sin mover un solo byte.

Todo lo que llega aquí YA pasó por el policy engine y por el human gate.
rsync puede destruir datos: este worker nunca usa --delete por su cuenta.

"Mover datos es fácil. Moverlos sin poder deshacerlo es imprudencia."
— Jekyll

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import shlex

try:
    from .ssh_worker import ssh_exec, SSH_PORT, SSH_USER, SSH_OPTS
except ImportError:  # ejecución directa fuera del paquete
    from ssh_worker import ssh_exec, SSH_PORT, SSH_USER, SSH_OPTS


# Flags base: archivo, recursivo, comprime, preserva permisos/tiempos.
# NUNCA --delete: la destrucción de datos no se automatiza.
RSYNC_FLAGS = "-avz --human-readable"


async def rsync_transfer(
    origen_host: str,
    origen_path: str,
    destino_host: str,
    destino_path: str,
    dry_run: bool = False,
    timeout: float = 600.0,
) -> dict:
    """
    Transfiere `origen_host:origen_path` → `destino_host:destino_path`.

    El comando rsync se ejecuta EN el host de destino (pull): el destino
    se conecta por SSH al origen y jala los archivos. Puerto SSH 58291.

    dry_run=True añade --dry-run nativo de rsync: muestra qué pasaría.
    """
    # rsync sobre SSH al origen, en el puerto custom y con opciones seguras.
    inner_ssh = "ssh -p {port} {opts}".format(
        port=SSH_PORT,
        opts=" ".join(SSH_OPTS),
    )

    flags = RSYNC_FLAGS
    if dry_run:
        flags = f"{flags} --dry-run"

    remote_source = f"{SSH_USER}@{origen_host}:{shlex.quote(origen_path)}"

    # Este es el comando que correrá DENTRO del host de destino.
    rsync_cmd = (
        f"rsync {flags} -e {shlex.quote(inner_ssh)} "
        f"{remote_source} {shlex.quote(destino_path)}"
    )

    # Ejecutamos el rsync en el destino vía ssh_exec.
    # Si dry_run=True, NO usamos el dry_run de ssh_exec: queremos que rsync
    # corra de verdad con su propio --dry-run y nos devuelva el plan real.
    result = await ssh_exec(destino_host, rsync_cmd, dry_run=False, timeout=timeout)

    result["operation"] = "rsync"
    result["dry_run"] = dry_run
    result["origen"] = f"{origen_host}:{origen_path}"
    result["destino"] = f"{destino_host}:{destino_path}"
    result["mode"] = "pull-desde-destino"
    return result
