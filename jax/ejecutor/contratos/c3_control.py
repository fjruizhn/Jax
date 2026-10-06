"""Cliente del canal de control autenticado del proxy C3."""
from __future__ import annotations

import asyncio
import json
import os
import struct
from pathlib import Path


async def registrar_auditor_c5(*, mision_id: str, faceta: str, proveedor_id: str, local: bool,
                               modo: str, env=None) -> None:
    """Solicita al único escritor del C3 que registre la selección efectiva de auditor."""
    env = os.environ if env is None else env
    ruta = env.get("JAX_PROXY_CARRIL_C5_SOCKET", "").strip()
    if not ruta or not Path(ruta).is_absolute() or modo != "SOLO_ORDENES" or local:
        raise ValueError("registro_c3_c5_sin_configuracion")
    evento = {"evento": "c5_auditor_elegido", "mision_id": mision_id, "faceta": faceta,
              "proveedor_id": proveedor_id, "local": local, "modo": modo}
    cuerpo = json.dumps(evento, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if len(cuerpo) > 4096:
        raise ValueError("registro_c3_c5_evento_invalido")
    reader = writer = None
    try:
        reader, writer = await asyncio.wait_for(asyncio.open_unix_connection(ruta), 3)
        writer.write(struct.pack("!I", len(cuerpo)) + cuerpo)
        await asyncio.wait_for(writer.drain(), 3)
        respuesta = await asyncio.wait_for(reader.readline(), 3)
        if respuesta != b"OK\n":
            raise ValueError("registro_c3_c5_rechazado")
    except (OSError, asyncio.TimeoutError, asyncio.IncompleteReadError):
        raise ValueError("registro_c3_c5_no_disponible") from None
    finally:
        if writer is not None:
            writer.close()
            try:
                await writer.wait_closed()
            except OSError:  # fail-soft: el cierre local no cambia el resultado de la anotación C3
                pass
