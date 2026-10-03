"""Raiz del workspace de JAX, SIN valor por defecto (falla cerrado).

`JAX_WORKSPACE_DIR` (/etc/jax/.env) es la unica fuente de verdad. El
2026-10-03 el workspace de produccion se movio de la ruta vieja bajo el home
a `/srv/jax-data/jax-workspace`; mientras la ruta vieja era el default en tres
sitios, un proceso que arrancara sin el `.env` escribia en silencio en otro
lugar y partia los datos. Ahora falta la variable => excepcion con el motivo.

Modulo liviano a proposito: solo stdlib, sin imports del repo, para que lo
puedan usar LAS MANOS (que no ve el paquete `jax`; se importa por el symlink
`las_manos/workspace_dir.py`, mismo patron que `db_connect_config.py`), los
jacobs y el procesamiento sin arrastrar dependencias.
"""
from __future__ import annotations

import os
from pathlib import Path

VARIABLE = "JAX_WORKSPACE_DIR"
_PREFIJO = (
    f"{VARIABLE} no está configurada (falta, está vacía, no es absoluta o no existe): "
    "agrégala a /etc/jax/.env; si corres a mano sin leer ese archivo, pásala en el "
    f"entorno con `{VARIABLE}=$(sudo -n grep '^{VARIABLE}=' /etc/jax/.env | cut -d= -f2-)`"
)


class WorkspaceNoConfigurado(RuntimeError):
    """JAX_WORKSPACE_DIR falta, esta vacia o no es una ruta absoluta."""


def workspace_dir() -> Path:
    """Devuelve `Path(JAX_WORKSPACE_DIR).resolve()` (debe existir y ser directorio)
    o lanza WorkspaceNoConfigurado."""
    bruto = os.environ.get(VARIABLE)
    valor = (bruto or "").strip()
    if not valor:
        raise WorkspaceNoConfigurado(_PREFIJO)
    if not os.path.isabs(valor):
        raise WorkspaceNoConfigurado(
            f"{_PREFIJO}: el valor debe ser una ruta absoluta, no {valor!r}"
        )
    ruta = Path(valor).resolve()
    # Mismo contrato que almacen.py de jax-platform: una ruta mal escrita en
    # .env falla en vez de crear un arbol nuevo en otro lugar.
    if not ruta.is_dir():
        raise WorkspaceNoConfigurado(
            f"{_PREFIJO} -- motivo: {str(ruta)!r} no existe o no es un directorio"
        )
    return ruta
