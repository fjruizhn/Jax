# jax/ejecutor/contratos/pausa.py
"""La pausa PROPIA del Ejecutor y el latido del vigía de C5.

DECISIÓN (plan 4 de SP1, 2026-09-17; corrige el punto 5 de «lo que el spec dice mal»
del índice): C5 NO escribe el interruptor global de JAX (`JAX_KILL_SWITCH_PATH`).
- Ese archivo frena TODO JAX (la Mesa responde 423): un falso positivo del auditor
  —un LLM— dejaría sin servicio a la plataforma entera por un paso del Ejecutor.
- Su activación la audita jax-platform en `kill_switch_audit` cuando la pone un
  superadmin por la API; escrita por un proceso, quedaría fuera de esa auditoría.
- El Ejecutor es un solo cliente (la cuenta `axioma`) detrás de un solo punto de paso
  (el proxy de C3): frenarlo ahí alcanza, sin tocar a nadie más.
Por eso C5 escribe `JAX_EJECUTOR_PAUSA` (junto al interruptor, en el mismo directorio
`root:fruiz 2770`) y el proxy la obedece (423, y el trozo que completa un `tool_use` no
sale). La activación queda en el propio archivo (origen, motivo, paso, momento) y en el
journal del vigía. El freno root de C4 (plan 3) tiene que matar también con ella puesta.
El interruptor global sigue frenando al Ejecutor (plan 3); esto se le SUMA.

Fail-closed, como el interruptor del frente B:
- sin la variable, vacía o relativa: `PausaSinConfigurar` (sin saber dónde está el freno
  no se ejecuta nada);
- `pausa_puesta` sólo da False ante «no existe»: permiso, ENOTDIR o E/S → PUESTA;
- `poner_pausa` publica el archivo completo de una vez (temporal + `os.link`, que falla
  si ya existe: el primer motivo no se pisa) y sincroniza el directorio.

El latido: el vigía toca `JAX_EJECUTOR_VIGIA_LATIDO` mientras vive. Un vigía muerto con
SIGKILL no llega a escribir la pausa; su latido envejece y el proxy deja de servir. Así
«vigía caído → freno» vale también cuando no hay `except` que lo atrape.

Sólo biblioteca estándar.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

VARIABLE_RUTA = "JAX_EJECUTOR_PAUSA"
VARIABLE_LATIDO = "JAX_EJECUTOR_VIGIA_LATIDO"
VARIABLE_LATIDO_MAX_S = "JAX_EJECUTOR_VIGIA_LATIDO_MAX_S"


class PausaSinConfigurar(RuntimeError):
    """La variable falta, está vacía o no es una ruta absoluta. `args[0]` es la variable."""


def _ruta(env, variable: str) -> Path:
    env = os.environ if env is None else env
    valor = env.get(variable, "").strip()
    if not valor or not Path(valor).is_absolute():
        raise PausaSinConfigurar(variable)
    return Path(valor)


def ruta_de_la_pausa(env=None) -> Path:
    return _ruta(env, VARIABLE_RUTA)


def ruta_del_latido(env=None) -> Path:
    return _ruta(env, VARIABLE_LATIDO)


def pausa_puesta(ruta) -> bool:
    try:
        os.stat(ruta)
    except FileNotFoundError:
        return False
    except OSError:  # fail-closed: sin poder mirar la pausa (permiso, ENOTDIR, E/S) se la da por PUESTA
        return True
    return True


def _sincronizar_directorio(directorio: Path) -> None:
    descriptor = os.open(directorio, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def poner_pausa(ruta: Path, datos: dict) -> bool:
    """Pone la pausa. True si la puso esta llamada; False si ya estaba (no la pisa).
    Un error de E/S se propaga: quien llama no puede creer que frenó."""
    ruta = Path(ruta)
    contenido = json.dumps({**datos, "momento": datetime.now(timezone.utc).isoformat()},
                           ensure_ascii=True, sort_keys=True)
    descriptor, temporal = tempfile.mkstemp(prefix=".pausa-", dir=ruta.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as archivo:
            archivo.write(contenido)
            archivo.flush()
            os.fsync(archivo.fileno())
        os.chmod(temporal, 0o660)
        try:
            os.link(temporal, ruta)
        except FileExistsError:
            return False
        _sincronizar_directorio(ruta.parent)
        return True
    finally:
        os.unlink(temporal)


def latir(ruta: Path) -> None:
    """Marca que el vigía vive: crea el archivo o le actualiza la fecha."""
    ruta = Path(ruta)
    with open(ruta, "a", encoding="utf-8"):
        pass
    os.utime(ruta)


def latido_fresco(ruta, max_s: float, *, ahora: float | None = None) -> bool:
    """¿El vigía latió hace menos de `max_s`? Ausente o ilegible → False (no hay vigía)."""
    try:
        mtime = os.stat(ruta).st_mtime
    except OSError:  # fail-closed: sin poder mirar el latido no hay vigía que conste; el proxy no sirve
        return False
    return (time.time() if ahora is None else ahora) - mtime < max_s
