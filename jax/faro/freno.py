"""El freno del Puerto: lee el interruptor global (`jax.core.interruptor`, FUENTE UNICA).

No reimplementa nada: `interruptor_activo` ya mira el archivo de `JAX_KILL_SWITCH_PATH` y la
ruta heredada, y ya trata cualquier error que no sea «no existe» como PUESTO. Aqui solo se
cierra el ultimo hueco: si no se sabe DONDE esta el freno (`InterruptorSinConfigurar`), el
Puerto tambien lo da por puesto. Sin saber si esta frenado, no se ejecuta (falla cerrado).
"""
from __future__ import annotations

from jax.core.interruptor import InterruptorSinConfigurar, interruptor_activo


def puesto() -> bool:
    try:
        return interruptor_activo()
    except InterruptorSinConfigurar:
        return True
