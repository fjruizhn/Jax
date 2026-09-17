#!/usr/bin/env python3
"""Prueba real de C3 en hall9000, re-ejecutable por un tercero.
Uso: set -a; . /etc/jax/.env; set +a; PYTHONPATH=.:las_manos python3 scripts/ejecutor_contratos/probar_c3.py
Lee /etc/jax/.env (producción) sólo para JAX_EJECUTOR_* y JAX_PROXY_CARRIL_*; no toca la DB."""
import asyncio
import os
import sys
from pathlib import Path

from jax.ejecutor.contratos import canario_c3, cuenta_axioma, formato


def main() -> int:
    fallos = asyncio.run(canario_c3.verificar_c3(
        cuenta_axioma.cuenta_desde_entorno(), registro=Path(os.environ["JAX_EJECUTOR_REGISTRO"]),
        puerto_proxy=int(os.environ["JAX_PROXY_CARRIL_PUERTO"]),
        sondas=tuple(int(p) for p in os.environ["JAX_EJECUTOR_CERCO_SONDAS"].split(","))))
    for f in fallos:
        print(formato.campos((("contrato", f.contrato), ("codigo", f.codigo)) + tuple(f.datos)))
    print(formato.campos((("c3_vivo", not fallos),)))
    return 0 if not fallos else 1


if __name__ == "__main__":
    sys.exit(main())
