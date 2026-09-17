#!/usr/bin/env python3
# scripts/ejecutor_contratos/probar_c1.py
"""Prueba real de C1 en hall9000, para que un tercero la vuelva a correr.

Corre verificar_c1 contra la cuenta real, el gancho instalado y el arnés real en la
jaula. Imprime cada Fallo en formato neutro y sale 0 sólo si no hay ninguno.
Uso:  set -a; . /etc/jax/.env; set +a
      PYTHONPATH=.:las_manos python3 scripts/ejecutor_contratos/probar_c1.py
Lee /etc/jax/.env (producción) sólo para JAX_EJECUTOR_*: no toca la DB.
"""
import asyncio
import os
import sys

from jax.ejecutor.contratos import canario_c1, cuenta_axioma, formato


def main() -> int:
    c = cuenta_axioma.cuenta_desde_entorno()
    fallos = asyncio.run(canario_c1.verificar_c1(c, puerto_canario=int(os.environ["JAX_EJECUTOR_CANARIO_PUERTO"])))
    for f in fallos:
        print(formato.campos((("contrato", f.contrato), ("codigo", f.codigo)) + tuple(f.datos)))
    print(formato.campos((("c1_vivo", not fallos),)))
    return 0 if not fallos else 1


if __name__ == "__main__":
    sys.exit(main())
