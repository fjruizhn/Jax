#!/usr/bin/env python3
# scripts/ejecutor_contratos/probar_arranque.py
"""Arranque real del Ejecutor en hall9000 SIN lanzar ninguna misión ni abrir el proxy: corre
exigir_contratos y dice si arrancaría, con cada fallo y el tiempo.

Sin máquinas: el arranque del Ejecutor (seis contratos + proveedores distintos). Con máquinas:
el arranque de una misión sobre ellas (además, la compuerta de datos de clientes).

Uso: set -a; . <(sudo -n cat /etc/jax/.env); set +a
     PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:las_manos .venv/bin/python scripts/ejecutor_contratos/probar_arranque.py \
       [--instrucciones-auditor ARCHIVO] [hall9000 otra …]
`--instrucciones-auditor` existe sólo para VER FALLAR a C5 sin escribir en la DB: cambia, en este
proceso, el archivo de instrucciones del auditor. El arranque de producción (vigia_servicio) no la tiene.
Exporta la política (C1) a JAX_EJECUTOR_POLITICA como lo hace cada arranque; no escribe en la DB.
Salida: `clave=valor`; sale 0 sólo con `arrancaria=true`.
"""
import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

from jax.ejecutor.contratos import arranque, auditor_cliente, formato


async def principal(args) -> int:
    ctx = arranque.contexto_desde_entorno(os.environ, frozenset(args.hosts) if args.hosts else None)
    if args.instrucciones_auditor:
        auditor_cliente._INSTRUCCIONES = Path(args.instrucciones_auditor)
    inicio = time.monotonic()
    try:
        await arranque.exigir_contratos(ctx)
    except arranque.ContratosNoVerificados as exc:
        for f in exc.fallos:
            print(formato.campos((("contrato", f.contrato), ("codigo", f.codigo)) + tuple(f.datos)))
        print(formato.campos((("arrancaria", False), ("fallos", len(exc.fallos)),
                              ("segundos", str(round(time.monotonic() - inicio, 1))))))
        return 1
    print(formato.campos((("arrancaria", True), ("segundos", str(round(time.monotonic() - inicio, 1))))))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--instrucciones-auditor")
    ap.add_argument("hosts", nargs="*")
    sys.exit(asyncio.run(principal(ap.parse_args())))
