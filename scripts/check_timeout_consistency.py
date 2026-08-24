#!/usr/bin/env python3
"""
Chequeo de consistencia entre las DOS fuentes que gobiernan el timeout de
ejecución de un step de Jacobs (Régimen A del inventario de la ronda 5,
CONTEXT.md): el default por-capability en código
(jacobs/plan.py::_CAPABILITY_TIMEOUT_SECONDS/_DEFAULT_TIMEOUT_SECONDS,
la fuente REAL que corre hoy) contra el techo de admisión en DB
(capability.max_execution_minutes, consultado por MotorPolicy.check()
como techo de lo que un caller puede pedir).

Por qué SOLO estas dos, y no los otros 3 regímenes del inventario: código y
DB gobiernan la MISMA pregunta ("cuánto puede tardar un step de esta
capability") por dos caminos distintos que deberían decir lo mismo -- si
divergen, es exactamente el riesgo de confianza documentado en la ronda 5:
alguien edita max_execution_minutes en un panel admin esperando que cambie
el comportamiento real, y no pasa nada, porque el código sigue usando su
propio dict. Los otros regímenes NO son divergencia: OLLAMA_TIMEOUT/
ADA_TIMEOUT (régimen B) gobiernan la LLAMADA que genera el plan, no la
ejecución de un step ya planeado -- comparalos sería comparar dos preguntas
distintas. config.toml (régimen C) es el REPL directo, sin relación con
capability. Los timeouts de cliente HTTP de jax-platform (régimen D) y los
umbrales del reaper (régimen E) tampoco gobiernan "cuánto puede tardar un
step" -- gobiernan cuánto espera un cliente, o cuándo el reaper decide que
algo se estancó. Comparar Régimen A contra sí mismo (código vs DB) es la
única comparación con la MISMA semántica en ambos lados.

Divergencia problemática: cualquier capability donde el valor efectivo de
_CAPABILITY_TIMEOUT_SECONDS/_DEFAULT_TIMEOUT_SECONDS (segundos) no
coincide con capability.max_execution_minutes*60 (DB). No hay "diferencia
legítima" posible acá -- a diferencia de los otros regímenes, ambos dicen
gobernar la misma capability con el mismo significado (ver ronda 4/5:
ambos fueron realineados a los mismos valores a propósito). Cualquier
divergencia es drift real, no diseño.

Dónde corre: NO en CI de GitHub Actions -- este chequeo necesita la DB
real (jax_memory), y los runners de GitHub no tienen red hacia
127.0.0.1:3308 de hall9000 (mismo motivo que el CI de P10 solo escanea
código fuente, nunca DB). Corre en dos lugares:
  1. Standalone (este script) -- para sesiones de pago de deuda o chequeo
     manual, mismo patrón que find_unread_columns.py.
  2. Al arranque de jax-las-manos.service (las_manos/server.py::_jacobs_init)
     -- WARNING en el log si hay divergencia, no bloquea el arranque. Corre
     en el ambiente real, con DB real, en cada restart -- que es exactamente
     cuando alguien ya está mirando los logs para confirmar que el servicio
     levantó sano (T0 de rondas anteriores). Reemplaza "esperar a que una
     ronda de deuda futura lo note a mano" (la alternativa que se descartó
     en la ronda 5) por una señal automática y real.

Qué hace al encontrar divergencia: SOLO REPORTA (exit 1 en modo standalone
para que un script/CI que sí tuviera DB pueda fallar sobre esto; WARNING
logueado, no excepción, al arranque del servicio). No falla el arranque
del servicio a propósito -- # fail-soft: la divergencia es un problema de
CONFIANZA/mantenimiento (alguien puede editar la DB sin efecto), no de
CORRECCIÓN funcional -- el código YA es la fuente real que gobierna la
ejecución (ronda 5), así que ninguna divergencia deja un job sin timeout
o con un timeout incorrecto. Bloquear el arranque de producción por un
desacuerdo de metadata sería el mismo error de proporción que P10 ya
señaló en otros lados: castigar con dureza máxima algo que no es una
falla de ejecución.

Uso:
  set -a; source /etc/jax/.env; set +a
  /home/fruiz/jax/las_manos/.venv/bin/python scripts/check_timeout_consistency.py

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "las_manos"))  # credential_resolver, model_catalog
sys.path.insert(0, str(_REPO_ROOT))  # jacobs.*

import aiomysql  # noqa: E402

from jacobs.plan import _CAPABILITY_TIMEOUT_SECONDS, _DEFAULT_TIMEOUT_SECONDS  # noqa: E402


async def _fetch_db_minutes() -> dict[str, int]:
    host = os.environ.get("JAX_DB_HOST")
    port = os.environ.get("JAX_DB_PORT")
    if not host or not port:
        raise RuntimeError(
            "JAX_DB_HOST/JAX_DB_PORT no están seteados -- sin default "
            "silencioso a localhost:3306 (esa instancia está muerta, ver "
            "memoria jax-dual-mariadb-instances). Sourceá /etc/jax/.env o "
            "exportalos a mano antes de conectar."
        )
    conn = await aiomysql.connect(
        host=host,
        port=int(port),
        user=os.getenv("JAX_DB_USER", ""),
        password=os.getenv("JAX_DB_PASSWORD", ""),
        db=os.getenv("JAX_DB_NAME", "jax_memory"),
        charset="utf8mb4",
        autocommit=True,
    )
    try:
        async with conn.cursor() as cur:
            await cur.execute("SELECT `key`, max_execution_minutes FROM capability")
            return {row[0]: row[1] for row in await cur.fetchall()}
    finally:
        conn.close()


async def find_timeout_divergences() -> list[str]:
    """Devuelve una lista de líneas "capability: código=Xs DB=Ys" por cada
    capability donde código y DB no coinciden. [] si todo coincide."""
    db_minutes = await _fetch_db_minutes()
    divergences = []
    for key, db_min in sorted(db_minutes.items()):
        code_seconds = _CAPABILITY_TIMEOUT_SECONDS.get(key, _DEFAULT_TIMEOUT_SECONDS)
        db_seconds = db_min * 60
        if code_seconds != db_seconds:
            divergences.append(
                f"{key}: código={code_seconds}s (_CAPABILITY_TIMEOUT_SECONDS) "
                f"DB={db_seconds}s ({db_min}min, capability.max_execution_minutes)"
            )
    return divergences


async def main_async() -> int:
    divergences = await find_timeout_divergences()
    if divergences:
        print(f"DIVERGENCIA — {len(divergences)} capability(ies) donde código y DB no coinciden:")
        for d in divergences:
            print(f"  {d}")
        return 1
    print("OK — código (_CAPABILITY_TIMEOUT_SECONDS) y DB (capability.max_execution_minutes) coinciden en todas las capabilities.")
    return 0


def main() -> int:
    import asyncio
    return asyncio.run(main_async())


if __name__ == "__main__":
    sys.exit(main())
