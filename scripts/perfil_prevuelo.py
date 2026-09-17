#!/usr/bin/env python3
"""Perfil del pre-vuelo de Jacobs EN PROCESO (Task 15b, LAS CUATRO #2 y #4).

POR QUÉ EXISTE. La Task 15b cambió el pre-vuelo con números medidos por
scripts sueltos del scratchpad; la revisión pidió que la medición se pueda
repetir (la re-medición de la Task 15 la usa). Esto corre `jacobs.prevuelo.
prevuelo()` REAL contra `jax_memory_test` en un solo event loop, sin HTTP, y
separa CPU por pedido de espera por fase.

QUÉ ES REAL Y QUÉ NO:
  - real: MariaDB, aiomysql, el pool del store, MotorCatalog, las consultas
    del catálogo, el armado del prompt y las reglas;
  - doble: `jacobs.sonda.sondear` responde ok al instante -- NUNCA llama a un
    proveedor pago. Sin eventos de salud sembrados, las claves "se sondean"
    en cada pedido, pero sólo contra el doble;
  - sin HTTP: uvicorn/FastAPI/pydantic del cuerpo quedan afuera (se miden con
    scripts/load_test.py contra una instancia aislada).

SOLO LECTURA Y SOLO jax_memory_test. JAX_DB_NAME tiene que ser exactamente
`jax_memory_test` (si no, sale con 2 antes de importar jacobs), y cada
`cursor.execute` que no empiece con SELECT/SHOW se rechaza con RuntimeError.

MODOS (el de la conclusión importa):
  trabajadores  c tareas que piden sin pausa durante --duracion s. Carga
                sostenida de concurrencia fija.
  abierto       llegadas a --rps fijo durante --duracion s, la latencia se
                cuenta desde la llegada PROGRAMADA (incluye la cola: no hay
                omisión coordinada). Por debajo de la saturación, la latencia
                no crece con la carga ofrecida; al pasarla, crece sin límite.
  lotes         rondas cerradas de `gather` de c pedidos (el método de la
                primera medición). OJO: con CPU limitada en un solo loop, un
                lote de c pedidos que llegan JUNTOS espera en cola ~c x CPU por
                construcción -- el p95 crece con c aunque nada se degrade. Se
                deja sólo para comparar con los números viejos.
  cpu           N pedidos en serie: CPU por fase (process_time, p50/p95) y, con
                cProfile, la fracción de CPU en el catálogo de motores
                (MotorCatalog.from_db / _leer), en pydantic y en json. El armado
                del prompt corre en línea (no en un hilo) para que cProfile lo
                vea; en serie la CPU es la misma.

POR FASE (modos de carga), p50/p95 por pedido: espera de conexión del pool
(`acquire`), tiempo dentro de `execute` (suma por pedido), from_db,
leer_catalogo y armado del prompt; y la CPU agregada / pedidos.

Uso:
  set -a; source /etc/jax/.env; set +a; export JAX_DB_NAME=jax_memory_test
  PYTHONPATH=.:las_manos las_manos/.venv/bin/python scripts/perfil_prevuelo.py \\
      trabajadores -c 1,25,50 --duracion 10
  ... abierto --rps 100,400,700 --duracion 10
  ... lotes -c 1,25,50 --pedidos 600
  ... cpu --pedidos 1000

El resultado es VERDAD OPERACIONAL con fecha: se escribe en la Biblioteca.

En honor al Prof. Raúl Jacobs.
"""
from __future__ import annotations

import argparse
import asyncio
import contextvars
import json
import math
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

BASE_PERMITIDA = "jax_memory_test"
FASES = ("total", "acquire", "execute", "from_db", "leer_catalogo", "prompt")

# El plan de la Task 15: dos facetas HTTP, una de catálogo sin contrato
# completo en la semilla (thot) y una de motor (kimi), con dependencias.
PLAN_POR_DEFECTO = [
    {"step_index": 0, "facet": "hipatia", "capability": "research", "input": {"prompt": "x"}},
    {"step_index": 1, "facet": "jekyll", "capability": "analysis", "depends_on": [0], "input": {"prompt": "x"}},
    {"step_index": 2, "facet": "thot", "capability": "critique", "depends_on": [1], "input": {"prompt": "x"}},
    {"step_index": 3, "facet": "kimi", "capability": "generate", "depends_on": [0, 1, 2], "input": {"prompt": "x"}},
]


# --- Partes puras (tests/test_perfil_prevuelo.py) ----------------------------

def verificar_base(nombre: str | None) -> None:
    if nombre != BASE_PERMITIDA:
        print(f"JAX_DB_NAME={nombre!r}: este perfil sólo corre contra {BASE_PERMITIDA!r}", file=sys.stderr)
        raise SystemExit(2)


def es_lectura(sql: str) -> bool:
    palabras = sql.lstrip().split(None, 1)
    return bool(palabras) and palabras[0].upper() in {"SELECT", "SHOW"}


def percentil(valores: list[float], p: float) -> float:
    """Rango más cercano: el menor valor con al menos p·n valores <= él."""
    if not valores:
        raise ValueError("percentil de una lista vacía")
    if not 0 < p <= 1:
        raise ValueError(f"p={p} fuera de (0, 1]")
    ordenados = sorted(valores)
    return ordenados[max(0, math.ceil(p * len(ordenados)) - 1)]


def _lista_de_enteros(texto: str) -> list[int]:
    try:
        valores = [int(x) for x in texto.split(",") if x.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"se esperaba una lista de enteros separada por comas: {texto!r}") from exc
    if not valores or any(v <= 0 for v in valores):
        raise argparse.ArgumentTypeError(f"los valores tienen que ser enteros positivos: {texto!r}")
    return valores


def _positivo(texto: str) -> float:
    try:
        valor = float(texto)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"no es un número: {texto!r}") from exc
    if valor <= 0:
        raise argparse.ArgumentTypeError(f"tiene que ser mayor que cero: {texto!r}")
    return valor


def parsear_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Perfil en proceso del pre-vuelo de Jacobs (sólo lectura).")
    modos = parser.add_subparsers(dest="modo", required=True)
    t = modos.add_parser("trabajadores", help="c tareas pidiendo sin pausa")
    t.add_argument("-c", "--concurrencias", type=_lista_de_enteros, default=[1, 25, 50])
    t.add_argument("--duracion", type=_positivo, default=10.0)
    a = modos.add_parser("abierto", help="llegadas a ritmo fijo")
    a.add_argument("--rps", type=_lista_de_enteros, required=True)
    a.add_argument("--duracion", type=_positivo, default=10.0)
    lo = modos.add_parser("lotes", help="rondas cerradas de gather (infla p95 con c)")
    lo.add_argument("-c", "--concurrencias", type=_lista_de_enteros, default=[1, 25, 50])
    lo.add_argument("--pedidos", type=int, default=600)
    c = modos.add_parser("cpu", help="desglose de CPU en serie")
    c.add_argument("--pedidos", type=int, default=1000)
    for sub in (t, a, lo, c):
        sub.add_argument("--plan", type=Path, default=None, help="JSON con la lista de pasos")
        sub.add_argument("--calentamiento", type=int, default=20)
    args = parser.parse_args(argv)
    for nombre in ("pedidos",):
        if getattr(args, nombre, 1) <= 0:
            parser.error(f"--{nombre} tiene que ser mayor que cero")
    return args


def categorias_de_cpu(filas: list[tuple[str, str, float, float]], total: float) -> dict[str, float]:
    """filas: (archivo, función, tottime, cumtime) de pstats. Devuelve
    fracciones del total: catálogo de motores (cumtime de
    MotorCatalog.from_db; incluye decodificar sus filas), pydantic y json
    (tottime de todo lo que vive en esos paquetes)."""
    if total <= 0:
        raise ValueError("total de CPU no positivo")
    catalogo = sum(cum for arch, fn, _, cum in filas
                   if arch.endswith(os.path.join("motor_registry", "catalog.py")) and fn == "from_db")
    pydantic = sum(tot for arch, fn, tot, _ in filas if "pydantic" in arch or "pydantic" in fn)
    json_ = sum(tot for arch, fn, tot, _ in filas
                if f"{os.sep}json{os.sep}" in arch or "_json" in fn)
    return {"catalogo_motores": catalogo / total, "pydantic": pydantic / total, "json": json_ / total}


def resumen(latencias: list[dict], segundos: float, cpu: float) -> dict:
    n = len(latencias)
    salida = {"pedidos": n, "rps": n / segundos, "cpu_ms_por_pedido": cpu / n * 1e3}
    for fase in FASES:
        xs = [d.get(fase, 0.0) * 1e3 for d in latencias]
        salida[fase] = {"p50": percentil(xs, 0.5), "p95": percentil(xs, 0.95)}
    return salida


# --- Parte con base -----------------------------------------------------------

_MEDIDAS: contextvars.ContextVar[dict] = contextvars.ContextVar("medidas")
# Reloj de las fases: perf_counter (espera incluida) en los modos de carga;
# process_time (sólo CPU) en el modo cpu, que corre en serie y en un hilo.
_RELOJ = time.perf_counter


def _sumar(fase: str, dt: float) -> None:
    d = _MEDIDAS.get(None)
    if d is not None:
        d[fase] = d.get(fase, 0.0) + dt


def _instrumentar(inline_prompt: bool):
    import aiomysql.cursors

    global _RELOJ
    if inline_prompt:
        _RELOJ = time.process_time

    from jacobs import prevuelo as pv
    from jacobs import sonda, store
    from motor_registry.catalog import MotorCatalog

    async def sonda_doble(clave, d, **kw):
        return sonda.ResultadoSonda(True, None)
    sonda.sondear = sonda_doble

    execute_real = aiomysql.cursors.Cursor.execute

    async def execute(self, query, args=None):
        if not es_lectura(query):
            raise RuntimeError(f"perfil_prevuelo es de sólo lectura; se rechazó: {query[:60]!r}")
        t = _RELOJ()
        try:
            return await execute_real(self, query, args)
        finally:
            _sumar("execute", _RELOJ() - t)
    aiomysql.cursors.Cursor.execute = execute

    conexion_real = store.conexion_del_pool

    @asynccontextmanager
    async def conexion_del_pool():
        t = _RELOJ()
        async with conexion_real() as conn:
            _sumar("acquire", _RELOJ() - t)
            yield conn
    store.conexion_del_pool = conexion_del_pool

    from_db_real = MotorCatalog.from_db.__func__

    async def from_db(cls, conexion=None):
        t = _RELOJ()
        try:
            return await from_db_real(cls, conexion)
        finally:
            _sumar("from_db", _RELOJ() - t)
    MotorCatalog.from_db = classmethod(from_db)

    leer_real = pv.prevuelo_catalogo.leer_catalogo

    async def leer_catalogo(**kw):
        t = _RELOJ()
        try:
            return await leer_real(**kw)
        finally:
            _sumar("leer_catalogo", _RELOJ() - t)
    pv.prevuelo_catalogo.leer_catalogo = leer_catalogo

    chars_real = pv._chars_de_entrada

    def chars(*a):
        t = _RELOJ()
        try:
            return chars_real(*a)
        finally:
            _sumar("prompt", _RELOJ() - t)
    pv._chars_de_entrada = chars

    if inline_prompt:
        async def en_linea(fn, *a, **kw):
            return fn(*a, **kw)
        class _AsyncioEnLinea:
            """asyncio de prevuelo.py con to_thread en línea (sólo modo cpu)."""
            to_thread = staticmethod(en_linea)

            def __getattr__(self, nombre):
                return getattr(asyncio, nombre)
        pv.asyncio = _AsyncioEnLinea()
    return pv, store


def _plan(ruta: Path | None):
    from jacobs.models import Step
    crudo = json.loads(ruta.read_text(encoding="utf-8")) if ruta else PLAN_POR_DEFECTO
    return [Step(pipeline_id="perfil", **p) for p in crudo]


async def _pedido(pv, pasos, latencias, llegada: float | None = None) -> None:
    """`total` en el reloj de las fases (pared en carga, CPU en el modo cpu).
    En el modo abierto se cuenta desde la llegada programada."""
    d: dict = {}
    _MEDIDAS.set(d)
    inicio = llegada if llegada is not None else _RELOJ()
    await pv.prevuelo(pasos, {"objective": "perfil"})
    d["total"] = _RELOJ() - inicio
    latencias.append(d)


async def _conexiones_del_servidor(store) -> int:
    conn = await store.conexion_dedicada()
    try:
        async with conn.cursor() as cur:
            await cur.execute("SHOW GLOBAL STATUS LIKE 'Connections'")
            (_, valor), = await cur.fetchall()
        return int(valor)
    finally:
        conn.close()


async def _correr_carga(args, pv, store, pasos) -> list[dict]:
    resultados = []
    etiquetas = args.rps if args.modo == "abierto" else args.concurrencias
    for valor in etiquetas:
        latencias: list[dict] = []
        antes = await _conexiones_del_servidor(store)
        cpu0, t0 = time.process_time(), time.perf_counter()
        if args.modo == "trabajadores":
            fin = t0 + args.duracion

            async def trabajador():
                while time.perf_counter() < fin:
                    await _pedido(pv, pasos, latencias)
            await asyncio.gather(*[trabajador() for _ in range(valor)])
        elif args.modo == "abierto":
            tareas = []
            intervalo = 1.0 / valor
            for k in range(int(args.duracion * valor)):
                llegada = t0 + k * intervalo
                espera = llegada - time.perf_counter()
                if espera > 0:
                    await asyncio.sleep(espera)
                tareas.append(asyncio.ensure_future(_pedido(pv, pasos, latencias, llegada)))
            await asyncio.gather(*tareas)
        else:
            for _ in range(max(1, args.pedidos // valor)):
                await asyncio.gather(*[_pedido(pv, pasos, latencias) for _ in range(valor)])
        segundos, cpu = time.perf_counter() - t0, time.process_time() - cpu0
        despues = await _conexiones_del_servidor(store)
        r = resumen(latencias, segundos, cpu)
        r[args.modo if args.modo != "abierto" else "rps_ofrecido"] = valor
        r["modo"] = args.modo
        # -1: la conexión propia de _conexiones_del_servidor de "despues".
        r["conexiones_nuevas_por_pedido"] = (despues - antes - 1) / len(latencias)
        resultados.append(r)
        print(json.dumps(r, ensure_ascii=False))
    return resultados


async def _correr_cpu(args, pv, store, pasos) -> dict:
    import cProfile
    import pstats

    latencias: list[dict] = []
    cpu0 = time.process_time()
    for _ in range(args.pedidos):
        await _pedido(pv, pasos, latencias)
    cpu_por_pedido = (time.process_time() - cpu0) / args.pedidos

    # Reloj de CPU: con el de pared, el tiempo del epoll esperando a MariaDB
    # entraría al total y achicaría todas las fracciones.
    perfil = cProfile.Profile(time.process_time)
    perfil.enable()
    for _ in range(args.pedidos):
        await _pedido(pv, pasos, [])
    perfil.disable()
    stats = pstats.Stats(perfil)
    filas = [(arch, fn, tot, cum) for (arch, _, fn), (_, _, tot, cum, _) in stats.stats.items()]
    fracciones = categorias_de_cpu(filas, stats.total_tt)
    r = {"modo": "cpu", "pedidos": args.pedidos, "cpu_ms_por_pedido": cpu_por_pedido * 1e3,
         "fraccion_de_cpu_perfilada": fracciones}
    for fase in FASES:
        xs = [d.get(fase, 0.0) * 1e3 for d in latencias]
        r[fase] = {"p50": percentil(xs, 0.5), "p95": percentil(xs, 0.95)}
    print(json.dumps(r, ensure_ascii=False))
    return r


async def _main_async(args) -> None:
    pv, store = _instrumentar(inline_prompt=args.modo == "cpu")
    pasos = _plan(args.plan)
    try:
        for _ in range(args.calentamiento):
            await _pedido(pv, pasos, [])
        if args.modo == "cpu":
            await _correr_cpu(args, pv, store, pasos)
        else:
            await _correr_carga(args, pv, store, pasos)
    finally:
        await store.cerrar_pool()


def main(argv: list[str] | None = None) -> int:
    args = parsear_args(argv)
    verificar_base(os.environ.get("JAX_DB_NAME"))
    raiz = Path(__file__).resolve().parents[1]
    for ruta in (str(raiz), str(raiz / "las_manos")):
        if ruta not in sys.path:
            sys.path.insert(0, ruta)
    asyncio.run(_main_async(args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
