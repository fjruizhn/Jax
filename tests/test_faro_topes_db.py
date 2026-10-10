"""El Faro, paso 0.3b (P-4): el conteo ATOMICO de los topes contra una MariaDB 12.3.3 EFIMERA (nunca la de
produccion). Mismo arnes que `test_faro_bitacora_db.py` (`FARO_TEST_DB_ADMIN` o `FARO_DOCKER_CMD`; sin ninguno
las pruebas se SALTAN y el job de CI exige cero saltadas).

El defecto que se caza: leer el contador, comparar y escribir despues (dos peticiones que ven el mismo valor
y las dos pasan el tope). La defensa es un solo `UPDATE ... WHERE usado + x <= tope`; la prueba lanza N hilos
(cada uno con su bucle y su conexion) contra un tope de N-1 y exige EXACTAMENTE N-1 permitidos.
"""
from __future__ import annotations

import asyncio
import threading

import pymysql
import pytest

from jax.faro.bitacora import Bitacora
from jax.faro.bitacora_db import EmisorTabla, crear_pool, verificar_cadena
from tests.policy.catalogo_pin import catalogo_del_pin
from jax.faro.topes import AlmacenMariaDB, Topes
from tests.test_faro_bitacora_db import basedb, servidor_db  # noqa: F401 (fixtures)

N = 16
# El conteo con tope exige el recurso DEL catalogo (D-4/R-4, r6) y el catalogo
# SELLADO: el de un pin de prueba evaluado con el snapshot (r7, MAJOR-1), como
# en produccion — bytes sueltos ya no pueden fabricar uno.
CATALOGO = catalogo_del_pin()


def _usado(basedb, tenant, recurso, periodo="total") -> int | None:
    con = basedb.admin()
    try:
        with con.cursor() as cur:
            cur.execute("SELECT usado FROM faro_topes WHERE clave=%s AND periodo=%s", (f"{tenant}|{recurso}", periodo))
            fila = cur.fetchone()
            return None if fila is None else int(fila[0])
    finally:
        con.close()


def _en_hilos(basedb, cantidades, tope, tenant="t1", recurso="tokens_costo.tokens", periodo="total"):
    """Un hilo por consumo, cada uno con su bucle, su pool y su conexion, alineados en una barrera."""
    barrera = threading.Barrier(len(cantidades))
    resultados = [None] * len(cantidades)
    errores = []

    def uno(i, cantidad):
        async def correr():
            pool = await crear_pool(basedb.config_topes())
            try:
                t = Topes(AlmacenMariaDB(pool), Bitacora(emisores=[]), catalogo=CATALOGO)
                await asyncio.get_running_loop().run_in_executor(None, barrera.wait, 30)
                return await t.consumir(tenant=tenant, recurso=recurso, cantidad=cantidad, tope=tope, periodo=periodo)
            finally:
                pool.close()
                await pool.wait_closed()
        try:
            resultados[i] = asyncio.run(correr())
        except Exception as exc:  # noqa: BLE001  # fail-soft: el error del hilo se junta y la prueba lo afirma abajo
            errores.append(repr(exc))

    hilos = [threading.Thread(target=uno, args=(i, c)) for i, c in enumerate(cantidades)]
    for h in hilos:
        h.start()
    for h in hilos:
        h.join(60)
    assert not errores, errores
    return resultados


# --------------------------------------------------------------------------- #
# atomicidad                                                                  #
# --------------------------------------------------------------------------- #

def test_n_hilos_contra_un_tope_de_n_menos_uno_dejan_pasar_exactamente_n_menos_uno(basedb):
    rs = _en_hilos(basedb, [1] * N, tope=N - 1)
    assert sum(r.permitido for r in rs) == N - 1
    assert sum(not r.permitido for r in rs) == 1
    assert _usado(basedb, "t1", "tokens_costo.tokens") == N - 1                         # y la cuenta no se paso


def test_con_cantidades_distintas_lo_permitido_suma_exactamente_lo_usado_y_nunca_pasa_el_tope(basedb):
    cantidades = [1 + (i % 5) for i in range(24)]
    rs = _en_hilos(basedb, cantidades, tope=40)
    permitido = sum(c for c, r in zip(cantidades, rs) if r.permitido)
    assert permitido <= 40 and _usado(basedb, "t1", "tokens_costo.tokens") == permitido
    assert any(not r.permitido for r in rs)                                # 60 pedidos contra un tope de 40


def test_muchas_corrutinas_de_un_mismo_bucle_con_un_pool_chico_tampoco_se_pasan(basedb):
    async def caso():
        pool = await crear_pool(basedb.config_topes())             # maxsize=2: hay que hacer fila por conexion
        try:
            t = Topes(AlmacenMariaDB(pool), Bitacora(emisores=[]), catalogo=CATALOGO)
            return await asyncio.gather(*(t.consumir(tenant="t1", recurso="tokens_costo.tokens", cantidad=1, tope=29) for _ in range(40)))
        finally:
            pool.close()
            await pool.wait_closed()
    rs = asyncio.run(caso())
    assert sum(r.permitido for r in rs) == 29 and _usado(basedb, "t1", "tokens_costo.tokens") == 29


def test_sin_tope_todos_pasan_y_la_cuenta_es_exacta(basedb):
    rs = _en_hilos(basedb, [2] * N, tope=None)
    assert all(r.permitido and r.medido for r in rs)
    assert _usado(basedb, "t1", "tokens_costo.tokens") == 2 * N


def test_los_periodos_y_los_tenants_son_cuentas_independientes(basedb):
    _en_hilos(basedb, [1] * 4, tope=None, periodo="2026-10")
    _en_hilos(basedb, [1] * 3, tope=None, periodo="2026-11")
    _en_hilos(basedb, [1] * 2, tope=None, tenant="t2")
    assert (_usado(basedb, "t1", "tokens_costo.tokens", "2026-10"), _usado(basedb, "t1", "tokens_costo.tokens", "2026-11"), _usado(basedb, "t2", "tokens_costo.tokens")) == (4, 3, 2)


def test_una_cuenta_que_llega_al_tope_sigue_negando_en_una_conexion_nueva(basedb):
    _en_hilos(basedb, [1] * 3, tope=3)
    rs = _en_hilos(basedb, [1], tope=3)
    assert not rs[0].permitido and rs[0].usado == 3 and rs[0].motivo == "tope_alcanzado"


# --------------------------------------------------------------------------- #
# falla cerrado                                                               #
# --------------------------------------------------------------------------- #

def test_con_la_base_caida_el_tope_niega_y_sin_tope_deja_pasar(basedb):
    async def caso():
        pool = await crear_pool(basedb.config_topes())
        pool.close()
        await pool.wait_closed()                                     # la base "se cae": ya no hay conexiones
        t = Topes(AlmacenMariaDB(pool, plazo_s=2.0), Bitacora(emisores=[]), catalogo=CATALOGO)
        return (await t.consumir(tenant="t1", recurso="tokens_costo.tokens", cantidad=1, tope=5),
                await t.consumir(tenant="t1", recurso="tokens_costo.tokens", cantidad=1, tope=None))
    con_tope, sin_tope = asyncio.run(caso())
    assert not con_tope.permitido and con_tope.motivo == "almacen_no_disponible"
    assert sin_tope.permitido and not sin_tope.medido


def test_la_denegacion_por_tope_queda_en_la_bitacora_durable_y_la_cadena_verifica(basedb):
    async def caso():
        pool_t = await crear_pool(basedb.config_topes())
        pool_b = await crear_pool(basedb.config())
        try:
            t = Topes(AlmacenMariaDB(pool_t), Bitacora(emisores=[EmisorTabla(pool_b)]),
                      catalogo=CATALOGO)
            await t.consumir(tenant="t1", recurso="tokens_costo.tokens", cantidad=1, tope=1, run_id="run-7")
            await t.consumir(tenant="t1", recurso="tokens_costo.tokens", cantidad=1, tope=1, run_id="run-7")
        finally:
            for p in (pool_t, pool_b):
                p.close()
                await p.wait_closed()
    asyncio.run(caso())
    filas = basedb.filas()
    assert [f["evento"] for f in filas] == ["inicio_cadena", "tope_superado"]
    assert filas[1]["decision"] == "denegado" and filas[1]["run_id"] == "run-7"
    assert verificar_cadena(filas) == []


# --------------------------------------------------------------------------- #
# permisos e indice                                                           #
# --------------------------------------------------------------------------- #

def test_el_usuario_de_los_topes_solo_puede_leer_insertar_y_actualizar_su_tabla(basedb):
    con = basedb.admin()
    with con.cursor() as cur:
        cur.execute(f"SHOW GRANTS FOR `{basedb.usuario_topes}`@`%`")
        grants = " ".join(r[0] for r in cur.fetchall())
    con.close()
    assert f"`{basedb.base}`.`faro_topes`" in grants
    for permitido in ("SELECT", "INSERT", "UPDATE"):
        assert permitido in grants
    for prohibido in ("DELETE", "DROP", "ALTER", "CREATE", "ALL PRIVILEGES", "TRUNCATE"):
        assert prohibido not in grants, f"{prohibido} en {grants}"
    assert "faro_bitacora" not in grants                                    # nada sobre la bitacora


@pytest.mark.parametrize("sentencia", [
    "DELETE FROM faro_topes", "TRUNCATE TABLE faro_topes", "DROP TABLE faro_topes",
    "SELECT * FROM faro_bitacora", "INSERT INTO faro_bitacora (cadena_id, seq, momento, evento, registro, hash_previo, hash) "
    "VALUES ('a',0,0,'e','{}','h','h')",
])
def test_el_usuario_de_los_topes_no_borra_ni_toca_la_bitacora(basedb, sentencia):
    con = basedb.app_topes()
    try:
        with pytest.raises(pymysql.err.OperationalError) as exc, con.cursor() as cur:
            cur.execute(sentencia)
        assert exc.value.args[0] in (1142, 1044, 1143, 1227)
    finally:
        con.close()


def test_el_usuario_de_la_bitacora_no_toca_los_topes(basedb):
    con = basedb.app()
    try:
        for sentencia in ("SELECT * FROM faro_topes", "UPDATE faro_topes SET usado=0",
                          "INSERT INTO faro_topes (clave, periodo, usado) VALUES ('a','b',0)"):
            with pytest.raises(pymysql.err.OperationalError) as exc, con.cursor() as cur:
                cur.execute(sentencia)
            assert exc.value.args[0] in (1142, 1044, 1143, 1227), sentencia
    finally:
        con.close()


def test_el_update_del_conteo_usa_la_clave_primaria(basedb):
    _en_hilos(basedb, [1], tope=None)                                 # que haya al menos una fila
    con = basedb.admin()
    try:
        with con.cursor(pymysql.cursors.DictCursor) as cur:
            cur.execute("EXPLAIN UPDATE faro_topes SET usado = LAST_INSERT_ID(usado + 1) "
                        "WHERE clave = 't1|tokens_costo.tokens' AND periodo = 'total' AND usado + 1 <= 5")
            plan = cur.fetchall()
    finally:
        con.close()
    assert plan and plan[0]["key"] == "PRIMARY", plan


# --------------------------------------------------------------------------- #
# auditoria de 0.3bc, MINOR 6: el UPDATE que se confirma y cuya respuesta no llega #
# --------------------------------------------------------------------------- #
from jax.faro.topes import ResultadoDesconocido  # noqa: E402


class _PoolQueRompeDespues:
    """Envuelve un pool real: el UPDATE se ejecuta de verdad (se confirma) y luego la conexion 'se corta'."""
    def __init__(self, pool, romper_en):
        self._pool, self._romper_en = pool, romper_en

    def acquire(self):
        pool, romper_en = self._pool, self._romper_en

        class _Ctx:
            async def __aenter__(self_):
                self_.inner = pool.acquire()
                con = await self_.inner.__aenter__()
                return _Con(con)

            async def __aexit__(self_, *a):
                return await self_.inner.__aexit__(*a)

        class _Con:
            def __init__(self_, con):
                self_.con = con

            def cursor(self_):
                return _Cur(self_.con.cursor())

        class _Cur:
            def __init__(self_, cur):
                self_.cur = cur

            async def __aenter__(self_):
                self_.c = await self_.cur.__aenter__()
                return self_

            async def __aexit__(self_, *a):
                return await self_.cur.__aexit__(*a)

            async def execute(self_, sql, params=None):
                r = await self_.c.execute(sql, params)
                if sql.lstrip().startswith(romper_en):
                    raise ConnectionResetError("se corto despues de confirmar")
                return r

            def __getattr__(self_, nombre):
                return getattr(self_.c, nombre)
        return _Ctx()


def _con_pool(basedb, romper_en):
    async def caso(accion):
        pool = await crear_pool(basedb.config_topes())
        try:
            return await accion(_PoolQueRompeDespues(pool, romper_en), pool)
        finally:
            pool.close()
            await pool.wait_closed()
    return lambda accion: asyncio.run(caso(accion))


def test_minor6_un_update_confirmado_sin_respuesta_es_resultado_desconocido_y_se_reconcilia(basedb):
    async def accion(pool_roto, pool_real):
        t = Topes(AlmacenMariaDB(pool_roto), Bitacora(emisores=[]), catalogo=CATALOGO)
        r = await t.consumir(tenant="t1", recurso="tokens_costo.tokens", cantidad=3, tope=10)
        rec = await Topes(AlmacenMariaDB(pool_real), Bitacora(emisores=[]), catalogo=CATALOGO).reconciliar(tenant="t1", recurso="tokens_costo.tokens")
        return r, t.inciertos, rec
    r, inciertos, rec = _con_pool(basedb, "UPDATE")(accion)
    assert not r.permitido and r.motivo == "resultado_desconocido"
    assert inciertos == {("t1|tokens_costo.tokens", "total"): 3}
    assert _usado(basedb, "t1", "tokens_costo.tokens") == 3                       # la base SI lo conto: lo desconocido era real
    assert rec["usado"] == 3


def test_minor6_un_corte_antes_del_update_es_un_fallo_normal_y_no_cuenta_nada(basedb):
    async def accion(pool_roto, pool_real):
        t = Topes(AlmacenMariaDB(pool_roto), Bitacora(emisores=[]), catalogo=CATALOGO)
        return await t.consumir(tenant="t1", recurso="tokens_costo.tokens", cantidad=3, tope=10), t.inciertos
    r, inciertos = _con_pool(basedb, "INSERT")(accion)
    assert r.motivo == "almacen_no_disponible" and inciertos == {}
    assert _usado(basedb, "t1", "tokens_costo.tokens") in (None, 0)
