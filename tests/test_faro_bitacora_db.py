"""El Faro (auditoria MAJOR-5): la bitacora durable. Tabla encadenada con hash previo, un usuario de base que
SOLO puede insertar, y falla cerrado si el emisor falla. MariaDB EFIMERA (nunca la de produccion):

- `FARO_TEST_DB_ADMIN="host:puerto:usuario:clave"` -> un servidor ya levantado (el service de CI);
- si no, un contenedor con `FARO_DOCKER_CMD` (p. ej. "sudo -n docker"), en 127.0.0.1 y puerto aleatorio,
  destruido al terminar (mismo esquema que axioma-sar-operador).
Cada prueba usa su propia base `faro_t_<uuid>` y su propio usuario, que se borran al terminar.
Sin ninguna de las dos configuraciones las pruebas se SALTAN (el job de CI las configura y exige cero saltadas).
"""
from __future__ import annotations

import asyncio
import json
import os
import secrets
import shlex
import subprocess
import time
import uuid
from pathlib import Path

import aiomysql
import pymysql
import pytest
from mcp.shared.exceptions import MCPError

from jax.faro import paquete
from jax.faro.bitacora import Bitacora
from jax.faro.bitacora_db import ConfigBitacoraDB, EmisorTabla, GENESIS, crear_pool, verificar_cadena
from jax.faro.config import ConfigFaro, ConfigFaroInvalida, ConfigPuerto
from jax.faro.migraciones import aplicar
from jax.faro.paquete import cargar_paquete
from jax.faro.transporte import ServidorPuerto
from tests._faro_utils import _git, cliente_por_rele, ejecucion, repo_de_juguete, servidor

MIGRACIONES = Path(__file__).resolve().parents[1] / "ops" / "faro" / "migrations"
IMAGEN = os.environ.get("FARO_TEST_MARIADB_IMAGE", "mariadb:12.3.3")


def _esperar(host, puerto, usuario, clave, intentos=90):
    ultimo = None
    for _ in range(intentos):
        try:
            pymysql.connect(host=host, port=puerto, user=usuario, password=clave).close()
            return
        except Exception as exc:  # fail-soft: la base todavia arranca; se reintenta y, agotados los intentos, se levanta el ultimo error
            ultimo = exc
            time.sleep(1)
    raise RuntimeError(f"MariaDB de prueba no respondio: {ultimo}")


@pytest.fixture(scope="session")
def servidor_db():
    externo = os.environ.get("FARO_TEST_DB_ADMIN")
    if externo:
        host, puerto, usuario, clave = externo.split(":", 3)
        _esperar(host, int(puerto), usuario, clave)
        yield {"host": host, "puerto": int(puerto), "usuario": usuario, "clave": clave}
        return
    if not os.environ.get("FARO_DOCKER_CMD"):
        pytest.skip("sin FARO_TEST_DB_ADMIN ni FARO_DOCKER_CMD: no hay MariaDB efimera para estas pruebas")
    docker = shlex.split(os.environ["FARO_DOCKER_CMD"])
    clave = secrets.token_hex(12)
    nombre = f"faro-test-{uuid.uuid4().hex[:8]}"
    subprocess.run([*docker, "run", "-d", "--rm", "--name", nombre, "-p", "127.0.0.1::3306",
                    "-e", f"MARIADB_ROOT_PASSWORD={clave}", IMAGEN], capture_output=True, text=True, check=True)
    try:
        puerto = subprocess.run([*docker, "port", nombre, "3306/tcp"], capture_output=True, text=True,
                                check=True).stdout.strip().splitlines()[0].rsplit(":", 1)[1]
        _esperar("127.0.0.1", int(puerto), "root", clave)
        yield {"host": "127.0.0.1", "puerto": int(puerto), "usuario": "root", "clave": clave}
    finally:
        subprocess.run([*docker, "rm", "-f", nombre], capture_output=True)


class BaseDePrueba:
    def __init__(self, srv, base, usuario, clave_usuario):
        self.srv, self.base, self.usuario, self.clave_usuario = srv, base, usuario, clave_usuario

    def admin(self):
        return pymysql.connect(host=self.srv["host"], port=self.srv["puerto"], user=self.srv["usuario"],
                               password=self.srv["clave"], database=self.base, autocommit=True)

    def app(self):
        return pymysql.connect(host=self.srv["host"], port=self.srv["puerto"], user=self.usuario,
                               password=self.clave_usuario, database=self.base, autocommit=True)

    def config(self) -> ConfigBitacoraDB:
        return ConfigBitacoraDB(host=self.srv["host"], port=self.srv["puerto"], usuario=self.usuario,
                                clave=self.clave_usuario, base=self.base)

    def filas(self) -> list[dict]:
        con = self.admin()
        try:
            with con.cursor(pymysql.cursors.DictCursor) as cur:
                cur.execute("SELECT * FROM faro_bitacora ORDER BY id")
                return list(cur.fetchall())
        finally:
            con.close()


@pytest.fixture
def basedb(servidor_db):
    base = f"faro_t_{uuid.uuid4().hex[:12]}"
    usuario = f"u_{uuid.uuid4().hex[:10]}"
    clave = secrets.token_hex(8)
    adm = pymysql.connect(host=servidor_db["host"], port=servidor_db["puerto"], user=servidor_db["usuario"],
                          password=servidor_db["clave"], autocommit=True)
    try:
        with adm.cursor() as cur:
            cur.execute(f"CREATE DATABASE `{base}`")
            cur.execute(f"CREATE USER `{usuario}`@`%%` IDENTIFIED BY %s", (clave,))

        async def migrar():
            con = await aiomysql.connect(host=servidor_db["host"], port=servidor_db["puerto"], user=servidor_db["usuario"],
                                         password=servidor_db["clave"], connect_timeout=10)
            try:
                return await aplicar(con, MIGRACIONES, {"base": base, "usuario": usuario, "host_usuario": "%"})
            finally:
                con.close()
        asyncio.run(migrar())
        yield BaseDePrueba(servidor_db, base, usuario, clave)
    finally:
        with adm.cursor() as cur:
            cur.execute(f"DROP DATABASE IF EXISTS `{base}`")
            cur.execute(f"DROP USER IF EXISTS `{usuario}`@`%`")
        adm.close()


def _correr(basedb, coro_fn):
    async def todo():
        pool = await crear_pool(basedb.config())
        try:
            return await coro_fn(pool)
        finally:
            pool.close()
            await pool.wait_closed()
    return asyncio.run(todo())


def _reg(i, **kw):
    return {"evento": "llamada", "momento": 1000.0 + i, "run_id": "run-1", "id_correlacion": "corr-1",
            "decision": "permitido", "metodo": "tools/call", "objetivo": f"herramienta-{i}", **kw}


# --------------------------------------------------------------------------- #
# el usuario de la aplicacion solo puede insertar                             #
# --------------------------------------------------------------------------- #

def test_las_migraciones_crean_la_tabla_y_dan_solo_insert(basedb):
    con = basedb.admin()
    with con.cursor() as cur:
        cur.execute(f"SHOW GRANTS FOR `{basedb.usuario}`@`%`")
        grants = " ".join(r[0] for r in cur.fetchall())
    con.close()
    assert "INSERT" in grants and f"`{basedb.base}`.`faro_bitacora`" in grants
    for prohibido in ("SELECT", "UPDATE", "DELETE", "DROP", "ALL PRIVILEGES", "ALTER", "CREATE"):
        assert prohibido not in grants.replace("GRANT USAGE", ""), f"{prohibido} en {grants}"


@pytest.mark.parametrize("sentencia", [
    "SELECT * FROM faro_bitacora",
    "UPDATE faro_bitacora SET decision='denegado'",
    "DELETE FROM faro_bitacora",
    "TRUNCATE TABLE faro_bitacora",
    "DROP TABLE faro_bitacora",
    "ALTER TABLE faro_bitacora ADD COLUMN x INT",
    "REPLACE INTO faro_bitacora (cadena_id, seq, momento, evento, registro, hash_previo, hash) VALUES ('a',0,0,'e','{}','h','h')",
])
def test_el_usuario_de_la_aplicacion_no_puede_leer_editar_ni_borrar(basedb, sentencia):
    con = basedb.app()
    try:
        with pytest.raises(pymysql.err.OperationalError) as exc, con.cursor() as cur:
            cur.execute(sentencia)
        assert exc.value.args[0] in (1142, 1044, 1143, 1227)   # denegado por privilegios
    finally:
        con.close()


def test_el_usuario_de_la_aplicacion_si_puede_insertar(basedb):
    con = basedb.app()
    with con.cursor() as cur:
        cur.execute("INSERT INTO faro_bitacora (cadena_id, seq, momento, evento, registro, hash_previo, hash) "
                    "VALUES ('c', 0, 1.5, 'x', '{}', %s, %s)", (GENESIS, "h" * 64))
    con.close()
    assert len(basedb.filas()) == 1


# --------------------------------------------------------------------------- #
# la cadena                                                                   #
# --------------------------------------------------------------------------- #

def test_el_emisor_escribe_una_cadena_que_verifica(basedb):
    async def caso(pool):
        emisor = EmisorTabla(pool)
        for i in range(5):
            await emisor(_reg(i))
    _correr(basedb, caso)
    filas = basedb.filas()
    assert [f["evento"] for f in filas] == ["inicio_cadena"] + ["llamada"] * 5
    assert [f["seq"] for f in filas] == list(range(6))
    assert filas[0]["hash_previo"] == GENESIS and all(filas[i]["hash_previo"] == filas[i - 1]["hash"] for i in range(1, 6))
    assert verificar_cadena(filas) == []
    assert json.loads(filas[3]["registro"])["objetivo"] == "herramienta-2"


def test_una_fila_modificada_rompe_la_verificacion(basedb):
    _correr(basedb, lambda pool: _emitir(pool, 4))
    con = basedb.admin()
    with con.cursor() as cur:
        cur.execute("UPDATE faro_bitacora SET registro = REPLACE(registro, 'permitido', 'denegado') WHERE seq = 2")
    con.close()
    problemas = verificar_cadena(basedb.filas())
    assert any(p.codigo == "hash_no_coincide" and p.seq == 2 for p in problemas)


def test_una_fila_borrada_en_el_medio_se_detecta_como_hueco(basedb):
    _correr(basedb, lambda pool: _emitir(pool, 5))
    con = basedb.admin()
    with con.cursor() as cur:
        cur.execute("DELETE FROM faro_bitacora WHERE seq = 3")
    con.close()
    assert any(p.codigo == "hueco" for p in verificar_cadena(basedb.filas()))


def test_una_fila_borrada_al_principio_se_detecta(basedb):
    _correr(basedb, lambda pool: _emitir(pool, 3))
    con = basedb.admin()
    with con.cursor() as cur:
        cur.execute("DELETE FROM faro_bitacora WHERE seq = 0")
    con.close()
    assert any(p.codigo in ("hueco", "cadena_sin_inicio") for p in verificar_cadena(basedb.filas()))


def test_reordenar_filas_se_detecta(basedb):
    _correr(basedb, lambda pool: _emitir(pool, 4))
    filas = basedb.filas()
    filas[2], filas[3] = filas[3], filas[2]
    # `verificar_cadena` ordena por seq, asi que el reordenamiento tiene que mostrarse en los seq/hash
    filas[2]["seq"], filas[3]["seq"] = filas[3]["seq"], filas[2]["seq"]
    assert verificar_cadena(filas) != []


async def _emitir(pool, n):
    emisor = EmisorTabla(pool)
    for i in range(n):
        await emisor(_reg(i))


def test_cada_arranque_abre_su_propia_cadena_y_las_dos_verifican(basedb):
    _correr(basedb, lambda pool: _emitir(pool, 3))
    _correr(basedb, lambda pool: _emitir(pool, 2))
    filas = basedb.filas()
    assert len({f["cadena_id"] for f in filas}) == 2
    assert verificar_cadena(filas) == []
    assert [f["evento"] for f in filas].count("inicio_cadena") == 2


def test_los_registros_concurrentes_quedan_en_una_cadena_valida(basedb):
    async def caso(pool):
        emisor = EmisorTabla(pool)
        await asyncio.gather(*(emisor(_reg(i)) for i in range(60)))
    _correr(basedb, caso)
    filas = basedb.filas()
    assert len(filas) == 61 and verificar_cadena(filas) == []
    assert sorted(json.loads(f["registro"])["objetivo"] for f in filas[1:]) == sorted(f"herramienta-{i}" for i in range(60))


def test_un_fallo_de_insercion_levanta_error_y_la_cadena_siguiente_sigue_valida(basedb):
    async def caso(pool):
        emisor = EmisorTabla(pool)
        await emisor(_reg(0))
        adm = basedb.admin()
        with adm.cursor() as cur:
            cur.execute(f"REVOKE INSERT ON `{basedb.base}`.faro_bitacora FROM `{basedb.usuario}`@`%`")
        with pytest.raises(Exception):
            await emisor(_reg(1))                      # fallo cerrado: la excepcion sube
        with adm.cursor() as cur:
            cur.execute(f"GRANT INSERT ON `{basedb.base}`.faro_bitacora TO `{basedb.usuario}`@`%`")
        adm.close()
        await emisor(_reg(2))                          # se recupera abriendo una cadena nueva (la insercion fallida es ambigua)
    _correr(basedb, caso)
    filas = basedb.filas()
    assert verificar_cadena(filas) == []
    assert len({f["cadena_id"] for f in filas}) == 2


def test_la_configuracion_de_la_base_falla_cerrado(tmp_path):
    ok = {"JAX_FARO_BITACORA_DB_HOST": "127.0.0.1", "JAX_FARO_BITACORA_DB_PORT": "3306", "JAX_FARO_BITACORA_DB_USER": "u",
          "JAX_FARO_BITACORA_DB_PASSWORD": "p", "JAX_FARO_BITACORA_DB_NAME": "b"}
    assert ConfigBitacoraDB.desde_entorno(ok).base == "b"
    for falta in ok:
        with pytest.raises(ConfigFaroInvalida, match=falta):
            ConfigBitacoraDB.desde_entorno({k: v for k, v in ok.items() if k != falta})
    with pytest.raises(ConfigFaroInvalida):
        ConfigBitacoraDB.desde_entorno({**ok, "JAX_FARO_BITACORA_DB_NAME": "b; DROP DATABASE x"})
    assert "p" not in repr(ConfigBitacoraDB.desde_entorno(ok)).replace("port", "").replace("password", "")  # la clave no se imprime


# --------------------------------------------------------------------------- #
# integrada con el Puerto: con la tabla caida, nada se entrega                 #
# --------------------------------------------------------------------------- #

def test_con_la_tabla_de_la_bitacora_caida_el_puerto_no_entrega_resultados(basedb, tmp_path):
    repo = repo_de_juguete(tmp_path)
    cfg = ConfigFaro(repo=repo, sha=_git(repo, "rev-parse", "HEAD"), destino=tmp_path / "eco", uid_duenio=os.getuid())
    paquete.construir_paquete(cfg)
    d = tmp_path / "run"
    d.mkdir(mode=0o700)

    async def caso(pool):
        bit = Bitacora(emisores=[EmisorTabla(pool)])
        async with servidor(ConfigPuerto(socket_dir=d), ejecucion(), cargar_paquete(cfg), bit) as srv:
            async with cliente_por_rele(srv) as c:
                ok = await c.call_tool("skills.leer", {"nombre": "alfa"})
                assert not ok.is_error                           # con la tabla, funciona y queda registrado
                adm = basedb.admin()
                with adm.cursor() as cur:
                    cur.execute(f"REVOKE INSERT ON `{basedb.base}`.faro_bitacora FROM `{basedb.usuario}`@`%`")
                adm.close()
                with pytest.raises(MCPError) as exc:
                    await c.call_tool("skills.leer", {"nombre": "alfa"})
                assert "BITACORA" in exc.value.message
    _correr(basedb, caso)
    filas = basedb.filas()
    assert verificar_cadena(filas) == []
    assert sum(1 for f in filas if f["evento"] == "llamada") >= 3   # initialize + list + la llamada con tabla


# --------------------------------------------------------------------------- #
# reauditoria: ancla externa contra una base real y servicio arrancado         #
# --------------------------------------------------------------------------- #

def test_truncar_la_cola_en_la_tabla_real_se_detecta_con_el_ancla_publicada(basedb):
    anclas = []

    async def caso(pool):
        emisor = EmisorTabla(pool)
        for i in range(6):
            await emisor(_reg(i))
        anclas.append(emisor.ancla())
    _correr(basedb, caso)
    con = basedb.admin()
    with con.cursor() as cur:
        cur.execute("DELETE FROM faro_bitacora WHERE seq >= 4")           # quien puede borrar la cola
    con.close()
    filas = basedb.filas()
    assert verificar_cadena(filas) == []                                  # sola, la cadena truncada "valida"
    assert any(p.codigo == "cola_truncada" for p in verificar_cadena(filas, anclas=anclas))


def test_el_servicio_arranca_contra_la_base_real_y_escribe_su_sonda(basedb, tmp_path):
    from jax.faro.servicio import arrancar
    repo = repo_de_juguete(tmp_path)
    sha = _git(repo, "rev-parse", "HEAD")
    paquete.construir_paquete(ConfigFaro(repo=repo, sha=sha, destino=tmp_path / "eco", uid_duenio=os.getuid()))
    d = tmp_path / "run"
    d.mkdir(mode=0o700)
    env = {"JAX_FARO_REPO": str(repo), "JAX_FARO_SHA": sha, "JAX_FARO_ECOSISTEMA_DIR": str(tmp_path / "eco"),
           "JAX_FARO_DUENIO_UID": str(os.getuid()), "JAX_FARO_SOCKET_DIR": str(d),
           "JAX_FARO_BITACORA_DB_HOST": basedb.srv["host"], "JAX_FARO_BITACORA_DB_PORT": str(basedb.srv["puerto"]),
           "JAX_FARO_BITACORA_DB_USER": basedb.usuario, "JAX_FARO_BITACORA_DB_PASSWORD": basedb.clave_usuario,
           "JAX_FARO_BITACORA_DB_NAME": basedb.base}

    async def caso():
        s = await arrancar(env, solo_pruebas_mismo_uid=True)
        try:
            async with s.crear_puerto(ejecucion()) as srv, cliente_por_rele(srv) as c:
                assert not (await c.call_tool("skills.leer", {"nombre": "alfa"})).is_error
        finally:
            await s.cerrar()
    asyncio.run(caso())
    filas = basedb.filas()
    assert [f["evento"] for f in filas][:2] == ["inicio_cadena", "servicio_iniciado"] and verificar_cadena(filas) == []
    assert sum(1 for f in filas if f["evento"] == "llamada") >= 3


def test_sin_la_tabla_el_servicio_no_arranca(basedb, tmp_path):
    """Con la tabla borrada (la base existe pero sin migrar) la sonda falla y el servicio no arranca."""
    from jax.faro.servicio import arrancar
    con = basedb.admin()
    with con.cursor() as cur:
        cur.execute("DROP TABLE faro_bitacora")
    con.close()
    repo = repo_de_juguete(tmp_path)
    sha = _git(repo, "rev-parse", "HEAD")
    paquete.construir_paquete(ConfigFaro(repo=repo, sha=sha, destino=tmp_path / "eco", uid_duenio=os.getuid()))
    d = tmp_path / "run"
    d.mkdir(mode=0o700)
    env = {"JAX_FARO_REPO": str(repo), "JAX_FARO_SHA": sha, "JAX_FARO_ECOSISTEMA_DIR": str(tmp_path / "eco"),
           "JAX_FARO_DUENIO_UID": str(os.getuid()), "JAX_FARO_SOCKET_DIR": str(d),
           "JAX_FARO_BITACORA_DB_HOST": basedb.srv["host"], "JAX_FARO_BITACORA_DB_PORT": str(basedb.srv["puerto"]),
           "JAX_FARO_BITACORA_DB_USER": basedb.usuario, "JAX_FARO_BITACORA_DB_PASSWORD": basedb.clave_usuario,
           "JAX_FARO_BITACORA_DB_NAME": basedb.base}
    with pytest.raises(Exception):
        asyncio.run(arrancar(env, solo_pruebas_mismo_uid=True))
