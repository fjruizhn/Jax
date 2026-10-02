"""Guion de migracion E1 (`scripts/proyectos_e1_migrar.py`), contra MariaDB real.

Base DESECHABLE y propia: un fixture de modulo crea `jax_memory_test_e1mig_<hex>`
y la borra con `DROP DATABASE` al terminar, aun si las pruebas fallan. No se usa
la base compartida porque `jax_project_membership_event` es append-only (sus
triggers impiden borrar) y tiene FK al alcance: la prueba no podria limpiar el
alcance del proyecto 1.

Esquema: `jax_tenants`/`jax_users` con la DDL exacta de jax-platform
(`backend/db/migrations.py`); `projects` como `_projects_ddl()` de
`test_project_authority_mariadb.py` mas la migracion REAL
`apply_project_authority_migration`. Las cinco tablas de contenido
(conversations, messages, facts, decisions, action_items) son TABLAS MINIMAS
(`id`, `project_id INT NULL`, `user_id`, y `tenant_id` en conversations): el guion solo lee y reescribe
`project_id`, asi que el resto de las columnas reales no influye.

La guarda de produccion se prueba sin base.
"""
from __future__ import annotations

import asyncio
import functools
import importlib.util
import json
import os
import stat
import re
import uuid
from pathlib import Path

import aiomysql
import pytest

from base_de_test import es_base_de_test
from jax.core.db_connect_config import db_connect_timeout_seconds
from jax.memory.project_authority_migrations import apply_project_authority_migration

_RAIZ = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("proyectos_e1_migrar", _RAIZ / "scripts" / "proyectos_e1_migrar.py")
migrar = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(migrar)

_SCHEMA_SQL = _RAIZ / "jax_memory_schema.sql"
_TABLAS = ("conversations", "messages", "facts", "decisions", "action_items")
_HAY_SERVIDOR = bool(os.getenv("JAX_DB_HOST"))
requiere_servidor = pytest.mark.skipif(not _HAY_SERVIDOR, reason="necesita MariaDB real (JAX_DB_HOST)")

_DDL_TENANTS = """CREATE TABLE IF NOT EXISTS jax_tenants (
  tenant_id INT AUTO_INCREMENT PRIMARY KEY,
  name VARCHAR(100) NOT NULL,
  plan VARCHAR(20) DEFAULT 'personal',
  status VARCHAR(20) DEFAULT 'active',
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4"""
_DDL_USERS = """CREATE TABLE IF NOT EXISTS jax_users (
  user_id INT AUTO_INCREMENT PRIMARY KEY,
  tenant_id INT NOT NULL,
  email VARCHAR(320) NOT NULL UNIQUE,
  password_hash VARCHAR(255) NOT NULL,
  role VARCHAR(20) DEFAULT 'operator',
  status VARCHAR(20) DEFAULT 'active',
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (tenant_id) REFERENCES jax_tenants(tenant_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4"""

_DB: str = ""


def asincrono(fn):
    @functools.wraps(fn)
    def wrapper(*a, **k):
        return asyncio.run(fn(*a, **k))
    return wrapper


def _conn_params() -> dict:
    return dict(host=os.environ.get("JAX_DB_HOST", ""), port=int(os.environ.get("JAX_DB_PORT", "3306")),
                user=os.getenv("JAX_DB_USER", ""), password=os.getenv("JAX_DB_PASSWORD", ""))


def _projects_ddl() -> str:
    m = re.search(r"CREATE TABLE `projects` \(.*?\n\)[^;\n]*", _SCHEMA_SQL.read_text(encoding="utf-8"), re.S)
    assert m, "projects no esta en jax_memory_schema.sql"
    return m.group(0)


async def _crear_base_y_esquema(nombre: str) -> None:
    assert es_base_de_test(nombre)
    conn = await aiomysql.connect(autocommit=True, connect_timeout=db_connect_timeout_seconds(), **_conn_params())
    try:
        async with conn.cursor() as cur:
            await cur.execute(f"CREATE DATABASE `{nombre}` CHARACTER SET utf8mb4")
    finally:
        conn.close()
    conn = await aiomysql.connect(db=nombre, autocommit=True, connect_timeout=db_connect_timeout_seconds(),
                                  **_conn_params())
    try:
        async with conn.cursor() as cur:
            await cur.execute(_DDL_TENANTS)
            await cur.execute(_DDL_USERS)
            await cur.execute(_projects_ddl())
            await apply_project_authority_migration(cur)
            for t in _TABLAS:
                await cur.execute(
                    f"CREATE TABLE `{t}` (id INT AUTO_INCREMENT PRIMARY KEY, project_id INT NULL, "
                    f"user_id INT NULL, tag VARCHAR(20) NULL, KEY (project_id)) ENGINE=InnoDB")
            await cur.execute("ALTER TABLE `conversations` ADD COLUMN tenant_id INT NULL")
    finally:
        conn.close()


async def _borrar_base(nombre: str) -> None:
    assert es_base_de_test(nombre) and "_e1mig_" in nombre
    conn = await aiomysql.connect(autocommit=True, connect_timeout=db_connect_timeout_seconds(), **_conn_params())
    try:
        async with conn.cursor() as cur:
            await cur.execute(f"DROP DATABASE IF EXISTS `{nombre}`")
    finally:
        conn.close()


@pytest.fixture(scope="module", autouse=True)
def _base_desechable():
    global _DB
    if not _HAY_SERVIDOR:
        yield
        return
    _DB = f"jax_memory_test_e1mig_{uuid.uuid4().hex[:8]}"
    try:
        asyncio.run(_crear_base_y_esquema(_DB))
        yield
    finally:
        asyncio.run(_borrar_base(_DB))


async def _sql(query: str, args: tuple = (), *, fetch: bool = False):
    conn = await aiomysql.connect(db=_DB, autocommit=True, cursorclass=aiomysql.DictCursor,
                                  connect_timeout=db_connect_timeout_seconds(), **_conn_params())
    try:
        async with conn.cursor() as cur:
            await cur.execute(query, args)
            return await cur.fetchall() if fetch else cur.lastrowid
    finally:
        conn.close()


async def _pool():
    return await aiomysql.create_pool(db=_DB, autocommit=True, minsize=1, maxsize=4,
                                      cursorclass=aiomysql.DictCursor,
                                      connect_timeout=db_connect_timeout_seconds(), **_conn_params())


async def _asegurar_tenant(t: int) -> None:
    await _sql("INSERT IGNORE INTO jax_tenants (tenant_id,name,plan,status) VALUES (%s,'legacy','personal','active')", (t,))


async def _crear_usuario(t: int, role: str = "superadmin") -> int:
    return await _sql("INSERT INTO jax_users (tenant_id,email,password_hash,role,status) VALUES (%s,%s,'x',%s,'active')",
                      (t, f"u{uuid.uuid4().hex[:20]}@test.invalid", role))


async def _actor_unico() -> int:
    """Un solo actor por base: el digest de create_project incluye al usuario."""
    await _asegurar_tenant(1)
    fila = await _sql("SELECT user_id FROM jax_users WHERE tenant_id=1 AND role='superadmin' ORDER BY user_id LIMIT 1",
                      fetch=True)
    return fila[0]["user_id"] if fila else await _crear_usuario(1)


async def _asegurar_proyecto_1() -> None:
    await _sql("INSERT IGNORE INTO projects (id,project_uuid,name,status) VALUES (1,UUID(),'HAMURABI','active')")


async def _sembrar_huerfanos(ids: list[int], tablas=("conversations", "messages")) -> None:
    for pid in ids:
        for t in tablas:
            n = 1 if t == "conversations" else 2
            for _ in range(n):
                await _sql(f"INSERT INTO `{t}` (project_id,tag) VALUES (%s,'semilla')", (pid,))


async def _contar(tabla: str, pid: int) -> int:
    return (await _sql(f"SELECT COUNT(*) c FROM `{tabla}` WHERE project_id=%s", (pid,), fetch=True))[0]["c"]


async def _status_alcance(pid: int) -> str:
    return (await _sql("SELECT status FROM jax_project_scope WHERE project_id=%s", (pid,), fetch=True))[0]["status"]


async def _rol(pid: int, uid: int) -> str:
    return (await _sql("SELECT project_role r FROM jax_project_membership WHERE project_id=%s AND user_id=%s "
                       "AND status='ACTIVE'", (pid, uid), fetch=True))[0]["r"]


async def _limpiar_contenido() -> None:
    for t in _TABLAS:
        await _sql(f"DELETE FROM `{t}`")


@requiere_servidor
@asincrono
async def test_aplicar_mueve_huerfanos_con_conteos_iguales_y_es_idempotente(tmp_path):
    t = 1
    await _asegurar_tenant(t)
    actor = await _crear_usuario(t)
    await _asegurar_proyecto_1()
    await _sembrar_huerfanos([900001, 900002, 1400055])
    pool = await _pool()
    try:
        r1 = await migrar.aplicar(pool, actor_user_id=actor, salida_reversion=str(tmp_path / 'rev1.json'))
        r2 = await migrar.aplicar(pool, actor_user_id=actor, salida_reversion=str(tmp_path / 'rev2.json'))
    finally:
        pool.close()
        await pool.wait_closed()
    ev = r1["evaluacion_project_id"]
    assert r2["evaluacion_project_id"] == ev
    assert sorted(r1["antes"]["huerfanos_ids"]) == [900001, 900002, 1400055]
    assert r1["despues"]["huerfanos_ids"] == []
    assert r1["antes"]["filas_por_tabla"]["conversations"] == 3
    assert r1["antes"]["filas_por_tabla"]["messages"] == 6
    assert await _contar("conversations", ev) == 3 and await _contar("messages", ev) == 6
    assert await _contar("conversations", 900001) == 0
    assert (await _status_alcance(ev)) == "ARCHIVED"
    assert r1["despues"]["hamurabi_con_alcance"] is True
    assert await _rol(1, actor) == "OWNER"
    assert r2["antes"]["huerfanos_ids"] == []
    assert await _contar("messages", ev) == 6          # la segunda corrida no cambia nada


@requiere_servidor
@asincrono
async def test_verificar_no_escribe():
    await _sembrar_huerfanos([900777], tablas=("facts",))
    pool = await _pool()
    try:
        antes_proy = (await _sql("SELECT COUNT(*) c FROM projects", fetch=True))[0]["c"]
        antes_scope = (await _sql("SELECT COUNT(*) c FROM jax_project_scope", fetch=True))[0]["c"]
        m = await migrar.medir(pool)
        despues_proy = (await _sql("SELECT COUNT(*) c FROM projects", fetch=True))[0]["c"]
        despues_scope = (await _sql("SELECT COUNT(*) c FROM jax_project_scope", fetch=True))[0]["c"]
    finally:
        pool.close()
        await pool.wait_closed()
    assert 900777 in m["huerfanos_ids"]
    assert m["filas_por_tabla"]["facts"] == 2
    assert (antes_proy, antes_scope) == (despues_proy, despues_scope)
    assert await _contar("facts", 900777) == 2


class _CursorSaboteado:
    """Delega en el cursor real, salvo el UPDATE de `messages`, que no mueve nada."""
    def __init__(self, real):
        self._real = real

    async def execute(self, sql, args=None):
        if sql.lstrip().upper().startswith("UPDATE `MESSAGES`"):
            return await self._real.execute("UPDATE `messages` SET project_id=project_id WHERE 1=0")
        return await (self._real.execute(sql, args) if args is not None else self._real.execute(sql))

    def __getattr__(self, n):
        return getattr(self._real, n)


class _ConnSaboteada:
    def __init__(self, real):
        self._real = real

    def cursor(self, *a, **k):
        real_cm = self._real.cursor(*a, **k)

        class _CM:
            async def __aenter__(_s):
                return _CursorSaboteado(await real_cm.__aenter__())

            async def __aexit__(_s, *e):
                return await real_cm.__aexit__(*e)
        return _CM()

    def __getattr__(self, n):
        return getattr(self._real, n)


class _PoolSaboteado:
    def __init__(self, real):
        self._real = real

    def acquire(self):
        real_cm = self._real.acquire()

        class _CM:
            async def __aenter__(_s):
                return _ConnSaboteada(await real_cm.__aenter__())

            async def __aexit__(_s, *e):
                return await real_cm.__aexit__(*e)
        return _CM()


@requiere_servidor
@asincrono
async def test_conteos_distintos_hacen_rollback_y_error(tmp_path):
    """Si el UPDATE de una tabla no mueve lo contado, nada queda a medias."""
    await _limpiar_contenido()
    await _sembrar_huerfanos([910001], tablas=("conversations", "messages", "facts"))
    pool = await _pool()
    try:
        with pytest.raises(migrar.ConteosNoCoinciden):
            await migrar._reescribir_en_transaccion(_PoolSaboteado(pool), [910001], 999999, str(tmp_path / 'rev.json'))
    finally:
        pool.close()
        await pool.wait_closed()
    assert await _contar("conversations", 910001) == 1      # revertido: sigue huerfana
    assert await _contar("messages", 910001) == 2
    assert await _contar("conversations", 999999) == 0
    assert not (tmp_path / "rev.json").exists()              # sin commit, sin archivo


def test_guarda_de_produccion_sin_base(capsys, monkeypatch):
    for k in ("JAX_DB_HOST", "JAX_DB_PORT", "JAX_DB_USER", "JAX_DB_PASSWORD", "JAX_DB_NAME"):
        monkeypatch.delenv(k, raising=False)
    rc = migrar.main(["--aplicar", "--actor-user-id", "1", "--database", "jax_memory"])
    assert rc == 2
    assert "--confirmo-produccion" in capsys.readouterr().err
    monkeypatch.setenv("JAX_DB_NAME", "jax_memory")
    assert migrar.main(["--aplicar", "--actor-user-id", "1", "--salida-reversion", "/tmp/x.json"]) == 2


def test_aplicar_sin_salida_reversion_sale_con_2(capsys, monkeypatch):
    monkeypatch.setenv("JAX_DB_NAME", "jax_memory_test")
    assert migrar.main(["--aplicar", "--actor-user-id", "1"]) == 2
    assert "--salida-reversion" in capsys.readouterr().err


async def _estado_filas() -> dict:
    out = {}
    for t in _TABLAS:
        for r in await _sql(f"SELECT id, project_id FROM `{t}`", fetch=True):
            out[(t, r["id"])] = r["project_id"]
    return out


@requiere_servidor
@asincrono
async def test_archivo_de_reversion_registra_cada_fila_con_su_id_anterior(tmp_path):
    actor = await _actor_unico()
    await _asegurar_proyecto_1()
    await _limpiar_contenido()
    await _sembrar_huerfanos([920001, 920002], tablas=("conversations", "messages", "decisions"))
    antes = await _estado_filas()
    ruta = tmp_path / "rev.json"
    pool = await _pool()
    try:
        r = await migrar.aplicar(pool, actor_user_id=actor, salida_reversion=str(ruta))
    finally:
        pool.close()
        await pool.wait_closed()
    datos = json.loads(ruta.read_text())
    assert datos["evaluacion_project_id"] == r["evaluacion_project_id"]
    registradas = {(f["tabla"], f["id"]): f["project_id_anterior"] for f in datos["filas"]}
    assert len(datos["filas"]) == len(registradas) == len(antes) == 10
    assert registradas == antes
    assert stat.S_IMODE(os.stat(ruta).st_mode) == 0o600


@requiere_servidor
@asincrono
async def test_archivo_existente_aborta_antes_de_tocar_la_base(tmp_path):
    actor = await _actor_unico()
    await _limpiar_contenido()
    await _sembrar_huerfanos([930001])
    ruta = tmp_path / "ya.json"
    ruta.write_text("previo")
    estado = await _estado_filas()
    n_scope = (await _sql("SELECT COUNT(*) c FROM jax_project_scope", fetch=True))[0]["c"]
    pool = await _pool()
    try:
        with pytest.raises(FileExistsError):
            await migrar.aplicar(pool, actor_user_id=actor, salida_reversion=str(ruta))
    finally:
        pool.close()
        await pool.wait_closed()
    assert ruta.read_text() == "previo"
    assert await _estado_filas() == estado
    assert (await _sql("SELECT COUNT(*) c FROM jax_project_scope", fetch=True))[0]["c"] == n_scope


class _PoolCommitFalla:
    """Pool cuya conexion falla en commit() y registra si se llamo a rollback()."""
    def __init__(self, real):
        self._real = real
        self.rollbacks = 0

    def acquire(self):
        real_cm = self._real.acquire()
        pool = self

        class _Conn:
            def __init__(c, real):
                c._real = real

            async def commit(c):
                raise ConnectionResetError("conexion caida en el commit")

            async def rollback(c):
                pool.rollbacks += 1
                return await c._real.rollback()

            def __getattr__(c, n):
                return getattr(c._real, n)

        class _CM:
            async def __aenter__(_s):
                return _Conn(await real_cm.__aenter__())

            async def __aexit__(_s, *e):
                return await real_cm.__aexit__(*e)
        return _CM()


@requiere_servidor
@asincrono
async def test_commit_incierto_conserva_el_mapa_como_incierto(tmp_path):
    await _limpiar_contenido()
    await _sembrar_huerfanos([950001], tablas=("conversations", "messages"))
    antes = {k: v for k, v in (await _estado_filas()).items()}
    destino = await _sql("INSERT INTO projects (project_uuid,name,status) VALUES (UUID(),'destino','active')")
    ruta = tmp_path / "rev.json"
    pool = await _pool()
    falso = _PoolCommitFalla(pool)
    try:
        with pytest.raises(migrar.CommitIncierto) as e:
            await migrar._reescribir_en_transaccion(falso, [950001], destino, str(ruta))
    finally:
        pool.close()
        await pool.wait_closed()
    incierto = tmp_path / "rev.json.incierto"
    assert not ruta.exists() and incierto.exists()
    assert stat.S_IMODE(os.stat(incierto).st_mode) == 0o600
    datos = json.loads(incierto.read_text())
    assert {(f["tabla"], f["id"]): f["project_id_anterior"] for f in datos["filas"]} == antes
    assert falso.rollbacks == 0                               # sin rollback a ciegas
    assert str(incierto) in str(e.value) and "--verificar" in str(e.value) and "desconocido" in str(e.value)


@requiere_servidor
@asincrono
async def test_huerfano_que_aparece_tras_el_commit_hace_fallar_aplicar(tmp_path, monkeypatch):
    actor = await _actor_unico()
    await _limpiar_contenido()
    await _sembrar_huerfanos([940001])
    real = migrar.medir
    llamadas = []

    async def medir_con_carrera(pool):
        llamadas.append(1)
        if len(llamadas) == 2:                       # la medicion posterior al commit
            await _sembrar_huerfanos([940002], tablas=("facts",))
        return await real(pool)
    monkeypatch.setattr(migrar, "medir", medir_con_carrera)
    pool = await _pool()
    try:
        with pytest.raises(migrar.HuerfanosRestantes) as e:
            await migrar.aplicar(pool, actor_user_id=actor, salida_reversion=str(tmp_path / "r.json"))
    finally:
        pool.close()
        await pool.wait_closed()
    assert "940002" in str(e.value) and "volver a correr" in str(e.value)


# ---------------------------------------------------------------- M-1: alcance de los huerfanos

async def _estado_global() -> tuple:
    return (await _estado_filas(),
            (await _sql("SELECT COUNT(*) c FROM jax_project_scope", fetch=True))[0]["c"],
            (await _sql("SELECT COUNT(*) c FROM projects", fetch=True))[0]["c"])


@requiere_servidor
@asincrono
async def test_id_huerfano_fuera_del_rango_reservado_aborta_sin_escribir(tmp_path):
    actor = await _actor_unico()
    await _limpiar_contenido()
    await _sembrar_huerfanos([930001], tablas=("conversations",))      # en rango
    await _sembrar_huerfanos([77], tablas=("messages",))               # fuera de rango
    antes = await _estado_global()
    ruta = tmp_path / "rev.json"
    pool = await _pool()
    try:
        with pytest.raises(migrar.HuerfanosFueraDeAlcance) as e:
            await migrar.aplicar(pool, actor_user_id=actor, salida_reversion=str(ruta))
        m = await migrar.medir(pool)
    finally:
        pool.close()
        await pool.wait_closed()
    assert "77" in str(e.value)
    assert await _estado_global() == antes                 # ni proyectos, ni alcance, ni filas
    assert not ruta.exists()
    assert m["fuera_de_alcance"]["ids_fuera_de_rango"] == [77]


@requiere_servidor
@asincrono
@pytest.mark.parametrize("tabla", ["conversations", "messages", "facts", "decisions", "action_items"])
async def test_fila_huerfana_de_usuario_de_otro_tenant_aborta_sin_escribir(tmp_path, tabla):
    actor = await _actor_unico()
    await _limpiar_contenido()
    await _asegurar_tenant(2)
    ajeno = await _crear_usuario(2, role="operator")
    await _sembrar_huerfanos([930002], tablas=("conversations",))
    fila = await _sql(f"INSERT INTO `{tabla}` (project_id,user_id,tag) VALUES (930002,%s,'ajena')", (ajeno,))
    antes = await _estado_global()
    ruta = tmp_path / "rev.json"
    pool = await _pool()
    try:
        with pytest.raises(migrar.HuerfanosFueraDeAlcance) as e:
            await migrar.aplicar(pool, actor_user_id=actor, salida_reversion=str(ruta))
        m = await migrar.medir(pool)
    finally:
        pool.close()
        await pool.wait_closed()
    assert str(ajeno) in str(e.value) and tabla in str(e.value)
    assert await _estado_global() == antes
    assert not ruta.exists()
    assert {"tabla": tabla, "id": fila, "user_id": ajeno} in m["fuera_de_alcance"]["filas_otro_tenant"]
    assert m["fuera_de_alcance"]["ids_fuera_de_rango"] == []


@requiere_servidor
@asincrono
async def test_conversacion_con_tenant_propio_ajeno_aborta(tmp_path):
    actor = await _actor_unico()
    await _limpiar_contenido()
    await _sql("INSERT INTO conversations (project_id,user_id,tenant_id) VALUES (930003,NULL,2)")
    pool = await _pool()
    try:
        with pytest.raises(migrar.HuerfanosFueraDeAlcance):
            await migrar.aplicar(pool, actor_user_id=actor, salida_reversion=str(tmp_path / "r.json"))
    finally:
        pool.close()
        await pool.wait_closed()


@requiere_servidor
@asincrono
async def test_huerfanos_del_tenant_1_y_sin_usuario_siguen_migrando(tmp_path):
    actor = await _actor_unico()
    await _limpiar_contenido()
    propio = await _crear_usuario(1, role="operator")
    await _sql("INSERT INTO conversations (project_id,user_id,tenant_id) VALUES (930004,%s,1)", (propio,))
    await _sql("INSERT INTO messages (project_id,user_id) VALUES (930004,NULL)")
    pool = await _pool()
    try:
        r = await migrar.aplicar(pool, actor_user_id=actor, salida_reversion=str(tmp_path / "r.json"))
    finally:
        pool.close()
        await pool.wait_closed()
    assert r["despues"]["huerfanos_ids"] == []
    assert r["despues"]["fuera_de_alcance"] == {"ids_fuera_de_rango": [], "filas_otro_tenant": [],
                                                "total_filas_otro_tenant": 0}


def test_main_sale_con_4_si_hay_huerfanos_fuera_de_alcance(tmp_path, capsys):
    if not _HAY_SERVIDOR:
        pytest.skip("necesita MariaDB real")
    actor = asyncio.run(_actor_unico())
    asyncio.run(_limpiar_contenido())
    asyncio.run(_sembrar_huerfanos([78], tablas=("facts",)))
    rc = migrar.main(["--aplicar", "--actor-user-id", str(actor), "--database", _DB,
                      "--salida-reversion", str(tmp_path / "r.json")])
    assert rc == 4
    assert "78" in capsys.readouterr().err


# ---------------------------------------------------------------- m-2: codigos de salida de main

def _main_con(monkeypatch, exc, tmp_path):
    async def falla(pool, **kw):
        raise exc
    monkeypatch.setattr(migrar, "aplicar", falla)
    return migrar.main(["--aplicar", "--actor-user-id", "1", "--database", _DB,
                        "--salida-reversion", str(tmp_path / "r.json")])


@requiere_servidor
def test_main_sale_con_1_si_quedan_huerfanos(monkeypatch, tmp_path, capsys):
    assert _main_con(monkeypatch, migrar.HuerfanosRestantes("quedan [1]; volver a correr"), tmp_path) == 1
    assert "quedan [1]" in capsys.readouterr().err


@requiere_servidor
def test_main_sale_con_3_si_el_commit_es_incierto(monkeypatch, tmp_path, capsys):
    assert _main_con(monkeypatch, migrar.CommitIncierto("desconocido; mapa en x.incierto"), tmp_path) == 3
    assert "x.incierto" in capsys.readouterr().err


@requiere_servidor
def test_main_sale_con_5_ante_una_excepcion_no_prevista(monkeypatch, tmp_path, capsys):
    rc = _main_con(monkeypatch, KeyError("inesperado"), tmp_path)
    err = capsys.readouterr().err
    assert rc == 5
    assert "no previsto" in err and "KeyError" in err and "NO reintentar" in err


@requiere_servidor
def test_main_sale_con_5_si_ni_siquiera_conecta(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("JAX_DB_PORT", "1")              # nadie escucha: OperationalError, no prevista
    rc = migrar.main(["--aplicar", "--actor-user-id", "1", "--database", _DB,
                      "--salida-reversion", str(tmp_path / "r.json")])
    assert rc == 5
    assert "no previsto" in capsys.readouterr().err


@requiere_servidor
@asincrono
async def test_commit_incierto_con_os_replace_fallido_sigue_siendo_incierto(tmp_path, monkeypatch):
    """Si no se puede renombrar el mapa, igual sale CommitIncierto (codigo 3) y
    dice que el mapa sigue en la ruta original."""
    await _limpiar_contenido()
    await _sembrar_huerfanos([950002], tablas=("conversations",))
    destino = await _sql("INSERT INTO projects (project_uuid,name,status) VALUES (UUID(),'destino2','active')")
    ruta = tmp_path / "rev.json"

    def roto(a, b):
        raise PermissionError("no se puede renombrar")
    monkeypatch.setattr(migrar.os, "replace", roto)
    pool = await _pool()
    try:
        with pytest.raises(migrar.CommitIncierto) as e:
            await migrar._reescribir_en_transaccion(_PoolCommitFalla(pool), [950002], destino, str(ruta))
    finally:
        pool.close()
        await pool.wait_closed()
    assert ruta.exists()
    assert str(ruta) in str(e.value) and "sigue en" in str(e.value)


# ---------------------------------------------------------------- M-3: --revertir

async def _aplicar_y_mapa(tmp_path, ids, tablas=("conversations", "messages", "facts")):
    actor = await _actor_unico()
    await _asegurar_proyecto_1()
    await _limpiar_contenido()
    await _sembrar_huerfanos(ids, tablas=tablas)
    antes = await _estado_filas()
    ruta = tmp_path / "rev.json"
    pool = await _pool()
    try:
        r = await migrar.aplicar(pool, actor_user_id=actor, salida_reversion=str(ruta))
    finally:
        pool.close()
        await pool.wait_closed()
    assert await _estado_filas() != antes
    return antes, ruta, r["evaluacion_project_id"]


@requiere_servidor
@asincrono
async def test_aplicar_y_revertir_deja_los_project_id_originales(tmp_path):
    antes, ruta, ev = await _aplicar_y_mapa(tmp_path, [960001, 960002])
    pool = await _pool()
    try:
        r = await migrar.revertir(pool, str(ruta))
    finally:
        pool.close()
        await pool.wait_closed()
    assert await _estado_filas() == antes
    assert r["revertidas"] == len(antes) == 10


@requiere_servidor
@asincrono
@pytest.mark.parametrize("estropicio", ["id_inexistente", "ya_movida_a_otro_proyecto"])
async def test_mapa_que_no_cuadra_hace_rollback_y_error(tmp_path, estropicio):
    antes, ruta, ev = await _aplicar_y_mapa(tmp_path, [960003])
    despues_de_aplicar = await _estado_filas()
    datos = json.loads(ruta.read_text())
    if estropicio == "id_inexistente":
        datos["filas"].append({"tabla": "facts", "id": 987654321, "project_id_anterior": 960003})
    else:
        victima = datos["filas"][-1]
        await _sql(f"UPDATE `{victima['tabla']}` SET project_id=424242 WHERE id=%s", (victima["id"],))
        despues_de_aplicar = await _estado_filas()
    ruta.write_text(json.dumps(datos))
    pool = await _pool()
    try:
        with pytest.raises(migrar.ReversionNoCuadra):
            await migrar.revertir(pool, str(ruta))
    finally:
        pool.close()
        await pool.wait_closed()
    assert await _estado_filas() == despues_de_aplicar        # nada a medias


@requiere_servidor
def test_mapa_con_tabla_no_valida_se_rechaza_antes_de_tocar_nada(tmp_path):
    ruta = tmp_path / "mal.json"
    ruta.write_text(json.dumps({"evaluacion_project_id": 5, "filas": [
        {"tabla": "jax_users; DROP TABLE x", "id": 1, "project_id_anterior": 2}]}))
    os.chmod(ruta, 0o600)
    with pytest.raises(migrar.MapaInvalido):
        migrar.cargar_mapa(str(ruta))
    assert migrar.main(["--revertir", str(ruta), "--database", _DB]) == 2


def test_revertir_sobre_produccion_exige_confirmacion(tmp_path, capsys, monkeypatch):
    for k in ("JAX_DB_HOST", "JAX_DB_PORT", "JAX_DB_USER", "JAX_DB_PASSWORD", "JAX_DB_NAME"):
        monkeypatch.delenv(k, raising=False)
    ruta = tmp_path / "ok.json"
    ruta.write_text(json.dumps({"evaluacion_project_id": 5, "filas": []}))
    os.chmod(ruta, 0o600)
    assert migrar.main(["--revertir", str(ruta), "--database", "jax_memory"]) == 2
    assert "--confirmo-produccion" in capsys.readouterr().err


@requiere_servidor
def test_main_revertir_por_cli_restaura_y_el_que_no_cuadra_sale_distinto_de_0(tmp_path, capsys):
    antes, ruta, ev = asyncio.run(_aplicar_y_mapa(tmp_path, [960004], tablas=("conversations", "decisions")))
    assert migrar.main(["--revertir", str(ruta), "--database", _DB]) == 0
    assert asyncio.run(_estado_filas()) == antes
    assert migrar.main(["--revertir", str(ruta), "--database", _DB]) == 6      # ya revertido: 0 filas != mapa
    assert "no cuadra" in capsys.readouterr().err


# ---------------------------------------------------------------- N-1/N-2: --revertir no confia en el mapa

async def _mapa_modificado(tmp_path, ids, cambio):
    antes, ruta, ev = await _aplicar_y_mapa(tmp_path, ids)
    datos = json.loads(ruta.read_text())
    await cambio(datos)
    ruta.write_text(json.dumps(datos))
    return await _estado_filas(), ruta


async def _revertir_debe_dar_6(ruta, estado):
    pool = await _pool()
    try:
        with pytest.raises(migrar.ReversionNoCuadra) as e:
            await migrar.revertir(pool, str(ruta))
    finally:
        pool.close()
        await pool.wait_closed()
    assert await _estado_filas() == estado                  # ningun UPDATE
    return str(e.value)


@requiere_servidor
@asincrono
async def test_revertir_rechaza_evaluacion_que_no_es_la_de_la_llave(tmp_path):
    ajeno = await _sql("INSERT INTO projects (project_uuid,name,status) VALUES (UUID(),'ajeno','active')")

    async def cambio(d):
        d["evaluacion_project_id"] = ajeno
    estado, ruta = await _mapa_modificado(tmp_path, [970001], cambio)
    msg = await _revertir_debe_dar_6(ruta, estado)
    assert "evaluacion" in msg.lower()


@requiere_servidor
@asincrono
async def test_revertir_rechaza_project_id_anterior_fuera_del_rango_reservado(tmp_path):
    async def cambio(d):
        d["filas"][0]["project_id_anterior"] = 77
    estado, ruta = await _mapa_modificado(tmp_path, [970002], cambio)
    assert "77" in await _revertir_debe_dar_6(ruta, estado)


@requiere_servidor
@asincrono
async def test_revertir_rechaza_project_id_anterior_que_existe_en_projects(tmp_path):
    await _sql("INSERT IGNORE INTO projects (id,project_uuid,name,status) VALUES (970100,UUID(),'real','active')")

    async def cambio(d):
        d["filas"][0]["project_id_anterior"] = 970100        # en rango, pero existe
    estado, ruta = await _mapa_modificado(tmp_path, [970003], cambio)
    assert "970100" in await _revertir_debe_dar_6(ruta, estado)


@requiere_servidor
@asincrono
async def test_reversion_no_cuadra_lista_las_entradas_que_fallaron(tmp_path):
    async def cambio(d):
        d["filas"].append({"tabla": "facts", "id": 987654321, "project_id_anterior": 970004})
    estado, ruta = await _mapa_modificado(tmp_path, [970004], cambio)
    msg = await _revertir_debe_dar_6(ruta, estado)
    assert "facts" in msg and "987654321" in msg


def test_mapa_con_permisos_abiertos_o_ajeno_se_rechaza_con_2(tmp_path, capsys, monkeypatch):
    ruta = tmp_path / "abierto.json"
    ruta.write_text(json.dumps({"evaluacion_project_id": 5, "filas": []}))
    os.chmod(ruta, 0o644)
    monkeypatch.setenv("JAX_DB_NAME", "jax_memory_test")
    assert migrar.main(["--revertir", str(ruta)]) == 2
    assert "0600" in capsys.readouterr().err
    os.chmod(ruta, 0o600)
    monkeypatch.setattr(migrar.os, "geteuid", lambda: os.getuid() + 1)       # otro usuario
    with pytest.raises(migrar.MapaInvalido, match="propiedad"):
        migrar.cargar_mapa(str(ruta))
