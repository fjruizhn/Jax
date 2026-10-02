"""Guion de FKs de project_id (`scripts/proyectos_e1_fks.py`), contra MariaDB real.

Base DESECHABLE y propia por prueba: `jax_memory_test_e1fks_<8 hex>`, que el
fixture crea y borra siempre (DROP solo si la creo el fixture). Esquema minimo:
`projects(id)` y las cinco tablas con `id` y `project_id INT NULL` indexado;
`messages` puede llevar ademas una columna VECTOR con su indice HNSW.

La guarda de produccion se prueba sin base.
"""
from __future__ import annotations

import asyncio
import functools
import importlib.util
import json
import os
import uuid
from pathlib import Path

import aiomysql
import pytest

from base_de_test import es_base_de_test
from jax.core.db_connect_config import db_connect_timeout_seconds
from jax.memory import indice_vectorial as iv

_RAIZ = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("proyectos_e1_fks", _RAIZ / "scripts" / "proyectos_e1_fks.py")
fks = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fks)

_TABLAS = ("conversations", "messages", "facts", "decisions", "action_items")
_HAY_SERVIDOR = bool(os.getenv("JAX_DB_HOST"))
requiere_servidor = pytest.mark.skipif(not _HAY_SERVIDOR, reason="necesita MariaDB real (JAX_DB_HOST)")


def asincrono(fn):
    @functools.wraps(fn)
    def wrapper(*a, **k):
        return asyncio.run(fn(*a, **k))
    return wrapper


def _conn_params() -> dict:
    return dict(host=os.environ.get("JAX_DB_HOST", ""), port=int(os.environ.get("JAX_DB_PORT", "3306")),
                user=os.getenv("JAX_DB_USER", ""), password=os.getenv("JAX_DB_PASSWORD", ""))


async def _crear_base(nombre: str, *, vector: bool) -> None:
    assert es_base_de_test(nombre) and "_e1fks_" in nombre
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
            await cur.execute("CREATE TABLE projects (id INT AUTO_INCREMENT PRIMARY KEY, name VARCHAR(20)) "
                              "ENGINE=InnoDB")
            for t in _TABLAS:
                extra = ""
                if t == "messages" and vector:
                    extra = ", emb VECTOR(3) NOT NULL, VECTOR KEY idx_msg_emb (emb) DISTANCE=cosine"
                await cur.execute(f"CREATE TABLE `{t}` (id INT AUTO_INCREMENT PRIMARY KEY, project_id INT NULL"
                                  f"{extra}, KEY (project_id)) ENGINE=InnoDB")
    finally:
        conn.close()


async def _borrar_base(nombre: str) -> None:
    assert es_base_de_test(nombre) and "_e1fks_" in nombre
    conn = await aiomysql.connect(autocommit=True, connect_timeout=db_connect_timeout_seconds(), **_conn_params())
    try:
        async with conn.cursor() as cur:
            await cur.execute(f"DROP DATABASE IF EXISTS `{nombre}`")
    finally:
        conn.close()


@pytest.fixture
def base(request):
    """Fabrica de bases desechables: `base(vector=False)` devuelve el nombre."""
    creadas: list[str] = []

    def _fabricar(*, vector: bool = False) -> str:
        nombre = f"jax_memory_test_e1fks_{uuid.uuid4().hex[:8]}"
        creadas.append(nombre)               # se anota ANTES de crear: el DROP corre aunque falle a medias
        asyncio.run(_crear_base(nombre, vector=vector))
        return nombre

    yield _fabricar
    for n in creadas:
        asyncio.run(_borrar_base(n))


async def _sql(db: str, query: str, args: tuple = (), *, fetch: bool = False):
    conn = await aiomysql.connect(db=db, autocommit=True, connect_timeout=db_connect_timeout_seconds(),
                                  **_conn_params())
    try:
        async with conn.cursor() as cur:
            await cur.execute(query, args)
            return await cur.fetchall() if fetch else cur.lastrowid
    finally:
        conn.close()


async def _fks_existentes(db: str) -> set[str]:
    filas = await _sql(db, "SELECT TABLE_NAME FROM information_schema.KEY_COLUMN_USAGE "
                           "WHERE TABLE_SCHEMA=DATABASE() AND COLUMN_NAME='project_id' "
                           "AND REFERENCED_TABLE_NAME='projects'", fetch=True)
    return {r[0] for r in filas}


async def _correr(db: str, *, aplicar: bool = True) -> list[dict]:
    conn = await aiomysql.connect(db=db, autocommit=True, connect_timeout=db_connect_timeout_seconds(),
                                  **_conn_params())
    try:
        return await fks.procesar(conn)
    finally:
        conn.close()


def _por_tabla(res: list[dict]) -> dict[str, dict]:
    return {r["tabla"]: r for r in res}


@requiere_servidor
def test_aplica_las_cinco_y_la_segunda_corrida_no_hace_nada(base):
    db = base()
    asyncio.run(_sql(db, "INSERT INTO projects (id,name) VALUES (1,'h')"))
    for t in _TABLAS:
        asyncio.run(_sql(db, f"INSERT INTO `{t}` (project_id) VALUES (1)"))
        asyncio.run(_sql(db, f"INSERT INTO `{t}` (project_id) VALUES (NULL)"))
    r1 = _por_tabla(asyncio.run(_correr(db)))
    assert set(r1) == set(_TABLAS)
    for t in _TABLAS:
        assert set(r1[t]) >= {"tabla", "filas", "algoritmo", "segundos", "hnsw_intacto", "aplicada"}
        assert r1[t]["filas"] == 2
        assert r1[t]["aplicada"] is True, r1[t]
        assert r1[t]["algoritmo"] == "INPLACE", r1[t]
        assert isinstance(r1[t]["segundos"], float)
        assert r1[t]["hnsw_intacto"] is None            # sin indice VECTOR en messages: el detector no aplica
    assert asyncio.run(_fks_existentes(db)) == set(_TABLAS)
    r2 = _por_tabla(asyncio.run(_correr(db)))
    for t in _TABLAS:
        assert r2[t]["aplicada"] is True and r2[t]["ya_existia"] is True and r2[t]["algoritmo"] is None
    assert asyncio.run(_fks_existentes(db)) == set(_TABLAS)


@requiere_servidor
def test_un_huerfano_aborta_esa_tabla_y_no_las_demas(base):
    db = base()
    asyncio.run(_sql(db, "INSERT INTO projects (id,name) VALUES (1,'h')"))
    asyncio.run(_sql(db, "INSERT INTO `facts` (project_id) VALUES (999)"))      # huerfano
    r = _por_tabla(asyncio.run(_correr(db)))
    assert r["facts"]["aplicada"] is False
    assert "huerfan" in r["facts"]["error"] and "999" in r["facts"]["error"]
    for t in _TABLAS:
        if t != "facts":
            assert r[t]["aplicada"] is True, r[t]
    assert asyncio.run(_fks_existentes(db)) == set(_TABLAS) - {"facts"}


@requiere_servidor
def test_messages_con_indice_vectorial_sano_da_hnsw_intacto_true(base):
    db = base(vector=True)
    asyncio.run(_sql(db, "INSERT INTO projects (id,name) VALUES (1,'h')"))
    for i in range(5):
        asyncio.run(_sql(db, "INSERT INTO messages (project_id, emb) VALUES (1, VEC_FromText(%s))",
                         (f"[{i + 1},0.5,0.25]",)))
    r = _por_tabla(asyncio.run(_correr(db)))
    assert r["messages"]["aplicada"] is True, r["messages"]
    assert r["messages"]["hnsw_intacto"] is True
    for t in _TABLAS:
        if t != "messages":
            assert r[t]["hnsw_intacto"] is None


@requiere_servidor
def test_messages_con_indice_que_miente_da_hnsw_intacto_false(base, monkeypatch):
    db = base(vector=True)
    asyncio.run(_sql(db, "INSERT INTO projects (id,name) VALUES (1,'h')"))
    asyncio.run(_sql(db, "INSERT INTO messages (project_id, emb) VALUES (1, VEC_FromText('[1,0.5,0.25]'))"))

    async def roto(cur, tabla, indice, columna, muestra=iv.MUESTRA):
        return iv.Informe(tabla, indice, columna, 10, 10, por_el_indice=1, por_scan=10)
    monkeypatch.setattr(fks.iv, "revisar_uno", roto)
    r = _por_tabla(asyncio.run(_correr(db)))
    assert r["messages"]["hnsw_intacto"] is False


@requiere_servidor
def test_si_mariadb_rechaza_inplace_se_registra_copy_y_no_se_aplica(base, monkeypatch):
    db = base()
    asyncio.run(_sql(db, "INSERT INTO projects (id,name) VALUES (1,'h')"))
    real = fks._alter

    async def alter_rechazado(cur, tabla):
        if tabla == "decisions":
            raise aiomysql.NotSupportedError(1846, "ALGORITHM=INPLACE is not supported. Try ALGORITHM=COPY.")
        return await real(cur, tabla)
    monkeypatch.setattr(fks, "_alter", alter_rechazado)
    r = _por_tabla(asyncio.run(_correr(db)))
    assert r["decisions"]["algoritmo"] == "COPY" and r["decisions"]["aplicada"] is False
    assert r["facts"]["aplicada"] is True
    assert "decisions" not in asyncio.run(_fks_existentes(db))


def test_guarda_de_produccion_sin_base(capsys, monkeypatch):
    for k in ("JAX_DB_HOST", "JAX_DB_PORT", "JAX_DB_USER", "JAX_DB_PASSWORD", "JAX_DB_NAME"):
        monkeypatch.delenv(k, raising=False)
    assert fks.main(["--aplicar", "--database", "jax_memory"]) == 2
    assert "--confirmo-produccion" in capsys.readouterr().err
    monkeypatch.setenv("JAX_DB_NAME", "jax_memory")
    assert fks.main(["--aplicar"]) == 2                 # la base sale del entorno: misma guarda
    assert fks.main(["--ensayar", "--database", "jax_memory"]) == 2   # el ensayo es sobre una copia
    assert "copia" in capsys.readouterr().err
    monkeypatch.delenv("JAX_DB_NAME")
    assert fks.main(["--ensayar"]) == 2                 # sin base no corre


@requiere_servidor
def test_main_imprime_un_json_por_tabla_y_sale_con_0(base, capsys):
    db = base()
    asyncio.run(_sql(db, "INSERT INTO projects (id,name) VALUES (1,'h')"))
    assert fks.main(["--ensayar", "--database", db]) == 0
    lineas = [json.loads(x) for x in capsys.readouterr().out.strip().splitlines()]
    assert [x["tabla"] for x in lineas] == list(_TABLAS)
    assert all(x["aplicada"] for x in lineas)


@requiere_servidor
def test_main_sale_con_1_si_alguna_tabla_no_se_aplico(base, capsys):
    db = base()
    asyncio.run(_sql(db, "INSERT INTO `facts` (project_id) VALUES (999)"))
    assert fks.main(["--aplicar", "--database", db]) == 1


@requiere_servidor
def test_huerfano_que_entra_durante_el_alter_se_reporta_como_error(base, monkeypatch):
    """INPLACE no valida filas existentes: el conteo posterior es la red."""
    db = base()
    asyncio.run(_sql(db, "INSERT INTO projects (id,name) VALUES (1,'h')"))
    real = fks._alter

    async def alter_con_carrera(cur, tabla):
        await real(cur, tabla)
        if tabla == "facts":
            await cur.execute("SET SESSION foreign_key_checks=0")
            await cur.execute("INSERT INTO facts (project_id) VALUES (777)")
            await cur.execute("SET SESSION foreign_key_checks=1")
    monkeypatch.setattr(fks, "_alter", alter_con_carrera)
    r = _por_tabla(asyncio.run(_correr(db)))
    assert r["facts"]["aplicada"] is False and "777" in r["facts"]["error"] and "DROP FOREIGN KEY" in r["facts"]["error"]
    assert r["decisions"]["aplicada"] is True
