"""Guion de LACTOVI (`scripts/proyectos_e2a_lactovi.py`), contra MariaDB real.

Base DESECHABLE y propia (`jax_memory_test_e2alac_<hex>`), creada y borrada por un
fixture de modulo. El disco es siempre `tmp_path`, nunca `~/jax-workspace`.
La fixture tiene la forma del dato real de LACTOVI: `fuente/` con subcarpetas,
`procesado/<sha>/ficha.json` y un `.claude-flow/` suelto en la raiz del proyecto.
Los DDL de tenants/usuarios/projects se toman de la prueba de E1 (misma fuente).
"""
from __future__ import annotations

import asyncio
import functools
import hashlib
import importlib.util
import json
import os
import stat
import uuid
from pathlib import Path

import aiomysql
import pytest

from base_de_test import es_base_de_test
from jax.core.db_connect_config import db_connect_timeout_seconds
from jax.memory.project_authority_migrations import apply_project_authority_migration
from test_proyectos_e1_migrar_mariadb import _DDL_TENANTS, _DDL_USERS, _conn_params, _projects_ddl

_RAIZ = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("proyectos_e2a_lactovi", _RAIZ / "scripts" / "proyectos_e2a_lactovi.py")
lactovi = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(lactovi)

_HAY_SERVIDOR = bool(os.getenv("JAX_DB_HOST"))
requiere_servidor = pytest.mark.skipif(not _HAY_SERVIDOR, reason="necesita MariaDB real (JAX_DB_HOST)")
_DB: str = ""
_CARPETA = "lacteos-victoria"


def asincrono(fn):
    @functools.wraps(fn)
    def wrapper(*a, **k):
        return asyncio.run(fn(*a, **k))
    return wrapper


async def _crear_base(nombre: str) -> None:
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
            await apply_project_authority_migration(cur)   # incluye 006a: project_documents
    finally:
        conn.close()


async def _borrar_base(nombre: str) -> None:
    assert es_base_de_test(nombre) and "_e2alac_" in nombre
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
    _DB = f"jax_memory_test_e2alac_{uuid.uuid4().hex[:8]}"
    try:
        asyncio.run(_crear_base(_DB))
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


async def _dueno() -> int:
    await _sql("INSERT IGNORE INTO jax_tenants (tenant_id,name,plan,status) VALUES (1,'legacy','personal','active')")
    return await _sql("INSERT INTO jax_users (tenant_id,email,password_hash,role,status) "
                      "VALUES (1,%s,'x','superadmin','active')", (f"u{uuid.uuid4().hex[:20]}@test.invalid",))


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _ficha(sha: str, origen: str, estado: str) -> str:
    return json.dumps({"sha256": sha, "origen": origen, "extractor": "x", "extractor_version": "1",
                       "fecha": "2026-09-21T17:32:36-06:00", "estado": estado, "detalle": {}})


def _armar(ws: Path, *, duplicado: bool = False) -> dict[str, str]:
    """Proyecto suelto con 3 archivos (ok, parcial, sin ficha) y `.claude-flow/`.
    Devuelve {ruta relativa en fuente/: sha}."""
    base = ws / "proyectos" / _CARPETA
    (base / "fuente" / "02-modelo").mkdir(parents=True)
    (base / "fuente" / "avaluos").mkdir(parents=True)
    (base / ".claude-flow").mkdir()
    (base / ".claude-flow" / "estado.json").write_text("{}")
    contenidos = {"Escanear.pdf": b"pdf-ok", "02-modelo/modelo.XLSX": b"xlsx-parcial",
                  "avaluos/Avalúo 1.png": b"png-sin-ficha"}
    if duplicado:
        contenidos["avaluos/copia.png"] = b"png-sin-ficha"
    shas = {}
    for rel, data in contenidos.items():
        (base / "fuente" / rel).write_bytes(data)
        shas[rel] = _sha(data)
    for rel, estado in (("Escanear.pdf", "ok"), ("02-modelo/modelo.XLSX", "parcial")):
        d = base / "procesado" / shas[rel]
        d.mkdir(parents=True)
        (d / "ficha.json").write_text(_ficha(shas[rel], f"fuente/{rel}", estado))
    return shas


def _foto(raiz: Path) -> dict[str, str]:
    """Todo el arbol: ruta relativa -> sha (o 'dir')."""
    out = {}
    for p in sorted(raiz.rglob("*")):
        out[str(p.relative_to(raiz))] = "dir" if p.is_dir() else _sha(p.read_bytes())
    return out


def _correr(ws: Path, dueno: int, *extra: str) -> int:
    return lactovi.main(["--workspace", str(ws), "--carpeta", _CARPETA, "--nombre", "Lácteos Victoria",
                         "--dueno-user-id", str(dueno), "--tenant-id", "1", "--database", _DB, *extra])


def _salida(capsys) -> dict:
    return json.loads(capsys.readouterr().out)


async def _filas(project_id: int) -> list[dict]:
    return await _sql("SELECT * FROM project_documents WHERE project_id=%s ORDER BY nombre_original",
                      (project_id,), fetch=True)


async def _conteo(tabla: str) -> int:
    return (await _sql(f"SELECT COUNT(*) c FROM `{tabla}`", fetch=True))[0]["c"]


@requiere_servidor
@asincrono
async def test_ensayo_no_escribe_nada(tmp_path, capsys):
    dueno = await _dueno()
    _armar(tmp_path)
    antes_disco = _foto(tmp_path)
    antes_db = (await _conteo("projects"), await _conteo("project_documents"), await _conteo("jax_project_scope"))
    rc = await asyncio.to_thread(_correr, tmp_path, dueno)
    assert rc == 0
    assert _foto(tmp_path) == antes_disco
    assert (await _conteo("projects"), await _conteo("project_documents"),
            await _conteo("jax_project_scope")) == antes_db
    r = _salida(capsys)
    assert r["ensayo"] is True
    assert r["archivos_fuente"] == 3 and r["fichas"] == 2 and r["filas_a_insertar"] == 3
    assert ".claude-flow" in r["ignorado"]


@requiere_servidor
@asincrono
async def test_aplicar_mueve_registra_y_conserva_sha(tmp_path, capsys):
    dueno = await _dueno()
    shas = _armar(tmp_path)
    rc = await asyncio.to_thread(_correr, tmp_path, dueno, "--aplicar")
    assert rc == 0
    r = _salida(capsys)
    uid, pid = r["project_uuid"], r["project_id"]
    proyectos = tmp_path / "proyectos"
    assert not (proyectos / _CARPETA).exists()
    nueva = proyectos / uid
    assert nueva.is_dir() and (nueva / ".claude-flow" / "estado.json").exists()
    fila_proy = await _sql("SELECT project_uuid FROM projects WHERE id=%s", (pid,), fetch=True)
    assert fila_proy[0]["project_uuid"] == uid
    # sha iguales tras mover
    for rel, sha in shas.items():
        assert _sha((nueva / "fuente" / rel).read_bytes()) == sha
    filas = {f["nombre_original"]: f for f in await _filas(pid)}
    assert set(filas) == set(shas)
    assert filas["Escanear.pdf"]["estado"] == "listo"
    assert filas["02-modelo/modelo.XLSX"]["estado"] == "parcial"
    assert filas["avaluos/Avalúo 1.png"]["estado"] == "en_cola"
    for rel, f in filas.items():
        assert f["sha256"] == shas[rel] and f["subido_por"] == dueno and f["job_id"] is None
        assert f["bytes"] == len((nueva / "fuente" / rel).read_bytes())
    assert filas["02-modelo/modelo.XLSX"]["tipo"] == "xlsx"          # minusculas, sin punto
    assert filas["Escanear.pdf"]["carpeta_procesado"] == f"proyectos/{uid}/procesado/{shas['Escanear.pdf']}"
    assert filas["Escanear.pdf"]["ruta_entrada"] is None
    # Decision del controlador: sin ficha -> ruta_entrada = su ruta en fuente/ relativa al workspace.
    assert filas["avaluos/Avalúo 1.png"]["ruta_entrada"] == f"proyectos/{uid}/fuente/avaluos/Avalúo 1.png"
    assert filas["avaluos/Avalúo 1.png"]["carpeta_procesado"] is None
    # conteos y mapa
    assert (r["archivos_fuente"], r["fichas"], r["filas_insertadas"]) == (3, 2, 3)
    mapa = Path(r["mapa"])
    assert mapa.parent == proyectos and mapa.name.startswith(".e2a-lactovi-")
    assert stat.S_IMODE(mapa.stat().st_mode) == 0o600
    datos = json.loads(mapa.read_text())
    assert datos["project_uuid"] == uid and datos["project_id"] == pid
    assert datos["ruta_vieja"] == str(proyectos / _CARPETA) and datos["ruta_nueva"] == str(nueva)
    assert datos["sha256_antes"] == shas


@requiere_servidor
@asincrono
async def test_aplicar_dos_veces_no_duplica(tmp_path, capsys):
    dueno = await _dueno()
    _armar(tmp_path)
    assert await asyncio.to_thread(_correr, tmp_path, dueno, "--aplicar") == 0
    capsys.readouterr()
    antes = (await _conteo("projects"), await _conteo("project_documents"))
    rc = await asyncio.to_thread(_correr, tmp_path, dueno, "--aplicar")
    assert rc == 2
    assert "no existe" in capsys.readouterr().err
    assert (await _conteo("projects"), await _conteo("project_documents")) == antes


@requiere_servidor
@asincrono
async def test_sha_distinto_revierte_el_rename(tmp_path, capsys, monkeypatch):
    dueno = await _dueno()
    shas = _armar(tmp_path)
    real = lactovi._hashear_fuente
    llamadas = []

    def tramposo(fuente):
        llamadas.append(fuente)
        h = real(fuente)
        if len(llamadas) >= 2:                      # el recalculo de DESPUES
            h = {**h, "Escanear.pdf": "0" * 64}
        return h

    monkeypatch.setattr(lactovi, "_hashear_fuente", tramposo)
    filas_antes = await _conteo("project_documents")
    rc = await asyncio.to_thread(_correr, tmp_path, dueno, "--aplicar")
    assert rc == 3
    proyectos = tmp_path / "proyectos"
    assert (proyectos / _CARPETA / "fuente" / "Escanear.pdf").exists()
    assert sorted(p.name for p in proyectos.iterdir() if p.is_dir()) == [_CARPETA]
    for rel, sha in shas.items():
        assert _sha((proyectos / _CARPETA / "fuente" / rel).read_bytes()) == sha
    assert await _conteo("project_documents") == filas_antes


@requiere_servidor
@asincrono
async def test_revertir_deja_todo_como_antes_y_proyecto_archivado(tmp_path, capsys):
    dueno = await _dueno()
    _armar(tmp_path)
    antes_disco = _foto(tmp_path)
    assert await asyncio.to_thread(_correr, tmp_path, dueno, "--aplicar") == 0
    r = _salida(capsys)
    pid = r["project_id"]
    rc = await asyncio.to_thread(
        lambda: lactovi.main(["--revertir", r["mapa"], "--database", _DB]))
    assert rc == 0
    proyectos = tmp_path / "proyectos"
    assert not (proyectos / r["project_uuid"]).exists()
    despues = {k: v for k, v in _foto(tmp_path).items() if not k.endswith(Path(r["mapa"]).name)}
    assert despues == antes_disco                              # mismo arbol, mismos sha
    assert len(await _filas(pid)) == 0
    estado = await _sql("SELECT status FROM jax_project_scope WHERE project_id=%s", (pid,), fetch=True)
    assert estado[0]["status"] == "ARCHIVED"                   # archivado, no borrado
    assert (await _sql("SELECT COUNT(*) c FROM projects WHERE id=%s", (pid,), fetch=True))[0]["c"] == 1


@requiere_servidor
@asincrono
async def test_mismo_sha_en_dos_archivos_es_una_sola_fila(tmp_path, capsys):
    dueno = await _dueno()
    _armar(tmp_path, duplicado=True)
    assert await asyncio.to_thread(_correr, tmp_path, dueno, "--aplicar") == 0
    r = _salida(capsys)
    assert r["archivos_fuente"] == 4 and r["filas_insertadas"] == 3 and r["duplicados_sha"] == 1
    assert len(await _filas(r["project_id"])) == 3


def test_aplicar_contra_produccion_exige_confirmacion(tmp_path, capsys):
    rc = lactovi.main(["--workspace", str(tmp_path), "--carpeta", _CARPETA, "--nombre", "X",
                       "--dueno-user-id", "1", "--database", "jax_memory", "--aplicar"])
    assert rc == 2 and "--confirmo-produccion" in capsys.readouterr().err
