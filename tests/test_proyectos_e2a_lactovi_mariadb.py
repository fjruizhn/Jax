"""Guion de LACTOVI (`scripts/proyectos_e2a_lactovi.py`), contra MariaDB real.

Base DESECHABLE y propia (`jax_memory_test_e2alac_<hex>`), creada y borrada por un
fixture de modulo. El disco es siempre `tmp_path`, nunca `~/jax-workspace`.
La fixture tiene la forma del dato real de LACTOVI: `fuente/` con subcarpetas,
`procesado/<sha>/ficha.json` y un `.claude-flow/` suelto en la raiz del proyecto.
Los DDL de tenants/usuarios/projects se toman de la prueba de E1 (misma fuente).
"""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import contextlib
import functools
import hashlib
import importlib.util
import json
import os
import stat
import threading
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


def test_mover_usa_el_workspace_de_datos_para_el_lock(tmp_path, monkeypatch):
    workspace = tmp_path / "datos"
    proyectos = workspace / "proyectos"
    proyectos.mkdir(parents=True)
    origen, destino = proyectos / "suelto", proyectos / "uuid"
    origen.mkdir()
    # Simular el .git file de un linked worktree y su commondir sin subprocess:
    # el checkout del código y este workspace temporal son repos distintos.
    common_git_dir = workspace / ".git"
    linked_git_dir = common_git_dir / "worktrees" / "linked"
    linked_git_dir.mkdir(parents=True)
    (linked_git_dir / "commondir").write_text("../..\n")
    linked = tmp_path / "linked"
    linked.mkdir()
    linked_git_dir_from_worktree = os.path.relpath(linked_git_dir, linked)
    (linked / ".git").write_text(f"gitdir: {linked_git_dir_from_worktree}\n")
    raices = []
    esperando_lock = threading.Event()
    rename_ejecutado = threading.Event()
    lock_real = lactovi.project_tree_lock

    @contextlib.contextmanager
    def lock_observado(root):
        raices.append(Path(root))
        esperando_lock.set()
        with lock_real(root):
            yield

    rename_real = lactovi.os.rename

    def rename_observado(src, dst):
        rename_ejecutado.set()
        return rename_real(src, dst)

    monkeypatch.setattr(lactovi, "project_tree_lock", lock_observado)
    monkeypatch.setattr(lactovi.os, "rename", rename_observado)
    from jax.core.project_tree_lock import project_tree_lock

    with ThreadPoolExecutor(max_workers=1) as executor:
        with project_tree_lock(linked):
            movimiento = executor.submit(lactovi._mover, origen, destino, workspace)
            assert esperando_lock.wait(timeout=2), "E2A no intentó adquirir el lock"
            assert not rename_ejecutado.wait(timeout=0.05), "E2A no compartió el lock con el linked worktree"
        movimiento.result(timeout=2)
    assert raices == [workspace]
    assert rename_ejecutado.is_set() and destino.is_dir() and not origen.exists()


def _ficha(sha: str, origen: str, estado: str) -> str:
    return json.dumps({"sha256": sha, "origen": origen, "extractor": "x", "extractor_version": "1",
                       "fecha": "2026-09-21T17:32:36-06:00", "estado": estado, "detalle": {}})


def _armar(ws: Path, *, duplicado: bool = False) -> dict[str, str]:
    """Proyecto suelto con 8 archivos regulares (ok, parcial, y seis sin ficha: aceptados, ocultos,
    por contenido y de tipo sin extractor), un symlink en una subcarpeta y `.claude-flow/`.
    Devuelve {ruta relativa en fuente/: sha}."""
    (ws / ".git").mkdir(exist_ok=True)
    base = ws / "proyectos" / _CARPETA
    (base / "fuente" / "02-modelo").mkdir(parents=True)
    (base / "fuente" / "avaluos").mkdir(parents=True)
    (base / ".claude-flow").mkdir()
    (base / ".claude-flow" / "estado.json").write_text("{}")
    contenidos = {"Escanear.pdf": b"pdf-ok", "02-modelo/modelo.XLSX": b"xlsx-parcial",
                  "avaluos/Avalúo 1.png": b"png-sin-ficha",
                  "avaluos/.oculto.pdf": b"oculto",          # archivo oculto, tipo aceptado -> en_cola
                  "avaluos/macro.xlsm": b"xlsm-sin-ficha",   # el extractor real lo acepta (ROTURA-2) -> en_cola
                  "avaluos/sin-extension": b"%PDF-1.4 sin ficha",   # PDF por contenido (ROTURA-2) -> en_cola
                  "avaluos/notas.txt": b"texto",             # el extractor real NO lo extrae -> sin_extractor
                  "avaluos/datos.xyz": b"tipo-no-aceptado"}  # sin ficha y sin extractor -> sin_extractor
    if duplicado:
        contenidos["avaluos/copia.png"] = b"png-sin-ficha"
    shas = {}
    for rel, data in contenidos.items():
        (base / "fuente" / rel).write_bytes(data)
        shas[rel] = _sha(data)
    (base / "fuente" / "02-modelo" / "enlace.pdf").symlink_to("../Escanear.pdf")   # symlink NO en el nivel superior
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
    assert r["archivos_fuente"] == 8 and r["fichas"] == 2 and r["filas_a_insertar"] == 8
    assert ".claude-flow" in r["ignorado"]
    assert r["ignorados"] == ["02-modelo/enlace.pdf"]            # MINOR-8: symlink de un nivel profundo
    assert r["estados"] == {"en_cola": 4, "listo": 1, "parcial": 1, "sin_extractor": 2}


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
    assert filas["avaluos/.oculto.pdf"]["estado"] == "en_cola"
    assert filas["avaluos/macro.xlsm"]["estado"] == "en_cola"            # ROTURA-2: la lista sale de compuerta
    assert filas["avaluos/sin-extension"]["estado"] == "en_cola"         # PDF detectado por contenido
    assert filas["avaluos/notas.txt"]["estado"] == "sin_extractor"
    assert filas["avaluos/datos.xyz"]["estado"] == "sin_extractor"   # MINOR-4
    assert filas["avaluos/datos.xyz"]["ruta_entrada"] is None and filas["avaluos/datos.xyz"]["tipo"] == "xyz"
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
    assert (r["archivos_fuente"], r["fichas"], r["filas_insertadas"]) == (8, 2, 8)
    assert r["ignorados"] == ["02-modelo/enlace.pdf"]
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
    assert r["archivos_fuente"] == 9 and r["filas_insertadas"] == 8 and r["duplicados_sha"] == 1
    assert len(await _filas(r["project_id"])) == 8


def test_aplicar_contra_produccion_exige_confirmacion(tmp_path, capsys):
    rc = lactovi.main(["--workspace", str(tmp_path), "--carpeta", _CARPETA, "--nombre", "X",
                       "--dueno-user-id", "1", "--database", "jax_memory", "--aplicar"])
    assert rc == 2 and "--confirmo-produccion" in capsys.readouterr().err


def _mapa_de(ws: Path) -> Path:
    (m,) = sorted((ws / "proyectos").glob(".e2a-lactovi-*.json"))
    return m


def _revertir(ws: Path, mapa: Path) -> int:
    return lactovi.main(["--revertir", str(mapa), "--database", _DB])


def _completar(mapa: Path) -> int:
    return lactovi.main(["--completar", str(mapa), "--database", _DB])


async def _estado_proyecto(pid: int) -> str:
    return (await _sql("SELECT status FROM jax_project_scope WHERE project_id=%s", (pid,), fetch=True))[0]["status"]


@requiere_servidor
@asincrono
async def test_commit_incierto_no_toca_el_disco_y_completar_termina(tmp_path, capsys, monkeypatch):
    """MINOR-5 y MINOR-6: si commit() lanza, el rename NO se deshace (codigo 4); despues
    --completar registra las filas que faltan."""
    dueno = await _dueno()
    shas = _armar(tmp_path)
    real = lactovi._insertar

    async def insertar_y_romper_commit(conn, pid, filas):
        n = await real(conn, pid, filas)

        async def malo():
            raise RuntimeError("corte en el commit")
        conn.commit = malo
        return n

    monkeypatch.setattr(lactovi, "_insertar", insertar_y_romper_commit)
    rc = await asyncio.to_thread(_correr, tmp_path, dueno, "--aplicar")
    assert rc == 4
    assert "verifica las filas" in capsys.readouterr().err
    proyectos = tmp_path / "proyectos"
    mapa = _mapa_de(tmp_path)
    datos = json.loads(mapa.read_text())
    assert (proyectos / datos["project_uuid"] / "fuente" / "Escanear.pdf").exists()      # el disco NO se toco
    assert not (proyectos / _CARPETA).exists()
    assert len(await _filas(datos["project_id"])) == 0                                   # la conexion cerrada revierte
    monkeypatch.setattr(lactovi, "_insertar", real)
    assert await asyncio.to_thread(_completar, mapa) == 0
    r = _salida(capsys)
    assert r["filas_insertadas_ahora"] == 8
    assert {f["nombre_original"] for f in await _filas(datos["project_id"])} == set(shas)


@requiere_servidor
@asincrono
async def test_completar_registra_las_filas_que_faltan_y_es_idempotente(tmp_path, capsys):
    dueno = await _dueno()
    _armar(tmp_path)
    assert await asyncio.to_thread(_correr, tmp_path, dueno, "--aplicar") == 0
    r = _salida(capsys)
    pid = r["project_id"]
    await _sql("DELETE FROM project_documents WHERE project_id=%s AND tipo IN ('pdf','xyz')",  # marcador-propio: base propia del modulo (uuid)
               (pid,))
    assert len(await _filas(pid)) == 5
    mapa = Path(r["mapa"])
    assert await asyncio.to_thread(_completar, mapa) == 0
    assert _salida(capsys)["filas_insertadas_ahora"] == 3
    assert len(await _filas(pid)) == 8
    assert await asyncio.to_thread(_completar, mapa) == 0                # segunda vez: nada nuevo
    assert _salida(capsys)["filas_insertadas_ahora"] == 0


@requiere_servidor
@asincrono
async def test_completar_con_sha_distinto_no_registra(tmp_path, capsys):
    dueno = await _dueno()
    _armar(tmp_path)
    assert await asyncio.to_thread(_correr, tmp_path, dueno, "--aplicar") == 0
    r = _salida(capsys)
    (tmp_path / "proyectos" / r["project_uuid"] / "fuente" / "Escanear.pdf").write_bytes(b"cambiado")
    await _sql("DELETE FROM project_documents WHERE project_id=%s",  # marcador-propio: base propia del modulo (uuid)
               (r["project_id"],))
    assert await asyncio.to_thread(_completar, Path(r["mapa"])) == 3
    assert len(await _filas(r["project_id"])) == 0


@requiere_servidor
@asincrono
async def test_revertir_se_niega_con_trabajos_en_vuelo(tmp_path, capsys):
    """MAJOR-2."""
    dueno = await _dueno()
    _armar(tmp_path)
    assert await asyncio.to_thread(_correr, tmp_path, dueno, "--aplicar") == 0
    r = _salida(capsys)
    pid = r["project_id"]
    await _sql("UPDATE project_documents SET estado='procesando' WHERE project_id=%s AND tipo='png'", (pid,))
    antes = len(await _filas(pid))
    assert await asyncio.to_thread(_revertir, tmp_path, Path(r["mapa"])) == 7
    assert "jax-platform" in capsys.readouterr().err
    assert (tmp_path / "proyectos" / r["project_uuid"]).is_dir()
    assert not (tmp_path / "proyectos" / _CARPETA).exists()
    assert len(await _filas(pid)) == antes
    assert await _estado_proyecto(pid) == "ACTIVE"


@requiere_servidor
@asincrono
async def test_revertir_reintentable_si_falla_el_archivado(tmp_path, capsys, monkeypatch):
    """MINOR-7: la carpeta ya volvio y las filas ya no estan, pero el archivado fallo."""
    dueno = await _dueno()
    _armar(tmp_path)
    assert await asyncio.to_thread(_correr, tmp_path, dueno, "--aplicar") == 0
    r = _salida(capsys)
    pid = r["project_id"]

    async def falla(self, *a, **k):
        raise RuntimeError("corte en el archivado")
    real = lactovi.ProjectAuthorityAdmin.set_project_lifecycle
    monkeypatch.setattr(lactovi.ProjectAuthorityAdmin, "set_project_lifecycle", falla)
    assert await asyncio.to_thread(_revertir, tmp_path, Path(r["mapa"])) == 5
    capsys.readouterr()
    assert (tmp_path / "proyectos" / _CARPETA).is_dir() and len(await _filas(pid)) == 0
    assert await _estado_proyecto(pid) == "ACTIVE"
    monkeypatch.setattr(lactovi.ProjectAuthorityAdmin, "set_project_lifecycle", real)
    assert await asyncio.to_thread(_revertir, tmp_path, Path(r["mapa"])) == 0
    assert _salida(capsys)["reintento"] is True
    assert await _estado_proyecto(pid) == "ARCHIVED"


@requiere_servidor
@asincrono
async def test_reaplicar_tras_corte_despues_de_create_project_funciona(tmp_path, capsys, monkeypatch):
    """MINOR-9: mismo proyecto (misma llave), ACTIVE, carpeta sin mover."""
    dueno = await _dueno()
    _armar(tmp_path)

    def corte(*a, **k):
        raise OSError("corte antes del mapa")
    real = lactovi._escribir_mapa
    monkeypatch.setattr(lactovi, "_escribir_mapa", corte)
    assert await asyncio.to_thread(_correr, tmp_path, dueno, "--aplicar") == 5
    capsys.readouterr()
    assert (tmp_path / "proyectos" / _CARPETA).is_dir()
    n = await _conteo("projects")
    monkeypatch.setattr(lactovi, "_escribir_mapa", real)
    assert await asyncio.to_thread(_correr, tmp_path, dueno, "--aplicar") == 0
    r = _salida(capsys)
    assert await _conteo("projects") == n                               # no creo otro proyecto
    assert len(await _filas(r["project_id"])) == 8


@requiere_servidor
@asincrono
async def test_reaplicar_con_proyecto_archivado_se_niega_y_dice_que_hacer(tmp_path, capsys):
    dueno = await _dueno()
    _armar(tmp_path)
    assert await asyncio.to_thread(_correr, tmp_path, dueno, "--aplicar") == 0
    r = _salida(capsys)
    assert await asyncio.to_thread(_revertir, tmp_path, Path(r["mapa"])) == 0
    capsys.readouterr()
    Path(r["mapa"]).rename(tmp_path / "proyectos" / "mapa-anterior.json")
    n = await _conteo("projects")
    assert await asyncio.to_thread(_correr, tmp_path, dueno, "--aplicar") == 1
    err = capsys.readouterr().err
    assert "ARCHIVED" in err and "RESTORE_PROJECT" in err
    assert (tmp_path / "proyectos" / _CARPETA).is_dir() and await _conteo("projects") == n
