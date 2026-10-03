#!/usr/bin/env python3
"""Proyectos E2a: convierte la carpeta suelta de LACTOVI en un proyecto real.

Toma `<workspace>/proyectos/<carpeta>/` (con `fuente/` y `procesado/<sha256>/ficha.json`),
crea el proyecto con `ProjectAuthorityAdmin.create_project`, renombra la carpeta a
`proyectos/<project_uuid>/` y registra sus documentos en `project_documents`.

Uso:  python -m scripts.proyectos_e2a_lactovi --workspace RUTA --carpeta NOMBRE
          --nombre "Lácteos Victoria" --dueno-user-id N [--tenant-id 1]
          [--aplicar] [--database NOMBRE] [--confirmo-produccion]
      python -m scripts.proyectos_e2a_lactovi --revertir MAPA.json [--database NOMBRE]
          [--confirmo-produccion]

Sin `--aplicar` es un ENSAYO: solo lee (disco y base) e imprime lo que haria.
La conexion sale de JAX_DB_HOST/PORT/USER/PASSWORD/NAME del entorno; este guion
nunca abre /etc/jax/.env (el runbook carga el entorno).

`--aplicar`: sha256 de cada archivo de `fuente/` (antes) -> create_project ->
mapa de reversion (0600, O_EXCL, fsync) ANTES de mover -> os.rename -> sha256
(despues) y se exige igualdad exacta (si no, se deshace el rename: codigo 3) ->
una fila por sha256 en `project_documents` (INSERT IGNORE, en UNA transaccion; si
falla, rollback y se deshace el rename).
  - con ficha: estado de la ficha (ok->listo, parcial, error, sin_extractor),
    `carpeta_procesado`, `ruta_entrada` NULL;
  - sin ficha: `en_cola` con `ruta_entrada` = `proyectos/<uuid>/fuente/<ruta>` (relativa
    al workspace) para que el despachador de la plataforma la procese; la ingesta
    reconoce el mismo contenido ya presente en fuente/ y no lo duplica.
Todo lo que cuelga de la carpeta fuera de `fuente/` y `procesado/` (p. ej.
`.claude-flow/`) se ignora para el registro (viaja con el rename) y se menciona.

`--revertir MAPA`: en una transaccion borra las filas de `project_documents` del
proyecto, devuelve la carpeta a su nombre verificando los sha256 del mapa y deja el
proyecto ARCHIVADO (Destruir no existe). Solo vale mientras `fuente/` no haya
cambiado: si la plataforma ya recibio subidas, los sha256 no cuadran (codigo 3) y no
se toca nada.

Codigos de salida: 0 hecho; 1 estado que impide seguir (mapa o destino ya existen,
proyecto no ACTIVE, fallo al registrar: se deshizo el rename); 2 argumentos, guarda
o carpeta que no existe (p. ej. ya se movio); 3 los sha256 no cuadran (se deshizo el
rename); 5 error no previsto (no reintentar sin revisar); 6 (solo --revertir) el mapa
no cuadra con la base.
"""
from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import os
import sys
import uuid
from pathlib import Path

import aiomysql

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jax.core.db_connect_config import db_connect_timeout_seconds  # noqa: E402
from jax.memory.b9 import MutationAuthorizationRequest, ScopeContext, Visibility  # noqa: E402
from jax.memory.b9_mariadb import MariaDBB9Store  # noqa: E402
from jax.memory.project_authority import ProjectAuthorityAdmin  # noqa: E402
from jax.memory.scope_authority import ProjectLifecycle  # noqa: E402
from procesamiento.ficha import Ficha, sha256_de  # noqa: E402

BASE_PRODUCCION = "jax_memory"
COMPONENTE = "proyectos-e2a-lactovi"
ESTADO_DE_FICHA = {"ok": "listo", "parcial": "parcial", "error": "error", "sin_extractor": "sin_extractor"}
_LARGO_NOMBRE = 1024   # project_documents.nombre_original VARCHAR(1024)
_LARGO_TIPO = 16       # project_documents.tipo VARCHAR(16)


class GuardaFallida(RuntimeError):
    """Argumento o precondicion que impide empezar, antes de escribir nada (codigo 2)."""


class EstadoImpide(RuntimeError):
    """Algo ya existe o no esta en el estado esperado (codigo 1)."""


class ShaNoCuadra(RuntimeError):
    """Los sha256 despues de mover no son los de antes; el rename ya se deshizo (codigo 3)."""


class MapaNoCuadra(RuntimeError):
    """El mapa de reversion no coincide con la base (codigo 6)."""


# --------------------------------------------------------------------------- disco

def _hashear_fuente(fuente: Path) -> dict[str, str]:
    """{ruta relativa (posix) dentro de fuente/: sha256} de cada archivo regular."""
    out: dict[str, str] = {}
    for raiz, _dirs, archivos in os.walk(fuente, followlinks=False):
        for nombre in archivos:
            p = Path(raiz) / nombre
            if p.is_symlink() or not p.is_file():
                continue
            out[p.relative_to(fuente).as_posix()] = sha256_de(p)
    return dict(sorted(out.items()))


def _leer_fichas(procesado: Path) -> dict[str, Ficha]:
    """{sha256: Ficha}. Una ficha ilegible, o cuyo sha no es el de su carpeta, aborta
    ANTES de escribir nada: registrar a ciegas es peor que parar."""
    fichas: dict[str, Ficha] = {}
    if not procesado.is_dir():
        return fichas
    for d in sorted(procesado.iterdir()):
        archivo = d / "ficha.json"
        if not d.is_dir() or not archivo.is_file():
            continue
        try:
            ficha = Ficha.desde_json(archivo.read_text(encoding="utf-8"))
        except (ValueError, OSError) as e:
            raise GuardaFallida(f"ficha invalida {archivo}: {e}") from e
        if ficha.sha256 != d.name:
            raise GuardaFallida(f"la ficha {archivo} dice sha256 {ficha.sha256}, distinto del de su carpeta")
        fichas[ficha.sha256] = ficha
    return fichas


def _planear(base: Path) -> dict:
    """Todo lo que se registraria, sin tocar nada. `filas` aun sin uuid: ruta_entrada
    y carpeta_procesado se completan con `_filas`."""
    fuente = base / "fuente"
    if not fuente.is_dir():
        raise GuardaFallida(f"{base} no tiene fuente/")
    hashes = _hashear_fuente(fuente)
    fichas = _leer_fichas(base / "procesado")
    primero: dict[str, str] = {}              # sha -> primera ruta (orden alfabetico)
    for rel, sha in hashes.items():
        primero.setdefault(sha, rel)
    demasiado_largas = [r for r in primero.values() if len(r) > _LARGO_NOMBRE]
    if demasiado_largas:
        raise GuardaFallida(f"rutas de mas de {_LARGO_NOMBRE} caracteres: {demasiado_largas[:5]}")
    ignorado = sorted(p.name for p in base.iterdir() if p.name not in ("fuente", "procesado"))
    return {
        "hashes": hashes, "fichas": fichas, "primero": primero, "ignorado": ignorado,
        "duplicados_sha": len(hashes) - len(primero),
        "fichas_sin_archivo": sorted(set(fichas) - set(primero)),
        "bytes": {sha: (fuente / rel).stat().st_size for sha, rel in primero.items()},
    }


def _filas(plan: dict, project_uuid: str, dueno: int) -> list[dict]:
    filas = []
    for sha, rel in plan["primero"].items():
        ficha = plan["fichas"].get(sha)
        ext = Path(rel).suffix.lstrip(".").lower()[:_LARGO_TIPO]
        fila = {"sha256": sha, "nombre_original": rel, "bytes": plan["bytes"][sha], "tipo": ext,
                "subido_por": dueno}
        if ficha is not None:
            fila.update(estado=ESTADO_DE_FICHA[ficha.estado], ruta_entrada=None,
                        carpeta_procesado=f"proyectos/{project_uuid}/procesado/{sha}")
        else:
            fila.update(estado="en_cola", carpeta_procesado=None,
                        ruta_entrada=f"proyectos/{project_uuid}/fuente/{rel}")
        filas.append(fila)
    return filas


def _escribir_mapa(ruta: Path, datos: dict) -> None:
    """0600, O_EXCL (jamas pisa uno existente), fsync. Se llama ANTES de mover."""
    fd = os.open(ruta, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(datos, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
    except BaseException:
        try:
            os.unlink(ruta)
        except OSError:  # fail-soft: limpieza del archivo parcial; el `raise` relanza el error ORIGINAL
            pass
        raise


def _mover(origen: Path, destino: Path) -> None:
    """rename sin pisar: `os.rename` reemplazaria un directorio destino vacio."""
    if os.path.lexists(destino):
        raise EstadoImpide(f"el destino ya existe: {destino}")
    os.rename(origen, destino)


# --------------------------------------------------------------------------- base

def _scope(actor: int, tenant: int, project_id: int | None) -> ScopeContext:
    return ScopeContext(actor_principal=f"user:{actor}", actor_type="USER", subject_user_id=str(actor),
                        tenant_id=str(tenant), project_id=str(project_id) if project_id is not None else None,
                        calling_component=COMPONENTE)


def _req(actor: int, tenant: int, op: str, project_id: int | None) -> MutationAuthorizationRequest:
    return MutationAuthorizationRequest(_scope(actor, tenant, project_id), op, Visibility.PROJECT_SHARED)


def _llave(base: Path) -> str:
    """Identifica ESTA carpeta de ESTE workspace: create_project exige un UUID de 36."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"e2a-lactovi:{base.resolve()}"))


async def _dueno_activo(pool, dueno: int, tenant: int) -> bool:
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute("SELECT status FROM jax_users WHERE user_id=%s AND tenant_id=%s", (dueno, tenant))
            fila = await cur.fetchone()
    return bool(fila) and str(fila["status"]).upper() == "ACTIVE"


async def _estado_alcance(pool, project_id: int) -> str | None:
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute("SELECT status FROM jax_project_scope WHERE project_id=%s", (project_id,))
            fila = await cur.fetchone()
    return str(fila["status"]).upper() if fila else None


_INSERT = ("INSERT IGNORE INTO project_documents (project_id,sha256,nombre_original,ruta_entrada,"
           "carpeta_procesado,bytes,tipo,estado,job_id,subido_por) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,NULL,%s)")


async def _insertar(conn, project_id: int, filas: list[dict]) -> int:
    """Dentro de la transaccion abierta por quien llama."""
    insertadas = 0
    async with conn.cursor() as cur:
        for f in filas:
            await cur.execute(_INSERT, (project_id, f["sha256"], f["nombre_original"], f["ruta_entrada"],
                                        f["carpeta_procesado"], f["bytes"], f["tipo"], f["estado"],
                                        f["subido_por"]))
            insertadas += int(cur.rowcount)
    return insertadas


# --------------------------------------------------------------------------- operaciones

def _resumen(plan: dict) -> dict:
    return {"archivos_fuente": len(plan["hashes"]), "fichas": len(plan["fichas"]),
            "duplicados_sha": plan["duplicados_sha"], "fichas_sin_archivo": plan["fichas_sin_archivo"],
            "ignorado": plan["ignorado"]}


def _validar_carpeta(proyectos: Path, carpeta: str) -> Path:
    if not carpeta or "/" in carpeta or carpeta in (".", ".."):
        raise GuardaFallida(f"--carpeta debe ser un nombre simple, no {carpeta!r}")
    if not proyectos.is_dir():
        raise GuardaFallida(f"{proyectos} no existe")
    base = proyectos / carpeta
    if base.is_symlink() or not base.is_dir():
        raise GuardaFallida(f"la carpeta {base} no existe (¿ya se movio? mirar el mapa .e2a-lactovi-*.json "
                            f"en {proyectos}); no se creo ningun proyecto")
    return base


def _ruta_mapa(proyectos: Path) -> Path:
    return proyectos / f".e2a-lactovi-{datetime.date.today():%Y%m%d}.json"


async def ensayar(pool, args) -> dict:
    proyectos = Path(args.workspace) / "proyectos"
    base = _validar_carpeta(proyectos, args.carpeta)
    plan = _planear(base)
    if not await _dueno_activo(pool, args.dueno_user_id, args.tenant_id):
        raise GuardaFallida(f"el usuario {args.dueno_user_id} no esta activo en el tenant {args.tenant_id}")
    mapa = _ruta_mapa(proyectos)
    if os.path.lexists(mapa):
        raise EstadoImpide(f"el mapa de reversion ya existe: {mapa}")
    return {"ensayo": True, **_resumen(plan), "filas_a_insertar": len(plan["primero"]),
            "mapa_previsto": str(mapa), "carpeta_actual": str(base),
            "estados": sorted({"en_cola" if s not in plan["fichas"] else ESTADO_DE_FICHA[plan["fichas"][s].estado]
                               for s in plan["primero"]})}


async def aplicar(pool, args) -> dict:
    proyectos = Path(args.workspace) / "proyectos"
    base = _validar_carpeta(proyectos, args.carpeta)
    if not await _dueno_activo(pool, args.dueno_user_id, args.tenant_id):
        raise GuardaFallida(f"el usuario {args.dueno_user_id} no esta activo en el tenant {args.tenant_id}")
    mapa = _ruta_mapa(proyectos)
    if os.path.lexists(mapa):
        raise EstadoImpide(f"el mapa de reversion ya existe: {mapa}")
    plan = _planear(base)                                                   # 1. sha256 (antes)
    admin = ProjectAuthorityAdmin(MariaDBB9Store(pool))
    creado = await admin.create_project(                                     # 2. proyecto
        _req(args.dueno_user_id, args.tenant_id, "CREATE_PROJECT", None), name=args.nombre,
        description=None, idempotency_key=_llave(base))
    estado = await _estado_alcance(pool, creado.project_id)
    if estado != "ACTIVE":
        raise EstadoImpide(f"el proyecto {creado.project_id} de esta carpeta ya existia y esta {estado}, no ACTIVE; "
                           f"no se movio nada")
    nueva = proyectos / creado.project_uuid
    _escribir_mapa(mapa, {                                                   # 3. mapa, antes de mover
        "version": 1, "project_id": creado.project_id, "project_uuid": creado.project_uuid,
        "tenant_id": args.tenant_id, "dueno_user_id": args.dueno_user_id, "nombre": args.nombre,
        "ruta_vieja": str(base), "ruta_nueva": str(nueva), "sha256_antes": plan["hashes"]})
    _mover(base, nueva)                                                      # 4. rename (si falla, nada se borra)
    try:
        despues = _hashear_fuente(nueva / "fuente")                          # 5. sha256 (despues)
        if despues != plan["hashes"]:
            diff = sorted(set(despues.items()) ^ set(plan["hashes"].items()))[:5]
            raise ShaNoCuadra(f"los sha256 despues de mover no son los de antes (primeras diferencias: {diff}); "
                              f"se deshizo el rename y la carpeta volvio a {base}")
        filas = _filas(plan, creado.project_uuid, args.dueno_user_id)
        async with pool.acquire() as conn:                                   # 6. registro, UNA transaccion
            await conn.begin()
            try:
                insertadas = await _insertar(conn, creado.project_id, filas)
                await conn.commit()
            except BaseException:
                await conn.rollback()
                raise
    except ShaNoCuadra:
        os.rename(nueva, base)
        raise
    except BaseException:
        os.rename(nueva, base)
        print(f"ERROR al registrar documentos: se deshizo el rename (la carpeta volvio a {base}); el proyecto "
              f"{creado.project_id} sigue creado y ACTIVO, y el mapa {mapa} se conserva", file=sys.stderr)
        raise
    return {"ensayo": False, "project_id": creado.project_id, "project_uuid": creado.project_uuid,
            **_resumen(plan), "filas_insertadas": insertadas, "mapa": str(mapa)}   # 7. conteos


def cargar_mapa(ruta: str) -> dict:
    try:
        with open(ruta, encoding="utf-8") as f:
            st = os.fstat(f.fileno())
            if st.st_uid != os.geteuid():
                raise GuardaFallida(f"el mapa {ruta} no es del usuario que corre; se rechaza")
            if st.st_mode & 0o177:
                raise GuardaFallida(f"el mapa {ruta} tiene permisos {oct(st.st_mode & 0o777)}; tiene que ser 0600")
            datos = json.load(f)
    except (OSError, ValueError) as e:
        raise GuardaFallida(f"no se pudo leer el mapa {ruta}: {type(e).__name__}: {e}") from e
    try:
        for k in ("project_id", "tenant_id", "dueno_user_id"):
            if not isinstance(datos[k], int) or isinstance(datos[k], bool):
                raise TypeError(f"{k} debe ser entero")
        for k in ("project_uuid", "ruta_vieja", "ruta_nueva"):
            if not isinstance(datos[k], str) or not datos[k]:
                raise TypeError(f"{k} debe ser texto")
        if not isinstance(datos["sha256_antes"], dict):
            raise TypeError("sha256_antes debe ser un objeto")
        nueva, vieja = Path(datos["ruta_nueva"]), Path(datos["ruta_vieja"])
        if nueva.name != datos["project_uuid"] or nueva.parent != vieja.parent:
            raise ValueError("ruta_nueva debe ser <proyectos>/<project_uuid> junto a ruta_vieja")
    except (KeyError, TypeError, ValueError) as e:
        raise GuardaFallida(f"mapa de reversion invalido: {type(e).__name__}: {e}") from e
    return datos


async def revertir(pool, ruta_mapa: str) -> dict:
    mapa = cargar_mapa(ruta_mapa)
    pid, uid = mapa["project_id"], mapa["project_uuid"]
    nueva, vieja = Path(mapa["ruta_nueva"]), Path(mapa["ruta_vieja"])
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute("SELECT project_uuid FROM projects WHERE id=%s", (pid,))
            fila = await cur.fetchone()
    if not fila or str(fila["project_uuid"]) != uid:
        raise MapaNoCuadra(f"el project_uuid del mapa ({uid}) no es el del proyecto {pid} en la base; "
                           f"no se toco nada")
    if not nueva.is_dir():
        raise GuardaFallida(f"{nueva} no existe: nada que revertir")
    if os.path.lexists(vieja):
        raise EstadoImpide(f"{vieja} ya existe: no se puede devolver la carpeta")
    async with pool.acquire() as conn:
        await conn.begin()
        movida = False
        try:
            async with conn.cursor() as cur:
                await cur.execute("DELETE FROM project_documents WHERE project_id=%s", (pid,))
                borradas = int(cur.rowcount)
            _mover(nueva, vieja)
            movida = True
            if _hashear_fuente(vieja / "fuente") != mapa["sha256_antes"]:
                raise ShaNoCuadra(f"los sha256 de fuente/ ya no son los del mapa (¿hubo subidas despues de "
                                  f"aplicar?); no se revirtio nada, la carpeta sigue en {nueva}")
            await conn.commit()
        except BaseException:
            await conn.rollback()
            if movida:
                os.rename(vieja, nueva)
            raise
    admin = ProjectAuthorityAdmin(MariaDBB9Store(pool))
    cambio = await admin.set_project_lifecycle(
        _req(mapa["dueno_user_id"], mapa["tenant_id"], "SET_PROJECT_LIFECYCLE", pid), pid,
        ProjectLifecycle.ARCHIVED)
    return {"revertido": True, "project_id": pid, "project_uuid": uid, "filas_borradas": borradas,
            "carpeta": str(vieja), "proyecto_archivado": True, "archivado_ahora": bool(cambio)}


# --------------------------------------------------------------------------- CLI

def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--workspace", help="raiz del workspace (contiene proyectos/)")
    p.add_argument("--carpeta", help="nombre de la carpeta suelta dentro de proyectos/")
    p.add_argument("--nombre", help="nombre del proyecto")
    p.add_argument("--dueno-user-id", type=int, default=None)
    p.add_argument("--tenant-id", type=int, default=1)
    p.add_argument("--aplicar", action="store_true", help="escribe; sin esto es un ensayo")
    p.add_argument("--revertir", metavar="MAPA.json", default=None)
    p.add_argument("--database", default=None, help="pisa a JAX_DB_NAME")
    p.add_argument("--confirmo-produccion", action="store_true",
                   help=f"obligatorio con --aplicar y --revertir si la base es {BASE_PRODUCCION}")
    return p


async def _correr(args, database: str) -> dict:
    pool = await aiomysql.create_pool(
        host=os.environ.get("JAX_DB_HOST", ""), port=int(os.environ.get("JAX_DB_PORT", "3306")),
        user=os.environ.get("JAX_DB_USER", ""), password=os.environ.get("JAX_DB_PASSWORD", ""),
        db=database, autocommit=True, minsize=1, maxsize=4, cursorclass=aiomysql.DictCursor,
        connect_timeout=db_connect_timeout_seconds())
    try:
        if args.revertir:
            return await revertir(pool, args.revertir)
        if args.aplicar:
            return await aplicar(pool, args)
        return await ensayar(pool, args)
    finally:
        pool.close()
        await pool.wait_closed()


class _CodigoDeSalida(Exception):
    def __init__(self, codigo: int) -> None:
        super().__init__(codigo)
        self.codigo = codigo


def main(argv: list[str] | None = None) -> int:
    try:
        return _main(argv)
    except _CodigoDeSalida as salida:
        return salida.codigo


def _main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    database = args.database or os.environ.get("JAX_DB_NAME", "")
    if not database:
        print("falta la base: use --database o JAX_DB_NAME", file=sys.stderr)
        return 2
    if args.revertir:
        if args.aplicar:
            print("--revertir y --aplicar son excluyentes", file=sys.stderr)
            return 2
    else:
        faltan = [n for n, v in (("--workspace", args.workspace), ("--carpeta", args.carpeta),
                                 ("--nombre", args.nombre), ("--dueno-user-id", args.dueno_user_id)) if v is None]
        if faltan:
            print(f"faltan argumentos: {', '.join(faltan)}", file=sys.stderr)
            return 2
    if (args.aplicar or args.revertir) and database == BASE_PRODUCCION and not args.confirmo_produccion:
        modo = "--aplicar" if args.aplicar else "--revertir"
        print(f"{modo} sobre {BASE_PRODUCCION} (produccion) exige --confirmo-produccion", file=sys.stderr)
        return 2
    try:
        resultado = asyncio.run(_correr(args, database))
    except GuardaFallida as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    except ShaNoCuadra as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 3
    except MapaNoCuadra as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 6
    except EstadoImpide as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    except Exception as e:                 # noqa: BLE001 - cualquier otra cosa NO es «volver a correr»
        print(f"ERROR no previsto ({type(e).__name__}: {e}); NO reintentar sin revisar: mirar el disco y la "
              f"base antes de cualquier otra accion", file=sys.stderr)
        raise _CodigoDeSalida(5) from e
    print(json.dumps(resultado, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
