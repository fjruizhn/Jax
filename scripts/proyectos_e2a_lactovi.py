#!/usr/bin/env python3
"""Proyectos E2a: convierte la carpeta suelta de LACTOVI en un proyecto real.

Toma `<workspace>/proyectos/<carpeta>/` (con `fuente/` y `procesado/<sha256>/ficha.json`),
crea el proyecto con `ProjectAuthorityAdmin.create_project`, renombra la carpeta a
`proyectos/<project_uuid>/` y registra sus documentos en `project_documents`.

Uso:  python -m scripts.proyectos_e2a_lactovi --workspace RUTA --carpeta NOMBRE
          --nombre "Lácteos Victoria" --dueno-user-id N [--tenant-id 1]
          [--aplicar] [--database NOMBRE] [--confirmo-produccion]
      python -m scripts.proyectos_e2a_lactovi {--revertir|--completar} MAPA.json
          [--database NOMBRE] [--confirmo-produccion]

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
  - sin ficha y que `procesamiento.compuerta.tiene_extractor` acepta (por extension o por
    contenido, la misma decision que `extraer`): `en_cola` con `ruta_entrada` =
    `proyectos/<uuid>/fuente/<ruta>` (relativa al workspace) para que el despachador de la
    plataforma la procese; la ingesta ve que el archivo ya esta en fuente/ y lo procesa en
    el lugar, sin copiarlo;
    REGLA: esa `ruta_entrada` bajo `fuente/` es el ORIGINAL que trajo LACTOVI. Nadie la borra
    nunca (ni el despachador al terminar el trabajo): solo se borran las de
    `proyectos/<uuid>/entrada/`;
  - sin ficha y sin extractor: `sin_extractor` (ruta_entrada NULL).
Todo lo que cuelga de la carpeta fuera de `fuente/` y `procesado/` (p. ej.
`.claude-flow/`) se ignora para el registro (viaja con el rename) y se menciona.

`--revertir MAPA`: en una transaccion borra las filas de `project_documents` del
proyecto, devuelve la carpeta a su nombre verificando los sha256 del mapa y deja el
proyecto ARCHIVADO (Destruir no existe). Solo vale mientras `fuente/` no haya
cambiado: si la plataforma ya recibio subidas, los sha256 no cuadran (codigo 3) y no
se toca nada. Se NIEGA (codigo 7) si alguna fila esta `pendiente` o `procesando`: hay
trabajos en vuelo y la ingesta recrearia `fuente/` tras devolver la carpeta (el runbook
manda detener jax-platform antes). Reintentable: con la carpeta ya devuelta, solo
termina de borrar filas y archivar.

`--completar MAPA`: para un corte DESPUES del rename y ANTES de registrar las filas (o con
commit incierto). Con el proyecto ACTIVE y la carpeta ya en `<uuid>`, verifica los sha256
contra el mapa y registra las filas que falten (INSERT IGNORE). Idempotente.

Codigos de salida: 0 hecho; 1 estado que impide seguir (mapa o destino ya existen,
proyecto no ACTIVE, fallo al registrar: se deshizo el rename); 2 argumentos, guarda
o carpeta que no existe (p. ej. ya se movio); 3 los sha256 no cuadran (se deshizo el
rename); 4 el COMMIT de las filas tiene desenlace desconocido (NO se toco el disco:
verificar las filas y, si faltan, correr --completar); 5 error no previsto (no reintentar
sin revisar); 6 (--revertir/--completar) el mapa no cuadra con la base; 7 (--revertir)
hay trabajos en vuelo.
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
from jax.core.project_tree_lock import project_tree_lock  # noqa: E402
from procesamiento.compuerta import tiene_extractor  # noqa: E402
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


class CommitIncierto(RuntimeError):
    """El commit de las filas fallo con desenlace desconocido: el disco NO se toca (codigo 4)."""


class TrabajosEnVuelo(RuntimeError):
    """Hay filas pendiente/procesando: no se revierte (codigo 7)."""


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


def _no_regulares(fuente: Path) -> list[str]:
    """Symlinks y no-regulares (a CUALQUIER nivel) que el guion no hashea ni registra."""
    out: list[str] = []
    for raiz, dirs, archivos in os.walk(fuente, followlinks=False):
        for nombre in [*dirs, *archivos]:
            p = Path(raiz) / nombre
            if p.is_symlink() or (not p.is_dir() and not p.is_file()):
                out.append(p.relative_to(fuente).as_posix())
    return sorted(out)


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
        "hashes": hashes, "fichas": fichas, "no_regulares": _no_regulares(fuente), "primero": primero, "ignorado": ignorado,
        "duplicados_sha": len(hashes) - len(primero),
        "fichas_sin_archivo": sorted(set(fichas) - set(primero)),
        "bytes": {sha: (fuente / rel).stat().st_size for sha, rel in primero.items()},
        # La misma decision que el extractor real (extension Y contenido), no una lista copiada.
        "extraible": {sha: tiene_extractor(fuente / rel) for sha, rel in primero.items()},
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
        elif plan["extraible"][sha]:
            fila.update(estado="en_cola", carpeta_procesado=None,
                        ruta_entrada=f"proyectos/{project_uuid}/fuente/{rel}")
        else:
            fila.update(estado="sin_extractor", carpeta_procesado=None, ruta_entrada=None)
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


def _mover(origen: Path, destino: Path, workspace_root: Path) -> None:
    """rename sin pisar: `os.rename` reemplazaria un directorio destino vacio."""
    with project_tree_lock(workspace_root):
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
            "ignorado": plan["ignorado"], "ignorados": plan["no_regulares"]}


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
        raise EstadoImpide(f"el mapa de reversion ya existe: {mapa} (si es de una corrida anterior, "
                           f"conservarlo renombrandolo antes de reintentar; nunca se pisa)")
    estados: dict[str, int] = {}
    for f in _filas(plan, "<uuid>", args.dueno_user_id):
        estados[f["estado"]] = estados.get(f["estado"], 0) + 1
    return {"ensayo": True, **_resumen(plan), "filas_a_insertar": len(plan["primero"]),
            "mapa_previsto": str(mapa), "carpeta_actual": str(base), "estados": dict(sorted(estados.items()))}


async def _registrar(pool, project_id: int, filas: list[dict]) -> int:
    """Las filas en UNA transaccion. Un fallo ANTES del commit hace rollback y se relanza tal cual
    (quien llama decide). Un fallo DEL commit tiene desenlace desconocido: CommitIncierto."""
    async with pool.acquire() as conn:
        await conn.begin()
        try:
            insertadas = await _insertar(conn, project_id, filas)
        except BaseException:
            await conn.rollback()
            raise
        try:
            await conn.commit()
        except BaseException as e:
            try:
                conn.close()
            except Exception:  # fail-soft: cierre de una conexion ya dudosa; se reporta CommitIncierto abajo
                pass
            raise CommitIncierto(
                f"el commit de las filas tiene desenlace desconocido ({type(e).__name__}: {e}); NO se toco el "
                f"disco: verifica las filas de project_documents del proyecto {project_id} antes de tocar nada; "
                f"si faltan, correr --completar con el mapa") from e
    return insertadas


async def aplicar(pool, args) -> dict:
    proyectos = Path(args.workspace) / "proyectos"
    base = _validar_carpeta(proyectos, args.carpeta)
    if not await _dueno_activo(pool, args.dueno_user_id, args.tenant_id):
        raise GuardaFallida(f"el usuario {args.dueno_user_id} no esta activo en el tenant {args.tenant_id}")
    mapa = _ruta_mapa(proyectos)
    if os.path.lexists(mapa):
        raise EstadoImpide(f"el mapa de reversion ya existe: {mapa} (si es de una corrida anterior, "
                           f"conservarlo renombrandolo antes de reintentar; nunca se pisa)")
    plan = _planear(base)                                                   # 1. sha256 (antes)
    admin = ProjectAuthorityAdmin(MariaDBB9Store(pool))
    creado = await admin.create_project(                                     # 2. proyecto
        _req(args.dueno_user_id, args.tenant_id, "CREATE_PROJECT", None), name=args.nombre,
        description=None, idempotency_key=_llave(base))
    estado = await _estado_alcance(pool, creado.project_id)
    if estado != "ACTIVE":
        raise EstadoImpide(
            f"el proyecto {creado.project_id} de esta carpeta ya existia y esta {estado}, no ACTIVE; no se movio "
            f"nada. Si vino de un --revertir, quedo archivado a proposito (Destruir no existe): restaurarlo a ACTIVE "
            f"(RESTORE_PROJECT, ARCHIVED -> ACTIVE) desde la plataforma o con set_project_lifecycle y volver a "
            f"correr --aplicar")
    nueva = proyectos / creado.project_uuid
    _escribir_mapa(mapa, {                                                   # 3. mapa, antes de mover
        "version": 1, "project_id": creado.project_id, "project_uuid": creado.project_uuid,
        "tenant_id": args.tenant_id, "dueno_user_id": args.dueno_user_id, "nombre": args.nombre,
        "ruta_vieja": str(base), "ruta_nueva": str(nueva), "sha256_antes": plan["hashes"]})
    _mover(base, nueva, proyectos.parent)                                    # 4. rename (si falla, nada se borra)
    try:
        despues = _hashear_fuente(nueva / "fuente")                          # 5. sha256 (despues)
        if despues != plan["hashes"]:
            diff = sorted(set(despues.items()) ^ set(plan["hashes"].items()))[:5]
            raise ShaNoCuadra(f"los sha256 despues de mover no son los de antes (primeras diferencias: {diff}); "
                              f"se deshizo el rename y la carpeta volvio a {base}")
        filas = _filas(plan, creado.project_uuid, args.dueno_user_id)
        insertadas = await _registrar(pool, creado.project_id, filas)        # 6. registro
    except CommitIncierto:
        raise                                                                # el disco NO se toca
    except ShaNoCuadra:
        _mover(nueva, base, proyectos.parent)
        raise
    except BaseException:
        _mover(nueva, base, proyectos.parent)
        print(f"ERROR al registrar documentos: se deshizo el rename (la carpeta volvio a {base}); el proyecto "
              f"{creado.project_id} sigue creado y ACTIVO, y el mapa {mapa} se conserva (borrarlo a mano antes de "
              f"reintentar --aplicar: no se pisa)", file=sys.stderr)
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


async def _verificar_mapa_contra_base(pool, mapa: dict) -> None:
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute("SELECT project_uuid FROM projects WHERE id=%s", (mapa["project_id"],))
            fila = await cur.fetchone()
    if not fila or str(fila["project_uuid"]) != mapa["project_uuid"]:
        raise MapaNoCuadra(f"el project_uuid del mapa ({mapa['project_uuid']}) no es el del proyecto "
                           f"{mapa['project_id']} en la base; no se toco nada")


async def completar(pool, ruta_mapa: str) -> dict:
    """Corte despues del rename y antes de las filas (o commit incierto): con el proyecto ACTIVE y la
    carpeta ya en <uuid>, verifica sha256 contra el mapa y registra las filas que falten."""
    mapa = cargar_mapa(ruta_mapa)
    pid, uid = mapa["project_id"], mapa["project_uuid"]
    nueva, vieja = Path(mapa["ruta_nueva"]), Path(mapa["ruta_vieja"])
    await _verificar_mapa_contra_base(pool, mapa)
    estado = await _estado_alcance(pool, pid)
    if estado != "ACTIVE":
        raise EstadoImpide(f"el proyecto {pid} esta {estado}, no ACTIVE; no se registro nada")
    if not nueva.is_dir():
        raise GuardaFallida(f"{nueva} no existe: la carpeta no esta en <uuid>, no hay nada que completar")
    if os.path.lexists(vieja):
        raise EstadoImpide(f"{vieja} existe junto a {nueva}: estado ambiguo, no se registro nada")
    plan = _planear(nueva)
    if plan["hashes"] != mapa["sha256_antes"]:
        raise ShaNoCuadra("los sha256 de fuente/ no son los del mapa; no se registro nada")
    filas = _filas(plan, uid, mapa["dueno_user_id"])
    insertadas = await _registrar(pool, pid, filas)
    return {"completado": True, "project_id": pid, "project_uuid": uid, **_resumen(plan),
            "filas_esperadas": len(filas), "filas_insertadas_ahora": insertadas}


async def revertir(pool, ruta_mapa: str) -> dict:
    mapa = cargar_mapa(ruta_mapa)
    pid, uid = mapa["project_id"], mapa["project_uuid"]
    nueva, vieja = Path(mapa["ruta_nueva"]), Path(mapa["ruta_vieja"])
    await _verificar_mapa_contra_base(pool, mapa)
    if nueva.is_dir():
        if os.path.lexists(vieja):
            raise EstadoImpide(f"{vieja} ya existe: no se puede devolver la carpeta")
        devuelta = False
    elif vieja.is_dir():
        devuelta = True            # reintento: la carpeta ya volvio; falta borrar filas y/o archivar
    else:
        raise GuardaFallida(f"ni {nueva} ni {vieja} existen: nada que revertir")
    async with pool.acquire() as conn:
        await conn.begin()
        movida = False
        try:
            async with conn.cursor() as cur:
                await cur.execute("SELECT COUNT(*) AS n FROM project_documents WHERE project_id=%s "
                                  "AND estado IN ('pendiente','procesando') FOR UPDATE", (pid,))
                en_vuelo = int((await cur.fetchone())["n"])
                if en_vuelo:
                    raise TrabajosEnVuelo(
                        f"hay {en_vuelo} documentos pendiente/procesando en el proyecto {pid}: la ingesta "
                        f"recrearia fuente/ tras devolver la carpeta. Detener jax-platform (el despachador vive "
                        f"ahi), esperar a que no quede nada en vuelo y volver a correr. No se toco nada")
                await cur.execute("DELETE FROM project_documents WHERE project_id=%s", (pid,))
                borradas = int(cur.rowcount)
            if not devuelta:
                _mover(nueva, vieja, vieja.parents[1])
                movida = True
            if _hashear_fuente(vieja / "fuente") != mapa["sha256_antes"]:
                raise ShaNoCuadra(f"los sha256 de fuente/ ya no son los del mapa (¿hubo subidas despues de "
                                  f"aplicar?); no se revirtio nada, la carpeta sigue en "
                                  f"{vieja if devuelta else nueva}")
        except BaseException:
            await conn.rollback()
            if movida:
                _mover(vieja, nueva, vieja.parents[1])
            raise
        try:
            await conn.commit()
        except BaseException as e:
            try:
                conn.close()
            except Exception:  # fail-soft: cierre de una conexion ya dudosa; se reporta CommitIncierto abajo
                pass
            raise CommitIncierto(
                f"el commit del borrado de filas tiene desenlace desconocido ({type(e).__name__}: {e}); NO se "
                f"toco el disco mas: verifica las filas del proyecto {pid} y la carpeta antes de repetir "
                f"--revertir (es reintentable)") from e
    admin = ProjectAuthorityAdmin(MariaDBB9Store(pool))
    cambio = await admin.set_project_lifecycle(
        _req(mapa["dueno_user_id"], mapa["tenant_id"], "SET_PROJECT_LIFECYCLE", pid), pid,
        ProjectLifecycle.ARCHIVED)
    return {"revertido": True, "project_id": pid, "project_uuid": uid, "filas_borradas": borradas,
            "carpeta": str(vieja), "proyecto_archivado": True, "archivado_ahora": bool(cambio),
            "reintento": devuelta}


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
    p.add_argument("--completar", metavar="MAPA.json", default=None,
                   help="registra las filas que faltan tras un corte posterior al rename")
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
        if args.completar:
            return await completar(pool, args.completar)
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
    if sum(bool(x) for x in (args.revertir, args.completar, args.aplicar)) > 1:
        print("--revertir, --completar y --aplicar son excluyentes", file=sys.stderr)
        return 2
    if not (args.revertir or args.completar):
        faltan = [n for n, v in (("--workspace", args.workspace), ("--carpeta", args.carpeta),
                                 ("--nombre", args.nombre), ("--dueno-user-id", args.dueno_user_id)) if v is None]
        if faltan:
            print(f"faltan argumentos: {', '.join(faltan)}", file=sys.stderr)
            return 2
    if (args.aplicar or args.revertir or args.completar) and database == BASE_PRODUCCION \
            and not args.confirmo_produccion:
        modo = "--aplicar" if args.aplicar else "--revertir" if args.revertir else "--completar"
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
    except CommitIncierto as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 4
    except TrabajosEnVuelo as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 7
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
