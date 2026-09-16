"""El respaldo en disco de las filas de uso que no pudieron entrar a
`axioma_usage` -- COPIA del repo `jax` (plan 2026-09-15-cola-durable-uso,
Task 7). El canonico es `jax-platform/backend/uso/cola.py`.

**Este modulo es el contrato entre los repos `jax-platform` y `jax`.** Se copia
verbatim: todo lo que no sea este docstring tiene que ser identico al canonico,
y `scripts/check_mirror_sync.py` (familia `cola_uso`) lo compara por AST en CI,
igual que `redaccion.py` y `db_connect_config.py`. Por eso el modulo no depende
de nada fuera de la biblioteca estandar: ni la base, ni el resto del repo.

El respaldo es un DIRECTORIO con un archivo por fila, no un archivo unico con
todas las lineas. Hay TRES procesos que depositan -- `jax-platform`
(`api/admin/usage.py::record_usage`), `jax/jacobs/usage_writer.py` y
`jax/las_manos/motor_registry/usage_writer.py` -- y uno solo que drena e
inserta: la plataforma, que es la duena de la tabla y la unica con migraciones.
Con un archivo unico, el que drena tendria que reescribirlo sin las filas ya
insertadas, y esa reescritura pisaria lo que otro proceso agrego entremedio: se
perderian filas justo en el mecanismo que existe para no perderlas. Con un
archivo por fila no hay reescritura -- se crea con temporal + `os.replace`
(atomico entre procesos) y el que drena lo borra recien despues de que la fila
entro.

**ESTE REPO NO DRENA.** De la API publica, los dos escritores de `jax`
(`jacobs/usage_writer.py` y `las_manos/motor_registry/usage_writer.py`) llaman
UNICAMENTE a `encolar`. `leer_pendientes`, `quitar` y `drenar` son de la
plataforma: es la duena de la tabla `axioma_usage` y la unica con migraciones
(la columna `spool_id` y su indice UNIQUE, que es lo que hace idempotente al
reintento). Se copian igual, sin usarlas, porque la copia es literal y la
comparacion por AST cubre el archivo entero: una copia recortada se driftea sin
que nadie lo vea. Si algun dia `jax` insertara por su cuenta, dos procesos
borrarian el mismo archivo sin coordinacion y el cobro se duplicaria en la
ventana entre el INSERT y el borrado.

La atomicidad ENTRE procesos la da `os.replace`, no el `asyncio.Lock` de
modulo: ese lock solo serializa las corrutinas de ESTE proceso. Todo el I/O de
disco sale del event loop por `asyncio.to_thread`, porque `encolar` se llama en
el camino del usuario.

Y por eso mismo `encolar` NO escribe bajo `_lock` (Task 11, 2026-09-16): como
la exclusion entre escrituras ya la da el nombre unico (`<uuid4>.json`) mas el
`os.replace`, el lock alrededor de los dos `fsync` no protegia nada y si
serializaba cada turno detras de los demas -- 111 ms a c=25 con la cola VACIA,
espera pura. `_lock` queda para lo unico que necesita exclusion: el tope
(contar y descartar la mas vieja, atomicos entre si), que es el camino raro.
Lo que se paga a cambio esta escrito en el docstring de `encolar`.
"""
import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# --- configuración (sin hardcoding en el llamador) --------------------------
VARIABLE_DIRECTORIO = "JAX_USAGE_SPOOL_DIR"
DIRECTORIO_POR_DEFECTO = Path("/srv/jax-data/usage-spool")
VARIABLE_MAX_FILAS = "JAX_USAGE_SPOOL_MAX_FILAS"
MAX_FILAS_POR_DEFECTO = 50000

SUFIJO = ".json"
SUFIJO_TEMPORAL = ".tmp"
SUBDIRECTORIO_CORRUPTOS = "corruptos"

# --- el formato compartido (el contrato entre los dos repos) ----------------
CAMPOS = (
    "spool_id", "created_at", "tenant_id", "user_id", "facet", "model",
    "tokens_in", "tokens_out", "cost_usd", "request_type", "origen",
    "status", "job_id",
)
#: `spool_id` y `created_at` los completa el módulo si el llamador no los trae;
#: `status` y `job_id` son opcionales PARA EL LLAMADOR (sólo `motor_registry`
#: los tiene) y `_normalizar` los deja en `None`. El resto son obligatorios.
#: `created_at` es la hora del TURNO, no la del reintento: si no, una caída de
#: dos horas movería el costo al día siguiente.
#: `status`/`job_id` están en el archivo (DECISIÓN de Fernando 2026-09-15,
#: opción (b)) porque sin ellos la fila recuperada entra a `axioma_usage` con
#: esas dos columnas en NULL y la reconciliación contra `motor_jobs.jsonl` no
#: la puede emparejar: se recupera el cobro y se pierde la trazabilidad.
CAMPOS_OBLIGATORIOS = tuple(
    c for c in CAMPOS
    if c not in ("spool_id", "created_at", "status", "job_id")
)
ORIGENES = frozenset({"platform", "jacobs", "motor_registry"})

_lock = asyncio.Lock()
#: Reloj lógico de las mediciones de la profundidad (Task 11). `encolar` mide
#: el disco y publica DESPUÉS de escribir el archivo, y esa escritura -- con
#: sus dos `fsync`, ~4,4 ms -- ya no corre bajo `_lock`: dos llamadas
#: concurrentes pueden TERMINAR en orden distinto al que MIDIERON. Sin el
#: reloj, la que midió primero y terminó última publicaría un número viejo y
#: `en_cola` retrocedería -- y eso se mira en Admin -> Costos justo durante una
#: caída, donde ver la cola bajar cuando en realidad sube es peor que no verla.
#: `asignada` la reserva el que acaba de mirar el disco; `publicada` es la de
#: la medición que hoy está en `_estado["en_cola"]`.
_medicion = {"asignada": 0, "publicada": 0}
_estado = {
    "en_cola": 0,
    "perdidas_por_desborde": 0,
    "corruptos": 0,
    "ultimo_error_de_respaldo": None,
}


# --- estado observable ------------------------------------------------------

def estadisticas() -> dict:
    """Lo que mira Admin → Costos. `en_cola` es la última profundidad medida
    del disco (no un contador en memoria): los otros dos procesos depositan sin
    pasar por acá."""
    return dict(_estado)


def reset_estado() -> None:
    """Vuelve los contadores a cero. Para los tests y para un arranque limpio."""
    _estado.update({
        "en_cola": 0,
        "perdidas_por_desborde": 0,
        "corruptos": 0,
        "ultimo_error_de_respaldo": None,
    })
    # El cero se publica como la medición MÁS NUEVA: si no, un `encolar` que
    # quedó en vuelo desde antes del reset resucitaría su profundidad vieja.
    _publicar_profundidad(0, _nueva_medicion())


def _anotar_error(mensaje: str) -> None:
    _estado["ultimo_error_de_respaldo"] = mensaje
    logger.warning("respaldo de uso: %s", mensaje)


def _nueva_medicion() -> int:
    """Reserva el número de orden de una medición de la profundidad.

    Se pide APENAS TERMINA de mirarse el disco, no antes: lo que ordena las
    mediciones es cuándo se miró, no cuándo se decidió mirar. Entre medir y
    reservar no hay ningún `await`, así que el par es atómico para el bucle de
    eventos.
    """
    _medicion["asignada"] += 1
    return _medicion["asignada"]


def _publicar_profundidad(cantidad: int, secuencia: int) -> None:
    """Deja `cantidad` en `_estado["en_cola"]`, salvo que ya se haya publicado
    una medición más nueva. Ver `_medicion`."""
    if secuencia < _medicion["publicada"]:
        return
    _medicion["publicada"] = secuencia
    _estado["en_cola"] = cantidad


# --- configuración ----------------------------------------------------------

def _ruta_configurada() -> Path:
    """La ruta del respaldo SIN crearla. Preguntar no crea nada: el default es
    una ruta de producción."""
    crudo = os.environ.get(VARIABLE_DIRECTORIO)
    return Path(crudo) if crudo else DIRECTORIO_POR_DEFECTO


def directorio_del_respaldo() -> Path:
    """La ruta del respaldo, creada si falta."""
    ruta = _ruta_configurada()
    ruta.mkdir(parents=True, exist_ok=True)
    return ruta


def max_filas() -> int:
    """Cota dura de filas pendientes. Una caída larga no puede llenar el disco."""
    crudo = os.environ.get(VARIABLE_MAX_FILAS)
    if not crudo:
        return MAX_FILAS_POR_DEFECTO
    try:
        valor = int(crudo)
    except (TypeError, ValueError):
        # fail-soft: un tope mal escrito en el entorno no puede tirar el
        # registro de uso; se usa el default y queda el aviso.
        logger.warning(
            "%s no es un número (%r); se usa el default %d",
            VARIABLE_MAX_FILAS, crudo, MAX_FILAS_POR_DEFECTO,
        )
        return MAX_FILAS_POR_DEFECTO
    return valor if valor > 0 else MAX_FILAS_POR_DEFECTO


# --- validación -------------------------------------------------------------

def _id_seguro(spool_id) -> bool:
    """Un `spool_id` es un nombre de archivo, no una ruta: sin separadores y
    sin `..`, para que `quitar` no pueda salirse del directorio."""
    if not isinstance(spool_id, str) or not spool_id:
        return False
    if spool_id in (".", ".."):
        return False
    if "/" in spool_id or os.sep in spool_id:
        return False
    if os.altsep and os.altsep in spool_id:
        return False
    return spool_id == os.path.basename(spool_id)


def _ahora_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalizar(fila) -> dict:
    """Devuelve la fila con los trece campos del contrato, o lanza ValueError.

    Los que el llamador no trae quedan en `None` por la comprensión sobre
    `CAMPOS` de más abajo: hoy eso alcanza para `status` y `job_id`, que sólo
    tiene `motor_registry`.
    """
    if not isinstance(fila, dict):
        raise ValueError(f"la fila no es un diccionario: {type(fila).__name__}")
    faltantes = [c for c in CAMPOS_OBLIGATORIOS if c not in fila]
    if faltantes:
        raise ValueError(f"faltan campos del contrato: {', '.join(faltantes)}")
    if fila["origen"] not in ORIGENES:
        raise ValueError(f"origen desconocido: {fila['origen']!r}")

    spool_id = fila.get("spool_id") or str(uuid.uuid4())
    if not _id_seguro(spool_id):
        raise ValueError(f"spool_id inválido: {spool_id!r}")

    normalizada = {c: fila.get(c) for c in CAMPOS}
    normalizada["spool_id"] = spool_id
    normalizada["created_at"] = fila.get("created_at") or _ahora_iso()
    return normalizada


def _motivo_de_corrupcion(datos) -> str | None:
    if not isinstance(datos, dict):
        return f"el contenido no es un objeto JSON ({type(datos).__name__})"
    # Se exigen los TRECE, no sólo los obligatorios del llamador, y es a
    # propósito: fail-closed. Un archivo escrito por una copia VIEJA del módulo
    # (once campos, sin `status`/`job_id`) cae en `corruptos/` en vez de entrar
    # a medias -- entraría con esas dos columnas en NULL y sin manera de saber,
    # después, que faltaban. El que encola completa el archivo; el que drena
    # exige el archivo completo.
    faltantes = [c for c in CAMPOS if c not in datos]
    if faltantes:
        return f"faltan campos del contrato: {', '.join(faltantes)}"
    if not _id_seguro(datos.get("spool_id")):
        return f"spool_id inválido: {datos.get('spool_id')!r}"
    return None


# --- I/O sincrónico (siempre detrás de asyncio.to_thread) -------------------

def _listar(directorio: Path) -> list:
    """`[(mtime, Path)]` de las filas pendientes. Sólo el primer nivel: lo que
    está en `corruptos/` ya no es pendiente. Propaga OSError."""
    salida = []
    with os.scandir(directorio) as entradas:
        for entrada in entradas:
            nombre = entrada.name
            if nombre.startswith(".") or not nombre.endswith(SUFIJO):
                continue
            try:
                if not entrada.is_file():
                    continue
                salida.append((entrada.stat().st_mtime, Path(entrada.path)))
            except OSError:
                # fail-soft: la fila desapareció entre el scandir y el stat
                # (otro proceso la drenó); no es un error, es la carrera normal.
                continue
    salida.sort(key=lambda par: (par[0], par[1].name))
    return salida


def _contar_barato(directorio: Path) -> int:
    """Cuántas filas pendientes hay, en UN recorrido y sin `stat()` por archivo.

    `_listar` cuesta un `stat()` por entrada más un `sort`: eso se paga sólo
    cuando hay que elegir la MÁS VIEJA para descartar (`_hacer_lugar`), que es
    el 0,01% de las llamadas. Para saber la profundidad alcanza con contar, y
    `entrada.is_file()` sale del `d_type` que ya vino en el `getdents` del
    `scandir`: no es un syscall más. Propaga OSError, como `_listar`.

    Cuenta lo mismo que `_listar`: primer nivel, con `SUFIJO` y sin ocultos --
    lo que está en `corruptos/` ya no es pendiente y el temporal de
    `_escribir_atomico` empieza con punto y no termina en `SUFIJO`.
    """
    cantidad = 0
    with os.scandir(directorio) as entradas:
        for entrada in entradas:
            nombre = entrada.name
            if nombre.startswith(".") or not nombre.endswith(SUFIJO):
                continue
            try:
                if not entrada.is_file():
                    continue
            except OSError:
                # fail-soft: la fila desapareció entre el `getdents` y el
                # `is_file` (otro proceso la drenó); es la carrera normal.
                continue
            cantidad += 1
    return cantidad


def _contar(directorio: Path) -> int:
    try:
        return _contar_barato(directorio)
    except OSError as error:
        # fail-soft: la profundidad es un dato de tablero; si el directorio no
        # se puede leer queda el aviso y el último valor conocido no miente
        # más que un cero inventado.
        _anotar_error(f"no se pudo contar el respaldo: {error}")
        return _estado["en_cola"]


def _escribir_atomico(directorio: Path, spool_id: str, fila: dict) -> None:
    """Temporal en el MISMO directorio, `fsync` del archivo, `os.replace` y
    `fsync` del directorio.

    El temporal tiene que estar en el mismo directorio o `replace` cruzaría
    sistemas de archivos y dejaría de ser atómico. El `fsync` del directorio es
    el que hace sobrevivir el rename a un corte de luz: sin él, el archivo
    existe y la entrada de directorio puede no estar.
    """
    temporal = directorio / f".{spool_id}{SUFIJO_TEMPORAL}"
    definitivo = directorio / f"{spool_id}{SUFIJO}"
    datos = json.dumps(fila, ensure_ascii=False).encode("utf-8")

    descriptor = os.open(temporal, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(descriptor, datos)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

    try:
        os.replace(temporal, definitivo)
    except BaseException:
        try:
            os.unlink(temporal)
        except OSError:  # fail-soft: limpieza del temporal; el error que importa es el de `replace`, que se relanza abajo
            pass
        raise

    descriptor = os.open(directorio, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _hacer_lugar(directorio: Path) -> int:
    """Deja lugar para una fila más: pasado el tope se descarta la MÁS VIEJA y
    se cuenta aparte. La pérdida sigue siendo visible. Devuelve cuántas
    descartó, para que el llamador ajuste la profundidad que ya midió sin
    recorrer el directorio otra vez.

    Acá SÍ se paga el `stat()` por archivo y el `sort` de `_listar`: elegir la
    más vieja necesita las fechas. Por eso `encolar` sólo llama a esta función
    cuando el conteo barato dice que se llegó al tope.
    """
    descartadas = 0
    try:
        entradas = _listar(directorio)
    except OSError as error:
        _anotar_error(f"no se pudo revisar el tope del respaldo: {error}")
        return descartadas
    sobrantes = len(entradas) - max_filas() + 1
    if sobrantes <= 0:
        return descartadas
    for _mtime, ruta in entradas[:sobrantes]:
        try:
            ruta.unlink()
        except FileNotFoundError:
            continue
        except OSError as error:
            _anotar_error(f"no se pudo descartar {ruta.name} por desborde: {error}")
            continue
        descartadas += 1
        _estado["perdidas_por_desborde"] += 1
        logger.warning(
            "respaldo de uso lleno (tope %d): se descarta la fila más vieja %s",
            max_filas(), ruta.name,
        )
    return descartadas


def _cuarentena(directorio: Path, ruta: Path, motivo: str) -> None:
    """Un archivo corrupto se saltea, se cuenta, se loguea UNA vez y se mueve a
    `corruptos/`: si se quedara, taparía el paso en cada ciclo. Se mueve, no se
    borra -- el dato no se tira."""
    _estado["corruptos"] += 1
    logger.warning(
        "registro de uso corrupto en el respaldo: %s (%s); se mueve a %s/",
        ruta.name, motivo, SUBDIRECTORIO_CORRUPTOS,
    )
    destino = directorio / SUBDIRECTORIO_CORRUPTOS
    try:
        destino.mkdir(parents=True, exist_ok=True)
        os.replace(ruta, destino / ruta.name)
    except OSError as error:
        # fail-soft: si no se puede mover, la fila corrupta se vuelve a ver en
        # el ciclo siguiente; queda el aviso y la cola sigue drenando el resto.
        _anotar_error(f"no se pudo mover {ruta.name} a {SUBDIRECTORIO_CORRUPTOS}/: {error}")


def _leer_lote(directorio: Path, entradas: list, limite: int) -> list:
    filas = []
    for _mtime, ruta in entradas:
        if len(filas) >= limite:
            break
        try:
            datos = json.loads(ruta.read_text(encoding="utf-8"))
        except FileNotFoundError:
            # otro ciclo (u otro proceso) ya la drenó: no es corrupción.
            continue
        except (OSError, ValueError) as error:
            _cuarentena(directorio, ruta, f"no se pudo leer: {error}")
            continue
        motivo = _motivo_de_corrupcion(datos)
        if motivo:
            _cuarentena(directorio, ruta, motivo)
            continue
        # el NOMBRE del archivo manda sobre el contenido: es lo que `quitar`
        # usa para borrarlo y lo que la columna UNIQUE va a guardar.
        datos["spool_id"] = ruta.name[:-len(SUFIJO)]
        filas.append(datos)
    return filas


def _borrar(directorio: Path, spool_ids: list) -> int:
    quitadas = 0
    for spool_id in spool_ids:
        try:
            (directorio / f"{spool_id}{SUFIJO}").unlink()
        except FileNotFoundError:
            # que ya no esté NO es error: otro ciclo pudo ganarle.
            continue
        except OSError as error:
            _anotar_error(f"no se pudo quitar {spool_id} del respaldo: {error}")
            continue
        quitadas += 1
    return quitadas


# --- API pública ------------------------------------------------------------

async def encolar(fila: dict) -> str | None:
    """Deja una fila de uso en el respaldo. Devuelve su `spool_id`, o None si
    no se pudo.

    NUNCA propaga: se llama desde el `except` de `record_usage`, en el camino
    del usuario, con el turno ya cobrado al proveedor y ya respondido. Un 500
    acá no recupera nada y sí le quita la respuesta.

    EL TOPE ES EXACTO EN EL USO SECUENCIAL. Bajo concurrencia se tolera un
    sobrepaso TRANSITORIO de a lo sumo `(llamadas en vuelo en este proceso) - 1`
    filas: mientras una cuenta y descarta, las otras escriben sin contar, así
    que el desborde lo ve una sola. Cuánto: con c=25 son 24 filas sobre 50.000
    (0,048%, ~7 KB). Por qué se acepta: `_hacer_lugar` deja SIEMPRE el
    directorio en `tope - 1` porque re-lista el disco real, así que la próxima
    medición devuelve la cola al tope -- el estado de régimen queda acotado a
    `[tope - 1, tope + c - 1]` y no deriva. El error es sólo hacia arriba:
    nunca se descarta de más. La alternativa era serializar la escritura o el
    conteo, y las dos cuestan cientos de milisegundos por turno del usuario
    para ahorrar 7 KB durante unos milisegundos.
    """
    try:
        normalizada = _normalizar(fila)
    except Exception as error:  # fail-soft: una fila mal formada no puede tirar el turno del usuario
        _anotar_error(f"fila de uso inválida, no se encola: {error}")
        return None

    try:
        directorio = await asyncio.to_thread(directorio_del_respaldo)
        # UN solo recorrido del directorio, y barato. Antes eran dos enteros
        # con `stat()` por archivo y `sort` (uno para el tope y otro para la
        # profundidad), los dos bajo `_lock`, que es de todo el proceso: el
        # turno k de la caída pagaba O(k) -- costo cuadrático sobre la caída, y
        # realimentado (cola más profunda, turnos más lentos). Medido a c=25 el
        # 2026-09-15: 122 ms vacío, 1.250 ms con 10.000 filas, 6.834 ms en el
        # tope.
        #
        # UNA medición a la vez, y el que llega con otra en curso NO la repite
        # (Task 11). El impulso es sacar también este recorrido del lock, y
        # medido es PEOR: `_contar_barato` es un bucle de Python sobre todas las
        # entradas, con el GIL tomado casi todo el tiempo, y 25 a la vez sobre
        # el mismo directorio no cuestan 25 veces uno -- cuestan 230 veces uno.
        # Medido el 2026-09-16 sobre XFS/NVMe con el cache caliente: 49.000
        # entradas, 7,98 ms un hilo solo contra 1.833 ms 25 hilos a la vez
        # (10.000 entradas: 1,26 ms contra 372 ms). A c=25 eso dio 2.039 ms por
        # turno con 49.000 filas, contra los 291 ms de la versión con el lock
        # grande: un 7x PEOR. El trabajo no hay que paralelizarlo, hay que no
        # repetirlo.
        #
        # El que se saltea el conteo no publica profundidad ni revisa el tope:
        # los hace el que sí está contando, milisegundos después. `en_cola`
        # sigue saliendo SIEMPRE de un recorrido real del disco -- nunca de un
        # contador en memoria -- porque `jacobs` y `motor_registry` depositan sin
        # pasar por este proceso.
        cantidad = None
        secuencia = None
        if not _lock.locked():  # entre el chequeo y el `acquire` no hay `await`
            # ACÁ el lock, y sólo acá: contar y descartar la más vieja tienen
            # que ser atómicos ENTRE SÍ o dos llamadas concurrentes descartan de
            # más. `_hacer_lugar` re-lista adentro (fechas + orden), así que el
            # descarte se decide sobre el disco real y NUNCA descarta de más; el
            # conteo barato es sólo el disparador. El descarte es el camino
            # raro: en una caída normal no se llega al tope.
            async with _lock:
                try:
                    cantidad = await asyncio.to_thread(_contar_barato, directorio)
                except OSError as error:  # fail-soft: si no se puede recorrer el directorio, igual se intenta escribir la fila -- perderla sería peor
                    # La profundidad queda en el último valor conocido, que no
                    # miente más que un cero inventado, y queda el aviso.
                    _anotar_error(f"no se pudo revisar el tope del respaldo: {error}")
                if cantidad is not None and cantidad + 1 > max_filas():
                    cantidad -= await asyncio.to_thread(_hacer_lugar, directorio)
                secuencia = _nueva_medicion()
        # FUERA de toda sección serializada: acá están los dos `fsync` (~4,4 ms),
        # que con el lock de todo el proceso alrededor eran el piso ENTERO de
        # 111 ms medido a c=25 con la cola VACÍA -- espera pura, no trabajo: con
        # la cola vacía no hay nada que contar. El lock no las protegía de nada:
        # cada fila va a su propio `<spool_id>.json` con `spool_id` uuid4, por
        # temporal OCULTO + `os.replace`, y dos escrituras concurrentes no
        # comparten ni el nombre definitivo ni el temporal.
        await asyncio.to_thread(
            _escribir_atomico, directorio, normalizada["spool_id"], normalizada
        )
        if cantidad is not None:
            # Sale del DISCO igual que antes -- es el conteo que se acaba de
            # hacer, más la fila que se acaba de escribir -- y no de un contador
            # recordado entre llamadas: `jacobs` y `motor_registry` depositan sin
            # pasar por este proceso. Va por `_publicar_profundidad` para que una
            # llamada que midió antes y terminó después no haga retroceder el
            # número (ver `_medicion`).
            _publicar_profundidad(cantidad + 1, secuencia)
        return normalizada["spool_id"]
    except Exception as error:  # fail-soft: un fallo de disco no puede tirar el turno del usuario
        _anotar_error(f"no se pudo encolar la fila de uso: {error}")
        return None


async def leer_pendientes(limite: int) -> list:
    """Las `limite` filas pendientes más viejas, cada una con su `spool_id`.

    Las viejas primero porque el reintento tiene que recuperar el gasto en el
    orden en que ocurrió. Un directorio que no existe es una cola vacía; uno
    que no se puede leer es un error visible -- silenciarlo escondería una cola
    que crece sin drenarse.
    """
    async with _lock:
        directorio = _ruta_configurada()
        try:
            entradas = await asyncio.to_thread(_listar, directorio)
        except FileNotFoundError:
            _publicar_profundidad(0, _nueva_medicion())
            return []
        except OSError as error:
            _anotar_error(f"no se pudo leer el respaldo: {error}")
            raise
        filas = await asyncio.to_thread(_leer_lote, directorio, entradas, limite)
        _publicar_profundidad(
            await asyncio.to_thread(_contar, directorio), _nueva_medicion())
        return filas


async def quitar(spool_ids) -> int:
    """Borra del respaldo las filas que ya entraron a la base. Devuelve cuántas
    se borraron de verdad; que un archivo ya no esté NO es error."""
    seguros = []
    for spool_id in spool_ids or ():
        if _id_seguro(spool_id):
            seguros.append(spool_id)
        else:
            _anotar_error(f"spool_id inválido, no se quita: {spool_id!r}")
    if not seguros:
        return 0

    async with _lock:
        directorio = _ruta_configurada()
        quitadas = await asyncio.to_thread(_borrar, directorio, seguros)
        _publicar_profundidad(
            await asyncio.to_thread(_contar, directorio), _nueva_medicion())
        return quitadas


async def contar_pendientes() -> int:
    """La profundidad de la cola, medida del disco. No de un contador en
    memoria: `jacobs` y `motor_registry` depositan en el mismo directorio sin
    pasar por este proceso."""
    async with _lock:
        directorio = _ruta_configurada()
        try:
            cantidad = await asyncio.to_thread(_contar_barato, directorio)
        except FileNotFoundError:
            cantidad = 0
        except OSError as error:
            _anotar_error(f"no se pudo contar el respaldo: {error}")
            raise
        _publicar_profundidad(cantidad, _nueva_medicion())
        return cantidad
