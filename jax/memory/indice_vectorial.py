"""Centinela del indice vectorial: detectar que MIENTE, y repararlo.

**El defecto que esto vigila** (medido en hall9000 el 2026-09-20, MariaDB
12.3.3). Un indice HNSW deja de devolver filas que SI estan en la tabla, y no
se recupera solo:

    300 filas -> borrar -> insertar 25 -> buscar las 10 mas cercanas
      por el indice ............ 1 de 10
      con IGNORE INDEX ......... 10 de 10
    insertar mas filas no lo arregla. `OPTIMIZE TABLE` tampoco.

**El disparador es el borrado EN CASCADA, no el volumen.** Bisecado con las
mismas filas y la misma consulta:

    DELETE FROM conversations  (cascada a messages) -> indice  1 / scan 10  ROTO
    DELETE FROM messages       (directo)            -> indice 10 / scan 10  sano

Y basta **UNA** conversacion: no hace falta vaciar nada. O sea que el dia que
exista un "borrar chat" -- una funcion de lo mas normal -- el indice de
`messages` empieza a mentir al primer uso.

**Hoy no hay ningun camino que borre conversaciones** en `jax` ni en
`jax-platform` (verificado el 2026-09-20), y produccion esta sana. Este modulo
existe para que el dia que ese camino aparezca, no nos enteremos tarde.

**Por que importa mas de lo que parece.** El que se queda ciego no es solo la
busqueda: es `_find_nearest_fact`, el dedup de la memoria. Para un dedup, "no
encontre nada parecido" significa "esto es nuevo". O sea que un indice
envenenado no hace que la memoria encuentre menos -- hace que **DUPLIQUE**, en
silencio y para siempre. Es corrupcion, no degradacion.

**La ceguera es PARCIAL.** Devolvio 1 de 10, no 0. Por eso la comprobacion NO
puede ser "devolvio vacio?": tiene que comparar el conteo por el indice contra
el mismo conteo con `IGNORE INDEX`. Un detector que solo mire el vacio da verde
con el indice roto, que es la peor clase de detector -- el que tranquiliza.

**Se repara sin perder datos** (medido): `DROP INDEX` + `ADD VECTOR INDEX`
reconstruye y devuelve 10 de 10, con todas las filas intactas. No hay que
exportar nada ni recalcular embeddings.

Runbook: `docs/runbooks/indice-vectorial-envenenado.md`.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

from dataclasses import dataclass

#: Cuantas filas se piden en la muestra. 10 alcanza: el caso medido daba 1.
#: Subirlo no detecta mas y cuesta mas en una tabla grande.
MUESTRA = 10


@dataclass(frozen=True)
class Informe:
    tabla: str
    indice: str
    columna: str
    filas: int
    muestra: int
    por_el_indice: int
    por_scan: int

    @property
    def sano(self) -> bool:
        """Sano = el indice devuelve lo MISMO que el scan completo.

        No "devuelve algo". Ver la ceguera parcial en la cabecera del modulo.
        """
        return self.por_el_indice == self.por_scan

    def __str__(self) -> str:
        estado = "sano" if self.sano else "ENVENENADO"
        return (f"{self.tabla}.{self.indice}: {estado} "
                f"(indice {self.por_el_indice} / scan {self.por_scan} "
                f"de una muestra de {self.muestra}, {self.filas} filas)")


async def indices_vectoriales(cur) -> list[tuple[str, str, str]]:
    """`(tabla, indice, columna)` de cada indice vectorial de la base actual.

    Se descubre por `INDEX_TYPE='VECTOR'` y no por el nombre de la columna: el
    dia que la columna cambie -- ya paso una vez, al pasar a bge-m3 -- esto
    sigue encontrandolos sin que nadie lo actualice.
    """
    await cur.execute(
        "SELECT TABLE_NAME, INDEX_NAME, COLUMN_NAME FROM information_schema.STATISTICS "
        "WHERE TABLE_SCHEMA = DATABASE() AND INDEX_TYPE = 'VECTOR' "
        "ORDER BY TABLE_NAME, INDEX_NAME")
    return [(t, i, c) for t, i, c in await cur.fetchall()]


async def revisar_uno(cur, tabla: str, indice: str, columna: str,
                      muestra: int = MUESTRA) -> Informe:
    """Compara la misma consulta por el indice y con `IGNORE INDEX`."""
    await cur.execute(f"SELECT COUNT(*) FROM `{tabla}`")
    filas = (await cur.fetchone())[0]
    if filas == 0:
        # Una tabla vacia no puede mentir: no hay nada que devolver.
        return Informe(tabla, indice, columna, 0, 0, 0, 0)

    # El vector de referencia sale de una fila REAL de la tabla: contra un
    # vector inventado, una tabla de embeddings normalizados puede dar
    # distancias degeneradas y el conteo dejaria de significar nada.
    await cur.execute(f"SELECT VEC_ToText(`{columna}`) FROM `{tabla}` ORDER BY id DESC LIMIT 1")
    referencia = (await cur.fetchone())[0]

    k = min(muestra, filas)

    async def _contar(ignorar: bool) -> int:
        hint = f" IGNORE INDEX (`{indice}`)" if ignorar else ""
        await cur.execute(
            f"SELECT COUNT(*) FROM (SELECT id FROM `{tabla}`{hint} "
            f"ORDER BY VEC_DISTANCE_COSINE(`{columna}`, VEC_FromText(%s)) ASC "
            f"LIMIT {k}) AS muestra", (referencia,))
        return (await cur.fetchone())[0]

    return Informe(tabla, indice, columna, filas, k,
                   por_el_indice=await _contar(False), por_scan=await _contar(True))


async def definicion_del_indice(cur, tabla: str, indice: str) -> str:
    """La linea EXACTA del `SHOW CREATE TABLE` que define este indice.

    No se reconstruye a mano. El indice de produccion es
    ``VECTOR KEY `idx_x` (`col`) `DISTANCE`='cosine' `M`='16'``, y un
    `ADD VECTOR INDEX (col)` pelado lo recrearia con los valores por omision:
    otra funcion de distancia y otro M. La busqueda seguiria "funcionando" y
    daria resultados distintos, sin un solo error. Una reparacion que degrada
    en silencio lo que venia a reparar es peor que no repararlo.
    """
    await cur.execute(f"SHOW CREATE TABLE `{tabla}`")
    create = (await cur.fetchone())[1]
    for linea in create.split("\n"):
        limpia = linea.strip().rstrip(",")
        if f"`{indice}`" in limpia and "VECTOR" in limpia.upper():
            return limpia
    raise LookupError(f"no encontre la definicion de {indice} en {tabla}")


async def reparar_uno(cur, tabla: str, indice: str, columna: str) -> str:
    """Reconstruye el indice CON SU MISMA DEFINICION. NO toca los datos.

    Medido: devuelve el indice a 10 de 10 con todas las filas intactas. Es DDL,
    asi que toma un lock sobre la tabla -- en produccion va con el servicio
    quieto, como cualquier migracion (ver el runbook). Devuelve la definicion
    que replico, para que quede en el registro.
    """
    definicion = await definicion_del_indice(cur, tabla, indice)
    await cur.execute(f"ALTER TABLE `{tabla}` DROP INDEX `{indice}`")
    await cur.execute(f"ALTER TABLE `{tabla}` ADD {definicion}")
    return definicion


async def revisar_todos(cur, muestra: int = MUESTRA) -> list[Informe]:
    return [await revisar_uno(cur, t, i, c, muestra)
            for t, i, c in await indices_vectoriales(cur)]


# ---------------------------------------------------------------------------
# La caché del índice: el otro modo de fallo, y este SÍ avisa antes de doler
# ---------------------------------------------------------------------------
#
# `mhnsw_max_cache_size` acota la caché DE CADA índice vectorial. Cuando los
# vectores no entran, la caché desaloja -- y bajo carga cada búsqueda recorre
# el grafo por un camino distinto. Medido el 2026-09-20 a 9.000 hechos:
# `/api/admin/memoria/grupos` devolvía entre 1.596 y 1.606 grupos en corridas
# consecutivas CON LOS MISMOS DATOS.
#
# Ojo con el diagnóstico fácil: NO es "el HNSW es aproximado y ya". Una
# consulta de vecinos suelta sale determinista 6 de 6 incluso con 9.000 filas,
# con `ef_search` 20 y 100. Lo que varía es el AGREGADO cuando la caché no
# alcanza. Por eso el centinela mira el tamaño, no la aproximación.
#
# A diferencia del envenenamiento por cascada, esto se puede ver venir: es
# aritmética. Por eso acá sí hay un aviso y allá no.

#: Un vector ocupa `dim * 4` bytes (float32).
BYTES_POR_FLOAT = 4

#: Se avisa a partir de este porcentaje de ocupación. 80 % deja margen para
#: actuar con calma: subir la variable es en caliente, pero hay que decidir el
#: número y persistirlo en el `conf.d` (si no, se pierde en el reinicio).
UMBRAL_AVISO = 0.80


@dataclass(frozen=True)
class Ocupacion:
    tabla: str
    columna: str
    filas: int
    dim: int
    cache_bytes: int

    @property
    def bytes_usados(self) -> int:
        return self.filas * self.dim * BYTES_POR_FLOAT

    @property
    def caben(self) -> int:
        return self.cache_bytes // (self.dim * BYTES_POR_FLOAT)

    @property
    def fraccion(self) -> float:
        return self.bytes_usados / self.cache_bytes if self.cache_bytes else float("inf")

    @property
    def holgada(self) -> bool:
        return self.fraccion < UMBRAL_AVISO

    def __str__(self) -> str:
        estado = "holgada" if self.holgada else "APRETADA"
        return (f"{self.tabla}.{self.columna}: caché {estado} "
                f"({self.filas:,} de ~{self.caben:,} vectores, "
                f"{self.fraccion * 100:.0f} % de {self.cache_bytes // 1024 // 1024} MB)")


async def ocupacion_de_cache(cur) -> list[Ocupacion]:
    """Cuántos vectores hay contra cuántos entran en la caché, por índice."""
    await cur.execute("SELECT @@mhnsw_max_cache_size")
    cache_bytes = int((await cur.fetchone())[0])

    salida = []
    for tabla, _indice, columna in await indices_vectoriales(cur):
        await cur.execute(
            "SELECT COLUMN_TYPE FROM information_schema.COLUMNS "
            "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = %s AND COLUMN_NAME = %s",
            (tabla, columna))
        fila = await cur.fetchone()
        # `vector(1024)` -> 1024. Si el tipo no se puede leer, no se inventa.
        dim = int(str(fila[0]).split("(")[1].split(")")[0]) if fila and "(" in str(fila[0]) else 0
        if dim <= 0:
            continue
        await cur.execute(f"SELECT COUNT(*) FROM `{tabla}`")
        filas = (await cur.fetchone())[0]
        salida.append(Ocupacion(tabla, columna, filas, dim, cache_bytes))
    return salida
