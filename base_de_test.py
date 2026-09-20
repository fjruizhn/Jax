"""La base de datos de tests de ESTA sesión.

**El problema real** (2026-09-17). Tres sesiones de Claude corriendo la suite
a la vez comparten una sola base, `jax_memory_test`, en la MariaDB de hall9000.
Se pisaron de verdad: filas con `mode` en NULL, un arnés expirando pipelines
ajenos, una sesión borrando una fila de uso de otra, un test dejando todas las
filas en NULL. El síntoma son rojos intermitentes que no son del código -- el
peor tipo de rojo, porque enseña a reintentar hasta que pase.

**La decisión** (Fernando, 2026-09-17). Cada sesión usa su propia base:
`JAX_TEST_DB_SUFIJO=<sufijo>` y la suite corre contra
`jax_memory_test_<sufijo>`.

**El mecanismo existía y colisionó igual** (medido 2026-09-20, dos worktrees
corriendo la suite a la vez, `tests/test_jacobs_reaper_cas_db.py`,
`test_memory_vector_zero_io.py` y `test_migrar_embeddings_io.py` variando
entre corridas). La causa: `JAX_TEST_DB_SUFIJO` era **opt-in** -- sin la
variable puesta a mano, la base seguía siendo `jax_memory_test` pelada y
CUALQUIER sesión que no se acordara de exportarla caía ahí, junto con
cualquier otra que tampoco se acordara. Un mecanismo de aislamiento que hay
que recordar activar no protege a la sesión que lo olvida.

**La decisión** (Fernando, 2026-09-20). El aislamiento pasa a ser el
comportamiento por defecto. Sin `JAX_TEST_DB_SUFIJO` puesto, cada proceso
genera uno propio (`auto<pid><random>`) la primera vez que resuelve el
nombre, lo fija en `os.environ` para que el resto del proceso -- y los
subprocesos que heredan el entorno -- vean el mismo valor, y sigue de ahí en
más como cualquier sesión con sufijo explícito. La única excepción es CI: los
jobs de `.github/workflows/policy.yml` que tocan la base ya corren cada uno
contra su propio contenedor MariaDB efímero (no hay sesiones concurrentes que
se puedan pisar ahí), así que agregar un sufijo ahí sólo metería un clonado
de esquema de más sin comprar nada. Se detecta con `CI`, la variable que
exportan GitHub Actions, GitLab CI y CircleCI por convención -- un hecho
verificable, no una adivinanza.

**Por qué un error y no un fallback.** Un sufijo inválido (puesto a mano, con
un valor que no matchea `SUFIJO_VALIDO`) sigue siendo un error explícito al
arrancar, nunca una caída silenciosa a la base compartida: caer a la
compartida en silencio ES el defecto que esto arregla. Misma razón por la que
`_verificar_que_no_es_produccion()` existe aunque el nombre se arme con un
f-string que no puede dar `jax_memory`: el control no está para el camino que
hoy se ve imposible, está para el refactor que mañana lo hace posible
(Principio IX -- el contrato va antes que la capacidad).

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import atexit
import os
import re
import secrets

#: La base compartida de siempre. Sin sufijo, la suite sigue corriendo acá.
BASE_COMPARTIDA = "jax_memory_test"

#: PRODUCCIÓN. Ninguna resolución de este módulo puede dar este nombre.
BASE_DE_PRODUCCION = "jax_memory"

#: La variable de entorno que elige la base de la sesión.
VARIABLE_DEL_SUFIJO = "JAX_TEST_DB_SUFIJO"

#: La variable que el código lee para saber contra qué base conectarse.
VARIABLE_DE_LA_BASE = "JAX_DB_NAME"

#: Minúsculas, números y guion bajo. Nada más: el nombre va a un `CREATE
#: DATABASE` y a un `DROP DATABASE`, y no se escapa un identificador con
#: parámetros. Lo que no matchea esto no llega a la SQL.
SUFIJO_VALIDO = re.compile(r"\A[a-z0-9_]+\Z")

#: El identificador de MariaDB tiene un tope de 64 caracteres. Se valida el
#: NOMBRE COMPLETO contra ese tope, no el sufijo contra un número inventado:
#: si mañana `BASE_COMPARTIDA` cambia de largo, el control sigue siendo cierto.
LARGO_MAXIMO_DEL_IDENTIFICADOR = 64


class BaseDeTestInvalida(RuntimeError):
    """El sufijo (o el nombre resuelto) no sirve. Se levanta al arrancar."""


def _verificar_que_no_es_produccion(nombre: str) -> str:
    """El último control antes de devolver un nombre: nunca `jax_memory`, y
    siempre derivado de `BASE_COMPARTIDA`.

    Es deliberadamente redundante con la construcción del nombre. Un control
    que sólo cubre lo que hoy puede pasar no protege del cambio de mañana, y
    lo que está en juego es la base de producción.
    """
    if nombre == BASE_DE_PRODUCCION:
        raise BaseDeTestInvalida(
            f"la base de tests resolvió a {BASE_DE_PRODUCCION!r}, que es PRODUCCIÓN. "
            f"La suite BORRA y ACTUALIZA filas: no corre contra esa base."
        )
    if nombre != BASE_COMPARTIDA and not nombre.startswith(f"{BASE_COMPARTIDA}_"):
        raise BaseDeTestInvalida(
            f"la base de tests resolvió a {nombre!r}, que no es {BASE_COMPARTIDA!r} "
            f"ni una base con el prefijo {BASE_COMPARTIDA + '_'!r}."
        )
    return nombre


def es_base_de_test(nombre: str | None) -> bool:
    """¿`nombre` es una base de tests legítima? La compartida o una con sufijo
    válido. `jax_memory` (y cualquier otra cosa) da False."""
    if not nombre:
        return False
    if nombre == BASE_COMPARTIDA:
        return True
    prefijo = f"{BASE_COMPARTIDA}_"
    if not nombre.startswith(prefijo):
        return False
    if len(nombre) > LARGO_MAXIMO_DEL_IDENTIFICADOR:
        return False
    return bool(SUFIJO_VALIDO.match(nombre[len(prefijo):]))


def _en_ci() -> bool:
    """¿Esta corrida es un job de CI? `CI` es la variable que exportan
    GitHub Actions, GitLab CI y CircleCI por convención propia -- se lee
    tal cual, no se inventa. Cada job de `.github/workflows/policy.yml` que
    toca la base corre contra su propio contenedor MariaDB efímero (ver
    `services: mariadb:` de cada job): no hay sesiones concurrentes que se
    puedan pisar ahí, así que el default automático de acá abajo no hace
    falta y sólo sumaría un clonado de esquema de más en cada corrida."""
    return os.environ.get("CI", "").strip().lower() in ("1", "true", "yes")


def _sufijo_automatico_de_sesion() -> str:
    """Un sufijo propio de ESTE proceso, para la sesión que no exportó
    `JAX_TEST_DB_SUFIJO` a mano. `os.getpid()` más unos bytes al azar: el PID
    solo no alcanza (se reutiliza entre procesos que ya terminaron), y el
    azar solo pierde la pista de qué proceso la creó al mirar el nombre en
    `SHOW DATABASES`.

    Antes de este mecanismo, crear una base con sufijo era un paso que
    alguien pedía a propósito (`export JAX_TEST_DB_SUFIJO=...`); ahora pasa
    en CADA corrida local que no lo pida. Sin limpiar, eso multiplica el
    ritmo al que se acumulan bases huérfanas (ya había una decena en
    hall9000 antes de este cambio). Por eso acá mismo se registra el borrado
    al salir del proceso -- ver `_borrar_al_salir()`. Una base con sufijo
    EXPLÍCITO (pasado a mano) NO se registra: alguien pudo poner ese sufijo
    a propósito para reusarla entre corridas, y borrarla forzaría un
    clonado de esquema de más en cada pytest suelto."""
    sufijo = f"auto{os.getpid()}{secrets.token_hex(4)}"
    atexit.register(_borrar_al_salir, f"{BASE_COMPARTIDA}_{sufijo}")
    return sufijo


def _borrar_al_salir(nombre: str) -> None:
    """Registrado en `atexit` SOLO para una base auto-generada (nunca para
    una pasada por `JAX_TEST_DB_SUFIJO` a mano). Sin `JAX_DB_HOST` no hay
    MariaDB a mano y no hay nada que borrar -- mismo criterio que
    `asegurar_base_de_test()`. Cualquier error (red caída, timeout) queda
    silenciado a propósito: es un best-effort de limpieza al cerrar, no una
    condición de salida del proceso; lo que esto no llegue a borrar lo
    barre después `scripts/limpiar_bases_de_test.py`."""
    if not os.environ.get("JAX_DB_HOST"):
        return
    import asyncio
    try:
        asyncio.run(_dropear_base_de_sesion(nombre))
    except Exception:
        pass


async def _dropear_base_de_sesion(nombre: str) -> None:
    """El DROP de verdad. El candado es el mismo `es_base_de_test()` que usa
    el resto del módulo, MÁS la exclusión explícita de `BASE_COMPARTIDA`:
    esta función nunca borra la base pelada ni nada que no lleve el prefijo
    de test. `tests/test_base_por_sesion.py` lo ejercita intentando borrar
    `jax_memory` y `jax_memory_test` de verdad."""
    if nombre == BASE_COMPARTIDA or not es_base_de_test(nombre):
        return
    import aiomysql

    from jax.core.db_connect_config import db_connect_timeout_seconds

    conn = await aiomysql.connect(
        db=BASE_PLANTILLA, autocommit=True,
        connect_timeout=db_connect_timeout_seconds(),
        **_parametros_de_conexion())
    try:
        async with conn.cursor() as cur:
            await cur.execute(f"DROP DATABASE IF EXISTS `{nombre}`")
    finally:
        conn.close()


def nombre_base_de_test(sufijo: str | None = None) -> str:
    """El nombre de la base de tests de esta sesión.

    Con `JAX_TEST_DB_SUFIJO` puesto (a mano, o ya fijado por una llamada
    anterior de este mismo proceso): `jax_memory_test_<sufijo>`.

    Sin la variable: en CI, `jax_memory_test` pelada, como siempre (cada job
    ya está aislado en su propio contenedor). Fuera de CI -- el caso de
    cualquier sesión o worktree local -- se genera un sufijo propio del
    proceso y se fija en `JAX_TEST_DB_SUFIJO` para que el resto de esta
    sesión (y los subprocesos que hereden el entorno) resuelvan la MISMA
    base. Decisión de Fernando, 2026-09-20: el aislamiento es el default,
    nadie tiene que acordarse de exportar nada.

    Un sufijo presente pero inválido (vacío, con mayúsculas, con guiones,
    con punto y coma, demasiado largo) es `BaseDeTestInvalida`. Un sufijo
    vacío TAMBIÉN es un error y no un "como si no estuviera": quien exportó
    la variable quiso una base propia, y darle la compartida en silencio es
    justo el defecto que este módulo arregla.
    """
    if sufijo is None:
        sufijo = os.environ.get(VARIABLE_DEL_SUFIJO)
    if sufijo is None:
        if _en_ci():
            return _verificar_que_no_es_produccion(BASE_COMPARTIDA)
        sufijo = _sufijo_automatico_de_sesion()
        os.environ[VARIABLE_DEL_SUFIJO] = sufijo
    if not SUFIJO_VALIDO.match(sufijo):
        raise BaseDeTestInvalida(
            f"{VARIABLE_DEL_SUFIJO}={sufijo!r} no sirve como sufijo de base: "
            f"sólo minúsculas, números y guion bajo (`[a-z0-9_]+`), sin vacío. "
            f"NO se cae a {BASE_COMPARTIDA!r}: la base compartida es lo que "
            f"este mecanismo evita."
        )
    nombre = f"{BASE_COMPARTIDA}_{sufijo}"
    if len(nombre) > LARGO_MAXIMO_DEL_IDENTIFICADOR:
        raise BaseDeTestInvalida(
            f"{VARIABLE_DEL_SUFIJO}={sufijo!r} da {nombre!r}, de {len(nombre)} "
            f"caracteres: el identificador de MariaDB tope en "
            f"{LARGO_MAXIMO_DEL_IDENTIFICADOR}."
        )
    return _verificar_que_no_es_produccion(nombre)


def fijar_base_de_test() -> str:
    """Pone `JAX_DB_NAME` en la base de esta sesión, pise lo que pise.

    Es el reemplazo exacto de `os.environ["JAX_DB_NAME"] = "jax_memory_test"`
    que cada archivo de test escribía a mano: mismo comportamiento (override
    incondicional), pero respetando el sufijo de la sesión.
    """
    nombre = nombre_base_de_test()
    os.environ[VARIABLE_DE_LA_BASE] = nombre
    return nombre


def exigir_base_de_test() -> str:
    """Como `fijar_base_de_test()`, pero respeta una `JAX_DB_NAME` que ya
    venga puesta -- y explota si esa que viene NO es una base de tests.

    Es el reemplazo del par
    `if _existing and _existing != "jax_memory_test": raise` + `setdefault`,
    que protege del `set -a; . <(sudo -n cat /etc/jax/.env)` (ahí `JAX_DB_NAME=jax_memory`).
    """
    actual = os.environ.get(VARIABLE_DE_LA_BASE)
    if actual:
        if not es_base_de_test(actual):
            raise BaseDeTestInvalida(
                f"{VARIABLE_DE_LA_BASE}={actual!r} ya está seteado (¿sourceaste "
                f"/etc/jax/.env?). Este test ESCRIBE en la base: sólo corre "
                f"contra {BASE_COMPARTIDA!r} o una base "
                f"{BASE_COMPARTIDA + '_<sufijo>'!r}."
            )
        return actual
    return fijar_base_de_test()


# --------------------------------------------------------------------------
# Creación de la base de la sesión
# --------------------------------------------------------------------------
#
# La base con sufijo no existe hasta que alguien la crea. Se crea CLONANDO el
# esquema de `jax_memory_test` -- la base que la suite usa hoy -- y corriendo
# después `jacobs.store.init_tables()`, que es el camino que este repo ya tiene
# para su propio esquema. No hay un DDL paralelo acá a propósito: una segunda
# fuente de verdad del esquema se desincroniza sola (Regla Absoluta).
#
# El esquema de `jax_memory_test` incluye tablas que NO son de este repo
# (`facet`, `motor`, `messages`, ... las crea jax-platform). Por eso la
# plantilla es la base compartida y no `init_tables()` a secas: una base vacía
# más `init_tables()` no reproduce lo que la suite necesita.

#: De dónde se copia el esquema de una base nueva.
BASE_PLANTILLA = BASE_COMPARTIDA

#: Además del esquema se copian los DATOS de las tablas chicas. No es un
#: capricho de tamaño: las tablas de catálogo y gobernanza (`facet`,
#: `capability`, `motor`, `provider`, `model`, `facet_binding`, `credential`,
#: `jax_tenants`...) son las que el código RESUELVE contra la base, y una base
#: recién clonada sin ellas hace fallar tests que hoy pasan -- medido el
#: 2026-09-17: `tests/test_facetas_de_gobernanza_db.py` en rojo contra una
#: clonación sólo de esquema. Las tablas grandes son historia transaccional
#: (`jacobs_events`, 953k filas / 222 MB; `jacobs_steps`; `shadow_messages`)
#: que la suite se escribe sola y que nadie afirma nada sobre ella: copiarlas
#: costaría minutos por sesión y cientos de MB.
#:
#: El corte se declara acá y se puede mover con `JAX_TEST_DB_FILAS_MAXIMAS`.
#: Es un umbral, no una lista de tablas: una lista se desactualiza en cuanto
#: alguien agrega una tabla de catálogo y nadie se entera hasta el rojo.
FILAS_MAXIMAS_A_COPIAR = int(os.environ.get("JAX_TEST_DB_FILAS_MAXIMAS", "2000"))


def _parametros_de_conexion() -> dict:
    """Host, puerto y credenciales de la MariaDB, del entorno (/etc/jax/.env).
    El NOMBRE de la base no sale de acá: lo elige quien llama.

    `connect_timeout` NO sale de acá aunque sea un parámetro de conexión: el
    tripwire `tests/test_aiomysql_connect_timeout_tripwire.py` lee el AST y
    exige el kwarg ESCRITO en la llamada, no escondido en un `**dict` --
    justamente para que no se pierda en una indirección. Me lo encontró a mí
    el 2026-09-17."""
    return {
        "host": os.environ.get("JAX_DB_HOST", "127.0.0.1"),
        "port": int(os.environ.get("JAX_DB_PORT", "3306")),
        "user": os.environ.get("JAX_DB_USER", ""),
        "password": os.environ.get("JAX_DB_PASSWORD", ""),
    }


async def _clonar_esquema(nombre: str) -> int:
    """Crea `nombre` y le copia el ESQUEMA (no los datos) de la plantilla.
    Devuelve cuántas tablas copió. Idempotente: si la base ya existe, no
    toca nada y devuelve -1."""
    import aiomysql  # import perezoso: la mayoría de los tests no crea bases

    _verificar_que_no_es_produccion(nombre)
    if nombre == BASE_COMPARTIDA:
        return -1

    from jax.core.db_connect_config import db_connect_timeout_seconds

    conn = await aiomysql.connect(
        db=BASE_PLANTILLA, autocommit=True,
        connect_timeout=db_connect_timeout_seconds(),
        **_parametros_de_conexion())
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT COUNT(*) FROM information_schema.SCHEMATA "
                "WHERE SCHEMA_NAME=%s", (nombre,))
            if (await cur.fetchone())[0]:
                return -1
            # El nombre está validado por `nombre_base_de_test()`/`es_base_de_test()`
            # contra `[a-z0-9_]`: un identificador no se pasa como parámetro.
            if not es_base_de_test(nombre):
                raise BaseDeTestInvalida(f"{nombre!r} no es una base de tests")
            await cur.execute(f"CREATE DATABASE `{nombre}`")
            await cur.execute(
                "SELECT TABLE_NAME FROM information_schema.TABLES "
                "WHERE TABLE_SCHEMA=%s AND TABLE_TYPE='BASE TABLE' ORDER BY TABLE_NAME",
                (BASE_PLANTILLA,))
            tablas = [fila[0] for fila in await cur.fetchall()]
            copiadas = 0
            # Sin chequeo de FKs mientras se copia: el orden alfabético no
            # respeta las dependencias y no hay datos que validar.
            await cur.execute("SET FOREIGN_KEY_CHECKS=0")
            for tabla in tablas:
                await cur.execute(f"SHOW CREATE TABLE `{BASE_PLANTILLA}`.`{tabla}`")
                ddl = (await cur.fetchone())[1]
                ddl = ddl.replace(
                    f"CREATE TABLE `{tabla}`", f"CREATE TABLE `{nombre}`.`{tabla}`", 1)
                await cur.execute(ddl)
                copiadas += 1
                # Los datos de las tablas chicas (catálogo y gobernanza). El
                # COUNT es exacto a propósito: `information_schema.TABLE_ROWS`
                # es una ESTIMACIÓN de InnoDB, y decidir con una estimación es
                # suponer. Se cuenta una vez, al crear la base.
                await cur.execute(f"SELECT COUNT(*) FROM `{BASE_PLANTILLA}`.`{tabla}`")
                filas = (await cur.fetchone())[0]
                if filas and filas <= FILAS_MAXIMAS_A_COPIAR:
                    await cur.execute(
                        f"INSERT INTO `{nombre}`.`{tabla}` "
                        f"SELECT * FROM `{BASE_PLANTILLA}`.`{tabla}`")
            # Las VISTAS, después de las tablas que miran. `motor_resolved` es
            # una: sin ella `MotorCatalog.from_db()` explota con un 1146 y
            # medio catálogo de LAS MANOS queda sin tests (medido el
            # 2026-09-17, 7 rojos). Su definición viene calificada con el
            # esquema de la plantilla y hay que reapuntarla a la base nueva.
            await cur.execute(
                "SELECT TABLE_NAME, VIEW_DEFINITION FROM information_schema.VIEWS "
                "WHERE TABLE_SCHEMA=%s", (BASE_PLANTILLA,))
            for vista, definicion in await cur.fetchall():
                definicion = definicion.replace(
                    f"`{BASE_PLANTILLA}`.", f"`{nombre}`.")
                await cur.execute(f"CREATE VIEW `{nombre}`.`{vista}` AS {definicion}")
                copiadas += 1
            await cur.execute("SET FOREIGN_KEY_CHECKS=1")

            # Lo que este clonador NO copia tiene que GRITAR, no faltar en
            # silencio: un trigger o un procedimiento nuevo en la plantilla
            # daría una base de sesión sutilmente distinta, y el rojo que
            # provoque va a parecer del código. Hoy la plantilla no tiene
            # ninguno (verificado 2026-09-17).
            for tipo, tabla_is, columna in (
                ("triggers", "TRIGGERS", "TRIGGER_SCHEMA"),
                ("rutinas", "ROUTINES", "ROUTINE_SCHEMA"),
                ("eventos", "EVENTS", "EVENT_SCHEMA"),
            ):
                await cur.execute(
                    f"SELECT COUNT(*) FROM information_schema.{tabla_is} "
                    f"WHERE {columna}=%s", (BASE_PLANTILLA,))
                cuantos = (await cur.fetchone())[0]
                if cuantos:
                    raise BaseDeTestInvalida(
                        f"{BASE_PLANTILLA} tiene {cuantos} {tipo} y este clonador "
                        f"no los copia: la base de sesión saldría distinta de la "
                        f"plantilla. Agregalos a `_clonar_esquema()` antes de seguir."
                    )
            return copiadas
    finally:
        conn.close()


def asegurar_base_de_test(nombre: str | None = None) -> str:
    """Deja lista la base de esta sesión: la crea con el esquema de la
    plantilla si no existía, y le corre `init_tables()` del repo.

    Sin `JAX_DB_HOST` no hay MariaDB a mano (los jobs de tests puros del CI):
    no se crea nada y no es un error.
    """
    import asyncio

    nombre = nombre or nombre_base_de_test()
    _verificar_que_no_es_produccion(nombre)
    if nombre == BASE_COMPARTIDA or not os.environ.get("JAX_DB_HOST"):
        return nombre

    anterior = os.environ.get(VARIABLE_DE_LA_BASE)
    os.environ[VARIABLE_DE_LA_BASE] = nombre
    try:
        asyncio.run(_clonar_esquema(nombre))
        # El esquema propio del repo, por SU camino. Corre siempre (no sólo al
        # crear): la plantilla puede estar atrasada respecto de esta rama.
        from jacobs import store  # import perezoso: arrastra el pool
        asyncio.run(store.init_tables())
    except Exception:
        if anterior is None:
            os.environ.pop(VARIABLE_DE_LA_BASE, None)
        else:
            os.environ[VARIABLE_DE_LA_BASE] = anterior
        raise
    return nombre
