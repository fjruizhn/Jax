"""
Config compartida de conexión a jax_memory -- connect_timeout de aiomysql.

Hallazgo de revisión de la Tarea 2b (tanda A, PR-B, hallazgo de la revisión
de la Tarea 6; ronda de arreglo 1, 2026-09-14): TODO `aiomysql.connect(...)`
del repo abría sin `connect_timeout`. Medido contra aiomysql 0.3.2
(`inspect.signature`, y el código real en `aiomysql/connection.py`): el
default es `connect_timeout=None`, y aiomysql lo pasa tal cual a
`asyncio.wait_for(timeout=None)` al abrir el socket TCP -- sin límite. Si la
DB se cuelga en vez de rechazar, cada `_db_conn()`/`get_conn()`/`_conn()` de
este repo esperaba indefinidamente.

Punto único de lectura y validación de `JAX_DB_CONNECT_TIMEOUT_SECONDS`,
compartido desde jax/core/, las_manos/ (raíz y motor_registry/) y jacobs/ --
decisión explícita del controller para NO duplicar esta validación en cada
uno de los sitios que abren `aiomysql.connect()`/`create_pool()`.

Import por CONTEXTO REAL, NO uno solo (corregido 2026-09-14, ronda de
arreglo 3 -- la versión anterior de este párrafo decía que el import era
"siempre absoluto" porque "el checkout siempre corre con la raíz del repo
en sys.path"; es falso, medido contra el servicio real de LAS MANOS). En
jax/core/ y en tests/ (siempre invocados con la raíz del repo en
`sys.path` -- el REPL, los workers de `jax.memory`, `python -m pytest`) el
import es absoluto y simple: `from jax.core.db_connect_config import
db_connect_timeout_seconds`. Pero LAS MANOS corre en producción como
`uvicorn server:app` con `WorkingDirectory=/home/fruiz/jax/las_manos` y SIN
`PYTHONPATH` -- medido con el python real de ese venv y ese cwd: ahí
`jax.core` NO es importable ("No module named 'jax'"), solo lo que vive
DIRECTO en `las_manos/` (bare). Por eso los archivos que corren (también o
solo) en ese contexto --`jax/core/facet_resolver.py` (symlinkeado como
`las_manos/facet_resolver.py`), `las_manos/credential_resolver.py`,
`las_manos/model_catalog.py`, `las_manos/motor_registry/{catalog,
facet_policy,usage_writer}.py`, `jacobs/store.py`, `jacobs/usage_writer.py`
-- usan el mismo `try/except` bare-primero-calificado-después que este
módulo comparte con `credential_resolver.py`: el bare `from
db_connect_config import ...` resuelve vía el symlink
`las_manos/db_connect_config.py -> ../jax/core/db_connect_config.py` (en
LAS MANOS y en cualquier job de CI con `las_manos/` en `PYTHONPATH`); el
`except ImportError` cae a `jax.core.db_connect_config` para los contextos
donde solo la raíz del repo está en `sys.path` (el REPL, jobs de CI sin
`PYTHONPATH` propio, como `governance`).

Es una excepción deliberada al patrón "espejo mínimo, sin paquete
compartido" que gobierna credential_resolver.py/model_catalog.py/etc (ver
sus propios docstrings): este módulo no tiene estado, no toca la DB y no
tiene dependencias más allá de `os` de la stdlib -- compartirlo no
reintroduce el acoplamiento que ese patrón evita (repos/venvs con su
propio código de I/O real, desplegables por separado).

LÍMITE (informativo, verificado 2026-09-14 contra aiomysql 0.3.2,
`aiomysql/connection.py` líneas ~519-532): `connect_timeout` SOLO acota la
apertura del socket TCP -- envuelve `asyncio.wait_for()` alrededor de
`_open_connection()`/`_open_unix_connection()`. El handshake MySQL posterior
(`_get_server_information`, `_request_authentication`) y cualquier
`cur.execute(...)` después de conectar corren SIN este límite ni ningún
otro. jax-platform acota la recarga COMPLETA del catálogo (socket +
handshake + queries) con su propio `wait_for`
(`GOVERNANCE_RELOAD_TIMEOUT_SECONDS`, Tarea 6 / PR-C,
`backend/governance_context.py`) -- este helper es la capa de abajo (el
socket), no un reemplazo de esa cota.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import os


def db_connect_timeout_seconds() -> int:
    """Lee y valida `JAX_DB_CONNECT_TIMEOUT_SECONDS` (default 10s -- alcanza
    para un connect local/LAN sano; no hay una medición de "el" valor
    correcto universal, así que se declara explícito en vez de inventar uno
    más fino sin evidencia).

    P10: un valor no numérico o no positivo NO cae a un default silencioso
    -- lanza `RuntimeError` antes de que el llamador intente conectar.
    "Sin límite" tiene que pedirse a propósito en otro lado (este helper
    nunca lo entrega), no colarse por un typo en la variable de entorno.
    """
    timeout_raw = os.getenv("JAX_DB_CONNECT_TIMEOUT_SECONDS", "10")
    try:
        connect_timeout = int(timeout_raw)
    except ValueError:
        connect_timeout = None
    if connect_timeout is None or connect_timeout <= 0:
        raise RuntimeError(
            f"JAX_DB_CONNECT_TIMEOUT_SECONDS={timeout_raw!r} inválido -- "
            "tiene que ser un entero positivo (segundos). Sin esto "
            "aiomysql.connect() espera sin límite si la DB se cuelga."
        )
    return connect_timeout
