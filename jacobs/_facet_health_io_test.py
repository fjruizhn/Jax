"""`check_facet_health()` -- la pieza de I/O del lector, contra una DB REAL.

POR QUE EXISTE, y por que no alcanzaba con `_facet_health_test.py`
-----------------------------------------------------------------
Aquel archivo cubre las funciones PURAS (`evaluate_states`,
`transitions_to_notify`) con 11 tests. `check_facet_health()` -- las tres
consultas, el envio, el ledger y la poda -- **no tenia ninguno**, y es la
pieza que conecta esa logica con los datos reales. Si falla ahi, la logica
pura corre impecablemente sobre datos que nunca llegaron.

Agrava el caso que su `except` en el reaper es fail-soft: si esto revienta,
el barrido sigue y solo queda un `logger.error` cada 300s. **Un fallo aca
se ve exactamente igual que un sistema sano.**

QUE RAMA CUBRE, y por que esa
-----------------------------
La rama `unknown` -> alerta agregada bajo `__system__`: la unica que avisa
si el detector entero muere. Nunca se ejercito en produccion (la tabla
`facet_health_alert` jamas tuvo una fila `__system__`) porque producirla
alli exige apagar la sonda dos horas. Un guard que nunca se probo con un
caso que DEBERIA dispararlo no es un guard, es una hipotesis.

Las tres propiedades se verifican POR SEPARADO -- una sola asercion que las
cubriera a las tres no diria cual se rompio:
  1. eventos fuera de la ventana -> `unknown`, NUNCA `ok`;
  2. la alerta va bajo la clave `__system__`, no una por facet y no lista
     vacia (el agujero del `if current and all(...)`: `{}` es falsy, asi
     que esa forma devuelve SILENCIO justo cuando el detector esta muerto);
  3. la supresion de 6h se respeta EN esa alerta agregada.
Mas el contrapositivo (`test_eventos_frescos_...`): sin el, un
`check_facet_health()` que devolviera siempre `__system__` pasaria 1-3.

SEGURIDAD
---------
- Solo corre contra una base cuyo nombre termina en `_test`. La funcion
  bajo prueba lee y BORRA sin filtrar por facet, asi que apuntar esto a
  `jax_memory` destruiria datos de produccion. El guard es fail-closed:
  sin `_test`, el modulo entero se salta.
- `send_telegram_alert` se parchea SIEMPRE. En hall9000 la DB y el token de
  Telegram salen del MISMO `/etc/jax/.env`: un test que se olvide del
  parche manda un mensaje real. Ademas se asserta que quedo parcheado.
"""
import asyncio
import functools
import os
import time

import aiomysql
import pytest

from jacobs import facet_health as fh
from jacobs import reaper


def asincrono(fn):
    """Corre el test en un event loop propio.

    Sin `pytest-asyncio` a proposito: una dependencia menos en el job de CI,
    y una fuente menos de verde/rojo segun la version del plugin."""
    @functools.wraps(fn)
    def wrapper(*a, **k):
        return asyncio.run(fn(*a, **k))
    return wrapper


_DB = os.getenv("JAX_DB_NAME", "")
requiere_db_de_prueba = pytest.mark.skipif(
    not os.getenv("JAX_DB_HOST") or not _DB.endswith("_test"),
    reason="necesita una MariaDB real con JAX_DB_NAME terminado en '_test'",
)

# Esquema esperado. Las tablas las CREA `jax-platform` (migrations.py), otro
# repo: aca solo se leen. Duplicar el DDL seria una segunda fuente de
# verdad, asi que la copia viene con guard de divergencia --
# `test_el_esquema_no_divergio` compara contra `information_schema` y falla
# si la tabla real dejo de coincidir. La copia no se cree la verdad: se cree
# una copia que sabe cuando quedo vieja.
_COLUMNAS = {
    "facet_health_event": ["id", "facet", "outcome", "source", "detail", "ts"],
    "facet_health_alert": ["facet", "state", "first_seen_ts", "notified_ts"],
}
_DDL = {
    "facet_health_event": """
        CREATE TABLE IF NOT EXISTS facet_health_event (
            id      BIGINT AUTO_INCREMENT PRIMARY KEY,
            facet   VARCHAR(50) NOT NULL,
            outcome ENUM('ok','provider_error','gate_denied','gate_unreachable',
                         'unbound','unsupported_transport','probe_error') NOT NULL,
            source  VARCHAR(30) NOT NULL,
            detail  VARCHAR(255) NULL,
            ts      DOUBLE NOT NULL,
            INDEX idx_facet_ts (facet, ts)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
    "facet_health_alert": """
        CREATE TABLE IF NOT EXISTS facet_health_alert (
            facet          VARCHAR(50) PRIMARY KEY,
            state          VARCHAR(20) NOT NULL,
            first_seen_ts  DOUBLE NOT NULL,
            notified_ts    DOUBLE NULL
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
}

VENCIDO = fh.HEALTH_WINDOW_SECONDS + 3600      # fuera de la ventana de 2h,
                                               # dentro de la retencion de 30d


async def _conn():
    return await aiomysql.connect(
        host=os.environ["JAX_DB_HOST"], port=int(os.environ["JAX_DB_PORT"]),
        user=os.getenv("JAX_DB_USER", ""), password=os.getenv("JAX_DB_PASSWORD", ""),
        db=os.environ["JAX_DB_NAME"], autocommit=True)


async def _sql(query, args=(), fetch=False):
    conn = await _conn()
    try:
        async with conn.cursor() as cur:
            await cur.execute(query, args)
            return list(await cur.fetchall()) if fetch else None
    finally:
        conn.close()


async def _tabla_limpia():
    for nombre, ddl in _DDL.items():
        await _sql(ddl)
        await _sql(f"DELETE FROM {nombre}")


@pytest.fixture
def sin_telegram(monkeypatch):
    """Captura los mensajes en vez de mandarlos. Ver SEGURIDAD arriba."""
    enviados = []

    async def fake(mensaje):
        enviados.append(mensaje)
        return {"ok": True, "message_id": 0, "error": None}

    monkeypatch.setattr(reaper, "send_telegram_alert", fake)
    assert reaper.send_telegram_alert is fake, "el parche no quedo puesto"
    return enviados


@requiere_db_de_prueba
@asincrono
async def test_el_esquema_no_divergio():
    """Guard de la copia del DDL: si `jax-platform` cambia estas tablas,
    este test falla y alguien actualiza la copia. Sin el, la copia envejece
    en silencio y los otros tests pasarian sobre un esquema que ya no es el
    de produccion -- la forma de 'los specs envejecen', aplicada a un DDL."""
    for tabla, esperadas in _COLUMNAS.items():
        await _sql(_DDL[tabla])
        filas = await _sql(
            "SELECT COLUMN_NAME FROM information_schema.COLUMNS "
            "WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME=%s "
            "ORDER BY ORDINAL_POSITION", (tabla,), fetch=True)
        assert [f[0] for f in filas] == esperadas, f"{tabla} divergio"


@requiere_db_de_prueba
@asincrono
async def test_eventos_vencidos_dan_unknown_y_NUNCA_ok(sin_telegram):
    """Propiedad 1. Ausencia de datos FRESCOS no es salud.

    Los eventos existen -- estan en la tabla y dicen `ok` -- pero cayeron
    fuera de la ventana. Un lector que mirara solo el outcome del ultimo
    evento diria `ok` sobre una sonda muerta hace horas."""
    await _tabla_limpia()
    ahora = time.time()
    for facet in ("thot", "ada"):
        await _sql("INSERT INTO facet_health_event (facet,outcome,source,ts) "
                   "VALUES (%s,'ok','canary_periodic',%s)",
                   (facet, ahora - VENCIDO))

    res = await fh.check_facet_health()

    assert res["states"] == {"thot": "unknown", "ada": "unknown"}
    assert "ok" not in res["states"].values()


@requiere_db_de_prueba
@asincrono
async def test_la_alerta_va_bajo___system___y_no_es_lista_vacia(sin_telegram):
    """Propiedad 2. Con TODO en unknown el diagnostico es 'la sonda no esta
    corriendo', no seis facets caidos: un mensaje por facet describiria mal
    el problema y ademas serian N mensajes por barrido.

    La asercion que importa es que `notified` NO sea `[]`. Ese es el modo de
    falla real del `if current and all(...)`: silencio total justo cuando el
    detector esta muerto."""
    await _tabla_limpia()
    ahora = time.time()
    for facet in ("thot", "ada", "jekyll"):
        await _sql("INSERT INTO facet_health_event (facet,outcome,source,ts) "
                   "VALUES (%s,'ok','canary_periodic',%s)",
                   (facet, ahora - VENCIDO))

    res = await fh.check_facet_health()

    assert res["notified"] == [fh.SYSTEM_KEY]
    assert res["notified"] != []
    assert len(sin_telegram) == 1
    assert "sonda de facets no esta corriendo" in sin_telegram[0]
    ledger = await _sql("SELECT facet, state FROM facet_health_alert", fetch=True)
    assert ledger == [(fh.SYSTEM_KEY, "unknown")]


@requiere_db_de_prueba
@asincrono
async def test_tabla_VACIA_alerta_igual_bajo___system__(sin_telegram):
    """Propiedad 2, en su forma mas fuerte y la que el codigo llama aparte:
    con la tabla VACIA, `current` es `{}` -- y `{}` es falsy.

    Un guard escrito `if current and all(...)` saltaria el bloque, el bucle
    no iteraria nada y esto devolveria `[]` = SILENCIO, justo cuando el
    detector esta del todo muerto (sonda sin cablear, escritor roto,
    servicio caido mas que la retencion). Ausencia TOTAL de datos es la
    senal mas fuerte que hay, no la mas debil. Los otros tests insertan
    eventos, asi que `current` nunca queda vacio y NINGUNO cubre esta rama."""
    await _tabla_limpia()

    res = await fh.check_facet_health()

    assert res["states"] == {}
    assert res["notified"] == [fh.SYSTEM_KEY]
    assert len(sin_telegram) == 1


@requiere_db_de_prueba
@asincrono
async def test_la_supresion_de_6h_se_respeta_en_la_alerta_agregada(sin_telegram):
    """Propiedad 3. Sin supresion, una sonda muerta el viernes produce 288
    mensajes el sabado (un barrido cada 300s). Con ella, uno cada 6h.

    Se verifican las DOS direcciones: que el segundo barrido inmediato NO
    avise, y que despues de 6h SI vuelva a avisar. Solo la primera dejaria
    pasar una supresion que no se levanta nunca -- que es dejar de avisar
    sobre un detector muerto."""
    await _tabla_limpia()
    ahora = time.time()
    await _sql("INSERT INTO facet_health_event (facet,outcome,source,ts) "
               "VALUES ('thot','ok','canary_periodic',%s)", (ahora - VENCIDO,))

    primero = await fh.check_facet_health()
    assert primero["notified"] == [fh.SYSTEM_KEY]

    segundo = await fh.check_facet_health()
    assert segundo["notified"] == [], "aviso de nuevo dentro de la ventana de 6h"
    assert len(sin_telegram) == 1

    await _sql("UPDATE facet_health_alert SET notified_ts=%s WHERE facet=%s",
               (time.time() - fh.ALERT_REPEAT_SECONDS - 1, fh.SYSTEM_KEY))
    tercero = await fh.check_facet_health()
    assert tercero["notified"] == [fh.SYSTEM_KEY], "la supresion no se levanta nunca"
    assert len(sin_telegram) == 2


@requiere_db_de_prueba
@asincrono
async def test_eventos_frescos_no_producen___system__(sin_telegram):
    """Contrapositivo, y no es decorado: sin el, un `check_facet_health()`
    que devolviera SIEMPRE `__system__` pasaria los tres tests de arriba.
    Un guard que dice violacion siempre no protege mas que uno que calla
    siempre; solo se rompe distinto."""
    await _tabla_limpia()
    ahora = time.time()
    await _sql("INSERT INTO facet_health_event (facet,outcome,source,ts) "
               "VALUES ('thot','ok','canary_periodic',%s)", (ahora - 60,))

    res = await fh.check_facet_health()

    assert res["states"] == {"thot": "ok"}
    assert fh.SYSTEM_KEY not in res["notified"]
    # Con el ledger vacio, `thot` transiciona None -> ok y una transicion
    # SIEMPRE avisa: es la recuperacion automatica, deliberada (un detector
    # que avisa cuando algo se rompe pero no cuando se arregla obliga a
    # mirar a mano). Lo que este test fija es que ese aviso nombre al FACET
    # y no sea la alerta agregada de sonda muerta.
    assert sin_telegram == ["JAX -- facet 'thot': ok"]
    assert not any(fh.SYSTEM_KEY in m for m in sin_telegram)
