"""Lectura del catálogo del pre-vuelo contra el esquema REAL de jax-platform
(job jacobs-gobernanza-db). Cada test siembra SU proveedor, modelo, faceta,
binding y credencial sintéticos y los borra: no depende de semillas.

Requiere el esquema del plan P: capability.min_output_tokens y
facet_health_event.source con 'preflight'.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import asyncio
import os
import time
import uuid
from decimal import Decimal

import pytest

_db = os.environ.get("JAX_DB_NAME", "")
if not _db.endswith("_test"):
    raise RuntimeError(f"JAX_DB_NAME={_db!r}: este test solo corre contra una base *_test.")

from jacobs import facet_health as fh  # noqa: E402
from jacobs import prevuelo_catalogo as pc  # noqa: E402
from jacobs import store  # noqa: E402


class _Semilla:
    def __init__(self):
        sufijo = uuid.uuid4().hex[:8]
        self.proveedor = f"zz-pv-{sufijo}"
        self.faceta = f"zz-pv-{sufijo}"
        self.modelo = f"zz-modelo-{sufijo}"


async def _sembrar(cur, s, *, credencial="active", faceta_status="active"):
    await cur.execute(
        "INSERT INTO provider (id, display_name, base_url, auth_type, is_local) "
        "VALUES (%s, 'prevuelo test', 'https://zz.example/v1', 'api_key', FALSE)", (s.proveedor,))
    await cur.execute(
        "INSERT INTO model (provider_id, model_id, is_alias, status, source, source_checked_at, "
        "max_tokens_param, max_output_tokens, price_input_per_1m_usd, price_output_per_1m_usd) "
        "VALUES (%s, %s, FALSE, 'available', 'manual', NOW(), 'max_tokens', 4096, 0.2700, 1.1000)",
        (s.proveedor, s.modelo))
    await cur.execute("SELECT id FROM model WHERE provider_id=%s AND model_id=%s", (s.proveedor, s.modelo))
    (model_ref,) = await cur.fetchone()
    await cur.execute(
        "INSERT INTO facet (`key`, display_name, transport, status) "
        "VALUES (%s, 'prevuelo test', 'http_openai_compat', %s)", (s.faceta, faceta_status))
    await cur.execute(
        "INSERT INTO facet_binding (facet_key, provider_id, model_id, role, model_ref) "
        "VALUES (%s, %s, %s, 'primary', %s)", (s.faceta, s.proveedor, s.modelo, model_ref))
    if credencial:
        await cur.execute(
            "INSERT INTO credential (provider_id, env_key, encrypted_value, state) "
            "VALUES (%s, 'ZZ_PV_KEY', 'no-se-descifra', %s)", (s.proveedor, credencial))


async def _limpiar(cur, s):
    await cur.execute("DELETE FROM facet_health_event WHERE facet LIKE %s", (s.faceta + "%",))
    await cur.execute("DELETE FROM credential WHERE provider_id=%s", (s.proveedor,))
    await cur.execute("DELETE FROM facet_binding WHERE facet_key=%s", (s.faceta,))
    await cur.execute("DELETE FROM facet WHERE `key`=%s", (s.faceta,))
    await cur.execute("DELETE FROM model WHERE provider_id=%s", (s.proveedor,))
    await cur.execute("DELETE FROM provider WHERE id=%s", (s.proveedor,))


def _con_semilla(cuerpo, **kw):
    async def correr():
        s = _Semilla()
        conn = await store.get_conn()
        try:
            async with conn.cursor() as cur:
                await _sembrar(cur, s, **kw)
            return s, await cuerpo(s, conn)
        finally:
            async with conn.cursor() as cur:
                await _limpiar(cur, s)
            conn.close()
    return asyncio.run(correr())


def _con_semillas(n, cuerpo):
    """Como _con_semilla pero con N proveedores/facetas/modelos sintéticos --
    para EXPLAIN con n>=2 (fix round 1, item 2: sql_modelos es una cadena OR
    de pares, sql_ultimo_evento_de_proveedor lleva N claves)."""
    async def correr():
        s_list = [_Semilla() for _ in range(n)]
        conn = await store.get_conn()
        try:
            async with conn.cursor() as cur:
                for s in s_list:
                    await _sembrar(cur, s)
            return s_list, await cuerpo(s_list, conn)
        finally:
            async with conn.cursor() as cur:
                for s in s_list:
                    await _limpiar(cur, s)
            conn.close()
    return asyncio.run(correr())


async def _evento(conn, faceta, outcome, source, ts):
    async with conn.cursor() as cur:
        await cur.execute(
            "INSERT INTO facet_health_event (facet, outcome, source, detail, ts) VALUES (%s, %s, %s, NULL, %s)",
            (faceta, outcome, source, ts))


async def _explain(conn, sql, params):
    async with conn.cursor() as cur:
        await cur.execute("EXPLAIN " + sql, params)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in await cur.fetchall()]


def test_lee_faceta_modelo_precio_y_contrato():
    async def cuerpo(s, conn):
        return await pc.leer_catalogo(facetas={s.faceta}, motores=[], capabilities=set(), ahora=time.time())
    s, cat = _con_semilla(cuerpo)
    fila = cat.facetas[s.faceta]
    assert (fila.provider_id, fila.model_id, fila.base_url, fila.transport) == (
        s.proveedor, s.modelo, "https://zz.example/v1", "http_openai_compat")
    assert cat.modelos[(s.proveedor, s.modelo)] == pc.FilaModelo(
        "max_tokens", 4096, Decimal("0.2700"), Decimal("1.1000"))
    assert s.proveedor in cat.proveedores_con_credencial


def test_faceta_inactiva_no_aparece():
    async def cuerpo(s, conn):
        return await pc.leer_catalogo(facetas={s.faceta}, motores=[], capabilities=set(), ahora=time.time())
    s, cat = _con_semilla(cuerpo, faceta_status="disabled")
    assert s.faceta not in cat.facetas


def test_credencial_revocada_no_cuenta():
    async def cuerpo(s, conn):
        return await pc.leer_catalogo(facetas={s.faceta}, motores=[], capabilities=set(), ahora=time.time())
    s, cat = _con_semilla(cuerpo, credencial="revoked")
    assert s.proveedor not in cat.proveedores_con_credencial


def test_salud_toma_el_ultimo_evento_de_proveedor_e_ignora_los_del_gate():
    ahora = time.time()

    async def cuerpo(s, conn):
        await _evento(conn, s.faceta, "ok", "chat", ahora - 600)
        await _evento(conn, s.faceta, "provider_error", "preflight", ahora - 300)
        await _evento(conn, s.faceta, "unsupported_transport", "canary_periodic", ahora - 10)
        await _evento(conn, s.faceta, "gate_denied", "chat", ahora - 5)
        return await pc.leer_catalogo(facetas={s.faceta}, motores=[], capabilities=set(), ahora=ahora)
    s, cat = _con_semilla(cuerpo)
    assert cat.salud[s.faceta] == (ahora - 300, "provider_error")


def test_salud_fuera_de_ventana_no_cuenta():
    ahora = time.time()

    async def cuerpo(s, conn):
        await _evento(conn, s.faceta, "ok", "preflight", ahora - fh.HEALTH_WINDOW_SECONDS - 5)
        return await pc.leer_catalogo(facetas={s.faceta}, motores=[], capabilities=set(), ahora=ahora)
    s, cat = _con_semilla(cuerpo)
    assert s.faceta not in cat.salud


def test_registrar_evento_de_sonda_escribe_source_preflight_y_recorta():
    ahora = time.time()

    async def cuerpo(s, conn):
        await fh.registrar_evento_de_sonda(s.faceta, "ok", "x" * 300, ahora)
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT outcome, source, CHAR_LENGTH(detail) FROM facet_health_event WHERE facet=%s",
                (s.faceta,))
            return await cur.fetchall()
    _, filas = _con_semilla(cuerpo)
    assert list(filas) == [("ok", "preflight", 255)]


def test_min_output_tokens_sale_de_la_capability():
    async def cuerpo(s, conn):
        async with conn.cursor() as cur:
            await cur.execute("SELECT min_output_tokens FROM capability WHERE `key`='research'")
            (esperado,) = await cur.fetchone()
        cat = await pc.leer_catalogo(facetas=set(), motores=[], capabilities={"research"}, ahora=time.time())
        return esperado, cat.min_output_tokens
    _, (esperado, minimos) = _con_semilla(cuerpo)
    assert minimos == {"research": int(esperado)}


def test_explain_salud_usa_idx_facet_ts():
    ahora = time.time()

    async def cuerpo(s, conn):
        # Volumen para que el optimizador no prefiera un scan por tabla chica.
        for n in range(30):
            for k in range(10):
                await _evento(conn, f"{s.faceta}-rel-{n:02d}", "ok", "chat", ahora - k)
        await _evento(conn, s.faceta, "ok", "preflight", ahora)
        return await _explain(conn, fh.sql_ultimo_evento_de_proveedor(1),
                              (s.faceta, ahora - fh.HEALTH_WINDOW_SECONDS))
    _, filas = _con_semilla(cuerpo)
    assert any(f["key"] == "idx_facet_ts" for f in filas), filas
    assert all("filesort" not in (f.get("Extra") or "") for f in filas), filas


def test_explain_credencial_y_modelos_usan_sus_indices():
    async def cuerpo(s, conn):
        cred = await _explain(conn, pc.sql_credenciales(1), (s.proveedor,))
        modelos = await _explain(conn, pc.sql_modelos(1), (s.proveedor, s.modelo))
        return cred, modelos
    _, (cred, modelos) = _con_semilla(cuerpo)
    assert [f["key"] for f in cred] == ["idx_provider_state"], cred
    assert [f["key"] for f in modelos] == ["uk_provider_model"], modelos


# ---------------------------------------------------------------------------
# Fix round 1 (revisión post-Task 6)
# ---------------------------------------------------------------------------

def test_registrar_evento_de_sonda_redacta_antes_de_recortar():
    """Item 1 (fix round 2: sensible al ORDEN de verdad). La forma libre
    `api_key=valor` NO tiene mínimo de longitud (`[^&\\s'",;<>}\\]]+`, un
    `+` sin cota) -- un valor cortado a la mitad sigue matcheando "api_key="
    + lo que quede, así que ese caso pasa aunque se recorte primero y se
    redacte después: no prueba el orden.

    La key con forma `AIza...` sí lo prueba: `_KEY_GOOGLE` exige un mínimo
    de 10 caracteres después de `AIza` (`{10,}`, jax/core/redaccion.py).
    Si se recorta ANTES de redactar y el corte deja MENOS de 10 caracteres
    visibles después de `AIza`, la regex ya NO reconoce la forma y el
    pedazo (`AIza` + unos pocos caracteres) queda en claro -- el mismo caso
    de tests/test_redaccion.py::test_una_key_AIza_que_cruza_el_corte_no_deja_un_pedazo,
    portado acá contra el escritor real de la sonda.

    Verificado a mano con la mutación slice-then-redact (recortar primero,
    redactar después): con esa mutación este test da rojo -- `AIza` +
    3 caracteres sobrevive en claro. Revertida, ver task-6-report.md ronda 2."""
    ahora = time.time()
    key_falsa = "AIza" + "Q" * 60  # forma real: AIza + >=10 (regex), 60 de sobra
    detalle = ("x" * 248) + key_falsa + " cola"
    assert len(detalle) > fh._LARGO_DETALLE
    inicio = detalle.index(key_falsa)
    fin_prefijo_AIza = inicio + len("AIza")
    # El corte de 255 cae DENTRO de la key, a menos de 10 caracteres del
    # prefijo "AIza": si alguien recortara antes de redactar, la regex de
    # forma (`{10,}`) ya no reconocería lo que queda.
    chars_visibles_tras_AIza = fh._LARGO_DETALLE - fin_prefijo_AIza
    assert 0 < chars_visibles_tras_AIza < 10, chars_visibles_tras_AIza
    assert fin_prefijo_AIza < fh._LARGO_DETALLE < inicio + len(key_falsa)

    async def cuerpo(s, conn):
        await fh.registrar_evento_de_sonda(s.faceta, "provider_error", detalle, ahora)
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT detail FROM facet_health_event WHERE facet=%s", (s.faceta,))
            return await cur.fetchone()
    _, (guardado,) = _con_semilla(cuerpo)
    assert guardado is not None
    assert len(guardado) <= fh._LARGO_DETALLE
    assert "AIza" not in guardado, guardado
    # Ningun fragmento reconocible de la key (ni un prefijo parcial largo)
    # sobrevive.
    for corte in range(10, len(key_falsa), 10):
        assert key_falsa[:corte] not in guardado, (corte, guardado)


def test_empate_de_ts_lo_gana_provider_error():
    """Item 3 / Ruling R12: con el mismo MAX(ts), provider_error gana el
    empate -- determinista, y el resultado es 'sondear', no 'sana' por
    casualidad de orden físico de filas. `provider_error` se inserta
    PRIMERO a propósito: contra el código viejo (sin orden explícito) esto
    devolvía 'ok' -- verificado a mano insertando en este orden, ver
    fix-round-1 en task-6-report.md."""
    ahora = time.time()
    empate = ahora - 10

    async def cuerpo(s, conn):
        await _evento(conn, s.faceta, "provider_error", "preflight", empate)
        await _evento(conn, s.faceta, "ok", "chat", empate)
        return await pc.leer_catalogo(facetas={s.faceta}, motores=[], capabilities=set(), ahora=ahora)
    s, cat = _con_semilla(cuerpo)
    assert cat.salud[s.faceta] == (empate, "provider_error")
    assert fh.salud_de_proveedor(cat.salud[s.faceta], ahora) == "sondear"


def test_explain_facetas_evita_full_scan():
    """Item 2: sql_facetas(n) es un JOIN de 4 tablas (facet, facet_binding,
    provider, model); cada salto va por PK o índice único, nunca un scan."""
    async def cuerpo(s, conn):
        return await _explain(conn, pc.sql_facetas(1), (s.faceta,))
    s, filas = _con_semilla(cuerpo)
    assert all(f["type"] != "ALL" for f in filas), filas
    assert all("filesort" not in (f.get("Extra") or "") for f in filas), filas


def test_explain_min_output_tokens_evita_full_scan():
    """Item 2: sql_min_output_tokens(n) contra capability.key (PRIMARY),
    con n=2 claves reales de la semilla de producción."""
    async def correr():
        conn = await store.get_conn()
        try:
            return await _explain(conn, pc.sql_min_output_tokens(2), ("research", "analysis"))
        finally:
            conn.close()
    filas = asyncio.run(correr())
    assert all(f["type"] != "ALL" for f in filas), filas
    assert all("filesort" not in (f.get("Extra") or "") for f in filas), filas


def test_explain_modelos_n2_usa_uk_provider_model():
    """Item 2: sql_modelos(n) es una cadena OR de pares (provider_id,
    model_id); con n=2 cada rama tiene que seguir yendo por uk_provider_model,
    no degradar a un scan de la tabla."""
    async def cuerpo(s_list, conn):
        s1, s2 = s_list
        return await _explain(
            conn, pc.sql_modelos(2), (s1.proveedor, s1.modelo, s2.proveedor, s2.modelo))
    _, filas = _con_semillas(2, cuerpo)
    assert all(f["key"] == "uk_provider_model" for f in filas), filas
    assert all(f["type"] != "ALL" for f in filas), filas
    assert all("filesort" not in (f.get("Extra") or "") for f in filas), filas


def test_explain_salud_n2_usa_idx_facet_ts():
    """Item 2: sql_ultimo_evento_de_proveedor(n) con n=2 claves, mismo
    volumen que el test de n=1 para que el optimizador no prefiera un scan
    por tabla chica."""
    ahora = time.time()

    async def cuerpo(s_list, conn):
        s1, s2 = s_list
        for n in range(30):
            for k in range(10):
                await _evento(conn, f"{s1.faceta}-rel-{n:02d}", "ok", "chat", ahora - k)
        await _evento(conn, s1.faceta, "ok", "preflight", ahora)
        await _evento(conn, s2.faceta, "ok", "preflight", ahora)
        return await _explain(
            conn, fh.sql_ultimo_evento_de_proveedor(2),
            (s1.faceta, s2.faceta, ahora - fh.HEALTH_WINDOW_SECONDS))
    _, filas = _con_semillas(2, cuerpo)
    assert any(f["key"] == "idx_facet_ts" for f in filas), filas
    assert all("filesort" not in (f.get("Extra") or "") for f in filas), filas


# ---------------------------------------------------------------------------
# Fix round 2 (revisión de la ronda de arreglo 1)
# ---------------------------------------------------------------------------

def test_explain_facetas_n3_evita_full_scan():
    """Item 2 (ronda 2): el pre-vuelo real pasa VARIAS facetas en la misma
    consulta, no una sola -- n=3 para que el plan de sql_facetas(n) siga
    siendo por PK/índice único en las 4 tablas y no degrade a un scan
    cuando la lista de claves crece."""
    async def cuerpo(s_list, conn):
        claves = sorted(s.faceta for s in s_list)
        return await _explain(conn, pc.sql_facetas(3), claves)
    _, filas = _con_semillas(3, cuerpo)
    assert all(f["type"] != "ALL" for f in filas), filas
    assert all("filesort" not in (f.get("Extra") or "") for f in filas), filas


# ---------------------------------------------------------------------------
# Fix round 3 (revisión de Task 7, fix round 1 -- Ruling R13b): la sonda
# también puede escribir 'config_error' (facet_health.OUTCOMES_DE_SONDA),
# pero SOLO 'ok'/'provider_error' cuentan para la salud -- ver
# tests/test_prevuelo_salud_pura.py::test_la_consulta_solo_cuenta_eventos_de_proveedor,
# que ya fija ese contrato a nivel de SQL y no cambió en esta ronda.
# ---------------------------------------------------------------------------

def test_registrar_evento_de_sonda_acepta_config_error():
    """R13b: la sonda escribe 'config_error' cuando no pudo ni preparar la
    llamada (credencial ausente, transporte desconocido, contrato sin tope)
    -- tiene que poder guardarse igual que 'ok'/'provider_error'."""
    ahora = time.time()

    async def cuerpo(s, conn):
        await fh.registrar_evento_de_sonda(
            s.faceta, "config_error", "la sonda no pudo preparar la llamada: sin credencial", ahora)
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT outcome, source FROM facet_health_event WHERE facet=%s", (s.faceta,))
            return await cur.fetchall()
    _, filas = _con_semilla(cuerpo)
    assert list(filas) == [("config_error", "preflight")]


def test_registrar_evento_de_sonda_rechaza_outcome_invalido():
    """R13b, contraparte: cualquier valor que NO sea ok/provider_error/
    config_error se sigue rechazando -- 'config_error' se agregó a la lista
    permitida, no se abrió la validación entera."""
    async def cuerpo():
        with pytest.raises(ValueError):
            await fh.registrar_evento_de_sonda("cualquiera", "bogus", None, time.time())
    asyncio.run(cuerpo())


def test_salud_ignora_config_error_como_los_gate():
    """R13b: el lector de salud (sql_ultimo_evento_de_proveedor,
    OUTCOMES_DE_PROVEEDOR) ignora 'config_error' igual que gate_*/unbound/
    unsupported_transport -- un config_error MÁS FRESCO que un 'ok' viejo no
    tiene que tapar ese 'ok': la salud sigue viendo el último evento de
    NIVEL PROVEEDOR, no el último evento a secas."""
    ahora = time.time()

    async def cuerpo(s, conn):
        await _evento(conn, s.faceta, "ok", "chat", ahora - 600)
        await _evento(conn, s.faceta, "config_error", "preflight", ahora - 10)
        return await pc.leer_catalogo(facetas={s.faceta}, motores=[], capabilities=set(), ahora=ahora)
    s, cat = _con_semilla(cuerpo)
    assert cat.salud[s.faceta] == (ahora - 600, "ok")
