"""`search_similar_facts`: JAX busca hechos por similitud en cada turno.

POR QUE EXISTE (2026-09-20, decision de Fernando tras un fallo real).
Fernando le pregunto a JAX "jax sabes a que me dedico?" y contesto que no
sabia, pese a tener el hecho GUARDADO Y VERIFICADO (fact #7: "Fernando Ruiz
es Licenciado en administracion de empresas, tiene un MBA en finanzas...").
Causa medida: `backend/api/chat.py` (jax-platform) solo inyecta facts cuando
`detect_completeness_intent(user_text)` reconoce una de seis categorias fijas
por palabra clave -- "jax sabes a que me dedico?" no matchea ninguna y
`detect_completeness_intent` devuelve None, asi que el turno NUNCA busca
facts. `_find_nearest_fact` (busqueda vectorial sobre `facts`, con el indice
HNSW `idx_embedding_bge_m3`) existe en este archivo pero solo la usa
`save_fact` al ESCRIBIR -- nadie la usa para LEER. `search_similar_facts` es
el metodo que le falta a `MemoryDB`: hermano de `search_similar_messages`
pero sobre `facts`, con umbral de similitud y sin devolver hechos superados
ni vencidos.

Mismo patron de aislamiento que los demas archivos de memoria: user_id
reservado, solo corre contra una base de tests, limpia lo que crea. Las
distancias no salen de Ollama (en la suite `JAX_OLLAMA_URL` apunta a un host
invalido a proposito, ver conftest.py raiz): se construyen vectores
SINTETICOS con un angulo exacto respecto del vector de consulta, para poder
afirmar sobre la distancia coseno sin depender de un modelo real.
"""
from __future__ import annotations

import asyncio
import functools
import math
import os
from datetime import datetime, timedelta

import aiomysql
import pytest

from jax.core.db_connect_config import db_connect_timeout_seconds
from jax.memory import db as dbmod
import _esquema_memoria

from base_de_test import es_base_de_test  # noqa: E402

_DB = os.getenv("JAX_DB_NAME", "")
requiere_db_de_prueba = pytest.mark.skipif(
    not os.getenv("JAX_DB_HOST") or not es_base_de_test(_DB),
    reason="necesita una MariaDB real y JAX_DB_NAME en una base de tests",
)

_DIM = dbmod.EMBEDDING_DIM
_USER = 990_050          # reservado para este archivo
_OTRO_USER = 990_051      # el vecino: sus hechos individuales NUNCA aparecen
_PROYECTO = 990_052


def _consulta() -> list[float]:
    """El vector de la CONSULTA: un basis vector puro (e0). Todo lo demas se
    construye a un angulo exacto de este."""
    v = [0.0] * _DIM
    v[0] = 1.0
    return v


def _vec_a_distancia(distancia: float) -> list[float]:
    """Un vector UNITARIO cuya distancia coseno contra `_consulta()` es
    EXACTAMENTE `distancia` (dentro del margen de punto flotante).

    Con e0 = _consulta() y v = cos(theta)*e0 + sin(theta)*e1 (e0, e1
    ortonormales), cos_sim(e0, v) = cos(theta) porque v ya tiene norma 1. La
    distancia coseno de MariaDB es 1 - cos_sim, asi que theta = arccos(1 -
    distancia) da el vector exacto. Permite afirmar sobre el umbral sin
    depender de Ollama (invalido a proposito en esta suite)."""
    cos_sim = 1.0 - distancia
    cos_sim = max(-1.0, min(1.0, cos_sim))
    v = [0.0] * _DIM
    v[0] = cos_sim
    v[1] = math.sqrt(max(0.0, 1.0 - cos_sim * cos_sim))
    return v


def asincrono(fn):
    @functools.wraps(fn)
    def wrapper(*a, **k):
        return asyncio.run(fn(*a, **k))
    return wrapper


async def _conn():
    return await aiomysql.connect(
        host=os.environ["JAX_DB_HOST"], port=int(os.environ["JAX_DB_PORT"]),
        user=os.getenv("JAX_DB_USER", ""), password=os.getenv("JAX_DB_PASSWORD", ""),
        db=os.environ["JAX_DB_NAME"], autocommit=True,
        connect_timeout=db_connect_timeout_seconds())


async def _sql(query, args=(), fetch=False):
    conn = await _conn()
    try:
        async with conn.cursor() as cur:
            await cur.execute(query, args)
            if fetch:
                return list(await cur.fetchall())
            return cur.lastrowid
    finally:
        conn.close()


_DDL = _esquema_memoria.ddl()


async def _preparar() -> list[str]:
    creadas = []
    for nombre, ddl in _DDL.items():
        existe = await _sql(
            "SELECT 1 FROM information_schema.TABLES "
            "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = %s", (nombre,), fetch=True)
        if not existe:
            await _sql(ddl)
            creadas.append(nombre)
    return creadas


async def _limpiar():
    """TRUNCATE, no DELETE: un DELETE masivo envenena el indice HNSW y no se
    recupera (ver tests/_esquema_memoria.py::vaciar, medido 2026-09-20).
    Vaciar tambien reproduce la proporcion de filas de CI (base nace vacia):
    con basura de otras sesiones en la tabla, el EXPLAIN de mas abajo puede
    elegir otro plan por estimacion de selectividad, no por regresion."""
    conn = await _conn()
    try:
        base = os.environ["JAX_DB_NAME"]
        assert es_base_de_test(base), f"me negue a vaciar tablas en {base!r}"
        await _esquema_memoria.vaciar(conn)
    finally:
        conn.close()


@pytest.fixture
def limpio():
    creadas = asyncio.run(_preparar())
    asyncio.run(_limpiar())
    yield
    async def _teardown():
        await _limpiar()
        for nombre in reversed(list(_DDL)):
            if nombre in creadas:
                await _sql(f"DROP TABLE {nombre}")
    asyncio.run(_teardown())


async def _memoria(monkeypatch, vec: list[float]) -> dbmod.MemoryDB:
    m = dbmod.MemoryDB()
    ok = await m.connect(
        os.environ["JAX_DB_HOST"], os.getenv("JAX_DB_USER", ""),
        os.getenv("JAX_DB_PASSWORD", ""), os.environ["JAX_DB_NAME"],
        port=int(os.environ["JAX_DB_PORT"]))
    assert ok, "MemoryDB no conecto: el test no probaria nada"

    async def _emb(_texto):
        return vec

    monkeypatch.setattr(m, "get_embedding", _emb)
    return m


async def _crear_fact(fact_text: str, embedding: list[float], **overrides) -> int:
    campos = {
        "fact_type": "user",
        "confidence": 0.7,
        "is_verified": True,
        "user_id": _USER,
        "project_id": None,
        "expires_at": None,
        "superseded_by": None,
    }
    campos.update(overrides)
    vec_txt = "[" + ",".join(str(x) for x in embedding) + "]"
    return await _sql(
        "INSERT INTO facts (fact_uuid, fact_text, fact_type, confidence, "
        "is_verified, user_id, project_id, expires_at, superseded_by, "
        f"{dbmod.EMBED.column}) "
        "VALUES (UUID(), %s, %s, %s, %s, %s, %s, %s, %s, VEC_FromText(%s))",
        (fact_text, campos["fact_type"], campos["confidence"], campos["is_verified"],
         campos["user_id"], campos["project_id"], campos["expires_at"],
         campos["superseded_by"], vec_txt))


# ---------------------------------------------------------------------------
# 1. El umbral de similitud: cerca entra, lejos no
# ---------------------------------------------------------------------------

@requiere_db_de_prueba
@asincrono
async def test_trae_un_hecho_cercano_y_descarta_uno_lejano(limpio, monkeypatch):
    """El caso real (2026-09-20): una pregunta trae el hecho parecido, no
    todo lo que hay guardado. 0.30 << FACT_SIMILARITY_THRESHOLD (0.62,
    documentado en jax/memory/db.py) y 0.90 >> umbral: no son casos limite,
    para que el test no dependa de d'onde se ponga exactamente la linea."""
    m = await _memoria(monkeypatch, _consulta())
    await _crear_fact("Fernando es Licenciado en administracion de empresas",
                       _vec_a_distancia(0.30))
    await _crear_fact("dato sin ninguna relacion con la consulta",
                       _vec_a_distancia(0.90))

    filas = await m.search_similar_facts("a que me dedico?", user_id=_USER)
    textos = [f["fact_text"] for f in filas]
    assert "Fernando es Licenciado en administracion de empresas" in textos
    assert "dato sin ninguna relacion con la consulta" not in textos, (
        f"un detector que trae todo no filtra nada: {textos}")


@requiere_db_de_prueba
@asincrono
async def test_una_pregunta_ajena_no_trae_nada(limpio, monkeypatch):
    """Control directo del criterio de aceptacion: sin ningun hecho cerca de
    la consulta, la lista vuelve vacia -- no hay relleno de "lo que haya"."""
    m = await _memoria(monkeypatch, _consulta())
    await _crear_fact("dato sin ninguna relacion con la consulta",
                       _vec_a_distancia(0.90))

    filas = await m.search_similar_facts("que clima hace hoy?", user_id=_USER)
    assert filas == []


@requiere_db_de_prueba
@asincrono
async def test_limit_por_defecto_es_8(limpio, monkeypatch):
    """Ronda 2 (2026-09-20, decision de Fernando sobre mediciones nuevas):
    el default baja de 30 a 8 -- este metodo es para RECUERDO ESPECIFICO,
    no para volcar la memoria. Con mas candidatos vigentes bajo el umbral
    que el tope, el default sigue truncando a 8, sin importar cuantos mas
    haya."""
    m = await _memoria(monkeypatch, _consulta())
    for i in range(15):
        await _crear_fact(f"hecho cercano {i}",
                           _vec_a_distancia(0.10 + i * 0.01))  # 0.10..0.24, todos < 0.45

    filas = await m.search_similar_facts("consulta", user_id=_USER)  # limit por defecto
    assert len(filas) == 8, (
        f"el limit por defecto tiene que ser 8 (ronda 2, ya no 30): "
        f"con 15 candidatos vigentes bajo el umbral se esperaban 8, "
        f"se obtuvieron {len(filas)}")


@requiere_db_de_prueba
@asincrono
async def test_limit_por_defecto_ya_no_compensa_preguntas_de_completeness(limpio, monkeypatch):
    """Contrapositivo del test retirado en esta misma ronda
    (`test_limit_por_defecto_alcanza_el_caso_real_del_bug`, ronda 1): agrandar
    `limit` para alcanzar un fact lejano (rank 26 de 28 por distancia,
    reproducido a escala) volcaba entre 16 y 30 facts ante frases sin
    contenido real -- medido contra produccion, "gracias" traia el tope
    (30). Con el `limit` nuevo (8), ese fact lejano YA NO vuelve por
    similitud -- y esta bien: el caso real ("jax sabes a que me dedico?")
    lo resuelve `detect_completeness_intent` devolviendo 'user'
    (tests/test_completeness_intent.py), que trae el fact completo via
    `get_facts()`, no por similitud."""
    m = await _memoria(monkeypatch, _consulta())
    for i in range(25):
        await _crear_fact(f"hecho sobre JAX, mas cercano {i}",
                           _vec_a_distancia(0.10 + i * 0.01))  # 0.10..0.34, todos < 0.45
    await _crear_fact("Fernando es Licenciado en administracion de empresas",
                       _vec_a_distancia(0.40))  # el mas lejano de los vigentes, igual < 0.45

    filas = await m.search_similar_facts("a que me dedico?", user_id=_USER)  # limit por defecto
    textos = [f["fact_text"] for f in filas]
    assert "Fernando es Licenciado en administracion de empresas" not in textos, (
        f"con el limit nuevo (8) un hecho fuera de los 8 mas cercanos no "
        f"tiene que volver por similitud -- ese caso lo resuelve "
        f"detect_completeness_intent, no este metodo: {textos}")
    assert len(textos) == 8


@requiere_db_de_prueba
@asincrono
async def test_limit_trunca_a_los_mas_cercanos_entre_los_que_pasan_el_umbral(limpio, monkeypatch):
    """`limit` no es el filtro principal (lo es el umbral, arriba) pero
    sigue siendo un tope real: con mas candidatos vigentes que `limit`,
    vuelven los `limit` MAS CERCANOS -- ni una cantidad arbitraria, ni por
    orden de insercion."""
    m = await _memoria(monkeypatch, _consulta())
    ids_por_distancia = []
    for i in range(8):
        d = 0.30 - i * 0.01  # 0.30 .. 0.23, todos < 0.45 con margen, insertados del MAS lejos al MAS cerca
        texto = f"hecho {i} a distancia {d:.2f}"
        await _crear_fact(texto, _vec_a_distancia(d))
        ids_por_distancia.append((d, texto))
    ids_por_distancia.sort(key=lambda t: t[0])
    esperados_en_orden = [texto for _, texto in ids_por_distancia[:3]]

    filas = await m.search_similar_facts("consulta", user_id=_USER, limit=3)
    textos = [f["fact_text"] for f in filas]
    assert textos == esperados_en_orden, (
        f"con limit=3 y 8 candidatos vigentes, se esperaban los 3 MAS "
        f"cercanos en orden: {esperados_en_orden}, se obtuvo {textos}")


# ---------------------------------------------------------------------------
# 2. Superados y vencidos NUNCA vuelven, aunque esten cerquisima
# ---------------------------------------------------------------------------

@requiere_db_de_prueba
@asincrono
async def test_un_hecho_superado_no_vuelve_aunque_sea_el_mas_cercano(limpio, monkeypatch):
    """Un hecho que Fernando fundio (superseded_by) no puede volver por la
    ventana de la busqueda semantica: es justo lo que la pantalla de Memoria
    vino a permitirle controlar."""
    m = await _memoria(monkeypatch, _consulta())
    vigente_id = await _crear_fact("hecho vigente", _vec_a_distancia(0.40))
    await _crear_fact("hecho superado, pero MUY cerca de la consulta",
                       _vec_a_distancia(0.05), superseded_by=vigente_id)

    filas = await m.search_similar_facts("consulta", user_id=_USER)
    textos = [f["fact_text"] for f in filas]
    assert "hecho superado, pero MUY cerca de la consulta" not in textos
    assert "hecho vigente" in textos


@requiere_db_de_prueba
@asincrono
async def test_un_hecho_vencido_no_vuelve_aunque_sea_el_mas_cercano(limpio, monkeypatch):
    """Mismo criterio que `_find_nearest_fact` (ver su docstring): un hecho
    caducado no es autoridad, ni siquiera para la busqueda semantica."""
    m = await _memoria(monkeypatch, _consulta())
    await _crear_fact("hecho vigente", _vec_a_distancia(0.40))
    await _crear_fact("hecho vencido, pero MUY cerca de la consulta",
                       _vec_a_distancia(0.05),
                       expires_at=datetime.now() - timedelta(days=1))

    filas = await m.search_similar_facts("consulta", user_id=_USER)
    textos = [f["fact_text"] for f in filas]
    assert "hecho vencido, pero MUY cerca de la consulta" not in textos
    assert "hecho vigente" in textos


# ---------------------------------------------------------------------------
# 3. Scope: igual que search_similar_messages / get_facts
# ---------------------------------------------------------------------------

@requiere_db_de_prueba
@asincrono
async def test_no_devuelve_hechos_individuales_de_otro_usuario(limpio, monkeypatch):
    m = await _memoria(monkeypatch, _consulta())
    await _crear_fact("SECRETO DEL VECINO", _vec_a_distancia(0.10), user_id=_OTRO_USER)
    await _crear_fact("cosa mia", _vec_a_distancia(0.10), user_id=_USER)

    filas = await m.search_similar_facts("consulta", user_id=_USER)
    textos = [f["fact_text"] for f in filas]
    assert "SECRETO DEL VECINO" not in textos, f"fuga de scope: {textos}"
    assert "cosa mia" in textos


@requiere_db_de_prueba
@asincrono
async def test_encuentra_el_hecho_del_proyecto_compartido(limpio, monkeypatch):
    m = await _memoria(monkeypatch, _consulta())
    await _crear_fact("dato del proyecto", _vec_a_distancia(0.10),
                       user_id=_OTRO_USER, project_id=_PROYECTO)

    filas = await m.search_similar_facts("consulta", project_id=_PROYECTO)
    textos = [f["fact_text"] for f in filas]
    assert "dato del proyecto" in textos


# ---------------------------------------------------------------------------
# 3b. El vocativo de faceta se pela ANTES de embeber (2026-09-20, ronda 2b)
# ---------------------------------------------------------------------------
# La logica del vocativo (que nombres cuentan, cuando se pela) esta cubierta
# a fondo en tests/test_vocativo_faceta.py, pura y sin DB. Este test cubre
# SOLO el cableado: que `search_similar_facts` de verdad llama a
# `_quitar_vocativo_faceta` antes de pedir el embedding, no que la funcion
# en si ande bien.

@requiere_db_de_prueba
@asincrono
async def test_pela_el_vocativo_antes_de_pedir_el_embedding(limpio, monkeypatch):
    m = await _memoria(monkeypatch, _consulta())
    vistos = []
    get_embedding_real = m.get_embedding

    async def _espia(texto):
        vistos.append(texto)
        return await get_embedding_real(texto)

    monkeypatch.setattr(m, "get_embedding", _espia)

    await m.search_similar_facts("jax sabes a que me dedico?", user_id=_USER)
    assert vistos == ["sabes a que me dedico?"], (
        f"search_similar_facts tiene que pelar el vocativo ANTES de pedir "
        f"el embedding: se le paso {vistos!r}")


# ---------------------------------------------------------------------------
# 4. El indice vectorial HNSW, a volumen (no con la tabla casi vacia)
# ---------------------------------------------------------------------------

@requiere_db_de_prueba
@asincrono
async def test_usa_el_indice_vectorial_a_volumen(limpio, monkeypatch):
    """El EXPLAIN de la consulta REAL, con volumen suficiente para que el
    optimizador de verdad elija el HNSW (con la tabla casi vacia elige otro
    plan y la medicion miente -- medido en tests/test_memory_scope_denormalized.py,
    misma leccion aplicada aca: se siembran 60 filas propias, no 1)."""
    m = await _memoria(monkeypatch, _consulta())
    for i in range(60):
        await _crear_fact(f"hecho de volumen {i}", _vec_a_distancia(0.1 + i * 0.01))

    consultas = []
    original = dbmod.aiomysql.cursors.DictCursor.execute

    async def espia(self, query, args=None):
        if "VEC_DISTANCE_COSINE" in (query or "") and "FROM facts" in (query or ""):
            consultas.append((query, args))
        return await original(self, query, args)

    monkeypatch.setattr(dbmod.aiomysql.cursors.DictCursor, "execute", espia)
    await m.search_similar_facts("consulta", user_id=_USER, limit=5)
    assert consultas, "no se capturo la consulta de busqueda de facts"

    query, args = consultas[0]
    assert "JOIN" not in query, (
        "la busqueda de facts no necesita JOIN -- user_id/project_id ya "
        "estan en la propia tabla facts")

    plan = await _sql("EXPLAIN " + query, args or (), fetch=True)
    texto = " | ".join(str(c) for fila in plan for c in fila)
    assert "idx_embedding_bge_m3" in texto, f"el plan no usa el indice vectorial: {texto}"


# ---------------------------------------------------------------------------
# 5. Fail-soft: sin pool o sin embedding, la busqueda no revienta el turno
# ---------------------------------------------------------------------------

def test_sin_pool_devuelve_lista_vacia():
    m = dbmod.MemoryDB()
    assert asyncio.run(m.search_similar_facts("consulta")) == []


@requiere_db_de_prueba
@asincrono
async def test_embedding_nulo_devuelve_lista_vacia(limpio, monkeypatch):
    m = dbmod.MemoryDB()
    ok = await m.connect(
        os.environ["JAX_DB_HOST"], os.getenv("JAX_DB_USER", ""),
        os.getenv("JAX_DB_PASSWORD", ""), os.environ["JAX_DB_NAME"],
        port=int(os.environ["JAX_DB_PORT"]))
    assert ok

    async def _sin_embedding(_texto):
        return None

    monkeypatch.setattr(m, "get_embedding", _sin_embedding)
    assert await m.search_similar_facts("consulta", user_id=_USER) == []
