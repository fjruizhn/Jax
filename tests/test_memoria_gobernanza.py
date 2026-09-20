# tests/test_memoria_gobernanza.py
"""Gobernanza de la memoria: autoria de la aprobacion y de la correccion.

Medido el 2026-09-20 contra produccion: `facts` tiene `verified_at` pero NO
`verified_by`, y `superseded_by` es el id del HECHO que reemplaza, no del
usuario. El spec 2026-09-18-memoria-admin §2.2 pide aprobar "con quien y
cuando" -- sin estas columnas el quien no se puede guardar.
"""
import re

import pytest

from jax.memory import migrations

COLUMNAS_NUEVAS = {"verified_by", "superseded_by_user"}


def test_las_columnas_de_autoria_estan_declaradas_en_el_migrador():
    declaradas = {col for (tabla, col, _ddl) in migrations._COLUMNAS if tabla == "facts"}
    assert COLUMNAS_NUEVAS <= declaradas, f"faltan: {COLUMNAS_NUEVAS - declaradas}"


def test_el_indice_de_revision_esta_declarado():
    """La pantalla filtra por verificado y caducidad y ordena por fecha: las
    tres columnas van en un indice compuesto, o el EXPLAIN de la Task 3 falla."""
    indices = {nombre for (tabla, nombre, _ddl) in migrations._INDICES if tabla == "facts"}
    assert "idx_facts_revision" in indices


def _columnas_de_facts_en_orden() -> list[str]:
    """Las columnas de `facts` tal como las declara jax_memory_schema.sql,
    EN ORDEN -- no un set: la posicion es justo lo que este archivo verifica."""
    import pathlib
    import re
    esquema = pathlib.Path(__file__).resolve().parent.parent / "jax_memory_schema.sql"
    texto = esquema.read_text(encoding="utf-8")
    m = re.search(r"CREATE TABLE `facts` \((.*?)\n\)[^;\n]*", texto, re.S)
    assert m, "facts no esta en jax_memory_schema.sql"
    columnas = []
    for linea in m.group(1).splitlines():
        cm = re.match(r"\s*`(\w+)`", linea)
        if cm:
            columnas.append(cm.group(1))
    return columnas


def test_el_esquema_declarado_y_el_migrador_no_se_contradicen():
    """jax_memory_schema.sql es la fuente de verdad (checker de deriva) y
    migrations.py lleva las bases existentes hacia adelante. Si una columna
    esta en uno y no en el otro, una base nueva y una vieja quedan distintas --
    que es exactamente como `depends_on` de jacobs_steps termino existiendo
    solo en produccion (DEUDA.md)."""
    columnas = _columnas_de_facts_en_orden()
    for col in COLUMNAS_NUEVAS:
        assert col in columnas, f"{col} no esta en jax_memory_schema.sql"


def test_el_migrador_agrega_las_columnas_en_la_MISMA_posicion_que_el_esquema():
    """No alcanza con que las dos fuentes tengan la columna (test anterior):
    tienen que tenerla en la MISMA posicion, o un `ALTER TABLE ADD COLUMN`
    sin `AFTER` la agrega al FINAL de la tabla en una base migrada, mientras
    jax_memory_schema.sql la declara en el medio (entre `verified_at` y
    `expires_at`). El checker de deriva (scripts/check_memory_schema_drift.py)
    compara el DDL completo, no solo que las columnas esten presentes: con
    las dos fuentes de acuerdo en QUE columnas hay pero en desacuerdo en
    DONDE, sale "DIFIERE" con el mismo texto a los dos lados -- el peor rojo
    posible, porque parece un bug del propio checker."""
    columnas = _columnas_de_facts_en_orden()
    ddl_por_columna = {
        col: ddl for (tabla, col, ddl) in migrations._COLUMNAS if tabla == "facts"
    }
    for col in COLUMNAS_NUEVAS:
        idx = columnas.index(col)
        assert idx > 0, f"{col} es la primera columna de facts en el .sql -- inesperado"
        anterior = columnas[idx - 1]
        ddl = ddl_por_columna[col]
        assert f"AFTER {anterior}" in ddl, (
            f"jax_memory_schema.sql pone {col!r} justo despues de "
            f"{anterior!r}, pero el DDL de migrations.py para {col!r} no "
            f"dice `AFTER {anterior}` ({ddl!r}) -- una base MIGRADA (no "
            f"creada desde cero con el .sql) va a terminar con {col!r} en "
            f"otra posicion.")


def test_la_columna_referenciada_por_AFTER_existe_de_antemano_o_se_agrega_antes():
    """m3 (auditoria adversarial 2026-09-20, SEGUNDA ronda). El test de
    arriba compara el DDL de migrations.py contra jax_memory_schema.sql,
    pero no exige que la columna de referencia (`AFTER <col>`) exista
    realmente en el momento en que `ensure_schema()` corre ese ALTER --
    `_COLUMNAS` se procesa en orden EXACTO (ver `ensure_schema()`): si
    alguien agregara una entrada nueva con `AFTER <otra-columna-nueva>` y la
    pusiera ANTES que esa otra en la lista, el `ALTER TABLE ... ADD COLUMN
    ... AFTER <otra-columna-nueva>` revienta con 'Unknown column' en
    cualquier base que todavia no la tenga -- el mismo tipo de fallo que
    dejaba a m2 (`verified_at`) sin cubrir, pero DENTRO de las columnas que
    este mismo migrador declara.

    Para cada `AFTER <col>`: si `<col>` es una de las que `_COLUMNAS`
    declara para la MISMA tabla, tiene que aparecer ANTES en la lista (se
    agrega ella misma antes de que se la use como referencia). Si `<col>` no
    esta en `_COLUMNAS` para esa tabla, se asume preexistente (como
    `verified_at`) y no hay nada que este test pueda validar sobre eso --
    ese caso lo cubre la migracion compensatoria (M2/m2), no esto."""
    after_re = re.compile(r"AFTER (\w+)")
    vistas_por_tabla: dict[str, set[str]] = {}
    for tabla, columna, ddl in migrations._COLUMNAS:
        vistas = vistas_por_tabla.setdefault(tabla, set())
        referencia = after_re.search(ddl)
        if referencia:
            ref = referencia.group(1)
            declaradas_en_esta_tabla = {
                c for (t, c, _ddl) in migrations._COLUMNAS if t == tabla
            }
            if ref in declaradas_en_esta_tabla:
                assert ref in vistas, (
                    f"{tabla}.{columna} tiene `AFTER {ref}`, pero {ref!r} "
                    f"TAMBIEN esta declarada en _COLUMNAS para {tabla!r} y "
                    f"no aparece ANTES que {columna!r} en la lista -- en una "
                    f"base que no tenga ninguna de las dos, el ALTER de "
                    f"{columna!r} va a fallar con 'Unknown column {ref!r}'")
        vistas.add(columna)


import inspect
from jax.memory.db import MemoryDB


def test_verify_fact_exige_saber_quien_aprueba():
    """Sin el quien, aprobar es un sello sin dueno -- el problema que la
    pantalla viene a resolver, no a repetir."""
    firma = inspect.signature(MemoryDB.verify_fact)
    assert "verified_by" in firma.parameters
    assert firma.parameters["verified_by"].default is inspect.Parameter.empty, \
        "verified_by no puede tener default: un aprobador implicito es un aprobador inventado"


def test_supersede_fact_registra_quien_corrigio():
    firma = inspect.signature(MemoryDB.supersede_fact)
    assert "superseded_by_user" in firma.parameters
    assert firma.parameters["superseded_by_user"].default is inspect.Parameter.empty


def test_existe_expire_fact():
    assert hasattr(MemoryDB, "expire_fact")
    firma = inspect.signature(MemoryDB.expire_fact)
    assert {"fact_id", "expires_at"} <= set(firma.parameters)


def test_get_facts_excluye_vencidos_por_defecto():
    """El spec §2.4: un hecho vencido no se borra, deja de pesar en la busqueda.
    Hoy `expires_at` no la lee NADIE (verificado 2026-09-20)."""
    fuente = inspect.getsource(MemoryDB.get_facts)
    assert "expires_at" in fuente, "get_facts sigue sin mirar la caducidad"
    firma = inspect.signature(MemoryDB.get_facts)
    assert firma.parameters["incluir_vencidos"].default is False
