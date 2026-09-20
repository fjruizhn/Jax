# tests/test_memoria_gobernanza.py
"""Gobernanza de la memoria: autoria de la aprobacion y de la correccion.

Medido el 2026-09-20 contra produccion: `facts` tiene `verified_at` pero NO
`verified_by`, y `superseded_by` es el id del HECHO que reemplaza, no del
usuario. El spec 2026-09-18-memoria-admin §2.2 pide aprobar "con quien y
cuando" -- sin estas columnas el quien no se puede guardar.
"""
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


def test_el_esquema_declarado_y_el_migrador_no_se_contradicen():
    """jax_memory_schema.sql es la fuente de verdad (checker de deriva) y
    migrations.py lleva las bases existentes hacia adelante. Si una columna
    esta en uno y no en el otro, una base nueva y una vieja quedan distintas --
    que es exactamente como `depends_on` de jacobs_steps termino existiendo
    solo en produccion (DEUDA.md)."""
    import pathlib
    esquema = pathlib.Path(__file__).resolve().parent.parent / "jax_memory_schema.sql"
    texto = esquema.read_text(encoding="utf-8")
    for col in COLUMNAS_NUEVAS:
        assert col in texto, f"{col} no esta en jax_memory_schema.sql"


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
