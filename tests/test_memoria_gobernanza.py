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
