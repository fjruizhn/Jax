"""Controls of `tests/memory_b9_provision.py`, the fixture builder of the B9 driver.

Pure (no database): the parts that can hurt are the statement splitter, the
removal of `CREATE DATABASE jax_memory`/`USE jax_memory` from the schema file, and
the guard. The real build is exercised by the `memory-b9-regression` CI job, which
provisions a fresh MariaDB and runs the driver on it.
"""
import importlib.util
from pathlib import Path

import pytest

_RAIZ = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("memory_b9_provision", _RAIZ / "tests/memory_b9_provision.py")
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)

_ENTORNO_OK = {"JAX_DB_HOST": "127.0.0.1", "JAX_DB_PORT": "3306", "JAX_DB_USER": "jax_test",
               "JAX_DB_PASSWORD": "x", "JAX_DB_NAME": "jax_memory_test_memb9_ci"}


def test_el_esquema_se_carga_sin_crear_ni_usar_la_base_de_produccion():
    sentencias = P.sentencias_de_esquema()
    assert sentencias, "el esquema no puede quedar vacio"
    for s in sentencias:
        cabeza = s.lstrip().upper()
        assert not cabeza.startswith("CREATE DATABASE") and not cabeza.startswith("USE "), s[:60]
    # El archivo SI las trae: si algun dia dejan de estar, el control de arriba no probaria nada.
    crudo = (_RAIZ / "jax_memory_schema.sql").read_text()
    assert "CREATE DATABASE IF NOT EXISTS jax_memory" in crudo and "USE jax_memory;" in crudo


def test_el_divisor_respeta_delimiter_de_la_migracion_001():
    texto = (_RAIZ / "jax/memory/b9_migrations/001_b9_shared_memory.sql").read_text()
    sentencias = P.dividir_sentencias(texto)
    triggers = [s for s in sentencias if s.lstrip().upper().startswith("CREATE TRIGGER")]
    assert len(triggers) == 2
    for t in triggers:  # el cuerpo BEGIN ... END no se parte en el `;` interno
        assert "BEGIN SIGNAL" in t and t.rstrip().endswith("END"), t
    assert not any(s.upper().startswith("DELIMITER") for s in sentencias)


def test_el_divisor_ignora_comentarios_y_conserva_el_ultimo_sin_punto_y_coma():
    assert P.dividir_sentencias("-- nota\nSELECT 1;\n-- otra\nSELECT 2") == ["SELECT 1", "SELECT 2"]


def test_la_guarda_rechaza_produccion_y_otro_usuario_o_prefijo():
    P.guarda(dict(_ENTORNO_OK))
    for cambio in ({"JAX_DB_NAME": "jax_memory"}, {"JAX_DB_NAME": "jax_memory_test"},
                   {"JAX_DB_NAME": "jax_memory_test_otra"}, {"JAX_DB_USER": "root"}):
        with pytest.raises(RuntimeError, match="test_database_guard"):
            P.guarda({**_ENTORNO_OK, **cambio})
    for faltante in ("JAX_DB_HOST", "JAX_DB_PORT", "JAX_DB_PASSWORD"):
        entorno = dict(_ENTORNO_OK); del entorno[faltante]
        with pytest.raises(RuntimeError, match="test_configuration_missing"):
            P.guarda(entorno)


def test_el_orden_de_migraciones_cubre_todo_el_directorio():
    """Una migracion nueva (014...) tiene que colocarse a proposito en el orden de
    `provisionar`; si no, el constructor de fixtures y la produccion divergen en silencio."""
    numeros = sorted(p.name[:3] for p in (_RAIZ / "jax/memory/b9_migrations").glob("[0-9][0-9][0-9]_*.sql"))
    assert numeros == ["001", "002", "003", "004", "006", "007", "008", "009", "010", "011", "012", "013"]
    fuente = (_RAIZ / "tests/memory_b9_provision.py").read_text()
    assert '["001", "002", "003", "004", "006", "007", "008", "009", "010", "011", "012", "013"]' in fuente
