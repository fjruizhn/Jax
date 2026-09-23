"""Controles PUROS de la siembra de definiciones (sin MariaDB).

La siembra contra una base real vive en
`tests/policy/test_execution_mariadb_integration.py`
(`test_sembrar_definiciones_de_control_es_idempotente`), que es el único job
con una MariaDB levantada. Acá va lo que se puede probar sin base: que el
catálogo empaquetado y el registro no se desincronicen, y que el llamador no
pueda aportar los bytes de una definición.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.sembrar_definiciones_de_control import RUTA_CATALOGO, controles_empaquetados


def test_el_catalogo_empaquetado_y_el_registro_no_se_desincronizan():
    """Un control agregado al registro y NO al catálogo no se sembraría nunca.

    El síntoma sería el mismo que el defecto original (jax#266): la fila
    falta, `readonly_status_snapshot` levanta "snapshot definition mismatch"
    y CUALQUIER consulta de ese control sale `UNAVAILABLE`, sin decir por
    qué. Al revés -- un ID en el catálogo que el registro no conoce -- el
    script muere con `UnknownControlError` a mitad del despliegue.
    """
    from policy.enforcement_evidence.control_registry import _controls

    assert set(controles_empaquetados()) == set(_controls)


def test_el_catalogo_vacio_o_ilegible_falla_cerrado(tmp_path: Path):
    """Un catálogo sin controles no puede leerse como "no había nada que sembrar"."""
    vacio = tmp_path / "v1.json"
    vacio.write_text(json.dumps({"schema_version": "1.0", "controls": []}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="catálogo de controles"):
        controles_empaquetados(vacio)

    sin_clave = tmp_path / "sin-clave.json"
    sin_clave.write_text(json.dumps({"schema_version": "1.0"}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="catálogo de controles"):
        controles_empaquetados(sin_clave)


def test_la_ruta_por_defecto_del_catalogo_existe():
    """El default no es una ruta escrita de memoria: tiene que resolver a un archivo real."""
    assert RUTA_CATALOGO.is_file()


def test_el_sembrador_nombra_el_control_y_nunca_aporta_la_definicion():
    """La firma pública acepta un ID, no una `ControlDefinition`.

    Es la diferencia entre sembrar y poder instalar una definición ajena: los
    bytes salen de `load_control_definition` adentro del store, que es la
    única fuente que `require_trusted_definition` acepta. Si alguien cambiara
    la firma para recibir el objeto, este test avisa.
    """
    import inspect

    from policy.enforcement_evidence.mariadb_store import MariaDBEvidenceStore

    firma = inspect.signature(MariaDBEvidenceStore.persist_packaged_control_definition)
    assert list(firma.parameters) == ["self", "control_id", "control_version"]
