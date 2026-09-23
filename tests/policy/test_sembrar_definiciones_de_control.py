"""Controles PUROS de la siembra de definiciones (sin MariaDB).

La siembra contra una base real vive en
`tests/policy/test_execution_mariadb_integration.py`
(`test_sembrar_definiciones_de_control_es_idempotente`, que se para en un
esquema VIRGEN a propósito, y
`test_persistir_una_definicion_exige_el_contexto_de_composicion_fija`).
Acá va lo que se puede probar sin base.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.sembrar_definiciones_de_control import (RUTA_CATALOGO, VERSION_DEL_CATALOGO,
                                                     controles_empaquetados, sembrar)


def test_el_catalogo_empaquetado_y_el_registro_no_se_desincronizan():
    """Un control agregado al registro y NO al catálogo no se sembraría nunca.

    El síntoma sería el mismo que el defecto original (jax#266): la fila
    falta, `readonly_status_snapshot` levanta "snapshot definition mismatch"
    y CUALQUIER consulta de ese control sale `UNAVAILABLE`, sin decir por
    qué. Al revés -- un ID en el catálogo que el registro no conoce -- el
    script muere con `UnknownControlError` a mitad del despliegue.
    """
    from policy.enforcement_evidence.control_registry import _controls

    assert {control_id for control_id, _version in controles_empaquetados()} == set(_controls)


def test_el_catalogo_declara_la_version_que_este_script_asume():
    """`controls/v1.json` lista IDs pelados: la versión la pone el script.

    Mientras el catálogo sea `schema_version: "1.0"`, asumir la versión
    `VERSION_DEL_CATALOGO` no inventa nada -- `ControlDefinition.__post_init__`
    rechaza cualquier otra. El día que exista un v2, el catálogo tiene que
    decirlo y este test se pone rojo ANTES de que el sembrador ignore el
    control nuevo en silencio (que volvería a dar `UNAVAILABLE` sin causa
    visible).
    """
    from policy.enforcement_evidence.control_registry import _controls

    catalogo = json.loads(RUTA_CATALOGO.read_text(encoding="utf-8"))
    assert catalogo["schema_version"] == "1.0"
    # Los IDs del catálogo son strings pelados, SIN versión embebida. Es lo
    # que autoriza a `controles_empaquetados()` a ponerle VERSION_DEL_CATALOGO
    # a cada uno. (Revisión adversarial de jax#266, MINOR nuevo: comparar
    # `{d.control_version for d in _controls.values()} == {1}` era tautológico
    # -- `ControlDefinition.__post_init__` YA rechaza cualquier versión != 1,
    # así que ese assert no podía fallar por la vía que declaraba.)
    assert all(isinstance(entrada, str) for entrada in catalogo["controls"])
    assert not [entrada for entrada in catalogo["controls"] if "@" in entrada or "/v" in entrada], (
        "el catálogo empezó a embeber versiones en el ID: `controles_empaquetados()` "
        "las está ignorando y les pone VERSION_DEL_CATALOGO a todas")
    assert all(version == VERSION_DEL_CATALOGO for _cid, version in controles_empaquetados())
    assert _controls


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

    otra_version = tmp_path / "v2.json"
    otra_version.write_text(json.dumps({"schema_version": "2.0", "controls": ["RULE.P10"]}),
                            encoding="utf-8")
    with pytest.raises(RuntimeError, match="schema_version inesperada"):
        controles_empaquetados(otra_version)


def test_la_ruta_por_defecto_del_catalogo_existe():
    """El default no es una ruta escrita de memoria: tiene que resolver a un archivo real."""
    assert RUTA_CATALOGO.is_file()


def test_los_bytes_sembrados_salen_del_REGISTRO_y_no_del_llamador():
    """La propiedad de fondo, ejercitada: el llamador no puede aportar la definición.

    (Revisión adversarial de jax#266, MINOR-1: la versión anterior de este
    test sólo miraba `inspect.signature`, y un cuerpo reescrito para leer los
    bytes de otro lado la pasaba intacta.) Acá se usa un store de mentira que
    CAPTURA lo que recibe: lo que llega tiene que ser exactamente la instancia
    del registro -- la única que `require_trusted_definition` acepta, porque
    compara por identidad de objeto contra `_trusted`.
    """
    import policy.enforcement_evidence.control_registry as registro
    from policy.enforcement_evidence.control_registry import (is_trusted_control_definition,
                                                              require_trusted_definition)
    from policy.enforcement_evidence.errors import EvidenceBlobMissingError

    recibidas = []

    class StoreDeMentira:
        def load_control_definition(self, control_id, control_version=1):
            raise EvidenceBlobMissingError("sin fila")

        def _MariaDBEvidenceStore__persist_control_definition(self, definition):
            recibidas.append(definition)
            return definition

    # El sello de `_trusted` es GLOBAL y persiste entre tests: si otro test ya
    # cargó este control, `is_trusted_control_definition` da True sin que ESTA
    # siembra haya pasado por la API. Se vacía el sello a propósito para que el
    # único sellador posible sea el `load_control_definition` de `sembrar()`.
    # (Revisión adversarial de jax#266, MINOR nuevo: la versión anterior hacía
    # `assert recibidas[0] is load_control_definition(...)` y esa MISMA llamada
    # sellaba el objeto, con lo que el assert siguiente no podía fallar nunca.
    # Mutación que la pasaba en verde: `sembrar()` tomando la definición del
    # dict interno `_controls` en vez de la API que sella.)
    sello_previo = dict(registro._trusted)
    registro._trusted.clear()
    try:
        filas = sembrar(StoreDeMentira(), [("RULE.P10", 1)], emitir=lambda _linea: None)

        assert filas == [("RULE.P10", 1, False)]
        assert len(recibidas) == 1
        # Sellada por la propia siembra, con el sello vacío al empezar.
        assert is_trusted_control_definition(recibidas[0])
        require_trusted_definition(recibidas[0])   # el mismo guardia que usa el store
    finally:
        registro._trusted.update(sello_previo)


def test_un_control_desconocido_corta_la_siembra():
    """Un ID que el registro no conoce no se siembra ni a medias: corta ahí."""
    from policy.enforcement_evidence.errors import UnknownControlError

    from policy.enforcement_evidence.errors import EvidenceBlobMissingError

    class StoreQueNuncaDeberiaEscribir:
        def load_control_definition(self, control_id, control_version=1):
            raise EvidenceBlobMissingError("sin fila")

        def _MariaDBEvidenceStore__persist_control_definition(self, definition):
            raise AssertionError("no debería escribirse un control desconocido")

    with pytest.raises(UnknownControlError):
        sembrar(StoreQueNuncaDeberiaEscribir(), [("CTL.NO.EXISTE", 1)], emitir=lambda _l: None)
