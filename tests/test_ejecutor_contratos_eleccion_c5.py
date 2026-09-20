# tests/test_ejecutor_contratos_eleccion_c5.py
"""Quién puede auditar a quién: nunca el mismo proveedor; datos de clientes sólo con
auditor local o con la compuerta abierta por DECISIÓN de Fernando."""
import pytest

from jax.ejecutor.contratos import eleccion_c5 as E
from jax.ejecutor.contratos.fallo import Fallo

FILAS = {"ejecutor.cerebro_faceta": "ejecutor", "ejecutor.auditor_faceta": "thot",
         "ejecutor.auditor_faceta_local": "auditor_local", "ejecutor.c5_lote_max": "20",
         "ejecutor.c5_intervalo_s": "15", "ejecutor.c5_max_tokens": "4000",
         "ejecutor.c5_auditor_admite_datos_de_clientes": "false",
         # Compuerta del mismo proveedor (2026-09-20): obligatoria y CERRADA, que es
         # como nace. Estos tests siguen midiendo el comportamiento estricto.
         "ejecutor.c5_auditor_admite_mismo_proveedor": "false"}


def test_config_desde_filas():
    assert E.config_desde_filas(FILAS) == E.ConfigC5("ejecutor", "thot", "auditor_local", 20, 15.0, 4000,
                                                    False, False)


@pytest.mark.parametrize("clave, valor", [("ejecutor.c5_lote_max", "0"), ("ejecutor.c5_intervalo_s", "x"),
                                          ("ejecutor.c5_intervalo_s", "nan"), ("ejecutor.c5_intervalo_s", "inf"),
                                          ("ejecutor.c5_max_tokens", "-1"),
                                          ("ejecutor.c5_auditor_admite_datos_de_clientes", "si"),
                                          ("ejecutor.c5_auditor_admite_datos_de_clientes", "True"),
                                          ("ejecutor.auditor_faceta", ""),
                                          ("ejecutor.auditor_faceta_local", "")])
def test_config_invalida(clave, valor):
    with pytest.raises(ValueError):
        E.config_desde_filas({**FILAS, clave: valor})


def test_config_incompleta():
    with pytest.raises(ValueError):
        E.config_desde_filas({k: v for k, v in FILAS.items() if k != "ejecutor.c5_max_tokens"})


def _validar(**cambios):
    base = dict(proveedor_cerebro="ollama", proveedor_auditor="openai", auditor_es_local=False,
                admite_datos_de_clientes=False, hosts_mision=frozenset({"hall9000"}),
                hosts_con_clientes=frozenset({"bridge"}), hosts_conocidos=frozenset({"hall9000", "bridge"}))
    return E.validar_eleccion(**{**base, **cambios})


def test_pareja_admisible():
    assert _validar() == ()


def test_mismo_proveedor():
    # El Fallo ahora lleva datos (el proveedor y la compuerta que lo permitiria),
    # asi que se compara por codigo: es lo que este test siempre quiso afirmar.
    assert "auditor_mismo_proveedor_que_el_cerebro" in [
        f.codigo for f in _validar(proveedor_auditor="ollama")]


def test_proveedor_vacio_no_cuenta_como_distinto():
    assert Fallo("c5", "proveedor_desconocido") in _validar(proveedor_auditor="")


def test_datos_de_clientes_con_auditor_de_nube_y_compuerta_cerrada():
    assert _validar(hosts_mision=frozenset({"bridge"})) == (
        Fallo("c5", "auditor_no_admite_datos_de_clientes", (("hosts", ("bridge",)),)),)


def test_host_desconocido_cuenta_como_con_clientes():
    assert _validar(hosts_mision=frozenset({"nueva"}))[0].codigo == "auditor_no_admite_datos_de_clientes"


def test_mision_sin_hosts_no_arranca():
    assert Fallo("c5", "mision_sin_maquinas") in _validar(hosts_mision=frozenset())


@pytest.mark.parametrize("cambios", [{"auditor_es_local": True}, {"admite_datos_de_clientes": True}])
def test_compuerta_abierta_o_auditor_local(cambios):
    assert _validar(hosts_mision=frozenset({"bridge"}), **cambios) == ()


def test_validar_proveedores_sin_mision():
    """El arranque sin misión (plan 6) sólo mira quién produce y quién aprueba."""
    assert E.validar_proveedores(proveedor_cerebro="ollama", proveedor_auditor="openai") == ()
    assert [f.codigo for f in E.validar_proveedores(proveedor_cerebro="ollama", proveedor_auditor="ollama")] == [
        "auditor_mismo_proveedor_que_el_cerebro"]
    assert E.validar_proveedores(proveedor_cerebro="", proveedor_auditor="openai") == (
        Fallo("c5", "proveedor_desconocido"),)


# --- elección del auditor según la máquina de la misión (spec 2026-09-18) -------------------

def test_elegir_auditor_faceta_segun_datos_de_clientes():
    """Decisión de Fernando: la máquina de la misión decide el auditor, no una clave global.
    Sin nombre de faceta hardcodeado -- los dos salen de axioma_config vía ConfigC5."""
    cfg = E.config_desde_filas(FILAS)
    assert E.elegir_auditor_faceta(cfg, hay_datos_de_clientes=True) == "auditor_local"
    assert E.elegir_auditor_faceta(cfg, hay_datos_de_clientes=False) == "thot"


def test_sensibles_son_las_con_datos_o_desconocidas():
    """Mismo hecho que gobierna la compuerta (validar_eleccion): con datos de clientes, o
    fuera del inventario/dada de baja (cuenta como con datos, cerrado)."""
    assert E.sensibles(frozenset({"hall9000", "bridge", "nueva"}), frozenset({"bridge"}),
                       frozenset({"hall9000", "bridge"})) == frozenset({"bridge", "nueva"})
    assert E.sensibles(frozenset({"hall9000"}), frozenset(), frozenset({"hall9000"})) == frozenset()
