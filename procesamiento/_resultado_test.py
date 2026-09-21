import json

import pytest

from procesamiento.resultado import Resultado


def test_error_no_puede_traer_salidas():
    """Fallo cerrado: si falló, NO hay extracto. Un extracto vacío dado por
    bueno hace que el modelo invente lo que no pudo leer (Principio VIII)."""
    with pytest.raises(ValueError, match="no puede traer salidas"):
        Resultado(
            estado="error", salidas={"a.csv": "x"},
            detalle={"razon": "roto"}, extractor="t", version="1",
        )


def test_sin_extractor_con_salidas_se_rechaza():
    """C-3 #1/#3: la regla de fallo cerrado dice 'error' y 'sin_extractor' --
    antes ningún test construía un Resultado con estado='sin_extractor', así
    que quitar ese estado de la regla no lo detectaba nadie."""
    with pytest.raises(ValueError, match="no puede traer salidas"):
        Resultado(
            estado="sin_extractor", salidas={"a.csv": "x"},
            detalle={}, extractor="t", version="1",
        )


def test_ok_sin_salidas_no_se_permite():
    with pytest.raises(ValueError, match="sin salidas"):
        Resultado(estado="ok", salidas={}, detalle={}, extractor="t", version="1")


def test_parcial_vacio_no_se_permite():
    """C-3 #4: no había ningún test de 'parcial' vacío -- sólo el de 'ok'."""
    with pytest.raises(ValueError, match="sin salidas"):
        Resultado(estado="parcial", salidas={}, detalle={}, extractor="t", version="1")


def test_ok_con_todas_las_salidas_vacias_se_rechaza():
    """I-1: un extracto vacío pasando como 'ok' es justo el escenario que el
    contrato existe para impedir."""
    with pytest.raises(ValueError, match="contenido"):
        Resultado(
            estado="ok", salidas={"h1.csv": "", "h2.csv": "   "},
            detalle={}, extractor="t", version="1",
        )


def test_ok_con_una_salida_vacia_y_otra_con_contenido_es_valido():
    """I-1, ruling del controlador: exigir que TODAS tengan contenido rompería
    un caso legítimo -- un libro de Excel con una hoja en blanco entre seis es
    un archivo válido. La regla es 'al menos una con contenido'."""
    r = Resultado(
        estado="ok", salidas={"h1.csv": "a,b", "h2_vacia.csv": ""},
        detalle={}, extractor="t", version="1",
    )
    assert r.hubo_extracto is True


def test_parcial_lleva_salidas_y_avisa():
    r = Resultado(
        estado="parcial", salidas={"h1.csv": "a,b"},
        detalle={"hojas": 6, "hojas_extraidas": 1}, extractor="t", version="1",
    )
    assert r.hubo_extracto is True
    assert r.detalle["hojas_extraidas"] < r.detalle["hojas"]


def test_hubo_extracto_en_false_con_sin_extractor():
    """C-3 #5: el único test previo de hubo_extracto sólo ejercitaba el True
    -- una mutación que devolviera True siempre quedaba en verde."""
    r = Resultado(
        estado="sin_extractor", salidas={}, detalle={"motivo": "sin driver"},
        extractor="t", version="1",
    )
    assert r.hubo_extracto is False


def test_estado_inventado_se_rechaza():
    """C-3 #2: nada probaba que ESTADOS siguiera siendo el único catálogo
    válido -- agregar un quinto estado inventado quedaba en verde."""
    with pytest.raises(ValueError, match="estado inválido"):
        Resultado(estado="mas_o_menos", salidas={}, detalle={}, extractor="t", version="1")


@pytest.mark.parametrize("salidas_invalidas", [["a.csv"], "a.csv"])
def test_salidas_que_no_son_un_mapa_se_rechazan(salidas_invalidas):
    """I-2: una lista o un string se aceptaban hoy y reventaban lejos del
    origen, o se leían como basura en silencio."""
    with pytest.raises(ValueError, match="mapa"):
        Resultado(
            estado="ok", salidas=salidas_invalidas, detalle={},
            extractor="t", version="1",
        )


def test_extractor_vacio_se_rechaza():
    with pytest.raises(ValueError, match="extractor"):
        Resultado(
            estado="ok", salidas={"a.csv": "x"}, detalle={}, extractor="", version="1",
        )


def test_version_vacia_se_rechaza():
    with pytest.raises(ValueError, match="version"):
        Resultado(
            estado="ok", salidas={"a.csv": "x"}, detalle={}, extractor="t", version="   ",
        )


def test_mutar_salidas_tras_construir_falla():
    r = Resultado(estado="ok", salidas={"a.csv": "x"}, detalle={}, extractor="t", version="1")
    with pytest.raises(TypeError):
        r.salidas["b.csv"] = "y"


def test_mutar_el_dict_original_no_afecta_al_resultado():
    """C-1/C-2: frozen congela el nombre del campo, no el dict que apunta --
    sin copia defensiva, esto pasaba: 'ok' quedaba vacío por atrás."""
    d = {"a.csv": "x"}
    r = Resultado(estado="ok", salidas=d, detalle={}, extractor="t", version="1")
    d.clear()
    assert r.salidas == {"a.csv": "x"}


def test_detalle_y_salidas_se_serializan_con_default_dict():
    """I-3 (task-3-hallazgos.md): el endurecimiento con MappingProxyType
    cerró la mutabilidad y rompió la serialización -- `json.dumps(r.detalle)`
    a secas revienta con `TypeError: Object of type mappingproxy is not JSON
    serializable`. No hay que tocar el tipo (no sabe qué va a escribirlo a
    disco, ni tiene por qué): quien serialice pasa `default=dict`."""
    r = Resultado(
        estado="parcial", salidas={"01-a.csv": "x"},
        detalle={"hojas": 2, "hojas_extraidas": 1, "no_tabulares": ["grafico"]},
        extractor="t", version="1",
    )
    assert json.loads(json.dumps(r.detalle, default=dict)) == dict(r.detalle)
    assert json.loads(json.dumps(r.salidas, default=dict)) == dict(r.salidas)
