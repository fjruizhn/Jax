import pytest
from procesamiento.resultado import Resultado


def test_un_error_no_puede_traer_salidas():
    """Fallo cerrado: si falló, NO hay extracto. Un extracto vacío dado por
    bueno hace que el modelo invente lo que no pudo leer (Principio VIII)."""
    with pytest.raises(ValueError, match="no puede traer salidas"):
        Resultado(
            estado="error", salidas={"a.csv": "x"},
            detalle={"razon": "roto"}, extractor="t", version="1",
        )


def test_ok_sin_salidas_tampoco_se_permite():
    with pytest.raises(ValueError, match="sin salidas"):
        Resultado(estado="ok", salidas={}, detalle={}, extractor="t", version="1")


def test_parcial_lleva_salidas_y_avisa():
    r = Resultado(
        estado="parcial", salidas={"h1.csv": "a,b"},
        detalle={"hojas": 6, "hojas_extraidas": 1}, extractor="t", version="1",
    )
    assert r.hubo_extracto is True
    assert r.detalle["hojas_extraidas"] < r.detalle["hojas"]
