import json
from pathlib import Path

import pytest

from procesamiento.ficha import Ficha, sha256_de


def test_sha256_de_un_archivo_conocido(tmp_path: Path):
    f = tmp_path / "x.txt"
    f.write_bytes(b"hola")
    # sha256("hola")
    assert sha256_de(f) == (
        "b221d9dbb083a7f33428d7c2a3c3198ae925614d70210e28716ccaa7cd4ddb79"
    )


def test_la_ficha_ida_y_vuelta_no_pierde_nada(tmp_path: Path):
    f = Ficha(
        sha256="abc",
        origen="fuente/x.xlsx",
        extractor="openpyxl",
        extractor_version="3.1.5",
        fecha="2026-09-20T23:14:00-06:00",
        estado="ok",
        detalle={"hojas": 6, "hojas_extraidas": 6},
    )
    assert Ficha.desde_json(f.a_json()) == f


def test_un_estado_inventado_se_rechaza():
    with pytest.raises(ValueError, match="estado inválido"):
        Ficha(
            sha256="a", origen="b", extractor="c", extractor_version="d",
            fecha="e", estado="casi_bien", detalle={},
        )


@pytest.mark.parametrize("campo", ["sha256", "origen", "extractor", "extractor_version", "fecha"])
def test_campos_de_procedencia_no_pueden_quedar_vacios(campo):
    """I-3: una `extractor_version=""` no permite decidir si hay que
    reprocesar, que es la única razón por la que ese campo existe."""
    kwargs = dict(
        sha256="a", origen="b", extractor="c", extractor_version="d",
        fecha="e", estado="ok", detalle={},
    )
    kwargs[campo] = "   "
    with pytest.raises(ValueError, match=campo):
        Ficha(**kwargs)


def test_detalle_con_tupla_no_sobrevive_el_viaje_a_json_y_se_rechaza():
    """I-6: el test de ida y vuelta sólo probaba dos enteros. Una tupla se
    convierte en lista JSON, así que Ficha.desde_json(f.a_json()) != f --
    se rechaza en la construcción, no tres capas después."""
    with pytest.raises(ValueError, match="no sobrevive"):
        Ficha(
            sha256="a", origen="b", extractor="c", extractor_version="d",
            fecha="e", estado="ok", detalle={"rango": (1, 5)},
        )


def test_detalle_con_clave_entera_no_sobrevive_y_se_rechaza():
    """I-6: json.dumps convierte la clave 1 en '1' -- distinto dict al volver."""
    with pytest.raises(ValueError, match="no sobrevive"):
        Ficha(
            sha256="a", origen="b", extractor="c", extractor_version="d",
            fecha="e", estado="ok", detalle={1: "uno"},
        )


def test_detalle_con_bytes_no_es_serializable_y_se_rechaza():
    """I-6: json.dumps no sabe serializar bytes -- TypeError capturado y
    reempaquetado como ValueError de dominio."""
    with pytest.raises(ValueError, match="serializable"):
        Ficha(
            sha256="a", origen="b", extractor="c", extractor_version="d",
            fecha="e", estado="ok", detalle={"v": b"x"},
        )


def test_mutar_detalle_de_ficha_tras_construir_falla():
    f = Ficha(
        sha256="a", origen="b", extractor="c", extractor_version="d",
        fecha="e", estado="ok", detalle={"hojas": 1},
    )
    with pytest.raises(TypeError):
        f.detalle["hojas"] = 2


def test_desde_json_con_clave_de_mas_se_rechaza():
    base = Ficha(
        sha256="a", origen="b", extractor="c", extractor_version="d",
        fecha="e", estado="ok", detalle={},
    )
    import json

    datos = json.loads(base.a_json())
    datos["campo_inventado"] = "x"
    with pytest.raises(ValueError, match="claves esperadas"):
        Ficha.desde_json(json.dumps(datos))


def test_desde_json_con_clave_faltante_se_rechaza():
    base = Ficha(
        sha256="a", origen="b", extractor="c", extractor_version="d",
        fecha="e", estado="ok", detalle={},
    )
    import json

    datos = json.loads(base.a_json())
    del datos["extractor_version"]
    with pytest.raises(ValueError, match="claves esperadas"):
        Ficha.desde_json(json.dumps(datos))


def test_desde_json_que_no_es_un_objeto_se_rechaza():
    """I-5: `[1,2,3]` es JSON válido pero no un objeto -- antes explotaba con
    un TypeError crudo de firma de constructor, no de dominio."""
    with pytest.raises(ValueError, match="objeto JSON"):
        Ficha.desde_json("[1,2,3]")


def test_detalle_se_serializa_con_default_dict():
    """I-3 (task-3-hallazgos.md): `.a_json()` ya envuelve `dict(self.detalle)`
    y no tiene el problema -- pero un consumidor que toque `.detalle`
    directamente (sin pasar por `a_json()`) se topa con el mismo
    `mappingproxy is not JSON serializable` que Resultado. `default=dict`
    lo resuelve sin tocar el tipo."""
    f = Ficha(
        sha256="a", origen="b", extractor="c", extractor_version="d",
        fecha="e", estado="ok", detalle={"hojas": 2, "hojas_extraidas": 2},
    )
    assert json.loads(json.dumps(f.detalle, default=dict)) == dict(f.detalle)
