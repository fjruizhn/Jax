from pathlib import Path
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
    import pytest

    with pytest.raises(ValueError, match="estado inválido"):
        Ficha(
            sha256="a", origen="b", extractor="c", extractor_version="d",
            fecha="e", estado="casi_bien", detalle={},
        )
