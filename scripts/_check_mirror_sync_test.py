"""
Tests del comparador de espejos (`scripts/check_mirror_sync.py`).

POR QUE EXISTEN. El checker se probo rompiendolo a mano el 2026-09-01 y esa
prueba encontro un defecto REAL: el mecanismo de "divergencia deliberada" **no
funcionaba para constantes de modulo**. El marcador se declara en el docstring,
y una constante no tiene docstring -- `ast.get_source_segment` de un `Assign`
devuelve la sentencia pelada, sin los comentarios de alrededor. O sea que
`FACET_SEAL_PATH`, agregado a la comparacion ese mismo dia, no se podia declarar
como divergencia deliberada de ninguna forma.

Una verificacion a mano que encuentra un defecto y no queda escrita es una
verificacion que hay que volver a hacer. Estos tests son esa prueba, hecha
permanente. Corren sobre fuentes SINTETICAS -- no sobre los archivos reales --
asi que fijan la LOGICA del comparador y no el estado de los espejos de hoy.

Corre con:
  python3 -m pytest scripts/_check_mirror_sync_test.py -v

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_mirror_sync import MARCADOR, Familia, _extract, revisar  # noqa: E402

COMPARTIDOS = ("UNA_CONSTANTE", "una_funcion", "UnaClase")

BASE = '''\
UNA_CONSTANTE = "original"


def una_funcion():
    """Docstring original."""
    return 1


class UnaClase:
    """Docstring original."""
'''


def _escribir(tmp_path: Path, nombre: str, texto: str) -> Path:
    ruta = tmp_path / nombre
    ruta.write_text(texto, encoding="utf-8")
    return ruta


def _familia(tmp_path: Path, canonico: str, espejo: str) -> Familia:
    return Familia(
        nombre="prueba",
        canonico=_escribir(tmp_path, "canonico.py", canonico),
        espejos=(("espejo", _escribir(tmp_path, "espejo.py", espejo)),),
        compartidos=COMPARTIDOS,
    )


# ---------------------------------------------------------------------------
# Lo basico
# ---------------------------------------------------------------------------

def test_copias_identicas_no_reportan_nada(tmp_path):
    drift, declaradas, faltantes = revisar(_familia(tmp_path, BASE, BASE))
    assert (drift, declaradas, faltantes) == ([], [], [])


def test_un_simbolo_que_falta_es_drift(tmp_path):
    """Fail-closed: que falte NO es lo mismo que que coincida."""
    incompleto = BASE.replace('UNA_CONSTANTE = "original"\n', "")
    drift, declaradas, faltantes = revisar(_familia(tmp_path, BASE, incompleto))
    assert any("UNA_CONSTANTE" in f for f in faltantes), faltantes


def test_los_comentarios_no_cuentan_como_drift(tmp_path):
    """El segmento comparado es el del AST: un comentario distinto en una copia
    no es una divergencia de comportamiento y no debe gritar."""
    con_comentario = "# un comentario que solo esta en el espejo\n" + BASE
    drift, _, faltantes = revisar(_familia(tmp_path, BASE, con_comentario))
    assert (drift, faltantes) == ([], [])


# ---------------------------------------------------------------------------
# El marcador, en las tres clases de simbolo
# ---------------------------------------------------------------------------

def test_funcion_que_diverge_sin_declarar_es_drift(tmp_path):
    otra = BASE.replace("return 1", "return 2")
    drift, declaradas, _ = revisar(_familia(tmp_path, BASE, otra))
    assert any("una_funcion" in d for d in drift), drift
    assert declaradas == []


def test_funcion_declarada_en_el_docstring_no_es_drift(tmp_path):
    otra = BASE.replace('"""Docstring original."""\n    return 1',
                        f'"""{MARCADOR}: razon escrita."""\n    return 2')
    drift, declaradas, _ = revisar(_familia(tmp_path, BASE, otra))
    assert drift == []
    assert any("una_funcion" in d for d in declaradas), declaradas


def test_constante_que_diverge_sin_declarar_es_drift(tmp_path):
    otra = BASE.replace('UNA_CONSTANTE = "original"', 'UNA_CONSTANTE = "otra"')
    drift, declaradas, _ = revisar(_familia(tmp_path, BASE, otra))
    assert any("UNA_CONSTANTE" in d for d in drift), drift


def test_constante_declarada_en_el_comentario_de_ARRIBA_no_es_drift(tmp_path):
    """EL DEFECTO REAL QUE ESTE ARCHIVO EXISTE PARA FIJAR.

    Una constante no tiene docstring. Antes del 2026-09-01 el marcador se
    buscaba solo dentro del segmento del AST, asi que una divergencia declarada
    sobre una constante se reportaba igual como DRIFT -- el mecanismo de
    declaracion no existia para esa clase de simbolo."""
    otra = BASE.replace(
        'UNA_CONSTANTE = "original"',
        f'# {MARCADOR}: razon escrita aca arriba.\nUNA_CONSTANTE = "otra"',
    )
    drift, declaradas, _ = revisar(_familia(tmp_path, BASE, otra))
    assert drift == [], drift
    assert any("UNA_CONSTANTE" in d for d in declaradas), declaradas


def test_constante_declarada_en_la_MISMA_LINEA_no_es_drift(tmp_path):
    """La otra forma natural de escribirlo. Tambien fallaba."""
    otra = BASE.replace(
        'UNA_CONSTANTE = "original"',
        f'UNA_CONSTANTE = "otra"  # {MARCADOR}: razon escrita.',
    )
    drift, declaradas, _ = revisar(_familia(tmp_path, BASE, otra))
    assert drift == [], drift
    assert any("UNA_CONSTANTE" in d for d in declaradas), declaradas


def test_un_comentario_SIN_marcador_no_declara_nada(tmp_path):
    """El arreglo de arriba no puede aflojar el detector: cualquier comentario
    no vale, tiene que estar el marcador."""
    otra = BASE.replace(
        'UNA_CONSTANTE = "original"',
        '# un comentario cualquiera, sin marcador\nUNA_CONSTANTE = "otra"',
    )
    drift, declaradas, _ = revisar(_familia(tmp_path, BASE, otra))
    assert declaradas == []
    assert any("UNA_CONSTANTE" in d for d in drift), drift


def test_el_marcador_de_OTRO_simbolo_no_declara_este(tmp_path):
    """El bloque declarativo es el de ESE nodo, no todo el archivo. Un marcador
    puesto en otro simbolo no puede tapar un drift ajeno."""
    otra = BASE.replace(
        "def una_funcion():", f"# {MARCADOR}: esto declara la FUNCION\ndef una_funcion():"
    ).replace('UNA_CONSTANTE = "original"', 'UNA_CONSTANTE = "otra"')
    drift, declaradas, _ = revisar(_familia(tmp_path, BASE, otra))
    assert any("UNA_CONSTANTE" in d for d in drift), drift


def test_alcanza_con_que_lo_declare_una_de_las_dos_copias(tmp_path):
    """La razon puede estar escrita del lado que tiene el campo extra -- que es
    como esta hoy en facet_resolver (el marcador vive en jax-platform)."""
    canonico = BASE.replace(
        'UNA_CONSTANTE = "original"',
        f'# {MARCADOR}: la razon vive en el canonico.\nUNA_CONSTANTE = "original"',
    )
    otra = BASE.replace('UNA_CONSTANTE = "original"', 'UNA_CONSTANTE = "otra"')
    drift, declaradas, _ = revisar(_familia(tmp_path, canonico, otra))
    assert drift == []
    assert any("UNA_CONSTANTE" in d for d in declaradas), declaradas


# ---------------------------------------------------------------------------
# Varios espejos
# ---------------------------------------------------------------------------

def test_compara_TODOS_los_espejos_no_solo_el_primero(tmp_path):
    """credential_resolver tiene TRES archivos reales (las_manos no es symlink,
    medido 2026-09-01). Un comparador que se quedara en el primer espejo
    dejaria al tercero sin vigilancia -- que es la situacion que habia."""
    sano = _escribir(tmp_path, "espejo_sano.py", BASE)
    roto = _escribir(tmp_path, "espejo_roto.py",
                     BASE.replace('UNA_CONSTANTE = "original"', 'UNA_CONSTANTE = "otra"'))
    familia = Familia(
        nombre="prueba",
        canonico=_escribir(tmp_path, "canonico.py", BASE),
        espejos=(("sano", sano), ("roto", roto)),
        compartidos=COMPARTIDOS,
    )
    drift, _, _ = revisar(familia)
    assert any("roto" in d for d in drift), drift
    assert not any("sano" in d for d in drift), drift


def test_el_reporte_dice_QUE_espejo_diverge(tmp_path):
    """Con tres copias, "difiere" sin decir cual no sirve para arreglarlo."""
    otra = BASE.replace('UNA_CONSTANTE = "original"', 'UNA_CONSTANTE = "otra"')
    drift, _, _ = revisar(_familia(tmp_path, BASE, otra))
    assert drift == ["UNA_CONSTANTE (espejo)"], drift


def test_extract_ignora_lo_que_no_esta_declarado_como_compartido(tmp_path):
    extra = BASE + '\n\nSOLO_EN_UNO = "no se compara"\n'
    encontrados = _extract(_escribir(tmp_path, "x.py", extra), COMPARTIDOS)
    assert set(encontrados) == set(COMPARTIDOS)
