"""La suite de `jax` no deja temporales propios en la raíz de /tmp.

POR QUÉ EXISTE (2026-10-07, encargo de Hyde). Los temporales de pruebas que
necesitan `dir="/tmp"` (la jaula real del Faro, los git-trust de las voces)
porque OTRO uid tiene que poder atravesarlos se limpiaban con
`rmtree(..., ignore_errors=True)` -- y un fallo de borrado ahí es invisible:
quedaba un `faro-real-*` o un `permisos-nucleo-*` huérfano por corrida, y el
propio `conftest.py` dejaba SEIS (`jax-test-*`) porque nadie los borraba
nunca. En el runner de CI eso se acumula corrida tras corrida.

El freno vive en el `conftest.py` de la raíz, igual que la barrera del
respaldo de uso: `pytest_sessionfinish` cuenta las entradas de /tmp contra
la foto tomada al importar (antes de cualquier test) y pone la corrida en
roja si queda una con prefijo de la suite. Sólo se vigilan los prefijos
propios (`PREFIJOS_TMP_PROPIOS`): /tmp es compartido con todo el host y un
contador sin filtro daría roja por ruido ajeno.

Y el freno no alcanza con declararlo: acá se ejercita el detector contra un
huérfano de mentira y se vigila la LISTA -- cada `mkdtemp`/`mkstemp`/
`TemporaryDirectory` de tests/ con prefijo tiene que estar cubierto, para
que el próximo temporal nuevo no nazca invisible al contador.

Corre con:
  PYTHONPATH=.:las_manos python -m pytest tests/test_higiene_tmp.py -v
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import conftest

RAIZ = Path(__file__).resolve().parents[1]

#: (mkdtemp|mkstemp|TemporaryDirectory|_temporal_de_sesion)(prefix="..." | "...").
#: Sólo cuenta el PRIMER argumento cuando es el prefijo (posicional o
#: `prefix=`): un `mkstemp(dir=...)` manda el temporal a otro lado (al árbol
#: de prueba, que ya se borra con él) y no es un fijo de /tmp raíz.
_PREFIJO_EN_FUENTE = re.compile(
    r"""(?:mkdtemp|mkstemp|TemporaryDirectory|_temporal_de_sesion)
        \(\s*(?:prefix\s*=\s*)?["']([A-Za-z0-9_.-]+)["']""",
    re.VERBOSE,
)


def test_el_detector_atrapa_un_huerfano_propio():
    """Un control que no falla cuando debe no valida nada: un temporal con
    prefijo de la suite que no estaba en la foto inicial, cuenta."""
    antes = frozenset({"un-archivo-ajeno", "faro-real-de-una-corr-ida-anterior"})
    ahora = list(antes | {"faro-real-abc123", "permisos-arbol-xyz"})
    assert conftest.fugaces_de_tmp(antes, ahora) == ["faro-real-abc123", "permisos-arbol-xyz"]


def test_el_detector_no_cuenta_lo_ajeno_ni_lo_que_ya_estaba():
    """Lo que ya estaba al iniciar la sesión no lo escribió esta corrida, y lo
    que no lleva prefijo propio (pytest, docker, el SO, otras sesiones) no es
    de esta suite: contarlos pondría el freno en roja permanente."""
    antes = frozenset({"faro-real-de-otra-corr-ida", "tmp-ajeno"})
    ahora = list(antes | {"pytest-of-fruiz", "docker.sock", "otra-cosa-0"})
    assert conftest.fugaces_de_tmp(antes, ahora) == []


def test_todo_prefijo_de_tests_esta_vigilado():
    """La lista es lo que el contador ve: un mkdtemp o TemporaryDirectory de
    tests/ con un prefijo que no esté cubierto por `PREFIJOS_TMP_PROPIOS`
    nacería invisible al contador. Se escanea el fuente de tests/ y del
    conftest raíz, no la memoria de quien escribió el test. Cubierto = que
    EMPIECE por un prefijo vigilado (así `jax-test-audit-log-` queda cubierto
    por `jax-test-`)."""
    descubiertos = {
        prefijo
        for fuente in [RAIZ / "conftest.py", *sorted((RAIZ / "tests").rglob("*.py"))]
        for prefijo in _PREFIJO_EN_FUENTE.findall(fuente.read_text(encoding="utf-8"))
    }
    sin_vigilar = sorted(
        prefijo for prefijo in descubiertos
        if not any(prefijo.startswith(vigilado) for vigilado in conftest.PREFIJOS_TMP_PROPIOS)
    )
    assert sin_vigilar == [], (
        f"prefijos de temporales que el contador de /tmp NO vigila: {sin_vigilar}. "
        f"Añádelos a conftest.PREFIJOS_TMP_PROPIOS."
    )


def test_NINGUN_temporal_quedo_huerfano_en_tmp():
    """El chequeo real, en vivo. Se restan los seis de aislamiento del conftest
    (`_temporal_de_sesion`): ésos se borran en `pytest_sessionfinish`,
    DESPUÉS de este test -- no son huérfanos, es su hora de borrarlos la que
    todavía no llegó. Se repite el chequeo completo (sin la resta) en
    `pytest_sessionfinish` (conftest): acá depende del orden de colección,
    allá no."""
    ahora = [
        entrada
        for directorio in conftest._DIRECTORIOS_DE_CONTEO
        for entrada in os.listdir(directorio)
    ]
    vigentes = {Path(ruta).name for ruta in conftest._TEMPORALES_DE_SESION}
    fugaces = [
        nombre for nombre in conftest.fugaces_de_tmp(conftest.ENTRADAS_TMP_AL_INICIO, ahora)
        if nombre not in vigentes
    ]
    assert fugaces == [], f"{len(fugaces)} temporal(es) propio(s) sin borrar en /tmp: {fugaces}"
