"""La base de tests es propia de cada sesión (decisión de Fernando, 2026-09-17).

Estos controles fallan contra el código viejo: antes del 2026-09-17 el nombre
de la base estaba escrito como literal en ~30 archivos, `JAX_TEST_DB_SUFIJO`
no existía, y tres sesiones de trabajo simultáneas compartían
`jax_memory_test`. (El nombre del asistente NO se escribe en
este archivo, a propósito: la política del repo marca cualquier archivo que
lance subprocesos y lo mencione, y este lanza subprocesos. Me marcó a mí el
2026-09-17; está en `policy/tests/`.)

El control central (`test_el_sufijo_manda_en_la_base_que_usan_los_tests`)
corre en un SUBPROCESO a propósito: lo que se prueba es la resolución en
tiempo de IMPORT -- el conftest y el módulo de test fijando `JAX_DB_NAME`
antes de que nada se conecte. En el proceso de pytest eso ya pasó y no se
puede volver a ejercitar.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from base_de_test import (
    BASE_COMPARTIDA,
    BASE_DE_PRODUCCION,
    BaseDeTestInvalida,
    es_base_de_test,
    exigir_base_de_test,
    nombre_base_de_test,
)

RAIZ = Path(__file__).resolve().parents[1]

#: Un módulo de test que fija la base por su cuenta, como todos: se importa en
#: el subproceso para comprobar que NO vuelve a la base compartida.
MODULO_QUE_FIJA_LA_BASE = RAIZ / "tests" / "test_jacobs_interruptor.py"


def _correr(codigo: str, **entorno: str) -> subprocess.CompletedProcess:
    """Corre `codigo` en un intérprete nuevo, con la raíz del repo en el path
    y SIN `JAX_DB_HOST` (así nada intenta crear ni tocar una base)."""
    env = dict(os.environ)
    env.pop("JAX_DB_HOST", None)
    env.pop("JAX_DB_NAME", None)
    env.pop("JAX_TEST_DB_SUFIJO", None)
    # las_manos/ también: los módulos de test importan `interruptor` por su
    # nombre corto, como corren en producción (igual que el CI).
    env["PYTHONPATH"] = f"{RAIZ}:{RAIZ / 'las_manos'}"
    env.update(entorno)
    return subprocess.run([sys.executable, "-c", codigo], cwd=RAIZ, env=env,
                          capture_output=True, text=True, timeout=180)


# ---------------------------------------------------------------- control 1
# Con el sufijo puesto, la base que van a usar los tests es la del sufijo.

CODIGO_QUE_RESUELVE_LA_BASE = """
import importlib.util, os, sys
import conftest                      # el mismo camino que corre pytest
spec = importlib.util.spec_from_file_location("_modulo_de_prueba", sys.argv[1] if len(sys.argv) > 1 else %r)
modulo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(modulo)      # el módulo fija la base por su cuenta
print(os.environ["JAX_DB_NAME"])
""" % str(MODULO_QUE_FIJA_LA_BASE)


def test_el_sufijo_manda_en_la_base_que_usan_los_tests():
    r = _correr(CODIGO_QUE_RESUELVE_LA_BASE, JAX_TEST_DB_SUFIJO="zz_control_sufijo")
    assert not r.returncode == 0, r.stderr
    resuelta = r.stdout.strip().splitlines()[-1]
    assert resuelta == f"{BASE_COMPARTIDA}_zz_control_sufijo", (
        f"con JAX_TEST_DB_SUFIJO puesto la suite resolvió {resuelta!r}: "
        f"sigue pisando la base compartida"
    )
    assert resuelta != BASE_COMPARTIDA


def test_sin_sufijo_la_base_sigue_siendo_la_compartida():
    """Compatibilidad hacia atrás DELIBERADA: el CI no exporta sufijo y no
    tiene que cambiar, y las ramas abiertas no se rompen."""
    r = _correr(CODIGO_QUE_RESUELVE_LA_BASE)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip().splitlines()[-1] == BASE_COMPARTIDA


# ---------------------------------------------------------------- control 2
# Un sufijo inválido es un error explícito, NO una caída a la compartida.

SUFIJOS_INVALIDOS = [
    "",                    # vacío: quien exportó la variable quiso base propia
    "Ronda",               # mayúsculas
    "con-guion",           # guion medio
    "con espacio",
    "punto.coma",
    "drop; drop database jax_memory",
    "acentuada_ñ",
    "x" * 80,              # pasa los 64 del identificador
]


@pytest.mark.parametrize("sufijo", SUFIJOS_INVALIDOS)
def test_sufijo_invalido_es_error_y_no_fallback(sufijo):
    with pytest.raises(BaseDeTestInvalida):
        nombre_base_de_test(sufijo)


@pytest.mark.parametrize("sufijo", ["", "con-guion", "drop; drop database jax_memory"])
def test_sufijo_invalido_corta_al_arrancar(sufijo):
    """Y corta de verdad al importar el conftest: no es una excepción que
    alguien atrapa más tarde."""
    r = _correr(CODIGO_QUE_RESUELVE_LA_BASE, JAX_TEST_DB_SUFIJO=sufijo)
    assert r.returncode != 0, (
        f"con JAX_TEST_DB_SUFIJO={sufijo!r} la suite arrancó igual "
        f"(salida: {r.stdout.strip()!r}) -- eso es el fallback silencioso"
    )
    assert "JAX_TEST_DB_SUFIJO" in r.stderr
    assert BASE_COMPARTIDA not in r.stdout


# ---------------------------------------------------------------- control 3
# Nunca, por ningún camino, la base de producción.

def test_nunca_resuelve_a_produccion():
    assert nombre_base_de_test() != BASE_DE_PRODUCCION
    assert nombre_base_de_test("loquesea") != BASE_DE_PRODUCCION
    # El sufijo que armaría el nombre de producción no existe: no hay sufijo
    # `s` tal que `jax_memory_test_<s>` sea `jax_memory`.
    assert not es_base_de_test(BASE_DE_PRODUCCION)
    assert not es_base_de_test("jax_memory_prod")
    assert not es_base_de_test("jax_memoryx_test")


def test_una_base_de_produccion_ya_exportada_es_error(monkeypatch):
    """El caso real: `set -a; . /etc/jax/.env; set +a` deja
    JAX_DB_NAME=jax_memory. Un test que escribe filas no puede correr así."""
    monkeypatch.setenv("JAX_DB_NAME", BASE_DE_PRODUCCION)
    with pytest.raises(BaseDeTestInvalida):
        exigir_base_de_test()
    assert os.environ["JAX_DB_NAME"] == BASE_DE_PRODUCCION  # no lo pisó en silencio


def test_el_verificador_de_produccion_no_se_puede_esquivar():
    from base_de_test import _verificar_que_no_es_produccion
    with pytest.raises(BaseDeTestInvalida):
        _verificar_que_no_es_produccion(BASE_DE_PRODUCCION)
    with pytest.raises(BaseDeTestInvalida):
        _verificar_que_no_es_produccion("otra_base_cualquiera")
    assert _verificar_que_no_es_produccion(BASE_COMPARTIDA) == BASE_COMPARTIDA


# ---------------------------------------------------------------- el limpiador

def test_el_limpiador_se_niega_a_tocar_produccion_y_la_compartida():
    sys.path.insert(0, str(RAIZ / "scripts"))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_limpiar_bases_de_test", RAIZ / "scripts" / "limpiar_bases_de_test.py")
    limpiar = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(limpiar)

    assert not limpiar._es_borrable(BASE_DE_PRODUCCION)
    assert not limpiar._es_borrable(BASE_COMPARTIDA)
    assert not limpiar._es_borrable("jax_memory_test_CON_MAYUSCULAS")
    assert not limpiar._es_borrable("otra_cosa")
    assert limpiar._es_borrable(f"{BASE_COMPARTIDA}_ronda_hyde")


def test_el_limpiador_no_borra_sin_el_flag_explicito():
    """El dry-run es el default: `--dias 0` sin `--borrar-de-verdad` no puede
    borrar nada. Se comprueba sobre el parser, sin tocar la MariaDB."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_limpiar_bases_de_test", RAIZ / "scripts" / "limpiar_bases_de_test.py")
    limpiar = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(limpiar)

    fuente = (RAIZ / "scripts" / "limpiar_bases_de_test.py").read_text(encoding="utf-8")
    assert '"--borrar-de-verdad", action="store_true"' in fuente
    assert "if not args.borrar_de_verdad:" in fuente
    # el DROP sólo se emite dentro de _borrar(), que revalida cada nombre
    assert fuente.count('cur.execute(f"DROP DATABASE') == 1
    assert "if not _es_borrable(nombre):" in fuente
