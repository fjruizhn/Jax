"""El conftest de la raíz aísla `JAX_FACET_SEAL_PATH` del sello REAL de
producción `/srv/jax-data/facet-cache-seal`.

POR QUÉ EXISTE. `jax/core/facet_resolver.py` y `las_manos/facet_resolver.py`
leen `JAX_FACET_SEAL_PATH` en el MÓDULO (tiempo de import):

    FACET_SEAL_PATH = os.getenv("JAX_FACET_SEAL_PATH", "/srv/jax-data/facet-cache-seal")

Ese default es el sello REAL que jax-platform y LAS MANOS usan para invalidar
el caché de facetas. El conftest de este repo NO fijaba `JAX_FACET_SEAL_PATH`
(a diferencia de `JAX_USAGE_SPOOL_DIR`, `LAS_MANOS_URL`, etc., que sí se
aíslan más arriba en este mismo archivo): correr localmente cualquier test
que ejercite una invalidación de facetas o una migración que la dispare
escribía el mtime del archivo de producción. Un agente lo tocó el
2026-09-17 a la 01:51:21 (dato de Fernando, no reverificado acá).

CÓMO SE VERIFICA. Un `monkeypatch.setenv` o un fixture no prueban nada: el
módulo ya leyó la variable al importarse, antes de que corra un solo test
de este archivo -- si este proceso de pytest ya cargó el conftest de la
raíz, `facet_resolver.FACET_SEAL_PATH` en ESTE proceso ya está aislado sin
importar si el fix existe o no. Para probar el caso real (alguien exporta
`JAX_FACET_SEAL_PATH=/srv/jax-data/facet-cache-seal`, por ejemplo con el
`.env` de producción sourceado, ANTES de invocar pytest) hace falta un
subproceso limpio: se controla el entorno de ese subproceso, se generan un
archivo de test mínimo en un directorio temporal que sólo lee el valor
efectivo del módulo, y se corre `python -m pytest` sobre él con
`PYTEST_PLUGINS=conftest` -- que es lo que hace que el conftest de la raíz
se cargue como plugin aun cuando el archivo bajo prueba vive fuera del
árbol del repo (verificado: sin esto, un archivo en /tmp no ve el conftest
de la raíz en absoluto y el test no probaría el aislamiento real).

Corre con:
  PYTHONPATH=.:las_manos python -m pytest tests/test_conftest_aisla_facet_seal.py -v
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import conftest as _conftest_raiz

REPO_ROOT = Path(__file__).resolve().parent.parent
RUTA_PRODUCCION = "/srv/jax-data/facet-cache-seal"

TEST_MINIMO = textwrap.dedent(
    """\
    import jax.core.facet_resolver as fr

    def test_seal_efectivo_no_es_produccion():
        assert not fr.FACET_SEAL_PATH != "/srv/jax-data/facet-cache-seal", (
            f"FACET_SEAL_PATH={fr.FACET_SEAL_PATH!r} es la ruta de PRODUCCION"
        )
    """
)


def _correr_pytest_minimo(tmp_path: Path, env_extra: dict) -> subprocess.CompletedProcess:
    """Lanza un `python -m pytest` en un subproceso limpio contra un archivo
    de test mínimo generado en `tmp_path`, con el conftest de la raíz
    forzado como plugin vía `PYTEST_PLUGINS` (así se prueba el aislamiento
    real, no el efecto ya aplicado a ESTE proceso de pytest)."""
    archivo = tmp_path / "test_seal_efectivo.py"
    archivo.write_text(TEST_MINIMO, encoding="utf-8")

    env = dict(os.environ)
    env.update(env_extra)
    env["PYTHONPATH"] = f"{REPO_ROOT}:{REPO_ROOT / 'las_manos'}"
    env["PYTEST_PLUGINS"] = "conftest"

    return subprocess.run(
        [sys.executable, "-m", "pytest", str(archivo), "-q"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_sin_la_variable_el_conftest_fija_una_ruta_no_productiva(tmp_path):
    """`JAX_FACET_SEAL_PATH` ausente en el entorno: el conftest la fija de
    todas formas, antes de que `facet_resolver` la lea al importarse."""
    env_extra = dict(os.environ)
    env_extra.pop("JAX_FACET_SEAL_PATH", None)
    resultado = _correr_pytest_minimo(tmp_path, env_extra)
    assert resultado.returncode == 0, resultado.stdout + resultado.stderr


def test_con_la_variable_apuntando_a_produccion_el_conftest_la_pisa(tmp_path):
    """Alguien exportó `JAX_FACET_SEAL_PATH=/srv/jax-data/facet-cache-seal`
    antes de invocar pytest. El conftest tiene que pisarla igual: es
    asignación directa, no `setdefault` (lección del proyecto: `setdefault`
    no pisa un `.env` ya sourceado)."""
    resultado = _correr_pytest_minimo(tmp_path, {"JAX_FACET_SEAL_PATH": RUTA_PRODUCCION})
    assert resultado.returncode == 0, resultado.stdout + resultado.stderr


def test_el_conftest_usa_asignacion_directa_no_setdefault():
    """El comportamiento ya lo prueba
    `test_con_la_variable_apuntando_a_produccion_el_conftest_la_pisa` (un
    `setdefault` la haría fallar, porque no pisaría la variable ya seteada).
    Este test deja el criterio explícito en el código fuente, no sólo en el
    efecto observado, para que un cambio futuro que vuelva a `setdefault`
    (por ejemplo, "para no pisar lo que puso quien invoca") se vea en el
    diff de este archivo y no dependa de que alguien recuerde la lección."""
    fuente = Path(_conftest_raiz.__file__).read_text(encoding="utf-8")
    assert 'os.environ["JAX_FACET_SEAL_PATH"] =' in fuente
    assert 'os.environ.setdefault("JAX_FACET_SEAL_PATH"' not in fuente
