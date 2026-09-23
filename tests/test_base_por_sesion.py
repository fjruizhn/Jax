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
import re
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
    env.pop("CI", None)
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
    assert r.returncode == 0, r.stderr
    resuelta = r.stdout.strip().splitlines()[-1]
    assert resuelta == f"{BASE_COMPARTIDA}_zz_control_sufijo", (
        f"con JAX_TEST_DB_SUFIJO puesto la suite resolvió {resuelta!r}: "
        f"sigue pisando la base compartida"
    )
    assert resuelta != BASE_COMPARTIDA


def test_sin_sufijo_fuera_de_ci_la_base_es_propia_y_no_la_compartida():
    """Decisión de Fernando, 2026-09-20. El choque medido ese día fue
    exactamente este default: dos worktrees sin `JAX_TEST_DB_SUFIJO` puesto a
    mano compartían `jax_memory_test` y se pisaron de verdad (filas en NULL,
    un test borrando la fila de otro). Fuera de CI, el aislamiento pasa a
    ser el comportamiento por defecto -- nadie tiene que acordarse de
    exportar nada."""
    r = _correr(CODIGO_QUE_RESUELVE_LA_BASE)
    assert r.returncode == 0, r.stderr
    resuelta = r.stdout.strip().splitlines()[-1]
    assert resuelta != BASE_COMPARTIDA, (
        f"sin JAX_TEST_DB_SUFIJO y fuera de CI, la suite resolvió {resuelta!r}: "
        f"sigue siendo la base compartida, que es el defecto que colisionó "
        f"el 2026-09-20."
    )
    assert resuelta.startswith(f"{BASE_COMPARTIDA}_")


def test_dos_procesos_sin_sufijo_se_aislan_entre_si():
    """El control central del choque del 2026-09-20: DOS procesos sin sufijo,
    cada uno resuelve una base DISTINTA. Antes de este arreglo los dos daban
    `jax_memory_test` -- la colisión medida ese mismo día entre worktrees."""
    r1 = _correr(CODIGO_QUE_RESUELVE_LA_BASE)
    r2 = _correr(CODIGO_QUE_RESUELVE_LA_BASE)
    assert r1.returncode == 0, r1.stderr
    assert r2.returncode == 0, r2.stderr
    base1 = r1.stdout.strip().splitlines()[-1]
    base2 = r2.stdout.strip().splitlines()[-1]
    assert base1 != base2, (
        f"dos procesos sin sufijo resolvieron la MISMA base ({base1!r}): "
        f"eso es la colisión, no el aislamiento."
    )


def test_sin_sufijo_en_ci_sigue_siendo_la_compartida():
    """Compatibilidad hacia atrás DELIBERADA para CI: cada job de
    `.github/workflows/policy.yml` que toca la base ya corre contra su propio
    contenedor MariaDB efímero (`services: mariadb:` por job) -- no hay
    sesiones concurrentes que se puedan pisar ahí, y sumar un sufijo metería
    un clonado de esquema de más en cada corrida sin comprar nada. `CI` es la
    variable que exportan GitHub Actions, GitLab CI y CircleCI por
    convención -- un hecho verificable, no una adivinanza."""
    r = _correr(CODIGO_QUE_RESUELVE_LA_BASE, CI="true")
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
    """El caso real: `set -a; . <(sudo -n cat /etc/jax/.env); set +a` deja
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


# ---------------------------------------------------------------------------
# La guarda vieja no vuelve (merge de prevuelo-y-continuar, 2026-09-18)
# ---------------------------------------------------------------------------

#: Los archivos de test que escriben en la base traían, desde antes de que
#: `JAX_TEST_DB_SUFIJO` existiera, su propia guarda con el nombre de la base
#: HARDCODEADO:
#:
#:     if _db and _db != "jax_memory_test": raise RuntimeError(...)
#:     os.environ.setdefault("JAX_DB_NAME", "jax_memory_test")
#:
#: o la variante `if not _db.endswith("_test"): raise`. Las dos rechazan
#: `jax_memory_test_<sufijo>`: fallan JUSTO cuando el mecanismo de base por
#: sesión funciona. Entraron siete archivos así con el merge de
#: `feat/prevuelo-y-continuar` (2026-09-18) y se migraron a
#: `exigir_base_de_test()`.
#:
#: Este control es el que impide que vuelvan. No mira "¿está el literal?" --
#: el literal es legítimo en un comentario o en `base_de_test.py` --: mira que
#: el literal esté DECIDIENDO (una comparación o un `setdefault` de
#: `JAX_DB_NAME`) fuera del módulo que es dueño de esa decisión.
_GUARDAS_VIEJAS = (
    re.compile(r'!=\s*["\']jax_memory_test["\']'),
    re.compile(r'\.endswith\(\s*["\']_test["\']\s*\)'),
    re.compile(r'os\.environ\.setdefault\(\s*["\']JAX_DB_NAME["\']'),
    re.compile(r'os\.environ\[\s*["\']JAX_DB_NAME["\']\s*\]\s*='),
)

#: Dueños legítimos de la decisión: el módulo del mecanismo, su propio test y
#: el conftest de la raíz (que la aplica para toda la suite).
_DUENIOS_DE_LA_DECISION = {"base_de_test.py", "test_base_por_sesion.py", "conftest.py"}


def _archivos_de_test():
    for patron in ("tests/test_*.py", "jacobs/_*_test.py",
                   "las_manos/**/_*_test.py", "las_manos/**/test_*.py"):
        yield from RAIZ.glob(patron)


def test_ningun_test_vuelve_a_hardcodear_el_nombre_de_la_base():
    """Validado por mutación: reponiendo la guarda vieja en cualquiera de los
    siete archivos migrados, este control se pone rojo y nombra el archivo y la
    línea."""
    hallazgos = []
    for f in _archivos_de_test():
        if f.name in _DUENIOS_DE_LA_DECISION:
            continue
        for n, linea in enumerate(f.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            desnuda = linea.split("#", 1)[0]
            if not desnuda.strip():
                continue
            for patron in _GUARDAS_VIEJAS:
                if patron.search(desnuda):
                    hallazgos.append(
                        f"{f.relative_to(RAIZ)}:{n}: {linea.strip()[:90]}")
                    break
    assert hallazgos == [], (
        "guarda vieja de la base de tests (rechaza jax_memory_test_<sufijo>): "
        "usar `exigir_base_de_test()` / `fijar_base_de_test()` de base_de_test.py\n"
        + "\n".join(hallazgos)
    )


# ---------------------------------------------------------------------------
# El borrado automático al salir (2026-09-20): el default de arriba crea una
# base nueva en CADA corrida local sin sufijo -- sin esto se acumulan más
# rápido de lo que se acumulaban antes, cuando exportar el sufijo era un paso
# que alguien pedía a propósito.
# ---------------------------------------------------------------------------

import asyncio  # noqa: E402

from base_de_test import (  # noqa: E402
    _borrar_al_salir,
    _dropear_base_de_sesion,
    _sufijo_automatico_de_sesion,
)


def test_el_borrado_al_salir_se_niega_a_tocar_produccion_y_la_compartida(monkeypatch):
    """El candado: intenta borrar `jax_memory` (producción), la compartida
    pelada, y un nombre cualquiera que no lleve el prefijo de test. Ninguno
    de los tres llega siquiera a abrir una conexión -- `aiomysql.connect`
    explota el test si algo lo intenta."""
    import aiomysql

    def _connect_prohibido(*_a, **_k):
        raise AssertionError("intentó conectar para borrar algo que no es una base de test")

    monkeypatch.setattr(aiomysql, "connect", _connect_prohibido)

    for nombre in (BASE_DE_PRODUCCION, BASE_COMPARTIDA, "otra_cosa_cualquiera"):
        asyncio.run(_dropear_base_de_sesion(nombre))  # no debe lanzar ni conectar


def test_el_borrado_al_salir_no_hace_nada_sin_jax_db_host(monkeypatch):
    monkeypatch.delenv("JAX_DB_HOST", raising=False)

    def _run_prohibido(*_a, **_k):
        raise AssertionError("no debería intentar correr nada sin JAX_DB_HOST")

    monkeypatch.setattr(asyncio, "run", _run_prohibido)
    _borrar_al_salir(f"{BASE_COMPARTIDA}_lo_que_sea")  # no debe lanzar


def test_el_sufijo_automatico_se_registra_para_borrarse_al_salir(monkeypatch):
    """Un sufijo AUTO-generado queda registrado en `atexit` para borrarse.
    Uno EXPLÍCITO (pasado a mano) NO se registra -- alguien pudo querer
    reusarlo entre corridas."""
    registrados = []
    monkeypatch.setattr(
        "base_de_test.atexit.register",
        lambda fn, *args: registrados.append((fn, args)),
    )
    sufijo = _sufijo_automatico_de_sesion()
    assert len(registrados) == 1
    fn, args = registrados[0]
    assert fn is _borrar_al_salir
    assert args == (f"{BASE_COMPARTIDA}_{sufijo}",)


# ---------------------------------------------------------------------------
# MINOR-3 (fix round 1 de Task 1-bis, 2026-09-22, Ruling 19e): clonar una
# tabla con una columna GENERATED y filas rompía con 1906 -- `visible`
# (Ruling 18, jacobs_pipelines) es la primera columna generada que pasa por
# `_clonar_esquema()`, y la plantilla compartida (`jax_memory_test`) SÍ
# puede tener filas en esa tabla. `_columnas_copiables()`/`_copiar_filas()`
# arreglan esto con una lista EXPLÍCITA de columnas (sin las GENERATED), no
# `SELECT *`. Se prueba contra la base de la SESIÓN (no `jax_memory_test`,
# la plantilla real, para no tocarla) con dos tablas propias, desechables:
# origen (con datos) y destino (vacía, misma forma) en una SEGUNDA base
# creada y borrada por el propio test.
# ---------------------------------------------------------------------------

import uuid as _uuid  # noqa: E402

from base_de_test import _columnas_copiables, _copiar_filas  # noqa: E402


@pytest.mark.skipif(not os.environ.get("JAX_DB_HOST"), reason="necesita la MariaDB real")
def test_copiar_filas_excluye_columnas_generadas_y_no_revienta_con_1906():
    async def _cuerpo():
        from jacobs import store

        origen = nombre_base_de_test()  # la base de ESTA sesión, ya existe
        destino = f"{origen}_clon{_uuid.uuid4().hex[:8]}"
        tabla = f"_diag_gen_{_uuid.uuid4().hex[:8]}"
        async with store.conexion() as conn:
            async with conn.cursor() as cur:
                await cur.execute(f"CREATE DATABASE `{destino}`")
                try:
                    ddl = (
                        f"CREATE TABLE `{{esquema}}`.`{tabla}` ("
                        "id INT PRIMARY KEY, status VARCHAR(20) NOT NULL, "
                        "visible TINYINT(1) GENERATED ALWAYS AS "
                        "(status NOT IN ('discarded','hidden')) VIRTUAL)"
                    )
                    await cur.execute(ddl.format(esquema=origen))
                    await cur.execute(ddl.format(esquema=destino))
                    await cur.executemany(
                        f"INSERT INTO `{origen}`.`{tabla}` (id, status) VALUES (%s,%s)",
                        [(1, "completed"), (2, "discarded"), (3, "running")],
                    )
                    # El paso que rompía: `SELECT *`/`INSERT ... SELECT *`
                    # incluiría `visible` -- 1906. `_copiar_filas` no.
                    await _copiar_filas(cur, origen, destino, tabla)
                    await cur.execute(
                        f"SELECT id, status, visible FROM `{destino}`.`{tabla}` ORDER BY id"
                    )
                    filas = await cur.fetchall()
                finally:
                    await cur.execute(f"DROP TABLE IF EXISTS `{origen}`.`{tabla}`")
                    await cur.execute(f"DROP DATABASE IF EXISTS `{destino}`")
            await conn.commit()
        return filas

    filas = asyncio.run(_cuerpo())
    assert list(filas) == [(1, "completed", 1), (2, "discarded", 0), (3, "running", 1)], (
        "las filas copiadas no coinciden -- o no llegaron, o `visible` no se "
        "recalculó igual en la tabla destino"
    )


@pytest.mark.skipif(not os.environ.get("JAX_DB_HOST"), reason="necesita la MariaDB real")
def test_columnas_copiables_excluye_solo_las_generadas():
    async def _cuerpo():
        from jacobs import store

        origen = nombre_base_de_test()
        tabla = f"_diag_cols_{_uuid.uuid4().hex[:8]}"
        async with store.conexion() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"CREATE TABLE `{origen}`.`{tabla}` ("
                    "id INT PRIMARY KEY, status VARCHAR(20) NOT NULL, "
                    "visible TINYINT(1) GENERATED ALWAYS AS "
                    "(status NOT IN ('discarded','hidden')) VIRTUAL)"
                )
                try:
                    return await _columnas_copiables(cur, origen, tabla)
                finally:
                    await cur.execute(f"DROP TABLE IF EXISTS `{origen}`.`{tabla}`")
            await conn.commit()

    columnas = asyncio.run(_cuerpo())
    assert columnas == ["id", "status"]
