"""Tripwire: un test que CREA un pipeline necesita la gobernanza, y tiene que
decir POR CUÁL DE LAS DOS SALIDAS la consigue.

Desde que crear un pipeline pasa por el pre-vuelo (`jacobs/routes.py` llama a
`jacobs.prevuelo.prevuelo`), la creación lee `facet`, `model`, `capability` y
`credential`: tablas que crean las migraciones de jax-platform y que
`store.init_tables()` de este repo NO crea. El mismo día (2026-09-17) eso rompió
tests por dos caminos distintos y a dos personas:

  * `tests/test_subpipeline_contrato_rutas.py` cayó en CI con
    "Table 'jax_memory_test.facet' doesn't exist".
  * `jacobs/_cupo_io_test.py::CreacionConcurrenteSinCandadoTest` cayó con
    "0 != 3" por lo mismo.

Las dos veces el arreglo fue legítimo, pero cada una eligió una salida distinta:
sembrar el esquema real en el job (clonar jax-platform y correr
`run_migrations()`) o sustituir el pre-vuelo con un doble en el archivo. El
defecto de fondo es el tercer caso: un test que crea pipelines SIN ninguna de las
dos pasa o falla según lo que haya sembrado otro job en la base compartida. Eso
no es un verde: es un estado que el test no fija.

Este tripwire es PURO (no toca la base; corre en `tests-puros`). Lee por AST
todos los archivos de test del repo, detecta los que crean un pipeline por el
camino real —`routes.create_pipeline(...)`, un `POST /jacobs/pipeline`, o un
ayudante del repo que haga una de esas dos (así lo hace
`tests/test_subpipeline_contrato_rutas.py`, que crea por `jacobs/_arnes_ada.py`)—
y exige una de las dos salidas.

LÍMITE DECLARADO: el detector es sintáctico. Una creación por un callee dinámico
(`getattr(cliente, metodo)(ruta)`, como en `tests/test_las_manos_auth_servicio.py`)
no se ve. Cubre la forma que el repo usa de verdad, no toda forma imaginable.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest
import yaml

RAIZ = Path(__file__).resolve().parents[1]
FLUJO = RAIZ / ".github" / "workflows" / "policy.yml"

RUTA_CREACION = "/jacobs/pipeline"
NOMBRE_CREACION = "create_pipeline"
MARCA_ESQUEMA = "run_migrations"

MENSAJE = (
    "Este test crea un pipeline y el pre-vuelo necesita la gobernanza "
    "(facet/model/capability/credential, que init_tables() no crea): o el job "
    "que lo corre trae el esquema de jax-platform (paso run_migrations), o el "
    "archivo sustituye el pre-vuelo con un doble. Sin una de las dos, el test "
    "pasa o falla según lo que haya sembrado otro job."
)


# ---------------------------------------------------------------------------
# Inventario de archivos
# ---------------------------------------------------------------------------

def _es_archivo_de_test(ruta: Path) -> bool:
    nombre = ruta.name
    return nombre.startswith("test_") or nombre.endswith("_test.py")


def archivos_de_test() -> list[Path]:
    """`tests/*.py`, `jacobs/*_test.py`, `las_manos/*_test.py` y subdirectorios."""
    vistos: dict[Path, None] = {}
    for patron in ("tests/**/*.py", "jacobs/**/*_test.py", "las_manos/**/*_test.py"):
        for ruta in sorted(RAIZ.glob(patron)):
            if ruta.is_file() and _es_archivo_de_test(ruta):
                vistos[ruta] = None
    return list(vistos)


def _modulos_del_repo() -> list[Path]:
    """Los módulos NO-test donde puede vivir un ayudante que cree pipelines."""
    vistos: dict[Path, None] = {}
    for patron in ("jacobs/**/*.py", "las_manos/**/*.py", "tests/**/*.py"):
        for ruta in sorted(RAIZ.glob(patron)):
            if ruta.is_file() and not _es_archivo_de_test(ruta):
                vistos[ruta] = None
    return list(vistos)


def _punteado(ruta: Path) -> str:
    relativa = ruta.relative_to(RAIZ).with_suffix("")
    return ".".join(relativa.parts)


def _arbol(ruta: Path) -> ast.AST | None:
    texto = ruta.read_text(encoding="utf-8", errors="replace")
    try:
        return ast.parse(texto, filename=str(ruta))
    except SyntaxError:
        return None


# ---------------------------------------------------------------------------
# Detección de la creación
# ---------------------------------------------------------------------------

def _nombre_llamado(nodo: ast.Call) -> str | None:
    f = nodo.func
    if isinstance(f, ast.Attribute):
        return f.attr
    if isinstance(f, ast.Name):
        return f.id
    return None


def _primer_texto(nodo: ast.Call) -> str | None:
    if nodo.args and isinstance(nodo.args[0], ast.Constant) and isinstance(nodo.args[0].value, str):
        return nodo.args[0].value
    return None


def creacion_directa(arbol: ast.AST) -> list[tuple[int, str]]:
    """Llamadas que crean un pipeline por el camino real, con línea y forma."""
    hallazgos: list[tuple[int, str]] = []
    for nodo in ast.walk(arbol):
        if not isinstance(nodo, ast.Call):
            continue
        nombre = _nombre_llamado(nodo)
        if nombre == NOMBRE_CREACION:
            hallazgos.append((nodo.lineno, "create_pipeline(...)"))
            continue
        if nombre == "post":
            texto = _primer_texto(nodo)
            if texto is not None and texto.rstrip("/") == RUTA_CREACION:
                hallazgos.append((nodo.lineno, f"POST {RUTA_CREACION}"))
    return sorted(hallazgos)


def ayudantes_que_crean() -> set[tuple[str, str]]:
    """(módulo punteado, función) de los ayudantes del repo que crean pipelines.

    `tests/test_subpipeline_contrato_rutas.py` no llama a `create_pipeline`: pide
    hijos con `ada.pedir_hijo()`, y ese ayudante (`jacobs/_arnes_ada.py`) sí
    llama. Sin este paso el tripwire no vería el archivo que originó la regla.
    """
    ayudantes: set[tuple[str, str]] = set()
    for ruta in _modulos_del_repo():
        arbol = _arbol(ruta)
        if arbol is None:
            continue
        modulo = _punteado(ruta)
        for nodo in ast.walk(arbol):
            if isinstance(nodo, (ast.FunctionDef, ast.AsyncFunctionDef)) and creacion_directa(nodo):
                ayudantes.add((modulo, nodo.name))
    return ayudantes


def _alias_de_modulos(arbol: ast.AST) -> dict[str, str]:
    """alias local -> módulo punteado, según los imports del archivo."""
    alias: dict[str, str] = {}
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Import):
            for nombre in nodo.names:
                alias[nombre.asname or nombre.name.split(".")[0]] = nombre.name
        elif isinstance(nodo, ast.ImportFrom) and nodo.module and not nodo.level:
            for nombre in nodo.names:
                alias[nombre.asname or nombre.name] = f"{nodo.module}.{nombre.name}"
    return alias


def _simbolos_importados(arbol: ast.AST) -> dict[str, tuple[str, str]]:
    """nombre local -> (módulo, símbolo), para `from x import y`."""
    simbolos: dict[str, tuple[str, str]] = {}
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.ImportFrom) and nodo.module and not nodo.level:
            for nombre in nodo.names:
                simbolos[nombre.asname or nombre.name] = (nodo.module, nombre.name)
    return simbolos


def creaciones(arbol: ast.AST, ayudantes: set[tuple[str, str]]) -> list[tuple[int, str]]:
    """Toda creación de pipeline visible en el archivo: directa o por ayudante."""
    hallazgos = list(creacion_directa(arbol))
    alias = _alias_de_modulos(arbol)
    simbolos = _simbolos_importados(arbol)
    for nodo in ast.walk(arbol):
        if not isinstance(nodo, ast.Call):
            continue
        f = nodo.func
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
            modulo = alias.get(f.value.id)
            if modulo and (modulo, f.attr) in ayudantes:
                hallazgos.append((nodo.lineno, f"{f.value.id}.{f.attr}(...) -> {modulo}"))
        elif isinstance(f, ast.Name) and f.id in simbolos:
            if simbolos[f.id] in ayudantes:
                modulo, simbolo = simbolos[f.id]
                hallazgos.append((nodo.lineno, f"{f.id}(...) -> {modulo}.{simbolo}"))
    return sorted(set(hallazgos))


# ---------------------------------------------------------------------------
# Salida (a): doble del pre-vuelo
# ---------------------------------------------------------------------------

_SUSTITUTOS = {"setattr", "patch", "object"}  # monkeypatch.setattr, patch, patch.object


def sustituye_el_prevuelo(arbol: ast.AST) -> tuple[int, str] | None:
    """El archivo cambia `prevuelo` (o `_prevuelo_o_503`) por un doble."""
    for nodo in ast.walk(arbol):
        if not isinstance(nodo, ast.Call):
            continue
        nombre = _nombre_llamado(nodo)
        if nombre not in _SUSTITUTOS:
            continue
        for argumento in nodo.args:
            if isinstance(argumento, ast.Constant) and isinstance(argumento.value, str):
                if "prevuelo" in argumento.value:
                    return (nodo.lineno, argumento.value)
    return None


# ---------------------------------------------------------------------------
# Salida (b): el job trae el esquema real
# ---------------------------------------------------------------------------

def sin_comentarios(texto: str) -> str:
    """Un `run:` de este flujo trae bloques enteros de comentario de shell (los
    pisos se documentan ahí). Encontrado al escribir este tripwire: el comentario
    que agregué al job `tests-puros` nombraba `run_migrations` y tres archivos, y
    el job PURO pasó a contar como "trae el esquema". Un control que se deja
    convencer por un comentario no es un control.
    """
    return "\n".join(linea for linea in texto.splitlines()
                     if not linea.lstrip().startswith("#"))


def _textos_del_paso(paso) -> list[str]:
    if not isinstance(paso, dict):
        return []
    textos = []
    for clave in ("name", "run", "uses"):
        valor = paso.get(clave)
        if isinstance(valor, str):
            textos.append(sin_comentarios(valor))
    return textos


def jobs_con_esquema(flujo: dict) -> dict[str, str]:
    """job -> texto de sus pasos, para los jobs que corren `run_migrations`."""
    resultado: dict[str, str] = {}
    for nombre, job in (flujo.get("jobs") or {}).items():
        if not isinstance(job, dict):
            continue
        textos = [t for paso in (job.get("steps") or []) for t in _textos_del_paso(paso)]
        junto = "\n".join(textos)
        if MARCA_ESQUEMA in junto:
            resultado[nombre] = junto
    return resultado


def _flujo() -> dict:
    return yaml.safe_load(FLUJO.read_text(encoding="utf-8"))


def job_que_trae_el_esquema(relativa: str, jobs: dict[str, str]) -> str | None:
    for nombre, texto in jobs.items():
        if relativa in texto:
            return nombre
    return None


# ---------------------------------------------------------------------------
# El tripwire
# ---------------------------------------------------------------------------

def _informe() -> tuple[dict[str, tuple[int, str]], dict[str, frozenset[str]], list[str]]:
    """(archivo -> primera creación), (archivo -> salidas) y los infractores."""
    ayudantes = ayudantes_que_crean()
    jobs = jobs_con_esquema(_flujo())
    detectados: dict[str, tuple[int, str]] = {}
    salidas: dict[str, frozenset[str]] = {}
    infractores: list[str] = []
    for ruta in archivos_de_test():
        arbol = _arbol(ruta)
        if arbol is None:
            continue
        encontradas = creaciones(arbol, ayudantes)
        if not encontradas:
            continue
        relativa = ruta.relative_to(RAIZ).as_posix()
        detectados[relativa] = encontradas[0]
        tiene: set[str] = set()
        if sustituye_el_prevuelo(arbol) is not None:
            tiene.add("doble")
        if job_que_trae_el_esquema(relativa, jobs) is not None:
            tiene.add("esquema")
        salidas[relativa] = frozenset(tiene)
        if tiene:
            continue
        linea, forma = encontradas[0]
        infractores.append(f"{relativa}:{linea} ({forma})")
    return detectados, salidas, infractores


def test_todo_test_que_crea_un_pipeline_declara_su_salida():
    """El tripwire. Rojo visto el 2026-09-17 con un archivo de prueba que creaba
    un pipeline sin ninguna de las dos salidas: lo nombró con archivo y línea.
    """
    _detectados, _salidas, infractores = _informe()
    assert not infractores, MENSAJE + "\nArchivos sin salida:\n  " + "\n  ".join(infractores)


def test_el_mensaje_nombra_las_DOS_salidas():
    """Si el mensaje pierde una de las dos, el que lo lea arregla por la otra."""
    for parte in ("run_migrations", "jax-platform", "doble", "sustituye el pre-vuelo",
                  "facet/model/capability/credential", "init_tables()", "otro job"):
        assert parte in MENSAJE, parte


def test_el_detector_no_esta_ciego_hoy():
    """Un detector que no detecta nada da verde por vacío. Los cuatro archivos
    que crean pipelines hoy tienen que seguir viéndose."""
    detectados, _salidas, _infractores = _informe()
    for esperado in ("tests/test_jacobs_preflight_endpoint.py",
                     "tests/test_jacobs_conexiones_por_pedido.py",
                     "tests/test_subpipeline_contrato_puro.py",
                     "tests/test_subpipeline_contrato_rutas.py",
                     "jacobs/_cupo_io_test.py"):
        assert esperado in detectados, (esperado, sorted(detectados))


def test_el_inventario_de_salidas_de_hoy_no_cambia_en_silencio():
    """Qué archivo cumple con cuál salida, medido el 2026-09-17. Fija el mapa
    entero: si alguien retira el `run_migrations` de un job, o el doble de un
    archivo, acá se ve ANTES de que el verde empiece a depender del otro job.
    Los dos de la base tienen las DOS salidas -- por eso el tripwire principal no
    se cae si se rompe el reconocimiento del esquema, y por eso existe este test.
    """
    _detectados, salidas, _infractores = _informe()
    esperado = {
        "jacobs/_cupo_io_test.py": {"doble", "esquema"},
        "tests/test_jacobs_conexiones_por_pedido.py": {"doble"},
        "tests/test_jacobs_preflight_endpoint.py": {"doble"},
        "tests/test_subpipeline_contrato_puro.py": {"doble"},
        "tests/test_subpipeline_contrato_rutas.py": {"doble", "esquema"},
    }
    assert {k: set(v) for k, v in salidas.items()} == esperado


def test_el_ayudante_del_arnes_cuenta_como_creacion():
    """`tests/test_subpipeline_contrato_rutas.py` crea por `ada.pedir_hijo()`.
    Sin seguir un salto, el archivo que originó la regla quedaría invisible."""
    assert ("jacobs._arnes_ada", "pedir_hijo") in ayudantes_que_crean()


# ---------------------------------------------------------------------------
# El detector se verifica a sí mismo (la herramienta también se verifica)
# ---------------------------------------------------------------------------

_CREA_DIRECTO = """
from jacobs import routes
async def t():
    await routes.create_pipeline(req, bg)
"""

_CREA_POR_HTTP = """
def t(cliente):
    r = cliente.post("/jacobs/pipeline", json={})
"""

_NO_CREA = """
import inspect
from jacobs import routes
def t():
    fuente = inspect.getsource(routes.create_pipeline)
    doc = routes.create_pipeline.__doc__
    cliente.post("/jacobs/pipeline/p1/resume", json={})
"""


@pytest.mark.parametrize("fuente", [_CREA_DIRECTO, _CREA_POR_HTTP])
def test_el_detector_ve_las_dos_formas_directas(fuente):
    assert creacion_directa(ast.parse(fuente))


def test_el_detector_no_confunde_una_referencia_con_una_creacion():
    """`inspect.getsource(routes.create_pipeline)` y `POST .../resume` no crean."""
    assert creacion_directa(ast.parse(_NO_CREA)) == []


@pytest.mark.parametrize("fuente", [
    'from jacobs import routes\ndef f(monkeypatch):\n    monkeypatch.setattr(routes, "prevuelo", x)\n',
    'def f():\n    with patch.object(routes, "_prevuelo_o_503", y):\n        pass\n',
    'def f():\n    with patch("jacobs.routes.prevuelo", y):\n        pass\n',
])
def test_reconoce_el_doble_del_prevuelo(fuente):
    assert sustituye_el_prevuelo(ast.parse(fuente)) is not None


def test_no_toma_cualquier_parche_por_un_doble_del_prevuelo():
    fuente = 'def f(monkeypatch):\n    monkeypatch.setattr(routes, "cupo", x)\n'
    assert sustituye_el_prevuelo(ast.parse(fuente)) is None


def test_reconoce_el_job_que_trae_el_esquema():
    flujo = yaml.safe_load("""
jobs:
  con-esquema:
    steps:
      - name: Esquema de gobernanza con SUS migraciones
        run: python -c "run_migrations()"
      - run: python -m pytest tests/test_x.py
  sin-esquema:
    steps:
      - run: python -m pytest tests/test_y.py
""")
    jobs = jobs_con_esquema(flujo)
    assert job_que_trae_el_esquema("tests/test_x.py", jobs) == "con-esquema"
    assert job_que_trae_el_esquema("tests/test_y.py", jobs) is None


def test_un_comentario_no_alcanza_para_traer_el_esquema():
    """Rojo visto el 2026-09-17: el comentario del piso en `tests-puros` nombra
    `run_migrations` y archivos, y sin este recorte el job puro contaba como si
    trajera la gobernanza -- tres archivos quedaban "cubiertos" por un texto."""
    flujo = yaml.safe_load("""
jobs:
  solo-lo-comenta:
    steps:
      - run: |
          python -m pytest tests/test_x.py
          # antes esto corria run_migrations; ya no
""")
    assert jobs_con_esquema(flujo) == {}
    assert "run_migrations" not in sin_comentarios("  # corre run_migrations\npytest")


def test_el_job_puro_no_cuenta_como_portador_del_esquema():
    """`tests-puros` no clona jax-platform: si alguna vez cuenta como portador,
    el tripwire está tapando justo el caso que vino a vigilar."""
    assert "tests-puros" not in jobs_con_esquema(_flujo())


def test_el_job_real_de_la_base_sigue_trayendo_el_esquema():
    """Si alguien saca el paso `run_migrations` del job de sub-pipelines, la
    salida (b) desaparece sin que nadie lo note: acá se nota."""
    jobs = jobs_con_esquema(_flujo())
    assert job_que_trae_el_esquema("tests/test_subpipeline_contrato_rutas.py", jobs) is not None
    assert job_que_trae_el_esquema("jacobs/_cupo_io_test.py", jobs) is not None
