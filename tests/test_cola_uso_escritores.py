"""Los dos escritores de `axioma_usage` del repo `jax` encolan en vez de perder
la fila (plan 2026-09-15-cola-durable-uso, Task 7).

Hasta hoy, cuando la base no estaba, la fila se perdia para siempre:
`jacobs/usage_writer.py` logueaba ERROR y seguia, y
`las_manos/motor_registry/usage_writer.py` reintentaba y despues logueaba ERROR.
El turno ya se le cobro al proveedor, asi que esa fila es dinero real que el
total de Admin -> Costos nunca vuelve a ver.

Estos tests NO tocan la base: la simulan caida (el pool compartido,
`jacobs.store.conexion`, que explota al pedir la conexion -- desde el
2026-09-17 los dos escritores ya no abren `aiomysql.connect` propio) y miran el
directorio del respaldo. `aiomysql.connect` queda armado para fallar el test si
alguien vuelve a abrir una conexion suelta. `JAX_USAGE_SPOOL_DIR` apunta a un `tmp_path`
en cada test -- nunca al default de produccion.

Corre con:
  PYTHONPATH=.:las_manos python -m pytest tests/test_cola_uso_escritores.py -v
"""
from __future__ import annotations

import ast
import asyncio
import contextlib
import json
import os
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parent.parent
CANONICO_EN_JAX = RAIZ / "jax" / "core" / "cola_uso.py"
ESPEJO_EN_LAS_MANOS = RAIZ / "las_manos" / "cola_uso.py"

#: Los TRECE del contrato (Task 8, 2026-09-15: `status` y `job_id` pasaron a
#: viajar en el archivo). Se escriben aca a mano, no se importan de `cola_uso`:
#: importarlos haria que el test dijera "el archivo tiene los campos que el
#: modulo dice tener" -- verde aunque el contrato cambiara solo de un lado.
CAMPOS_DEL_CONTRATO = {
    "spool_id", "created_at", "tenant_id", "user_id", "facet", "model",
    "tokens_in", "tokens_out", "cost_usd", "request_type", "origen",
    "status", "job_id",
}

# La API que SOLO puede usar la plataforma: es la duena de `axioma_usage` y la
# unica con migraciones (la columna `spool_id` y su UNIQUE, que es lo que hace
# idempotente al reintento). Si un proceso de jax drenara, dos procesos
# borrarian el mismo archivo sin coordinacion.
API_DE_DRENAJE = ("leer_pendientes", "quitar", "drenar", "_leer_lote", "_borrar")

ESCRITORES = (
    RAIZ / "jacobs" / "usage_writer.py",
    RAIZ / "las_manos" / "motor_registry" / "usage_writer.py",
)


# ---------------------------------------------------------------------------
# La copia
# ---------------------------------------------------------------------------

def test_el_modulo_existe_en_jax_core():
    assert CANONICO_EN_JAX.is_file(), CANONICO_EN_JAX


def test_las_manos_lo_ve_por_symlink():
    """Mismo patron que redaccion.py / db_connect_config.py / facet_resolver.py:
    la copia vive en jax/core y las_manos la ve por symlink, porque LAS MANOS
    corre con cwd=las_manos y ahi `jax.core` no es importable."""
    assert ESPEJO_EN_LAS_MANOS.is_symlink(), ESPEJO_EN_LAS_MANOS
    assert ESPEJO_EN_LAS_MANOS.resolve() == CANONICO_EN_JAX.resolve()


def test_la_copia_importa_SOLO_biblioteca_estandar():
    """La copia es literal entre dos repos con venvs distintos. Una dependencia
    de terceros la vuelve incopiable y rompe el contrato en silencio."""
    arbol = ast.parse(CANONICO_EN_JAX.read_text(encoding="utf-8"))
    modulos = set()
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Import):
            modulos.update(a.name.split(".")[0] for a in nodo.names)
        elif isinstance(nodo, ast.ImportFrom) and nodo.level == 0 and nodo.module:
            modulos.add(nodo.module.split(".")[0])
    de_terceros = sorted(m for m in modulos if m not in sys.stdlib_module_names)
    assert de_terceros == [], de_terceros


def test_la_familia_de_espejos_esta_declarada_y_COMPLETA():
    """Fail-closed contra el modo de falla real de este repo: una familia
    declarada a medias deja simbolos fuera de la comparacion y el checker da
    verde sin haberlos mirado. `compartidos` tiene que cubrir TODOS los simbolos
    de nivel superior del canonico -- si manana se agrega uno y no se declara,
    este test se pone rojo antes que el drift."""
    sys.path.insert(0, str(RAIZ / "scripts"))
    from check_mirror_sync import FAMILIAS  # noqa: E402

    familia = next((f for f in FAMILIAS if f.nombre == "cola_uso"), None)
    assert familia is not None, [f.nombre for f in FAMILIAS]
    assert familia.canonico == CANONICO_EN_JAX
    etiquetas = {e for e, _ in familia.espejos}
    assert "jax-platform" in etiquetas, etiquetas
    assert "las_manos" in etiquetas, etiquetas

    arbol = ast.parse(CANONICO_EN_JAX.read_text(encoding="utf-8"))
    del_archivo = set()
    for nodo in arbol.body:
        nombre = getattr(nodo, "name", None)
        if nombre is None and isinstance(nodo, ast.Assign) and len(nodo.targets) == 1:
            destino = nodo.targets[0]
            if isinstance(destino, ast.Name):
                nombre = destino.id
        if nombre:
            del_archivo.add(nombre)
    assert del_archivo - set(familia.compartidos) == set(), (
        "simbolos del canonico fuera de la comparacion: "
        f"{sorted(del_archivo - set(familia.compartidos))}"
    )


def test_la_copia_es_identica_al_canonico_de_jax_platform():
    """La comparacion REAL por AST. En CI la corre el job `mirror-sync`, que
    clona jax-platform y falla con exit 2 si el archivo no esta (fail-closed).
    Aca, sin ese checkout, no hay nada que comparar."""
    raiz_platform = Path(
        os.environ.get("JAX_PLATFORM_REPO_ROOT", Path.home() / "jax-platform")
    )
    if not (raiz_platform / "backend" / "uso" / "cola.py").is_file():
        pytest.skip(
            f"sin checkout de jax-platform en {raiz_platform} "
            "(seteá JAX_PLATFORM_REPO_ROOT); en CI lo cubre el job mirror-sync"
        )
    sys.path.insert(0, str(RAIZ / "scripts"))
    from check_mirror_sync import FAMILIAS, revisar  # noqa: E402

    familia = next(f for f in FAMILIAS if f.nombre == "cola_uso")
    drift, _declaradas, faltantes = revisar(familia)
    assert (drift, faltantes) == ([], [])


def test_los_escritores_de_jax_NO_drenan():
    """Solo la plataforma inserta. Que este escrito en un comentario no alcanza:
    esto lo verifica."""
    for ruta in ESCRITORES:
        fuente = ruta.read_text(encoding="utf-8")
        arbol = ast.parse(fuente)
        usados = {
            n.id for n in ast.walk(arbol) if isinstance(n, ast.Name)
        } | {
            n.attr for n in ast.walk(arbol) if isinstance(n, ast.Attribute)
        } | {
            a.asname or a.name
            for n in ast.walk(arbol)
            if isinstance(n, (ast.Import, ast.ImportFrom))
            for a in n.names
        }
        prohibidos = sorted(usados & set(API_DE_DRENAJE))
        assert prohibidos == [], f"{ruta.name} usa API de drenaje: {prohibidos}"


# ---------------------------------------------------------------------------
# Los dos escritores
# ---------------------------------------------------------------------------

class _CursorFalso:
    def __init__(self, filas):
        self._filas = filas

    async def execute(self, *_a, **_k):
        return None

    async def fetchone(self):
        return self._filas

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_a):
        return False


class _ConexionFalsa:
    """La base responde bien: el lookup de precio devuelve una fila y el INSERT
    no explota."""

    def __init__(self):
        self.cerrada = False

    def cursor(self):
        return _CursorFalso((1.0, 2.0))

    def close(self):
        self.cerrada = True


@pytest.fixture
def respaldo(tmp_path, monkeypatch):
    """Directorio del respaldo aislado + la config de DB que hace falta para
    llegar al pool (y explotar ahi, no antes). Una conexion suelta queda
    registrada en `sueltas` y el test la rechaza al final."""
    import aiomysql
    import cola_uso

    sueltas = []

    async def _connect_suelto(*_a, **_k):
        sueltas.append(1)
        raise OSError("conexion suelta: el escritor tiene que usar el pool")

    monkeypatch.setattr(aiomysql, "connect", _connect_suelto)

    directorio = tmp_path / "usage-spool"
    monkeypatch.setenv("JAX_USAGE_SPOOL_DIR", str(directorio))
    monkeypatch.setenv("JAX_DB_HOST", "127.0.0.1")
    monkeypatch.setenv("JAX_DB_PORT", "1")
    monkeypatch.setenv("JAX_DB_NAME", "jax_memory_test")
    cola_uso.reset_estado()
    yield directorio
    assert sueltas == [], "un escritor abrio aiomysql.connect propio en vez del pool"


def _filas_del_respaldo(directorio: Path) -> list[dict]:
    if not directorio.is_dir():
        return []
    return [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(directorio.glob("*.json"))
    ]


class _Pool:
    """Doble de `jacobs.store.conexion`: cuenta los pedidos y entrega una
    conexion sana o explota al pedirla (base caida o pool agotado)."""

    def __init__(self, error: Exception | None = None, al_pedir=None):
        self.error = error
        self.al_pedir = al_pedir
        self.pedidos = 0

    def __call__(self, desechable: bool = False):
        self.pedidos += 1
        if self.al_pedir:
            self.al_pedir()

        @contextlib.asynccontextmanager
        async def _ctx():
            if self.error is not None:
                raise self.error
            yield _ConexionFalsa()

        return _ctx()


def _pool_que_explota():
    return _Pool(OSError("la base no está"))


def _pool_ok():
    return _Pool()


def _usar_pool(monkeypatch, pool: _Pool) -> _Pool:
    from jacobs import store
    monkeypatch.setattr(store, "conexion", pool)
    return pool


# --- jacobs -----------------------------------------------------------------

def test_jacobs_encola_la_fila_cuando_la_base_falla(respaldo, monkeypatch):
    from jacobs import usage_writer

    _usar_pool(monkeypatch, _pool_que_explota())
    asyncio.run(usage_writer.record_direct_usage(
        user_id="7", tenant_id="77", facet="jekyll", provider_id="deepseek",
        model="deepseek-v4-flash", tokens_in=123, tokens_out=45,
    ))

    filas = _filas_del_respaldo(respaldo)
    assert len(filas) == 1, filas
    fila = filas[0]
    assert set(fila) == CAMPOS_DEL_CONTRATO, sorted(set(fila) ^ CAMPOS_DEL_CONTRATO)
    assert fila["origen"] == "jacobs"
    assert fila["request_type"] == "pipeline"
    assert fila["tenant_id"] == 77 and fila["user_id"] == 7
    assert fila["facet"] == "jekyll" and fila["model"] == "deepseek-v4-flash"
    assert fila["tokens_in"] == 123 and fila["tokens_out"] == 45
    # sin base no hay tabla `model` que consultar: el precio lo pone el que
    # drena, del lado de la plataforma.
    assert fila["cost_usd"] is None
    # jacobs no invoca trabajos del motor: los dos campos viajan explicitos en
    # null, no ausentes -- el que drena exige los TRECE.
    assert fila["status"] is None and fila["job_id"] is None


def test_jacobs_camino_feliz_no_deja_nada_en_el_respaldo(respaldo, monkeypatch):
    from jacobs import usage_writer

    _usar_pool(monkeypatch, _pool_ok())
    asyncio.run(usage_writer.record_direct_usage(
        user_id="7", tenant_id="77", facet="jekyll", provider_id="deepseek",
        model="m", tokens_in=1, tokens_out=2,
    ))
    assert _filas_del_respaldo(respaldo) == []


def test_jacobs_loguea_ERROR_si_tampoco_puede_encolar(respaldo, monkeypatch, caplog):
    """La cola es la red, no una excusa para bajar el volumen: si la red
    tampoco esta, la fila SI se perdio y eso es un ERROR."""
    from jacobs import usage_writer

    async def _encolar_que_no_puede(_fila):
        return None

    _usar_pool(monkeypatch, _pool_que_explota())
    monkeypatch.setattr(usage_writer, "encolar_uso", _encolar_que_no_puede)
    with caplog.at_level("ERROR", logger="jacobs.usage_writer"):
        asyncio.run(usage_writer.record_direct_usage(
            user_id="7", tenant_id="77", facet="jekyll", provider_id="p",
            model="m", tokens_in=1, tokens_out=2,
        ))
    errores = [r for r in caplog.records if r.levelname == "ERROR"]
    assert errores, caplog.records
    assert "jekyll" in errores[0].getMessage()


def test_jacobs_no_loguea_ERROR_cuando_pudo_encolar(respaldo, monkeypatch, caplog):
    """Encolada NO es perdida: un ERROR ahi entrena a ignorar el log."""
    from jacobs import usage_writer

    _usar_pool(monkeypatch, _pool_que_explota())
    with caplog.at_level("DEBUG", logger="jacobs.usage_writer"):
        asyncio.run(usage_writer.record_direct_usage(
            user_id="7", tenant_id="77", facet="jekyll", provider_id="p",
            model="m", tokens_in=1, tokens_out=2,
        ))
    assert [r for r in caplog.records if r.levelname == "ERROR"] == []


# --- motor_registry ---------------------------------------------------------

def test_motor_encola_tras_agotar_los_reintentos(respaldo, monkeypatch):
    from motor_registry import usage_writer as motor

    pool = _usar_pool(monkeypatch, _pool_que_explota())

    async def _sin_espera(_s):
        return None

    monkeypatch.setattr(motor.asyncio, "sleep", _sin_espera)
    asyncio.run(motor.record_motor_usage(
        user_id="7", tenant_id="77", facet="ada", provider_id="openai",
        model="gpt-x", tokens_in=10, tokens_out=20, job_id="j1", status="failed",
    ))

    assert pool.pedidos == motor._WRITE_MAX_ATTEMPTS, pool.pedidos
    filas = _filas_del_respaldo(respaldo)
    assert len(filas) == 1, filas
    fila = filas[0]
    assert set(fila) == CAMPOS_DEL_CONTRATO, sorted(set(fila) ^ CAMPOS_DEL_CONTRATO)
    assert fila["origen"] == "motor_registry"
    assert fila["request_type"] == "motor"
    assert fila["facet"] == "ada" and fila["model"] == "gpt-x"
    assert fila["tokens_in"] == 10 and fila["tokens_out"] == 20
    assert fila["status"] == "failed" and fila["job_id"] == "j1"


def test_motor_camino_feliz_no_deja_nada_en_el_respaldo(respaldo, monkeypatch):
    from motor_registry import usage_writer as motor

    _usar_pool(monkeypatch, _pool_ok())
    asyncio.run(motor.record_motor_usage(
        user_id="7", tenant_id="77", facet="ada", provider_id="openai",
        model="gpt-x", tokens_in=1, tokens_out=2, job_id="j1", status="completed",
    ))
    assert _filas_del_respaldo(respaldo) == []


def test_motor_loguea_ERROR_si_tampoco_puede_encolar(respaldo, monkeypatch, caplog):
    from motor_registry import usage_writer as motor

    async def _encolar_que_no_puede(_fila):
        return None

    async def _sin_espera(_s):
        return None

    _usar_pool(monkeypatch, _pool_que_explota())
    monkeypatch.setattr(motor.asyncio, "sleep", _sin_espera)
    monkeypatch.setattr(motor, "encolar_uso", _encolar_que_no_puede)
    with caplog.at_level("ERROR", logger="motor_registry.usage_writer"):
        asyncio.run(motor.record_motor_usage(
            user_id="7", tenant_id="77", facet="ada", provider_id="p",
            model="m", tokens_in=1, tokens_out=2, job_id="j9", status="failed",
        ))
    errores = [r for r in caplog.records if r.levelname == "ERROR"]
    assert errores, caplog.records
    assert "j9" in errores[0].getMessage()


def test_motor_manda_status_y_job_id_EN_EL_ARCHIVO(respaldo, monkeypatch, caplog):
    """CAMBIO DE SENTIDO (Task 8, 2026-09-15). Este test fijaba lo contrario:
    que `status` y `job_id` NO viajaban en el archivo y por eso tenian que
    quedar en el log. Con el contrato en TRECE campos viajan, y es lo que
    importa: sin ellos la fila recuperada entra a `axioma_usage` con esas dos
    columnas en NULL y la reconciliacion contra motor_jobs.jsonl por igualdad
    exacta (T3) no la puede emparejar -- se recupera el cobro y se pierde la
    trazabilidad.

    El log los sigue nombrando y se sigue exigiendo aca: el operador que ve el
    INFO no tiene que ir a abrir el archivo para saber de que trabajo era.
    """
    from motor_registry import usage_writer as motor

    async def _sin_espera(_s):
        return None

    _usar_pool(monkeypatch, _pool_que_explota())
    monkeypatch.setattr(motor.asyncio, "sleep", _sin_espera)
    with caplog.at_level("INFO", logger="motor_registry.usage_writer"):
        asyncio.run(motor.record_motor_usage(
            user_id="7", tenant_id="77", facet="ada", provider_id="p",
            model="m", tokens_in=1, tokens_out=2, job_id="j42", status="killed",
        ))

    filas = _filas_del_respaldo(respaldo)
    assert len(filas) == 1, filas
    assert filas[0]["status"] == "killed"
    assert filas[0]["job_id"] == "j42"
    texto = " ".join(r.getMessage() for r in caplog.records)
    assert "j42" in texto and "killed" in texto, texto


def test_created_at_es_la_hora_del_TURNO_no_la_del_reintento(respaldo, monkeypatch):
    """Una caida de dos horas movería el costo al dia siguiente si la hora se
    tomara al encolar. El reloj avanza en cada intento fallido: la fila tiene
    que quedarse con la lectura anterior al primer intento."""
    import cola_uso
    from motor_registry import usage_writer as motor

    reloj = ["2026-09-15T00:00:00+00:00"]

    def _avanza_el_reloj():
        reloj[0] = "2026-09-16T02:00:00+00:00"

    async def _sin_espera(_s):
        return None

    monkeypatch.setattr(cola_uso, "_ahora_iso", lambda: reloj[0])
    monkeypatch.setattr(motor, "_ahora_iso", lambda: reloj[0])
    _usar_pool(monkeypatch, _Pool(OSError("la base no está"), al_pedir=_avanza_el_reloj))
    monkeypatch.setattr(motor.asyncio, "sleep", _sin_espera)
    asyncio.run(motor.record_motor_usage(
        user_id="7", tenant_id="77", facet="ada", provider_id="p",
        model="m", tokens_in=1, tokens_out=2, job_id="j1", status="failed",
    ))

    filas = _filas_del_respaldo(respaldo)
    assert len(filas) == 1, filas
    assert filas[0]["created_at"] == "2026-09-15T00:00:00+00:00"
