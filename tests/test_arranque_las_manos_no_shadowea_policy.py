"""Arranque real de LAS MANOS: el cwd de uvicorn no puede colisionar con
`policy/` de la raíz.

**El bug real (jax#260, visto en producción, rollback a e09c3b3).**
`_configure_b7_trusted_runtime()` (`las_manos/server.py`, primera línea del
startup event `_jacobs_init`, agregada por #260) importa
`policy.enforcement_evidence.*` y `policy.execution_control.*` — el paquete
namespace de la RAÍZ del repo (sin `__init__.py`, a propósito). Pero
`las_manos/policy.py` (el motor de política viejo, `PolicyEngine`) EXISTE y
tiene el mismo nombre de primer nivel. El servicio real arranca con
`WorkingDirectory=las_manos/` vía `uvicorn server:app`, y con ESTE venv (y
esta versión de Python) eso deja `sys.path[0] == ''` — que Python resuelve
como el directorio de trabajo ACTUAL, no el del script — **por delante de
cualquier `PYTHONPATH`** (verificado a mano contra el venv real de
producción, `/srv/jax-prod/jax/las_manos/.venv/bin/uvicorn`, con y sin
`PYTHONPATH`: el orden de `sys.path` es siempre `['', '.venv/bin',
<PYTHONPATH...>, stdlib...]`). Por eso `PYTHONPATH=/srv/jax-prod/jax` —ya
agregado por el controlador, ver el encargo— NO alcanza: `import policy`
sigue resolviendo `las_manos/policy.py` antes de mirar `PYTHONPATH`, y
revienta con `ModuleNotFoundError: No module named 'policy.enforcement_evidence';
'policy' is not a package`.

**Por qué ningún test lo agarró.** Todos los tests que importan `server`
dentro del proceso de pytest hacen `sys.path.insert(0, las_manos)` — pero
para cuando corren, el paquete REAL `policy` (el de la raíz) casi siempre ya
está en `sys.modules` por otro test de gobernanza que corrió antes en la
misma sesión, y Python nunca vuelve a mirar el disco para un nombre ya
cacheado: el mecanismo que rompe en producción queda enmascarado dentro del
proceso de pytest. Por eso ESTE test arranca un PROCESO NUEVO — el único
jeito de reproducir el orden de `sys.path` real — con `cwd=las_manos/` y
`PYTHONPATH=<raíz>`, la misma composición que el drop-in de systemd de
producción.

**Por qué llama a `_configure_b7_trusted_runtime()` directo y no corre todo
`_jacobs_init()`.** `tests/_alcance_las_manos.py` ya documenta por qué un
`TestClient` con el lifespan completo NO se usa sin querer: dispara el
reaper y manda alertas REALES a Telegram, y además abre el pool de Jacobs
contra una MariaDB real. `_configure_b7_trusted_runtime()` es la función que
agregó #260 — literalmente "el startup event" en la parte que agregó el
bug — y se invoca EXACTAMENTE como la invoca `_jacobs_init()`: primera
llamada, sin argumentos. Se le da un `JAX_DB_PORT` que rechaza la conexión
al toque (127.0.0.1:1 — ningún servicio escucha ahí nunca) para no depender
de ninguna MariaDB real: si el import revienta, jamás llega a conectarse; si
el import está arreglado, revienta MÁS ADELANTE, con un error de conexión
(`OSError`/`pymysql.err.OperationalError`), nunca con el
`ModuleNotFoundError` envuelto en "B7 trusted composition dependencies
unavailable".

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import secrets
import subprocess
import sys
import tempfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
LAS_MANOS = RAIZ / "las_manos"

#: El script que corre en el proceso hijo. Se manda por STDIN (`python -`),
#: no como archivo (`python script.py`): un archivo pone el directorio DEL
#: SCRIPT en `sys.path[0]`, no el cwd -- justo el mecanismo que hay que
#: reproducir. `-` (como `-c`) deja `sys.path[0] == ''`, igual que el
#: `uvicorn server:app` real (verificado a mano, ver docstring del módulo).
_SCRIPT_HIJO = """
import sys

try:
    import server
    server._configure_b7_trusted_runtime()
except BaseException as exc:
    causa = type(exc.__cause__).__name__ if exc.__cause__ is not None else ""
    print(f"RESULTADO:{type(exc).__name__}:{causa}:{exc}")
    sys.exit(0)
print("RESULTADO:OK::llego-hasta-el-final-sin-excepcion")
"""


def _correr_arranque_real(tmp_path: Path) -> str:
    """Arranca `server` en un proceso nuevo, con el mismo `cwd`/`PYTHONPATH`
    que usa uvicorn en producción, y devuelve la línea `RESULTADO:...`.

    Entorno construido DESDE CERO (nunca `os.environ` heredado): ni lee ni
    depende de `/etc/jax/.env` -- todas las credenciales son de esta corrida,
    generadas con `secrets`, y el `JAX_DB_PORT` nunca alcanza una base real.
    """
    credencial_plataforma = secrets.token_urlsafe(32)
    credencial_jacobs = secrets.token_urlsafe(32)
    entorno = {
        "HOME": str(tmp_path),
        "PATH": "/usr/bin:/bin",
        "PYTHONPATH": str(RAIZ),
        # Mismo patrón que conftest.py de la raíz: nada de esto toca una ruta
        # real, todo vive en el tmp_path de ESTA corrida.
        "JAX_USAGE_SPOOL_DIR": str(tmp_path / "usage-spool"),
        "LAS_MANOS_URL": "http://las-manos.invalid:7777",
        "JAX_OLLAMA_URL": "http://ollama.invalid:11434",
        "JAX_LAS_MANOS_CREDENCIAL_PLATAFORMA": credencial_plataforma,
        "JAX_LAS_MANOS_CREDENCIAL_JACOBS": credencial_jacobs,
        "JAX_REPO_BASE": str(tmp_path / "repo"),
        "JAX_AUDIT_LOG_PATH": str(tmp_path / "audit" / "audit.jsonl"),
        "JAX_FACET_SEAL_PATH": str(tmp_path / "seal" / "facet-cache-seal"),
        "JAX_KILL_SWITCH_PATH": str(tmp_path / "interruptor" / "PAUSE"),
        # Los tres que exige el guard de _configure_b7_trusted_runtime():
        "JAX_DEPLOYMENT_ID": "test-arranque-real",
        "JAX_DB_HOST": "127.0.0.1",
        # Puerto que nunca escucha: la conexión se rechaza al toque, sin
        # tocar ninguna MariaDB real (ni la de test ni la de producción).
        "JAX_DB_PORT": "1",
    }
    proceso = subprocess.run(
        [sys.executable, "-"],
        input=_SCRIPT_HIJO,
        cwd=str(LAS_MANOS),
        env=entorno,
        capture_output=True,
        text=True,
        timeout=30,
    )
    lineas_resultado = [
        linea for linea in proceso.stdout.splitlines() if linea.startswith("RESULTADO:")
    ]
    assert lineas_resultado, (
        f"el proceso hijo no imprimió ningún RESULTADO -- salió con {proceso.returncode}.\n"
        f"stdout:\n{proceso.stdout}\nstderr:\n{proceso.stderr}"
    )
    return lineas_resultado[-1]


def test_arranque_real_no_colisiona_con_policy_de_la_raiz(tmp_path: Path) -> None:
    """Reproduce el arranque real (cwd=las_manos/, PYTHONPATH=raíz) y exige
    que la colisión de nombres esté resuelta: `_configure_b7_trusted_runtime`
    tiene que poder importar `policy.enforcement_evidence.*` y
    `policy.execution_control.*` de la RAÍZ, no de `las_manos/`.

    Contra 88f9a02 esto da
    `RESULTADO:RuntimeError:ModuleNotFoundError:B7 trusted composition
    dependencies unavailable` (medido a mano antes de este commit). Con el
    fix, el import ya no revienta -- lo que revienta después es la conexión
    a MariaDB (puerto 1, nadie escucha ahí), que es un fallo DISTINTO y
    esperado en este test, no el que se está cazando acá."""
    resultado = _correr_arranque_real(tmp_path)

    assert "ModuleNotFoundError" not in resultado, (
        f"el arranque real todavía colisiona con las_manos/policy.py (la "
        f"colisión de #260/e09c3b3): {resultado}"
    )
    assert "B7 trusted composition dependencies unavailable" not in resultado, (
        f"_configure_b7_trusted_runtime volvió a envolver un fallo de import "
        f"como 'dependencies unavailable': {resultado}"
    )


def test_arranque_real_sin_fix_falla_mas_adelante_no_por_import(tmp_path: Path) -> None:
    """Control: con el fix aplicado, `_configure_b7_trusted_runtime` sí llega
    a ejecutar código DESPUÉS del bloque `try/except` que envuelve los
    imports (líneas 205-218 de `server.py`) -- prueba que el test de arriba
    no está verde porque el guard de arriba
    (`JAX_DEPLOYMENT_ID`/`JAX_DB_HOST`/`JAX_DB_PORT`) cortó ANTES de llegar a
    los imports.

    Medido en este sandbox: `FileNotFoundError` sobre
    `/etc/jax/build/implementation-identity.json` (construyendo
    `TrustedImplementationIdentityProvider`, la primera línea después del
    `try/except` de imports) -- no depende de red ni de ninguna MariaDB
    real. En un entorno donde ese archivo SÍ existe, seguiría más adelante y
    fallaría en la conexión (puerto 1, nadie escucha ahí). Cualquiera de
    las dos es un fallo DISTINTO al de la colisión de imports, así que el
    control acepta las dos formas -- lo único que descarta es "OK" (llegó
    hasta el final, lo que en este sandbox sin MariaDB ni ese archivo no
    puede pasar) y la firma de la colisión."""
    resultado = _correr_arranque_real(tmp_path)

    assert not resultado.startswith("RESULTADO:OK:"), (
        f"se esperaba que este entorno de prueba (sin "
        f"/etc/jax/build/implementation-identity.json y con JAX_DB_PORT=1) "
        f"no pudiera completar el arranque -- si llegó OK, revisar el "
        f"entorno del test: {resultado}"
    )
    assert "ModuleNotFoundError" not in resultado, (
        f"debería haber pasado el punto de la colisión y fallar más "
        f"adelante, no en el import: {resultado}"
    )
