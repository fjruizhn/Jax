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
el import está arreglado, revienta MÁS ADELANTE, con un error DISTINTO
(`FileNotFoundError` de `implementation-identity.json` o `OperationalError`
de pymysql conectando al puerto 1) — nunca con el `ModuleNotFoundError`
envuelto en "B7 trusted composition dependencies unavailable".

**MAJOR-1 (ronda 1 de revisión de PR#262, encontrado por el reviewer,
verificado acá independientemente).** La primera versión de este archivo
afirmaba en NEGATIVO ("no contiene ModuleNotFoundError", "no es RESULTADO:OK").
Quitar `JAX_DEPLOYMENT_ID` del entorno alcanza para que el guard de
`_configure_b7_trusted_runtime` (`server.py:203-204`) corte ANTES de llegar
al `try/except` de imports — ni siquiera INTENTA importar
`policy.enforcement_evidence` — y el resultado
(`RuntimeError: B7 trusted composition requires deployment and MariaDB
configuration`) pasaba las dos aserciones en negativo IGUAL contra el
código viejo (con `las_manos/policy.py` todavía presente): el test daba
verde con el defecto adentro. Reescrito como ALLOWLIST
(`_firma_conocida_tras_el_fix`): el resultado tiene que ser una de las DOS
firmas conocidas de "ya pasé la colisión", o el test falla con el mensaje
real. `test_allowlist_rechaza_el_guard_cortando_antes_de_los_imports` deja
esa mutación (quitar `JAX_DEPLOYMENT_ID`) como regresión permanente.

**Sobre el despliegue (no es un test — nada de código lo puede verificar
desde acá).** El fix depende de que el archivo viejo quede BORRADO del
checkout de producción, no solo de que el nuevo exista: un `rsync` sin
`--delete` (o un `cp -r` sobre el árbol existente) deja `policy.py` Y
`motor_de_politica.py` conviviendo, con la colisión intacta y el merge en
verde. Quien despliegue tiene que confirmar, DESPUÉS de desplegar:
`ls /srv/jax-prod/jax/las_manos/policy.py` → `No such file or directory`.

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


def _correr_arranque_real(tmp_path: Path, *, omitir: frozenset[str] = frozenset()) -> str:
    """Arranca `server` en un proceso nuevo, con el mismo `cwd`/`PYTHONPATH`
    que usa uvicorn en producción, y devuelve la línea `RESULTADO:...`.

    Entorno construido DESDE CERO (nunca `os.environ` heredado): ni lee ni
    depende de `/etc/jax/.env` -- todas las credenciales son de esta corrida,
    generadas con `secrets`, y el `JAX_DB_PORT` nunca alcanza una base real.

    `omitir`: nombres de variable a NO incluir -- lo usa el test de
    mutación para simular el guard cortando antes de los imports (quitar
    `JAX_DEPLOYMENT_ID`), sin duplicar la construcción del entorno.
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
    for clave in omitir:
        entorno.pop(clave, None)
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


#: ALLOWLIST, no denylist (MAJOR-1, ronda 1 de revisión de PR#262 -- ver la
#: nota en el docstring del módulo). Las dos únicas formas en que ESTE
#: sandbox (sin `/etc/jax/build/implementation-identity.json`, sin ninguna
#: MariaDB escuchando en el puerto 1) puede fallar DESPUÉS de haber pasado
#: el bloque `try/except` de imports de `_configure_b7_trusted_runtime`
#: (server.py:205-218) -- es decir, después de haber importado
#: `policy.enforcement_evidence.*`/`policy.execution_control.*` con éxito.
#: Un resultado que NO matchea ninguna de las dos no prueba que el fix
#: funcione: hay que mirar cuál es antes de asumir nada.
def _firma_conocida_tras_el_fix(resultado: str) -> str | None:
    """Descripción corta de qué firma conocida matcheó `resultado`, o
    `None` si no es ninguna de las dos -- incluye el caso del guard
    cortando antes (falta `JAX_DEPLOYMENT_ID`) y el de una dependencia
    faltante en el intérprete (falta `fastapi`, por ejemplo: en hall9000
    con el `python3` del sistema en vez del venv del repo, MINOR-1 de la
    misma ronda)."""
    if (
        resultado.startswith("RESULTADO:FileNotFoundError:")
        and "implementation-identity.json" in resultado
    ):
        return (
            "FileNotFoundError sobre implementation-identity.json "
            "(TrustedImplementationIdentityProvider -- ya pasó los imports)"
        )
    if "OperationalError" in resultado:
        return "OperationalError de pymysql conectando al puerto 1 (ya pasó los imports)"
    return None


def test_arranque_real_no_colisiona_con_policy_de_la_raiz(tmp_path: Path) -> None:
    """Reproduce el arranque real (cwd=las_manos/, PYTHONPATH=raíz) y exige
    que la colisión de nombres esté resuelta: `_configure_b7_trusted_runtime`
    tiene que poder importar `policy.enforcement_evidence.*` y
    `policy.execution_control.*` de la RAÍZ, no de `las_manos/`.

    Contra 88f9a02 esto da
    `RESULTADO:RuntimeError:ModuleNotFoundError:B7 trusted composition
    dependencies unavailable` (medido a mano antes de este commit) -- NO
    matchea `_firma_conocida_tras_el_fix`. Con el fix, el import ya no
    revienta y el resultado sí matchea una de las dos firmas conocidas."""
    resultado = _correr_arranque_real(tmp_path)
    firma = _firma_conocida_tras_el_fix(resultado)

    assert firma is not None, (
        f"el arranque real NO dio ninguna de las firmas conocidas de 'ya "
        f"pasé la colisión' -- esto no prueba que el fix funcione, prueba "
        f"que pasó OTRA cosa y hay que mirar cuál antes de asumir nada. "
        f"Si el resultado es un ModuleNotFoundError sobre "
        f"'policy.enforcement_evidence' (o 'policy' a secas), es la "
        f"colisión de #260/e09c3b3 sin arreglar. Si es un RuntimeError con "
        f"'B7 trusted composition requires deployment and MariaDB "
        f"configuration', el guard cortó ANTES de los imports (revisar el "
        f"entorno del test, no el fix -- ver "
        f"test_allowlist_rechaza_el_guard_cortando_antes_de_los_imports). "
        f"Si es un ModuleNotFoundError sobre otra cosa (fastapi, pymysql, "
        f"...), a ESTE intérprete le falta una dependencia -- correr con "
        f"el venv del repo, no con un python3 del sistema. "
        f"resultado real: {resultado!r}"
    )


def test_allowlist_rechaza_el_guard_cortando_antes_de_los_imports(tmp_path: Path) -> None:
    """Mutación permanente (MAJOR-1, ronda 1 de revisión de PR#262): el
    reviewer encontró que quitar `JAX_DEPLOYMENT_ID` del entorno alcanzaba
    para que las dos aserciones EN NEGATIVO de la versión anterior de este
    archivo ("no contiene ModuleNotFoundError", "no es RESULTADO:OK")
    dieran verde -- el guard de `_configure_b7_trusted_runtime`
    (`server.py:203-204`) corta ANTES de llegar al `try/except` de imports,
    así que ni siquiera INTENTA importar `policy.enforcement_evidence`. Un
    denylist no distingue "no colisionó" de "nunca llegó a intentarlo".

    Este test deja esa mutación corriendo de verdad, para siempre: si
    `_firma_conocida_tras_el_fix` alguna vez se afloja de vuelta a un
    denylist, este test lo agarra."""
    resultado = _correr_arranque_real(tmp_path, omitir=frozenset({"JAX_DEPLOYMENT_ID"}))

    assert (
        "B7 trusted composition requires deployment and MariaDB configuration"
        in resultado
    ), (
        f"se esperaba que quitar JAX_DEPLOYMENT_ID disparara el guard de "
        f"_configure_b7_trusted_runtime (server.py:203-204) -- si el "
        f"mensaje cambió, actualizar este test, no borrarlo: {resultado}"
    )
    assert _firma_conocida_tras_el_fix(resultado) is None, (
        f"la allowlist NO debería aceptar el resultado del guard cortando "
        f"antes de los imports -- si esto pasa, la allowlist volvió a ser "
        f"lo bastante floja como para dar verde con el guard incompleto "
        f"(el defecto exacto de MAJOR-1, ronda 1 de revisión de PR#262): "
        f"{resultado}"
    )
