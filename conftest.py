"""Aislamiento de la suite de `jax` respecto de los directorios REALES de
producción. Primer conftest de este repo (2026-09-15, Task 7 de la cola durable
para el registro de uso).

**El respaldo de uso.** `jax/core/cola_uso.py` tiene como default
`/srv/jax-data/usage-spool`: el directorio del que `jax-platform` drena e
inserta en `axioma_usage`, la tabla de cobro. Desde que los dos escritores de
este repo encolan en vez de perder la fila, CUALQUIER test que ejercite
`record_motor_usage` o `record_direct_usage` contra una base inalcanzable
deposita ahí un archivo. No es hipotético: el 2026-09-15 la suite dejó 116
archivos de mentira en ese directorio, y habrían terminado como filas de uso
inventadas contra tenants reales en cuanto el reintento empezara a drenar.

Se fija con `os.environ` en tiempo de IMPORT, no con un fixture:
- un fixture hay que pedirlo, y el problema es exactamente el test que no sabe
  que está escribiendo ahí;
- `autouse` de sesión tampoco alcanzaría por sí solo si algún módulo leyera la
  ruta al importarse (es el mismo criterio con el que jax-platform fija
  `JAX_FACET_SEAL_PATH` antes de cualquier import).

`cola_uso` lee la variable en CADA llamada, así que un test que quiera su propio
directorio puede seguir haciendo `monkeypatch.setenv` sin pelearse con esto.

**El freno es verificable.** `pytest_sessionfinish` vuelve a mirar el directorio
de producción cuando ya corrieron TODOS los tests y pone la corrida en rojo si
apareció un archivo nuevo. `tests/test_respaldo_de_uso_aislado.py` ejercita el
detector contra una escritura de mentira: un control que no falla cuando debe no
valida nada.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import importlib
import os
import sys
import tempfile
import time
from pathlib import Path

import pytest

#: El default de `jax/core/cola_uso.py`. Se escribe acá, literal y a propósito:
#: si alguien cambiara el default del módulo sin tocar esto, el chequeo miraría
#: un directorio que ya no es el de producción y daría verde sin vigilar nada --
#: por eso `tests/test_respaldo_de_uso_aislado.py` compara esta constante contra
#: `cola_uso.DIRECTORIO_POR_DEFECTO`.
RESPALDO_DE_PRODUCCION = Path("/srv/jax-data/usage-spool")

#: Antes de que corra un solo test: lo anterior a esta marca no lo escribió la
#: suite. Contar lo que ya estaba pondría el freno en rojo permanente, y un
#: freno siempre rojo se termina ignorando.
INICIO_DE_SESION = time.time()

os.environ["JAX_USAGE_SPOOL_DIR"] = tempfile.mkdtemp(prefix="jax-test-respaldo-uso-")

#: E-21 (2026-09-16): jacobs/executor.py y plan.py leen LAS_MANOS_URL y
#: JAX_OLLAMA_URL al importarse y NO arrancan sin ellas (fail-closed). Se fijan
#: acá, antes de cualquier import. NO con los valores de producción: un test
#: que olvide parchear el transporte le pegaría a LAS MANOS o al Ollama vivos.
#: El dominio `.invalid` (RFC 6761) nunca resuelve, así que ese olvido falla con
#: un error de DNS; el valor sigue siendo una URL válida y estable para las
#: aserciones de ruta. tests/test_config_entorno.py lo vigila.
os.environ["LAS_MANOS_URL"] = "http://las-manos.invalid:7777"
os.environ["JAX_OLLAMA_URL"] = "http://ollama.invalid:11434"

#: 2026-09-17 (autenticación de servicio de LAS MANOS): jacobs/executor.py manda
#: la credencial `jacobs` en cada pedido a LAS MANOS y server.py no arranca sin
#: las dos. Valores de prueba generados por sesión, NUNCA los de /etc/jax/.env.
import secrets as _secrets  # noqa: E402
os.environ["JAX_LAS_MANOS_CREDENCIAL_PLATAFORMA"] = _secrets.token_urlsafe(32)
os.environ["JAX_LAS_MANOS_CREDENCIAL_JACOBS"] = _secrets.token_urlsafe(32)

#: E-22 (2026-09-16): el .md de cortesía de cada step se escribe en
#: $JAX_REPO_BASE/documents. En producción es /home/fruiz/jax/repo, que el
#: admin de jax-platform lista. Un test que ejercite _persist_step_to_repo no
#: puede dejar ahí un documento de mentira: mismo criterio que el respaldo de uso.
os.environ["JAX_REPO_BASE"] = tempfile.mkdtemp(prefix="jax-test-repo-")

#: 2026-09-17: `jax/core/facet_resolver.py` y `las_manos/facet_resolver.py`
#: leen JAX_FACET_SEAL_PATH al importarse, con default
#: /srv/jax-data/facet-cache-seal -- el sello REAL que jax-platform y LAS
#: MANOS usan para invalidar el caché de facetas. Sin esto, cualquier test
#: local que dispare una invalidación de facetas (o una migración que la
#: dispare) le toca el mtime al archivo de producción -- pasó el
#: 2026-09-17 01:51:21. Asignación DIRECTA, no `setdefault`: si alguien
#: llega con JAX_FACET_SEAL_PATH ya apuntando a /srv (p.ej. el .env de
#: producción sourceado antes de invocar pytest), `setdefault` lo
#: respetaría; el conftest tiene que pisarlo igual.
#: tests/test_conftest_aisla_facet_seal.py lo vigila.
os.environ["JAX_FACET_SEAL_PATH"] = os.path.join(
    tempfile.mkdtemp(prefix="jax-test-facet-seal-"), "facet-cache-seal"
)
# El kill switch, aislado por la misma razón (2026-09-16, frente B). Desde
# ese día la ruta del freno sale de JAX_KILL_SWITCH_PATH y /etc/jax/.env la
# define: un test que pusiera el freno sin esto detendría a JAX en producción.
# Asignación y no setdefault, a propósito.
os.environ["JAX_KILL_SWITCH_PATH"] = os.path.join(
    tempfile.mkdtemp(prefix="jax-test-interruptor-"), "PAUSE")

#: La base de tests de ESTA sesión (decisión de Fernando, 2026-09-17).
#: Tres sesiones de Claude compartían `jax_memory_test` y se pisaban de
#: verdad: filas con `mode` en NULL, un arnés expirando pipelines ajenos, una
#: sesión borrando la fila de uso de otra. Con `JAX_TEST_DB_SUFIJO=<sufijo>`
#: cada sesión corre contra `jax_memory_test_<sufijo>`; sin la variable, la
#: base sigue siendo `jax_memory_test` -- el CI no cambia.
#:
#: Se fija acá, en tiempo de IMPORT y antes de cualquier import del repo, por
#: la misma razón que el resto de este archivo: hay módulos que leen
#: `JAX_DB_NAME` al importarse. Cada archivo de test sigue llamando a
#: `fijar_base_de_test()`/`exigir_base_de_test()` por su cuenta -- muchos se
#: corren sueltos, sin pasar por acá.
#:
#: `asegurar_base_de_test()` crea la base de la sesión si no existía (esquema
#: clonado de `jax_memory_test` + `init_tables()` del repo). Sin sufijo no
#: hace nada. `tests/test_base_por_sesion.py` lo vigila.
from base_de_test import (  # noqa: E402
    asegurar_base_de_test,
    fijar_base_de_test,
)

fijar_base_de_test()
asegurar_base_de_test()

#: El freno de PRODUCCIÓN que la barrera de sesión vigila: los DOS ARCHIVOS
#: del interruptor (la ruta de JAX_KILL_SWITCH_PATH y la ruta heredada), NUNCA
#: el directorio que los contiene. La heredada se toma del módulo ANTES de
#: que el fixture de abajo la desvíe, y no se escribe acá: su literal sólo
#: vive en el módulo interruptor (tests/test_interruptor_sin_rutas_fijas.py).
#: FRENO_DE_PRODUCCION SÍ es literal acá, a propósito, igual criterio que
#: RESPALDO_DE_PRODUCCION: es el valor real de JAX_KILL_SWITCH_PATH en
#: /etc/jax/.env (verificado 2026-09-17 con `cat /etc/jax/.env`). Los dos son
#: ARCHIVOS, no directorios: se anota si cada uno existía al empezar para
#: distinguir "apareció" de "ya estaba".
#:
#: Hasta esta revisión (2026-09-17, false positive del frente B) el chequeo
#: miraba el DIRECTORIO /etc/jax/interruptor entero (su mtime y sus
#: entradas). Ese directorio es COMPARTIDO: otra sesión pone y suelta ahí
#: `EJECUTOR_PAUSA` (la pausa del Ejecutor, que NO es el kill switch) como
#: parte de una prueba de contrato legítima, y eso disparaba esta barrera
#: contra corridas de tests completamente ajenas al freno real. Mirar SÓLO
#: los dos archivos del freno (nunca el directorio ni otras entradas)
#: elimina el falso positivo sin perder la detección de un toque real.
from jax.core import interruptor as _interruptor  # noqa: E402

FRENO_DE_PRODUCCION = Path("/etc/jax/interruptor/PAUSE")
HEREDADA_DE_PRODUCCION = Path(_interruptor.RUTA_HEREDADA)
FRENO_EXISTIA_AL_INICIO = os.path.lexists(FRENO_DE_PRODUCCION)
HEREDADA_EXISTIA_AL_INICIO = os.path.lexists(HEREDADA_DE_PRODUCCION)

#: Los objetos módulo del interruptor. Son DOS distintos para el mismo archivo:
#: `jax.core.interruptor` (REPL, `jax --task`) e `interruptor` pelado vía el
#: symlink de las_manos (LAS MANOS, Jacobs). Cada uno tiene su RUTA_HEREDADA.
MODULOS_DEL_INTERRUPTOR = ("jax.core.interruptor", "interruptor")


@pytest.fixture(autouse=True)
def _ruta_heredada_del_freno_aislada(monkeypatch, tmp_path_factory):
    """La ruta vieja del freno (Task H del frente B, 2026-09-17) es una
    constante del módulo, no una variable (ruling R15): se desvía acá, en
    CADA test y en CADA objeto módulo, a un temporal que no existe. Sin esto,
    un host con la ruta vieja puesta daría vuelta todos los tests de "freno
    suelto". Importar los módulos antes de desviar evita que uno importado
    más tarde en el test quede sin aislar. También se reinicia el anti-spam
    del WARNING para que un test no herede el aviso de otro."""
    inexistente = tmp_path_factory.mktemp("jax-test-heredada") / "PAUSE"
    for nombre in MODULOS_DEL_INTERRUPTOR:
        try:
            modulo = importlib.import_module(nombre)
        except ImportError:
            # sin las_manos en sys.path el módulo pelado no se puede importar
            # tampoco desde el test: no hay nada que desviar
            continue
        monkeypatch.setattr(modulo, "RUTA_HEREDADA", inexistente)
        monkeypatch.setattr(modulo, "_heredada_avisada", False)


#: Task 6 (2026-09-18, aviso por Telegram al terminar un pipeline): desde que
#: `_correr_pipeline` (jacobs/executor.py) llama a
#: `jacobs.aviso.avisar_fin_pipeline` en cada transición terminal
#: (completed/aborted), CUALQUIER test que corra un pipeline hasta el final
#: agenda un envío real vía `send_telegram_alert` (jacobs/reaper.py:82) --
#: que sólo se abstiene si TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID no están
#: seteadas en el proceso. Un dev shell que sourceó /etc/jax/.env de
#: producción antes de invocar pytest SÍ las tiene seteadas (misma fuga que
#: motivó la barrera de DB de este archivo, T1 2026-09-14) -- sin este freno,
#: correr la suite local mandaría mensajes reales al chat de producción.
#: Mismo criterio que `_aviso_pipeline_no_dispara_solo` en el conftest de
#: jax-platform (Task 8 de esta misma ronda): no-op por defecto para TODA la
#: suite; los tests de `jacobs/_aviso_test.py` que quieren el comportamiento
#: real lo reponen con su propio `monkeypatch.setattr`, que corre DESPUÉS de
#: este fixture (dentro del cuerpo del test) y gana.
@pytest.fixture(autouse=True)
def _telegram_no_manda_de_verdad(monkeypatch):
    try:
        import jacobs.reaper as _reaper
    except ImportError:
        # sin las_manos en sys.path jacobs.reaper no se puede importar --
        # tampoco hay nada que pueda dispararlo desde este test
        return
    from unittest.mock import AsyncMock
    monkeypatch.setattr(
        _reaper, "send_telegram_alert",
        AsyncMock(return_value={"ok": False, "message_id": None, "error": "no-op de test (conftest raíz)"}),
    )


def archivos_nuevos_en(directorio: Path, desde: float) -> list[Path]:
    """Los archivos de `directorio` (y sus subdirectorios, que es donde
    `cola_uso` manda los corruptos) con mtime igual o posterior a `desde`.

    Que el directorio no exista son CERO escrituras, no un error: en el runner
    de CI `/srv/jax-data` no existe, y hacer explotar el chequeo ahí lo
    convertiría en ruido que alguien apagaría.
    """
    directorio = Path(directorio)
    if not directorio.is_dir():
        return []
    nuevos = []
    for ruta in directorio.rglob("*"):
        try:
            if ruta.is_file() and ruta.stat().st_mtime >= desde:
                nuevos.append(ruta)
        except OSError:
            # fail-soft: un archivo que desaparece entre el rglob y el stat no es una escritura de la suite
            continue
    return sorted(nuevos)


def _cambio_de_archivo(archivo: Path, inicio: float, archivo_existia: bool) -> list[str]:
    """Qué tocó la sesión en ESTE archivo. Sólo LEE (os.lstat): nunca crea
    nada, y nunca mira el directorio que lo contiene.

    Apareció (no existía al inicio), desapareció, o su mtime es de esta
    sesión: cambió. Fail-closed igual que `interruptor.pausa_presente`: un
    error de stat que no sea "no existe" (permiso, ENOTDIR, E/S) cuenta como
    PUESTO -- y sin poder saber si es nuevo, cuenta como cambio.
    """
    try:
        estado = os.lstat(archivo)
    except FileNotFoundError:
        if archivo_existia:
            return [f"{archivo} (desapareció)"]
        return []
    except OSError:
        return [str(archivo)]
    if not archivo_existia or estado.st_mtime >= inicio:
        return [str(archivo)]
    return []


def cambios_del_freno_de_produccion(
    freno: Path, heredada: Path, inicio: float,
    freno_existia: bool, heredada_existia: bool,
) -> list[str]:
    """Qué tocó la sesión en el freno de producción: SÓLO los dos archivos
    (`freno` = JAX_KILL_SWITCH_PATH, `heredada` = la ruta vieja), nunca el
    directorio que los contiene.

    El directorio (`/etc/jax/interruptor`) es compartido con la pausa del
    Ejecutor (`EJECUTOR_PAUSA`), que otra sesión pone y suelta como parte de
    una prueba de contrato legítima -- mirar el directorio confundía esa
    prueba con un toque al freno real (falso positivo, revisión del frente B
    2026-09-17). Mirar sólo estos dos archivos detecta el freno real
    (puesto, quitado, o su mtime tocado) sin reaccionar a un hermano suyo.
    """
    return (
        _cambio_de_archivo(freno, inicio, freno_existia)
        + _cambio_de_archivo(heredada, inicio, heredada_existia)
    )


def pytest_sessionfinish(session, exitstatus):
    """El chequeo que NO depende del orden de colección.

    Un test que mira el directorio de producción sólo ve lo escrito por los
    tests que corrieron ANTES que él. Acá ya corrieron todos.
    """
    nuevos = archivos_nuevos_en(RESPALDO_DE_PRODUCCION, INICIO_DE_SESION)
    if nuevos:
        print(
            f"\nBARRERA DEL RESPALDO DE USO: la suite escribió {len(nuevos)} "
            f"archivo(s) en {RESPALDO_DE_PRODUCCION}, el directorio REAL del que "
            f"jax-platform drena e inserta en axioma_usage. Esas filas se cobrarían "
            f"a un tenant de verdad.\n"
            f"  primeros: {[p.name for p in nuevos[:5]]}\n"
            f"  el test que las escribió no respetó JAX_USAGE_SPOOL_DIR "
            f"(¿un monkeypatch.delenv, o un subproceso sin el entorno?).",
            file=sys.stderr,
        )
        session.exitstatus = 1
    freno = cambios_del_freno_de_produccion(
        FRENO_DE_PRODUCCION, HEREDADA_DE_PRODUCCION, INICIO_DE_SESION,
        FRENO_EXISTIA_AL_INICIO, HEREDADA_EXISTIA_AL_INICIO)
    if freno:
        print(
            f"\nBARRERA DEL KILL SWITCH: la suite tocó el freno de producción: {freno}. "
            f"Un freno puesto ahí detiene a JAX de verdad; uno borrado lo suelta.",
            file=sys.stderr,
        )
        session.exitstatus = 1
