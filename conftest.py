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

import os
import sys
import tempfile
import time
from pathlib import Path

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


def pytest_sessionfinish(session, exitstatus):
    """El chequeo que NO depende del orden de colección.

    Un test que mira el directorio de producción sólo ve lo escrito por los
    tests que corrieron ANTES que él. Acá ya corrieron todos.
    """
    nuevos = archivos_nuevos_en(RESPALDO_DE_PRODUCCION, INICIO_DE_SESION)
    if not nuevos:
        return
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
