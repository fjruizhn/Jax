"""La suite de `jax` NO puede escribir en el respaldo de uso de PRODUCCIÓN.

POR QUÉ EXISTE. El 2026-09-15, al cablear la cola durable (Task 7), la suite
depositó 116 archivos en `/srv/jax-data/usage-spool` -- el directorio REAL del
que `jax-platform` drena e inserta en `axioma_usage`. No fue un test nuevo: fue
que los tests que ya existían (`las_manos/_worker_max_tokens_test.py`,
`tests/test_motor_job_cancel_and_length.py`, ...) ejercitan `record_motor_usage`
contra una base inalcanzable, y desde este cambio ese camino ya no tira la fila:
la encola. Con el default del módulo, "encolar" es el directorio de producción.
Si esos archivos siguieran ahí cuando la Task 3 empiece a drenar, se insertan
filas de uso INVENTADAS en la tabla de cobro.

El aislamiento vive en el `conftest.py` de la raíz, en un `os.environ[...]` de
tiempo de import: un fixture que hay que pedir no sirve: el problema es
justamente el test que no sabe que está escribiendo ahí.

Y el aislamiento no alcanza con declararlo. Este archivo es el freno probado:
un detector que se ejercita contra una escritura de mentira (si no falla cuando
debe, su verde no significa nada) y el chequeo real sobre el directorio de
producción, que además corre en `pytest_sessionfinish` -- ahí sí después de
TODOS los tests, sin depender del orden de colección.

Corre con:
  PYTHONPATH=.:las_manos python -m pytest tests/test_respaldo_de_uso_aislado.py -v
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import conftest


def test_el_conftest_manda_el_respaldo_a_un_tempdir():
    """Y no al default del módulo, que es el directorio de producción."""
    crudo = os.environ.get("JAX_USAGE_SPOOL_DIR")
    assert crudo, "JAX_USAGE_SPOOL_DIR no está seteado: la suite escribiría en el default"
    ruta = Path(crudo)
    assert ruta.resolve() != conftest.RESPALDO_DE_PRODUCCION.resolve()
    assert ruta.is_dir(), ruta


def test_el_modulo_resuelve_al_tempdir_y_no_al_default():
    """La prueba de que la variable es la que manda: se le pregunta al módulo,
    no a la convención."""
    import cola_uso

    assert cola_uso._ruta_configurada().resolve() == Path(
        os.environ["JAX_USAGE_SPOOL_DIR"]
    ).resolve()
    assert cola_uso.DIRECTORIO_POR_DEFECTO == conftest.RESPALDO_DE_PRODUCCION


def test_el_detector_ATRAPA_una_escritura_nueva(tmp_path):
    """Un control que no falla cuando debe no valida nada. Se ejercita contra
    un directorio de mentira, no contra el real: probar el freno no puede
    ensuciar justo lo que el freno protege.

    Los tiempos van FIJADOS con `os.utime`, no leídos del reloj: el mtime lo
    estampa el reloj grueso del kernel (un tick atrás de `time.time()`), así
    que escribir justo después de tomar `desde` podía dar mtime < desde y el
    test fallaba sin que el detector lo hiciera (visto una vez en CI). En
    segundos enteros para que el float de `st_mtime` sea exactamente `desde`.
    La escritura nueva cae JUSTO en el borde (mtime == desde: cuenta), y al
    lado hay una anterior que no cuenta: un detector que no mire el mtime
    devuelve las dos y cae."""
    falso = tmp_path / "usage-spool"
    falso.mkdir()
    desde_ns = (time.time_ns() // 10**9) * 10**9
    desde = desde_ns / 1e9
    viejo = falso / "anterior.json"
    viejo.write_text("{}", encoding="utf-8")
    os.utime(viejo, ns=(desde_ns - 10**9, desde_ns - 10**9))
    nuevo = falso / "abc.json"
    nuevo.write_text("{}", encoding="utf-8")
    os.utime(nuevo, ns=(desde_ns, desde_ns))
    encontrados = conftest.archivos_nuevos_en(falso, desde)
    assert [p.name for p in encontrados] == ["abc.json"], encontrados


def test_el_detector_NO_cuenta_lo_que_ya_estaba(tmp_path):
    """Los archivos anteriores a la sesión no son de la suite -- contarlos
    pondría el freno en rojo permanente y se terminaría ignorando."""
    falso = tmp_path / "usage-spool"
    falso.mkdir()
    viejo = falso / "viejo.json"
    viejo.write_text("{}", encoding="utf-8")
    os.utime(viejo, (time.time() - 3600, time.time() - 3600))
    assert conftest.archivos_nuevos_en(falso, time.time() - 60) == []


def test_un_directorio_que_no_existe_no_es_una_escritura(tmp_path):
    """En el runner de CI `/srv/jax-data` no existe. Que no exista es cero
    escrituras, no un error que tape el chequeo."""
    assert conftest.archivos_nuevos_en(tmp_path / "no-existe", 0.0) == []


def test_NINGUN_test_escribio_en_el_respaldo_de_produccion():
    """El chequeo real. Se repite en `pytest_sessionfinish` (conftest): acá
    depende del orden de colección, allá no."""
    nuevos = conftest.archivos_nuevos_en(
        conftest.RESPALDO_DE_PRODUCCION, conftest.INICIO_DE_SESION
    )
    assert nuevos == [], (
        f"{len(nuevos)} archivo(s) escritos por la suite en el respaldo de "
        f"PRODUCCIÓN {conftest.RESPALDO_DE_PRODUCCION}: {[p.name for p in nuevos[:5]]}"
    )
