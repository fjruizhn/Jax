# tests/test_ejecutor_vigia_stderr.py
"""El stderr del vigia no se tira a la basura.

**Por que existe (2026-09-20).** `abrir_vigia` lanzaba el subproceso con
`stderr=asyncio.subprocess.DEVNULL`. Cuando una mision fallo con `vigia_no_latio`,
NO habia una sola linea para investigar: ni en el journal, ni en la bitacora, ni
en el resultado del turno. El diagnostico salio corriendo el vigia A MANO con el
entorno completo, que es exactamente lo que un log existe para evitar.

Fue la TERCERA vez en el mismo dia que un `stderr=DEVNULL` escondio el
diagnostico (las otras dos: el runner de contratos y el arranque del vigia).

**La trampa que este diseno evita.** Cambiar `DEVNULL` por `PIPE` a secas
BLOQUEA al vigia: escribe INFO de httpx en stderr (medido: tres lineas por lote
auditado), el pipe del sistema son ~64 KB, y un turno largo lo llena. Con nadie
drenando, el vigia se cuelga en su propio write -- y el sintoma seria
`vigia_no_latio`, el MISMO fallo que esto viene a poder diagnosticar. Por eso hay
un drenaje continuo con tope: se conserva la COLA, que es donde esta el error.
"""
import asyncio

import pytest

from jax.ejecutor import mision_servicio as S


def _correr(guion, tmp_path, tope_s=30):
    """Abrir y cerrar van en el MISMO bucle de eventos: el drenaje es una tarea de
    asyncio atada a su loop, y en produccion las dos cosas pasan en el mismo turno.
    Separarlas en dos `asyncio.run` da `Future attached to a different loop` -- que es
    un defecto del arnes, no del codigo."""
    async def escenario():
        v = await S.abrir_vigia(str(tmp_path), "m", "mision", ["bridge"],
                                argv=["python3", "-c", guion])
        # Se deja terminar al proceso ANTES de cerrar, como en produccion: el vigia
        # corre durante todo el turno y `cerrar()` llega despues. Sin esto, el SIGTERM
        # de `cerrar()` lo mata antes de que escriba -- un defecto del arnes, no del
        # codigo. El tope evita colgarse si el drenaje no funcionara.
        await asyncio.wait_for(v._proc.wait(), tope_s)
        return await asyncio.wait_for(v.cerrar(), tope_s)
    return asyncio.run(escenario())


def test_el_stderr_llega_al_cierre(tmp_path):
    """Lo que faltaba el 2026-09-20: una linea para investigar."""
    rc, salida, err = _correr("import sys; sys.stderr.write('ModuleNotFoundError: no existe tal cosa\\n'); "
                  "sys.stdout.write('cerrada=false\\n')", tmp_path)
    assert "ModuleNotFoundError" in err, err


def test_un_vigia_que_escribe_MUCHO_no_se_bloquea(tmp_path):
    """LA prueba de este diseno. Sin drenaje, el proceso se cuelga al llenar el
    pipe (~64 KB) y `cerrar()` se iria al tope. Con drenaje, termina."""
    rc, salida, err = _correr("import sys\n"
                  "for i in range(20000): sys.stderr.write('INFO:httpx:HTTP Request: POST ... 200 OK\\n')\n"
                  "sys.stdout.write('cerrada=true\\n')", tmp_path)
    assert rc == 0
    assert "cerrada=true" in salida


def test_se_conserva_la_COLA_no_la_cabeza(tmp_path):
    """El error esta al final: una traza aparece DESPUES de mil lineas de INFO.
    Guardar la cabeza seria guardar el ruido y tirar el diagnostico."""
    rc, salida, err = _correr("import sys\n"
                  "for i in range(20000): sys.stderr.write('ruido\\n')\n"
                  "sys.stderr.write('Traceback (most recent call last): EL ERROR\\n')\n"
                  "sys.stdout.write('cerrada=true\\n')", tmp_path)
    assert "EL ERROR" in err, err[-200:]
    assert len(err) <= S.TOPE_STDERR_VIGIA


def test_los_secretos_no_salen_en_el_stderr(tmp_path):
    """Un traceback puede traer la llave. `redaccion` ya existe para esto; lo que
    faltaba era usarla acá."""
    rc, salida, err = _correr("import sys; sys.stderr.write('falló con Authorization: Bearer sk-abc123secreto\\n'); "
                  "sys.stdout.write('cerrada=true\\n')", tmp_path)
    assert "sk-abc123secreto" not in err, err


def test_sin_stderr_devuelve_vacio_no_None(tmp_path):
    """Quien lee no tiene que defenderse de un None."""
    rc, salida, err = _correr("import sys; sys.stdout.write('cerrada=true\\n')", tmp_path)
    assert err == ""
