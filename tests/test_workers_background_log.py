# tests/test_workers_background_log.py
"""B9 da a systemd la propiedad exclusiva de workers programados.

`_lanzar_workers_background` lanzaba `jax.memory.worker` y
`jax.memory.embedding_worker` con `stdout=DEVNULL` y `stderr=DEVNULL`. Los dos
usan `logging.basicConfig`, que por defecto escribe en **stderr**: con DEVNULL se
tiraba TODO su registro. Si un worker moria al importar, nadie se enteraba.

Quinto caso del mismo patron el 2026-09-20.

**Por que archivo y no pipe.** Estos procesos son DESPRENDIDOS
(`start_new_session=True`) a proposito: sobreviven al cierre de JAX. Un pipe
muere con el padre, asi que la solucion del vigia y del runner no sirve aca.

**Fail-soft, a proposito.** Si el archivo no se puede abrir, se cae a DEVNULL y
el worker se lanza igual: perder el log es malo, no lanzar el worker es peor.
"""
from jax.core import main as M


def test_repl_no_lanza_workers_desprendidos(monkeypatch):
    vistos = []
    monkeypatch.setattr(M.subprocess, "Popen", lambda *a, **kw: vistos.append((a, kw)))
    M._lanzar_workers_background()
    assert not vistos
