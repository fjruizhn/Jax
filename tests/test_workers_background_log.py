# tests/test_workers_background_log.py
"""Los workers desprendidos no tiran su log a la basura.

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
import subprocess

from jax.core import main as M


def test_el_log_del_worker_va_a_un_archivo_no_a_devnull(tmp_path, monkeypatch):
    monkeypatch.setenv("JAX_WORKSPACE_DIR", str(tmp_path))
    vistos = []

    def falso_popen(argv, **kw):
        vistos.append(kw)
        class P: pass
        return P()

    monkeypatch.setattr(subprocess, "Popen", falso_popen)
    M._lanzar_workers_background()
    assert vistos, "no se lanzo ningun worker"
    for kw in vistos:
        assert kw["stderr"] is not subprocess.DEVNULL, "el stderr sigue yendo a DEVNULL"
        assert hasattr(kw["stderr"], "write"), kw["stderr"]


def test_los_dos_workers_escriben_en_archivos_DISTINTOS(tmp_path, monkeypatch):
    """Un solo archivo compartido mezclaria dos procesos y haria ilegible el log."""
    monkeypatch.setenv("JAX_WORKSPACE_DIR", str(tmp_path))
    nombres = []
    monkeypatch.setattr(subprocess, "Popen",
                        lambda argv, **kw: nombres.append(getattr(kw["stderr"], "name", None)) or type("P", (), {})())
    M._lanzar_workers_background()
    assert len(set(nombres)) == len(nombres) == 2, nombres


def test_si_el_archivo_no_se_puede_abrir_el_worker_se_lanza_igual(tmp_path, monkeypatch):
    """Perder el log es malo; no lanzar el worker es peor."""
    monkeypatch.setenv("JAX_WORKSPACE_DIR", "/proc/no-se-puede-escribir-aca")
    lanzados = []
    monkeypatch.setattr(subprocess, "Popen", lambda argv, **kw: lanzados.append(argv) or type("P", (), {})())
    M._lanzar_workers_background()
    assert len(lanzados) == 2, lanzados
