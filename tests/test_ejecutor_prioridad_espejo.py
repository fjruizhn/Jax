# tests/test_ejecutor_prioridad_espejo.py
"""SP3 (2026-09-17): la Mesa toma `carril_mesa_async` desde jax-platform con una COPIA verbatim
de `jax/ejecutor/prioridad.py` (backend/ejecutor/prioridad.py). No se importa jax: api/chat.py
pone en sys.path el checkout de producción de jax, que puede ir detrás, y el runner de
jax-platform no lo tiene. Si la copia diverge, la Mesa y el proxy dejan de hablar el mismo
protocolo de locks y la prioridad se pierde EN SILENCIO (la Mesa no espera nada, el Ejecutor
no ve a la Mesa). Por eso la familia cubre TODOS los símbolos del archivo, constantes incluidas:
una familia a medias da verde sin mirar lo que falta."""
import ast
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]


def _familias():
    sys.path.insert(0, str(RAIZ / "scripts"))
    from check_mirror_sync import FAMILIAS  # noqa: E402
    return {f.nombre: f for f in FAMILIAS}


def _simbolos(ruta: Path) -> set[str]:
    nombres = set()
    for nodo in ast.parse(ruta.read_text(encoding="utf-8")).body:
        if hasattr(nodo, "name"):
            nombres.add(nodo.name)
        elif isinstance(nodo, ast.Assign) and len(nodo.targets) == 1 and isinstance(nodo.targets[0], ast.Name):
            nombres.add(nodo.targets[0].id)
    return nombres


def test_prioridad_es_una_familia_de_espejos_completa():
    familia = _familias().get("prioridad")
    assert familia is not None
    assert familia.canonico == RAIZ / "jax" / "ejecutor" / "prioridad.py"
    assert [(e, r.parts[-3:]) for e, r in familia.espejos] == [("jax-platform", ("backend", "ejecutor", "prioridad.py"))]
    assert _simbolos(familia.canonico) == set(familia.compartidos)


def test_motivo_que_usa_la_copia_tambien_es_familia():
    # prioridad.py importa Motivo de cita.py; la copia lo importa de backend/ejecutor/motivo.py.
    # Los ImportFrom no se comparan: sin esta familia, un Motivo distinto pasaría callado.
    familia = _familias().get("motivo")
    assert familia is not None
    assert familia.canonico == RAIZ / "jax" / "ejecutor" / "cita.py"
    assert familia.compartidos == ("Motivo",)
    assert [(e, r.parts[-3:]) for e, r in familia.espejos] == [("jax-platform", ("backend", "ejecutor", "motivo.py"))]


def test_la_pausa_del_ejecutor_que_escribe_la_plataforma_es_familia():
    # SP2 (2026-09-17): el modo Ejecutor de jax-platform PONE la pausa del Ejecutor (el kill switch
    # del modo) con una copia de pausa.py: la plataforma no importa jax (mismo motivo que prioridad).
    # El proxy de C3 y el arranque la LEEN con este archivo. Un `pausa_puesta` o un `poner_pausa`
    # distintos serían un botón que dice «pausado» mientras el proxy sigue sirviendo.
    familia = _familias().get("pausa_ejecutor")
    assert familia is not None
    assert familia.canonico == RAIZ / "jax" / "ejecutor" / "contratos" / "pausa.py"
    assert [(e, r.parts[-3:]) for e, r in familia.espejos] == [("jax-platform", ("backend", "ejecutor", "pausa.py"))]
    assert set(familia.compartidos) == {"VARIABLE_RUTA", "PausaSinConfigurar", "pausa_puesta",
                                        "_sincronizar_directorio", "poner_pausa"}
    assert set(familia.compartidos) <= _simbolos(familia.canonico)
