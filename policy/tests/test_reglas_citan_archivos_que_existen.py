"""Toda ruta de ESTE repo citada en una regla de política existe (código, docs, config).

POR QUE EXISTE (2026-09-23). jax#262 renombró `las_manos/policy.py` a
`las_manos/motor_de_politica.py`. P04 y su `policy/generated/CORPUS.md` siguieron
citando el archivo viejo: recordaban falso con autoridad, y nada se puso rojo
porque P04 es NORMATIVA_PENDIENTE sin mecanismo. Esto no juzga la regla, solo
que lo que cita como evidencia siga existiendo.

Solo rutas que empiezan por una carpeta de código de este repo. Las de
jax-platform (otro repo) y las relativas (`tools/…` dentro de policy/) no se
pueden resolver sin adivinar, y adivinar da rojos falsos.
"""
import re
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
REGLAS = RAIZ / "policy" / "rules"
RAICES_DE_CODIGO = ("las_manos", "jax", "jacobs", "policy", "scripts", "procesamiento",
                    "jaxctl", "loadtest", "ops", "config", "tests", "docs")
# Fuera a propósito: rutas de OTRAS instalaciones (`/opt/jax/docs/…`, citadas con
# su sha256 como texto verbatim), `~/…`, `./…` y URLs: no hay repo en CI contra el
# cual comprobarlas, y adivinar da rojos falsos.
EXTENSIONES = "py|md|toml|yaml|yml|json|sh"
# Ruta relativa a una carpeta de código de este repo, o absoluta bajo el checkout
# (/home/fruiz/jax/…). El lookbehind excluye `:` para que una cita HISTÓRICA con
# revisión (`f6c8e7d^:missions/…`, archivo que ya no está en HEAD) no se exija
# en el árbol: en CI el checkout es superficial y la revisión no está.
RUTA = re.compile(r"(?<![\w/.:-])((?:%s)/[\w./-]+\.(?:%s))\b" % ("|".join(RAICES_DE_CODIGO), EXTENSIONES))
ABSOLUTA = re.compile(r"/home/fruiz/jax/([\w./-]+\.(?:%s))\b" % EXTENSIONES)


def _citas():
    for regla in sorted(REGLAS.glob("*.yaml")):
        for n, linea in enumerate(regla.read_text(encoding="utf-8").splitlines(), 1):
            for ruta in RUTA.findall(linea) + ABSOLUTA.findall(linea):
                yield f"{regla.name}:{n}", ruta


def test_hay_citas_que_revisar():
    # Si el patrón deja de encontrar nada, el test de abajo da verde sin mirar.
    assert sum(1 for _ in _citas()) >= 5


def test_toda_ruta_citada_existe():
    faltan = [f"{donde} -> {ruta}" for donde, ruta in _citas() if not (RAIZ / ruta).exists()]
    assert faltan == [], "reglas que citan archivos que ya no existen:\n" + "\n".join(faltan)
