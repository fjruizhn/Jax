"""Los `run:` de los workflows tienen que ser bash válido.

Por qué existe: el 2026-09-20, resolviendo un conflicto de merge en el piso de
`tests-puros`, se borraron los dos `grep -qE ... || {` y se dejaron los dos
`echo ...; exit 1; }`. Quedó una llave abierta sin cerrar y dos huérfanas. El
job murió con `syntax error near unexpected token '}'` y **exit 2**, o sea sin
llegar a comprobar nada de lo que tenía que comprobar.

Lo peor fue la verificación que se hizo antes de subirlo: `yaml.safe_load`
sobre el archivo, que dio OK. **El YAML parseaba perfecto con el bash roto
adentro**, porque para el YAML un `run:` es una cadena y nada más. Parsear no
es validar --- la misma familia del `EXPLAIN` sobre una tabla vacía y del
caché que "existe" pero no se usa: el control responde que sí a una pregunta
que no era la que había que hacer.

Un bloque roto así no falla ruidosamente donde se lo ve: hace que el job
muera ANTES de correr su comprobación. Un piso que no llega a compararse es un
piso que no existe, y el verde del resto de la matriz tapa el hueco.

El control se ejercita a sí mismo (`test_el_detector_atrapa_el_bloque_roto`):
un detector que no atrapa nada da un verde que no significa nada.
"""
import pathlib
import subprocess
import tempfile
import unittest

import yaml

RAIZ = pathlib.Path(__file__).resolve().parents[2]
WORKFLOWS = RAIZ / ".github" / "workflows"

# El bloque exacto que se rompió el 2026-09-20, reducido a lo mínimo.
BLOQUE_ROTO = """\
echo hola
  echo "PISO ROTO: faltaba su grep"; exit 1; }
grep -qE "^1 passed" /tmp/out || {
"""


def _bash_valido(guion):
    """(ok, error). Usa `bash -n`: analiza sintaxis sin ejecutar una sola línea."""
    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as f:
        f.write(guion)
        ruta = f.name
    try:
        r = subprocess.run(["bash", "-n", ruta], capture_output=True, text=True)
        return r.returncode == 0, r.stderr.strip()
    finally:
        pathlib.Path(ruta).unlink(missing_ok=True)


def _pasos_con_run():
    """Cada paso `run:` de cada job de cada workflow, con su procedencia."""
    for wf in sorted(WORKFLOWS.glob("*.yml")) + sorted(WORKFLOWS.glob("*.yaml")):
        datos = yaml.safe_load(wf.read_text(encoding="utf-8")) or {}
        for nombre_job, job in (datos.get("jobs") or {}).items():
            for i, paso in enumerate(job.get("steps") or []):
                guion = paso.get("run")
                if guion:
                    yield wf.name, nombre_job, i, paso.get("name", "sin nombre"), guion


class WorkflowsBashValidoTest(unittest.TestCase):
    def test_todos_los_run_son_bash_valido(self):
        rotos = []
        revisados = 0
        for wf, job, i, nombre, guion in _pasos_con_run():
            revisados += 1
            ok, err = _bash_valido(guion)
            if not ok:
                ultima = err.splitlines()[-1] if err else "(sin detalle)"
                rotos.append(f"{wf} :: job {job} :: paso #{i} ({nombre}): {ultima}")

        self.assertGreater(revisados, 0, "no se encontró ningún `run:`: el control no está mirando nada")
        self.assertEqual(
            rotos, [],
            "hay `run:` que no son bash válido -- el job muere ANTES de comprobar "
            "lo suyo y el piso nunca llega a compararse:\n  " + "\n  ".join(rotos),
        )

    def test_el_detector_atrapa_el_bloque_roto(self):
        """El control negativo. Si esto pasa, el de arriba no prueba nada."""
        ok, err = _bash_valido(BLOQUE_ROTO)
        self.assertFalse(ok, "bash -n aceptó el bloque que rompió el CI el 2026-09-20")
        self.assertIn("}", err)

    def test_un_bloque_bien_formado_no_da_falso_positivo(self):
        """El control tiene que distinguir, no solo rechazar."""
        ok, _ = _bash_valido(
            'grep -qE "^1 passed" /tmp/out || {\n'
            '  echo "PISO ROTO"; exit 1; }\n'
        )
        self.assertTrue(ok, "el detector rechaza bash correcto: castiga al que escribe bien")


if __name__ == "__main__":
    unittest.main()
