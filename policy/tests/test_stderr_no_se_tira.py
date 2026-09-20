#!/usr/bin/env python3
"""Ningun subproceso tira su stderr sin un motivo ESCRITO.

**Por que existe (2026-09-20).** En un solo dia, `stderr=DEVNULL` escondio el
diagnostico CINCO veces:

  1. el vigia de C5 -- una mision fallo con `vigia_no_latio` y no habia una linea
     para investigar; el diagnostico salio corriendo el vigia a mano;
  2. el runner de contratos -- `runner_sin_cierre` sin explicacion;
  3. el binario de `/api/command` -- `comando_sin_resultado` sin explicacion;
  4. los workers desprendidos de memoria -- usan `logging.basicConfig`, que escribe
     en stderr, asi que se tiraba TODO su registro;
  5. y el propio `abrir_vigia`, que fue el que lo destapo.

Arreglar los cinco no impide el sexto. Esto si: un `stderr=DEVNULL` nuevo rompe CI
salvo que su archivo este aca abajo con su motivo. No prohibe DEVNULL -- hay casos
donde es correcto-- obliga a ESCRIBIR POR QUE.

**Lo que aprendimos y no se puede perder:** cambiar DEVNULL por PIPE a secas
BLOQUEA al proceso. El pipe del sistema son ~64 KB; sin nadie drenando, el hijo se
cuelga en su propio `write`. Si hay que capturar, se drena en continuo (vigia,
runner) o se usa `communicate()` cuando nadie mas lee los flujos (command). Si el
proceso es DESPRENDIDO y sobrevive al padre, va a ARCHIVO: un pipe muere con el
padre (workers de memoria).
"""
import pathlib
import re
import unittest

RAIZ = pathlib.Path(__file__).resolve().parents[2]

#: archivo -> por que tirar el stderr es CORRECTO ahi. Sin entrada, CI falla.
PERMITIDOS = {
    "scripts/ejecutor_contratos/probar_c4.py":
        "Procesos CENTINELA (`sleep`) para contar procesos vivos en la prueba de C4. Su "
        "salida es irrelevante --lo que se mide es CUANTOS hay-- y el propio comando ya "
        "redirige `>/dev/null 2>&1` adentro del shell remoto.",
    "scripts/ejecutor_contratos/probar_c6.py":
        "Igual que probar_c4: un `sleep` centinela para probar la revocacion de C6. No "
        "produce salida que signifique algo.",
}

_DEVNULL = re.compile(r"stderr\s*=\s*(?:asyncio\.)?subprocess\.DEVNULL")


def _archivos():
    for ruta in RAIZ.rglob("*.py"):
        rel = ruta.relative_to(RAIZ).as_posix()
        if rel.startswith((".venv/", "tests/", "policy/tests/")) or "/tests/" in rel or rel.endswith("_test.py"):
            continue
        yield rel, ruta


class StderrNoSeTira(unittest.TestCase):
    def test_todo_stderr_a_devnull_tiene_motivo_escrito(self):
        culpables = {rel for rel, ruta in _archivos()
                     if _DEVNULL.search(ruta.read_text(encoding="utf-8", errors="replace"))}
        nuevos = culpables - set(PERMITIDOS)
        self.assertFalse(nuevos, f"stderr=DEVNULL sin motivo escrito: {sorted(nuevos)}. "
                                 "Si es correcto, agregalo a PERMITIDOS con su porque.")

    def test_no_sobran_permitidos(self):
        """Un permitido que ya no tira stderr es una excepcion vencida: se borra, para
        que la lista siga significando algo."""
        culpables = {rel for rel, ruta in _archivos()
                     if _DEVNULL.search(ruta.read_text(encoding="utf-8", errors="replace"))}
        self.assertFalse(set(PERMITIDOS) - culpables,
                         f"permitidos que ya no aplican: {sorted(set(PERMITIDOS) - culpables)}")

    def test_cada_motivo_dice_algo(self):
        for archivo, motivo in PERMITIDOS.items():
            self.assertGreater(len(motivo), 60, f"{archivo}: el motivo es demasiado corto para servir")

    def test_el_detector_ve_una_violacion_nueva(self):
        """Un control que no falla no valida."""
        self.assertTrue(_DEVNULL.search("proc = Popen(x, stderr=subprocess.DEVNULL)"))
        self.assertTrue(_DEVNULL.search("stderr=asyncio.subprocess.DEVNULL, start_new_session=True"))
        self.assertFalse(_DEVNULL.search("stdout=subprocess.DEVNULL"))


if __name__ == "__main__":
    unittest.main()
