#!/usr/bin/env python3
"""Un artifact que no se puede leer NO puede salir como un paso exitoso.

POR QUÉ EXISTE (auditoría P10, 2026-09-16). `_load_ref()` devolvía `{}` tanto
cuando no había ref como cuando el artifact era ilegible, y ese `{}` viajaba río
abajo como si fuera la salida real de la dependencia:

  - `_build_context_input` lo convertía en la cadena `'{}'` y se la entregaba al
    step siguiente como el output de aquello de lo que DEPENDE, y el step corría
    igual y terminaba `completed`.
  - `_assemble_mechanical` concatenaba un módulo VACÍO al documento final, que
    se devolvía con `success: True`.
  - `routes._resolve_ref` devolvía `("", [])` sin una línea de log, y el usuario
    veía un paso `completed` con resultado vacío, indistinguible de uno que
    legítimamente no produjo texto.

Error tragado -> paso completado con producto incorrecto. Es el patrón exacto
que P10 prohíbe, y en el sitio más caro: el paquete que se entrega.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jacobs import executor, routes  # noqa: E402
from jacobs.models import Pipeline, Step  # noqa: E402


def _pipeline_con_dos_steps(depends_on: list[int]) -> tuple[Pipeline, Step]:
    s0 = Step(step_index=0, facet="hipatia", capability="research")
    s1 = Step(step_index=1, facet="ada", capability="design", depends_on=depends_on)
    p = Pipeline(name="t", invoked_by="test", mode="auto", plan=[s0, s1])
    p.context["step_0_ref"] = "artifact://roto/no-existe.json"
    return p, s1


class LoadRefTest(unittest.TestCase):
    def test_sin_ref_devuelve_vacio_sin_lanzar(self):
        self.assertEqual(executor._load_ref(""), {})

    def test_artifact_ilegible_lanza_en_vez_de_devolver_vacio(self):
        with mock.patch.object(executor, "read_artifact", side_effect=OSError("disco")):
            with self.assertRaises(executor.RefIlegible):
                executor._load_ref("artifact://x/y.json")

    def test_inline_roto_lanza(self):
        with self.assertRaises(executor.RefIlegible):
            executor._load_ref("inline:{esto no es json")

    def test_formato_desconocido_lanza(self):
        with self.assertRaises(executor.RefIlegible):
            executor._load_ref("vete-a-saber://x")


class ContextoDeDependenciaTest(unittest.TestCase):
    def test_una_dependencia_declarada_ilegible_corta_el_step(self):
        """Antes se sustituía por '[ref: ...]' con truncated=False y el step
        corría sin su dependencia, terminando `completed`."""
        pipeline, step = _pipeline_con_dos_steps(depends_on=[0])
        with mock.patch.object(executor, "read_artifact", side_effect=OSError("disco")):
            with self.assertRaises(executor.RefIlegible):
                executor._build_context_input(step, pipeline)

    def test_el_contexto_opcional_ilegible_se_declara_pero_no_corta(self):
        """Sin `depends_on` es contexto de cortesía: se sigue, pero diciéndolo
        —— nunca fingiendo que ese texto es la salida del paso anterior."""
        pipeline, step = _pipeline_con_dos_steps(depends_on=[])
        with mock.patch.object(executor, "read_artifact", side_effect=OSError("disco")):
            datos = executor._build_context_input(step, pipeline)
        previos = datos["previous_outputs"]
        self.assertEqual(len(previos), 1)
        self.assertTrue(previos[0]["perdido"])
        self.assertIn("no disponible", previos[0]["summary"])
        self.assertNotIn("{}", previos[0]["summary"])


class EnsambleTest(unittest.TestCase):
    def test_un_modulo_ilegible_hace_que_el_paquete_NO_sea_exitoso(self):
        pipeline, step = _pipeline_con_dos_steps(depends_on=[])
        with mock.patch.object(executor, "read_artifact", side_effect=OSError("disco")):
            salida = executor._assemble_mechanical(step, pipeline)
        self.assertFalse(salida["success"], "se entregó un paquete con un módulo faltante como exitoso")
        self.assertIn("no se pudieron leer", salida.get("error", ""))
        self.assertIn("NO DISPONIBLE", salida["result"])

    def test_el_caso_sano_sigue_siendo_exitoso(self):
        """Control del control: sin módulos perdidos, nada cambia."""
        pipeline, step = _pipeline_con_dos_steps(depends_on=[])
        with mock.patch.object(executor, "read_artifact", return_value={"result": "contenido real"}):
            salida = executor._assemble_mechanical(step, pipeline)
        self.assertTrue(salida["success"])
        self.assertNotIn("error", salida)
        self.assertIn("contenido real", salida["result"])


class ResolveRefTest(unittest.TestCase):
    def test_un_ref_ilegible_devuelve_motivo_y_no_un_vacio_mudo(self):
        with mock.patch.object(routes, "read_artifact", side_effect=OSError("disco")):
            result, sources, error = routes._resolve_ref("artifact://x/y.json")
        self.assertEqual(result, "")
        self.assertEqual(sources, [])
        self.assertIsNotNone(error, "un resultado ilegible se presentó como resultado vacío")
        self.assertIn("no se pudo leer", error)

    def test_sin_ref_no_es_error(self):
        self.assertEqual(routes._resolve_ref(""), ("", [], None))


if __name__ == "__main__":
    unittest.main()
