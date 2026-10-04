"""Nombres de jobs, checks y pasos de policy.yml tras la migración de los pisos a ci/pisos.json.

La fotografía `fixtures/policy_master_364ded9.json` se extrajo UNA vez de
`git show 364ded9:.github/workflows/policy.yml` (master antes de la migración) con un
extractor independiente del que escribió los datos (yaml.safe_load). Estas pruebas comparan el
estado actual con esa fotografía:

  * los nombres de los jobs (id y `name:`) son IDÉNTICOS, y los de los pasos de cada job
    conservan los de master en el mismo orden (un paso nuevo solo puede ir al final);
    el nombre de un job es el nombre del check que ve GitHub.

Los VALORES de los pisos (el patrón, el mensaje y el mínimo del B9) ya no se comparan aquí contra la
fotografía: subir un piso es legítimo, y que ninguno baje lo cubre de forma permanente el job
`pisos-no-bajan` contra la punta de master (ver docs/ci/pisos.md).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

RAIZ = Path(__file__).resolve().parents[2]
FOTO = json.loads((Path(__file__).parent / "fixtures" / "policy_master_364ded9.json").read_text(encoding="utf-8"))
ACTUAL = yaml.safe_load((RAIZ / ".github" / "workflows" / "policy.yml").read_text(encoding="utf-8"))


JOBS_NUEVOS = ["pisos-no-bajan"]  # el comparador de pisos: único job agregado, sin tocar los existentes


def test_los_jobs_son_los_mismos_mas_el_comparador():
    assert sorted(ACTUAL["jobs"]) == sorted([*FOTO["jobs"], *JOBS_NUEVOS])


@pytest.mark.parametrize("jid", sorted(FOTO["jobs"]))
def test_nombre_del_job_identico(jid):
    assert ACTUAL["jobs"][jid].get("name") == FOTO["jobs"][jid]["name"]


@pytest.mark.parametrize("jid", sorted(FOTO["jobs"]))
def test_los_pasos_de_master_siguen_con_su_nombre_y_orden(jid):
    ahora = [s.get("name") for s in ACTUAL["jobs"][jid]["steps"]]
    antes = FOTO["jobs"][jid]["pasos"]
    assert ahora[:len(antes)] == antes
