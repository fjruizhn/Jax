"""Migración de los pisos de policy.yml a ci/pisos.json: nada cambió de valor ni de nombre.

La fotografía `fixtures/policy_master_364ded9.json` se extrajo UNA vez de
`git show 364ded9:.github/workflows/policy.yml` (master antes de la migración) con un
extractor independiente del que escribió los datos (yaml.safe_load + regex sobre cada
`run:`). Estas pruebas comparan el estado actual con esa fotografía:

  1. los nombres de los jobs (id y `name:`) son IDÉNTICOS, y los de los pasos de cada job
     conservan los de master en el mismo orden (un paso nuevo solo puede ir al final);
     el nombre de un job es el nombre del check que ve GitHub;
  2. cada piso de master está en el mismo paso, con el mismo patrón, el mismo archivo de
     salida y el mismo mensaje, BYTE A BYTE, y el mínimo del B9 vale igual.

OJO: son pruebas de MIGRACIÓN. Cuando un piso suba o un paso se renombre a propósito, estas
pruebas fallarán por diseño: se actualizan o se retiran (ver docs/ci/pisos.md).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

RAIZ = Path(__file__).resolve().parents[2]
FOTO = json.loads((Path(__file__).parent / "fixtures" / "policy_master_364ded9.json").read_text(encoding="utf-8"))
ACTUAL = yaml.safe_load((RAIZ / ".github" / "workflows" / "policy.yml").read_text(encoding="utf-8"))
DATOS = json.loads((RAIZ / "ci" / "pisos.json").read_text(encoding="utf-8"))
LLAMADA = re.compile(r"python3 \.github/ci/piso\.py verificar (\S+) (\S+)")


def test_los_jobs_son_los_mismos():
    assert sorted(ACTUAL["jobs"]) == sorted(FOTO["jobs"])


@pytest.mark.parametrize("jid", sorted(FOTO["jobs"]))
def test_nombre_del_job_identico(jid):
    assert ACTUAL["jobs"][jid].get("name") == FOTO["jobs"][jid]["name"]


@pytest.mark.parametrize("jid", sorted(FOTO["jobs"]))
def test_los_pasos_de_master_siguen_con_su_nombre_y_orden(jid):
    ahora = [s.get("name") for s in ACTUAL["jobs"][jid]["steps"]]
    antes = FOTO["jobs"][jid]["pasos"]
    assert ahora[:len(antes)] == antes


@pytest.mark.parametrize("piso", FOTO["pisos"], ids=lambda p: f"{p['job']}#{p['paso']}")
def test_piso_de_master_copiado_byte_a_byte(piso):
    paso = ACTUAL["jobs"][piso["job"]]["steps"][piso["paso"]]
    llamadas = LLAMADA.findall(paso["run"])
    assert len(llamadas) == 1, "el paso que tenía un piso tiene que llamar exactamente una vez a piso.py"
    clave, archivo = llamadas[0]
    assert archivo == piso["archivo"]
    entrada = DATOS["pisos"][clave]
    assert entrada["patron"] == piso["patron"]
    assert entrada["mensaje"] == piso["mensaje"]


def test_los_42_pisos_de_master_tienen_cada_uno_su_clave():
    migrados = {LLAMADA.findall(ACTUAL["jobs"][p["job"]]["steps"][p["paso"]]["run"])[0][0] for p in FOTO["pisos"]}
    assert len(migrados) == len(FOTO["pisos"]) == 42


def test_minimo_del_b9_igual_al_de_master():
    assert DATOS["minimos"]["memory-b9-regression/casos"] == FOTO["minimo_b9"] == 102
