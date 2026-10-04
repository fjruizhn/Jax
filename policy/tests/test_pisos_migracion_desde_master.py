"""Migración de los pisos de policy.yml a ci/pisos.json: nada cambió de valor ni de nombre.

La fotografía `fixtures/policy_master_364ded9.json` se extrajo UNA vez de
`git show 364ded9:.github/workflows/policy.yml` (master antes de la migración) con un
extractor independiente del que escribió los datos (yaml.safe_load + regex sobre cada
`run:`). Estas pruebas comparan el estado actual con esa fotografía:

  1. los nombres de los jobs (id y `name:`) son IDÉNTICOS, y los de los pasos de cada job
     conservan los de master en el mismo orden (un paso nuevo solo puede ir al final);
     el nombre de un job es el nombre del check que ve GitHub;
  2. cada piso de master está en el mismo paso, con su clave en ci/pisos.json, el mismo archivo de
     salida y un patrón de la MISMA forma, con N (passed) mayor o igual y M (skipped) menor o igual
     al de la fotografía, y el mínimo del B9 mayor o igual a 102.

La regla es genérica (vale para cada clave de la fotografía): un piso que SUBE no hace fallar estas
pruebas, y el número de pruebas no cambia (importa: el piso de `archivos-de-test-en-ci/pisos` cuenta
estas pruebas). Que ningún piso baje lo cubre también el job `pisos-no-bajan` (ver docs/ci/pisos.md).
Un paso que se renombre a propósito sí las hace fallar por diseño.
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


def _partes(patron: str) -> tuple[str, int, int | None]:
    """(forma, N, M) de un patron de piso: la FORMA es el patron con N (el primer numero, los tests `passed`) y M
    (el que precede a ` skipped`, si lo hay) cambiados por marcas; cualquier otro numero se queda literal, asi que
    tiene que ser IGUAL para que la forma coincida."""
    numeros = list(re.finditer(r"\d+", patron))
    assert numeros, f"el patron {patron!r} no trae ningun numero"
    n = numeros[0]
    m = next((x for x in numeros[1:] if patron[x.end():].startswith(" skipped")), None)
    forma = patron
    for x, marca in sorted(((x, mk) for x, mk in ((n, "<N>"), (m, "<M>")) if x is not None),
                           key=lambda par: -par[0].start()):
        forma = forma[:x.start()] + marca + forma[x.end():]
    return forma, int(n.group()), (int(m.group()) if m is not None else None)


@pytest.mark.parametrize("piso", FOTO["pisos"], ids=lambda p: f"{p['job']}#{p['paso']}")
def test_piso_de_master_existe_con_su_forma_y_no_bajo(piso):
    """Regla GENÉRICA para cada clave de la fotografía (nadie amplía una lista cuando un piso sube): la clave existe
    en ci/pisos.json y el paso la llama una vez con el mismo archivo temporal; el patrón tiene la MISMA forma; N (passed)
    del head >= el de la fotografía; M (skipped) del head <= el de la fotografía, si la forma lo tiene. El valor exacto
    ya no se compara byte a byte: subir un piso es legítimo y que ninguno baje lo cubre también `pisos-no-bajan`."""
    paso = ACTUAL["jobs"][piso["job"]]["steps"][piso["paso"]]
    llamadas = LLAMADA.findall(paso["run"])
    assert len(llamadas) == 1, "el paso que tenía un piso tiene que llamar exactamente una vez a piso.py"
    clave, archivo = llamadas[0]
    assert archivo == piso["archivo"], "el archivo temporal que lee el piso cambió"
    assert clave in DATOS["pisos"], f"la clave {clave} ya no está en ci/pisos.json"
    entrada = DATOS["pisos"][clave]
    forma_antes, n_antes, m_antes = _partes(piso["patron"])
    forma_ahora, n_ahora, m_ahora = _partes(entrada["patron"])
    assert forma_ahora == forma_antes, f"cambió la forma del patrón: {piso['patron']!r} -> {entrada['patron']!r}"
    assert n_ahora >= n_antes, f"N (passed) bajó de {n_antes} a {n_ahora}"
    if m_antes is not None:
        assert m_ahora is not None and m_ahora <= m_antes, f"M (skipped) subió de {m_antes} a {m_ahora}"
    assert entrada["mensaje"].strip(), "el piso no tiene mensaje"


def test_los_42_pisos_de_master_tienen_cada_uno_su_clave():
    migrados = {LLAMADA.findall(ACTUAL["jobs"][p["job"]]["steps"][p["paso"]]["run"])[0][0] for p in FOTO["pisos"]}
    assert len(migrados) == len(FOTO["pisos"]) == 42


def test_minimo_del_b9_no_bajo_de_el_de_master():
    assert FOTO["minimo_b9"] == 102
    assert DATOS["minimos"]["memory-b9-regression/casos"] >= FOTO["minimo_b9"]
