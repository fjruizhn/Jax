"""Activación con contrato (auditoría única, MAJOR 4): la fila de `axioma_config` se siembra
ANTES de reiniciar LAS MANOS con este código, y la siembra no pisa una decisión ya tomada.

El script nunca corre solo: por defecto imprime el SQL (--simular); aplicarlo exige
`--aplicar` y lo hace una persona con GO. Pasar a duro lo decide Fernando tras U5.
"""
from __future__ import annotations

import asyncio
import importlib.util
import pathlib
import sys

import pytest

RAIZ = pathlib.Path(__file__).resolve().parents[1]
_RUTA = RAIZ / "scripts" / "anti_fabricacion" / "sembrar_modo.py"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("sembrar_modo", _RUTA)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


class _Cur:
    def __init__(self, log, fila):
        self.log, self.fila = log, fila

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def execute(self, sql, args=None):
        self.log.append((sql, args))

    async def fetchone(self):
        return self.fila


class _Conn:
    def __init__(self, log, fila):
        self.log, self.fila = log, fila

    def cursor(self):
        return _Cur(self.log, self.fila)

    def close(self):
        pass

    async def commit(self):
        pass


def _conectar(log, fila=None):
    async def conectar():
        return _Conn(log, fila)
    return conectar


def test_por_defecto_solo_simula_e_imprime_el_sql(mod, capsys):
    log = []
    assert mod.main([], conectar=_conectar(log)) == 0
    salida = capsys.readouterr().out
    assert "INSERT IGNORE INTO axioma_config" in salida and "anti_fabricacion.modo" in salida and "sombra" in salida
    assert log == [], "sin --aplicar no se toca la base"


def test_aplicar_inserta_sombra_sin_pisar_lo_que_ya_hay(mod):
    log = []
    assert mod.main(["--aplicar"], conectar=_conectar(log)) == 0
    sql, args = log[-1]
    assert sql.startswith("INSERT IGNORE INTO axioma_config") and args == ("anti_fabricacion.modo", "sombra")
    assert not any("UPDATE" in q.upper() or "REPLACE" in q.upper().replace("INSERT IGNORE", "") for q, _ in log)


def test_verificar_sale_0_si_la_clave_es_valida_y_1_si_falta_o_es_invalida(mod):
    assert mod.main(["--verificar"], conectar=_conectar([], ("sombra",))) == 0
    assert mod.main(["--verificar"], conectar=_conectar([], ("duro",))) == 0
    assert mod.main(["--verificar"], conectar=_conectar([], None)) == 1
    assert mod.main(["--verificar"], conectar=_conectar([], ("quizas",))) == 1


def test_el_runbook_y_el_contexto_dicen_el_orden_de_activacion():
    runbook = (RAIZ / "docs" / "runbooks" / "anti-fabricacion-activacion.md").read_text(encoding="utf-8")
    assert "anti_fabricacion.modo" in runbook and "sombra" in runbook
    assert "antes de reiniciar" in runbook.lower() and "Fernando" in runbook and "U5" in runbook
    assert "sembrar_modo.py" in runbook
    contexto = (RAIZ / "CONTEXT.md").read_text(encoding="utf-8")
    assert "anti-fabricacion-activacion.md" in contexto
