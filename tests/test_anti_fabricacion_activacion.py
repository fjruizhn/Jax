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
    """Cursor falso: `usuarios` = {texto: user_id}, `config` = valor actual o None; registra todo."""
    def __init__(self, conn):
        self.c = conn
        self.rowcount = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def execute(self, sql, args=None):
        self.c.log.append((sql, args))
        self._sql = sql
        if sql.startswith("INSERT IGNORE INTO axioma_config "):
            self.rowcount = 0 if self.c.config is not None else 1
            if self.c.config is None:
                self.c.config = args[1]
        elif sql.startswith("UPDATE axioma_config "):
            self.c.config = args[0]
            self.rowcount = 1

    async def fetchone(self):
        if "FROM jax_users" in self._sql:
            return (self.c.usuario,) if self.c.usuario is not None else None
        return None if self.c.config is None else (self.c.config,)

    async def fetchall(self):
        return [(self.c.usuario,)] if self.c.usuario is not None else []


class _Conn:
    def __init__(self, log, config, usuario):
        self.log, self.config, self.usuario = log, config, usuario
        self.commits = 0

    def cursor(self):
        return _Cur(self)

    def close(self):
        pass

    async def commit(self):
        self.commits += 1


def _conectar(log, fila=None, usuario=7):
    conn = _Conn(log, None if not fila else fila[0], usuario)

    async def conectar():
        return conn
    conectar.conn = conn
    return conectar


def _auditorias(log):
    return [a for q, a in log if q.startswith("INSERT INTO axioma_config_audit")]


def test_por_defecto_solo_simula_e_imprime_el_sql(mod, capsys):
    log = []
    assert mod.main([], conectar=_conectar(log)) == 0
    salida = capsys.readouterr().out
    assert "INSERT IGNORE INTO axioma_config" in salida and "anti_fabricacion.modo" in salida and "sombra" in salida
    assert log == [], "sin --aplicar no se toca la base"


def test_aplicar_exige_actor(mod):
    log = []
    assert mod.main(["--aplicar"], conectar=_conectar(log)) == 2
    assert log == []


def test_aplicar_inserta_sombra_sin_pisar_y_deja_su_fila_de_auditoria(mod):
    log = []
    c = _conectar(log)
    assert mod.main(["--aplicar", "--actor", "fernando"], conectar=c) == 0
    sql, args = next((q, a) for q, a in log if q.startswith("INSERT IGNORE INTO axioma_config "))
    assert args == ("anti_fabricacion.modo", "sombra")
    (aud,) = _auditorias(log)
    # (ts, actor_user_id, config_key, valor_anterior, valor_nuevo, origen, ip) -- convención de axioma_config_audit
    assert aud[1] == 7 and aud[2] == "anti_fabricacion.modo" and aud[3] is None and aud[4] == "sombra" and aud[5] == "config"
    assert aud[0] is not None
    assert not any(q.startswith(("UPDATE", "REPLACE")) for q, _ in log)
    assert c.conn.commits == 1


def test_aplicar_sobre_una_clave_ya_sembrada_no_cambia_nada_ni_audita(mod):
    log = []
    assert mod.main(["--aplicar", "--actor", "fernando"], conectar=_conectar(log, ("duro",))) == 0
    assert _auditorias(log) == []


def test_cambiar_el_modo_exige_actor_y_motivo_y_nunca_es_por_defecto(mod):
    log = []
    assert mod.main(["--modo", "duro"], conectar=_conectar(log, ("sombra",))) == 2
    assert mod.main(["--modo", "duro", "--actor", "fernando"], conectar=_conectar(log, ("sombra",))) == 2
    assert mod.main(["--modo", "duro", "--motivo", "U5 cumplido"], conectar=_conectar(log, ("sombra",))) == 2
    assert log == []
    assert mod.main(["--modo", "otro", "--actor", "f", "--motivo", "x"], conectar=_conectar(log, ("sombra",))) == 2


def test_cambiar_a_duro_actualiza_y_audita_con_valor_anterior_y_motivo(mod, capsys):
    log = []
    c = _conectar(log, ("sombra",))
    assert mod.main(["--modo", "duro", "--actor", "fernando", "--motivo", "U5 cumplido: 30 salidas"], conectar=c) == 0
    assert c.conn.config == "duro"
    upd = [a for q, a in log if q.startswith("UPDATE axioma_config ")]
    assert upd == [("duro", "anti_fabricacion.modo")]
    (aud,) = _auditorias(log)
    assert aud[1] == 7 and aud[3] == "sombra" and aud[4] == "duro" and aud[5] == "config"
    assert "U5 cumplido: 30 salidas" in capsys.readouterr().out, "el motivo queda impreso (la tabla no tiene columna para él)"
    assert c.conn.commits == 1


def test_cambiar_al_mismo_valor_no_hace_nada(mod):
    log = []
    assert mod.main(["--modo", "sombra", "--actor", "f", "--motivo", "x"], conectar=_conectar(log, ("sombra",))) == 0
    assert _auditorias(log) == [] and not any(q.startswith("UPDATE") for q, _ in log)


def test_un_actor_que_no_existe_o_es_ambiguo_aborta_sin_escribir(mod):
    log = []
    assert mod.main(["--modo", "duro", "--actor", "nadie", "--motivo", "x"], conectar=_conectar(log, ("sombra",), usuario=None)) == 2
    assert not any(q.startswith(("UPDATE", "INSERT")) for q, _ in log)


def test_verificar_sale_0_si_la_clave_es_valida_y_1_si_falta_o_es_invalida(mod):
    assert mod.main(["--verificar"], conectar=_conectar([], ("sombra",))) == 0
    assert mod.main(["--verificar"], conectar=_conectar([], ("duro",))) == 0
    assert mod.main(["--verificar"], conectar=_conectar([], None)) == 1
    assert mod.main(["--verificar"], conectar=_conectar([], ("quizas",))) == 1


def test_el_runbook_y_el_contexto_dicen_el_orden_de_activacion():
    runbook = (RAIZ / "docs" / "runbooks" / "anti-fabricacion-activacion.md").read_text(encoding="utf-8")
    assert "anti_fabricacion.modo" in runbook and "sombra" in runbook
    assert "antes de reiniciar" in runbook.lower() and "Fernando" in runbook and "U5" in runbook
    assert "sembrar_modo.py" in runbook and "--modo duro --actor" in runbook and "UPDATE" not in runbook
    contexto = (RAIZ / "CONTEXT.md").read_text(encoding="utf-8")
    assert "anti-fabricacion-activacion.md" in contexto
