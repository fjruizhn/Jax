"""Salud de nivel proveedor y motor resuelto, sin DB (spec 2026-09-17 §4.5,
desvío 1 del plan).

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

os.environ["JAX_DB_NAME"] = "jax_memory_test"

from jacobs import facet_health as fh  # noqa: E402
from motor_registry.policy import MotorPolicy  # noqa: E402


def test_ok_reciente_es_sana():
    assert fh.salud_de_proveedor((1000.0, "ok"), 1060.0) == "sana"


def test_sin_evento_o_fuera_de_ventana_se_sondea():
    assert fh.salud_de_proveedor(None, 1060.0) == "sondear"
    assert fh.salud_de_proveedor((0.0, "ok"), fh.HEALTH_WINDOW_SECONDS + 1.0) == "sondear"


def test_provider_error_reciente_se_vuelve_a_medir():
    assert fh.salud_de_proveedor((1000.0, "provider_error"), 1060.0) == "sondear"


def test_la_consulta_solo_cuenta_eventos_de_proveedor():
    sql = fh.sql_ultimo_evento_de_proveedor(2)
    assert "outcome IN ('ok','provider_error')" in sql
    assert sql.count("%s") == 3  # dos claves + el inicio de la ventana


def test_motor_que_despacharia_respeta_el_pedido_y_la_prioridad():
    class _Catalogo:
        def get_capability(self, nombre):
            return SimpleNamespace(allowed_motors=["kimi", "ada"]) if nombre == "implementation" else None

        def get_motor(self, nombre):
            return {"kimi": SimpleNamespace(enabled=False), "ada": SimpleNamespace(enabled=True)}.get(nombre)

    politica = MotorPolicy(_Catalogo())
    assert politica.motor_que_despacharia(None, "implementation") == "ada"
    assert politica.motor_que_despacharia("kimi", "implementation") is None
    assert politica.motor_que_despacharia("ada", "implementation") == "ada"
    assert politica.motor_que_despacharia(None, "no_existe") is None
