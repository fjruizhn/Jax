"""Partes puras de scripts/perfil_prevuelo.py (Task 15b, fix round 1): la
guardia de base, el filtro de sólo lectura, argumentos, percentiles y la
agrupación de CPU. Sin base: el script importa jacobs sólo al correr.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path

import pytest

_RUTA = Path(__file__).resolve().parents[1] / "scripts" / "perfil_prevuelo.py"
_spec = importlib.util.spec_from_file_location("perfil_prevuelo_bajo_test", _RUTA)
perfil = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(perfil)


def test_importar_el_script_no_importa_jacobs():
    import ast
    arbol = ast.parse(_RUTA.read_text(encoding="utf-8"))
    de_modulo = [n for n in arbol.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    nombres = {a.name for n in de_modulo for a in n.names} | {getattr(n, "module", None) for n in de_modulo}
    assert not any(str(x).startswith(("jacobs", "motor_registry", "aiomysql")) for x in nombres)


@pytest.mark.parametrize("nombre", ["jax_memory", "jax_memory_test2", "otra_test", "", None])
def test_guardia_rechaza_toda_base_que_no_sea_jax_memory_test(nombre, capsys):
    with pytest.raises(SystemExit) as salida:
        perfil.verificar_base(nombre)
    assert salida.value.code == 2
    assert "jax_memory_test" in capsys.readouterr().err


def test_guardia_acepta_jax_memory_test():
    perfil.verificar_base("jax_memory_test")


def test_main_sale_2_antes_de_tocar_la_base(monkeypatch):
    monkeypatch.setenv("JAX_DB_NAME", "jax_memory")
    with pytest.raises(SystemExit) as salida:
        perfil.main(["cpu", "--pedidos", "1"])
    assert salida.value.code == 2


@pytest.mark.parametrize("sql,es", [
    ("SELECT 1", True), ("  select `key` FROM capability", True), ("SHOW GLOBAL STATUS", True),
    ("INSERT INTO x VALUES (1)", False), ("UPDATE x SET a=1", False), ("DELETE FROM x", False),
    ("", False), ("KILL 3", False),
])
def test_solo_lectura(sql, es):
    assert perfil.es_lectura(sql) is es


def test_percentil_rango_mas_cercano():
    xs = list(range(1, 101))  # 1..100
    assert perfil.percentil(xs, 0.5) == 50
    assert perfil.percentil(xs, 0.95) == 95
    assert perfil.percentil(xs, 1.0) == 100
    assert perfil.percentil([7.0], 0.95) == 7.0
    assert perfil.percentil([3, 1, 2], 0.5) == 2  # ordena


@pytest.mark.parametrize("p", [0, -0.1, 1.5])
def test_percentil_rechaza_p_fuera_de_rango(p):
    with pytest.raises(ValueError):
        perfil.percentil([1, 2], p)


def test_percentil_de_lista_vacia_falla():
    with pytest.raises(ValueError):
        perfil.percentil([], 0.5)


def test_argumentos_de_cada_modo():
    a = perfil.parsear_args(["trabajadores", "-c", "1,25,50", "--duracion", "5"])
    assert a.modo == "trabajadores" and a.concurrencias == [1, 25, 50] and a.duracion == 5.0
    b = perfil.parsear_args(["abierto", "--rps", "100,400"])
    assert b.modo == "abierto" and b.rps == [100, 400]
    c = perfil.parsear_args(["cpu"])
    assert c.modo == "cpu" and c.pedidos == 1000


@pytest.mark.parametrize("argv", [
    [], ["abierto"], ["trabajadores", "-c", "0"], ["trabajadores", "-c", "a,b"],
    ["abierto", "--rps", "10", "--duracion", "-1"], ["cpu", "--pedidos", "0"],
])
def test_argumentos_invalidos_salen_2(argv):
    with pytest.raises(SystemExit) as salida:
        perfil.parsear_args(argv)
    assert salida.value.code == 2


def test_categorias_de_cpu():
    catalogo = os.path.join("x", "las_manos", "motor_registry", "catalog.py")
    filas = [
        (catalogo, "from_db", 0.1, 4.0),
        (catalogo, "_leer", 0.5, 3.5),  # ya dentro de from_db: no se suma dos veces
        ("~", "<method 'validate_python' of 'pydantic_core._pydantic_core.SchemaValidator' objects>", 1.0, 1.0),
        (os.path.join("usr", "lib", "python3", "json", "decoder.py"), "raw_decode", 0.5, 0.5),
        ("~", "<built-in method _json.scanstring>", 0.5, 0.5),
        ("otro.py", "f", 3.0, 3.0),
    ]
    assert perfil.categorias_de_cpu(filas, 10.0) == {"catalogo_motores": 0.4, "pydantic": 0.1, "json": 0.1}


def test_categorias_de_cpu_rechaza_total_no_positivo():
    with pytest.raises(ValueError):
        perfil.categorias_de_cpu([], 0.0)


def test_resumen_calcula_rps_cpu_y_fases():
    latencias = [{"total": 0.001 * k, "acquire": 0.0001} for k in range(1, 21)]
    r = perfil.resumen(latencias, segundos=2.0, cpu=0.04)
    assert r["pedidos"] == 20 and r["rps"] == 10.0 and r["cpu_ms_por_pedido"] == pytest.approx(2.0)
    assert r["total"]["p50"] == pytest.approx(10.0) and r["total"]["p95"] == pytest.approx(19.0)
    assert r["execute"] == {"p50": 0.0, "p95": 0.0}  # fase ausente cuenta como 0
