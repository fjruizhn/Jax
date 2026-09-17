"""Aritmética de la medición de la semilla de capability.min_output_tokens
(spec 2026-09-17 §4.4). La lectura de producción no se testea acá: la hace el
script con GO, en solo lectura. La medición YA SE HIZO UNA VEZ (plan P,
jax-platform, Task 2, commit d79b0f9): este archivo no la repite (Ruling R2
del ledger de este plan).

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from medir_min_output_tokens import combinar, maximos_de_jobs, redondear  # noqa: E402

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "medir_min_output_tokens.py"


def test_redondea_hacia_arriba_al_multiplo_de_1024():
    assert [redondear(n) for n in (1, 1024, 1025, 7997)] == [1024, 1024, 2048, 8192]


def test_cero_o_negativo_queda_en_cero():
    assert redondear(0) == 0 and redondear(-5) == 0


def test_maximos_de_jobs_solo_completed_con_completion_tokens():
    lineas = [
        json.dumps({"status": "completed", "capability": "generate", "_usage": {"completion_tokens": 3000}}),
        json.dumps({"status": "completed", "capability": "generate", "_usage": {"completion_tokens": 5000}}),
        json.dumps({"status": "failed", "capability": "generate", "_usage": {"completion_tokens": 8000}}),
        json.dumps({"status": "completed", "capability": "design", "_usage": {}}),
        "",
    ]
    assert maximos_de_jobs(lineas) == ({"generate": 5000}, 0)


def test_una_linea_rota_se_cuenta_no_se_esconde():
    lineas = ["{roto", json.dumps({"status": "completed", "capability": "reason", "_usage": {"completion_tokens": 10}})]
    assert maximos_de_jobs(lineas) == ({"reason": 10}, 1)


def test_combinar_toma_el_maximo_por_capability():
    assert combinar({"a": 10, "b": 5}, {"a": 7, "c": 1}) == {"a": 10, "b": 5, "c": 1}


def test_el_join_http_directo_lleva_la_collate_del_esquema_real():
    """Plan P (jax-platform Task 2, commit d79b0f9) midió contra producción y
    tuvo que agregar `COLLATE utf8mb4_uca1400_ai_ci` al JOIN por facet: error
    1267 "Illegal mix of collations" entre axioma_usage.facet
    (utf8mb4_uca1400_ai_ci) y jacobs_steps.facet (utf8mb4_unicode_ci). Sin
    esto, este script fallaría contra el esquema real la primera vez que se
    corriera con GO."""
    from medir_min_output_tokens import _SQL_HTTP_DIRECTO

    assert "COLLATE utf8mb4_uca1400_ai_ci" in _SQL_HTTP_DIRECTO
    assert "u.facet = s.facet COLLATE utf8mb4_uca1400_ai_ci" in _SQL_HTTP_DIRECTO.replace("\n", " ")


def _sql_ejecutados_por_el_script() -> list[str]:
    """SQL que el script EJECUTA vía `cur.execute(...)`, no el que solo
    imprime. Resuelve constantes de módulo (p.ej. `_SQL_HTTP_DIRECTO`) para
    no depender de que el literal esté inline en la llamada."""
    arbol = ast.parse(_SCRIPT.read_text(encoding="utf-8"))
    constantes = {
        nodo.targets[0].id: nodo.value.value
        for nodo in ast.walk(arbol)
        if isinstance(nodo, ast.Assign)
        and len(nodo.targets) == 1
        and isinstance(nodo.targets[0], ast.Name)
        and isinstance(nodo.value, ast.Constant)
        and isinstance(nodo.value.value, str)
    }
    sentencias = []
    for nodo in ast.walk(arbol):
        if (
            isinstance(nodo, ast.Call)
            and isinstance(nodo.func, ast.Attribute)
            and nodo.func.attr == "execute"
            and nodo.args
        ):
            arg = nodo.args[0]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                sentencias.append(arg.value)
            elif isinstance(arg, ast.Name) and arg.id in constantes:
                sentencias.append(constantes[arg.id])
    return sentencias


def test_el_script_es_de_solo_lectura_nada_de_insert_update_delete_ejecutado():
    """El script es de solo lectura POR CONSTRUCCIÓN: cada `cur.execute(...)`
    real lleva una sentencia SELECT. Distinto de los `print(f"UPDATE
    capability ...")` que arma para que la migración del plan P los consuma
    (ver el siguiente test) -- esos nunca pasan por `cur.execute`."""
    sentencias = _sql_ejecutados_por_el_script()
    assert sentencias, "no se encontró ningún cur.execute(...) -- el detector no vigila nada"
    prohibidas = ("INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE", "REPLACE")
    ofensoras = [s for s in sentencias if any(p in s.upper() for p in prohibidas)]
    assert ofensoras == []
    assert all(s.strip().upper().startswith("SELECT") for s in sentencias)


def test_los_update_que_imprime_para_la_migracion_no_se_ejecutan():
    fuente = _SCRIPT.read_text(encoding="utf-8")
    assert "UPDATE capability SET min_output_tokens" in fuente, (
        "el script tiene que imprimir los UPDATE que consume la migración del plan P"
    )
    sentencias = _sql_ejecutados_por_el_script()
    assert not any("UPDATE capability SET min_output_tokens" in s for s in sentencias)
