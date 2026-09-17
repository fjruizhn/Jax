"""E-13 (2026-09-16): el tope de steps por pipeline es UNA constante.

Estaba repetido como literal 20 en jacobs/models.py (request y validador),
jacobs/routes.py (plan_only) y jacobs/policy.py. Cambiar la constante no movía
los otros dos. Vive en models.py porque policy.py ya importa models: al revés
sería un import circular (verificador, 2026-09-16).
"""
from __future__ import annotations

import ast
import asyncio
import os
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

os.environ["JAX_DB_NAME"] = "jax_memory_test"

RAIZ = Path(__file__).resolve().parents[1]
_ARCHIVOS = ("jacobs/models.py", "jacobs/routes.py", "jacobs/policy.py", "jacobs/plan.py")


def _nombre(nodo: ast.AST) -> str | None:
    if isinstance(nodo, ast.Name):
        return nodo.id
    if isinstance(nodo, ast.Attribute):
        return nodo.attr
    return None


def _literales_de_max_steps(fuente: str) -> list[int]:
    lineas = []
    for nodo in ast.walk(ast.parse(fuente)):
        if isinstance(nodo, ast.Compare):
            lados = [nodo.left, *nodo.comparators]
            if any(_nombre(l) == "max_steps" for l in lados) and any(
                    isinstance(l, ast.Constant) and isinstance(l.value, int) and l.value > 1 for l in lados):
                lineas.append(nodo.lineno)
        elif isinstance(nodo, ast.AnnAssign) and _nombre(nodo.target) == "max_steps" \
                and isinstance(nodo.value, ast.Constant):
            lineas.append(nodo.lineno)
        elif isinstance(nodo, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = nodo.args.args[len(nodo.args.args) - len(nodo.args.defaults):]
            for arg, default in zip(args, nodo.args.defaults):
                if arg.arg == "max_steps" and isinstance(default, ast.Constant):
                    lineas.append(default.lineno)
    return lineas


def test_la_constante_vive_en_models_y_policy_la_importa():
    from jacobs import models, policy
    assert models.MAX_STEPS_PER_PIPELINE == 20
    arbol = ast.parse((RAIZ / "jacobs" / "policy.py").read_text(encoding="utf-8"))
    definiciones = [n for n in arbol.body if isinstance(n, ast.Assign)
                    and any(_nombre(t) == "MAX_STEPS_PER_PIPELINE" for t in n.targets)]
    assert definiciones == [], "policy.py vuelve a definir su propio tope"
    assert policy.MAX_STEPS_PER_PIPELINE is models.MAX_STEPS_PER_PIPELINE


def test_ningun_literal_de_tope_para_max_steps():
    hallazgos = {rel: _literales_de_max_steps((RAIZ / rel).read_text(encoding="utf-8")) for rel in _ARCHIVOS}
    assert {rel: l for rel, l in hallazgos.items() if l} == {}


def test_el_request_lee_la_constante(monkeypatch):
    from jacobs import models
    monkeypatch.setattr(models, "MAX_STEPS_PER_PIPELINE", 5, raising=False)
    with pytest.raises(ValueError) as e:
        models.PipelineCreateRequest(name="n", objective="o", invoked_by="plataforma", mode="dry_run", max_steps=6)
    assert "entre 1 y 5" in str(e.value)


def test_plan_only_lee_la_constante(monkeypatch):
    from fastapi import HTTPException
    from jacobs import routes
    monkeypatch.setattr(routes, "MAX_STEPS_PER_PIPELINE", 5, raising=False)
    monkeypatch.setattr(routes, "check_kill_switch", lambda: False)
    monkeypatch.setattr(routes, "_build_plan_or_reject", AsyncMock(side_effect=AssertionError("no debía planificar")))
    req = routes.PlanRequest(name="n", objective="o", invoked_by="plataforma", mode="dry_run", max_steps=6)
    with pytest.raises(HTTPException) as e:
        asyncio.run(routes.plan_only(req))
    assert e.value.status_code == 422
    assert "(5)" in e.value.detail
