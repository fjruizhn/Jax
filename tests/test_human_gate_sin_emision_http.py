"""El human gate de LAS MANOS no se puede autoaprobar (2026-09-17).

Hasta este cambio `POST /human_gate/token` (sin autenticación) le entregaba un
token de aprobación a CUALQUIER proceso local: Hyde, cualquier proceso de
`fruiz`, o `axioma`. Y `/motor/dispatch` aceptaba como `human_gate_token`
cualquier string no vacío. El contrato ahora es el de los tokens de
sub-pipelines (jacobs/subpipelines.py, frente F):

- Emisión SOLO del lado del servidor, sin ruta HTTP: la hace
  `las_manos/emitir_token_gate.py`, que necesita las credenciales de la base
  (/etc/jax/.env). Un proceso sin esa credencial no puede emitir.
- En la base queda SOLO el sha256; un solo uso; consumo atómico con
  `UPDATE … WHERE usado_at IS NULL`.
- `/execute` y `/motor/dispatch` consumen contra la base: un token inventado
  no aprueba nada.

Tests puros (sin base): la parte contra MariaDB real está en
las_manos/_human_gate_io_test.py.

Corre con:
  PYTHONPATH=.:las_manos python -m pytest tests/test_human_gate_sin_emision_http.py -v

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import ast
import asyncio
import hashlib
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

RAIZ = Path(__file__).resolve().parents[1]
MODULOS_CON_RUTAS = [
    RAIZ / "las_manos" / "server.py",
    RAIZ / "las_manos" / "motor_registry" / "routes.py",
    RAIZ / "jacobs" / "routes.py",
]
_METODOS = {"get", "post", "put", "patch", "delete", "api_route", "websocket"}


def _arbol(ruta: Path) -> ast.Module:
    return ast.parse(ruta.read_text(encoding="utf-8"))


def _rutas(ruta: Path) -> list[tuple[str, ast.AST]]:
    """(path, función) de cada `@app.<método>(path)` / `@router.<método>(path)`."""
    halladas = []
    for nodo in ast.walk(_arbol(ruta)):
        if not isinstance(nodo, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for deco in nodo.decorator_list:
            if (isinstance(deco, ast.Call) and isinstance(deco.func, ast.Attribute)
                    and deco.func.attr in _METODOS and deco.args
                    and isinstance(deco.args[0], ast.Constant)):
                halladas.append((deco.args[0].value, nodo))
    return halladas


def _llamadas(nodo: ast.AST) -> set[str]:
    nombres = set()
    for n in ast.walk(nodo):
        if isinstance(n, ast.Call):
            f = n.func
            nombres.add(f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", ""))
    return nombres


def _funcion(ruta: Path, nombre: str) -> ast.AST:
    (hallada,) = [n for n in ast.walk(_arbol(ruta))
                  if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == nombre]
    return hallada


# ---------------------------------------------------------------------------
#  Sin ruta HTTP de emisión
# ---------------------------------------------------------------------------

def test_el_escaneo_de_rutas_ve_las_rutas_reales():
    """Control del escaneo: si dejara de ver decoradores, el test de abajo
    daría verde sin mirar nada."""
    paths = {p for m in MODULOS_CON_RUTAS for p, _ in _rutas(m)}
    assert {"/execute", "/health", "/dispatch", "/pipeline/{pipeline_id}/approve-step"} <= paths


def test_ninguna_ruta_de_las_manos_emite_tokens_de_aprobacion():
    """El hueco: `POST /human_gate/token` sin autenticación. Ninguna ruta puede
    tener `human_gate` o `token` en el path, ni llamar al emisor."""
    culpables = []
    for modulo in MODULOS_CON_RUTAS:
        for path, funcion in _rutas(modulo):
            if "human_gate" in path or "token" in path.lower():
                culpables.append(f"{modulo.name}: {path}")
            if {"emitir_token_gate", "issue"} & _llamadas(funcion):
                culpables.append(f"{modulo.name}: {path} llama al emisor")
    assert culpables == []


def test_server_no_emite_ni_guarda_tokens_en_memoria():
    arbol = _arbol(RAIZ / "las_manos" / "server.py")
    clases = {n.name for n in ast.walk(arbol) if isinstance(n, ast.ClassDef)}
    assert "HumanGate" not in clases
    assert "emitir_token_gate" not in _llamadas(arbol)


def test_ningun_modulo_de_servicio_llama_al_emisor():
    """El emisor sólo lo llama la herramienta de línea de comandos (y los tests)."""
    permitidos = {RAIZ / "las_manos" / "emitir_token_gate.py", RAIZ / "las_manos" / "human_gate.py"}
    llamadores = []
    for carpeta in (RAIZ / "las_manos", RAIZ / "jacobs"):
        for py in carpeta.rglob("*.py"):
            if ".venv" in py.parts or py.name.startswith("_") or py in permitidos:
                continue
            if "emitir_token_gate" in _llamadas(_arbol(py)):
                llamadores.append(str(py.relative_to(RAIZ)))
    assert llamadores == []


def test_execute_consume_el_token_contra_la_base():
    execute = _funcion(RAIZ / "las_manos" / "server.py", "execute")
    llamadas = _llamadas(execute)
    assert "consumir_token_gate" in llamadas
    assert "validate" not in llamadas


# ---------------------------------------------------------------------------
#  El contrato del token (store sustituido: la base real está en el test de IO)
# ---------------------------------------------------------------------------

@pytest.fixture
def store_falso(monkeypatch):
    """La base sustituida. raising=False: antes del arreglo estas funciones no
    existían y los tests de /motor/dispatch tienen que poder verse en ROJO por
    lo que hacen, no por un error del fixture."""
    from jacobs import store
    monkeypatch.setattr(store, "human_gate_token_emitir", AsyncMock(), raising=False)
    monkeypatch.setattr(store, "human_gate_token_consumir", AsyncMock(return_value=False),
                        raising=False)
    monkeypatch.setattr(store, "human_gate_token_diagnostico", AsyncMock(return_value=None),
                        raising=False)
    return store


@pytest.fixture
def gate(store_falso):
    import human_gate
    return human_gate, store_falso


def test_la_emision_guarda_solo_el_sha256(gate):
    human_gate, store = gate
    emitido = asyncio.run(human_gate.emitir_token_gate("fernando", ttl_segundos=300))
    (args,) = [c.args for c in store.human_gate_token_emitir.await_args_list]
    assert args[0] == hashlib.sha256(emitido.token.encode()).hexdigest()
    assert emitido.token not in [str(a) for a in args]
    assert len(emitido.token) >= 43  # 32 bytes en base64url


@pytest.mark.parametrize("token", [None, "", "   "])
def test_sin_token_rechaza_sin_tocar_la_base(gate, token):
    human_gate, store = gate
    r = asyncio.run(human_gate.consumir_token_gate(token, uso="execute:r1"))
    assert (r.aceptado, r.motivo) == (False, human_gate.Motivo.FALTA_TOKEN)
    store.human_gate_token_consumir.assert_not_awaited()


def test_un_token_inventado_no_aprueba(gate):
    human_gate, store = gate
    r = asyncio.run(human_gate.consumir_token_gate("inventado", uso="execute:r1"))
    assert (r.aceptado, r.motivo) == (False, human_gate.Motivo.TOKEN_DESCONOCIDO)
    (args,) = [c.args for c in store.human_gate_token_consumir.await_args_list]
    assert args[0] == hashlib.sha256(b"inventado").hexdigest()


@pytest.mark.parametrize("diagnostico, motivo", [
    ({"usado_at": 1.0, "vence_at": 9e18}, "TOKEN_USADO"),
    ({"usado_at": None, "vence_at": 1.0}, "TOKEN_VENCIDO"),
    ({"usado_at": None, "vence_at": 9e18}, "ESTADO_CAMBIO"),
])
def test_motivo_del_rechazo(gate, diagnostico, motivo):
    human_gate, store = gate
    store.human_gate_token_diagnostico.return_value = diagnostico
    r = asyncio.run(human_gate.consumir_token_gate("t", uso="execute:r1"))
    assert (r.aceptado, r.motivo) == (False, human_gate.Motivo[motivo])


def test_consumo_aceptado(gate):
    human_gate, store = gate
    store.human_gate_token_consumir.return_value = True
    r = asyncio.run(human_gate.consumir_token_gate("t", uso="execute:r1"))
    assert (r.aceptado, r.motivo) == (True, None)
    store.human_gate_token_diagnostico.assert_not_awaited()


@pytest.mark.parametrize("valor", [0, 9, 3601, "300", None, True])
def test_ttl_fuera_de_rango_o_mal_tipado_falla_cerrado(valor):
    import human_gate
    with pytest.raises(ValueError):
        human_gate.ttl_segundos({"token_ttl_seconds": valor})


def test_ttl_valido():
    import human_gate
    assert human_gate.ttl_segundos({"token_ttl_seconds": 300}) == 300


# ---------------------------------------------------------------------------
#  /motor/dispatch: un string cualquiera ya no satisface el gate
# ---------------------------------------------------------------------------

_CAP_CON_GATE = {
    "allowed_motors": ["kimi"], "allowed_callers": ["hyde"], "risk_level": "high",
    "sandbox_only": True, "requires_human_gate": True, "max_execution_minutes": 15,
    "max_recursion_depth": 0, "output_schema": "",
}
_CFG = {
    "motors": {"kimi": {
        "enabled": True, "provider": "kimi", "transport": "http_openai_compat",
        "provider_id": "moonshot", "api_key_env": "KIMI_API_KEY",
        "api_url": "https://api.example/v1/chat/completions", "model": "kimi-k3",
        "max_context_tokens": 256000, "sandbox_only": True, "default_timeout_seconds": 600,
        "supports_reasoning": True, "reasoning_default_visibility": "audit_only",
        "max_tokens": 8000,
    }},
    "capabilities": {"code_swarm": _CAP_CON_GATE},
}


@pytest.fixture
def motor(tmp_path, monkeypatch):
    from motor_registry import routes
    from motor_registry.catalog import MotorCatalog
    from motor_registry.job_store import JobStore
    from motor_registry.policy import MotorPolicy

    catalogo = MotorCatalog(_CFG)
    lanzado = AsyncMock()
    monkeypatch.setattr(routes, "_STORE", JobStore(str(tmp_path / "jobs.jsonl")))
    monkeypatch.setattr(routes, "_CATALOG", catalogo)
    monkeypatch.setattr(routes, "_POLICY", MotorPolicy(catalogo))
    monkeypatch.setattr(routes, "_ensure_catalog_fresh", AsyncMock())
    monkeypatch.setattr(routes.motor_worker, "run", lanzado)
    monkeypatch.setenv("JAX_KILL_SWITCH_PATH", str(tmp_path / "PAUSE"))
    return routes, lanzado


def _pedido(token):
    from motor_registry.models import MotorDispatchRequest
    return MotorDispatchRequest(caller="hyde", capability="code_swarm", motor="kimi",
                                prompt="p", human_gate_token=token, timeout_seconds=60)


def test_motor_dispatch_con_token_inventado_se_rechaza(motor, store_falso):
    routes, lanzado = motor
    from fastapi import HTTPException
    with pytest.raises(HTTPException, match="GOVERNED_EXECUTION_REQUIRED") as exc:
        asyncio.run(routes.dispatch(_pedido("cualquier-cosa")))
    assert exc.value.status_code == 410
    lanzado.assert_not_called()


def test_motor_dispatch_con_token_emitido_pasa_y_lo_consume(motor, store_falso):
    routes, _ = motor
    store = store_falso
    store.human_gate_token_consumir.return_value = True
    from fastapi import HTTPException
    with pytest.raises(HTTPException, match="GOVERNED_EXECUTION_REQUIRED"):
        asyncio.run(routes.dispatch(_pedido("emitido")))
    store.human_gate_token_consumir.assert_not_awaited()


def test_motor_dispatch_rechazado_por_politica_no_quema_el_token(motor, store_falso):
    routes, _ = motor
    store = store_falso
    from motor_registry.models import MotorDispatchRequest
    pedido = MotorDispatchRequest(caller="ada", capability="code_swarm", motor="kimi",
                                  prompt="p", human_gate_token="emitido", timeout_seconds=60)
    from fastapi import HTTPException
    with pytest.raises(HTTPException, match="GOVERNED_EXECUTION_REQUIRED"):
        asyncio.run(routes.dispatch(pedido))
    store.human_gate_token_consumir.assert_not_awaited()


def test_governed_dispatch_without_authoritative_store_rejects(motor):
    routes, launched = motor
    from motor_registry.models import GovernedDispatchRequest
    from fastapi import HTTPException
    routes.configure_governed_execution_store(None)
    with pytest.raises(HTTPException, match="store no inicializado") as exc:
        asyncio.run(routes.governed_dispatch(GovernedDispatchRequest(execution_id="x")))
    assert exc.value.status_code == 503
    launched.assert_not_called()


def test_governed_dispatch_missing_binding_rejects_before_worker(motor, monkeypatch):
    routes, launched = motor
    from motor_registry.models import GovernedDispatchRequest
    from fastapi import HTTPException
    class Store:
        def load_execution(self, _):
            from types import SimpleNamespace
            return SimpleNamespace(execution_id="x", authorization_id="a", decision_id="wrong",
                execution_authorization_hash="h", execution_request_hash="r")
        def load_authorization(self, _):
            from types import SimpleNamespace
            return SimpleNamespace(decision_id="right", execution_authorization_hash="h")
    routes.configure_governed_execution_store(Store())
    with pytest.raises(HTTPException, match="GOVERNED_DISPATCH_REJECTED"):
        asyncio.run(routes.governed_dispatch(GovernedDispatchRequest(execution_id="x")))
    launched.assert_not_called()


def test_governed_dispatch_claims_before_mocked_worker(motor, monkeypatch):
    routes, launched = motor
    from types import SimpleNamespace
    from motor_registry.models import GovernedDispatchRequest
    calls = []
    request = SimpleNamespace(capability="code_swarm", motor="kimi", authenticated_caller_id="hyde",
        execution_request_hash="r", prompt="p", user_id=None, tenant_id=None, timeout_seconds=60,
        projection=lambda: {"context": {}})
    auth = SimpleNamespace(decision_id="d", execution_authorization_hash="h", execution_request=request)
    record = SimpleNamespace(execution_id="x", authorization_id="a", decision_id="d",
        execution_authorization_hash="h", execution_request_hash="r")
    class Store:
        def load_execution(self, _): return record
        def load_authorization(self, _): return auth
    routes.configure_governed_execution_store(Store(), lambda *args, **kwargs: calls.append(args))
    result = asyncio.run(routes.governed_dispatch(GovernedDispatchRequest(execution_id="x")))
    assert calls and result.status.value == "pending"
    launched.assert_called_once()
