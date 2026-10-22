"""LAS MANOS autentica a sus llamadores (2026-09-17).

Hasta este cambio LAS MANOS no autenticaba nada y la identidad salía del
cuerpo: un proceso cualquiera (Hyde desde su jaula con red local, cualquier
proceso de `fruiz`) mandaba `{"invoked_by": "plataforma"}` a
`POST /jacobs/pipeline/{id}/approve-step` y aprobaba los pasos de Hyde
bloqueados en el gate.

Lo que se prueba acá, contra los routers reales con la base sustituida:

- sin credencial -> 401 y el paso NO se aprueba (visto en rojo contra c7d59cc:
  200 y el paso pasaba a pending);
- con la credencial `jacobs` (la del propio proceso de LAS MANOS) tampoco se
  aprueba: 403 aunque declare `invoked_by=plataforma`;
- la credencial `plataforma` sigue aprobando (el flujo real no se rompe);
- deny by default: una ruta nueva sin proteger explícitamente queda cubierta;
- `/health` sigue público;
- la identidad declarada en el cuerpo tiene que ser la de la credencial;
- fail-closed al cargar las credenciales;
- server.py instala la protección y Jacobs presenta su credencial;
- la jaula de Hyde no recibe la credencial.

Corre con:
  PYTHONPATH=.:las_manos python -m pytest tests/test_las_manos_auth_servicio.py -v

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import ast
import secrets
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import auth_servicio
from processing_ownership import ProcessingOwnershipError, processing_ownership_from_headers
from auth_servicio import (
    ENCABEZADO, IDENTIDAD_JACOBS, IDENTIDAD_PLATAFORMA, VARIABLES,
    cargar_credenciales, proteger,
)
from config_entorno import EntornoInvalido
from jacobs import routes as jacobs_routes
from jacobs.models import Pipeline, PipelineStatus, Step, StepStatus

RAIZ = Path(__file__).resolve().parents[1]

CRED = {
    IDENTIDAD_PLATAFORMA: secrets.token_urlsafe(32),
    IDENTIDAD_JACOBS: secrets.token_urlsafe(32),
}


def _credenciales() -> dict[str, bytes]:
    return {k: v.encode("ascii") for k, v in CRED.items()}


def test_processing_owner_headers_are_closed_and_canonical():
    headers = [(b"x-jax-processing-owner-version", b"processing-owner.1"),
               (b"x-jax-processing-tenant-id", b"1"), (b"x-jax-processing-user-id", b"2"),
               (b"x-jax-processing-project-id", b"3")]
    assert processing_ownership_from_headers(headers).project_id == "3"
    with pytest.raises(ProcessingOwnershipError):
        processing_ownership_from_headers(headers + [(b"x-jax-processing-tenant-id", b"1")])
    with pytest.raises(ProcessingOwnershipError):
        processing_ownership_from_headers(headers[:-1])
    with pytest.raises(ProcessingOwnershipError):
        processing_ownership_from_headers(headers[:-1] + [(b"x-jax-processing-project-id", b"03")])


def _app() -> FastAPI:
    """Mismo armado que server.py: routers reales + proteger()."""
    from motor_registry.routes import router as motor_router

    app = FastAPI()

    @app.get("/health")
    async def _health() -> dict:
        return {"status": "alive"}

    @app.get("/audit/tail")
    async def _tail() -> dict:
        return {"events": ["forense"]}

    @app.post("/execute")
    async def _execute() -> dict:
        return {"ejecutado": True}

    # Una ruta que alguien agrega mañana sin acordarse de protegerla.
    @app.post("/jacobs/ruta-nueva-de-manana")
    async def _nueva() -> dict:
        return {"hecho": True}

    app.include_router(motor_router)
    app.include_router(jacobs_routes.router)
    proteger(app, _credenciales())
    return app


def _h(identidad: str | None) -> dict[str, str]:
    return {ENCABEZADO: CRED[identidad]} if identidad else {}


@pytest.fixture
def paso_de_hyde_en_gate():
    """Pipeline interrumpido con un paso de Hyde en blocked_human_gate."""
    paso = Step(pipeline_id="p1", step_index=0, facet="hyde", capability="code",
                status=StepStatus.blocked_human_gate)
    pipeline = Pipeline(pipeline_id="p1", name="n", invoked_by="plataforma",
                        mode="autonomous", status=PipelineStatus.interrupted)
    step_upsert = AsyncMock()
    with patch.object(jacobs_routes.cupo, "activos", AsyncMock(return_value=0)), \
         patch.object(jacobs_routes.store, "pipeline_get", AsyncMock(return_value=pipeline)), \
         patch.object(jacobs_routes.store, "steps_by_pipeline", AsyncMock(return_value=[paso])), \
         patch.object(jacobs_routes.store, "event_append", AsyncMock()), \
         patch.object(jacobs_routes.store, "step_upsert", step_upsert), \
         patch.object(jacobs_routes.store, "pipeline_update_status", AsyncMock()), \
         patch.object(jacobs_routes.store, "pipeline_tomar_epoca", AsyncMock(return_value=1)), \
         patch.object(jacobs_routes, "_prevuelo_de_reanudacion",
                      AsyncMock(return_value=({}, {}, []))), \
         patch.object(jacobs_routes, "check_kill_switch", return_value=False), \
         patch.object(jacobs_routes, "run_pipeline", AsyncMock()):
        yield step_upsert


APROBAR = "/jacobs/pipeline/p1/approve-step"


# ---------------------------------------------------------------------------
#  El hueco: aprobar un paso de Hyde
# ---------------------------------------------------------------------------

def test_sin_credencial_no_se_aprueba_un_paso_de_hyde(paso_de_hyde_en_gate):
    with TestClient(_app()) as c:
        r = c.post(APROBAR, json={"invoked_by": "plataforma"})
    assert r.status_code == 401, r.text
    paso_de_hyde_en_gate.assert_not_called()


def test_credencial_inventada_no_aprueba(paso_de_hyde_en_gate):
    with TestClient(_app()) as c:
        r = c.post(APROBAR, json={"invoked_by": "plataforma"},
                   headers={ENCABEZADO: secrets.token_urlsafe(32)})
    assert r.status_code == 401, r.text
    paso_de_hyde_en_gate.assert_not_called()


def test_hyde_dentro_del_proceso_legitimo_no_se_autoaprueba(paso_de_hyde_en_gate):
    """La credencial `jacobs` es la del proceso que corre los pasos de Hyde:
    con ella no se aprueba ni se reanuda, declare lo que declare."""
    with TestClient(_app()) as c:
        for cuerpo in ({"invoked_by": "plataforma"}, {"invoked_by": "ada"}, {}):
            r = c.post(APROBAR, json=cuerpo, headers=_h(IDENTIDAD_JACOBS))
            assert r.status_code == 403, (cuerpo, r.text)
        r = c.post("/jacobs/pipeline/p1/resume", json={"invoked_by": "plataforma"},
                   headers=_h(IDENTIDAD_JACOBS))
        assert r.status_code == 403, r.text
    paso_de_hyde_en_gate.assert_not_called()


def test_la_plataforma_sigue_aprobando(paso_de_hyde_en_gate):
    with TestClient(_app()) as c:
        r = c.post(APROBAR, json={"invoked_by": "plataforma"}, headers=_h(IDENTIDAD_PLATAFORMA))
    assert r.status_code == 200, r.text
    assert r.json()["approved_steps"] == [0]
    paso_de_hyde_en_gate.assert_called_once()


def test_claves_duplicadas_no_esquivan_el_chequeo(paso_de_hyde_en_gate):
    cuerpo = b'{"invoked_by": "ada", "invoked_by": "plataforma"}'
    with TestClient(_app()) as c:
        r = c.post(APROBAR, content=cuerpo, headers={**_h(IDENTIDAD_JACOBS),
                                                     "content-type": "application/json"})
        assert r.status_code == 403, r.text
        r = c.post(APROBAR, content=b'{"invoked_by": "plataforma", "invoked_by": "plataforma"}',
                   headers={**_h(IDENTIDAD_PLATAFORMA), "content-type": "application/json"})
        assert r.status_code == 400, r.text
    paso_de_hyde_en_gate.assert_not_called()


# ---------------------------------------------------------------------------
#  Deny by default y rutas públicas
# ---------------------------------------------------------------------------

def test_health_sigue_publico():
    with TestClient(_app()) as c:
        assert c.get("/health").status_code == 200


@pytest.mark.parametrize("metodo,ruta", [
    ("post", "/execute"), ("get", "/audit/tail"), ("post", "/jacobs/ruta-nueva-de-manana"),
    ("post", "/motor/dispatch"), ("get", "/motor/job/x"), ("post", "/motor/authorize-facet"),
    ("get", "/jacobs/pipeline/p1"), ("post", "/jacobs/pipeline"), ("post", "/jacobs/plan"),
    ("get", "/ruta-que-no-existe"),
])
def test_toda_ruta_no_publica_exige_credencial(metodo, ruta):
    with TestClient(_app()) as c:
        r = getattr(c, metodo)(ruta)
    assert r.status_code == 401, (ruta, r.text)
    assert r.json() == {"detail": {"code": auth_servicio.CODIGO_SIN_CREDENCIAL}}


@pytest.mark.parametrize("identidad", [IDENTIDAD_PLATAFORMA, IDENTIDAD_JACOBS])
def test_execute_y_audit_no_los_alcanza_ninguna_identidad(identidad):
    """Sin llamador real (journal de 7 días + grep en jax y jax-platform): ni
    `/execute` (staging sin gate para hyde) ni el log forense."""
    with TestClient(_app()) as c:
        assert c.post("/execute", headers=_h(identidad)).status_code == 403
        assert c.get("/audit/tail", headers=_h(identidad)).status_code == 403


def test_la_ruta_nueva_bajo_jacobs_es_solo_de_la_plataforma():
    with TestClient(_app()) as c:
        assert c.post("/jacobs/ruta-nueva-de-manana", headers=_h(IDENTIDAD_JACOBS)).status_code == 403
        assert c.post("/jacobs/ruta-nueva-de-manana", headers=_h(IDENTIDAD_PLATAFORMA)).status_code == 200


# ---------------------------------------------------------------------------
#  La identidad del cuerpo es la de la credencial
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("identidad,ruta,cuerpo", [
    (IDENTIDAD_JACOBS, "/jacobs/pipeline", {"invoked_by": "plataforma"}),
    (IDENTIDAD_JACOBS, "/jacobs/pipeline", {"invoked_by": "jax_local"}),
    (IDENTIDAD_PLATAFORMA, "/jacobs/pipeline", {"invoked_by": "ada"}),
    (IDENTIDAD_PLATAFORMA, "/motor/authorize-facet", {"caller": "jacobs", "facet": "kimi"}),
    (IDENTIDAD_PLATAFORMA, "/jacobs/pipeline", {"invoked_by": ["plataforma"]}),
])
def test_declarar_otra_identidad_se_rechaza(identidad, ruta, cuerpo):
    with TestClient(_app()) as c:
        r = c.post(ruta, json=cuerpo, headers=_h(identidad))
    assert r.status_code == 403, r.text
    assert r.json() == {"detail": {"code": auth_servicio.CODIGO_IDENTIDAD_DECLARADA}}


@pytest.mark.parametrize("metodo,ruta", [
    ("post", "/motor/dispatch"),
    ("get", "/motor/job/j1"),
    ("post", "/motor/job/j1/cancel"),
])
def test_jacobs_ya_no_tiene_permiso_sobre_los_motores(metodo, ruta):
    """Jacobs no despacha motores (despacho legacy cerrado, 410) ni consulta o
    cancela motor jobs: nada del arbol usa esas rutas con la credencial
    `jacobs`. Un permiso que nadie usa es superficie sin dueno: 403 de ruta."""
    with TestClient(_app()) as c:
        r = getattr(c, metodo)(ruta, headers=_h(IDENTIDAD_JACOBS))
    assert r.status_code == 403, r.text
    assert r.json() == {"detail": {"code": auth_servicio.CODIGO_RUTA_NO_PERMITIDA}}


def test_jacobs_conserva_solo_el_pipeline_de_sub_pipelines():
    permiso = auth_servicio.PERMISOS[IDENTIDAD_JACOBS]
    assert [(m, p.pattern) for m, p in permiso.rutas] == [("POST", "/jacobs/pipeline")]


def test_la_plataforma_no_despacha_motores():
    with TestClient(_app()) as c:
        r = c.post("/motor/dispatch", json={"caller": "jax_platform_chat"},
                   headers=_h(IDENTIDAD_PLATAFORMA))
    assert r.status_code == 403
    assert r.json() == {"detail": {"code": auth_servicio.CODIGO_RUTA_NO_PERMITIDA}}


def test_el_cuerpo_llega_intacto_a_la_ruta():
    """El middleware lee el cuerpo y lo reproduce: la ruta ve lo mismo."""
    with patch("motor_registry.routes.check_facet_admission",
               AsyncMock(return_value=(True, "ok"))) as admision, TestClient(_app()) as c:
        r = c.post("/motor/authorize-facet", json={"caller": "jax_platform_chat", "facet": "kimi"},
                   headers=_h(IDENTIDAD_PLATAFORMA))
    assert r.status_code == 200, r.text
    admision.assert_awaited_once_with("jax_platform_chat", "kimi")


# ---------------------------------------------------------------------------
#  Fail-closed al cargar
# ---------------------------------------------------------------------------

def _entorno(**valores) -> dict[str, str]:
    base = {VARIABLES[k]: v for k, v in CRED.items()}
    base.update(valores)
    return {k: v for k, v in base.items() if v is not None}


@pytest.mark.parametrize("variable", sorted(VARIABLES.values()))
def test_sin_la_variable_no_arranca(variable):
    with pytest.raises(EntornoInvalido, match=variable):
        cargar_credenciales(_entorno(**{variable: None}))
    with pytest.raises(EntornoInvalido, match=variable):
        cargar_credenciales(_entorno(**{variable: "   "}))


def test_credencial_corta_o_con_caracteres_raros_no_arranca():
    var = VARIABLES[IDENTIDAD_PLATAFORMA]
    with pytest.raises(EntornoInvalido):
        cargar_credenciales(_entorno(**{var: "a" * 42}))
    with pytest.raises(EntornoInvalido):
        cargar_credenciales(_entorno(**{var: "a" * 42 + "\n" + "b"}))


def test_credenciales_iguales_no_arranca():
    igual = secrets.token_urlsafe(32)
    with pytest.raises(EntornoInvalido):
        cargar_credenciales({v: igual for v in VARIABLES.values()})


def test_proteger_sin_credenciales_en_el_entorno_falla_cerrado(monkeypatch):
    for variable in VARIABLES.values():
        monkeypatch.delenv(variable, raising=False)
    with pytest.raises(EntornoInvalido):
        proteger(FastAPI())


# ---------------------------------------------------------------------------
#  Cableado real: server.py, Jacobs y la jaula de Hyde
# ---------------------------------------------------------------------------

def test_server_instala_la_proteccion_al_importar():
    arbol = ast.parse((RAIZ / "las_manos" / "server.py").read_text(encoding="utf-8"))
    llamadas = [n for n in arbol.body
                if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                and getattr(n.value.func, "id", None) == "proteger"
                and [getattr(a, "id", None) for a in n.value.args] == ["app"]]
    assert len(llamadas) == 1, "server.py tiene que llamar proteger(app) a nivel de módulo"


def test_jacobs_ya_no_hace_ningun_pedido_http_a_las_manos():
    """Pipeline no despacha motores directamente (despacho legacy cerrado, 410): el
    ultimo pedido a LAS MANOS era la cancelacion de un motor job, borrada con el
    resto del codigo muerto. Se fija el comportamiento, no una constante: ningun
    modulo de jacobs/ hace un pedido HTTP que apunte a LAS MANOS ni presenta su
    credencial. Si vuelve un pedido, vuelve con ejecucion gobernada y esta prueba
    se reescribe a proposito."""
    hallazgos = []
    for p in sorted((RAIZ / "jacobs").glob("*.py")):
        if p.name.endswith("_test.py"):
            continue
        fuente = p.read_text(encoding="utf-8")
        for n in ast.walk(ast.parse(fuente)):
            if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    and n.func.attr in {"get", "post", "put", "delete", "request", "stream"} and n.args):
                destino = ast.unparse(n.args[0]).lower()
                if any(m in destino for m in ("las_manos", "/motor/", "7777")):
                    hallazgos.append(f"{p.name}:{n.lineno} {ast.unparse(n)[:80]}")
        # Office callbacks authenticate to the paired JXP service with the
        # existing Jacobs identity.  That is a bounded outbound callback, not
        # a request back into LAS MANOS; all other Jacobs modules remain
        # forbidden from presenting a LAS MANOS service credential.
        if p.name != "office_callbacks.py" and ("auth_servicio" in fuente or "encabezado_propio" in fuente):
            hallazgos.append(f"{p.name}: usa la credencial de servicio de LAS MANOS")
    assert hallazgos == []


def test_encabezado_propio_sale_del_entorno_y_falla_cerrado(monkeypatch):
    for identidad, variable in VARIABLES.items():
        monkeypatch.setenv(variable, CRED[identidad])
    assert auth_servicio.encabezado_propio(IDENTIDAD_JACOBS) == {ENCABEZADO: CRED[IDENTIDAD_JACOBS]}
    monkeypatch.delenv(VARIABLES[IDENTIDAD_JACOBS])
    with pytest.raises(EntornoInvalido):
        auth_servicio.encabezado_propio(IDENTIDAD_JACOBS)
