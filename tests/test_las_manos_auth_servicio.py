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
    ("get", "/motor/job/j1"),
    ("post", "/motor/job/j1/cancel"),
])
def test_jacobs_ya_no_tiene_permiso_sobre_los_motor_jobs(metodo, ruta):
    """Jacobs no consulta ni cancela motor jobs: nada del arbol usa esas rutas
    con la credencial `jacobs`. Un permiso que nadie usa es superficie sin
    dueno: 403 de ruta (cuerpo de siempre, sin correlacion: no es el dispatch)."""
    with TestClient(_app()) as c:
        r = getattr(c, metodo)(ruta, headers=_h(IDENTIDAD_JACOBS))
    assert r.status_code == 403, r.text
    assert r.json() == {"detail": {"code": auth_servicio.CODIGO_RUTA_NO_PERMITIDA}}


# --- POST /motor/dispatch: la denegacion REAL ocurre en el middleware ---------
# `proteger(app)` responde 403 antes de que dispatch() corra, para las dos
# identidades; ahi es donde la evidencia B7 y la correlacion tienen que vivir.
# Estas pruebas van por HTTP (TestClient sobre la app envuelta), no llaman a
# routes.dispatch() directo.

_HEX32 = r"[0-9a-f]{32}"


@pytest.fixture
def evidencia_b7(monkeypatch):
    from unittest.mock import Mock
    from motor_registry import routes
    registrador = Mock()
    monkeypatch.setattr(routes, "_B7_EVIDENCE_RECORDER", registrador)
    return registrador


def _correlaciones_del_log(caplog):
    import re
    return [m.group(1) for r in caplog.records
            for m in [re.search(rf"correlacion=({_HEX32})", r.getMessage())] if m]


@pytest.mark.parametrize("identidad", [IDENTIDAD_JACOBS, IDENTIDAD_PLATAFORMA])
def test_dispatch_denegado_por_el_middleware_registra_evidencia_y_trae_correlacion(identidad, evidencia_b7, caplog):
    import logging
    import re
    with caplog.at_level(logging.INFO), TestClient(_app()) as c:
        r = c.post("/motor/dispatch", json={"caller": "hyde" if identidad == IDENTIDAD_JACOBS else "jax_platform_chat",
                                            "capability": "code", "prompt": "PROMPT-SECRETO"},
                   headers=_h(identidad))
    assert r.status_code == 403
    cuerpo = r.json()
    assert set(cuerpo) == {"detail"} and set(cuerpo["detail"]) == {"code", "correlacion", "evidencia_registrada"}
    assert cuerpo["detail"]["code"] == auth_servicio.CODIGO_RUTA_NO_PERMITIDA
    assert re.fullmatch(_HEX32, cuerpo["detail"]["correlacion"])
    assert cuerpo["detail"]["evidencia_registrada"] is True
    evidencia_b7.record_governed_dispatch_denied.assert_called_once_with()
    # el log lleva el MISMO id y nada del pedido
    assert set(_correlaciones_del_log(caplog)) == {cuerpo["detail"]["correlacion"]}
    assert "PROMPT-SECRETO" not in caplog.text


def test_dispatch_denegado_aunque_falle_la_evidencia_sigue_siendo_403_con_el_mismo_id(evidencia_b7, caplog):
    import logging
    evidencia_b7.record_governed_dispatch_denied.side_effect = RuntimeError("sin disco")
    with caplog.at_level(logging.INFO), TestClient(_app()) as c:
        r = c.post("/motor/dispatch", json={"caller": "hyde"}, headers=_h(IDENTIDAD_JACOBS))
    assert r.status_code == 403
    id_cliente = r.json()["detail"]["correlacion"]
    assert r.json()["detail"]["evidencia_registrada"] is False
    errores = [x for x in caplog.records if x.levelname == "ERROR"]
    assert len(errores) == 1 and errores[0].exc_info and "sin disco" in str(errores[0].exc_info[1])
    assert set(_correlaciones_del_log(caplog)) == {id_cliente}


def test_dispatch_sin_credencial_es_401_y_no_registra_evidencia(evidencia_b7):
    with TestClient(_app()) as c:
        r = c.post("/motor/dispatch", json={"caller": "hyde"})
    assert r.status_code == 401
    assert r.json() == {"detail": {"code": auth_servicio.CODIGO_SIN_CREDENCIAL}}
    evidencia_b7.record_governed_dispatch_denied.assert_not_called()


def test_otras_denegaciones_del_middleware_no_tocan_la_evidencia_de_dispatch(evidencia_b7):
    with TestClient(_app()) as c:
        assert c.get("/motor/job/j1", headers=_h(IDENTIDAD_JACOBS)).status_code == 403
        assert c.post("/execute", headers=_h(IDENTIDAD_JACOBS)).status_code == 403
    evidencia_b7.record_governed_dispatch_denied.assert_not_called()


def test_jacobs_conserva_solo_el_pipeline_de_sub_pipelines():
    permiso = auth_servicio.PERMISOS[IDENTIDAD_JACOBS]
    assert [(m, p.pattern) for m, p in permiso.rutas] == [("POST", "/jacobs/pipeline")]


def test_la_plataforma_no_despacha_motores(evidencia_b7):
    with TestClient(_app()) as c:
        r = c.post("/motor/dispatch", json={"caller": "jax_platform_chat"},
                   headers=_h(IDENTIDAD_PLATAFORMA))
    assert r.status_code == 403
    detalle = r.json()["detail"]
    assert detalle["code"] == auth_servicio.CODIGO_RUTA_NO_PERMITIDA and set(detalle) == {"code", "correlacion", "evidencia_registrada"}


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
        if "auth_servicio" in fuente or "encabezado_propio" in fuente:
            hallazgos.append(f"{p.name}: usa la credencial de servicio de LAS MANOS")
    assert hallazgos == []


def test_encabezado_propio_sale_del_entorno_y_falla_cerrado(monkeypatch):
    for identidad, variable in VARIABLES.items():
        monkeypatch.setenv(variable, CRED[identidad])
    assert auth_servicio.encabezado_propio(IDENTIDAD_JACOBS) == {ENCABEZADO: CRED[IDENTIDAD_JACOBS]}
    monkeypatch.delenv(VARIABLES[IDENTIDAD_JACOBS])
    with pytest.raises(EntornoInvalido):
        auth_servicio.encabezado_propio(IDENTIDAD_JACOBS)


def test_la_jaula_de_hyde_no_recibe_la_credencial(monkeypatch, tmp_path):
    """Hyde corre como fruiz igual que los servicios: lo único que lo separa de
    la credencial es la jaula. Entorno limpio, PID propio, sin /etc/jax.

    Se prueba sobre el argv que arma la función REAL (`wrap_hyde_command`), sin
    ejecutar bwrap: el runner de CI no lo tiene. Lo único que se sustituye es la
    ruta del binario (un ejecutable vacío, para pasar el chequeo fail-closed de
    existencia), el directorio del template de $HOME (el real vive bajo
    /home/fruiz, que en el runner no existe) y la credencial de Anthropic (ver
    abajo). La forma de la jaula no cambia.
    """
    import hyde_sandbox

    bwrap_falso = tmp_path / "bwrap"
    bwrap_falso.write_text("#!/bin/sh\nexit 99\n", encoding="ascii")
    bwrap_falso.chmod(0o755)
    monkeypatch.setattr(hyde_sandbox, "_BWRAP_BIN", str(bwrap_falso))
    monkeypatch.setattr(hyde_sandbox, "_TEMPLATE_DIR", tmp_path / "home-template")
    # Credencial de Anthropic determinista: este test no es sobre ESA
    # credencial (es sobre JAX_LAS_MANOS_CREDENCIAL_*, más abajo), pero desde
    # que wrap_hyde_command falla cerrado sin ninguna credencial usable
    # (HydeCredentialUnavailable, ver hyde_sandbox.py) necesita una para
    # poder construir el argv que el resto del test inspecciona. No se puede
    # depender de que la máquina que corre la suite tenga
    # ~/.claude/.credentials.json (no existe en el runner de CI) ni de que
    # CLAUDE_CODE_OAUTH_TOKEN esté en el entorno ambiente.
    monkeypatch.delenv(hyde_sandbox.HYDE_OAUTH_TOKEN_ENV, raising=False)
    credencial_anthropic = tmp_path / "credencial-anthropic-de-mentira.json"
    credencial_anthropic.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(hyde_sandbox, "REAL_CREDENTIALS", str(credencial_anthropic))
    # El proceso padre TIENE las credenciales en su entorno, como LAS MANOS.
    for identidad, variable in VARIABLES.items():
        monkeypatch.setenv(variable, CRED[identidad])

    workspace = tmp_path / "workspace"
    # wrap_hyde_command devuelve (argv, env) desde B-1 (auditoría
    # adversarial 2026-09-27, ver hyde_sandbox.py): la frontera de entorno
    # ya no es --clearenv/--setenv dentro del argv, sino el `env` mínimo
    # que el llamador (run_sandboxed_claude) pasa TAL CUAL a
    # create_subprocess_exec -- nunca fusionado con os.environ. El motivo
    # es que el argv completo de un proceso es legible por CUALQUIER
    # usuario del host vía /proc/<pid>/cmdline, a diferencia del entorno
    # (/proc/<pid>/environ exige el mismo UID o CAP_SYS_PTRACE).
    argv, env = hyde_sandbox.wrap_hyde_command(["claude", "-p"], str(workspace))
    assert argv[0] == str(bwrap_falso) and argv[-2:] == ["claude", "-p"]
    jaula = argv[:argv.index("--")]

    # (a) namespaces propios, y SIN --clearenv/--setenv en absoluto (B-1):
    # sin esas banderas bwrap hereda sin tocarlo el entorno de quien lo
    # exec-ea, que es exactamente `env` -- por eso ya no hace falta
    # --clearenv para bloquear nada acá.
    assert "--unshare-all" in jaula and "--proc" in jaula
    assert "--clearenv" not in jaula, jaula
    assert "--setenv" not in jaula, jaula

    # (b) ningún montaje de /etc/jax ni de /etc/jax/.env (ni de /etc entero).
    montajes = {"--bind", "--ro-bind", "--dev-bind", "--bind-try", "--ro-bind-try",
                "--dev-bind-try", "--file", "--bind-data", "--ro-bind-data"}
    for i, a in enumerate(jaula):
        if a in montajes:
            origen, destino = jaula[i + 1], jaula[i + 2]
            for ruta in (origen, destino):
                assert ruta.rstrip("/") not in ("/etc", "/etc/jax"), jaula[i:i + 3]
                assert not ruta.startswith("/etc/jax"), jaula[i:i + 3]
    for arg in argv:
        assert "/etc/jax" not in arg, arg
        assert not any(v in arg for v in CRED.values()), "credencial en el argv de la jaula"

    # (c) el `env` de retorno -- que es la única vía de entrada de
    # variables ahora -- es SIEMPRE {HOME, PATH, LANG} (sin token en este
    # test, se borró arriba): ninguna JAX_LAS_MANOS_CREDENCIAL_* (ni otra
    # JAX_*) entra por ahí.
    assert set(env) == {"HOME", "PATH", "LANG"}, env
    assert not any(v.startswith("JAX_LAS_MANOS_CREDENCIAL_") for v in env)
    assert all(v.startswith("JAX_LAS_MANOS_CREDENCIAL_") for v in VARIABLES.values())
    assert set(env).isdisjoint(VARIABLES.values())
    assert not any(v in env.values() for v in CRED.values()), "credencial en el env de la jaula"


# --- El registro de la evidencia esta ACOTADO: hilo, plazo y tope --------------
# La denegacion de POST /motor/dispatch corre en el camino de cualquier pedido
# denegado; escribir la evidencia (sincrona, a MariaDB) no puede bloquear el bucle,
# colgar al cliente ni acumular hilos sin limite.

@pytest.fixture
def registro_acotado(monkeypatch):
    """Devuelve una funcion que (re)configura timeout/concurrencia y el aviso."""
    from motor_registry import routes

    def configurar(timeout="3", concurrencia="4", aviso_s=60.0):
        monkeypatch.setattr(routes, "INTERVALO_AVISO_OMITIDAS_S", aviso_s)
        return routes.configurar_registro_de_denegaciones({
            routes.VARIABLE_TIMEOUT_DENEGACION: timeout,
            routes.VARIABLE_CONCURRENCIA_DENEGACION: concurrencia,
        })
    yield configurar
    routes.configurar_registro_de_denegaciones({})


def _cliente_asgi(app):
    import httpx
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://las-manos.test")


def test_la_evidencia_se_escribe_fuera_del_hilo_del_bucle(evidencia_b7, registro_acotado):
    import asyncio
    import threading
    from motor_registry import routes
    registro_acotado()
    visto = {}

    def escribir():
        visto["hilo"] = threading.get_ident()

    evidencia_b7.record_governed_dispatch_denied.side_effect = escribir

    async def correr():
        visto["bucle"] = threading.get_ident()
        return await routes.registrar_denegacion_de_dispatch("a" * 32)

    assert asyncio.run(correr()) == routes.EVIDENCIA_REGISTRADA
    assert visto["hilo"] != visto["bucle"]


def test_un_registrador_lento_no_retiene_el_403_mas_alla_del_plazo(evidencia_b7, registro_acotado, caplog):
    import asyncio
    import logging
    import threading
    import time
    registro_acotado(timeout="0.1")
    terminado = threading.Event()

    def lento():
        time.sleep(0.8)
        terminado.set()

    evidencia_b7.record_governed_dispatch_denied.side_effect = lento

    async def correr():
        async with _cliente_asgi(_app()) as c:
            t0 = time.monotonic()
            r = await c.post("/motor/dispatch", json={"caller": "hyde"}, headers=_h(IDENTIDAD_JACOBS))
            return r, time.monotonic() - t0, terminado.is_set()

    with caplog.at_level(logging.INFO):
        r, demora, ya_termino = asyncio.run(correr())
    assert r.status_code == 403
    assert demora < 0.6 and not ya_termino, (demora, ya_termino)  # salio ANTES de que el registrador terminara
    detalle = r.json()["detail"]
    assert detalle["evidencia_registrada"] is False
    errores = [x for x in caplog.records if x.levelname == "ERROR"]
    assert len(errores) == 1 and detalle["correlacion"] in errores[0].getMessage()
    assert terminado.wait(2)  # el hilo no se cancela; termina por su cuenta


def test_cincuenta_denegaciones_simultaneas_respetan_el_tope_y_cuentan_las_omitidas(
        evidencia_b7, registro_acotado, caplog):
    import asyncio
    import logging
    import threading
    import time
    from motor_registry import routes
    registro_acotado(timeout="5", concurrencia="4", aviso_s=0.2)
    liberar = threading.Event()
    lock = threading.Lock()
    estadisticas = {"en_vuelo": 0, "maximo": 0, "llamadas": 0}

    def escribir():
        with lock:
            estadisticas["en_vuelo"] += 1
            estadisticas["llamadas"] += 1
            estadisticas["maximo"] = max(estadisticas["maximo"], estadisticas["en_vuelo"])
        liberar.wait(5)
        with lock:
            estadisticas["en_vuelo"] -= 1

    evidencia_b7.record_governed_dispatch_denied.side_effect = escribir

    async def correr():
        async with _cliente_asgi(_app()) as c:
            async def pedir():
                return await c.post("/motor/dispatch", json={"caller": "hyde"}, headers=_h(IDENTIDAD_JACOBS))

            async def soltar_luego():
                await asyncio.sleep(0.3)
                liberar.set()

            resultados = await asyncio.gather(*[pedir() for _ in range(50)], soltar_luego())
            await asyncio.sleep(0.5)  # deja correr el aviso agregado
            return resultados[:-1]

    with caplog.at_level(logging.INFO):
        respuestas = asyncio.run(correr())
    assert all(r.status_code == 403 for r in respuestas)
    registradas = [r for r in respuestas if r.json()["detail"]["evidencia_registrada"]]
    omitidas = [r for r in respuestas if not r.json()["detail"]["evidencia_registrada"]]
    assert len(registradas) == 4 and len(omitidas) == 46
    assert estadisticas["maximo"] <= 4 and estadisticas["llamadas"] == 4
    assert routes._DENEGACION["omitidas_total"] == 46
    avisos = [x for x in caplog.records if x.levelname == "WARNING" and "omitida" in x.getMessage()]
    assert len(avisos) == 1 and "46" in avisos[0].getMessage()
    # a cada omitida igual le llega su correlacion unica
    assert len({r.json()["detail"]["correlacion"] for r in respuestas}) == 50


def test_el_cupo_se_libera_al_terminar_la_escritura(evidencia_b7, registro_acotado):
    import asyncio
    from motor_registry import routes
    registro_acotado(concurrencia="1")

    async def correr():
        a = await routes.registrar_denegacion_de_dispatch("a" * 32)
        b = await routes.registrar_denegacion_de_dispatch("b" * 32)
        return a, b

    assert asyncio.run(correr()) == (routes.EVIDENCIA_REGISTRADA, routes.EVIDENCIA_REGISTRADA)


@pytest.mark.parametrize("variable,valor", [
    ("JAX_B7_DENEGACION_TIMEOUT_S", "0"), ("JAX_B7_DENEGACION_TIMEOUT_S", "-1"),
    ("JAX_B7_DENEGACION_TIMEOUT_S", "abc"), ("JAX_B7_DENEGACION_TIMEOUT_S", "nan"),
    ("JAX_B7_DENEGACION_TIMEOUT_S", "61"),
    ("JAX_B7_DENEGACION_CONCURRENCIA", "0"), ("JAX_B7_DENEGACION_CONCURRENCIA", "-2"),
    ("JAX_B7_DENEGACION_CONCURRENCIA", "2.5"), ("JAX_B7_DENEGACION_CONCURRENCIA", "65"),
])
def test_la_configuracion_invalida_del_registro_impide_arrancar(variable, valor, monkeypatch):
    """Se valida al arrancar: configure_b7_evidence_recorder (composicion de
    startup) levanta EntornoInvalido y el servicio no arranca."""
    from motor_registry import routes
    monkeypatch.setattr(routes, "_B7_EVIDENCE_RECORDER", None)
    with pytest.raises(EntornoInvalido):
        routes.configurar_registro_de_denegaciones({variable: valor})
    monkeypatch.setenv(variable, valor)
    with pytest.raises(EntornoInvalido):
        routes.configure_b7_evidence_recorder(object())


def test_la_configuracion_por_defecto_es_3_segundos_y_4_escrituras(registro_acotado):
    from motor_registry import routes
    estado = routes.configurar_registro_de_denegaciones({})
    assert (estado["timeout"], estado["cupo"]) == (3.0, 4)


def test_sin_registrador_B7_la_evidencia_no_figura_como_registrada(monkeypatch):
    from motor_registry import routes
    monkeypatch.setattr(routes, "_B7_EVIDENCE_RECORDER", None)
    with TestClient(_app()) as c:
        r = c.post("/motor/dispatch", json={"caller": "hyde"}, headers=_h(IDENTIDAD_JACOBS))
    assert r.status_code == 403 and r.json()["detail"]["evidencia_registrada"] is False


def test_el_log_une_la_correlacion_con_el_observation_id_de_la_fila_b7(evidencia_b7, caplog):
    """El registrador devuelve la observacion persistida: su observation_id va en
    el log junto a la correlacion, para unir el 403 que vio el cliente con la fila
    B7 sin depender de la hora."""
    import logging
    from types import SimpleNamespace
    evidencia_b7.record_governed_dispatch_denied.return_value = SimpleNamespace(observation_id="obs-1234")
    with caplog.at_level(logging.INFO), TestClient(_app()) as c:
        r = c.post("/motor/dispatch", json={"caller": "hyde"}, headers=_h(IDENTIDAD_JACOBS))
    correlacion = r.json()["detail"]["correlacion"]
    lineas = [x.getMessage() for x in caplog.records if "observation_id=obs-1234" in x.getMessage()]
    assert len(lineas) == 1 and f"correlacion={correlacion}" in lineas[0]
