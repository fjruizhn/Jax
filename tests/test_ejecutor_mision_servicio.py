# tests/test_ejecutor_mision_servicio.py
"""SP2: el proceso que lanza la plataforma para un turno. Pedido por stdin, eventos por stdout
(una línea JSON cada uno) y SIEMPRE una última línea `resultado`, también cuando el pedido es
ilegible, falta configuración o algo revienta: la plataforma nunca queda adivinando."""
import asyncio
import io
import json
import uuid
from pathlib import Path

import pytest

from jax.ejecutor import mision as M
from jax.ejecutor import mision_servicio as S
from jax.ejecutor.contratos import arranque as AR
from jax.ejecutor.contratos.cuenta_axioma import CuentaSinConfigurar

TURNO = {"mision_id": str(uuid.uuid4()), "n": 1, "sesion": str(uuid.uuid4()), "objetivo": "x",
         "instruccion": "x", "hosts": ["ejecutor-prueba"]}
ENV = {"JAX_EJECUTOR_TURNO_TOPE_S": "900", "JAX_EJECUTOR_VIGIA_ESPERA_S": "120"}


def _correr(monkeypatch, entrada: bytes, correr_turno=None, env=ENV):
    salida = io.StringIO()
    if correr_turno is not None:
        monkeypatch.setattr(M, "correr_turno", correr_turno)
    monkeypatch.setattr(S, "dependencias_reales", lambda env, turno, **topes: "deps")
    rc = S.principal(io.BytesIO(entrada), salida, dict(env))
    lineas = [json.loads(l) for l in salida.getvalue().splitlines()]
    return rc, lineas


def test_turno_completo_sale_0_con_los_eventos_y_el_resultado_al_final(monkeypatch):
    async def turno(t, deps, emitir):
        assert deps == "deps" and t.sesion == TURNO["sesion"]
        emitir(M.evento("turno_lanzado", 1))
        return {"estado": "completado", "codigo": None}
    rc, lineas = _correr(monkeypatch, json.dumps(TURNO).encode(), turno)
    assert rc == 0
    assert lineas == [{"evento": "turno_lanzado", "turno": 1, "datos": {}},
                      {"evento": "resultado", "turno": 1, "datos": {"estado": "completado", "codigo": None}}]


def test_turno_fallido_o_rechazado_sale_1(monkeypatch):
    async def turno(t, deps, emitir):
        return {"estado": "rechazado", "codigo": "arranque_rechazado"}
    rc, lineas = _correr(monkeypatch, json.dumps(TURNO).encode(), turno)
    assert rc == 1 and lineas[-1]["datos"]["estado"] == "rechazado"


def test_pedido_ilegible_da_resultado_con_codigo_y_sale_2(monkeypatch):
    rc, lineas = _correr(monkeypatch, b"{no json")
    assert rc == 2
    assert lineas == [{"evento": "resultado", "turno": None,
                       "datos": {"estado": "fallido", "codigo": "turno_ilegible", "detalle": "turno_no_es_json"}}]


@pytest.mark.parametrize("variable", sorted(ENV))
def test_sin_topes_configurados_no_se_lanza_nada(monkeypatch, variable):
    llamado = []

    async def turno(*a):
        llamado.append(1)
    rc, lineas = _correr(monkeypatch, json.dumps(TURNO).encode(), turno,
                         env={k: v for k, v in ENV.items() if k != variable})
    assert rc == 2 and llamado == []
    assert lineas[-1]["datos"] == {"estado": "fallido", "codigo": "sin_configurar", "detalle": variable}


@pytest.mark.parametrize("valor", ["0", "-1", "x", "inf", "nan"])
def test_tope_invalido_es_sin_configurar(monkeypatch, valor):
    rc, lineas = _correr(monkeypatch, json.dumps(TURNO).encode(), env={**ENV, "JAX_EJECUTOR_TURNO_TOPE_S": valor})
    assert rc == 2 and lineas[-1]["datos"]["codigo"] == "sin_configurar"


def test_configuracion_de_la_cuenta_ausente_es_sin_configurar(monkeypatch):
    async def turno(t, deps, emitir):
        raise CuentaSinConfigurar("JAX_EJECUTOR_CUENTA")
    rc, lineas = _correr(monkeypatch, json.dumps(TURNO).encode(), turno)
    assert rc == 2 and lineas[-1]["datos"] == {"estado": "fallido", "codigo": "sin_configurar",
                                               "detalle": "JAX_EJECUTOR_CUENTA"}


def test_una_excepcion_inesperada_es_runner_error_con_el_tipo_y_sin_el_mensaje(monkeypatch):
    async def turno(t, deps, emitir):
        raise RuntimeError("secreto que no debe salir")
    rc, lineas = _correr(monkeypatch, json.dumps(TURNO).encode(), turno)
    assert rc == 1 and lineas[-1]["datos"] == {"estado": "fallido", "codigo": "runner_error", "tipo": "RuntimeError"}
    assert "secreto" not in json.dumps(lineas)


def test_leer_pausa_fail_closed(tmp_path):
    ruta = tmp_path / "pausa"
    assert S.leer_pausa(ruta) == {"puesta": False, "legible": True, "origen": None, "motivo": None, "paso": None,
                                  "momento": None}
    ruta.write_text(json.dumps({"origen": "c5", "motivo": "prohibido", "paso": 4, "momento": "2026-09-17T10:00:00+00:00"}))
    assert S.leer_pausa(ruta) == {"puesta": True, "legible": True, "origen": "c5", "motivo": "prohibido", "paso": 4,
                                  "momento": "2026-09-17T10:00:00+00:00"}
    ruta.write_text("{roto")
    assert S.leer_pausa(ruta)["puesta"] is True and S.leer_pausa(ruta)["legible"] is False
    ruta.write_text(json.dumps({"motivo": 3, "paso": "x"}))
    assert S.leer_pausa(ruta) == {"puesta": True, "legible": True, "origen": None, "motivo": None, "paso": None,
                                  "momento": None}


def test_leer_pausa_trae_el_detalle_solo_si_es_texto(tmp_path):
    ruta = tmp_path / "pausa"
    ruta.write_text(json.dumps({"origen": "c5", "motivo": "auditor_ilegible", "detalle": "proveedor_fallo"}))
    assert S.leer_pausa(ruta)["detalle"] == "proveedor_fallo"
    ruta.write_text(json.dumps({"origen": "c5", "motivo": "auditor_ilegible", "detalle": {"x": 1}}))
    assert "detalle" not in S.leer_pausa(ruta)


# --- ronda 5, auditoría adversarial 2026-09-22: SIN unidad systemd -----------------------
#
# `ejecutor-vigia@.service` se retiró: código muerto, nunca arrancó en producción (verificado
# por el coordinador contra el journal). El camino REAL es este -- `abrir_vigia` lanza
# `vigia_servicio` como subproceso DIRECTO, heredando la identidad de quien corre ESTE
# proceso (jax-platform, `fruiz`). Antes esto lo cubría (débilmente, indirecto) un test sobre
# el contenido del archivo de la unidad; con la unidad fuera, el default de `abrir_vigia` es
# lo único que documenta el comando real, y no tenía una prueba propia.
def test_abrir_vigia_por_defecto_lanza_el_modulo_como_subproceso_directo(tmp_path, monkeypatch):
    """Sin `argv=` explícito (el caso real, el que usa `dependencias_reales`), `abrir_vigia`
    tiene que lanzar exactamente `python -m jax.ejecutor.contratos.vigia_servicio <ruta>` --
    ni una unidad systemd, ni `sudo`, ni ningún cambio de cuenta: el proceso hereda la
    identidad de quien lo llama."""
    import sys

    vistos = {}
    original = asyncio.create_subprocess_exec

    async def espia(*argv, **kwargs):
        vistos["argv"] = argv
        falso = tmp_path / "no_arranca_de_verdad.py"
        falso.write_text("import sys; sys.exit(0)\n")
        return await original(sys.executable, str(falso), **kwargs)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", espia)
    asyncio.run(S.abrir_vigia(tmp_path, "m-t9", "texto", frozenset({"a"})))
    argv = vistos["argv"]
    assert argv[0] == sys.executable
    assert argv[1:3] == ("-m", "jax.ejecutor.contratos.vigia_servicio")
    assert argv[3] == str(tmp_path / "m-t9.json")


def test_el_vigia_se_abre_con_el_archivo_de_mision_y_se_cierra_con_sigterm(tmp_path, monkeypatch):
    """El vigía de verdad es `vigia_servicio`; acá un proceso falso que imprime lo que el vigía
    imprime al cerrar y termina con SIGTERM, para probar el manejo del proceso y del archivo."""
    import sys
    falso = tmp_path / "vigia_falso.py"
    falso.write_text("import signal, sys, time\n"
                     "signal.signal(signal.SIGTERM, lambda *a: (print('arranco=true cerrada=true', flush=True), sys.exit(0)))\n"
                     "print(open(sys.argv[-1]).read(), file=sys.stderr, flush=True)\n"
                     "time.sleep(30)\n")

    async def probar():
        v = await S.abrir_vigia(tmp_path, "m-t1", "texto", frozenset({"b", "a"}),
                                argv=[sys.executable, str(falso)])
        await asyncio.sleep(0.3)
        assert v.vive()
        assert json.loads((tmp_path / "m-t1.json").read_text()) == {"mision": "texto", "hosts": ["a", "b"],
                                                                     "tipo": None}
        rc, salida, err = await v.cerrar()
        # El proceso falso escribe el archivo de mision en stderr: sirve para
        # comprobar, de paso, que el stderr YA NO se tira (2026-09-20).
        assert "texto" in err, err
        return rc, salida
    rc, salida = asyncio.run(probar())
    assert rc == 0 and "cerrada=true" in salida and not (tmp_path / "m-t1.json").exists()


def test_abrir_vigia_incluye_el_tipo_recibido_en_el_archivo_de_mision(tmp_path, monkeypatch):
    """Ruling del coordinador (seguimiento Tarea 12/13): sin el tipo en el archivo que lee el
    vigía, el contrato "codigo" queda dormido -- el vigía es quien de verdad exige los
    contratos, en un proceso aparte que no comparte memoria con el turno."""
    import sys
    falso = tmp_path / "no_arranca_de_verdad.py"
    falso.write_text("import sys; sys.exit(0)\n")
    asyncio.run(S.abrir_vigia(tmp_path, "m-t2", "texto", frozenset({"a"}),
                              argv=[sys.executable, str(falso)], tipo="codigo"))
    assert json.loads((tmp_path / "m-t2.json").read_text())["tipo"] == "codigo"


def test_el_cerebro_le_da_bash_y_skill_al_arnes(monkeypatch):
    """El Ejecutor tiene skills (cerebros.toml `skills`, 2026-09-22): `remoto_claude`
    necesita `Skill` en --allowedTools para poder invocarlas, además de `Bash`.
    `Read` no hace falta -- el Ejecutor lee con `cat` (Bash), como siempre."""
    vistos = {}

    def remoto_falso(cuenta, *, base_url, modelo, prompt, herramientas, max_salida_tokens, sesion, reanudar,
                     directorio_projects):
        vistos["herramientas"] = herramientas
        return "remoto-de-prueba"

    async def correr_falso(cuenta, remoto, *, entrada, tope_s):
        return 0, b"{}", b""

    async def preparar_falso(cuenta, ruta):
        vistos["directorio_preparado"] = ruta

    monkeypatch.setattr(S.cuenta_axioma, "remoto_claude", remoto_falso)
    monkeypatch.setattr(S.cuenta_axioma, "correr_en_la_cuenta", correr_falso)
    monkeypatch.setattr(S.cuenta_axioma, "preparar_directorio_projects", preparar_falso)

    turno = M.Turno(**{**TURNO, "hosts": frozenset(TURNO["hosts"])})
    env = {"JAX_PROXY_CARRIL_MODELO": "canario", "JAX_PROXY_CARRIL_MAX_SALIDA_TOKENS": "1024",
          "JAX_EJECUTOR_MISIONES": "/var/lib/jax-ejecutor-misiones"}
    deps = S.dependencias_reales(env, turno, tope_s=1.0, espera_s=1.0)

    class _CtxFalso:
        cuenta = object()
        puerto_proxy = 18435

    asyncio.run(deps.correr_cerebro(_CtxFalso(), "prompt", None, False))
    assert vistos["herramientas"] == "Bash,Skill"
    assert vistos["directorio_preparado"] == Path(
        f"/var/lib/jax-ejecutor-misiones/{TURNO['mision_id']}/claude-projects")


# --- misión de CÓDIGO (spec 2026-09-28 v1.3, Tarea 9) ------------------------------------------

from jax.ejecutor.codigo import mision_codigo as MC  # noqa: E402
from jax.ejecutor.codigo import preparar as P  # noqa: E402
from jax.ejecutor import transporte, cita  # noqa: E402

TOKEN = "github_pat_TOKEN_FALSO_" + "0" * 30
ENV_CODIGO = {"JAX_PROXY_CARRIL_MODELO": "qwen-carril", "JAX_PROXY_CARRIL_MAX_SALIDA_TOKENS": "1024",
              "JAX_EJECUTOR_MISIONES": "/var/lib/jax-ejecutor-misiones", "JAX_GITHUB_TOKEN": TOKEN}


def _turno_codigo() -> M.Turno:
    return M.Turno(**{**TURNO, "hosts": frozenset(TURNO["hosts"])}, tipo="codigo",
                   repo={"owner_repo": "o/r", "comandos_prueba": ("pytest -q",)})


class _Cuenta:
    nombre = "axioma"
    puerto = 2222
    llave = Path("/etc/jax-ejecutor/llave")
    node_bin = Path("/opt/ejecutor/node/bin")


class _CtxCodigo:
    cuenta = _Cuenta()
    puerto_proxy = 18435


CLON = P.Clon(Path(f"/var/lib/jax-ejecutor-misiones/{TURNO['mision_id']}/repo"), f"axioma/{TURNO['mision_id']}",
              "trunk", ("package-lock.json",), Path(f"/var/lib/jax-ejecutor-misiones/{TURNO['mision_id']}/espejo.git"))


@pytest.fixture
def codigo(monkeypatch):
    """Todo lo de afuera (API de GitHub, DB, preparar, entregar, la jaula) reemplazado por dobles
    que anotan con qué se los llamó."""
    vistos: dict = {"config_leida": 0}

    async def config():
        vistos["config_leida"] += 1
        return S.ConfigCodigo("Axioma Prueba <ax@prueba.io>", 1234, 5678)

    async def rama(cliente, repo):
        vistos["rama"] = (repo, cliente.token)
        return "trunk"

    async def preparar(repo, **kw):
        vistos["preparar"] = (repo, kw)
        return CLON

    async def entregar(clon, **kw):
        vistos["entregar"] = (clon, kw)
        return {"estado_entrega": "abierto", "pr_url": "https://gh/pr/1", "violaciones": [], "notas": []}

    def remoto(cuenta, **kw):
        vistos["remoto"] = kw
        return "remoto-de-prueba"

    async def correr(cuenta, remoto, *, entrada, tope_s):
        return 0, b"{}", b""

    async def nada(*a, **k):
        return None

    monkeypatch.setattr(S, "leer_config_codigo", config)
    monkeypatch.setattr(P, "rama_por_omision", rama)
    monkeypatch.setattr(P, "preparar", preparar)
    monkeypatch.setattr(MC, "entregar", entregar)
    monkeypatch.setattr(S.cuenta_axioma, "remoto_claude", remoto)
    monkeypatch.setattr(S.cuenta_axioma, "correr_en_la_cuenta", correr)
    monkeypatch.setattr(S.cuenta_axioma, "preparar_directorio_projects", nada)
    return vistos


def test_codigo_sin_token_no_se_configura():
    env = {k: v for k, v in ENV_CODIGO.items() if k != "JAX_GITHUB_TOKEN"}
    with pytest.raises(S.SinConfigurar) as exc:
        S.dependencias_reales(env, _turno_codigo(), tope_s=1.0, espera_s=1.0)
    assert exc.value.args[0] == "JAX_GITHUB_TOKEN"
    with pytest.raises(S.SinConfigurar):
        S.dependencias_reales({**env, "JAX_GITHUB_TOKEN": "  "}, _turno_codigo(), tope_s=1.0, espera_s=1.0)


def test_servidor_no_trae_dependencias_de_codigo_ni_pide_token():
    turno = M.Turno(**{**TURNO, "hosts": frozenset(TURNO["hosts"])})
    deps = S.dependencias_reales({}, turno, tope_s=1.0, espera_s=1.0)
    assert deps.preparar_codigo is None and deps.entregar_codigo is None


# --- ruling del coordinador, seguimiento Tarea 12/13: turno.tipo llega de verdad al contrato
# de arranque -- "un contrato dormido en producción no es un contrato" (Principio IX). Con
# DOBLES MÍNIMOS: `contexto()`/`arranque.contexto_desde_entorno` son parseo puro de entorno,
# sin red ni DB, así que corren de verdad; sólo el proceso del vigía (subprocess real) se
# reemplaza. ----------------------------------------------------------------------------

def _entorno_arranque(tmp_path):
    return {"JAX_EJECUTOR_CUENTA": "axioma", "JAX_EJECUTOR_SSH_PUERTO": "58291",
            "JAX_EJECUTOR_CONTROLADOR_LLAVE": "/k", "JAX_EJECUTOR_NODE_BIN": "/n",
            "JAX_EJECUTOR_LIB": str(tmp_path / "lib"), "JAX_EJECUTOR_CUENTA_HOME": "/home/axioma",
            "JAX_EJECUTOR_POLITICA": str(tmp_path / "politica.json"), "JAX_EJECUTOR_CANARIO_PUERTO": "18436",
            "JAX_EJECUTOR_REGISTRO": str(tmp_path / "r.jsonl"), "JAX_PROXY_CARRIL_PUERTO": "18435",
            "JAX_EJECUTOR_CERCO_SONDAS": "7777,11434", "JAX_EJECUTOR_FRENO_ESTADO": str(tmp_path / "e.json"),
            "JAX_EJECUTOR_LLAVES_ROOT": "/etc/ssh/authorized_keys.d/axioma", "JAX_EJECUTOR_GANCHO_TOPE_S": "10",
            "JAX_EJECUTOR_PAUSA": str(tmp_path / "PAUSA"),
            "JAX_EJECUTOR_VIGIA_LATIDO": str(tmp_path / "v.latido"), "JAX_EJECUTOR_VIGIA_LATIDO_MAX_S": "30"}


def test_deps_contexto_de_un_turno_de_codigo_trae_tipo_codigo_de_verdad(tmp_path):
    env = {**_entorno_arranque(tmp_path), **ENV_CODIGO}
    deps = S.dependencias_reales(env, _turno_codigo(), tope_s=1.0, espera_s=1.0)
    ctx = asyncio.run(deps.contexto())
    assert ctx.tipo == "codigo"
    assert "codigo" in AR.pruebas_reales(ctx)


def test_deps_contexto_de_un_turno_de_servidor_no_trae_codigo_de_verdad(tmp_path):
    env = _entorno_arranque(tmp_path)
    turno = M.Turno(**{**TURNO, "hosts": frozenset(TURNO["hosts"])})
    deps = S.dependencias_reales(env, turno, tope_s=1.0, espera_s=1.0)
    ctx = asyncio.run(deps.contexto())
    assert ctx.tipo == "servidor"
    assert "codigo" not in AR.pruebas_reales(ctx)


def _sin_arrancar_de_verdad(tmp_path, monkeypatch):
    import sys
    original = asyncio.create_subprocess_exec

    async def espia(*argv, **kwargs):
        falso = tmp_path / "vigia_no_arranca_de_verdad.py"
        falso.write_text("import sys; sys.exit(0)\n")
        return await original(sys.executable, str(falso), **kwargs)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", espia)


def test_deps_abrir_vigia_de_un_turno_de_codigo_escribe_tipo_codigo(tmp_path, monkeypatch):
    _sin_arrancar_de_verdad(tmp_path, monkeypatch)
    env = {**_entorno_arranque(tmp_path), **ENV_CODIGO, "JAX_EJECUTOR_MISIONES": str(tmp_path)}
    deps = S.dependencias_reales(env, _turno_codigo(), tope_s=1.0, espera_s=1.0)
    asyncio.run(deps.abrir_vigia(None, "m-tc", "texto", frozenset({"a"})))
    assert json.loads((tmp_path / "m-tc.json").read_text())["tipo"] == "codigo"


def test_deps_abrir_vigia_de_un_turno_de_servidor_escribe_tipo_servidor(tmp_path, monkeypatch):
    _sin_arrancar_de_verdad(tmp_path, monkeypatch)
    env = {**_entorno_arranque(tmp_path), "JAX_EJECUTOR_MISIONES": str(tmp_path)}
    turno = M.Turno(**{**TURNO, "hosts": frozenset(TURNO["hosts"])})
    deps = S.dependencias_reales(env, turno, tope_s=1.0, espera_s=1.0)
    asyncio.run(deps.abrir_vigia(None, "m-ts", "texto", frozenset({"a"})))
    assert json.loads((tmp_path / "m-ts.json").read_text())["tipo"] == "servidor"


def test_codigo_prepara_con_la_rama_de_la_api_el_autor_de_la_config_y_los_accesos_de_la_cuenta(codigo):
    deps = S.dependencias_reales(ENV_CODIGO, _turno_codigo(), tope_s=1.0, espera_s=1.0)
    assert asyncio.run(deps.preparar_codigo(_CtxCodigo())) is CLON
    assert codigo["rama"] == ("o/r", TOKEN)
    repo, kw = codigo["preparar"]
    assert repo == P.Repo("o/r", ("pytest -q",))
    assert kw["mision_id"] == TURNO["mision_id"] and kw["raiz"] == Path("/var/lib/jax-ejecutor-misiones")
    assert kw["rama_por_omision"] == "trunk" and kw["token"] == TOKEN and kw["autor"] == "Axioma Prueba <ax@prueba.io>"
    assert kw["node_bin"] == _Cuenta.node_bin and isinstance(kw["accesos"], P.Accesos)


def test_codigo_en_turno_2_usa_la_rama_guardada_sin_llamar_a_la_api(codigo, tmp_path):
    """MINOR-5: si ya hay `preparado.json` (turno >= 2), la base sale de ahí -- no de un GET
    fresco a GitHub, que además podría no coincidir con la que se usó en el turno 1."""
    mision_id = TURNO["mision_id"]
    (tmp_path / mision_id).mkdir()
    (tmp_path / mision_id / "preparado.json").write_text(
        json.dumps({"rama_por_omision": "trunk-guardada", "dependencias": ["package-lock.json"]}))
    env = {**ENV_CODIGO, "JAX_EJECUTOR_MISIONES": str(tmp_path)}
    deps = S.dependencias_reales(env, _turno_codigo(), tope_s=1.0, espera_s=1.0)
    asyncio.run(deps.preparar_codigo(_CtxCodigo()))
    assert "rama" not in codigo
    repo, kw = codigo["preparar"]
    assert kw["rama_por_omision"] == "trunk-guardada"


def test_codigo_el_cerebro_trabaja_en_el_clon_con_las_herramientas_de_codigo(codigo):
    deps = S.dependencias_reales(ENV_CODIGO, _turno_codigo(), tope_s=1.0, espera_s=1.0)
    asyncio.run(deps.preparar_codigo(_CtxCodigo()))
    asyncio.run(deps.correr_cerebro(_CtxCodigo(), "prompt", TURNO["sesion"], False))
    assert codigo["remoto"]["herramientas"] == "Bash,Read,Edit,Write,Glob,Grep,Skill"
    assert codigo["remoto"]["directorio_trabajo"] == CLON.ruta
    assert TOKEN not in json.dumps({k: str(v) for k, v in codigo["remoto"].items()})


def test_codigo_sin_clon_preparado_el_cerebro_no_corre(codigo):
    deps = S.dependencias_reales(ENV_CODIGO, _turno_codigo(), tope_s=1.0, espera_s=1.0)
    with pytest.raises(RuntimeError, match="codigo_sin_clon"):
        asyncio.run(deps.correr_cerebro(_CtxCodigo(), "prompt", TURNO["sesion"], False))
    assert "remoto" not in codigo


def test_codigo_entrega_con_token_tope_autor_y_upload_pack_de_la_cuenta(codigo):
    deps = S.dependencias_reales(ENV_CODIGO, _turno_codigo(), tope_s=1.0, espera_s=1.0)
    afirmacion = cita.Afirmacion("m", "ssh -tt m pytest", "3 passed", "3 passed", "¿pasan?")
    entrega = transporte.Entrega((), (afirmacion,), ())
    r = asyncio.run(deps.entregar_codigo(_CtxCodigo(), CLON, entrega, True))
    assert r["estado_entrega"] == "abierto"
    clon, kw = codigo["entregar"]
    assert clon is CLON and kw["repo"] == "o/r" and kw["mision_id"] == TURNO["mision_id"]
    assert kw["token"] == TOKEN and kw["tope_bytes"] == 1234 and kw["autor"] == "Axioma Prueba <ax@prueba.io>"
    assert kw["tope_total_bytes"] == 5678
    assert kw["modelo"] == "qwen-carril" and kw["revision_legible"] is True
    assert kw["upload_pack"].startswith("ssh ") and kw["upload_pack"].endswith("axioma@127.0.0.1 git-upload-pack")
    assert "'3 passed'" in kw["informe"] and "'ssh -tt m pytest'" in kw["informe"]
    assert kw["cliente"].token == TOKEN
    assert codigo["config_leida"] == 1


def test_codigo_la_entrega_relee_la_pausa_del_contexto(codigo, tmp_path):
    """MINOR-1: la pausa que `mision_codigo` vuelve a leer es la de ESTE contexto (ilegible = puesta)."""
    class Ctx(_CtxCodigo):
        pausa = tmp_path / "pausa"
    deps = S.dependencias_reales(ENV_CODIGO, _turno_codigo(), tope_s=1.0, espera_s=1.0)
    asyncio.run(deps.entregar_codigo(Ctx(), CLON, transporte.Entrega((), (), ()), True))
    leer = codigo["entregar"][1]["pausa_puesta"]
    assert asyncio.run(leer()) is False
    (tmp_path / "pausa").write_text("{}")
    assert asyncio.run(leer()) is True


def test_codigo_la_config_se_lee_una_vez_por_turno(codigo):
    deps = S.dependencias_reales(ENV_CODIGO, _turno_codigo(), tope_s=1.0, espera_s=1.0)
    asyncio.run(deps.preparar_codigo(_CtxCodigo()))
    asyncio.run(deps.entregar_codigo(_CtxCodigo(), CLON, transporte.Entrega((), (), ()), False))
    assert codigo["config_leida"] == 1
    assert codigo["preparar"][1]["autor"] == codigo["entregar"][1]["autor"]


def test_informe_c5_va_en_un_bloque_de_codigo_sangrado_sin_saltos_crudos():
    """Cada valor sale con `repr` (cita.presentar): sin saltos de línea crudos, así que nada
    puede salirse del bloque sangrado y convertirse en Markdown del PR."""
    a = cita.Afirmacion("m", "cmd", "linea\n## titulo falso", "linea", "p")
    texto = S.informe_c5(transporte.Entrega((), (a,), ()), "thot")
    assert "thot" in texto
    lineas = texto.splitlines()
    (item,) = [l for l in lineas if "cmd" in l]
    assert item.startswith("    ") and "\\n## titulo falso" in item
    assert not any(l.startswith("#") for l in lineas)


def test_informe_c5_incluye_las_descartadas_con_estado_y_motivo():
    """MAJOR-2 (DC5: Fernando revisa): lo que el verificador o C5 descartó también va en el PR. «3
    passed» contra una salida real «3 failed» no se respalda -- y se ve."""
    captura = cita.Captura(maquina="hall9000", comando="pytest -q", salida="3 failed in 0.10s\n", stderr="",
                           truncada=False)
    mala = cita.Afirmacion("hall9000", "pytest -q", "3 passed in 0.10s", "3 passed", "¿pasan?")
    entrega = transporte.entregar((mala,), (captura,))
    assert entrega.respaldadas == () and len(entrega.descartadas) == 1
    texto = S.informe_c5(entrega, "thot")
    assert "Descartadas por el verificador/C5" in texto
    (item,) = [l for l in texto.splitlines() if "'3 passed'" in l]
    d = entrega.descartadas[0]
    assert item.startswith("    ") and f"estado={d.estado}" in item and f"motivo={d.motivo.codigo}" in item


def test_informe_enorme_conserva_las_descartadas_y_recorta_primero_las_respaldadas():
    """MINOR-B (ronda final): «Descartadas» va ANTES que las respaldadas, así el recorte del cuerpo
    del PR (`cuerpo_del_pr`, que corta la cola del informe) se lleva primero las respaldadas."""
    captura = cita.Captura(maquina="hall9000", comando="pytest -q", salida="3 failed in 0.10s\n", stderr="",
                           truncada=False)
    mala = cita.Afirmacion("hall9000", "pytest -q", "3 passed in 0.10s", "3 passed", "¿pasan?")
    buenas = tuple(cita.Afirmacion("hall9000", "pytest -q", f"linea {i} " + "x" * 200, f"linea {i}", "p")
                   for i in range(400))
    entrega = transporte.Entrega((), buenas, transporte.entregar((mala,), (captura,)).descartadas)
    informe = S.informe_c5(entrega, "thot")
    assert len(informe) > MC.TOPE_CUERPO
    assert informe.index("Descartadas por el verificador/C5") < informe.index("'linea 0'")
    cuerpo = MC.cuerpo_del_pr(informe, TURNO["mision_id"], "qwen")
    assert len(cuerpo) <= MC.TOPE_CUERPO and "[informe recortado" in cuerpo
    assert "'3 passed'" in cuerpo and "'linea 0'" in cuerpo and "'linea 399'" not in cuerpo


@pytest.mark.parametrize("filas, esperado", [
    ({}, S.ConfigCodigo(S.AUTOR_POR_OMISION, S.TOPE_BYTES_POR_OMISION, 100 * 1024 * 1024)),
    ({"ejecutor.codigo.autor": "Otro <o@x.io>", "ejecutor.codigo.tope_bytes": "10",
      "ejecutor.codigo.tope_total_bytes": "20"}, S.ConfigCodigo("Otro <o@x.io>", 10, 20)),
])
def test_config_de_codigo_con_y_sin_filas(filas, esperado):
    assert S.config_codigo_desde_filas(filas) == esperado


@pytest.mark.parametrize("filas", [{"ejecutor.codigo.autor": "sin correo"}, {"ejecutor.codigo.tope_bytes": "0"},
                                   {"ejecutor.codigo.tope_bytes": "x"}, {"ejecutor.codigo.tope_bytes": "-5"},
                                   {"ejecutor.codigo.tope_total_bytes": "0"}, {"ejecutor.codigo.tope_total_bytes": "y"}])
def test_config_de_codigo_invalida_falla_cerrado(filas):
    with pytest.raises(ValueError):
        S.config_codigo_desde_filas(filas)


def test_el_turno_audita_con_el_plazo_de_axioma_config(monkeypatch):
    """`ejecutor.c5_tope_s` (cfg.tope_s) llega a auditor_cliente.auditar desde el turno del cerebro."""
    import jacobs.store as jstore
    from jax.ejecutor.contratos import auditor_cliente, eleccion_c5

    class _Conexion:
        async def __aenter__(self):
            return object()

        async def __aexit__(self, *exc):
            return False

    async def leer_config(conn):
        return eleccion_c5.ConfigC5("x", "y", "z", 5, 1.0, 100, 555, False, False)

    async def elegir(conn, *, cfg, hosts_mision, resolve_facet):
        return ("faceta-fake", None, None)

    vistas = {}

    async def auditar_falso(lote, **kw):
        vistas.update(kw)
        return "revision"

    monkeypatch.setattr(jstore, "conexion", lambda **kw: _Conexion())
    monkeypatch.setattr(eleccion_c5, "leer_config", leer_config)
    monkeypatch.setattr(eleccion_c5, "elegir_y_resolver_auditor", elegir)
    monkeypatch.setattr(auditor_cliente, "auditar", auditar_falso)
    monkeypatch.setattr(S.A, "afirmaciones_auditables", lambda entrega: ())
    turno = M.Turno(**{**TURNO, "hosts": frozenset(TURNO["hosts"])})
    deps = S.dependencias_reales({}, turno, tope_s=1.0, espera_s=1.0)
    assert asyncio.run(deps.auditar("texto", object(), (S.A.Maquina("m", "192.0.2.9", 58291),))) == "revision"
    assert vistas == {"faceta": "faceta-fake", "max_tokens": 100, "tope_s": 555}
