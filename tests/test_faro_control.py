"""El Faro, paso 0.3c (R3): el canal de control autenticado. Quien puede crear (o cerrar) una `Ejecucion`.

Las cuatro pruebas del plan:
 (a) un proceso con el uid de la jaula no puede crear nada por el socket de control (se cierra sin leerle);
 (b) ni un motor con el token de su ejecucion puede crear, alterar ni cerrar una `Ejecucion` ni elegir tenant;
 (c) un pedido con campos de identidad de mas se ignora;
 (d) cada creacion y cada rechazo queda en la bitacora durable y se avisa.

Como las pruebas corren con UN solo usuario, el «par de otro uid» sale de dos formas: configurando el uid del
orquestador distinto del propio (el proceso de la prueba pasa a ser un extraño, con credenciales REALES del
kernel) o inyectando las credenciales del par (`leer_credenciales`). La prueba con uids del sistema distintos
(sudo) esta en `test_faro_jaula_real.py`.
"""
from __future__ import annotations

import asyncio
import json
import os
import stat
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from mcp.shared.exceptions import MCPError

from jax.faro.aviso import Avisador, ConfigAviso, Credenciales
from jax.faro.bitacora import Bitacora
from jax.faro.bitacora_db import EmisorTabla, verificar_cadena
from jax.faro.config import ConfigFaroInvalida, ConfigPuerto
from jax.faro.control import ConfigControl, ServidorControl
from jax.faro.servicio import arrancar
from jax.faro.transporte import ServidorPuerto
from tests._faro_falsos import FalsoPool
from tests._faro_utils import cliente_por_rele, corre, ejecucion, paquete_listo, puerto
from tests.test_faro_cadena import _fabrica, entorno  # noqa: F401 (fixture y fabrica de pool)

UID = os.getuid()
JAULA_MIN, JAULA_MAX = 50000, 50050
PEDIDO = {"op": "crear", "usuario": "u-real", "tenant": "t-real", "faceta": "hyde", "motor": "codex",
          "pipeline": "p-real", "entry_point": "repl", "uid_jaula": 50001}


def _cfg_control(tmp_path, **kw) -> ConfigControl:
    d = tmp_path / "control"
    if not d.exists():
        d.mkdir(mode=0o750)
    base = dict(control_dir=d, orquestador_uid=UID, jaula_uid_min=JAULA_MIN, jaula_uid_max=JAULA_MAX,
                plazo_s=2.0, solo_pruebas_mismo_uid=True)
    base.update(kw)
    return ConfigControl(**base)


@pytest.fixture
def mundo(tmp_path):
    _cfg_faro, cargado = paquete_listo(tmp_path)
    d = tmp_path / "run"
    d.mkdir(mode=0o700)
    m = type("Mundo", (), {})()
    m.tmp, m.cargado, m.cfg_puerto, m.registros = tmp_path, cargado, ConfigPuerto(socket_dir=d), []
    return m


@asynccontextmanager
async def control(mundo, *, cfg=None, bitacora=None, **kw):
    bit = bitacora if bitacora is not None else Bitacora(emisores=[mundo.registros.append])
    cfg = cfg or _cfg_control(mundo.tmp)

    def crear_puerto(ej):
        return ServidorPuerto(mundo.cfg_puerto, ej, mundo.cargado, bit)
    async with ServidorControl(cfg, crear_puerto, bit, **kw) as srv:
        srv.calls = crear_puerto
        yield srv


async def pedir(ruta, pedido=None, *, crudo=None, plazo=5.0):
    """Manda una linea y devuelve la respuesta decodificada (None si la conexion se cerro sin responder)."""
    lector, escritor = await asyncio.open_unix_connection(str(ruta))
    try:
        escritor.write(crudo if crudo is not None else json.dumps(pedido).encode() + b"\n")
        await escritor.drain()
        linea = await asyncio.wait_for(lector.readline(), plazo)
        return json.loads(linea) if linea else None
    except (ConnectionResetError, BrokenPipeError):
        return None         # el servidor cerro sin leer lo que se le mando (RST): para un extraño es lo esperado
    finally:
        escritor.close()


def _eventos(registros, evento):
    return [r for r in registros if r.get("evento") == evento]


# --------------------------------------------------------------------------- #
# configuracion: falla cerrado                                                #
# --------------------------------------------------------------------------- #

ENV = {"JAX_FARO_CONTROL_DIR": "/run/faro-control", "JAX_FARO_ORQUESTADOR_UID": "990",
       "JAX_FARO_JAULA_UID_MIN": "50000", "JAX_FARO_JAULA_UID_MAX": "60000"}


@pytest.mark.parametrize("falta", sorted(ENV))
def test_sin_cada_variable_del_canal_de_control_el_servicio_no_arranca(falta):
    e = {k: v for k, v in ENV.items() if k != falta}
    with pytest.raises(ConfigFaroInvalida, match=falta):
        ConfigControl.desde_entorno(e)


def test_la_configuracion_sale_del_entorno():
    cfg = ConfigControl.desde_entorno({**ENV, "JAX_FARO_CONTROL_PLAZO_S": "3", "JAX_FARO_CONTROL_MAX_PEDIDO": "2048"})
    assert (cfg.control_dir, cfg.orquestador_uid, cfg.jaula_uid_min, cfg.jaula_uid_max, cfg.plazo_s, cfg.max_pedido) == (
        Path("/run/faro-control"), 990, 50000, 60000, 3.0, 2048)
    assert cfg.ruta_socket == Path("/run/faro-control/control.sock")


@pytest.mark.parametrize("cambio", [
    {"JAX_FARO_CONTROL_DIR": "relativa"},
    {"JAX_FARO_ORQUESTADOR_UID": "0"},                         # nunca root
    {"JAX_FARO_ORQUESTADOR_UID": str(UID)},                    # nunca el usuario del servicio
    {"JAX_FARO_ORQUESTADOR_UID": "x"}, {"JAX_FARO_ORQUESTADOR_UID": "-5"},
    {"JAX_FARO_JAULA_UID_MIN": "0"},                           # el rango no incluye root
    {"JAX_FARO_JAULA_UID_MIN": str(UID), "JAX_FARO_JAULA_UID_MAX": str(UID + 5)},     # ni el servicio
    {"JAX_FARO_JAULA_UID_MIN": "985", "JAX_FARO_JAULA_UID_MAX": "995"},               # ni el orquestador (990)
    {"JAX_FARO_JAULA_UID_MIN": "60001"},                       # MIN > MAX
    {"JAX_FARO_CONTROL_PLAZO_S": "0"}, {"JAX_FARO_CONTROL_MAX_PEDIDO": "10"},
])
def test_una_configuracion_insegura_no_arranca(cambio):
    with pytest.raises(ConfigFaroInvalida):
        ConfigControl.desde_entorno({**ENV, **cambio})


def test_el_mismo_uid_solo_se_acepta_con_la_bandera_de_pruebas_que_no_sale_del_entorno():
    ConfigControl.desde_entorno({**ENV, "JAX_FARO_ORQUESTADOR_UID": str(UID)}, solo_pruebas_mismo_uid=True)
    with pytest.raises(ConfigFaroInvalida):
        ConfigControl.desde_entorno({**ENV, "JAX_FARO_ORQUESTADOR_UID": str(UID), "JAX_FARO_SOLO_PRUEBAS_MISMO_UID": "1"})


# --------------------------------------------------------------------------- #
# el socket de control                                                        #
# --------------------------------------------------------------------------- #

def test_el_socket_de_control_es_0660_y_su_directorio_no_admite_a_otros(mundo):
    async def caso():
        async with control(mundo) as srv:
            return stat.S_IMODE(os.stat(srv.ruta_socket).st_mode)
    assert corre(caso()) == 0o660


@pytest.mark.parametrize("modo", [0o755, 0o751, 0o705, 0o770, 0o775, 0o777])
def test_un_directorio_de_control_con_acceso_para_otros_o_escritura_de_grupo_no_se_acepta(mundo, modo):
    d = mundo.tmp / "control"
    d.mkdir()
    d.chmod(modo)

    async def caso():
        async with control(mundo):
            pass
    with pytest.raises(ConfigFaroInvalida, match="grupo|otros"):
        corre(caso())


@pytest.mark.parametrize("modo", [0o700, 0o710, 0o750])
def test_los_directorios_de_control_sin_acceso_ajeno_se_aceptan(mundo, modo):
    d = mundo.tmp / "control"
    d.mkdir()
    d.chmod(modo)

    async def caso():
        async with control(mundo):
            pass
    corre(caso())


def test_un_enlace_o_un_archivo_en_lugar_del_directorio_de_control_no_se_acepta(mundo):
    real = mundo.tmp / "real"
    real.mkdir(mode=0o750)
    (mundo.tmp / "control").symlink_to(real)

    async def caso():
        async with control(mundo):
            pass
    with pytest.raises(ConfigFaroInvalida, match="no es un directorio"):
        corre(caso())


def test_un_socket_de_control_previo_impide_arrancar_y_al_salir_no_queda_nada(mundo):
    async def caso():
        async with control(mundo) as srv:
            ruta = srv.ruta_socket
            r = await pedir(ruta, PEDIDO)
            assert r["ok"]
            run_id = r["run_id"]
            sock = mundo.cfg_puerto.socket_dir / f"{run_id}.sock"
            assert sock.exists()
            with pytest.raises(FileExistsError):
                async with control(mundo):
                    pass
        return ruta, sock
    ruta, sock = corre(caso())
    assert not ruta.exists() and not sock.exists()             # al salir se cierran tambien las ejecuciones vivas


# --------------------------------------------------------------------------- #
# (a) y (b): solo el orquestador                                              #
# --------------------------------------------------------------------------- #

def test_a_un_proceso_que_no_es_el_orquestador_se_le_cierra_sin_leerle_ni_responderle(mundo):
    cfg = _cfg_control(mundo.tmp, orquestador_uid=UID + 1, solo_pruebas_mismo_uid=False)

    async def caso():
        async with control(mundo, cfg=cfg) as srv:
            r = await pedir(srv.ruta_socket, PEDIDO)            # credenciales REALES del kernel: uid UID != UID+1
            return r, srv.ejecuciones
    r, ejecuciones = corre(caso())
    assert r is None and ejecuciones == {}
    rechazos = _eventos(mundo.registros, "control_rechazado")
    assert len(rechazos) == 1 and rechazos[0]["motivo"] == "uid_no_autorizado" and rechazos[0]["peer_uid"] == UID
    assert not _eventos(mundo.registros, "control_creado")
    assert not list(mundo.cfg_puerto.socket_dir.iterdir())     # no nacio ni un socket ni un token


@pytest.mark.parametrize("uid_del_par", [0, 50001, 12345, UID + 7])
def test_ni_root_ni_el_uid_de_una_jaula_ni_ningun_otro_pasa_por_orquestador(mundo, uid_del_par):
    cfg = _cfg_control(mundo.tmp, orquestador_uid=990, solo_pruebas_mismo_uid=False)

    async def caso():
        async with control(mundo, cfg=cfg, leer_credenciales=lambda w: (4242, uid_del_par, 4242)) as srv:
            return await pedir(srv.ruta_socket, PEDIDO), srv.ejecuciones
    r, ejecuciones = corre(caso())
    assert r is None and ejecuciones == {}
    assert [x["peer_uid"] for x in _eventos(mundo.registros, "control_rechazado")] == [uid_del_par]


def test_un_par_sin_credenciales_se_rechaza(mundo):
    async def caso():
        async with control(mundo, leer_credenciales=lambda w: None) as srv:
            return await pedir(srv.ruta_socket, PEDIDO)
    assert corre(caso()) is None
    r = _eventos(mundo.registros, "control_rechazado")
    assert [x["motivo"] for x in r] == ["par_sin_credenciales"] and r[0]["peer_uid"] == -1


def test_un_motor_con_el_token_de_su_ejecucion_no_crea_altera_ni_cierra_nada(mundo):
    """El motor corre con el uid de su jaula (50001) y conoce el token de SU ejecucion; el control no mira tokens."""
    async def caso():
        creadas = []
        async with control(mundo) as srv:                                          # el orquestador crea
            r = await pedir(srv.ruta_socket, PEDIDO)
            run_id = r["run_id"]
            token = (mundo.cfg_puerto.socket_dir / f"{run_id}.token").read_text().strip()
            srv._credenciales = lambda w: (777, 50001, 777)                       # ahora habla «el motor»
            intentos = [
                {**PEDIDO, "uid_jaula": 50002},                                    # crear otra ejecucion
                {**PEDIDO, "uid_jaula": 50002, "tenant": "OTRO-TENANT"},           # elegir tenant
                {"op": "cerrar", "run_id": run_id},                                # cerrar la suya
                {"op": "cerrar", "run_id": run_id, "token": token},                # ... mostrando el token
                {"op": "crear", "token": token, "run_id": run_id, "tenant": "x"},  # alterar la suya
            ]
            respuestas = [await pedir(srv.ruta_socket, p) for p in intentos]
            creadas = dict(srv.ejecuciones)
            existe = (mundo.cfg_puerto.socket_dir / f"{run_id}.sock").exists()
        return respuestas, creadas, existe, run_id
    respuestas, creadas, existe, run_id = corre(caso())
    assert respuestas == [None] * 5
    assert list(creadas) == [run_id] and existe                                   # sigue viva y sin tocar
    assert creadas[run_id].ejecucion.tenant == "t-real"
    assert [r["motivo"] for r in _eventos(mundo.registros, "control_rechazado")] == ["uid_no_autorizado"] * 5


def test_ninguna_herramienta_del_puerto_crea_altera_ni_cierra_una_ejecucion(mundo):
    async def caso():
        async with puerto(mundo.cfg_puerto, mundo.cargado) as srv, cliente_por_rele(srv) as c:
            return {t.name for t in (await c.list_tools()).tools}, {r.name for r in (await c.list_resources()).resources}
    herramientas, recursos = corre(caso())
    assert herramientas == {"skills.buscar", "skills.leer", "agentes.listar"}
    assert not any("ejecucion" in n or "control" in n or "tenant" in n for n in herramientas | recursos)


def test_el_motor_no_puede_elegir_tenant_ni_usuario_por_los_argumentos_de_una_herramienta(mundo):
    async def caso():
        async with puerto(mundo.cfg_puerto, mundo.cargado, ej=ejecucion(tenant="t-real", usuario="u-real")) as srv, cliente_por_rele(srv) as c:
            for extra in ({"tenant": "OTRO"}, {"usuario": "OTRO"}, {"_meta": {"tenant": "OTRO"}}):
                try:
                    await c.call_tool("skills.leer", {"nombre": "alfa", **extra})
                except MCPError:
                    pass                                                           # rechazarlo tambien vale
        return srv.registros
    llamadas = [r for r in corre(caso()) if r.get("metodo") == "tools/call"]
    assert llamadas and all(r["tenant"] == "t-real" and r["usuario"] == "u-real" for r in llamadas)


# --------------------------------------------------------------------------- #
# (c): el pedido                                                              #
# --------------------------------------------------------------------------- #

def test_c_los_campos_de_identidad_de_mas_se_ignoran(mundo):
    pedido = {**PEDIDO, "run_id": "evil", "id_correlacion": "evil", "uid_esperado": 0, "uid": 0, "tenant_efectivo": "x",
              "admin": True, "token": "t", "socket": "/etc/passwd", "es_root": True}

    async def caso():
        async with control(mundo) as srv:
            r = await pedir(srv.ruta_socket, pedido)
            return r, srv.ejecuciones
    r, ejecuciones = corre(caso())
    assert r["ok"] and r["run_id"] != "evil" and r["run_id"].startswith("r-") and r["id_correlacion"] != "evil"
    e = ejecuciones[r["run_id"]].ejecucion
    assert (e.usuario, e.tenant, e.faceta, e.motor, e.pipeline, e.entry_point) == (
        "u-real", "t-real", "hyde", "codex", "p-real", "repl")
    assert e.uid_esperado == 50001 == r["uid_jaula"] and e.run_id == r["run_id"] and e.id_correlacion == r["id_correlacion"]
    assert e.id_correlacion != "evil"


def test_la_ejecucion_creada_tiene_su_socket_y_su_token_y_un_par_de_otro_uid_no_entra(mundo):
    async def caso():
        async with control(mundo) as srv:
            r = await pedir(srv.ruta_socket, PEDIDO)
            puerto_ = srv.ejecuciones[r["run_id"]]
            lector, escritor = await asyncio.open_unix_connection(str(puerto_.ruta_socket))
            cerrada = await asyncio.wait_for(lector.read(), 5) == b""              # el proceso de la prueba no es la jaula (uid 50001)
            escritor.close()
            return puerto_, cerrada
    p, cerrada = corre(caso())
    assert cerrada and any(r.get("motivo") == "uid_distinto_del_esperado" for r in _eventos(mundo.registros, "conexion_rechazada"))


@pytest.mark.parametrize("campo", ["usuario", "tenant", "faceta", "motor", "pipeline", "entry_point"])
@pytest.mark.parametrize("valor", ["__falta__", "", "   ", 5, None, ["x"], "x" * 201, "con\nsalto", "con\x00nul"])
def test_un_pedido_sin_un_campo_de_identidad_valido_se_rechaza(mundo, campo, valor):
    pedido = {k: v for k, v in PEDIDO.items() if k != campo}
    if valor != "__falta__":
        pedido[campo] = valor

    async def caso():
        async with control(mundo) as srv:
            return await pedir(srv.ruta_socket, pedido), srv.ejecuciones
    r, ejecuciones = corre(caso())
    assert r == {"ok": False, "error": f"campo_invalido:{campo}"} and ejecuciones == {}
    assert _eventos(mundo.registros, "control_rechazado")[0]["motivo"] == f"campo_invalido:{campo}"


@pytest.mark.parametrize("uid", ["__falta__", "50001", 50001.0, True, None, -1, 0, 49999, 50051, UID, 990])
def test_un_uid_de_jaula_fuera_de_rango_o_de_tipo_raro_se_rechaza(mundo, uid):
    pedido = {k: v for k, v in PEDIDO.items() if k != "uid_jaula"}
    if uid != "__falta__":
        pedido["uid_jaula"] = uid

    async def caso():
        async with control(mundo, cfg=_cfg_control(mundo.tmp)) as srv:
            return await pedir(srv.ruta_socket, pedido), srv.ejecuciones
    r, ejecuciones = corre(caso())
    assert r == {"ok": False, "error": "uid_jaula_invalido"} and ejecuciones == {}


def test_dos_ejecuciones_vivas_no_comparten_uid_de_jaula_y_al_cerrar_se_libera(mundo):
    async def caso():
        async with control(mundo) as srv:
            a = await pedir(srv.ruta_socket, PEDIDO)
            b = await pedir(srv.ruta_socket, PEDIDO)
            await pedir(srv.ruta_socket, {"op": "cerrar", "run_id": a["run_id"]})
            c = await pedir(srv.ruta_socket, PEDIDO)
            return a, b, c
    a, b, c = corre(caso())
    assert a["ok"] and b == {"ok": False, "error": "uid_jaula_en_uso"} and c["ok"] and c["run_id"] != a["run_id"]


def test_diez_pedidos_a_la_vez_con_el_mismo_uid_crean_exactamente_uno(mundo):
    async def caso():
        async with control(mundo) as srv:
            rs = await asyncio.gather(*(pedir(srv.ruta_socket, PEDIDO) for _ in range(10)))
            return rs, srv.ejecuciones
    rs, ejecuciones = corre(caso())
    assert sum(r["ok"] for r in rs) == 1 and len(ejecuciones) == 1


def test_d4_sesenta_ejecuciones_a_la_vez_no_chocan_con_ningun_tope_de_agentes(mundo):
    pedidos = [{**PEDIDO, "uid_jaula": 50000 + i, "tenant": f"t{i}"} for i in range(51)]

    async def caso():
        async with control(mundo, cfg=_cfg_control(mundo.tmp, plazo_s=20.0)) as srv:
            rs = await asyncio.gather(*(pedir(srv.ruta_socket, p, plazo=30) for p in pedidos))
            return rs, srv.ejecuciones
    rs, ejecuciones = corre(caso())
    assert all(r["ok"] for r in rs) and len(ejecuciones) == 51
    assert len({e.ejecucion.run_id for e in ejecuciones.values()}) == 51


@pytest.mark.parametrize("crudo,motivo", [
    (b"esto no es json\n", "pedido_invalido"),
    (b"[1, 2, 3]\n", "pedido_invalido"),
    (b"\"texto\"\n", "pedido_invalido"),
    (b"\n", "pedido_invalido"),
    (b"{}\n", "operacion_desconocida"),
    (b'{"op": "borrar"}\n', "operacion_desconocida"),
    (b'{"op": ["crear"]}\n', "operacion_desconocida"),
])
def test_un_pedido_mal_formado_o_con_una_operacion_desconocida_se_rechaza(mundo, crudo, motivo):
    async def caso():
        async with control(mundo) as srv:
            return await pedir(srv.ruta_socket, crudo=crudo)
    assert corre(caso()) == {"ok": False, "error": motivo}


def test_un_pedido_demasiado_largo_se_rechaza(mundo):
    async def caso():
        async with control(mundo, cfg=_cfg_control(mundo.tmp, max_pedido=2048)) as srv:
            return await pedir(srv.ruta_socket, crudo=b'{"op":"crear","usuario":"' + b"x" * 5000 + b'"}\n')
    assert corre(caso()) == {"ok": False, "error": "pedido_demasiado_largo"}


def test_un_pedido_que_no_termina_vence_por_el_plazo(mundo):
    async def caso():
        async with control(mundo, cfg=_cfg_control(mundo.tmp, plazo_s=0.3)) as srv:
            return await pedir(srv.ruta_socket, crudo=b'{"op":"crear"')                  # sin salto de linea
    assert corre(caso()) == {"ok": False, "error": "pedido_ilegible"}


def test_cerrar_cierra_la_ejecucion_libera_sus_archivos_y_se_anota(mundo):
    async def caso():
        async with control(mundo) as srv:
            r = await pedir(srv.ruta_socket, PEDIDO)
            run_id = r["run_id"]
            antes = sorted(p.name for p in mundo.cfg_puerto.socket_dir.iterdir())
            c = await pedir(srv.ruta_socket, {"op": "cerrar", "run_id": run_id})
            otra_vez = await pedir(srv.ruta_socket, {"op": "cerrar", "run_id": run_id})
            return run_id, antes, c, otra_vez, sorted(p.name for p in mundo.cfg_puerto.socket_dir.iterdir()), srv.ejecuciones
    run_id, antes, c, otra_vez, despues, ejecuciones = corre(caso())
    assert antes == [f"{run_id}.sock", f"{run_id}.token"] and despues == [] and ejecuciones == {}
    assert c == {"ok": True, "run_id": run_id} and otra_vez == {"ok": False, "error": "run_id_desconocido"}
    assert [r["run_id"] for r in _eventos(mundo.registros, "control_cerrado")] == [run_id]


@pytest.mark.parametrize("run_id", ["no-existe", "", None, 5, ["x"], "../etc/passwd"])
def test_cerrar_una_ejecucion_desconocida_se_rechaza(mundo, run_id):
    async def caso():
        async with control(mundo) as srv:
            return await pedir(srv.ruta_socket, {"op": "cerrar", "run_id": run_id})
    assert corre(caso()) == {"ok": False, "error": "run_id_desconocido"}


# --------------------------------------------------------------------------- #
# (d): bitacora durable y aviso                                               #
# --------------------------------------------------------------------------- #

def test_d_cada_creacion_y_cada_rechazo_queda_en_la_bitacora_durable_y_la_cadena_verifica(mundo):
    pool = FalsoPool()

    async def caso():
        bit = Bitacora(emisores=[EmisorTabla(pool)])
        async with control(mundo, bitacora=bit) as srv:
            ok = await pedir(srv.ruta_socket, PEDIDO)
            mal = await pedir(srv.ruta_socket, {**PEDIDO, "uid_jaula": 1})
            srv._credenciales = lambda w: (1, 50001, 1)
            intruso = await pedir(srv.ruta_socket, PEDIDO)
        return ok, mal, intruso
    ok, mal, intruso = corre(caso())
    assert ok["ok"] and not mal["ok"] and intruso is None
    eventos = [f["evento"] for f in pool.filas]
    assert eventos[0] == "inicio_cadena" and eventos[1:4] == ["control_creado", "control_rechazado", "control_rechazado"]
    creado = json.loads(pool.filas[1]["registro"])
    assert creado["run_id"] == ok["run_id"] and creado["tenant"] == "t-real" and creado["usuario"] == "u-real"
    assert creado["motor"] == "codex" and creado["entry_point"] == "repl" and creado["uid_jaula"] == 50001
    assert [json.loads(f["registro"])["motivo"] for f in pool.filas[2:4]] == ["uid_jaula_invalido", "uid_no_autorizado"]
    assert verificar_cadena(pool.filas) == []


def test_d_la_creacion_y_los_rechazos_se_avisan(mundo, tmp_path):
    ruta = tmp_path / "t.env"
    ruta.write_text("TELEGRAM_BOT_TOKEN=x\nTELEGRAM_CHAT_ID=1\n")
    ruta.chmod(0o600)
    enviados = []

    async def caso():
        cfg = ConfigAviso(creds=ruta, intervalo_s=0.05, rafaga=10)
        async with Avisador(cfg, Credenciales("x", "1"), enviar=enviados.append, host="h") as av:
            bit = Bitacora(emisores=[], observadores=[av])
            async with control(mundo, bitacora=bit) as srv:
                await pedir(srv.ruta_socket, PEDIDO)
                await pedir(srv.ruta_socket, {**PEDIDO, "uid_jaula": 1})
                srv._credenciales = lambda w: (1, 50001, 1)
                await pedir(srv.ruta_socket, PEDIDO)
            await asyncio.sleep(0.2)
    corre(caso())
    assert any("EJECUCION CREADA" in t and "u-real" in t and "codex" in t for t in enviados)
    assert any("RECHAZADO" in t and "uid_jaula_invalido" in t for t in enviados)
    assert any("RECHAZADO" in t and "uid_no_autorizado" in t for t in enviados)


def test_d_si_la_bitacora_no_puede_anotar_la_creacion_esta_se_deshace(mundo):
    falla = [True]

    def emisor(registro):
        if falla[0] and registro.get("evento") == "control_creado":
            raise OSError("bitacora caida")
        mundo.registros.append(registro)

    async def caso():
        bit = Bitacora(emisores=[emisor])
        async with control(mundo, bitacora=bit) as srv:
            mal = await pedir(srv.ruta_socket, PEDIDO)
            quedo = (srv.ejecuciones, sorted(p.name for p in mundo.cfg_puerto.socket_dir.iterdir()))
            falla[0] = False
            bien = await pedir(srv.ruta_socket, PEDIDO)           # el mismo uid de jaula vuelve a estar libre
        return mal, quedo, bien
    mal, quedo, bien = corre(caso())
    assert mal == {"ok": False, "error": "no_se_pudo_crear_o_anotar"} and quedo == ({}, []) and bien["ok"]


def test_d_un_rechazo_se_mantiene_aunque_la_bitacora_falle(mundo):
    def rota(registro):
        raise OSError("sin bitacora")

    async def caso():
        async with control(mundo, bitacora=Bitacora(emisores=[rota])) as srv:
            srv._credenciales = lambda w: (1, 50001, 1)
            r = await pedir(srv.ruta_socket, PEDIDO)
            return r, srv.ejecuciones
    r, ejecuciones = corre(caso())
    assert r is None and ejecuciones == {}


def test_un_puerto_que_no_se_puede_crear_rechaza_y_no_deja_reservado_el_uid(mundo):
    estado = {"romper": True}

    async def caso():
        bit = Bitacora(emisores=[mundo.registros.append])

        def crear_puerto(ej):
            if estado["romper"]:
                raise ConfigFaroInvalida("el directorio de sockets no sirve")
            return ServidorPuerto(mundo.cfg_puerto, ej, mundo.cargado, bit)
        async with ServidorControl(_cfg_control(mundo.tmp), crear_puerto, bit) as srv:
            mal = await pedir(srv.ruta_socket, PEDIDO)
            estado["romper"] = False
            bien = await pedir(srv.ruta_socket, PEDIDO)
        return mal, bien
    mal, bien = corre(caso())
    assert mal == {"ok": False, "error": "ejecucion_invalida"} and bien["ok"]


# --------------------------------------------------------------------------- #
# el servicio                                                                 #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("falta", ["JAX_FARO_AVISO_CREDS", "JAX_FARO_CONTROL_DIR", "JAX_FARO_ORQUESTADOR_UID",
                                   "JAX_FARO_JAULA_UID_MIN", "JAX_FARO_JAULA_UID_MAX"])
def test_sin_aviso_o_sin_canal_de_control_el_servicio_no_arranca_ni_toca_la_base(entorno, falta):
    entorno.pop(falta)
    crear = _fabrica()
    with pytest.raises(ConfigFaroInvalida, match=falta):
        corre(arrancar(entorno, crear_pool=crear, solo_pruebas_mismo_uid=True))
    assert crear.llamadas == []


def test_con_credenciales_de_aviso_inservibles_el_servicio_no_arranca(entorno, tmp_path):
    ruta = Path(entorno["JAX_FARO_AVISO_CREDS"])
    ruta.write_text("TELEGRAM_BOT_TOKEN=solo-el-token\n")
    crear = _fabrica()
    with pytest.raises(ConfigFaroInvalida, match="TELEGRAM"):
        corre(arrancar(entorno, crear_pool=crear, solo_pruebas_mismo_uid=True))
    assert crear.llamadas == []


def test_el_servicio_arranca_con_el_aviso_como_observador_y_el_control_sirve_ejecuciones(entorno):
    async def caso():
        pool = FalsoPool()
        s = await arrancar(entorno, crear_pool=_fabrica(pool), solo_pruebas_mismo_uid=True)
        try:
            assert isinstance(s.avisador, Avisador) and s.avisador in s.bitacora.observadores
            async with s.control() as srv:
                r = await pedir(srv.ruta_socket, PEDIDO)
                assert r["ok"] and srv.ejecuciones[r["run_id"]].presupuesto is s.presupuesto
                await pedir(srv.ruta_socket, {"op": "cerrar", "run_id": r["run_id"]})
        finally:
            await s.cerrar()
        assert pool.cerrado and s.avisador._tarea is None
        return pool.filas
    filas = corre(caso())
    assert [f["evento"] for f in filas if f["evento"].startswith("control_")] == ["control_creado", "control_cerrado"]
    assert verificar_cadena(filas) == []


def test_si_la_sonda_falla_tampoco_queda_el_avisador_corriendo(entorno):
    pool = FalsoPool("fallar")
    with pytest.raises(OSError):
        corre(arrancar(entorno, crear_pool=_fabrica(pool), solo_pruebas_mismo_uid=True))
    assert pool.cerrado
