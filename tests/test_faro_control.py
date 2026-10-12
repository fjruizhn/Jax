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
import contextlib
import json
import os
import socket
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
from jax.faro.scope import ConfigScope, EstadoUnidad, ScopeError, ScopeManager
from jax.faro.servicio import arrancar
from jax.faro.transporte import ServidorPuerto
from tests._faro_falsos import FalsoPool, FalsoTelegram
from tests._faro_utils import cliente_por_rele, corre, ejecucion, paquete_listo, puerto
from tests.test_faro_cadena import _fabrica, entorno  # noqa: F401 (fixture y fabrica de pool)

UID = os.getuid()
JAULA_MIN, JAULA_MAX = 50000, 50050
PEDIDO = {"op": "crear", "usuario": "u-real", "tenant": "t-real", "faceta": "hyde", "motor": "codex",
          "pipeline": "p-real", "entry_point": "repl", "uid_jaula": 50001}


class FalsoScope:
    """El doble EXPLICITO del gestor de unidades (0.6/R-2): sin el no hay canal de control, y un uid sin
    estado concluyente (UNKNOWN) retiene. `estados` mapea uid -> EstadoUnidad; lo que no esta, EMPTY."""

    def __init__(self, estados: dict[int, EstadoUnidad] | None = None):
        self.estados = dict(estados or {})
        self.consultas: list[int] = []
        self.muertes: list[int] = []
        self.esperas: list[int] = []
        self.error: Exception | None = None          # si se pone, toda consulta revienta
        self.error_cierre: Exception | None = None
        self.error_recuperacion: Exception | None = None
        self.al_matar = None                          # callable(uid): espiar el mundo en el momento del kill
        self.admision_cerrada = False
        self.uid_bloqueados: list[int] = []
        self.cierres_uid: list[int] = []
        self._propietario_instancia = None

    async def adquirir_instancia(self):
        if self._propietario_instancia is not None:
            raise ScopeError("instancia única ya adquirida")
        self._propietario_instancia = object()
        return self._propietario_instancia

    async def liberar_instancia(self, propietario):
        if propietario is not self._propietario_instancia:
            raise ScopeError("token de instancia ajeno")
        self._propietario_instancia = None

    def cerrar_admision(self) -> None:
        self.admision_cerrada = True

    def bloquear_uid(self, uid: int) -> None:
        self.uid_bloqueados.append(uid)

    async def cerrar_uid(self, uid: int) -> None:
        self.cierres_uid.append(uid)
        self.esperas.append(uid)                       # close incluye el wait causal del cliente/unidad
        if self.error_cierre is not None:
            raise self.error_cierre
        self.muertes.append(uid)
        if self.al_matar is not None:
            self.al_matar(uid)
        self.estados[uid] = EstadoUnidad.EMPTY

    async def recuperar_leases(self) -> None:
        if self.error_recuperacion is not None:
            raise self.error_recuperacion
        self.recuperacion_terminada = True

    async def liberar(self, uid: int) -> None:
        return None

    async def estado(self, uid: int) -> EstadoUnidad:
        self.consultas.append(uid)
        if self.error is not None:
            raise self.error
        return self.estados.get(uid, EstadoUnidad.EMPTY)

    async def matar(self, uid: int) -> None:
        self.muertes.append(uid)
        if self.al_matar is not None:
            self.al_matar(uid)

    async def esperar_vacio(self, uid: int, *, plazo_s: float | None = None) -> EstadoUnidad:
        self.esperas.append(uid)
        if self.error is not None:
            raise self.error
        return self.estados.get(uid, EstadoUnidad.EMPTY)

    async def esperar_limpiezas(self) -> None:
        return None


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
async def control(mundo, *, cfg=None, bitacora=None, scope=None, **kw):
    bit = bitacora if bitacora is not None else Bitacora(emisores=[mundo.registros.append])
    cfg = cfg or _cfg_control(mundo.tmp)

    def crear_puerto(ej):
        return ServidorPuerto(mundo.cfg_puerto, ej, mundo.cargado, bit)
    async with ServidorControl(cfg, crear_puerto, bit, scope if scope is not None else FalsoScope(), **kw) as srv:
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


def test_el_recovery_fallido_impide_abrir_el_socket_de_control(mundo):
    scope = FalsoScope()
    scope.error_recuperacion = RuntimeError("lease corrupto")

    async def caso():
        async with control(mundo, scope=scope):
            raise AssertionError("no se debe abrir el canal")

    with pytest.raises(RuntimeError, match="lease corrupto"):
        corre(caso())
    assert not (_cfg_control(mundo.tmp).ruta_socket).exists()


def test_scope_con_contrato_incompleto_falla_antes_de_crear_socket(mundo):
    bit = Bitacora(emisores=[mundo.registros.append])
    cfg = _cfg_control(mundo.tmp)
    with pytest.raises(ConfigFaroInvalida, match="contrato.*scope"):
        ServidorControl(cfg, lambda _ej: None, bit, object())
    assert not cfg.ruta_socket.exists()


def test_control_duplicado_con_mismo_scope_no_suelta_el_lock_del_servidor_activo(mundo):
    leases = mundo.tmp / "leases-control"
    leases.mkdir(mode=0o700)
    cfg_scope = ConfigScope(memory_max="256M", cpu_quota="200%", tasks_max="128", state_dir=leases)
    scope = ScopeManager(cfg_scope)
    bit = Bitacora(emisores=[mundo.registros.append])
    cfg = _cfg_control(mundo.tmp)
    primero = ServidorControl(cfg, lambda _ej: None, bit, scope)
    segundo = ServidorControl(cfg, lambda _ej: None, bit, scope)

    async def caso():
        await primero.__aenter__()
        with pytest.raises(ScopeError, match="instancia única"):
            await segundo.__aenter__()
        tercero = ScopeManager(cfg_scope)
        with pytest.raises(ScopeError, match="instancia única"):
            await tercero.adquirir_instancia()
        assert primero._servidor is not None and primero._servidor.is_serving()
        await primero.__aexit__(None, None, None)

    corre(caso())


def test_fallo_chmod_cierra_servidor_retirando_solo_su_socket_y_suelta_su_lock(mundo, monkeypatch):
    leases = mundo.tmp / "leases-chmod"
    leases.mkdir(mode=0o700)
    cfg_scope = ConfigScope(memory_max="256M", cpu_quota="200%", tasks_max="128", state_dir=leases)
    scope = ScopeManager(cfg_scope)
    cfg = _cfg_control(mundo.tmp)
    bit = Bitacora(emisores=[mundo.registros.append])
    srv = ServidorControl(cfg, lambda _ej: None, bit, scope)
    chmod_real = os.chmod

    def chmod_falla(path, modo):
        if Path(path) == cfg.ruta_socket:
            raise OSError("chmod de socket simulado")
        return chmod_real(path, modo)

    monkeypatch.setattr("jax.faro.control.os.chmod", chmod_falla)

    async def caso():
        with pytest.raises(OSError, match="chmod de socket simulado"):
            await srv.__aenter__()
        assert srv._servidor is None
        assert not cfg.ruta_socket.exists()
        tercero = ScopeManager(cfg_scope)
        token = await tercero.adquirir_instancia()
        await tercero.liberar_instancia(token)

    corre(caso())


def test_rollback_chmod_no_desvincula_socket_ajeno_que_reemplazo_el_propio(mundo, monkeypatch):
    cfg = _cfg_control(mundo.tmp)
    bit = Bitacora(emisores=[mundo.registros.append])
    srv = ServidorControl(cfg, lambda _ej: None, bit, FalsoScope())
    chmod_real = os.chmod
    socket_ajeno = []

    def reemplazar_y_fallar(path, modo):
        if Path(path) == cfg.ruta_socket:
            cfg.ruta_socket.unlink()
            nuevo = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            nuevo.bind(str(cfg.ruta_socket))
            socket_ajeno.append(nuevo)
            raise OSError("chmod de socket simulado")
        return chmod_real(path, modo)

    monkeypatch.setattr("jax.faro.control.os.chmod", reemplazar_y_fallar)

    async def caso():
        with pytest.raises(OSError, match="chmod de socket simulado"):
            await srv.__aenter__()
        assert cfg.ruta_socket.exists()
        assert srv._servidor is None

    try:
        corre(caso())
    finally:
        for sock in socket_ajeno:
            sock.close()
        cfg.ruta_socket.unlink(missing_ok=True)


def test_inicio_socket_no_admite_pedido_antes_de_completar_start_serving(mundo, monkeypatch):
    leases = mundo.tmp / "leases-start-serving"
    leases.mkdir(mode=0o700)
    cfg_scope = ConfigScope(memory_max="256M", cpu_quota="200%", tasks_max="128", state_dir=leases)
    scope = ScopeManager(cfg_scope)
    consultas = []
    estado_original = scope.estado

    async def estado_contado(uid):
        consultas.append(uid)
        return await estado_original(uid)

    scope.estado = estado_contado
    cfg = _cfg_control(mundo.tmp)
    bit = Bitacora(emisores=[mundo.registros.append])
    servidores = []
    creados = []
    listo = asyncio.Event()
    liberar_start = asyncio.Event()
    start_original = asyncio.start_unix_server

    class PuertoFalso:
        def __init__(self, ejecucion):
            self.ejecucion = ejecucion

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return None

    def crear_puerto(ejecucion):
        creado = PuertoFalso(ejecucion)
        creados.append(creado)
        return creado

    srv = ServidorControl(cfg, crear_puerto, bit, scope)

    async def start_controlado(callback, *, path, limit, backlog, start_serving=True):
        server = await start_original(callback, path=path, limit=limit, backlog=backlog,
                                      start_serving=start_serving)
        servidores.append(server)
        if not start_serving:
            iniciar_real = server.start_serving

            async def detener_retorno():
                await iniciar_real()
                listo.set()
                await liberar_start.wait()

            server.start_serving = detener_retorno
        else:
            listo.set()  # la API vieja ya está aceptando aunque __aenter__ aún no recibió el Server
            await liberar_start.wait()
        return server

    monkeypatch.setattr("jax.faro.control.asyncio.start_unix_server", start_controlado)

    async def caso():
        entrada = asyncio.create_task(srv.__aenter__())
        try:
            await asyncio.wait_for(listo.wait(), 2)
            lector, escritor = await asyncio.open_unix_connection(str(cfg.ruta_socket))
            escritor.write(json.dumps(PEDIDO).encode() + b"\n")
            await escritor.drain()
            try:
                respuesta = await asyncio.wait_for(lector.readline(), 2)
            except ConnectionResetError:
                respuesta = b""
            escritor.close()
            with contextlib.suppress(ConnectionResetError):
                await escritor.wait_closed()
            assert respuesta == b""
            assert creados == []
            assert consultas == []
            assert not _eventos(mundo.registros, "control_creado")
            entrada.cancel()
            with pytest.raises(asyncio.CancelledError):
                await entrada
            assert not cfg.ruta_socket.exists()
            assert srv._tareas == set()
            assert srv.ejecuciones == {}
            assert srv._uids == {}
            tercero = ScopeManager(cfg_scope)
            token = await tercero.adquirir_instancia()
            await tercero.liberar_instancia(token)
        finally:
            liberar_start.set()
            if not entrada.done():
                entrada.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await entrada
            for server in servidores:
                server.close()
                await server.wait_closed()
            for tarea in tuple(srv._tareas):
                tarea.cancel()
            if srv._tareas:
                await asyncio.gather(*srv._tareas, return_exceptions=True)
            if srv._ejecuciones:
                with contextlib.suppress(BaseException):
                    await srv.__aexit__(None, None, None)
            cfg.ruta_socket.unlink(missing_ok=True)

    corre(caso())


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
    assert herramientas == {"skills.buscar", "skills.leer", "agentes.listar", "memoria.buscar"}
    assert not any("ejecucion" in n or "control" in n or "tenant" in n for n in herramientas | recursos)


def test_el_motor_no_puede_elegir_tenant_ni_usuario_por_los_argumentos_de_una_herramienta(mundo):
    async def caso():
        async with puerto(mundo.cfg_puerto, mundo.cargado, ej=ejecucion(tenant="t-real", usuario="u-real")) as srv, cliente_por_rele(srv) as c:
            for extra in ({"tenant": "OTRO"}, {"usuario": "OTRO"}, {"_meta": {"tenant": "OTRO"}}):
                try:
                    await c.call_tool("skills.leer", {"nombre": "alfa", **extra})
                except MCPError:  # fail-soft: que el Puerto rechace el argumento extra tambien cumple la prueba
                    pass
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
    assert (mal["ok"], mal["error"]) == (False, "no_se_pudo_crear_o_anotar") and quedo == ({}, []) and bien["ok"]


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


def test_d_un_rechazo_a_un_orquestador_se_le_responde_aunque_la_bitacora_falle(mundo):
    def rota(registro):
        raise OSError("sin bitacora")

    async def caso():
        async with control(mundo, bitacora=Bitacora(emisores=[rota])) as srv:
            return await pedir(srv.ruta_socket, {**PEDIDO, "uid_jaula": 1}), srv.ejecuciones
    r, ejecuciones = corre(caso())
    assert r == {"ok": False, "error": "uid_jaula_invalido"} and ejecuciones == {}


def test_un_puerto_que_no_se_puede_crear_rechaza_y_no_deja_reservado_el_uid(mundo):
    estado = {"romper": True}

    async def caso():
        bit = Bitacora(emisores=[mundo.registros.append])

        def crear_puerto(ej):
            if estado["romper"]:
                raise ConfigFaroInvalida("el directorio de sockets no sirve")
            return ServidorPuerto(mundo.cfg_puerto, ej, mundo.cargado, bit)
        async with ServidorControl(_cfg_control(mundo.tmp), crear_puerto, bit, FalsoScope()) as srv:
            mal = await pedir(srv.ruta_socket, PEDIDO)
            estado["romper"] = False
            bien = await pedir(srv.ruta_socket, PEDIDO)
        return mal, bien
    mal, bien = corre(caso())
    assert (mal["ok"], mal["error"]) == (False, "ejecucion_invalida") and bien["ok"]


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
            async with s.control(FalsoScope()) as srv:
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


def test_el_servicio_no_abre_b9_si_la_compuerta_de_memoria_no_esta_habilitada(entorno, monkeypatch):
    from jax.faro import servicio as servicio_mod

    def no_debe_conectar(_cfg):
        raise AssertionError("la B9 no se conecta cuando memoria está deshabilitada")

    monkeypatch.setattr(servicio_mod, "crear_pool_memoria_prueba", no_debe_conectar)
    async def caso():
        servicio = await arrancar(entorno, crear_pool=_fabrica(), solo_pruebas_mismo_uid=True)
        try:
            assert not servicio.cfg_memoria.habilitada
            assert servicio.pool_memoria is None
            assert servicio.adaptador_memoria is None
            assert servicio.crear_puerto(ejecucion())._adaptador_memoria is None
        finally:
            await servicio.cerrar()
    corre(caso())


def test_el_servicio_habilitado_pasa_al_helper_solo_el_perfil_de_prueba(entorno, monkeypatch):
    from jax.faro import servicio as servicio_mod
    from jax.faro.herramientas.memoria import AdaptadorMemoria

    entorno.update({
        "JAX_FARO_MEMORIA_HABILITADA": "true",
        "JAX_FARO_MEMORIA_TEST_DB_HOST": "127.0.0.1",
        "JAX_FARO_MEMORIA_TEST_DB_PORT": "3308",
        "JAX_FARO_MEMORIA_TEST_DB_USER": "jax_test",
        "JAX_FARO_MEMORIA_TEST_DB_NAME": "jax_memory_test",
        "JAX_FARO_MEMORIA_TEST_DB_PASSWORD": "clave-local-de-prueba",
        "JAX_FARO_MEMORIA_TIMEOUT_S": "1.25",
    })
    pool, lector, capturado = FalsoPool(), object(), []

    async def abrir(cfg):
        capturado.append(cfg)
        return pool, lector

    monkeypatch.setattr(servicio_mod, "crear_pool_memoria_prueba", abrir)

    async def caso():
        servicio = await arrancar(entorno, crear_pool=_fabrica(FalsoPool()), solo_pruebas_mismo_uid=True)
        try:
            cfg = capturado[0]
            assert (cfg.host, cfg.port, cfg.usuario, cfg.base, cfg.timeout_s) == (
                "127.0.0.1", 3308, "jax_test", "jax_memory_test", 1.25)
            assert servicio.pool_memoria is pool
            assert isinstance(servicio.adaptador_memoria, AdaptadorMemoria)
            assert servicio.adaptador_memoria._timeout_s == 1.25
        finally:
            await servicio.cerrar()

    corre(caso())


def test_el_servicio_arrancado_avisa_de_verdad_por_http_un_rechazo_del_canal_de_control(entorno):
    with FalsoTelegram() as tg:
        entorno["JAX_FARO_AVISO_API_URL"] = tg.url

        async def caso():
            s = await arrancar(entorno, crear_pool=_fabrica(FalsoPool()), solo_pruebas_mismo_uid=True)
            try:
                async with s.control(FalsoScope()) as srv:
                    r = await pedir(srv.ruta_socket, {**PEDIDO, "uid_jaula": 1})
                    assert r["error"] == "uid_jaula_invalido"
                    assert await asyncio.to_thread(tg.esperar, 1, 5.0)
            finally:
                await s.cerrar()
        corre(caso())
    assert "RECHAZADO" in tg.textos()[0] and "uid_jaula_invalido" in tg.textos()[0]
    assert tg.recibidos[0]["chat_id"] == "1" and tg.recibidos[0]["ruta"] == "/bottoken-de-prueba/sendMessage"


def test_r2_el_servicio_no_construye_el_control_sin_el_gestor_de_unidades(entorno):
    async def caso():
        s = await arrancar(entorno, crear_pool=_fabrica(), solo_pruebas_mismo_uid=True)
        try:
            with pytest.raises(ConfigFaroInvalida, match="scope|unidad"):
                s.control(None)
        finally:
            await s.cerrar()
    corre(caso())


def test_major1_el_servicio_tiene_un_solo_gestor_y_el_lanzador_lo_comparte(entorno, tmp_path):
    """MAJOR 1: no hay dos proveedores de vida por uid. `control(scope)` adopta EL gestor del servicio;
    un gestor distinto se rechaza, y el lanzador se construye solo sobre ese mismo gestor."""
    from jax.faro.jaula import ConfigJaula
    bwrap = tmp_path / "bwrap"
    bwrap.write_text("#!/bin/sh\nexit 0\n")
    bwrap.chmod(0o755)
    cfg_jaula = ConfigJaula(bwrap=bwrap, elevar=("/usr/bin/sudo", "-n", "-u", "#{uid}", "--"))

    async def caso():
        s = await arrancar(entorno, crear_pool=_fabrica(), solo_pruebas_mismo_uid=True)
        try:
            with pytest.raises(ConfigFaroInvalida, match="control|gestor"):
                s.crear_lanzador(cfg_jaula)                   # sin control(scope) previo no hay gestor
            gestor = FalsoScope()
            async with s.control(gestor) as srv:
                assert srv._scope is gestor
                lanzador = s.crear_lanzador(cfg_jaula)
                assert lanzador._scope is gestor              # EL MISMO objeto, no otro igual
            with pytest.raises(ConfigFaroInvalida, match="UNA sola|unico|dos"):
                s.control(FalsoScope())                       # un segundo gestor distinto: no entra
            s.control(gestor)                                 # el mismo gestor siempre vale
        finally:
            await s.cerrar()
    corre(caso())


def test_si_la_sonda_falla_tampoco_queda_el_avisador_corriendo(entorno):
    pool = FalsoPool("fallar")
    with pytest.raises(OSError):
        corre(arrancar(entorno, crear_pool=_fabrica(pool), solo_pruebas_mismo_uid=True))
    assert pool.cerrado


# --------------------------------------------------------------------------- #
# auditoria de 0.3bc                                                          #
# --------------------------------------------------------------------------- #
from jax.faro.control import CUENTAS_CON_CODIGO_DE_MODELOS, cuentas_prohibidas, validar_directorio_control  # noqa: E402


def _pwd(tabla):
    def getpwnam(nombre):
        if nombre not in tabla:
            raise KeyError(nombre)
        return type("P", (), {"pw_uid": tabla[nombre]})()
    return getpwnam


TABLA = {"jaxsvc": 994, "axioma": 1001, "fruiz": 1000, "orq-faro": 990, "otra": 995}


def test_major1_la_lista_por_defecto_incluye_las_cuentas_que_ejecutan_codigo_de_modelos():
    assert {"jaxsvc", "axioma", "fruiz"} <= set(CUENTAS_CON_CODIGO_DE_MODELOS)
    assert cuentas_prohibidas({}, getpwnam=_pwd(TABLA)) == {994, 1001, 1000}


@pytest.mark.parametrize("cuenta,uid", [("jaxsvc", 994), ("axioma", 1001), ("fruiz", 1000)])
def test_major1_el_orquestador_no_puede_ser_una_cuenta_que_ejecuta_codigo_de_modelos(cuenta, uid, monkeypatch):
    monkeypatch.setattr(os, "geteuid", lambda: 5)            # que la causa sea la lista y no «es el usuario del servicio»
    with pytest.raises(ConfigFaroInvalida, match=cuenta):
        ConfigControl.desde_entorno({**ENV, "JAX_FARO_ORQUESTADOR_UID": str(uid)}, getpwnam=_pwd(TABLA))


def test_major1_una_cuenta_propia_del_orquestador_si_se_acepta(monkeypatch):
    monkeypatch.setattr(os, "geteuid", lambda: 5)
    assert ConfigControl.desde_entorno({**ENV, "JAX_FARO_ORQUESTADOR_UID": "990"}, getpwnam=_pwd(TABLA)).orquestador_uid == 990


def test_major1_la_lista_se_extiende_por_configuracion_pero_no_se_puede_acortar(monkeypatch):
    monkeypatch.setattr(os, "geteuid", lambda: 5)
    e = {**ENV, "JAX_FARO_ORQUESTADOR_PROHIBIDOS": "otra, no-existe"}
    assert cuentas_prohibidas(e, getpwnam=_pwd(TABLA)) == {994, 1001, 1000, 995}           # lo que no existe en el sistema se salta
    with pytest.raises(ConfigFaroInvalida, match="otra"):
        ConfigControl.desde_entorno({**e, "JAX_FARO_ORQUESTADOR_UID": "995"}, getpwnam=_pwd(TABLA))
    sola = {**ENV, "JAX_FARO_ORQUESTADOR_PROHIBIDOS": "otra"}                                # poner solo «otra» no quita jaxsvc
    with pytest.raises(ConfigFaroInvalida, match="jaxsvc"):
        ConfigControl.desde_entorno({**sola, "JAX_FARO_ORQUESTADOR_UID": "994"}, getpwnam=_pwd(TABLA))


def test_major1_un_nombre_de_cuenta_invalido_en_la_configuracion_falla_cerrado():
    with pytest.raises(ConfigFaroInvalida):
        cuentas_prohibidas({"JAX_FARO_ORQUESTADOR_PROHIBIDOS": "ok,con espacio;rm"}, getpwnam=_pwd(TABLA))


def test_major1_el_rango_de_jaulas_tampoco_puede_incluir_esas_cuentas(monkeypatch):
    monkeypatch.setattr(os, "geteuid", lambda: 5)
    with pytest.raises(ConfigFaroInvalida, match="jaxsvc"):
        ConfigControl.desde_entorno({**ENV, "JAX_FARO_JAULA_UID_MIN": "993", "JAX_FARO_JAULA_UID_MAX": "1100"}, getpwnam=_pwd(TABLA))


def test_major1_en_el_sistema_real_las_cuentas_resueltas_son_las_de_pwd():
    import pwd
    esperadas = set()
    for n in CUENTAS_CON_CODIGO_DE_MODELOS:
        try:
            esperadas.add(pwd.getpwnam(n).pw_uid)
        except KeyError:  # fail-soft: una cuenta que no existe en esta maquina no entra en lo esperado
            pass
    assert cuentas_prohibidas({}) == esperadas


def test_minor1_el_rechazo_por_fallo_de_la_bitacora_lleva_run_id_e_id_de_correlacion(mundo):
    def emisor(registro):
        if registro.get("evento") == "control_creado":
            raise OSError("bitacora caida")
        mundo.registros.append(registro)

    async def caso():
        async with control(mundo, bitacora=Bitacora(emisores=[emisor])) as srv:
            return await pedir(srv.ruta_socket, PEDIDO)
    r = corre(caso())
    rech = _eventos(mundo.registros, "control_rechazado")
    assert r["ok"] is False and len(rech) == 1
    assert rech[0]["run_id"].startswith("r-") and len(rech[0]["id_correlacion"]) == 32 and rech[0]["uid_jaula"] == 50001
    assert r["run_id"] == rech[0]["run_id"]                                                  # y el orquestador lo sabe


def test_minor2_el_uid_no_se_libera_mientras_su_unidad_siga_viva(mundo):
    scope = FalsoScope()                                   # al crear, la unidad del uid no existe: EMPTY

    async def caso():
        async with control(mundo, scope=scope) as srv:
            a = await pedir(srv.ruta_socket, PEDIDO)
            scope.estados[50001] = EstadoUnidad.LIVE       # al cerrar, la unidad sigue poblada
            scope.error_cierre = OSError("kill no confirmó EMPTY")
            await pedir(srv.ruta_socket, {"op": "cerrar", "run_id": a["run_id"]})
            b = await pedir(srv.ruta_socket, PEDIDO)        # la unidad sigue viva: el uid sigue ocupado
            scope.error_cierre = None
            scope.estados[50001] = EstadoUnidad.EMPTY       # murio hasta el ultimo nieto
            c = await pedir(srv.ruta_socket, PEDIDO)        # confirmado EMPTY: se libera
            return a, b, c
    a, b, c = corre(caso())
    assert a["ok"] and b == {"ok": False, "error": "uid_jaula_en_uso"} and c["ok"]
    assert scope.muertes and scope.esperas                  # el cierre mato la unidad y espero el vacio


def test_r2_sin_proveedor_de_unidad_no_hay_canal_de_control(mundo):
    """R-2: el gancho ya no es opcional; ninguna ruta productiva construye el control sin el."""
    async def caso():
        with pytest.raises(ConfigFaroInvalida, match="scope|unidad"):
            ServidorControl(_cfg_control(mundo.tmp), lambda ej: None, Bitacora(emisores=[]), None)
    corre(caso())


# --------------------------------------------------------------------------- #
# R-2: el estado vive en systemd, no en la memoria del Faro                    #
# --------------------------------------------------------------------------- #

def test_r2_un_reinicio_del_faro_no_libera_un_uid_con_cgroup_superviviente(mundo):
    """El caso central de R-2: el mapa en memoria esta VACIO (el Faro reinicio) pero systemd sigue
    reportando procesos en la unidad de ese uid. El uid NO se reutiliza."""
    scope = FalsoScope({50001: EstadoUnidad.LIVE})

    async def caso():
        async with control(mundo, scope=scope) as srv:      # mapa vacio: nada reservado localmente
            a = await pedir(srv.ruta_socket, PEDIDO)
            scope.estados[50001] = EstadoUnidad.EMPTY       # el cgroup quedo vacio y systemd lo confirma
            b = await pedir(srv.ruta_socket, PEDIDO)
            return a, b, srv.ejecuciones
    a, b, ejecuciones = corre(caso())
    assert a == {"ok": False, "error": "uid_jaula_en_uso"} and b["ok"] and len(ejecuciones) == 1
    assert 50001 in scope.consultas                         # se miro systemd, no solo el mapa local


def test_r2_un_estado_desconocido_rechaza_el_uid_y_es_fallo_cerrado(mundo):
    scope = FalsoScope({50001: EstadoUnidad.UNKNOWN})

    async def caso():
        async with control(mundo, scope=scope) as srv:
            r = await pedir(srv.ruta_socket, PEDIDO)
            return r, srv.ejecuciones
    r, ejecuciones = corre(caso())
    assert r == {"ok": False, "error": "uid_jaula_desconocido"} and ejecuciones == {}
    assert _eventos(mundo.registros, "control_rechazado")[0]["motivo"] == "uid_jaula_desconocido"


def test_r2_si_la_consulta_al_scope_falla_el_crear_rechaza_sin_dejar_nada_reservado(mundo):
    scope = FalsoScope()
    scope.error = OSError("systemd no responde")

    async def caso():
        async with control(mundo, scope=scope) as srv:
            r = await pedir(srv.ruta_socket, PEDIDO)
            scope.error = None
            despues = await pedir(srv.ruta_socket, PEDIDO)  # sin la consulta rota, el mismo uid pasa
            return r, despues, srv.ejecuciones
    r, despues, _ = corre(caso())
    assert r == {"ok": False, "error": "uid_jaula_desconocido"} and despues["ok"]
    assert not any(motivo.get("motivo") == "no_se_pudo_crear_o_anotar"
                   for motivo in _eventos(mundo.registros, "control_rechazado"))


def test_r2_el_cierre_revoca_el_puerto_antes_de_matar_y_mata_solo_esa_unidad(mundo):
    scope = FalsoScope()
    vista_al_matar = {}
    scope.al_matar = lambda uid: vista_al_matar.update(
        {uid: sorted(p.name for p in mundo.cfg_puerto.socket_dir.iterdir())})

    async def caso():
        async with control(mundo, scope=scope) as srv:
            a = await pedir(srv.ruta_socket, PEDIDO)
            b = await pedir(srv.ruta_socket, {**PEDIDO, "uid_jaula": 50002})
            c = await pedir(srv.ruta_socket, {"op": "cerrar", "run_id": a["run_id"]})
            # se mide DENTRO del contexto: al salir, el apagado cierra (como debe) a la hermana tambien
            return a, b, c, dict(srv.ejecuciones), list(scope.muertes), list(scope.esperas), dict(vista_al_matar)
    a, b, c, ejecuciones, muertes, esperas, vista = corre(caso())
    assert a["ok"] and b["ok"] and c["ok"]
    # al matar, el Puerto de ESA ejecucion ya no existe (revocado ANTES del kill); la hermana, intacta
    assert f"{a['run_id']}.sock" not in vista[50001] and f"{a['run_id']}.token" not in vista[50001]
    assert f"{b['run_id']}.sock" in vista[50001]
    assert 50002 not in vista                               # a la hermana no se la toco
    assert muertes == [50001]                               # solo muere la unidad de ese uid
    assert esperas == [50001]
    assert list(ejecuciones) == [b["run_id"]]               # la hermana sigue viva


def test_r2_el_cierre_con_la_unidad_aun_viva_o_desconocida_mantiene_el_uid(mundo):
    for estado in (EstadoUnidad.LIVE, EstadoUnidad.UNKNOWN):
        scope = FalsoScope()

        async def caso():
            async with control(mundo, scope=scope) as srv:
                a = await pedir(srv.ruta_socket, PEDIDO)
                scope.estados[50001] = estado               # al cerrar, la unidad no se confirma vacia
                scope.error_cierre = OSError("no se pudo certificar EMPTY")
                await pedir(srv.ruta_socket, {"op": "cerrar", "run_id": a["run_id"]})
                return await pedir(srv.ruta_socket, PEDIDO)
        assert corre(caso()) == {"ok": False, "error": "uid_jaula_en_uso" if estado is EstadoUnidad.LIVE
                                 else "uid_jaula_desconocido"}


def test_r2_el_fallo_del_kill_no_impide_el_cierre_ni_la_retencion(mundo):
    class ScopeQueNoMata(FalsoScope):
        async def cerrar_uid(self, uid):
            self.cierres_uid.append(uid)
            self.muertes.append(uid)
            raise OSError("systemctl kill fallo")

    scope = ScopeQueNoMata()

    async def caso():
        async with control(mundo, scope=scope) as srv:
            a = await pedir(srv.ruta_socket, PEDIDO)
            scope.estados[50001] = EstadoUnidad.LIVE        # y el kill que debia vaciarla fallo
            c = await pedir(srv.ruta_socket, {"op": "cerrar", "run_id": a["run_id"]})
            b = await pedir(srv.ruta_socket, PEDIDO)
            return c, b
    c, b = corre(caso())
    assert c == {"ok": False, "error": "cierre_incompleto"}
    assert b == {"ok": False, "error": "uid_jaula_en_uso"}   # el UID queda retenido


def test_shutdown_limpia_hermanas_aunque_falle_una(mundo):
    class ScopeConUnaFalla(FalsoScope):
        async def cerrar_uid(self, uid):
            self.cierres_uid.append(uid)
            if uid == 50002:
                raise OSError("fallo deliberado de B")

    scope = ScopeConUnaFalla()
    async def caso():
        cm = control(mundo, scope=scope)
        srv = await cm.__aenter__()
        await pedir(srv.ruta_socket, PEDIDO)
        await pedir(srv.ruta_socket, {**PEDIDO, "uid_jaula": 50002, "tenant": "t-second"})
        with pytest.raises(BaseExceptionGroup, match="cierre completo"):
            await cm.__aexit__(None, None, None)
    corre(caso())
    assert sorted(scope.cierres_uid) == [50001, 50002]


def test_shutdown_observa_fallo_de_cierre_terminado_antes_del_snapshot(mundo):
    class ScopeFallaAlCerrar(FalsoScope):
        def __init__(self):
            super().__init__()
            self.entro = asyncio.Event()
            self.liberaciones_instancia = 0

        async def cerrar_uid(self, uid):
            self.cierres_uid.append(uid)
            self.entro.set()
            raise OSError("fallo antes del snapshot de shutdown")

        async def liberar_instancia(self, propietario):
            self.liberaciones_instancia += 1
            await super().liberar_instancia(propietario)

    scope = ScopeFallaAlCerrar()

    async def caso():
        cm = control(mundo, scope=scope)
        srv = await cm.__aenter__()
        creado = await pedir(srv.ruta_socket, PEDIDO)
        run_id = creado["run_id"]
        tareas_capturadas = []
        programar = srv._programar_cierre

        def capturar(uid, puerto_srv):
            tarea = programar(uid, puerto_srv)
            tareas_capturadas.append(tarea)
            return tarea

        srv._programar_cierre = capturar
        cierre_conexion = asyncio.create_task(
            srv._cerrar_ejecucion(run_id, srv._ejecuciones[run_id]))
        await asyncio.wait_for(scope.entro.wait(), 5)
        cierre_conexion.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cierre_conexion
        # La tarea propia completa con error antes de que shutdown tome su snapshot.
        tarea = tareas_capturadas[0]
        async with asyncio.timeout(5):
            while not tarea.done():
                await asyncio.sleep(0)
        try:
            with pytest.raises(BaseExceptionGroup, match="cierre completo") as error:
                await cm.__aexit__(None, None, None)
        finally:
            await asyncio.gather(tarea, return_exceptions=True)
        return srv, run_id, error.value

    srv, run_id, error = corre(caso())
    assert error.subgroup(OSError) is not None
    assert srv._retenidos == {50001}
    assert srv._uids == {50001: run_id}
    assert srv._cierres == {}
    assert scope.liberaciones_instancia == 0


def test_block2_el_apagado_que_cancela_un_cierre_a_medias_termina_el_kill(mundo):
    """El caso exacto del BLOCK 2: la peticion `cerrar` corre en una conexion; el apagado cancela esa
    conexion mientras el kill esta en vuelo. La tarea de kill+confirmacion es propia y el apagado la
    espera: la unidad muere y se confirma el vacio antes de salir."""
    class ScopeLento(FalsoScope):
        def __init__(self):
            super().__init__()
            self.entro = asyncio.Event()
            self.salir = asyncio.Event()

        async def cerrar_uid(self, uid):
            self.entro.set()
            await self.salir.wait()                # cierre lento: la cancelacion del apagado cae aqui
            await super().cerrar_uid(uid)

    scope = ScopeLento()

    async def caso():
        cm = control(mundo, scope=scope)
        srv = await cm.__aenter__()
        r = await pedir(srv.ruta_socket, PEDIDO)
        cierre = asyncio.create_task(pedir(srv.ruta_socket, {"op": "cerrar", "run_id": r["run_id"]}))
        await asyncio.wait_for(scope.entro.wait(), 5)        # el cierre ya está en vuelo en la conexión

        async def soltar():
            await asyncio.sleep(0.05)
            scope.salir.set()
        asyncio.create_task(soltar())                        # libera el cierre lento durante el apagado
        await cm.__aexit__(None, None, None)                 # apagado: cancela la conexion `cerrar`
        cierre.cancel()
        return list(scope.cierres_uid), list(scope.esperas), sorted(p.name for p in mundo.cfg_puerto.socket_dir.iterdir())
    cierres, esperas, archivos = corre(caso())
    assert cierres == [50001]                               # el cierre corrió pese a la cancelación
    assert archivos == []                                   # y el Puerto de la ejecucion quedo revocado


def test_r2_el_apagado_espera_limpiezas_de_arranque_cancelado(mundo):
    class ScopeConLimpiezaPendiente(FalsoScope):
        def __init__(self):
            super().__init__()
            self.entro = asyncio.Event()
            self.salir = asyncio.Event()

        async def esperar_limpiezas(self):
            self.entro.set()
            await self.salir.wait()

    scope = ScopeConLimpiezaPendiente()

    async def caso():
        cm = control(mundo, scope=scope)
        await cm.__aenter__()
        apagado = asyncio.create_task(cm.__aexit__(None, None, None))
        await asyncio.sleep(0.05)
        espero_limpieza = not apagado.done()
        scope.salir.set()
        await apagado
        return scope.entro.is_set(), espero_limpieza

    entro, espero_limpieza = corre(caso())
    assert entro and espero_limpieza


@pytest.mark.parametrize("tenant", ["con espacio", "a|b", "x" * 65, "ñandú", "t\nx", "a;b", "a/b"])
def test_minor5_el_control_rechaza_un_tenant_que_los_topes_no_aceptarian(mundo, tenant):
    async def caso():
        async with control(mundo) as srv:
            return await pedir(srv.ruta_socket, {**PEDIDO, "tenant": tenant}), srv.ejecuciones
    r, e = corre(caso())
    assert r == {"ok": False, "error": "campo_invalido:tenant"} and e == {}


@pytest.mark.parametrize("tenant", ["t-real", "Tenant_1", "a.b:c@d", "x" * 64])
def test_minor5_un_tenant_aceptado_por_el_control_nunca_rompe_consumir_sin_regla(mundo, tenant):
    from jax.faro.topes import Topes
    from tests._faro_falsos import AlmacenMemoria

    async def caso():
        async with control(mundo) as srv:
            r = await pedir(srv.ruta_socket, {**PEDIDO, "tenant": tenant})
            t = srv.ejecuciones[r["run_id"]].ejecucion.tenant
        return await Topes(AlmacenMemoria(), Bitacora(emisores=[])).consumir(tenant=t, recurso="tokens", cantidad=1, tope=None)
    assert corre(caso()).permitido


@pytest.mark.parametrize("modo", [0o777, 0o755, 0o770, 0o705])
def test_minor9_el_directorio_de_control_inseguro_impide_arrancar_el_servicio(entorno, modo):
    Path(entorno["JAX_FARO_CONTROL_DIR"]).chmod(modo)
    crear = _fabrica()
    with pytest.raises(ConfigFaroInvalida, match="grupo|otros"):
        corre(arrancar(entorno, crear_pool=crear, solo_pruebas_mismo_uid=True))
    assert crear.llamadas == []


def test_minor9_main_sale_con_2_si_el_directorio_de_control_es_0777(entorno, capsys):
    from jax.faro.servicio import main
    Path(entorno["JAX_FARO_CONTROL_DIR"]).chmod(0o777)
    entorno["JAX_FARO_DUENIO_UID"] = "0"                    # main no lleva la bandera de pruebas: ni dueño ni orquestador pueden ser este uid
    entorno["JAX_FARO_ORQUESTADOR_UID"] = "990"
    assert main([], env=entorno) == 2 and "otros" in capsys.readouterr().err


def test_minor9_la_validacion_se_puede_llamar_sola(tmp_path):
    d = tmp_path / "c"
    d.mkdir(mode=0o750)
    validar_directorio_control(_cfg_control(tmp_path, control_dir=d))
    d.chmod(0o777)
    with pytest.raises(ConfigFaroInvalida):
        validar_directorio_control(_cfg_control(tmp_path, control_dir=d))
