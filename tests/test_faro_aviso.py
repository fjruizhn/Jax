"""El Faro, paso 0.3b (P-3): el aviso inmediato de cada denegacion y rechazo de conexion.

El aviso va por Telegram (mismo patron que `ci-aislada-vigia`: credenciales en un archivo tipo
`/etc/restic/telegram.env` cuya ruta sale de `JAX_FARO_AVISO_CREDS`, nunca del codigo). Aqui el Telegram es
SIEMPRE un HTTP falso en 127.0.0.1 (`FalsoTelegram`); jamas el real.

Lo que se defiende: una denegacion por freno produce un aviso; un aviso que falla (emisor roto, HTTP 500,
destino inalcanzable, colgado) NO cambia la denegacion; la tasa esta limitada (una tormenta no inunda el
chat ni oculta una denegacion de otra clase); el aviso nunca bloquea el bucle de eventos; el token nunca sale
en un log; la configuracion falla cerrado.
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from pathlib import Path

import pytest
from mcp.shared.exceptions import MCPError

from jax.faro.aviso import (Avisador, ConfigAviso, Credenciales, LimiteTasa, es_avisable, leer_credenciales,
                            redactar)
from jax.faro.bitacora import Bitacora
from jax.faro.config import ConfigFaroInvalida, ConfigPuerto
from tests._faro_falsos import FalsoTelegram
from tests._faro_utils import cliente_por_rele, corre, ejecucion, paquete_listo, puerto

TOKEN = "123456:TOKEN-SECRETO-DE-PRUEBA-xyz"
CHAT = "-100777"


def _creds(tmp_path, texto=None, modo=0o600) -> Path:
    ruta = tmp_path / "telegram.env"
    ruta.write_text(texto if texto is not None else f"export TELEGRAM_BOT_TOKEN={TOKEN}\nexport TELEGRAM_CHAT_ID={CHAT}\n")
    ruta.chmod(modo)
    return ruta


def _cfg(tmp_path, url="http://127.0.0.1:9", **kw) -> ConfigAviso:
    base = dict(creds=_creds(tmp_path), api_url=url, rafaga=5, intervalo_s=0.05, timeout_s=2.0, cola=100)
    base.update(kw)
    return ConfigAviso(**base)


CRED = Credenciales(token=TOKEN, chat_id=CHAT)
DENEGACION = {"evento": "llamada", "decision": "denegado", "motivo": "freno", "run_id": "run-1", "usuario": "u-real",
              "tenant": "t-real", "motor": "codex", "entry_point": "repl", "metodo": "tools/call", "objetivo": "skills.leer",
              "argumentos": "ARGUMENTO-SECRETO-NO-VA-AL-CHAT"}
RECHAZO = {"evento": "conexion_rechazada", "motivo": "uid_distinto_del_esperado", "run_id": "run-1", "peer_uid": 4242}


# --------------------------------------------------------------------------- #
# configuracion: falla cerrado                                                #
# --------------------------------------------------------------------------- #

def test_sin_la_ruta_de_las_credenciales_no_hay_aviso_y_el_servicio_no_arranca():
    with pytest.raises(ConfigFaroInvalida, match="JAX_FARO_AVISO_CREDS"):
        ConfigAviso.desde_entorno({})


def test_la_configuracion_sale_del_entorno_con_valores_por_defecto_razonables(tmp_path):
    cfg = ConfigAviso.desde_entorno({"JAX_FARO_AVISO_CREDS": str(tmp_path / "t.env")})
    assert cfg.api_url.startswith("https://") and cfg.rafaga >= 1 and cfg.intervalo_s > 0 and cfg.cola >= 1
    cfg = ConfigAviso.desde_entorno({"JAX_FARO_AVISO_CREDS": str(tmp_path / "t.env"), "JAX_FARO_AVISO_API_URL": "http://127.0.0.1:1",
                                     "JAX_FARO_AVISO_RAFAGA": "9", "JAX_FARO_AVISO_INTERVALO_S": "2.5",
                                     "JAX_FARO_AVISO_TIMEOUT_S": "3", "JAX_FARO_AVISO_COLA": "7"})
    assert (cfg.api_url, cfg.rafaga, cfg.intervalo_s, cfg.timeout_s, cfg.cola) == ("http://127.0.0.1:1", 9, 2.5, 3.0, 7)


@pytest.mark.parametrize("env", [
    {"JAX_FARO_AVISO_CREDS": "relativa/t.env"},
    {"JAX_FARO_AVISO_API_URL": "ftp://x"},
    {"JAX_FARO_AVISO_API_URL": "file:///etc/passwd"},
    {"JAX_FARO_AVISO_RAFAGA": "0"}, {"JAX_FARO_AVISO_RAFAGA": "x"},
    {"JAX_FARO_AVISO_INTERVALO_S": "0"}, {"JAX_FARO_AVISO_INTERVALO_S": "-1"},
    {"JAX_FARO_AVISO_TIMEOUT_S": "0"}, {"JAX_FARO_AVISO_COLA": "0"},
])
def test_una_configuracion_invalida_no_arranca(tmp_path, env):
    base = {"JAX_FARO_AVISO_CREDS": str(tmp_path / "t.env")}
    base.update(env)
    with pytest.raises(ConfigFaroInvalida):
        ConfigAviso.desde_entorno(base)


@pytest.mark.parametrize("texto", [
    f"export TELEGRAM_BOT_TOKEN={TOKEN}\nexport TELEGRAM_CHAT_ID={CHAT}\n",          # formato de hall9000
    f'TELEGRAM_BOT_TOKEN="{TOKEN}"\nTELEGRAM_CHAT_ID="{CHAT}"\n',                    # formato de .11
    f'  export TELEGRAM_BOT_TOKEN="{TOKEN}"   # comentario\nTELEGRAM_CHAT_ID={CHAT}\n',
])
def test_las_credenciales_se_leen_en_los_dos_formatos_del_ecosistema(tmp_path, texto):
    c = leer_credenciales(_creds(tmp_path, texto))
    assert (c.token, c.chat_id) == (TOKEN, CHAT)


def test_el_token_no_sale_en_el_repr_de_las_credenciales():
    assert TOKEN not in repr(CRED) and TOKEN not in str(CRED)


@pytest.mark.parametrize("texto", ["", "TELEGRAM_BOT_TOKEN=x\n", "TELEGRAM_CHAT_ID=1\n", "TELEGRAM_BOT_TOKEN=\nTELEGRAM_CHAT_ID=1\n"])
def test_credenciales_incompletas_fallan_cerrado(tmp_path, texto):
    with pytest.raises(ConfigFaroInvalida):
        leer_credenciales(_creds(tmp_path, texto))


def test_credenciales_ausentes_o_con_escritura_ajena_o_enlace_fallan_cerrado(tmp_path):
    with pytest.raises(ConfigFaroInvalida):
        leer_credenciales(tmp_path / "no-existe.env")
    with pytest.raises(ConfigFaroInvalida, match="escritura"):
        leer_credenciales(_creds(tmp_path, modo=0o666))
    real = _creds(tmp_path)
    enlace = tmp_path / "enlace.env"
    enlace.symlink_to(real)
    with pytest.raises(ConfigFaroInvalida):
        leer_credenciales(enlace)


# --------------------------------------------------------------------------- #
# que se avisa y como se redacta                                              #
# --------------------------------------------------------------------------- #

def test_se_avisan_las_denegaciones_y_los_rechazos_y_nada_de_lo_corriente():
    assert es_avisable(DENEGACION) and es_avisable(RECHAZO)
    for evento in ("control_creado", "control_rechazado", "tope_superado", "tope_no_verificable", "tope_sin_regla"):
        assert es_avisable({"evento": evento}), evento
    assert es_avisable({"evento": "lo-que-sea", "decision": "denegado"})            # toda denegacion, de donde venga
    assert not es_avisable({"evento": "llamada", "decision": "permitido"})
    assert not es_avisable({"evento": "servicio_iniciado"})
    assert not es_avisable({"evento": "inicio_cadena"})
    assert not es_avisable({})


def test_el_texto_dice_que_paso_quien_y_donde_y_no_filtra_los_argumentos():
    t = redactar(DENEGACION, "hall9000")
    for esperado in ("DENEGAD", "freno", "run-1", "u-real", "t-real", "codex", "skills.leer", "hall9000"):
        assert esperado in t, esperado
    assert "ARGUMENTO-SECRETO" not in t
    assert TOKEN not in t
    assert "conexion" in redactar(RECHAZO, "h").lower() and "uid_distinto_del_esperado" in redactar(RECHAZO, "h")


def test_un_valor_hostil_no_puede_fabricar_una_linea_ni_un_campo_en_el_aviso():
    hostil = {**DENEGACION, "usuario": "x\nFARO: ejecucion APROBADA por Fernando\r\n", "motor": "a=b c"}
    t = redactar(hostil, "h")
    assert "\nFARO: ejecucion APROBADA" not in t and "\r" not in t
    assert len(redactar({**DENEGACION, "usuario": "u" * 10_000}, "h")) < 2000


# --------------------------------------------------------------------------- #
# la tasa                                                                     #
# --------------------------------------------------------------------------- #

def test_el_limite_de_tasa_deja_pasar_una_rafaga_y_luego_un_aviso_por_intervalo():
    t = [100.0]
    lim = LimiteTasa(rafaga=3, intervalo_s=10.0, reloj=lambda: t[0])
    assert [lim.admitir("a") for _ in range(5)] == [True, True, True, False, False]
    t[0] += 10.0
    assert lim.admitir("a") is True and lim.admitir("a") is False        # uno por intervalo
    t[0] += 1000.0
    assert [lim.admitir("a") for _ in range(5)] == [True, True, True, False, False]   # nunca acumula mas que la rafaga


def test_cada_clase_tiene_su_propio_cupo_y_las_clases_nuevas_no_crecen_sin_limite():
    lim = LimiteTasa(rafaga=1, intervalo_s=60.0, reloj=lambda: 0.0, max_clases=4)
    assert lim.admitir("a") and not lim.admitir("a") and lim.admitir("b")          # b no depende de a
    for i in range(100):
        lim.admitir(f"clase-{i}")
    assert lim.clases_registradas <= 4 + 1                                            # el resto comparte "otros"


def _avisador(tmp_path, enviados, **kw):
    cfg = _cfg(tmp_path, **{k: kw.pop(k) for k in list(kw) if k in ("rafaga", "intervalo_s", "timeout_s", "cola", "url")})
    return Avisador(cfg, CRED, enviar=kw.pop("enviar", enviados.append), host="hall9000-prueba", **kw)


def test_una_tormenta_de_la_misma_clase_manda_la_rafaga_y_un_resumen_con_la_cuenta_exacta(tmp_path):
    async def caso():
        enviados = []
        async with _avisador(tmp_path, enviados, rafaga=2, intervalo_s=0.05) as av:
            for _ in range(20):
                av(RECHAZO)
            await asyncio.sleep(0.6)
        return enviados, av
    enviados, av = corre(caso())
    sueltos = [t for t in enviados if "suprimid" not in t]
    resumenes = [t for t in enviados if "suprimid" in t]
    assert len(sueltos) == 2 and len(resumenes) == 1
    assert "18" in resumenes[0] and "uid_distinto_del_esperado" in resumenes[0]
    assert av.suprimidos == 18 and av.enviados == 3


def test_una_tormenta_de_rechazos_no_oculta_una_denegacion_de_otra_clase(tmp_path):
    async def caso():
        enviados = []
        async with _avisador(tmp_path, enviados, rafaga=1, intervalo_s=60.0) as av:
            for _ in range(50):
                av(RECHAZO)
            av(DENEGACION)
            await asyncio.sleep(0.2)
        return enviados
    enviados = corre(caso())
    assert any("freno" in t for t in enviados)


# --------------------------------------------------------------------------- #
# nunca bloquea, nunca lanza, nunca cuelga a quien lo llama                   #
# --------------------------------------------------------------------------- #

def test_avisar_no_bloquea_aunque_el_envio_sea_lentisimo(tmp_path):
    soltar = threading.Event()

    def lento(texto):
        soltar.wait(5)

    async def caso():
        async with _avisador(tmp_path, [], enviar=lento, rafaga=100, timeout_s=10) as av:
            t0 = time.monotonic()
            bit = Bitacora(emisores=[], observadores=[av])
            for i in range(5):
                await bit.registrar("conexion_rechazada", motivo=f"m{i}", run_id="r")
            dt = time.monotonic() - t0
            soltar.set()
            return dt
    assert corre(caso()) < 0.2


def test_con_la_cola_llena_se_descarta_y_se_cuenta_sin_bloquear(tmp_path):
    soltar = threading.Event()
    enviados = []

    def lento(texto):
        soltar.wait(5)
        enviados.append(texto)

    async def caso():
        async with _avisador(tmp_path, [], enviar=lento, rafaga=100, cola=2, timeout_s=10) as av:
            t0 = time.monotonic()
            for i in range(10):
                av({**RECHAZO, "motivo": f"m{i}"})
            dt = time.monotonic() - t0
            soltar.set()
            await asyncio.sleep(0.3)
            return dt, av
    dt, av = corre(caso())
    assert dt < 0.1 and av.descartados >= 7


def test_un_envio_colgado_vence_por_el_plazo_y_el_siguiente_aviso_sale(tmp_path):
    llamadas = []

    def a_veces_colgado(texto):
        llamadas.append(texto)
        if len(llamadas) == 1:
            time.sleep(1.5)

    async def caso():
        async with _avisador(tmp_path, [], enviar=a_veces_colgado, rafaga=100, timeout_s=0.1) as av:
            av({**RECHAZO, "motivo": "uno"})
            await asyncio.sleep(0.4)
            av({**RECHAZO, "motivo": "dos"})
            await asyncio.sleep(0.3)
            return av
    av = corre(caso())
    assert av.fallidos == 1 and av.enviados == 1 and len(llamadas) == 2


@pytest.mark.parametrize("registro", [None, {}, {"evento": None}, {"evento": 5, "decision": "denegado"}, {"evento": object()},
                                      {"evento": "conexion_rechazada", "run_id": object(), "peer_uid": [1, 2], "motivo": b"\xff\xfe"}])
def test_avisar_nunca_lanza_con_un_registro_raro(tmp_path, registro):
    async def caso():
        async with _avisador(tmp_path, []) as av:
            av(registro)
            await asyncio.sleep(0.05)
    corre(caso())


def test_si_el_avisador_falla_por_dentro_no_lanza_y_sigue_funcionando(tmp_path, monkeypatch):
    import jax.faro.aviso as modulo
    original = modulo.redactar
    fallos = []

    def redactar_roto(registro, host):
        if not fallos:
            fallos.append(1)
            raise RuntimeError("fallo interno del avisador")
        return original(registro, host)
    monkeypatch.setattr(modulo, "redactar", redactar_roto)

    async def caso():
        enviados = []
        async with _avisador(tmp_path, enviados, rafaga=10) as av:
            av(RECHAZO)                 # el primero falla por dentro: NO lanza
            av(DENEGACION)              # el siguiente sale
            await asyncio.sleep(0.2)
        return enviados
    enviados = corre(caso())
    assert fallos == [1] and len(enviados) == 1 and "freno" in enviados[0]


def test_al_cerrar_se_entrega_lo_pendiente(tmp_path):
    async def caso():
        enviados = []
        av = _avisador(tmp_path, enviados, rafaga=10)
        await av.iniciar()
        for i in range(3):
            av({**RECHAZO, "motivo": f"m{i}"})
        await av.cerrar()
        return enviados
    assert len(corre(caso())) == 3


# --------------------------------------------------------------------------- #
# HTTP real contra un Telegram falso                                          #
# --------------------------------------------------------------------------- #

def test_el_aviso_llega_al_endpoint_de_telegram_con_el_chat_y_el_texto(tmp_path):
    with FalsoTelegram() as tg:
        async def caso():
            async with Avisador(_cfg(tmp_path, tg.url), CRED, host="hall9000-prueba") as av:
                av(DENEGACION)
                assert await asyncio.to_thread(tg.esperar, 1)
        corre(caso())
        r = tg.recibidos[0]
        assert r["ruta"] == f"/bot{TOKEN}/sendMessage" and r["chat_id"] == CHAT
        assert "freno" in r["text"] and "ARGUMENTO-SECRETO" not in r["text"]


@pytest.mark.parametrize("modo", ["http500", "inalcanzable", "retraso"])
def test_un_destino_roto_se_cuenta_como_fallido_y_el_token_nunca_sale_en_un_log(tmp_path, caplog, modo):
    caplog.set_level(logging.DEBUG)
    with FalsoTelegram(estado=500 if modo == "http500" else 200, retraso_s=2.0 if modo == "retraso" else 0.0) as tg:
        url = "http://127.0.0.1:9" if modo == "inalcanzable" else tg.url

        async def caso():
            async with Avisador(_cfg(tmp_path, url, timeout_s=0.3), CRED, host="h") as av:
                av(DENEGACION)
                await asyncio.sleep(1.0)
                return av
        av = corre(caso())
    assert av.fallidos == 1 and av.enviados == 0
    assert TOKEN not in caplog.text and TOKEN.split(":")[1] not in caplog.text


# --------------------------------------------------------------------------- #
# extremo a extremo con el Puerto real                                        #
# --------------------------------------------------------------------------- #

@pytest.fixture
def mundo(tmp_path):
    _cfg_faro, cargado = paquete_listo(tmp_path)
    d = tmp_path / "run"
    d.mkdir(mode=0o700)
    return ConfigPuerto(socket_dir=d), cargado


def _bitacora_con_aviso(av, registros):
    return Bitacora(emisores=[registros.append], observadores=[av])


def test_una_denegacion_por_freno_produce_un_aviso_inmediato(tmp_path, mundo):
    cfg_puerto, cargado = mundo
    with FalsoTelegram() as tg:
        async def caso():
            registros = []
            freno = [False]
            async with Avisador(_cfg(tmp_path, tg.url), CRED, host="hall9000-prueba") as av:
                bit = _bitacora_con_aviso(av, registros)
                async with puerto(cfg_puerto, cargado, bitacora=bit, freno=lambda: freno[0]) as srv, cliente_por_rele(srv) as c:
                    freno[0] = True
                    t0 = time.monotonic()
                    with pytest.raises(MCPError) as exc:
                        await c.call_tool("skills.leer", {"nombre": "alfa"})
                    assert exc.value.code == 423
                    assert await asyncio.to_thread(tg.esperar, 1, 3.0)
                    return time.monotonic() - t0, registros
        dt, registros = corre(caso())
    assert "freno" in tg.textos()[0] and "skills.leer" in tg.textos()[0] and "run-1" in tg.textos()[0]
    assert dt < 3.0
    assert any(r.get("decision") == "denegado" for r in registros)


@pytest.mark.parametrize("modo", ["emisor_roto", "http500", "inalcanzable"])
def test_un_aviso_que_falla_no_cambia_la_denegacion(tmp_path, mundo, modo):
    cfg_puerto, cargado = mundo

    def roto(texto):
        raise OSError("canal caido")

    with FalsoTelegram(estado=500) as tg:
        async def caso():
            registros = []
            freno = [False]
            url = "http://127.0.0.1:9" if modo == "inalcanzable" else tg.url
            enviar = roto if modo == "emisor_roto" else None
            async with Avisador(_cfg(tmp_path, url, timeout_s=0.3), CRED, enviar=enviar, host="h") as av:
                bit = _bitacora_con_aviso(av, registros)
                async with puerto(cfg_puerto, cargado, bitacora=bit, freno=lambda: freno[0]) as srv, cliente_por_rele(srv) as c:
                    freno[0] = True
                    for _ in range(3):
                        with pytest.raises(MCPError) as exc:
                            await c.call_tool("skills.leer", {"nombre": "alfa"})
                        assert exc.value.code == 423
                await asyncio.sleep(0.5)
                return av, registros
        av, registros = corre(caso())
    assert av.fallidos >= 1 and av.enviados == 0
    assert [r["decision"] for r in registros if r.get("metodo") == "tools/call"] == ["denegado"] * 3


def test_aunque_la_bitacora_durable_falle_la_denegacion_se_avisa(tmp_path, mundo):
    cfg_puerto, cargado = mundo

    def sin_bitacora(registro):
        if registro.get("metodo") == "tools/call":      # el handshake pasa; la llamada denegada no se puede anotar
            raise OSError("tabla caida")

    enviados = []

    async def caso():
        freno = [False]
        async with Avisador(_cfg(tmp_path), CRED, enviar=enviados.append, host="h") as av:
            bit = Bitacora(emisores=[sin_bitacora], observadores=[av])
            async with puerto(cfg_puerto, cargado, bitacora=bit, freno=lambda: freno[0]) as srv, cliente_por_rele(srv) as c:
                freno[0] = True
                with pytest.raises(MCPError) as exc:
                    await c.call_tool("skills.leer", {"nombre": "alfa"})
                assert exc.value.code == 423
            await asyncio.sleep(0.2)
    corre(caso())
    assert any("freno" in t for t in enviados)


def test_un_rechazo_de_conexion_se_avisa(tmp_path, mundo):
    cfg_puerto, cargado = mundo
    enviados = []

    async def caso():
        async with Avisador(_cfg(tmp_path), CRED, enviar=enviados.append, host="h") as av:
            bit = Bitacora(emisores=[], observadores=[av])
            async with puerto(cfg_puerto, cargado, ej=ejecucion(uid_esperado=os.getuid() + 1), bitacora=bit) as srv:
                lector, escritor = await asyncio.open_unix_connection(str(srv.ruta_socket))
                assert await asyncio.wait_for(lector.read(), 5) == b""
                escritor.close()
            await asyncio.sleep(0.2)
    corre(caso())
    assert len(enviados) == 1 and "uid_distinto_del_esperado" in enviados[0]


# --------------------------------------------------------------------------- #
# los observadores de la bitacora                                             #
# --------------------------------------------------------------------------- #

def test_los_observadores_corren_antes_que_los_emisores_y_aunque_un_emisor_falle():
    orden = []

    def emisor_roto(registro):
        orden.append("emisor")
        raise OSError("sin bitacora")

    bit = Bitacora(emisores=[emisor_roto], observadores=[lambda r: orden.append("observador")])
    with pytest.raises(OSError):
        corre(bit.registrar("x"))
    assert orden == ["observador", "emisor"]


def test_un_observador_que_lanza_no_rompe_la_bitacora_ni_a_los_demas():
    vistos = []

    def malo(registro):
        raise RuntimeError("observador roto")

    bit = Bitacora(emisores=[vistos.append], observadores=[malo, lambda r: vistos.append("otro")])
    r = corre(bit.registrar("evento", a=1))
    assert r["evento"] == "evento" and "otro" in vistos and any(isinstance(v, dict) for v in vistos)


# --------------------------------------------------------------------------- #
# auditoria de 0.3bc                                                          #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("url", ["http://127.0.0.1:9", "http://localhost:80", "http://[::1]:9", "http://127.5.5.5", "https://api.telegram.org",
                                 "https://proxy.interno:8443"])
def test_minor10_http_solo_hacia_loopback_y_https_hacia_cualquiera(tmp_path, url):
    assert ConfigAviso(creds=tmp_path / "t.env", api_url=url).api_url == url


@pytest.mark.parametrize("url", ["http://example.com", "http://api.telegram.org", "http://10.0.0.5:9", "http://localhost.evil.com",
                                 "http://127.0.0.1.evil.com", "http://0.0.0.0:9", "http://[::]:9", "http://", "https://", "http:///x"])
def test_minor10_http_hacia_un_destino_no_loopback_no_se_acepta(tmp_path, url):
    with pytest.raises(ConfigFaroInvalida):
        ConfigAviso(creds=tmp_path / "t.env", api_url=url)


def test_minor3_un_aviso_fallido_se_mide_por_clase_y_run_id_y_se_registra_sin_secretos(tmp_path, caplog):
    caplog.set_level(logging.WARNING)

    def roto(texto):
        raise OSError("canal caido")

    async def caso():
        async with _avisador(tmp_path, [], enviar=roto, rafaga=10) as av:
            av(DENEGACION)
            av(RECHAZO)
            av({**RECHAZO, "run_id": "run-2"})
            await asyncio.sleep(0.3)
            return av
    av = corre(caso())
    assert av.fallidos == 3
    assert av.fallidos_por_clase == {"llamada|freno": 1, "conexion_rechazada|uid_distinto_del_esperado": 2}
    assert av.ultimo_fallo["clase"] == "conexion_rechazada|uid_distinto_del_esperado" and av.ultimo_fallo["run_id"] == "run-2"
    assert av.ultimo_fallo["error"] == "OSError"
    log = caplog.text
    assert "llamada|freno" in log and "run-1" in log and "run-2" in log and TOKEN not in log


def test_minor3_un_resumen_fallido_se_cuenta_como_clase_resumen(tmp_path):
    def roto(texto):
        raise OSError("x")

    async def caso():
        async with _avisador(tmp_path, [], enviar=roto, rafaga=1, intervalo_s=0.05) as av:
            for _ in range(5):
                av(RECHAZO)
            await asyncio.sleep(0.5)
            return av
    av = corre(caso())
    assert av.fallidos_por_clase.get("resumen", 0) >= 1



# --------------------------------------------------------------------------- #
# F1.1 paso 8 r2: el aviso de la decision del kernel de reglas (§11, #371)    #
#                                                                             #
# El aviso NO es parte de la autoridad: el kernel decide y persiste, y ESTO   #
# corre despues. Contrato contra el RuleDecision REAL de #371 (instancias     #
# reales, no un doble). Todo fail-soft y sin red: el envio va con un doble.   #
# --------------------------------------------------------------------------- #

import ast
import builtins
import collections
import io
import json as _json
import pathlib
import pkgutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone

import policy.rule_authority as _rule_authority
from policy.rule_authority.models import RuleDecision, RuleDecisionStatus

from jax.faro.aviso import (_MAX_TEXTO, AvisoRegla, acumular_para_resumen, aviso_de_decision,
                            confirmar_resumen, emitir_aviso_inmediato, resumen_diario)

HASH_OK = "sha256:" + "a1b2c3d4e5f67890deadbeeffeedface00112233445566778899aabbccddeeff"
RAIZ = pathlib.Path(__file__).resolve().parents[1]
def _resumir(cola, host="hall9000", **kw):
    """Las dos fases juntas (el envio salio bien): la LISTA de mensajes, o None."""
    r = resumen_diario(cola, host=host, **kw)
    if r is None:
        return None
    assert confirmar_resumen(r) is True
    return list(r.mensajes)


_RAZON_DE = {RuleDecisionStatus.DENY: "RULE_EXPIRED", RuleDecisionStatus.MISSING_RULE: "RULE_NOT_FOUND",
             RuleDecisionStatus.PERMIT: None}


def _decision(status=RuleDecisionStatus.DENY, regla="RL-para-gastar-dinero", cuando=None, **extra) -> RuleDecision:
    """Un RuleDecision REAL de #371 (pasa su validacion). `extra` cuelga atributos
    AJENOS al contrato (subject, arguments...) para probar que no viajan."""
    d = RuleDecision(request_id="018f3b2a-9c1d-7abc-9ef0-1234567890ab", request_hash=HASH_OK, status=status,
                     required_rule_id=regla, reason_code=_RAZON_DE[status],
                     decided_at_utc=cuando or datetime(2026, 10, 7, 3, 4, 5, tzinfo=timezone.utc))
    for k, v in extra.items():
        object.__setattr__(d, k, v)
    return d


def _forzar(decision: RuleDecision, **campos) -> RuleDecision:
    """Salta la validacion de #371 para fabricar la decision CORRUPTA que el aviso tiene que aguantar."""
    for k, v in campos.items():
        object.__setattr__(decision, k, v)
    return decision


# Lo que NO puede salir (§11): subject y argumentos de la decision.
SECRETOS_DE_LA_DECISION = {
    "subject": "usuario-con-correo@secreto.hn",
    "arguments": {"clave": "sk-SECRETO-DE-PRUEBA-no-sale", "ruta": "/etc/secreto/llaves.env"},
    "amount": 987654,
    "contenido": "MENSAJE-SECRETO-DEL-USUARIO",
    "archivo": "/tmp/plan-secreto.md",
}
SECRETOS_PLANOS = ["usuario-con-correo@secreto.hn", "sk-SECRETO-DE-PRUEBA-no-sale", "/etc/secreto/llaves.env",
                   "987654", "MENSAJE-SECRETO-DEL-USUARIO", "/tmp/plan-secreto.md"]


def test_los_tres_status_reales_de_371_se_avisan_al_instante_con_su_titulo():
    for status, titulo in ((RuleDecisionStatus.DENY, "REGLA DENEGADA"),
                           (RuleDecisionStatus.MISSING_RULE, "SIN REGLA QUE CUBRA EL ACTO"),
                           (RuleDecisionStatus.PERMIT, "PERMISO (OBLIGA O NO SE PUDO DESCARTAR QUE OBLIGUE)")):
        av = aviso_de_decision(_decision(status), host="hall9000")
        assert av.inmediato is True, status
        assert titulo in av.texto, status


def test_un_permiso_se_difiere_solo_si_el_llamador_dice_que_no_obliga():
    """Mutante d: un PERMIT NO obligante no puede salir como inmediato. Si hay duda (obliga ausente
    o que no es un bool), obliga."""
    permit = _decision(RuleDecisionStatus.PERMIT)
    assert aviso_de_decision(permit, host="h", obliga=False).inmediato is False
    assert aviso_de_decision(permit, host="h", obliga=True).inmediato is True
    assert aviso_de_decision(permit, host="h").inmediato is True
    assert aviso_de_decision(permit, host="h", obliga="no").inmediato is True
    assert aviso_de_decision(permit, host="h", obliga=0).inmediato is True
    # una negativa nunca se difiere, diga lo que diga `obliga`
    assert aviso_de_decision(_decision(RuleDecisionStatus.DENY), host="h", obliga=False).inmediato is True
    assert aviso_de_decision(_decision(RuleDecisionStatus.MISSING_RULE), host="h", obliga=False).inmediato is True


def test_el_texto_lleva_regla_razon_hash_de_12_hex_y_la_hora_utc_de_la_decision():
    cuando = datetime(2026, 10, 7, 3, 4, 5, tzinfo=timezone.utc)
    av = aviso_de_decision(_decision(cuando=cuando), host="hall9000")
    lineas = av.texto.splitlines()
    assert len(lineas) == 2                                    # ni una linea fabricada
    assert "regla=RL-para-gastar-dinero" in lineas[1]
    assert "razon=RULE_EXPIRED" in lineas[1]
    assert "req=sha256:a1b2c3d4e5f6 " in lineas[1]             # sha256: + EXACTAMENTE 12 hex (no 5)
    assert f"a={cuando.isoformat(timespec='seconds')}" in lineas[1]
    assert av.creado_utc == "2026-10-07T03:04:05+00:00"
    assert av.clase == "DENY|RULE_EXPIRED|RL-para-gastar-dinero"   # la clase de tasa incluye la REGLA


@pytest.mark.parametrize("campo", ["subject", "arguments", "amount", "contenido", "archivo"])
def test_el_texto_no_filtra_subject_ni_argumentos_ni_montos_ni_contenido(campo):
    """Mutantes b (arguments) y g (subject): cada secreto, colgado de una decision VALIDA, uno por uno."""
    for status in RuleDecisionStatus:
        av = aviso_de_decision(_decision(status, **{campo: SECRETOS_DE_LA_DECISION[campo]}), host="hall9000")
        for secreto in SECRETOS_PLANOS:
            assert secreto not in av.texto, f"{status}: el aviso filtro {secreto!r} ({campo})"
        assert av.inmediato is True


def test_una_decision_no_reconocible_es_aviso_inmediato_y_no_filtra_nada():
    """BLOCK-2: ni None ni raise. Aviso INMEDIATO que nombra la CLASE y nada mas: el repr de un objeto
    cualquiera puede traer secretos."""
    import types
    raras = [
        types.SimpleNamespace(**SECRETOS_DE_LA_DECISION),                       # sin status: no es el contrato
        types.SimpleNamespace(decision="DENY", rule_id="RL-1", obliga=False, **SECRETOS_DE_LA_DECISION),
        None, "DENY", 7, {"status": "DENY", **SECRETOS_DE_LA_DECISION},
        _forzar(_decision(**SECRETOS_DE_LA_DECISION), status="TAL_VEZ"),        # status que no es del enum
        _forzar(_decision(**SECRETOS_DE_LA_DECISION), request_hash=12345),      # MINOR 7: tipo raro
        _forzar(_decision(**SECRETOS_DE_LA_DECISION), request_hash="sha256:abc"),
        _forzar(_decision(**SECRETOS_DE_LA_DECISION), required_rule_id=None),
        _forzar(_decision(**SECRETOS_DE_LA_DECISION), decided_at_utc="ayer"),
    ]
    for rara in raras:
        av = aviso_de_decision(rara, host="hall9000", obliga=False)    # obliga=False tampoco lo difiere
        assert av is not None and av.inmediato is True, rara
        assert "DECISION NO RECONOCIBLE" in av.texto, rara
        assert type(rara).__qualname__ in av.texto
        assert av.creado_utc.endswith("+00:00")
        for secreto in SECRETOS_PLANOS:
            assert secreto not in av.texto, f"el aviso de lo no reconocible filtro {secreto!r}"


def test_una_decision_que_lanza_al_leerse_tambien_cierra_cerrado():
    class _Rota(RuleDecision):
        @property
        def status(self):
            raise TypeError("campo ilegible")
    rota = object.__new__(_Rota)
    av = aviso_de_decision(rota, host="hall9000")
    assert av.inmediato is True and "DECISION NO RECONOCIBLE" in av.texto and "_Rota" in av.texto


def test_un_required_rule_id_hostil_no_fabrica_lineas_ni_campos():
    av = aviso_de_decision(_forzar(_decision(), required_rule_id="RL-1\ninyectada=SI\rSECRETO"), host="hall9000")
    assert len(av.texto.splitlines()) == 2
    assert "\ninyectada" not in av.texto and "inyectada=SI" not in av.texto


def test_las_horas_son_utc_aunque_la_maquina_este_en_otra_zona(monkeypatch, tmp_path):
    """Mutante l: la hora local desplaza el offset. Se fija una zona (UTC-6) y se mira CADA hora que
    el modulo escribe: la de la decision, la del no reconocible y la del resumen."""
    import time as t
    monkeypatch.setenv("TZ", "XXX6")
    t.tzset()
    try:
        otra_zona = timezone(timedelta(hours=5))
        av = aviso_de_decision(_decision(cuando=datetime(2026, 10, 7, 8, 4, 5, tzinfo=otra_zona)), host="h")
        assert av.creado_utc == "2026-10-07T03:04:05+00:00" and "a=2026-10-07T03:04:05+00:00" in av.texto
        def es_ahora_en_utc(iso: str) -> None:       # no basta el sufijo: la hora misma tiene que ser UTC
            assert iso.endswith("+00:00"), iso
            assert abs((datetime.fromisoformat(iso) - datetime.now(timezone.utc)).total_seconds()) < 120, iso

        nr = aviso_de_decision(None, host="h")
        es_ahora_en_utc(nr.creado_utc)
        assert "a=" + nr.creado_utc in nr.texto
        cola = tmp_path / "cola.jsonl"
        assert acumular_para_resumen(av, cola) is True
        cabecera = _resumir(cola, host="h")[0].splitlines()[0]
        es_ahora_en_utc(cabecera.rsplit(" · ", 1)[1])
    finally:
        monkeypatch.undo()
        t.tzset()


def test_una_hora_ingenua_no_se_acepta_como_utc():
    """Mutante l (datetime.now() sin zona): la hora local ingenua no se convierte en silencio."""
    from jax.faro.aviso import _ahora_utc, _iso_utc
    assert _ahora_utc().utcoffset() == timedelta(0)
    with pytest.raises(ValueError):
        _iso_utc(datetime(2026, 10, 7, 3, 4, 5))


def test_emitir_exige_limite_y_ruta_de_cola_sin_defaults(tmp_path):
    """MINOR 12: un limite que no se pasa no limita nada y una cola que no se pasa pierde lo suprimido."""
    av = aviso_de_decision(_decision(), host="hall9000")
    cfg = _cfg(tmp_path)
    with pytest.raises(TypeError):
        emitir_aviso_inmediato(av, cfg, CRED, enviar=lambda *a: None, ruta_cola=tmp_path / "cola.jsonl")
    with pytest.raises(TypeError):
        emitir_aviso_inmediato(av, cfg, CRED, enviar=lambda *a: None, limite=LimiteTasa(5, 10.0))


def test_emitir_envia_el_texto_y_devuelve_true(tmp_path):
    av = aviso_de_decision(_decision(), host="hall9000")
    mandados = []
    cola = tmp_path / "cola.jsonl"
    ok = emitir_aviso_inmediato(av, _cfg(tmp_path), CRED, enviar=lambda cfg, cred, texto: mandados.append(texto),
                                limite=LimiteTasa(5, 10.0), ruta_cola=cola)
    assert ok is True and mandados == [av.texto]
    assert not cola.exists()            # nada suprimido: la cola ni se crea


def test_emitir_es_fail_soft_cuando_el_envio_explota(tmp_path, caplog):
    """Mutante c: un envio que falla no se propaga hacia quien aplico la decision."""
    av = aviso_de_decision(_decision(), host="hall9000")

    def explota(cfg, cred, texto):
        raise RuntimeError("el chat de Telegram no existe")

    with caplog.at_level(logging.WARNING):
        ok = emitir_aviso_inmediato(av, _cfg(tmp_path), CRED, enviar=explota,
                                    limite=LimiteTasa(5, 10.0), ruta_cola=tmp_path / "cola.jsonl")
    assert ok is False
    assert "aviso de regla no entregado" in caplog.text and "RuntimeError" in caplog.text


def test_lo_suprimido_por_tasa_va_a_la_cola_con_marca_y_no_tapa_otra_regla(tmp_path):
    """MAJOR-4 / mutante j: la clase de tasa incluye la REGLA (la tormenta de RL-A no gasta el cupo de
    RL-B) y lo suprimido NO se descarta: va a la cola con suprimido_por_tasa=true."""
    cola = tmp_path / "cola.jsonl"
    reloj = [0.0]
    limite = LimiteTasa(2, 1000.0, reloj=lambda: reloj[0])
    envios = []
    cfg = _cfg(tmp_path)

    def emitir(av):
        return emitir_aviso_inmediato(av, cfg, CRED, enviar=lambda c, k, texto: envios.append(texto),
                                      limite=limite, ruta_cola=cola)

    a = aviso_de_decision(_decision(regla="RL-A"), host="hall9000")
    b = aviso_de_decision(_decision(regla="RL-B"), host="hall9000")
    assert emitir(a) is True and emitir(a) is True
    assert emitir(a) is False                    # RL-A agoto su rafaga: suprimido, NO enviado
    assert emitir(b) is True                     # el DENY de RL-B tiene su propio cupo
    assert len(envios) == 3
    lineas = cola.read_text(encoding="utf-8").splitlines()
    assert len(lineas) == 1                      # el suprimido de RL-A esta en la cola, no se perdio
    dato = _json.loads(lineas[0])
    assert dato["suprimido_por_tasa"] is True and "RL-A" in dato["clase"] and dato["texto"] == a.texto
    resumen = _resumir(cola, host="hall9000")
    assert "suprimidos_por_tasa=1" in resumen[0]  # y el resumen lo cuenta


def test_la_cola_y_el_candado_nacen_y_se_mantienen_0600(tmp_path):
    """MAJOR-3 / mutante e: con el umask MAS laxo (000) la cola y el candado salen 0600 igual, y tras
    resumir y volver a acumular (la cola NUEVA) tambien."""
    cola = tmp_path / "cola.jsonl"
    av = aviso_de_decision(_decision(RuleDecisionStatus.PERMIT), host="hall9000", obliga=False)
    old = os.umask(0)
    try:
        assert acumular_para_resumen(av, cola) is True
        assert cola.stat().st_mode & 0o777 == 0o600
        assert (tmp_path / "cola.jsonl.candado").stat().st_mode & 0o777 == 0o600
        assert _resumir(cola, host="hall9000") is not None
        assert acumular_para_resumen(av, cola) is True
        assert cola.stat().st_mode & 0o777 == 0o600
        assert (tmp_path / "cola.jsonl.candado").stat().st_mode & 0o777 == 0o600
    finally:
        os.umask(old)


def test_la_carrera_escritor_resumen_no_pierde_ni_duplica(tmp_path):
    """BLOCK-1: un escritor REAL en otro proceso mete 20 000 lineas mientras este proceso resume en bucle.
    Perdidos = 0 y duplicados = 0; y la carrera tiene que haber ocurrido de verdad (varias rotaciones)."""
    cola = tmp_path / "cola.jsonl"
    total = 20000
    escritor = tmp_path / "escritor.py"
    escritor.write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(RAIZ)!r})\n"
        "from jax.faro.aviso import AvisoRegla, acumular_para_resumen\n"
        f"cola = {str(cola)!r}\n"
        f"for i in range({total}):\n"
        "    acumular_para_resumen(AvisoRegla(texto=f'cuerpo-{i}', clase='PERMIT|ok|RL-A',"
        " inmediato=False, creado_utc='2026-10-07T03:04:05+00:00'), cola)\n", encoding="utf-8")
    proc = subprocess.Popen([sys.executable, str(escritor)], stdout=subprocess.DEVNULL)
    vistos: list[str] = []
    rotaciones = 0
    try:
        while proc.poll() is None:
            mensajes = _resumir(cola, host="hall9000")
            if mensajes:
                rotaciones += 1
                vistos.extend(mensajes)
        assert proc.wait(timeout=120) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
    mensajes = _resumir(cola, host="hall9000")       # lo que quedo tras el ultimo
    if mensajes:
        vistos.extend(mensajes)
    cuerpos = [l for m in vistos for l in m.splitlines() if l.startswith("cuerpo-")]
    assert rotaciones >= 3, f"la carrera no ocurrio de verdad ({rotaciones} rotaciones)"
    assert len(set(cuerpos)) == total, f"perdidos: {total - len(set(cuerpos))}"
    duplicados = [c for c, n in collections.Counter(cuerpos).items() if n > 1]
    assert duplicados == [] and len(cuerpos) == total, f"duplicados: {duplicados[:5]}"
    assert _resumir(cola, host="hall9000") is None   # nada quedo, nada se repite


def test_el_resumen_vacia_la_cola_y_no_deja_rotados_ni_repite(tmp_path):
    """Mutante f: tras resumir no queda cola, ni rotado, y un segundo resumen no repite nada."""
    cola = tmp_path / "cola.jsonl"
    for i in range(3):
        av = AvisoRegla(texto=f"cuerpo-{i}", clase="PERMIT|ok|RL-A", inmediato=False, creado_utc="2026-10-07T03:04:05+00:00")
        assert acumular_para_resumen(av, cola) is True
    primero = _resumir(cola, host="hall9000")
    assert primero is not None and all(f"cuerpo-{i}" in "\n".join(primero) for i in range(3))
    assert not cola.exists()
    assert list(tmp_path.glob("*.procesando*")) == []
    assert _resumir(cola, host="hall9000") is None


def test_un_rotado_que_sobrevivio_a_una_caida_se_recoge_y_un_fallo_de_lectura_no_borra_nada(tmp_path, monkeypatch):
    cola = tmp_path / "cola.jsonl"
    huerfano = tmp_path / "cola.jsonl.00000000000000000001.procesando"
    huerfano.write_text(_json.dumps({"creado_utc": "x", "clase": "DENY|r|RL-A", "texto": "huerfano-1",
                                     "suprimido_por_tasa": False}) + "\n", encoding="utf-8")
    huerfano.chmod(0o600)
    av = AvisoRegla(texto="nuevo-1", clase="DENY|r|RL-A", inmediato=True, creado_utc="2026-10-07T03:04:05+00:00")
    assert acumular_para_resumen(av, cola) is True
    # un fallo de lectura: devuelve None y NO se borra ni se pierde nada
    import jax.faro.aviso as modulo
    real = modulo._leer_seguro

    def lee_mal(ruta):
        raise OSError("disco roto")

    monkeypatch.setattr(modulo, "_leer_seguro", lee_mal)
    assert _resumir(cola, host="hall9000") is None                  # un fallo de disco (no un archivo inseguro) aborta
    monkeypatch.setattr(modulo, "_leer_seguro", real)
    assert len(list(tmp_path.glob("*.procesando*"))) == 2            # el huerfano y el recien rotado, reclamados
    mensajes = "\n".join(_resumir(cola, host="hall9000", lease_s=0))
    assert "huerfano-1" in mensajes and "nuevo-1" in mensajes
    assert list(tmp_path.glob("*.procesando*")) == [] and not cola.exists()


def test_el_resumen_numera_los_mensajes_y_no_trunca_en_silencio(tmp_path):
    """MAJOR-5: 200 avisos con cuerpos largos -> mensajes numerados (k/n), cada uno cabe en _MAX_TEXTO SIN
    cortar cuerpos, la cuenta dice 200 y la cola queda vacia: ninguno se pierde."""
    cola = tmp_path / "cola.jsonl"
    texto_largo = "x" * 300
    for i in range(200):
        av = AvisoRegla(texto=f"cuerpo-{i:03d} {texto_largo}", clase=f"PERMIT|ok|RL-{i % 7}",
                        inmediato=False, creado_utc="2026-10-07T03:04:05+00:00")
        assert acumular_para_resumen(av, cola) is True
    mensajes = _resumir(cola, host="hall9000")
    assert mensajes is not None and len(mensajes) > 1
    assert "200 avisos" in mensajes[0]
    for k, m in enumerate(mensajes, 1):
        assert len(m) <= _MAX_TEXTO
        assert m.splitlines()[-1] == f"· {k}/{len(mensajes)}"
    todo = "\n".join(mensajes)
    for i in range(200):
        assert f"cuerpo-{i:03d} {texto_largo}" in todo, f"el aviso {i} se perdio o se trunco"
    assert _resumir(cola, host="hall9000") is None


def test_un_cuerpo_gigante_se_recorta_con_marca_y_el_mensaje_cabe(tmp_path):
    cola = tmp_path / "cola.jsonl"
    av = AvisoRegla(texto="y" * 9000, clase="DENY|r|RL-A", inmediato=True, creado_utc="2026-10-07T03:04:05+00:00")
    assert acumular_para_resumen(av, cola) is True
    mensajes = _resumir(cola, host="hall9000")
    assert all(len(m) <= _MAX_TEXTO for m in mensajes)
    assert "car. recortados]" in "\n".join(mensajes)       # se dice, no se calla


def test_el_resumen_cuenta_las_lineas_ilegibles(tmp_path):
    cola = tmp_path / "cola.jsonl"
    assert acumular_para_resumen(AvisoRegla("ok-1", "DENY|r|RL-A", True, "2026-10-07T03:04:05+00:00"), cola) is True
    with open(cola, "a", encoding="utf-8") as f:
        f.write("esto no es json\n[1, 2]\n")
    mensajes = _resumir(cola, host="hall9000")
    assert "ilegibles=2" in mensajes[0] and "ok-1" in "\n".join(mensajes)


def test_el_resumen_no_se_borra_hasta_confirmar_y_se_reentrega_sin_duplicar(tmp_path):
    """MAJOR-1 / mutante «borra antes de confirmar»: si el envio falla (el llamador no confirma), los
    rotados siguen en disco; la corrida siguiente los re-entrega UNA vez cada linea; tras confirmar, nada."""
    cola = tmp_path / "cola.jsonl"
    for i in range(3):
        assert acumular_para_resumen(AvisoRegla(f"cuerpo-{i}", "DENY|r|RL-A", True, "2026-10-07T03:04:05+00:00"), cola)
    primero = resumen_diario(cola, host="hall9000")
    assert len(list(tmp_path.glob("*.procesando*"))) == 1, "los rotados se borraron antes de confirmar"
    # Telegram cae: no se confirma. Llega un aviso nuevo; la segunda corrida recoge ambos.
    assert acumular_para_resumen(AvisoRegla("cuerpo-3", "DENY|r|RL-A", True, "2026-10-07T03:04:05+00:00"), cola)
    segundo = resumen_diario(cola, host="hall9000", lease_s=0)         # el reclamo del primero vencio (lease 0)
    cuerpos = [l for m in segundo.mensajes for l in m.splitlines() if l.startswith("cuerpo-")]
    assert sorted(cuerpos) == [f"cuerpo-{i}" for i in range(4)]      # los 4, cada uno UNA vez
    assert confirmar_resumen(segundo) is True
    assert list(tmp_path.glob("*.procesando*")) == [] and not cola.exists()
    assert confirmar_resumen(primero) is True                          # token viejo: idempotente
    assert resumen_diario(cola, host="hall9000") is None


def test_confirmar_resumen_solo_borra_rotados_de_esa_cola(tmp_path):
    from jax.faro.aviso import ResumenDiario
    ajeno = tmp_path / "importante.txt"
    ajeno.write_text("no me borres")
    falso = ResumenDiario(("x",), (str(ajeno),), str(tmp_path / "cola.jsonl"))
    assert confirmar_resumen(falso) is False and ajeno.exists()


def test_un_envio_que_falla_va_a_la_cola_con_marca_y_el_resumen_lo_cuenta(tmp_path):
    """MAJOR-2 / mutante «solo log»: con Telegram caido (excepcion o False) la negativa no queda solo en un
    log: esta en la cola con envio_fallido y el resumen la cuenta."""
    def explota(cfg, cred, texto):
        raise RuntimeError("Telegram caido")

    for enviar in (explota, lambda cfg, cred, texto: False):
        cola = tmp_path / f"cola-{id(enviar)}.jsonl"
        av = aviso_de_decision(_decision(), host="hall9000")
        assert emitir_aviso_inmediato(av, _cfg(tmp_path), CRED, enviar=enviar, limite=LimiteTasa(5, 10.0),
                                      ruta_cola=cola) is False
        dato = _json.loads(cola.read_text(encoding="utf-8").splitlines()[0])
        assert dato["envio_fallido"] is True and dato["texto"] == av.texto
        resumen = _resumir(cola)
        assert "envio_fallido=1" in resumen[0] and "REGLA DENEGADA" in "\n".join(resumen)


def _cola_con_linea(ruta, texto="legitimo"):
    assert acumular_para_resumen(AvisoRegla(texto, "DENY|r|RL-A", True, "2026-10-07T03:04:05+00:00"), ruta)


def test_una_cola_que_es_un_enlace_no_se_sigue_y_va_a_cuarentena(tmp_path):
    """MINOR / mutante «sin O_NOFOLLOW»: un symlink plantado en lugar de la cola NO se lee (el destino es
    del usuario y 0600: solo O_NOFOLLOW lo detiene); queda en cuarentena y el resumen lo dice."""
    objetivo = tmp_path / "ajeno.jsonl"
    _cola_con_linea(objetivo, "SECRETO-DEL-DESTINO")
    antes = objetivo.read_text()
    cola = tmp_path / "cola.jsonl"
    cola.symlink_to(objetivo)
    r = resumen_diario(cola, host="hall9000")
    assert r is not None and "en_cuarentena=1" in r.mensajes[0]
    assert "SECRETO-DEL-DESTINO" not in "\n".join(r.mensajes)
    assert objetivo.read_text() == antes and objetivo.exists()
    assert confirmar_resumen(r) is True
    assert len(list(tmp_path.glob("*.cuarentena"))) == 1
    assert resumen_diario(cola, host="hall9000") is None             # la cuarentena no se vuelve a leer


def test_un_rotado_malo_no_aborta_el_resumen_los_buenos_se_entregan(tmp_path):
    """MINOR (1) / mutante «abortar todo»: un `.procesando` plantado como enlace va a cuarentena y los
    legitimos SE ENTREGAN."""
    objetivo = tmp_path / "ajeno.jsonl"
    _cola_con_linea(objetivo, "SECRETO-DEL-DESTINO")
    cola = tmp_path / "cola.jsonl"
    (tmp_path / "cola.jsonl.00000000000000000001.procesando").symlink_to(objetivo)
    _cola_con_linea(cola, "legitimo-1")
    r = resumen_diario(cola, host="hall9000")
    assert r is not None
    todo = "\n".join(r.mensajes)
    assert "legitimo-1" in todo and "en_cuarentena=1" in r.mensajes[0] and "SECRETO-DEL-DESTINO" not in todo
    assert confirmar_resumen(r) is True
    assert [p.name for p in tmp_path.glob("*.cuarentena")] == [
        p.name for p in tmp_path.glob("cola.jsonl.00000000000000000001.procesando.*.cuarentena")]
    assert objetivo.read_text().count("SECRETO-DEL-DESTINO") == 1


def test_una_cola_con_modo_o_dueno_ajeno_no_se_lee(tmp_path, monkeypatch):
    """MINOR / mutantes «sin modo» y «sin dueno»: van a cuarentena, no se leen."""
    cola = tmp_path / "cola.jsonl"
    _cola_con_linea(cola, "no-me-leas")
    cola.chmod(0o644)
    r = resumen_diario(cola, host="hall9000")
    assert "en_cuarentena=1" in r.mensajes[0] and "no-me-leas" not in "\n".join(r.mensajes)
    _cola_con_linea(cola, "no-me-leas-2")
    monkeypatch.setattr(os, "geteuid", lambda: 12345)
    r = resumen_diario(cola, host="hall9000")
    assert "en_cuarentena=1" in r.mensajes[0] and "no-me-leas-2" not in "\n".join(r.mensajes)


def test_el_patron_de_rotados_es_exacto_y_dos_colas_no_se_mezclan(tmp_path):
    """MINOR (2) / mutante «glob laxo»: `cola.jsonl.otra` en la misma carpeta no se mezcla con `cola.jsonl`."""
    a, b = tmp_path / "cola.jsonl", tmp_path / "cola.jsonl.otra"
    _cola_con_linea(a, "de-A")
    _cola_con_linea(b, "de-B")
    rb = resumen_diario(b, host="hall9000")              # rota B: `cola.jsonl.otra.<ts>.procesando`
    ra = resumen_diario(a, host="hall9000")
    assert "de-A" in "\n".join(ra.mensajes) and "de-B" not in "\n".join(ra.mensajes)
    assert "de-B" in "\n".join(rb.mensajes) and "de-A" not in "\n".join(rb.mensajes)
    assert confirmar_resumen(ra) is True
    assert len(list(tmp_path.glob("cola.jsonl.otra.*.procesando.*"))) == 1     # lo de B sigue intacto
    cruzado = type(ra)(rb.mensajes, rb.rotados, str(a))                          # token de B contra la cola A
    assert confirmar_resumen(cruzado) is False
    assert len(list(tmp_path.glob("cola.jsonl.otra.*.procesando.*"))) == 1
    assert confirmar_resumen(rb) is True


def test_confirmar_resumen_no_borra_un_rotado_sin_reclamar(tmp_path):
    """q5 / mutante «quitar la guarda m.group(2) is None»: un token que liste un `.procesando` SIN el
    sufijo de reclamo (de otro, o fabricado) devuelve False y no borra nada."""
    from jax.faro.aviso import ResumenDiario
    cola = tmp_path / "cola.jsonl"
    sin_reclamar = tmp_path / "cola.jsonl.00000000000000000007.procesando"
    sin_reclamar.write_text("x\n")
    token = ResumenDiario(("m",), (str(sin_reclamar),), str(cola))
    assert confirmar_resumen(token) is False
    assert sin_reclamar.exists()


def test_el_patron_de_rotados_no_acepta_cuarentena(tmp_path):
    """q6 / mutante «patron acepta cuarentena»: lo apartado a cuarentena no es un rotado y no se vuelve a leer."""
    from jax.faro.aviso import _patron_rotado
    patron = _patron_rotado("cola.jsonl")
    ok = "cola.jsonl.00000000000000000007.procesando.123-0123456789abcdef"
    assert patron.fullmatch("cola.jsonl.00000000000000000007.procesando") and patron.fullmatch(ok)
    for malo in ("cola.jsonl.00000000000000000007.procesando.cuarentena", ok + ".cuarentena",
                 "cola.jsonl.cuarentena"):
        assert patron.fullmatch(malo) is None, malo
    cola = tmp_path / "cola.jsonl"
    for nombre in ("cola.jsonl.00000000000000000007.procesando.cuarentena", ok + ".cuarentena"):
        p = tmp_path / nombre
        p.write_text(_json.dumps({"creado_utc": "x", "clase": "c", "texto": "NO-ME-LEAS", "suprimido_por_tasa": False}) + "\n")
        p.chmod(0o600)
    assert resumen_diario(cola, host="hall9000") is None
    assert len(list(tmp_path.glob("*.cuarentena"))) == 2


def test_dos_resumidores_simultaneos_no_duplican(tmp_path):
    """MINOR (3) / mutante «sin reclamo»: lo reclamado por un resumidor no lo ve el otro; con lease vencido
    se re-reclama. Y dos hilos concurrentes: 0 duplicados."""
    cola = tmp_path / "cola.jsonl"
    for i in range(50):
        _cola_con_linea(cola, f"cuerpo-{i}")
    r1 = resumen_diario(cola, host="hall9000")
    assert resumen_diario(cola, host="hall9000") is None             # r1 lo reclamo: el segundo no lo ve
    assert resumen_diario(cola, host="hall9000", lease_s=0) is not None   # lease vencido: se re-reclama
    assert confirmar_resumen(r1) is True                              # el token viejo no borra lo del nuevo dueno
    # dos hilos a la vez sobre una cola recien llenada
    for p in tmp_path.glob("cola.jsonl.*"):
        p.unlink()
    for i in range(300):
        _cola_con_linea(cola, f"cuerpo-{i}")
    salidas: list = []

    def resumir():
        r = resumen_diario(cola, host="hall9000")
        if r is not None:
            salidas.append(r)

    hilos = [threading.Thread(target=resumir) for _ in range(4)]
    for h in hilos:
        h.start()
    for h in hilos:
        h.join()
    cuerpos = [l for r in salidas for m in r.mensajes for l in m.splitlines() if l.startswith("cuerpo-")]
    assert len(cuerpos) == 300 and len(set(cuerpos)) == 300, f"duplicados o perdidos: {len(cuerpos)}"
    assert all(confirmar_resumen(r) for r in salidas)


def test_solo_cuentan_las_lineas_json_con_texto_str(tmp_path):
    cola = tmp_path / "cola.jsonl"
    _cola_con_linea(cola, "bueno-1")
    with open(cola, "a", encoding="utf-8") as f:
        f.write('{"texto": 5}\n{"clase": "x"}\n"suelta"\n')
    mensajes = _resumir(cola)
    assert "1 avisos" in mensajes[0] and "ilegibles=3" in mensajes[0]


def test_acumular_es_una_linea_json_por_aviso(tmp_path):
    cola = tmp_path / "cola.jsonl"
    for i in range(3):
        av = AvisoRegla(texto=f"cuerpo-{i}", clase="DENY|RULE_EXPIRED|RL-A", inmediato=True,
                        creado_utc="2026-10-07T03:04:05+00:00")
        assert acumular_para_resumen(av, cola) is True
    lineas = cola.read_text(encoding="utf-8").splitlines()
    assert len(lineas) == 3
    for linea in lineas:
        assert set(_json.loads(linea)) == {"creado_utc", "clase", "texto", "suprimido_por_tasa", "envio_fallido"}


def test_acumular_es_fail_soft_sin_disco(tmp_path, caplog):
    av = AvisoRegla("t", "DENY|r|RL-A", True, "2026-10-07T03:04:05+00:00")
    with caplog.at_level(logging.WARNING):
        assert acumular_para_resumen(av, tmp_path / "no-existe" / "cola.jsonl") is False
    assert "no se pudo acumular" in caplog.text


def test_aviso_y_emitir_sin_io_por_ninguna_via(monkeypatch, tmp_path):
    """Mutantes a y a2: armar el aviso y emitirlo (sin suprimir) no abren NI UN archivo: ni open, ni io.open,
    ni os.open, ni Path.open/write_text/write_bytes. Los vigilantes llaman a la funcion REAL capturada
    antes de parchear: no hay recursion (MINOR 9)."""
    cfg, av = _cfg(tmp_path), aviso_de_decision(_decision(), host="hall9000")   # la config escribe: ANTES de vigilar
    abiertos = []
    reales = {"open": builtins.open, "io.open": io.open, "os.open": os.open, "P.open": pathlib.Path.open,
              "P.write_text": pathlib.Path.write_text, "P.write_bytes": pathlib.Path.write_bytes}

    def vigilar(nombre):
        real = reales[nombre]

        def vigilante(*a, **k):
            abiertos.append(nombre)
            return real(*a, **k)
        return vigilante

    monkeypatch.setattr(builtins, "open", vigilar("open"))
    monkeypatch.setattr(io, "open", vigilar("io.open"))
    monkeypatch.setattr(os, "open", vigilar("os.open"))
    monkeypatch.setattr(pathlib.Path, "open", vigilar("P.open"))
    monkeypatch.setattr(pathlib.Path, "write_text", vigilar("P.write_text"))
    monkeypatch.setattr(pathlib.Path, "write_bytes", vigilar("P.write_bytes"))
    assert emitir_aviso_inmediato(av, cfg, CRED, enviar=lambda *a: None,
                                  limite=LimiteTasa(5, 10.0), ruta_cola=tmp_path / "cola.jsonl") is True
    assert abiertos == [], "armar y emitir no tocan ni config, ni log a archivo, ni store"


def _imports_de(texto: str) -> set[str]:
    hallados = set()
    for nodo in ast.walk(ast.parse(texto)):
        if isinstance(nodo, ast.Import):
            hallados.update(alias.name for alias in nodo.names)
        elif isinstance(nodo, ast.ImportFrom):
            hallados.add(nodo.module or "")
            hallados.update(f"{nodo.module}.{alias.name}" for alias in nodo.names)
    return hallados


def test_ningun_modulo_de_rule_authority_importa_el_aviso_estaticamente():
    """Mutante i: la frontera del paso 8, lado estatico: por AST, ningun modulo de policy/rule_authority
    importa el aviso (el resto de jax.faro SI lo usa, p. ej. snapshot). (aviso -> modelos esta permitido; esta es la direccion contraria.)"""
    archivos = sorted((RAIZ / "policy" / "rule_authority").rglob("*.py"))
    assert archivos, "no se encontro policy/rule_authority"
    for py in archivos:
        for nombre in _imports_de(py.read_text(encoding="utf-8")):
            assert "aviso" not in nombre.split("."), \
                f"{py} importa {nombre!r}: el aviso no es parte de la autoridad"


def test_rule_authority_carga_sin_el_aviso_en_un_interprete_limpio():
    """Mutante h: el AST no ve un import dinamico ni transitivo; sys.modules de un interprete NUEVO si.
    Se importan TODOS los submodulos de policy.rule_authority."""
    nombres = [m.name for m in pkgutil.walk_packages(_rule_authority.__path__, "policy.rule_authority.")]
    assert "policy.rule_authority.models" in nombres
    codigo = ("import importlib, sys\n"
              "import policy.rule_authority, policy.rule_authority.models\n"
              f"for n in {nombres!r}: importlib.import_module(n)\n"
              "assert 'jax.faro.aviso' not in sys.modules, 'la autoridad cargo el aviso'\n"
              )
    r = subprocess.run([sys.executable, "-c", codigo], capture_output=True, text=True, cwd=str(RAIZ),
                       env={**os.environ, "PYTHONPATH": str(RAIZ)}, timeout=120)
    assert r.returncode == 0, (r.stdout, r.stderr)


def test_aviso_importa_el_modelo_real_de_371():
    """La direccion permitida: el aviso usa EL RuleDecision de #371, no un espejo."""
    import jax.faro.aviso as modulo
    assert modulo.RuleDecision is RuleDecision and modulo.RuleDecisionStatus is RuleDecisionStatus
