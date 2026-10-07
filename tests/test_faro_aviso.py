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
# corre despues. Contrato r2 contra el RuleDecision REAL de #371 (status,     #
# required_rule_id, reason_code, request_hash, decided_at_utc). Todo lo de    #
# aca es fail-soft y sin red (el envio va con un doble); la unica "red" es    #
# la de siempre: nunca la de verdad.                                          #
# --------------------------------------------------------------------------- #

import ast
import dataclasses
import json as _json
import pathlib
import subprocess
import sys
import types
from datetime import datetime, timezone

import builtins
import collections
import time as _time

from jax.faro.aviso import (AvisoRegla, RuleDecisionStatus, _MAX_TEXTO,
                            acumular_para_resumen, aviso_de_decision,
                            emitir_aviso_inmediato, resumen_diario)

DENY = "DENY"
MISSING_RULE = "MISSING_RULE"
PERMIT = "PERMIT"

HASH_OK = "sha256:" + "a1b2c3d4e5f67890deadbeeffeedface00112233445566778899aabbccddeeff"


@dataclasses.dataclass(frozen=True)
class _Decision:
    """Fiel al RuleDecision de #371 (31eb14bf): SUS seis campos, nada mas."""
    request_id: str
    request_hash: str
    status: str
    required_rule_id: str
    reason_code: str | None
    decided_at_utc: datetime


def _decision(status=DENY, regla="RL-para-gastar-dinero", razon="RULE_EXPIRED",
              cuando=None, hash_=HASH_OK, **extra):
    """Una decision del contrato; `extra` cuelga atributos EXTRA (subject,
    arguments...) para probar que no viajan: al alcance del getattr."""
    base = dict(request_id="018f3b2a-9c1d-7abc-def0-1234567890ab", request_hash=hash_,
                status=status, required_rule_id=regla, reason_code=razon,
                decided_at_utc=cuando or datetime(2026, 10, 7, 3, 4, 5, tzinfo=timezone.utc))
    base.update(extra)
    return types.SimpleNamespace(**base)


# El subject y los argumentos de la decision: LO QUE NO PUEDE SALIR (§11).
SECRETOS_DE_LA_DECISION = {
    "subject": "usuario-con-correo@secreto.hn",
    "arguments": ["sk-SECRETO-DE-PRUEBA-no-sale", "/etc/secreto/llaves.env"],
    "amount": 987654.32,
    "contenido": "MENSAJE-SECRETO-DEL-USUARIO",
    "archivo": "/tmp/plan-secreto.md",
}
SECRETOS_PLANOS = [s for v in SECRETOS_DE_LA_DECISION.values()
                   for s in (v if isinstance(v, list) else [str(v)])]


def test_los_tres_status_generan_aviso_inmediato_con_los_nombres_de_371():
    deny = aviso_de_decision(_decision(status=DENY), host="hall9000")
    assert deny.inmediato is True and "REGLA DENEGADA" in deny.texto
    missing = aviso_de_decision(_decision(status=MISSING_RULE, razon="RULE_NOT_FOUND"), host="hall9000")
    assert missing.inmediato is True and "SIN REGLA QUE CUBRA EL ACTO" in missing.texto
    permit = aviso_de_decision(_decision(status=PERMIT, razon=None), host="hall9000")
    # el contrato real no trae la clase del acto: no se puede EXCLUIR que obligue
    assert permit.inmediato is True and "SI HAY DUDA, OBLIGA" in permit.texto


def test_el_texto_lleva_regla_razon_hash_y_la_hora_utc_de_la_decision():
    cuando = datetime(2026, 10, 7, 3, 4, 5, tzinfo=timezone.utc)
    av = aviso_de_decision(_decision(status=DENY, cuando=cuando), host="hall9000")
    lineas = av.texto.splitlines()
    assert len(lineas) == 2                                   # ni una linea fabricada
    assert "regla=RL-para-gastar-dinero" in lineas[1]          # SUS nombres: required_rule_id
    assert "razon=RULE_EXPIRED" in lineas[1]
    assert "req=sha256:a1b2c3d4e5f6" in lineas[1]              # sha256: + 12 hex del digest
    assert f"a={cuando.isoformat(timespec='seconds')}" in lineas[1]   # la hora de la DECISION
    assert av.creado_utc == cuando.isoformat(timespec="seconds")
    assert av.creado_utc.endswith("+00:00")                    # UTC, no hora local


def test_una_decision_desconocida_es_aviso_inmediato_no_reconocible_sin_secretos():
    """BLOCK-2: nunca None silencioso. El aviso nombra la decision por su
    CLASE y nada mas -- el repr de un objeto cualquiera puede traer secretos."""
    rara = _decision(status="TAL_VEZ", **SECRETOS_DE_LA_DECISION)
    av = aviso_de_decision(rara, host="hall9000")
    assert av is not None and av.inmediato is True
    assert "DECISION NO RECONOCIBLE" in av.texto
    assert "SimpleNamespace" in av.texto
    for secreto in SECRETOS_PLANOS:
        assert secreto not in av.texto, f"el aviso de lo no reconocible filtro {secreto!r}"


def test_un_status_ausente_o_un_hash_ilegible_tambien_cierran_cerrado():
    sin_status = types.SimpleNamespace(**SECRETOS_DE_LA_DECISION)  # ni status tiene
    av = aviso_de_decision(sin_status, host="hall9000")
    assert av.inmediato is True and "DECISION NO RECONOCIBLE" in av.texto
    con_hash_int = _decision(request_hash=12345)  # MINOR 7: TypeError potencial
    av2 = aviso_de_decision(con_hash_int, host="hall9000")
    assert av2.inmediato is True and "DECISION NO RECONOCIBLE" in av2.texto


def test_el_texto_no_filtra_argumentos_montos_subject_ni_archivos():
    """La decision trae subject, arguments, monto, contenido y archivos como
    ATRIBUTOS REALES de una decision VALIDA: nada de eso viaja (§11, mutante g)."""
    decision = _decision(status=PERMIT, razon=None, **SECRETOS_DE_LA_DECISION)
    av = aviso_de_decision(decision, host="hall9000")
    for secreto in SECRETOS_PLANOS:
        assert secreto not in av.texto, f"el aviso filtro {secreto!r}"
    assert av.inmediato is True  # la decision es valida: se avisa, no se calla


def test_un_required_rule_id_hostil_no_fabrica_lineas():
    av = aviso_de_decision(_decision(regla="RL-1\ninyectada=SI\rSECRETO"), host="hall9000")
    assert len(av.texto.splitlines()) == 2


def test_emitir_exige_limite_y_ruta_de_cola_sin_defaults():
    """MINOR 12: un limite que no se pasa no limita nada y una cola que no se
    pasa pierde lo suprimido -- los dos son obligatorios."""
    av = aviso_de_decision(_decision(), host="hall9000")
    with pytest.raises(TypeError):
        emitir_aviso_inmediato(av, _cfg(tmp_path), CRED, enviar=lambda *a: None,
                               ruta_cola=tmp_path / "cola.jsonl")
    with pytest.raises(TypeError):
        emitir_aviso_inmediato(av, _cfg(tmp_path), CRED, enviar=lambda *a: None,
                               limite=LimiteTasa(5, 10.0))


def test_emitir_envia_el_texto_y_devuelve_true(tmp_path):
    av = aviso_de_decision(_decision(), host="hall9000")
    mandados = []

    def enviar(cfg, cred, texto):
        mandados.append(texto)

    limite = LimiteTasa(5, 10.0)
    cola = tmp_path / "cola.jsonl"
    assert emitir_aviso_inmediato(av, _cfg(tmp_path), CRED, enviar=enviar, limite=limite, ruta_cola=cola) is True
    assert mandados == [av.texto]
    assert not cola.exists()            # nada suprimido: la cola ni se crea


def test_emitir_es_fail_soft_cuando_el_envio_explota(tmp_path, caplog):
    av = aviso_de_decision(_decision(), host="hall9000")

    def explota(cfg, cred, texto):
        raise RuntimeError("el chat de Telegram no existe")

    with caplog.at_level(logging.WARNING):
        ok = emitir_aviso_inmediato(av, _cfg(tmp_path), CRED, enviar=explota,
                                    limite=LimiteTasa(5, 10.0), ruta_cola=tmp_path / "cola.jsonl")
    assert ok is False
    assert "aviso de regla no entregado" in caplog.text and "RuntimeError" in caplog.text


def test_lo_suprimido_por_tasa_va_a_la_cola_con_marca_y_no_tapa_otra_regla(tmp_path):
    """MAJOR-4: la clase de tasa incluye la REGLA -- una tormenta de RL-A no
    gasta el cupo de RL-B -- y lo suprimido NO se descarta: va a la cola con
    suprimido_por_tasa=true."""
    cola = tmp_path / "cola.jsonl"
    reloj = [0.0]
    limite = LimiteTasa(2, 1000.0, reloj=lambda: reloj[0])
    envios = []

    def enviar(cfg, cred, texto):
        envios.append(texto)

    cfg = _cfg(tmp_path)
    deneg_a = aviso_de_decision(_decision(status=DENY, regla="RL-A", razon="RULE_EXPIRED"), host="hall9000")
    deneg_b = aviso_de_decision(_decision(status=DENY, regla="RL-B", razon="RULE_EXPIRED"), host="hall9000")

    assert emitir_aviso_inmediato(deneg_a, cfg, CRED, enviar=enviar, limite=limite, ruta_cola=cola) is True
    reloj[0] += 1.0
    assert emitir_aviso_inmediato(deneg_a, cfg, CRED, enviar=enviar, limite=limite, ruta_cola=cola) is True
    reloj[0] += 1.0
    assert emitir_aviso_inmediato(deneg_a, cfg, CRED, enviar=enviar, limite=limite, ruta_cola=cola) is False  # RL-A agoto su rafaga
    assert emitir_aviso_inmediato(deneg_b, cfg, CRED, enviar=enviar, limite=limite, ruta_cola=cola) is True   # RL-B tiene la suya
    assert len(envios) == 3

    lineas = cola.read_text(encoding="utf-8").splitlines()
    assert len(lineas) == 1                              # el suprimido de RL-A esta en la cola
    dato = _json.loads(lineas[0])
    assert dato["suprimido_por_tasa"] is True
    assert "RL-A" in dato["clase"]


def test_la_cola_y_el_candado_nacen_y_se_mantienen_0600(tmp_path):
    """MAJOR-3: ni la cola ni el candado dependen del umask de quien llama --
    se mide con un umask laxo 022 y tras resumir y volver a acumular."""
    cola = tmp_path / "cola.jsonl"
    av = aviso_de_decision(_decision(status=PERMIT, razon=None), host="hall9000")
    old = os.umask(0o022)
    try:
        assert acumular_para_resumen(av, cola) is True
        assert cola.stat().st_mode & 0o777 == 0o600
        assert (tmp_path / "cola.jsonl.candado").stat().st_mode & 0o777 == 0o600
        assert resumen_diario(cola, host="hall9000") is not None
        assert acumular_para_resumen(av, cola) is True      # la cola NUEVA tras el resumen
        assert cola.stat().st_mode & 0o777 == 0o600
    finally:
        os.umask(old)


def test_la_carrera_escritor_resumen_no_pierde_ni_duplica(tmp_path):
    """BLOCK-1: un escritor REAL en otro proceso mete 20 000 lineas mientras
    este proceso resume en bucle. Perdidos = 0, duplicados = 0."""
    import tempfile
    cola = tmp_path / "cola.jsonl"
    total = 20000
    escritor = tmp_path / "escritor.py"
    escritor.write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(pathlib.Path(__file__).resolve().parents[1])!r})\n"
        "from jax.faro.aviso import AvisoRegla, acumular_para_resumen\n"
        f"cola = {str(cola)!r}\n"
        f"for i in range({total}):\n"
        "    acumular_para_resumen(AvisoRegla(texto=f'cuerpo-{i}', clase='PERMIT|ok|RL-A',"
        " inmediato=False, creado_utc='2026-10-07T03:04:05+00:00'), cola)\n"
        "print('listo')\n", encoding="utf-8")
    proc = subprocess.Popen([sys.executable, str(escritor)], stdout=subprocess.DEVNULL)
    vistos: list[str] = []
    try:
        while proc.poll() is None or cola.exists():
            mensajes = resumen_diario(cola, host="hall9000")
            if mensajes:
                vistos.extend(mensajes)
            if proc.poll() is not None and not cola.exists():
                break
            _time.sleep(0.01)
    finally:
        proc.wait(timeout=120)
    mensajes = resumen_diario(cola, host="hall9000")
    if mensajes:
        vistos.extend(mensajes)
    cuerpos = [l for m in vistos for l in m.splitlines() if l.startswith("cuerpo-")]
    conteo = collections.Counter(cuerpos)
    assert len(cuerpos) == total, f"perdidos: {total - len(cuerpos)}"
    duplicados = [c for c, n in conteo.items() if n > 1]
    assert duplicados == [], f"duplicados: {duplicados[:5]}"


def test_el_resumen_numera_los_mensajes_y_no_trunca_en_silencio(tmp_path):
    """MAJOR-5: 200 avisos con cuerpos largos -> mensajes numerados (k/n), cada
    uno cabe en _MAX_TEXTO SIN cortar cuerpos, y la cola queda vacia."""
    cola = tmp_path / "cola.jsonl"
    texto_largo = "x" * 300
    for i in range(200):
        av = AvisoRegla(texto=f"cuerpo-{i:03d} {texto_largo}", clase=f"PERMIT|ok|RL-{i % 7}",
                        inmediato=False, creado_utc="2026-10-07T03:04:05+00:00")
        assert acumular_para_resumen(av, cola) is True
    mensajes = resumen_diario(cola, host="hall9000")
    assert mensajes is not None and len(mensajes) > 1
    for k, m in enumerate(mensajes, 1):
        assert len(m) <= _MAX_TEXTO
        assert m.splitlines()[-1] == f"· {k}/{len(mensajes)}"
    todo = "\n".join(mensajes)
    for i in range(200):                      # ningun cuerpo cortado a la mitad
        assert f"cuerpo-{i:03d} {texto_largo}" in todo, f"el aviso {i} se perdio o trunco"
    assert resumen_diario(cola, host="hall9000") is None      # la cola quedo vacia


def test_resumen_sin_cola_devuelve_none(tmp_path):
    assert resumen_diario(tmp_path / "no-existe.jsonl", host="hall9000") is None


def test_acumular_es_una_linea_json_por_aviso(tmp_path):
    cola = tmp_path / "cola.jsonl"
    for i in range(3):
        av = AvisoRegla(texto=f"cuerpo-{i}", clase="DENY|RULE_EXPIRED|RL-A",
                        inmediato=True, creado_utc="2026-10-07T03:04:05+00:00")
        assert acumular_para_resumen(av, cola) is True
    lineas = cola.read_text(encoding="utf-8").splitlines()
    assert len(lineas) == 3
    for linea in lineas:
        assert set(_json.loads(linea)) == {"creado_utc", "clase", "texto", "suprimido_por_tasa"}


def test_aviso_y_emitir_sin_io_por_ninguna_via(monkeypatch, tmp_path):
    """Mutantes a y a2: armar el aviso y emitirlo no abren NI UN archivo --
    ni con open, ni con os.open, ni con Path.write_text. Los vigilantes
    capturan la funcion REAL antes de parchear: sin recursion (MINOR 9)."""
    abiertos = []
    open_real, os_open_real = open, os.open
    write_text_real = pathlib.Path.write_text

    def open_vigilado(*a, **k):
        abiertos.append(("open", a))
        return open_real(*a, **k)

    def os_open_vigilado(*a, **k):
        abiertos.append(("os.open", a))
        return os_open_real(*a, **k)

    def write_text_vigilado(self, *a, **k):
        abiertos.append(("write_text", (str(self),)))
        return write_text_real(self, *a, **k)

    monkeypatch.setattr(builtins, "open", open_vigilado)
    monkeypatch.setattr(os, "open", os_open_vigilado)
    monkeypatch.setattr(pathlib.Path, "write_text", write_text_vigilado)
    av = aviso_de_decision(_decision(), host="hall9000")
    assert av.inmediato is True
    assert emitir_aviso_inmediato(av, _cfg(tmp_path), CRED, enviar=lambda *a: None,
                                  limite=LimiteTasa(5, 10.0), ruta_cola=tmp_path / "cola.jsonl") is True
    assert abiertos == [], "armar y emitir no tocan ni config, ni log a archivo, ni store"


def test_ningun_modulo_de_rule_authority_importa_aviso():
    """La frontera del paso 8 (§14.8), lado estatico: por AST, ningun modulo de
    policy/rule_authority importa el aviso (directo o por alias). Que aviso
    importe los MODELOS de #371 si esta permitido ( direccion aviso->modelos)."""
    raiz = pathlib.Path(__file__).resolve().parents[1]

    def _imports(texto):
        hallados = set()
        for nodo in ast.walk(ast.parse(texto)):
            if isinstance(nodo, ast.Import):
                hallados.update(alias.name for alias in nodo.names)
            elif isinstance(nodo, ast.ImportFrom) and nodo.module:
                hallados.add(nodo.module)
        return hallados

    for py in sorted((raiz / "policy" / "rule_authority").rglob("*.py")):
        for nombre in _imports(py.read_text(encoding="utf-8")):
            assert "aviso" not in nombre, f"{py} importa {nombre!r}: el aviso no es parte de la autoridad"


def test_rule_authority_carga_sin_el_aviso_en_un_interprete_limpio(tmp_path):
    """MINOR 10: el invariante en un interprete REAL -- importar TODA la
    autoridad no puede arrastrar el aviso (import estatico, transitivo o
    dinamico: sys.modules lo ve todo)."""
    raiz = pathlib.Path(__file__).resolve().parents[1]
    codigo = "import sys\nimport policy.rule_authority\nassert 'jax.faro.aviso' not in sys.modules, 'la autoridad cargo el aviso'\n"
    r = subprocess.run([sys.executable, "-c", codigo], capture_output=True, text=True,
                       cwd=str(raiz), timeout=120)
    assert r.returncode == 0, (r.stdout, r.stderr)
