# tests/test_ejecutor_vigia_servicio.py
"""El arranque de una misión: primero los contratos, después el vigía; sin contratos
nunca late (y sin latido el proxy no sirve). Fin normal: el latido se borra."""
import asyncio
import json
from pathlib import Path

import pytest

from jax.ejecutor.contratos import arranque as AR
from jax.ejecutor.contratos import auditor as A
from jax.ejecutor.contratos import pausa as P
from jax.ejecutor.contratos import vigia_servicio as S
from jax.ejecutor.contratos.cuenta_axioma import Cuenta
from jax.ejecutor.contratos.fallo import Fallo
from jax.ejecutor.contratos.registro import Registro

MISION = S.Mision("uptime de hall9000", frozenset({"hall9000"}))
MAQUINAS = (A.Maquina("hall9000", "192.0.2.5", 58291),)


def _ctx(tmp_path, hosts=frozenset({"hall9000"})):
    reg = Registro(tmp_path / "registro.jsonl")
    reg.anotar({"evento": "registro_abierto", "pid": 1})
    reg.cerrar()
    return AR.Contexto(cuenta=Cuenta("axioma", 58291, Path("/k"), Path("/n"), tmp_path / "lib", tmp_path / "p.json", Path("/home/axioma")),
                       repo=tmp_path, puerto_canario=1, registro=tmp_path / "registro.jsonl", puerto_proxy=2,
                       sondas=(3,), estado_freno=tmp_path / "e.json", llaves_root=tmp_path / "llaves",
                       tope_gancho_s=10, hosts_mision=hosts, pausa=tmp_path / "PAUSA", latido=tmp_path / "latido",
                       latido_max_s=5.0)


async def _auditar(lote):
    return A.Revision(False, None, None, (), frozenset(), frozenset())


def _correr(ctx, *, exigir, vigilar, fin_antes=False):
    async def escenario():
        f = asyncio.Event()
        if fin_antes:
            f.set()
        await S.correr_mision(ctx, MISION, latido_cada_s=0.05, lote_max=5, intervalo_s=1.0, auditar=_auditar,
                              fin=f, exigir=exigir, vigilar=vigilar, maquinas=MAQUINAS)
    asyncio.run(escenario())


def test_sin_contratos_no_hay_vigia_ni_latido(tmp_path):
    ctx = _ctx(tmp_path)
    vigilado = []

    async def exigir(c):
        raise AR.ContratosNoVerificados((Fallo("c4", "freno_sin_latido"),))

    async def vigilar(cfg, auditar, fin):
        vigilado.append(cfg)

    with pytest.raises(AR.ContratosNoVerificados):
        _correr(ctx, exigir=exigir, vigilar=vigilar)
    assert vigilado == [] and not ctx.latido.exists()


def test_primero_los_contratos_despues_el_vigia_desde_el_final_del_registro(tmp_path):
    ctx = _ctx(tmp_path)
    orden = []

    async def exigir(c):
        orden.append(("exigir", c.hosts_mision))

    async def vigilar(cfg, auditar, fin):
        orden.append(("vigilar", cfg.desde_byte, cfg.mision, cfg.pausa, cfg.latido))
        P.latir(cfg.latido)

    _correr(ctx, exigir=exigir, vigilar=vigilar)
    assert orden == [("exigir", frozenset({"hall9000"})),
                     ("vigilar", ctx.registro.stat().st_size, MISION.texto, ctx.pausa, ctx.latido)]
    assert not ctx.latido.exists()


def test_parado_mientras_se_verificaban_los_contratos_no_abre(tmp_path):
    ctx = _ctx(tmp_path)
    vigilado = []

    async def exigir(c):
        return None

    async def vigilar(cfg, auditar, fin):
        vigilado.append(cfg)

    _correr(ctx, exigir=exigir, vigilar=vigilar, fin_antes=True)
    assert vigilado == [] and not ctx.latido.exists()


def test_vigia_que_revienta_no_borra_el_latido(tmp_path):
    ctx = _ctx(tmp_path)

    async def exigir(c):
        return None

    async def vigilar(cfg, auditar, fin):
        P.latir(cfg.latido)
        raise RuntimeError("caido")

    with pytest.raises(RuntimeError):
        _correr(ctx, exigir=exigir, vigilar=vigilar)
    assert ctx.latido.exists()


def test_contexto_de_otra_mision_no_arranca(tmp_path):
    ctx = _ctx(tmp_path, hosts=frozenset({"bridge"}))
    llamado = []

    async def exigir(c):
        llamado.append(c)

    with pytest.raises(ValueError):
        _correr(ctx, exigir=exigir, vigilar=None)
    assert llamado == []


def test_con_el_vigia_real_el_latido_abre_y_el_fin_lo_cierra(tmp_path):
    """Lo que ve el proxy (`pausa.latido_fresco`): cerrado antes, abierto con el vigía, cerrado al terminar."""
    ctx = _ctx(tmp_path)

    async def exigir(c):
        assert not P.latido_fresco(c.latido, c.latido_max_s)

    async def escenario():
        fin = asyncio.Event()
        tarea = asyncio.create_task(S.correr_mision(ctx, MISION, latido_cada_s=0.05, lote_max=5, intervalo_s=1.0,
                                                    auditar=_auditar, fin=fin, exigir=exigir, maquinas=MAQUINAS))
        for _ in range(100):
            if P.latido_fresco(ctx.latido, ctx.latido_max_s):
                break
            await asyncio.sleep(0.02)
        abierto = P.latido_fresco(ctx.latido, ctx.latido_max_s)
        fin.set()
        await asyncio.wait_for(tarea, 5)
        return abierto

    assert asyncio.run(escenario()) is True
    assert not P.latido_fresco(ctx.latido, ctx.latido_max_s) and not ctx.pausa.exists()


@pytest.mark.parametrize("datos, codigo", [
    (b"{", "mision_no_es_json"),
    (b"[]", "mision_no_es_objeto"),
    (b'{"hosts": ["hall9000"]}', "mision_sin_texto"),
    (b'{"mision": "  ", "hosts": ["hall9000"]}', "mision_sin_texto"),
    (b'{"mision": "x"}', "mision_sin_maquinas"),
    (b'{"mision": "x", "hosts": []}', "mision_sin_maquinas"),
    (b'{"mision": "x", "hosts": ["", "a"]}', "mision_sin_maquinas"),
])
def test_mision_ilegible(datos, codigo):
    with pytest.raises(S.MisionIlegible) as e:
        S.mision_desde_bytes(datos)
    assert e.value.args[0] == codigo


def test_mision_legible():
    assert S.mision_desde_bytes(b'{"mision": " uptime ", "hosts": ["hall9000", "atemai"]}') == S.Mision(
        "uptime", frozenset({"hall9000", "atemai"}))


@pytest.mark.parametrize("valor", [None, "x", "0", "-1", "30", "31"])
def test_latido_cada_tiene_que_ser_menor_que_el_maximo(valor):
    env = {} if valor is None else {S.VARIABLE_LATIDO_CADA_S: valor}
    with pytest.raises(ValueError):
        S.latido_cada_desde_entorno(env, 30.0)
    assert S.latido_cada_desde_entorno({S.VARIABLE_LATIDO_CADA_S: "5"}, 30.0) == 5.0


def test_el_vigia_audita_con_las_maquinas_de_la_mision(tmp_path):
    """El vigía en vuelo también juzga «esta máquina»: sus lotes llevan las máquinas elegidas."""
    ctx = _ctx(tmp_path)
    vistas = []

    async def exigir(c):
        return None

    async def vigilar(cfg, auditar, fin):
        vistas.append(cfg.maquinas)

    maquinas = (A.Maquina("hall9000", "172.16.20.5", 58291),)

    async def escenario():
        await S.correr_mision(ctx, MISION, latido_cada_s=0.05, lote_max=5, intervalo_s=1.0, auditar=_auditar,
                              fin=asyncio.Event(), exigir=exigir, vigilar=vigilar, maquinas=maquinas)
    asyncio.run(escenario())
    assert vistas == [maquinas]


# --- huella (M-1/M-2, ronda 3, auditoría adversarial 2026-09-22) ---------------------

async def _exigir_ok(c):
    return None


async def _vigilar_noop(cfg, auditar, fin):
    return None


MISION_ID = "11111111-1111-1111-1111-111111111111"
OTRA_MISION_ID = "22222222-2222-2222-2222-222222222222"


def _h(host, controles=None):
    from jax.ejecutor.contratos import huella as H
    return H.huella_desde_salida(host, b"abc  /etc/sudoers\n" if controles is None else controles)


def test_sin_tomar_huella_no_se_toma_ninguna(tmp_path):
    """`tomar_huella=None` (el default): cero llamadas, cero pausa -- los llamadores
    que no la necesitan no cambian de comportamiento."""
    ctx = _ctx(tmp_path)
    llamadas = []

    async def tomar_huella_falsa(h):
        llamadas.append(h)

    async def escenario():
        return await S.correr_mision(ctx, MISION, latido_cada_s=0.05, lote_max=5, intervalo_s=1.0, auditar=_auditar,
                                     fin=asyncio.Event(), exigir=_exigir_ok, vigilar=_vigilar_noop, maquinas=MAQUINAS,
                                     hosts_con_sudo=("atemai",))  # sin tomar_huella: no debería usarse
    pausas = asyncio.run(escenario())
    assert llamadas == [] and pausas == ()
    assert not ctx.pausa.exists()


def _tomador_secuencia(mapa_por_llamada):
    """`mapa_por_llamada = {host: [huella_1, huella_2, ...]}` -- la primera llamada por
    host devuelve el primer elemento, la segunda el segundo, etc."""
    contadores = {h: 0 for h in mapa_por_llamada}

    async def tomar(host):
        i = contadores[host]
        contadores[host] += 1
        valor = mapa_por_llamada[host][i]
        if isinstance(valor, Exception):
            raise valor
        return valor
    return tomar


def _correr_con_huella(ctx, mision, *, hosts_con_sudo, tomar_huella, misiones, mision_id):
    async def escenario():
        return await S.correr_mision(
            ctx, mision, latido_cada_s=0.05, lote_max=5, intervalo_s=1.0, auditar=_auditar, fin=asyncio.Event(),
            exigir=_exigir_ok, vigilar=_vigilar_noop, maquinas=MAQUINAS, hosts_con_sudo=hosts_con_sudo,
            misiones=misiones, mision_id=mision_id, tomar_huella=tomar_huella)
    return asyncio.run(escenario())


def test_huella_cambiada_pone_la_pausa(tmp_path):
    """Ronda 6: sin declarado -- cualquier cambio pausa, sin importar el texto de la
    misión."""
    antes = _h("atemai")
    despues = _h("atemai", controles=b"abc  /etc/sudoers\ndef  /root/.ssh/authorized_keys\n")
    tomar_huella = _tomador_secuencia({"atemai": [antes, despues]})
    ctx = _ctx(tmp_path)

    pausas = _correr_con_huella(ctx, MISION, hosts_con_sudo=("atemai",), tomar_huella=tomar_huella,
                                misiones=tmp_path / "misiones", mision_id=MISION_ID)
    assert pausas == (("atemai", "huella_cambio_no_declarado"),)
    assert ctx.pausa.exists()
    import json
    datos = json.loads(ctx.pausa.read_text())
    assert datos["origen"] == "huella" and datos["motivo"] == "huella_cambio_no_declarado"
    assert datos["host"] == "atemai"
    assert "authorized_keys" in datos["detalle"][0]


def test_sin_cambio_en_la_huella_no_pausa(tmp_path):
    antes = _h("atemai")
    despues = _h("atemai")
    tomar_huella = _tomador_secuencia({"atemai": [antes, despues]})
    ctx = _ctx(tmp_path)

    pausas = _correr_con_huella(ctx, MISION, hosts_con_sudo=("atemai",), tomar_huella=tomar_huella,
                                misiones=tmp_path / "misiones", mision_id=MISION_ID)
    assert pausas == ()
    assert not ctx.pausa.exists()


def test_huella_de_cierre_vacia_pausa_como_no_medible(tmp_path):
    """MINOR (ronda 6): una huella de cierre vacía (no parsea/no midió nada real) no
    es "sin cambios" -- es no-medible, y pausa."""
    antes = _h("atemai")
    tomar_huella = _tomador_secuencia({"atemai": [antes, _h("atemai", controles=b"")]})
    ctx = _ctx(tmp_path)

    pausas = _correr_con_huella(ctx, MISION, hosts_con_sudo=("atemai",), tomar_huella=tomar_huella,
                                misiones=tmp_path / "misiones", mision_id=MISION_ID)
    assert pausas == (("atemai", "huella_no_medible"),)


def test_huella_de_cierre_ilegible_pausa(tmp_path):
    """El CIERRE falla cerrado: perder la MEDICIÓN de cierre (ssh caído, timeout) es,
    por sí sola, un hallazgo `huella_no_medible` que pausa."""
    antes = _h("atemai")
    tomar_huella = _tomador_secuencia({"atemai": [antes, RuntimeError("ssh caído")]})
    ctx = _ctx(tmp_path)

    pausas = _correr_con_huella(ctx, MISION, hosts_con_sudo=("atemai",), tomar_huella=tomar_huella,
                                misiones=tmp_path / "misiones", mision_id=MISION_ID)
    assert pausas == (("atemai", "huella_no_medible"),)
    assert ctx.pausa.exists()


def test_apertura_que_revienta_no_abre_la_mision(tmp_path):
    """M-3/M-4: `tomar_huella` que revienta al ABRIR se PROPAGA -- no hay try/except
    que la trague. Sin huella de apertura, la misión no debe seguir."""
    tomar_huella = _tomador_secuencia({"atemai": [RuntimeError("sin ssh")]})
    ctx = _ctx(tmp_path)

    with pytest.raises(RuntimeError):
        _correr_con_huella(ctx, MISION, hosts_con_sudo=("atemai",), tomar_huella=tomar_huella,
                          misiones=tmp_path / "misiones", mision_id=MISION_ID)


def test_apertura_vacia_revienta(tmp_path):
    """MINOR (ronda 6): una huella de APERTURA vacía es fallo -- la misión no abre."""
    tomar_huella = _tomador_secuencia({"atemai": [_h("atemai", controles=b"")]})
    ctx = _ctx(tmp_path)

    with pytest.raises(RuntimeError, match="huella_apertura_vacia"):
        _correr_con_huella(ctx, MISION, hosts_con_sudo=("atemai",), tomar_huella=tomar_huella,
                          misiones=tmp_path / "misiones", mision_id=MISION_ID)


def test_turno_n_mas_1_no_blanquea_un_cambio_del_turno_n(tmp_path):
    """La línea base es la de la APERTURA DE LA MISIÓN, persistida por `mision_id` --
    no la del turno anterior. El turno 1 fija la huella de apertura; el turno 2 (un
    `correr_mision` DISTINTO, misma `mision_id`) tiene que seguir comparando contra esa
    MISMA apertura."""
    misiones = tmp_path / "misiones"
    original = _h("atemai")
    cambiado_sin_declarar = _h("atemai", controles=b"abc  /etc/sudoers\ndef  /root/.ssh/authorized_keys\n")

    # Turno 1: abre limpio, cierra limpio (nadie detecta nada -- el cambio pasa DESPUÉS).
    t1 = _tomador_secuencia({"atemai": [original, _h("atemai")]})
    ctx1 = _ctx(tmp_path)
    pausas1 = _correr_con_huella(ctx1, MISION, hosts_con_sudo=("atemai",), tomar_huella=t1,
                                 misiones=misiones, mision_id=MISION_ID)
    assert pausas1 == ()

    # Turno 2: nadie dejó una marca pendiente (turno 1 cerró limpio) y ya existe una
    # apertura persistida para esta `mision_id` -- así que la ÚNICA llamada a
    # `tomar_huella` en todo el turno 2 tiene que ser la del CIERRE. Si el turno 2
    # volviera a TOMAR la apertura (en vez de cargar la persistida), vería el estado
    # YA cambiado y lo tomaría como su propio punto de partida -- exactamente lo que
    # este test prueba que NO pasa.
    llamadas = []

    async def tomar_turno2(host):
        llamadas.append(host)
        return cambiado_sin_declarar
    ctx2 = _ctx(tmp_path)
    pausas2 = _correr_con_huella(ctx2, MISION, hosts_con_sudo=("atemai",), tomar_huella=tomar_turno2,
                                 misiones=misiones, mision_id=MISION_ID)
    assert len(llamadas) == 1  # sólo el cierre: ni la huérfana (no hay) ni una apertura nueva
    assert pausas2 == (("atemai", "huella_cambio_no_declarado"),)
    assert ctx2.pausa.exists()


# --- M-1 (ronda 6): deuda de verificación huérfana --------------------------------------

def test_huerfana_de_otra_mision_limpia_deja_abrir_y_la_marca(tmp_path):
    misiones = tmp_path / "misiones"
    huella_vieja = _h("atemai")
    S._escribir_huella_persistida(S.ruta_huella(misiones, OTRA_MISION_ID, "atemai"), huella_vieja, pendiente=True)

    # La huérfana se revisa PRIMERO (misma huella=sin cambio), y RECIÉN DESPUÉS se toma
    # la de esta misión nueva -- dos llamadas: la revisión de la huérfana, y la propia.
    tomar = _tomador_secuencia({"atemai": [huella_vieja, _h("atemai")]})
    ctx = _ctx(tmp_path)

    async def abrir():
        return await S.huella_de_apertura_de_la_mision(
            misiones=misiones, mision_id=MISION_ID, host="atemai", tomar_huella=tomar,
            pausar=P.poner_pausa, pausa_ruta=ctx.pausa)
    resultado = asyncio.run(abrir())
    assert resultado is not None
    assert not ctx.pausa.exists()
    # La huérfana quedó resuelta (pendiente=False), no borrada.
    datos = json.loads(S.ruta_huella(misiones, OTRA_MISION_ID, "atemai").read_text())
    assert datos["pendiente"] is False


def test_huerfana_con_cambio_no_deja_abrir_la_mision_nueva(tmp_path):
    """M-1: «así un kill -9 o un reinicio no blanquean nada» -- una marca huérfana de
    OTRA misión que cambió bloquea la apertura de ÉSTA, aunque su propia huella fuera
    a salir limpia."""
    misiones = tmp_path / "misiones"
    huella_vieja = _h("atemai")
    S._escribir_huella_persistida(S.ruta_huella(misiones, OTRA_MISION_ID, "atemai"), huella_vieja, pendiente=True)

    async def apertura_no_deberia_llamarse(host):
        raise AssertionError("no debe abrir su propia huella si la huérfana falló")

    huella_cambiada = _h("atemai", controles=b"abc  /etc/sudoers\ndef  /root/.ssh/authorized_keys\n")
    llamadas = {"n": 0}

    async def tomar(host):
        llamadas["n"] += 1
        return huella_cambiada  # la revisión de la huérfana ve el cambio
    ctx = _ctx(tmp_path)

    with pytest.raises(S.HuellaHuerfanaNoResuelta):
        _correr_con_huella(ctx, MISION, hosts_con_sudo=("atemai",), tomar_huella=tomar,
                          misiones=misiones, mision_id=MISION_ID)
    assert llamadas["n"] == 1  # sólo la revisión de la huérfana -- nunca llegó a abrir la propia
    assert ctx.pausa.exists()
    datos = json.loads(ctx.pausa.read_text())
    assert datos["motivo"] == "huella_cambio_no_declarado"
    # La marca huérfana SIGUE pendiente (no se limpió: siguió cambiada).
    persistida = json.loads(S.ruta_huella(misiones, OTRA_MISION_ID, "atemai").read_text())
    assert persistida["pendiente"] is True


def test_huerfana_ilegible_no_deja_abrir(tmp_path):
    misiones = tmp_path / "misiones"
    ruta = S.ruta_huella(misiones, OTRA_MISION_ID, "atemai")
    ruta.parent.mkdir(parents=True)
    ruta.write_text("{esto no es json valido")

    async def tomar(host):
        raise AssertionError("no debería llamarse: la marca ilegible ya corta antes")
    ctx = _ctx(tmp_path)

    with pytest.raises(S.HuellaHuerfanaNoResuelta):
        _correr_con_huella(ctx, MISION, hosts_con_sudo=("atemai",), tomar_huella=tomar,
                          misiones=misiones, mision_id=MISION_ID)
    assert ctx.pausa.exists()


def test_marca_no_pendiente_no_se_revisa_de_nuevo(tmp_path):
    misiones = tmp_path / "misiones"
    S._escribir_huella_persistida(S.ruta_huella(misiones, OTRA_MISION_ID, "atemai"), _h("atemai"), pendiente=False)

    llamadas = []

    async def tomar(host):
        llamadas.append(host)
        return _h("atemai")
    ctx = _ctx(tmp_path)

    async def abrir():
        return await S.huella_de_apertura_de_la_mision(
            misiones=misiones, mision_id=MISION_ID, host="atemai", tomar_huella=tomar,
            pausar=P.poner_pausa, pausa_ruta=ctx.pausa)
    resultado = asyncio.run(abrir())
    assert resultado is not None
    assert llamadas == ["atemai"]  # una sola llamada: la propia apertura, sin volver a medir la ya resuelta


def test_kill_9_deja_marca_pendiente_que_la_proxima_mision_encuentra(tmp_path):
    """Simula un `kill -9`: la misión 1 abre (queda `pendiente=true`) y NUNCA cierra
    (el proceso muere entre abrir y cerrar -- no se llega a
    `_verificar_huellas_al_cierre`). La misión 2 (otra `mision_id`) tiene que
    encontrarla al abrir."""
    misiones = tmp_path / "misiones"
    huella_real = _h("atemai")

    async def escenario_mision_1():
        ctx = _ctx(tmp_path)
        await S.huella_de_apertura_de_la_mision(
            misiones=misiones, mision_id=MISION_ID, host="atemai",
            tomar_huella=_tomador_secuencia({"atemai": [huella_real]}), pausar=P.poner_pausa, pausa_ruta=ctx.pausa)
        # "muere" acá -- nunca cierra.
    asyncio.run(escenario_mision_1())

    persistida = json.loads(S.ruta_huella(misiones, MISION_ID, "atemai").read_text())
    assert persistida["pendiente"] is True

    # Misión 2, mismo host, OTRA mision_id, huella actual IGUAL a la que dejó la 1 (nada
    # cambió de verdad) -- tiene que abrir limpio (dos llamadas: la revisión de la
    # huérfana de la 1, y la propia apertura de la 2) y limpiar la marca de la 1.
    ctx2 = _ctx(tmp_path)
    tomar2 = _tomador_secuencia({"atemai": [huella_real, huella_real]})

    async def abrir_mision_2():
        return await S.huella_de_apertura_de_la_mision(
            misiones=misiones, mision_id=OTRA_MISION_ID, host="atemai", tomar_huella=tomar2,
            pausar=P.poner_pausa, pausa_ruta=ctx2.pausa)
    resultado2 = asyncio.run(abrir_mision_2())
    assert resultado2 is not None
    assert not ctx2.pausa.exists()
    persistida = json.loads(S.ruta_huella(misiones, MISION_ID, "atemai").read_text())
    assert persistida["pendiente"] is False



# --- correr_huella_por_ssh (M-4, ronda 6: "test de _principal._tomar_huella") -----------

class _ProcesoFalso:
    def __init__(self, rc, stdout=b"", stderr=b""):
        self.returncode = rc
        self._stdout, self._stderr = stdout, stderr

    async def communicate(self):
        return self._stdout, self._stderr

    def kill(self):
        pass

    async def wait(self):
        return None


def _correr_falso(rc, stdout=b"", stderr=b""):
    async def correr(*argv, **kwargs):
        return _ProcesoFalso(rc, stdout, stderr)
    return correr


def test_correr_huella_por_ssh_rc_cero_arma_la_huella():
    async def escenario():
        return await S.correr_huella_por_ssh(
            ["ssh", "fruiz@atemai"], "atemai", tope_s=5,
            correr=_correr_falso(0, stdout=b"abc  /etc/sudoers\n"))
    h = asyncio.run(escenario())
    assert h.host == "atemai" and "abc" in h.texto


def test_correr_huella_por_ssh_rc_no_cero_revienta():
    """M-4 (ronda 6): un rc != 0 NUNCA se convierte en huella -- ni vacía ni parcial."""
    async def escenario():
        return await S.correr_huella_por_ssh(
            ["ssh", "fruiz@atemai"], "atemai", tope_s=5,
            correr=_correr_falso(1, stdout=b"", stderr=b"Permission denied"))
    with pytest.raises(RuntimeError, match="huella_rc_1"):
        asyncio.run(escenario())


def test_correr_huella_por_ssh_rc_no_cero_con_salida_parcial_tambien_revienta():
    """El mutante concreto que pide matar: "quitar el chequeo del código de salida"
    dejaría pasar ESTO como una huella válida -- rc=1 pero con líneas de salida (un
    comando que empezó a andar y se cortó a mitad de camino)."""
    async def escenario():
        return await S.correr_huella_por_ssh(
            ["ssh", "fruiz@atemai"], "atemai", tope_s=5,
            correr=_correr_falso(1, stdout=b"abc  /etc/sudoers\n", stderr=b"algo fallo a mitad"))
    with pytest.raises(RuntimeError, match="huella_rc_1"):
        asyncio.run(escenario())


def test_el_mutante_sin_chequeo_de_rc_muere(monkeypatch):
    """Mata el mutante de verdad: parchea `correr_huella_por_ssh` para que se
    comporte como si el chequeo de rc no existiera, y confirma que ESE comportamiento
    es el que los dos tests de arriba no dejan pasar."""
    import jax.ejecutor.contratos.vigia_servicio as modulo

    async def version_mutada(argv, host, *, tope_s, correr=None):
        correr = correr or asyncio.create_subprocess_exec
        proc = await correr(*argv, stdout=None, stderr=None, start_new_session=True)
        salida, _ = await proc.communicate()
        # SIN el "if proc.returncode != 0: raise" -- el mutante.
        return modulo.huella.huella_desde_salida(host, salida)

    async def escenario():
        return await version_mutada(
            ["x"], "atemai", tope_s=5, correr=_correr_falso(1, stdout=b"abc  /etc/sudoers\n"))
    resultado = asyncio.run(escenario())
    assert modulo.huella.huella_valida(resultado) is True  # el mutante "logra" pasar -- por eso hay que matarlo


# --- hosts_con_sudo (M-3, ronda 4: función extraída y testable por su cuenta) ------------

def test_hosts_con_sudo_es_solo_las_remotas_de_la_mision():
    from jax.ejecutor.contratos.destinos import Host
    pol = {
        "hall9000": Host("hall9000", "172.16.20.5", 58291, "controlador", True),
        "atemai": Host("atemai", "172.16.20.11", 58291, "remota", False),
        "bridge": Host("bridge", "172.16.20.20", 58291, "remota", False),
    }
    assert S.hosts_con_sudo(frozenset({"hall9000", "atemai", "bridge"}), pol) == ("atemai", "bridge")


def test_hosts_con_sudo_omite_lo_que_no_esta_en_la_politica():
    from jax.ejecutor.contratos.destinos import Host
    pol = {"atemai": Host("atemai", "172.16.20.11", 58291, "remota", False)}
    assert S.hosts_con_sudo(frozenset({"atemai", "fantasma"}), pol) == ("atemai",)


# --- mision_id_desde_ruta (M-1, ronda 4) --------------------------------------------------

def test_mision_id_desde_ruta_le_quita_el_sufijo_de_turno():
    assert S.mision_id_desde_ruta(Path(f"/x/{MISION_ID}-t3.json")) == MISION_ID


def test_mision_id_desde_ruta_bare_se_queda_igual():
    assert S.mision_id_desde_ruta(Path(f"/x/{MISION_ID}.json")) == MISION_ID
