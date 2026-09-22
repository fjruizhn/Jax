# tests/test_ejecutor_vigia_servicio.py
"""El arranque de una misión: primero los contratos, después el vigía; sin contratos
nunca late (y sin latido el proxy no sirve). Fin normal: el latido se borra."""
import asyncio
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


def test_la_unidad_lanza_este_modulo_con_la_mision_de_su_instancia():
    unidad = (Path(__file__).resolve().parents[1] / "ops" / "ejecutor" / "ejecutor-vigia@.service").read_text()
    assert "-m jax.ejecutor.contratos.vigia_servicio ${JAX_EJECUTOR_MISIONES}/%i.json" in unidad
    assert "Restart=no" in unidad and "User=jaxsvc" in unidad and "KillSignal=SIGTERM" in unidad


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


def test_sin_tomar_huella_no_se_toma_ninguna(tmp_path):
    """`tomar_huella=None` (el default): cero llamadas, cero pausa -- los llamadores que
    no la necesitan no cambian de comportamiento."""
    ctx = _ctx(tmp_path)
    llamadas = []

    async def tomar_huella_falsa(h):
        llamadas.append(h)

    async def escenario():
        await S.correr_mision(ctx, MISION, latido_cada_s=0.05, lote_max=5, intervalo_s=1.0, auditar=_auditar,
                              fin=asyncio.Event(), exigir=_exigir_ok, vigilar=_vigilar_noop, maquinas=MAQUINAS,
                              hosts_con_sudo=("atemai",))  # sin tomar_huella: no debería usarse
    asyncio.run(escenario())
    assert llamadas == []
    assert not ctx.pausa.exists()


def _huella_secuencia(mapa_por_llamada):
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


def test_huella_cambiada_y_no_declarada_pone_la_pausa(tmp_path):
    from jax.ejecutor.contratos import huella as H
    ctx = _ctx(tmp_path)
    antes = H.huella_desde_salida("atemai", b"abc  /etc/sudoers.d/50-ejecutor-axioma-registro\n")
    despues = H.huella_desde_salida(
        "atemai", b"abc  /etc/sudoers.d/50-ejecutor-axioma-registro\n"
                  b"def  /root/.ssh/authorized_keys\n")
    tomar_huella = _huella_secuencia({"atemai": [antes, despues]})

    async def escenario():
        await S.correr_mision(ctx, MISION, latido_cada_s=0.05, lote_max=5, intervalo_s=1.0, auditar=_auditar,
                              fin=asyncio.Event(), exigir=_exigir_ok, vigilar=_vigilar_noop, maquinas=MAQUINAS,
                              hosts_con_sudo=("atemai",), tomar_huella=tomar_huella)
    asyncio.run(escenario())
    assert ctx.pausa.exists()
    import json
    datos = json.loads(ctx.pausa.read_text())
    assert datos["origen"] == "huella" and datos["motivo"] == "huella_cambio_no_declarado"
    assert datos["host"] == "atemai"
    assert "authorized_keys" in datos["detalle"][0]


def test_huella_cambiada_pero_declarada_en_la_mision_no_pausa(tmp_path):
    from jax.ejecutor.contratos import huella as H
    ctx = _ctx(tmp_path)
    antes = H.huella_desde_salida("hall9000", b"abc  /etc/sudoers.d/50-ejecutor-axioma-registro\n")
    despues = H.huella_desde_salida(
        "hall9000", b"abc  /etc/sudoers.d/50-ejecutor-axioma-registro\n"
                    b"def  /etc/systemd/system/mi-servicio.service\n")
    tomar_huella = _huella_secuencia({"hall9000": [antes, despues]})
    mision = S.Mision("Crear /etc/systemd/system/mi-servicio.service en hall9000", frozenset({"hall9000"}))

    async def escenario():
        await S.correr_mision(ctx, mision, latido_cada_s=0.05, lote_max=5, intervalo_s=1.0, auditar=_auditar,
                              fin=asyncio.Event(), exigir=_exigir_ok, vigilar=_vigilar_noop, maquinas=MAQUINAS,
                              hosts_con_sudo=("hall9000",), tomar_huella=tomar_huella)
    asyncio.run(escenario())
    assert not ctx.pausa.exists()


def test_sin_cambio_en_la_huella_no_pausa(tmp_path):
    from jax.ejecutor.contratos import huella as H
    ctx = _ctx(tmp_path)
    igual = H.huella_desde_salida("atemai", b"abc  /etc/sudoers.d/50-ejecutor-axioma-registro\n")
    tomar_huella = _huella_secuencia({"atemai": [igual, igual]})

    async def escenario():
        await S.correr_mision(ctx, MISION, latido_cada_s=0.05, lote_max=5, intervalo_s=1.0, auditar=_auditar,
                              fin=asyncio.Event(), exigir=_exigir_ok, vigilar=_vigilar_noop, maquinas=MAQUINAS,
                              hosts_con_sudo=("atemai",), tomar_huella=tomar_huella)
    asyncio.run(escenario())
    assert not ctx.pausa.exists()


def test_huella_de_cierre_ilegible_no_pausa_por_eso_solo(tmp_path):
    """Fail-soft: perder la VERIFICACIÓN (ssh caído, timeout) no es, por sí solo, un
    hallazgo -- eso ya lo cubre C6 (la máquina tiene que estar viva)."""
    from jax.ejecutor.contratos import huella as H
    ctx = _ctx(tmp_path)
    antes = H.huella_desde_salida("atemai", b"abc  /etc/sudoers.d/50-ejecutor-axioma-registro\n")
    tomar_huella = _huella_secuencia({"atemai": [antes, RuntimeError("ssh caído")]})

    async def escenario():
        await S.correr_mision(ctx, MISION, latido_cada_s=0.05, lote_max=5, intervalo_s=1.0, auditar=_auditar,
                              fin=asyncio.Event(), exigir=_exigir_ok, vigilar=_vigilar_noop, maquinas=MAQUINAS,
                              hosts_con_sudo=("atemai",), tomar_huella=tomar_huella)
    asyncio.run(escenario())
    assert not ctx.pausa.exists()
