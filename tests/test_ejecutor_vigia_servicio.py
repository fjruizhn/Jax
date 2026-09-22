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


MISION_ID = "11111111-1111-1111-1111-111111111111"


def _log(sha="a" * 64):
    return f"LOG /var/log/sudo-axioma.log 1 1 {sha}\n".encode()


def _apertura(host, controles=b"", log=None):
    from jax.ejecutor.contratos import huella as H
    return H.huella_de_apertura_desde_salida(
        host, (log or _log()) + b"===CONTROLES===\n" + controles + b"===PERSISTENCIA===\n===FIN===\n")


def _cierre(host, controles=b"", persistencia=b"", log=None):
    from jax.ejecutor.contratos import huella as H
    return H.huella_de_cierre_desde_salida(
        host, (log or _log()) + b"===CONTROLES===\n" + controles + b"===PERSISTENCIA===\n" + persistencia
        + b"===FIN===\n")


def test_sin_tomar_huella_apertura_no_se_toma_ninguna(tmp_path):
    """`tomar_huella_apertura=None` (el default): cero llamadas, cero pausa -- los
    llamadores que no la necesitan no cambian de comportamiento."""
    ctx = _ctx(tmp_path)
    llamadas = []

    async def tomar_huella_falsa(h):
        llamadas.append(h)

    async def escenario():
        return await S.correr_mision(ctx, MISION, latido_cada_s=0.05, lote_max=5, intervalo_s=1.0, auditar=_auditar,
                                     fin=asyncio.Event(), exigir=_exigir_ok, vigilar=_vigilar_noop, maquinas=MAQUINAS,
                                     hosts_con_sudo=("atemai",))  # sin tomar_huella_*: no debería usarse
    pausas = asyncio.run(escenario())
    assert llamadas == [] and pausas == ()
    assert not ctx.pausa.exists()


def _tomadores(apertura_por_host: dict, cierre_por_host: dict):
    async def tomar_apertura(host):
        v = apertura_por_host[host]
        if isinstance(v, Exception):
            raise v
        return v

    async def tomar_cierre(host, *, tamano_apertura_log):
        v = cierre_por_host[host]
        if isinstance(v, Exception):
            raise v
        return v
    return tomar_apertura, tomar_cierre


def _correr_con_huella(ctx, mision, *, hosts_con_sudo, tomar_apertura, tomar_cierre, misiones, mision_id):
    async def escenario():
        return await S.correr_mision(
            ctx, mision, latido_cada_s=0.05, lote_max=5, intervalo_s=1.0, auditar=_auditar, fin=asyncio.Event(),
            exigir=_exigir_ok, vigilar=_vigilar_noop, maquinas=MAQUINAS, hosts_con_sudo=hosts_con_sudo,
            misiones=misiones, mision_id=mision_id, tomar_huella_apertura=tomar_apertura,
            tomar_huella_cierre=tomar_cierre)
    return asyncio.run(escenario())


def test_huella_cambiada_y_no_declarada_pone_la_pausa(tmp_path):
    antes = _apertura("atemai")
    despues = _cierre("atemai", controles=b"def  /root/.ssh/authorized_keys\n")
    tomar_apertura, tomar_cierre = _tomadores({"atemai": antes}, {"atemai": despues})
    ctx = _ctx(tmp_path)

    pausas = _correr_con_huella(ctx, MISION, hosts_con_sudo=("atemai",), tomar_apertura=tomar_apertura,
                                tomar_cierre=tomar_cierre, misiones=tmp_path / "misiones", mision_id=MISION_ID)
    assert pausas == (("atemai", "huella_cambio_no_declarado"),)
    assert ctx.pausa.exists()
    import json
    datos = json.loads(ctx.pausa.read_text())
    assert datos["origen"] == "huella" and datos["motivo"] == "huella_cambio_no_declarado"
    assert datos["host"] == "atemai"
    assert "authorized_keys" in datos["detalle"][0]


def test_huella_cambiada_pero_declarada_en_la_mision_no_pausa(tmp_path):
    antes = _apertura("hall9000")
    despues = _cierre("hall9000", controles=b"def  /etc/crontab\n")
    tomar_apertura, tomar_cierre = _tomadores({"hall9000": antes}, {"hall9000": despues})
    mision = S.Mision("Voy a tocar /etc/crontab para el cliente en hall9000", frozenset({"hall9000"}))
    ctx = _ctx(tmp_path)

    pausas = _correr_con_huella(ctx, mision, hosts_con_sudo=("hall9000",), tomar_apertura=tomar_apertura,
                                tomar_cierre=tomar_cierre, misiones=tmp_path / "misiones", mision_id=MISION_ID)
    assert pausas == ()
    assert not ctx.pausa.exists()


def test_sin_cambio_en_la_huella_no_pausa(tmp_path):
    antes = _apertura("atemai")
    despues = _cierre("atemai")
    tomar_apertura, tomar_cierre = _tomadores({"atemai": antes}, {"atemai": despues})
    ctx = _ctx(tmp_path)

    pausas = _correr_con_huella(ctx, MISION, hosts_con_sudo=("atemai",), tomar_apertura=tomar_apertura,
                                tomar_cierre=tomar_cierre, misiones=tmp_path / "misiones", mision_id=MISION_ID)
    assert pausas == ()
    assert not ctx.pausa.exists()


def test_persistencia_cambiada_y_no_declarada_informa_pero_no_pausa(tmp_path):
    antes = _apertura("atemai")
    despues = _cierre("atemai", persistencia=b"abc  /etc/systemd/system/cliente.service\n")
    tomar_apertura, tomar_cierre = _tomadores({"atemai": antes}, {"atemai": despues})
    ctx = _ctx(tmp_path)

    pausas = _correr_con_huella(ctx, MISION, hosts_con_sudo=("atemai",), tomar_apertura=tomar_apertura,
                                tomar_cierre=tomar_cierre, misiones=tmp_path / "misiones", mision_id=MISION_ID)
    assert pausas == ()
    assert not ctx.pausa.exists()


def test_huella_de_cierre_ilegible_ahora_PAUSA_ronda4(tmp_path):
    """Ronda 4 (M-1): «el CIERRE falla cerrado» -- invierte el fail-soft de la ronda 3.
    Perder la MEDICIÓN de cierre (ssh caído, timeout) es, por sí sola, un hallazgo
    `huella_no_medible` que pausa, visible en el resultado del turno."""
    antes = _apertura("atemai")
    tomar_apertura, tomar_cierre = _tomadores({"atemai": antes}, {"atemai": RuntimeError("ssh caído")})
    ctx = _ctx(tmp_path)

    pausas = _correr_con_huella(ctx, MISION, hosts_con_sudo=("atemai",), tomar_apertura=tomar_apertura,
                                tomar_cierre=tomar_cierre, misiones=tmp_path / "misiones", mision_id=MISION_ID)
    assert pausas == (("atemai", "huella_no_medible"),)
    assert ctx.pausa.exists()
    import json
    datos = json.loads(ctx.pausa.read_text())
    assert datos["origen"] == "huella" and datos["motivo"] == "huella_no_medible"
    assert datos["host"] == "atemai"


def test_apertura_de_apertura_que_revienta_no_abre_la_mision(tmp_path):
    """M-3: `tomar_huella_apertura` que revienta se PROPAGA -- no hay try/except que la
    trague. Sin huella de apertura, la misión no debe seguir."""
    tomar_apertura, tomar_cierre = _tomadores({"atemai": RuntimeError("sin ssh")}, {})
    ctx = _ctx(tmp_path)

    with pytest.raises(RuntimeError):
        _correr_con_huella(ctx, MISION, hosts_con_sudo=("atemai",), tomar_apertura=tomar_apertura,
                          tomar_cierre=tomar_cierre, misiones=tmp_path / "misiones", mision_id=MISION_ID)


def test_turno_n_mas_1_no_blanquea_un_cambio_del_turno_n(tmp_path):
    """M-1 (ronda 4): la línea base es la de la APERTURA DE LA MISIÓN, persistida por
    `mision_id` -- no la del turno anterior. El turno 1 fija la huella de apertura; el
    turno 2 (un `correr_mision` DISTINTO, misma `mision_id`) tiene que seguir
    comparando contra esa MISMA apertura, aunque para el turno 2 un tomador ingenuo
    "vería" el estado ya cambiado como su propio punto de partida."""
    misiones = tmp_path / "misiones"
    original = _apertura("atemai")
    cambiado_sin_declarar = _cierre("atemai", controles=b"def  /root/.ssh/authorized_keys\n")

    # Turno 1: abre limpio, cierra limpio (nadie detecta nada -- el cambio pasa DESPUÉS).
    ta1, tc1 = _tomadores({"atemai": original}, {"atemai": _cierre("atemai")})
    ctx1 = _ctx(tmp_path)
    pausas1 = _correr_con_huella(ctx1, MISION, hosts_con_sudo=("atemai",), tomar_apertura=ta1, tomar_cierre=tc1,
                                 misiones=misiones, mision_id=MISION_ID)
    assert pausas1 == ()

    # Turno 2: si `tomar_huella_apertura` se volviera a invocar y devolviera el estado YA
    # cambiado, un baseline "del turno anterior" lo tomaría como normal. Para probar que
    # NO pasa, el tomador de apertura del turno 2 directamente REVIENTA si se le llama --
    # la única forma de que el turno 2 pase es que cargue la apertura PERSISTIDA del
    # turno 1 sin volver a invocar `tomar_huella_apertura`.
    async def apertura_no_deberia_llamarse(host):
        raise AssertionError("turno 2 no debe volver a tomar la apertura: tiene que cargar la persistida")

    async def cierre_turno2(host, *, tamano_apertura_log):
        return cambiado_sin_declarar
    ctx2 = _ctx(tmp_path)
    pausas2 = _correr_con_huella(ctx2, MISION, hosts_con_sudo=("atemai",), tomar_apertura=apertura_no_deberia_llamarse,
                                 tomar_cierre=cierre_turno2, misiones=misiones, mision_id=MISION_ID)
    assert pausas2 == (("atemai", "huella_cambio_no_declarado"),)
    assert ctx2.pausa.exists()


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
