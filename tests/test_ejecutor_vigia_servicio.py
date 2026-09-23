# tests/test_ejecutor_vigia_servicio.py
"""El arranque de una misión: primero los contratos, después el vigía; sin contratos
nunca late (y sin latido el proxy no sirve). Fin normal: el latido se borra."""
import asyncio
import json
import os
import pwd
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

#: FIX CI (ronda 8, auditoría adversarial 2026-09-22): BLOCK-F/MAJOR-L pasaban
#: "fruiz" a `ruta_authorized_keys_admin()`/`JAX_EJECUTOR_ADMIN_USUARIO` -- las dos
#: rutas terminan en `pwd.getpwnam` real, que sólo resuelve en hall9000. En el
#: runner de CI esa cuenta no existe (medido en jax#263, 5 tests rotos). La cuenta
#: que de verdad corre el proceso existe en cualquier máquina, por definición.
_ADMIN_REAL = pwd.getpwuid(os.getuid()).pw_name


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

# Formas de hash REALES (MAJOR-6, ronda 2: huella_valida() exige 64 hex) -- no "abc"/"def".
_HASH_A = "a" * 64
_HASH_B = "b" * 64
_HASH_C = "c" * 64


def _base_completa() -> bytes:
    """MINOR (ronda 3)/RONDA 4: `huella_valida()` (llamada por `vigia_servicio.py` con
    `rutas=huella.RUTAS_DECLARADAS_POR_DEFAULT (+ rutas_extra)`) exige que CADA ruta
    declarada tenga un estado válido -- hash/D/A, nunca E ni ausencia total. Los
    TIPOS de acá reflejan lo que Fernando midió de verdad en hall9000/atemai/prod:
    `/etc/ssh/sshd_config` es un ARCHIVO (hash, nunca `D`). Escribir `D` para un
    archivo era ficción, y es justo lo que la ronda 4 vino a corregir.

    MAJOR-G (ronda 5, auditoría adversarial 2026-09-22): esta misma nota decía antes
    que `/root/.ssh/authorized_keys` "NO EXISTE en ninguna de las tres" -- ERA FALSO.
    Medido el 2026-09-22: SÍ existe (vacío, 0600, root) en hall9000, atemai y prod;
    sólo falta en `bridge`. El fixture sigue usando `A` acá (representa el estado
    SANO de `bridge`, que es la máquina donde el escenario de ronda 4 de verdad
    aplicaba) -- eso no cambia; lo que se corrige es la afirmación de a qué máquina
    describía.

    BLOCK-E, punto 2 (ronda 5): `RUTAS_DECLARADAS_POR_DEFAULT` ahora exige TAMBIÉN el
    glob de sbin (`RUTA_GLOB_SBIN_EJECUTOR`) representado -- antes quedaba fuera del
    chequeo por diseño."""
    from jax.ejecutor.contratos import huella as H
    lineas_por_ruta = {
        "/etc/sudoers": f"{_HASH_A}  /etc/sudoers",
        "/etc/sudoers.d": "D /etc/sudoers.d",
        "/etc/ssh/sshd_config": f"{_HASH_B}  /etc/ssh/sshd_config",
        "/etc/ssh/sshd_config.d": "D /etc/ssh/sshd_config.d",
        "/etc/ssh/authorized_keys.d": "D /etc/ssh/authorized_keys.d",
        "/root/.ssh/authorized_keys": "A /root/.ssh/authorized_keys",
        "/etc/ejecutor-huella": "D /etc/ejecutor-huella",
        H.RUTA_GLOB_SBIN_EJECUTOR: f"D {H.RUTA_GLOB_SBIN_EJECUTOR}",
    }
    assert set(lineas_por_ruta) == set(H.RUTAS_DECLARADAS_POR_DEFAULT), \
        "RUTAS_DECLARADAS_POR_DEFAULT cambió -- actualizar este fixture"
    lineas = [lineas_por_ruta[r] for r in H.RUTAS_DECLARADAS_POR_DEFAULT]
    return ("\n".join(lineas) + "\n").encode()


def _h(host, controles=None):
    from jax.ejecutor.contratos import huella as H
    return H.huella_desde_salida(host, _base_completa() if controles is None else controles)


def _base_completa_con_cambio() -> bytes:
    """La base completa, pero con `/root/.ssh/authorized_keys` pasando de `A`
    (ausente, confirmado) a un HASH real -- exactamente el caso que la ronda 4 señala:
    "que la ruta pase de A a existir es un CAMBIO y pausa". Sigue siendo una huella
    VÁLIDA (cada ruta declarada sigue teniendo un estado hash/D/A), pero DISTINTA de
    `_base_completa()`."""
    base = _base_completa().decode()
    sin_root = base.replace("A /root/.ssh/authorized_keys\n", "")
    return (sin_root + f"{_HASH_C}  /root/.ssh/authorized_keys\n").encode()


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


def _correr_con_huella(ctx, mision, *, hosts_con_sudo, tomar_huella, misiones, mision_id, rutas_extra=()):
    async def escenario():
        return await S.correr_mision(
            ctx, mision, latido_cada_s=0.05, lote_max=5, intervalo_s=1.0, auditar=_auditar, fin=asyncio.Event(),
            exigir=_exigir_ok, vigilar=_vigilar_noop, maquinas=MAQUINAS, hosts_con_sudo=hosts_con_sudo,
            misiones=misiones, mision_id=mision_id, tomar_huella=tomar_huella, rutas_extra=rutas_extra)
    return asyncio.run(escenario())


def test_huella_cambiada_pone_la_pausa(tmp_path):
    """Ronda 6: sin declarado -- cualquier cambio pausa, sin importar el texto de la
    misión."""
    antes = _h("atemai")
    despues = _h("atemai", controles=_base_completa_con_cambio())
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


# --- BLOCK-F (ronda 6, auditoría adversarial 2026-09-22): dos mutantes sobrevivían con
# el resto de la suite en verde -- quitar `+ rutas_extra` en `verificar_huellas_huerfanas`
# (línea 176), en `huella_de_apertura_de_la_mision` (línea 219) y en
# `_verificar_huellas_al_cierre` (línea 267); y `rutas_extra=()` en `_principal` (línea
# 458). Escenario real: con cualquiera de esos mutantes, el `authorized_keys` del
# ADMINISTRADOR -- donde vive la llave del servicio -- deja de exigirse, y una huella
# que nunca lo midió se declara completa igual. Los tres tests de abajo aíslan cada
# línea por separado: la huella "antes"/"después" es TEXTUALMENTE la misma salvo la
# ruta del admin, así que `huella.cambio()` por sí solo NUNCA alcanza para que el test
# pase -- sólo el chequeo de `rutas_extra` en `huella_valida()` lo hace. -----------------

def test_apertura_exige_la_ruta_extra_del_administrador_block_f(tmp_path):
    """Línea 219 (`huella_de_apertura_de_la_mision`): primer turno, sin marca
    persistida -- si `rutas_extra` no llegara de verdad a `huella_valida()`, una huella
    que jamás mide el `authorized_keys` del administrador pasaría por completa igual.

    MINOR (ronda 7, auditoría adversarial 2026-09-22): el mock trae una SEGUNDA
    huella (sin usar contra el código real, que corta en la primera) -- así, bajo el
    mutante que ronda 6 pedía matar, la ejecución NO revienta con un `IndexError` de
    mock agotado al llegar al cierre: falla por lo que importa, `pytest.raises` sin
    haber visto el `RuntimeError` esperado."""
    from jax.ejecutor.contratos import huella as H
    ruta_admin = H.ruta_authorized_keys_admin(_ADMIN_REAL)
    tomar_huella = _tomador_secuencia({"atemai": [_h("atemai"), _h("atemai")]})  # SIN la línea del admin
    ctx = _ctx(tmp_path)

    with pytest.raises(RuntimeError, match="huella_apertura_vacia"):
        _correr_con_huella(ctx, MISION, hosts_con_sudo=("atemai",), tomar_huella=tomar_huella,
                          misiones=tmp_path / "misiones", mision_id=MISION_ID, rutas_extra=(ruta_admin,))


def test_cierre_exige_la_ruta_extra_del_administrador_block_f(tmp_path):
    """Línea 267 (`_verificar_huellas_al_cierre`): la APERTURA sí mide el admin (para
    aislar SOLO el chequeo del cierre); el CIERRE deja de traer esa línea -- tiene que
    pausar como no-medible, no pasar como si nada."""
    from jax.ejecutor.contratos import huella as H
    ruta_admin = H.ruta_authorized_keys_admin(_ADMIN_REAL)
    con_admin = _base_completa() + f"A {ruta_admin}\n".encode()
    antes = _h("atemai", controles=con_admin)
    despues = _h("atemai")  # el cierre YA NO trae la línea del admin
    tomar_huella = _tomador_secuencia({"atemai": [antes, despues]})
    ctx = _ctx(tmp_path)

    pausas = _correr_con_huella(ctx, MISION, hosts_con_sudo=("atemai",), tomar_huella=tomar_huella,
                                misiones=tmp_path / "misiones", mision_id=MISION_ID, rutas_extra=(ruta_admin,))
    assert pausas == (("atemai", "huella_no_medible"),)


def test_huerfana_exige_la_ruta_extra_del_administrador_block_f(tmp_path):
    """Línea 176 (`verificar_huellas_huerfanas`): la marca persistida y la remedición
    son TEXTUALMENTE IGUALES (así `huella.cambio()` da `False` y no puede ser lo que
    hace fallar el test) -- ninguna de las dos trae la línea del admin. Si
    `rutas_extra` no llegara acá, esto se vería "limpio" (mismo texto, nada cambió) y
    la misión nueva abriría igual, con una deuda de vigilancia real sin cerrar.

    MINOR (ronda 7, auditoría adversarial 2026-09-22): bajo el mutante, la huérfana
    se da por resuelta y `huella_de_apertura_de_la_mision` sigue adelante a abrir SU
    PROPIA huella (esta misión, `MISION_ID`, todavía no tiene marca) -- eso consume
    una SEGUNDA llamada a `tomar_huella` que el mock (con un solo elemento) no podía
    servir, y el test moría con un `IndexError` de mock agotado en vez de fallar por
    lo que importa. El segundo elemento es una huella COMPLETA (con la línea del
    admin) -- si la ejecución llega a usarla, `resultado` deja de ser `None` y el
    `assert` de abajo lo dice con claridad."""
    from jax.ejecutor.contratos import huella as H
    misiones = tmp_path / "misiones"
    ruta_admin = H.ruta_authorized_keys_admin(_ADMIN_REAL)
    huella_vieja = _h("atemai")  # SIN la línea del admin
    H.escribir_marca(S.ruta_huella(misiones, OTRA_MISION_ID, "atemai"), H.Marca(huella=huella_vieja, estado=H.ABIERTA))

    huella_completa = _h("atemai", controles=_base_completa() + f"A {ruta_admin}\n".encode())
    tomar = _tomador_secuencia({"atemai": [_h("atemai"), huella_completa]})  # 1ra: idéntica a la persistida
    ctx = _ctx(tmp_path)

    async def abrir():
        return await S.huella_de_apertura_de_la_mision(
            misiones=misiones, mision_id=MISION_ID, host="atemai", tomar_huella=tomar,
            pausar=P.poner_pausa, pausa_ruta=ctx.pausa, rutas_extra=(ruta_admin,))
    resultado = asyncio.run(abrir())
    assert resultado is None  # no abre: la huérfana no pudo cerrarse limpia sin medir al admin
    assert ctx.pausa.exists()
    datos = json.loads(ctx.pausa.read_text())
    assert datos["motivo"] == "huella_cambio_no_declarado"
    assert "huella_de_ahora_vacia" in datos["detalle"]


def test_apertura_valida_la_marca_ya_persistida_major_l(tmp_path):
    """MAJOR-L (ronda 7, auditoría adversarial 2026-09-22): si ya hay una línea base
    persistida (la marca de ESTA `mision_id` quedó `CERRADA` al cerrar un turno
    anterior -- el caso normal de un turno 2+), se cargaba TAL CUAL, sin validar que
    siga representando las rutas EXIGIDAS de HOY -- sólo el camino "todavía no hay
    marca" (`except FileNotFoundError`) llamaba a `huella_valida()`. La persistida de
    acá es válida para `RUTAS_DECLARADAS_POR_DEFAULT` pero JAMÁS midió el
    `authorized_keys` del administrador -- si `admin_usuario` cambió entre turnos
    (`rutas_extra` ahora exige otra ruta), abrir con esa base sería abrir SIN vigilar
    lo que hoy hace falta. `tomar_huella` no debería llamarse en absoluto: con una
    marca ya persistida, no hay nada que remedir."""
    from jax.ejecutor.contratos import huella as H
    misiones = tmp_path / "misiones"
    ruta_admin = H.ruta_authorized_keys_admin(_ADMIN_REAL)
    persistida = _h("atemai")  # SIN la línea del admin
    H.escribir_marca(S.ruta_huella(misiones, MISION_ID, "atemai"), H.Marca(huella=persistida, estado=H.CERRADA))

    async def tomar_no_deberia_llamarse(host):
        raise AssertionError("con una marca ya persistida, no hay que volver a medir")

    ctx = _ctx(tmp_path)

    async def abrir():
        return await S.huella_de_apertura_de_la_mision(
            misiones=misiones, mision_id=MISION_ID, host="atemai", tomar_huella=tomar_no_deberia_llamarse,
            pausar=P.poner_pausa, pausa_ruta=ctx.pausa, rutas_extra=(ruta_admin,))

    with pytest.raises(RuntimeError, match="huella_apertura_persistida_invalida"):
        asyncio.run(abrir())


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
    cambiado_sin_declarar = _h("atemai", controles=_base_completa_con_cambio())

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

def _marca(host, estado, *, controles=None, diff=()):
    from jax.ejecutor.contratos import huella as H
    return H.Marca(huella=_h(host, controles=controles), estado=estado, diff=tuple(diff))


def test_huerfana_abierta_de_otra_mision_limpia_deja_abrir_y_la_cierra(tmp_path):
    from jax.ejecutor.contratos import huella as H
    misiones = tmp_path / "misiones"
    huella_vieja = _h("atemai")
    H.escribir_marca(S.ruta_huella(misiones, OTRA_MISION_ID, "atemai"), H.Marca(huella=huella_vieja, estado=H.ABIERTA))

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
    # La huérfana quedó CERRADA (limpia), no borrada.
    marca = H.leer_marca(S.ruta_huella(misiones, OTRA_MISION_ID, "atemai"))
    assert marca.estado == H.CERRADA


def test_huerfana_abierta_con_cambio_no_deja_abrir_y_queda_reportada(tmp_path):
    """M-1: «así un kill -9 o un reinicio no blanquean nada» -- una marca huérfana de
    OTRA misión que cambió bloquea la apertura de ÉSTA, aunque su propia huella fuera
    a salir limpia. Ronda 7: la huérfana pasa a REPORTADA, con el diff guardado."""
    from jax.ejecutor.contratos import huella as H
    misiones = tmp_path / "misiones"
    huella_vieja = _h("atemai")
    H.escribir_marca(S.ruta_huella(misiones, OTRA_MISION_ID, "atemai"), H.Marca(huella=huella_vieja, estado=H.ABIERTA))

    huella_cambiada = _h("atemai", controles=_base_completa_con_cambio())
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
    marca = H.leer_marca(S.ruta_huella(misiones, OTRA_MISION_ID, "atemai"))
    assert marca.estado == H.REPORTADA
    assert marca.diff and "authorized_keys" in marca.diff[0]


def test_huerfana_reportada_bloquea_de_una_sin_volver_a_medir(tmp_path):
    """Ronda 7, punto 2: una marca REPORTADA ya se comparó y ya pausó -- no hace falta
    (ni corresponde) volver a medir para saber que sigue bloqueando; sólo la
    aceptación explícita la resuelve."""
    from jax.ejecutor.contratos import huella as H
    misiones = tmp_path / "misiones"
    H.escribir_marca(S.ruta_huella(misiones, OTRA_MISION_ID, "atemai"),
                     _marca("atemai", H.REPORTADA, diff=("def  /root/.ssh/authorized_keys",)))

    async def tomar_no_deberia_llamarse(host):
        raise AssertionError("una REPORTADA no se vuelve a medir -- sólo se acepta")
    ctx = _ctx(tmp_path)

    with pytest.raises(S.HuellaHuerfanaNoResuelta):
        _correr_con_huella(ctx, MISION, hosts_con_sudo=("atemai",), tomar_huella=tomar_no_deberia_llamarse,
                          misiones=misiones, mision_id=MISION_ID)
    assert ctx.pausa.exists()
    datos = json.loads(ctx.pausa.read_text())
    assert datos["motivo"] == "huella_reportada_sin_aceptar"


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


def test_marca_cerrada_no_se_revisa_de_nuevo(tmp_path):
    from jax.ejecutor.contratos import huella as H
    misiones = tmp_path / "misiones"
    H.escribir_marca(S.ruta_huella(misiones, OTRA_MISION_ID, "atemai"), _marca("atemai", H.CERRADA))

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


def test_kill_9_deja_marca_abierta_que_la_proxima_mision_encuentra(tmp_path):
    """Simula un `kill -9`: la misión 1 abre (queda ABIERTA) y NUNCA cierra (el
    proceso muere entre abrir y cerrar -- no se llega a `_verificar_huellas_al_cierre`).
    La misión 2 (otra `mision_id`) tiene que encontrarla al abrir."""
    from jax.ejecutor.contratos import huella as H
    misiones = tmp_path / "misiones"
    huella_real = _h("atemai")

    async def escenario_mision_1():
        ctx = _ctx(tmp_path)
        await S.huella_de_apertura_de_la_mision(
            misiones=misiones, mision_id=MISION_ID, host="atemai",
            tomar_huella=_tomador_secuencia({"atemai": [huella_real]}), pausar=P.poner_pausa, pausa_ruta=ctx.pausa)
        # "muere" acá -- nunca cierra.
    asyncio.run(escenario_mision_1())

    marca = H.leer_marca(S.ruta_huella(misiones, MISION_ID, "atemai"))
    assert marca.estado == H.ABIERTA

    # Misión 2, mismo host, OTRA mision_id, huella actual IGUAL a la que dejó la 1 (nada
    # cambió de verdad) -- tiene que abrir limpio (dos llamadas: la revisión de la
    # huérfana de la 1, y la propia apertura de la 2) y CERRAR la marca de la 1.
    ctx2 = _ctx(tmp_path)
    tomar2 = _tomador_secuencia({"atemai": [huella_real, huella_real]})

    async def abrir_mision_2():
        return await S.huella_de_apertura_de_la_mision(
            misiones=misiones, mision_id=OTRA_MISION_ID, host="atemai", tomar_huella=tomar2,
            pausar=P.poner_pausa, pausa_ruta=ctx2.pausa)
    resultado2 = asyncio.run(abrir_mision_2())
    assert resultado2 is not None
    assert not ctx2.pausa.exists()
    marca = H.leer_marca(S.ruta_huella(misiones, MISION_ID, "atemai"))
    assert marca.estado == H.CERRADA


# --- M-3, ronda 7: el primer ssh falla y el segundo responde --------------------------

def test_el_primer_ssh_falla_y_el_segundo_responde_no_abre(tmp_path):
    """Mata el mutante `return False` -> `continue` en `verificar_huellas_huerfanas`:
    dos marcas ABIERTA huérfanas para el MISMO host (de dos misiones viejas
    distintas). La primera revisión revienta (ssh caído); la segunda, si se llegara a
    intentar, respondería LIMPIA. Con el código correcto, la función CORTA en la
    primera falla y nunca llega a la segunda -- si alguien cambiara el `return False`
    por un `continue`, la segunda taparía la primera y la misión abriría igual."""
    from jax.ejecutor.contratos import huella as H
    misiones = tmp_path / "misiones"
    primera_mision = "33333333-3333-3333-3333-333333333333"
    segunda_mision = "44444444-4444-4444-4444-444444444444"
    huella_vieja = _h("atemai")
    H.escribir_marca(S.ruta_huella(misiones, primera_mision, "atemai"), H.Marca(huella=huella_vieja, estado=H.ABIERTA))
    H.escribir_marca(S.ruta_huella(misiones, segunda_mision, "atemai"), H.Marca(huella=huella_vieja, estado=H.ABIERTA))

    llamadas = []

    async def tomar(host):
        llamadas.append(host)
        if len(llamadas) == 1:
            raise RuntimeError("ssh caído")
        return huella_vieja  # limpia -- pero NO debería llegar a usarse
    ctx = _ctx(tmp_path)

    ok = asyncio.run(S.verificar_huellas_huerfanas(misiones, "atemai", tomar_huella=tomar,
                                                    pausar=P.poner_pausa, pausa_ruta=ctx.pausa))
    assert ok is False
    assert len(llamadas) == 1  # NUNCA llega a la segunda marca
    assert ctx.pausa.exists()
    datos = json.loads(ctx.pausa.read_text())
    assert datos["motivo"] == "huella_no_medible"
    # Las dos marcas siguen ABIERTA -- ninguna se tocó (la función cortó antes).
    assert H.leer_marca(S.ruta_huella(misiones, primera_mision, "atemai")).estado == H.ABIERTA
    assert H.leer_marca(S.ruta_huella(misiones, segunda_mision, "atemai")).estado == H.ABIERTA


def test_el_mutante_continue_en_vez_de_return_false_deja_pasar_la_mision(tmp_path):
    """Prueba el mutante DE VERDAD: una versión de `verificar_huellas_huerfanas` que
    hace `continue` en vez de `return False` SÍ dejaría abrir la misión -- por eso el
    test de arriba, que exige `ok is False` y una sola llamada, es el que lo mata."""
    from jax.ejecutor.contratos import huella as H

    async def version_mutada(misiones, host, *, tomar_huella, pausar, pausa_ruta):
        for ruta in sorted(Path(misiones).glob(f"*/huella/{host}.json")):
            marca = H.leer_marca(ruta)
            if marca.estado != H.ABIERTA:
                continue
            try:
                despues = await tomar_huella(host)
            except Exception:  # fail-soft: ESTE except es el MUTANTE bajo prueba (debería ser `return False`, ver el test de arriba que lo mata)
                continue
            if H.cambio(marca.huella, despues):
                continue
        return True

    misiones = tmp_path / "misiones"
    primera_mision = "33333333-3333-3333-3333-333333333333"
    huella_vieja = _h("atemai")
    H.escribir_marca(H.ruta_huella(misiones, primera_mision, "atemai"), H.Marca(huella=huella_vieja, estado=H.ABIERTA))

    async def tomar(host):
        raise RuntimeError("ssh caído")

    ok = asyncio.run(version_mutada(misiones, "atemai", tomar_huella=tomar, pausar=P.poner_pausa,
                                    pausa_ruta=misiones / "PAUSA"))
    assert ok is True  # el mutante "logra" abrir -- por eso hay que matarlo con el test de arriba



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

    _HASH = "a" * 64  # forma de hash real (MAJOR-6, ronda 2): huella_valida() exige 64 hex

    async def escenario():
        return await version_mutada(
            ["x"], "atemai", tope_s=5, correr=_correr_falso(1, stdout=f"{_HASH}  /etc/sudoers\n".encode()))
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


# --- BLOCK-F, línea 458 (`_principal`, ronda 6): el mutante `rutas_extra=()` -- en vez
# de `(huella.ruta_authorized_keys_admin(admin_usuario),)` -- sobrevivía porque nada
# corría `_principal` de punta a punta y miraba qué recibía `correr_mision`. Acá se
# monkeypatchea SOLO el borde caro (conexión a la base, resolución de faceta C5, y
# `correr_mision` mismo -- que ya tiene sus propios tests de comportamiento arriba) --
# el resto de `_principal` (parseo del entorno, cómputo de `rutas_extra`) corre DE
# VERDAD, sin atajos. -----------------------------------------------------------------

class _ConexionFalsa:
    async def __aenter__(self):
        return object()

    async def __aexit__(self, *exc):
        return False


def _entorno_principal(tmp_path, *, admin=_ADMIN_REAL):
    politica_ruta = tmp_path / "politica.json"
    def _bash(comando):
        return {"tool_name": "Bash", "tool_input": {"command": comando}}

    doc = {
        "version": 1, "generada_at": "2026-09-22T00:00:00+00:00",
        "hosts": [{"nombre": "hall9000", "ip": "127.0.0.1", "puerto": 58291, "rol": "hypervisor",
                  "es_local": True},
                 {"nombre": "atemai", "ip": "192.0.2.11", "puerto": 58291, "rol": "desarrollo",
                  "es_local": False}],
        "reglas": [
            {"id": 1, "codigo": "canario_c1", "tipo": "prohibido", "herramientas": "Bash", "campo": "command",
             "patron": "ejecutor-canario-c1", "ambito_hosts": [], "ambito_roles": [], "es_canario": True,
             "ejemplos_coincide": [_bash("echo ejecutor-canario-c1")], "ejemplos_no_coincide": [_bash("echo x")]},
        ],
        "respaldos": {}, "c2_edad_max_s": 86400,
    }
    from jax.ejecutor.contratos import politica as POL
    politica_ruta.write_text(json.dumps(POL.firmar(doc)))
    registro = tmp_path / "r.jsonl"
    Registro(registro).cerrar()
    return {
        "JAX_EJECUTOR_CUENTA": "axioma", "JAX_EJECUTOR_SSH_PUERTO": "58291",
        "JAX_EJECUTOR_CONTROLADOR_LLAVE": "/k", "JAX_EJECUTOR_NODE_BIN": "/n", "JAX_EJECUTOR_LIB": "/opt/lib",
        "JAX_EJECUTOR_CUENTA_HOME": "/home/axioma", "JAX_EJECUTOR_POLITICA": str(politica_ruta),
        "JAX_EJECUTOR_CANARIO_PUERTO": "18436", "JAX_EJECUTOR_REGISTRO": str(registro),
        "JAX_PROXY_CARRIL_PUERTO": "18435", "JAX_EJECUTOR_CERCO_SONDAS": "7777,11434",
        "JAX_EJECUTOR_FRENO_ESTADO": str(tmp_path / "e.json"), "JAX_EJECUTOR_LLAVES_ROOT": str(tmp_path / "llaves"),
        "JAX_EJECUTOR_GANCHO_TOPE_S": "10", "JAX_EJECUTOR_PAUSA": str(tmp_path / "PAUSA"),
        "JAX_EJECUTOR_VIGIA_LATIDO": str(tmp_path / "latido"), "JAX_EJECUTOR_VIGIA_LATIDO_MAX_S": "30",
        "JAX_EJECUTOR_VIGIA_LATIDO_CADA_S": "1",
        "JAX_EJECUTOR_ADMIN_USUARIO": admin, "JAX_EJECUTOR_HUELLA_LLAVE": str(tmp_path / "id_huella"),
        "JAX_EJECUTOR_HUELLA_KNOWN_HOSTS": str(tmp_path / "kh"), "JAX_EJECUTOR_MISIONES": str(tmp_path / "misiones"),
    }


def test_principal_pasa_la_ruta_extra_del_administrador_de_verdad_block_f(tmp_path, monkeypatch):
    import jacobs.store as jstore
    from jax.ejecutor.contratos import eleccion_c5, huella as H

    admin = _ADMIN_REAL
    env = _entorno_principal(tmp_path, admin=admin)
    for k, v in env.items():
        monkeypatch.setenv(k, v)

    ruta_mision = tmp_path / f"{MISION_ID}.json"
    ruta_mision.write_text(json.dumps({"mision": "uptime de atemai", "hosts": ["atemai"]}))

    monkeypatch.setattr(jstore, "conexion", lambda **kw: _ConexionFalsa())

    async def _leer_config_falso(conn):
        return eleccion_c5.ConfigC5(
            cerebro_faceta="x", auditor_faceta="y", auditor_faceta_local="z",
            lote_max=5, intervalo_s=1.0, max_tokens=100,
            admite_datos_de_clientes=False, admite_mismo_proveedor=False)

    async def _elegir_falso(conn, *, cfg, hosts_mision, resolve_facet):
        return ("faceta-fake", None, None)

    monkeypatch.setattr(eleccion_c5, "leer_config", _leer_config_falso)
    monkeypatch.setattr(eleccion_c5, "elegir_y_resolver_auditor", _elegir_falso)

    llamadas = {}

    async def _correr_mision_falso(*a, **kw):
        llamadas["rutas_extra"] = kw.get("rutas_extra")
        return ()

    monkeypatch.setattr(S, "correr_mision", _correr_mision_falso)

    rc = asyncio.run(S._principal(ruta_mision))

    assert rc == 0
    assert llamadas["rutas_extra"] == (H.ruta_authorized_keys_admin(admin),)
