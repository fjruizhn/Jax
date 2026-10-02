"""El Faro (reauditoria): la cadena de la bitacora sin necesidad de una base real. Ancla externa (publicar el
ultimo hash para que truncar la cola se note), plazo al INSERT (nada colgado detras del Lock) y el punto de
entrada del servicio, que NO arranca sin bitacora durable."""
from __future__ import annotations

import asyncio
import os
import time

import pytest

from jax.faro import paquete
from jax.faro.bitacora import Bitacora
from jax.faro.bitacora_db import ConfigBitacoraDB, EmisorTabla, publicar_anclas_periodicamente, verificar_cadena
from jax.faro.config import ConfigFaro, ConfigFaroInvalida
from jax.faro.servicio import Servicio, arrancar, main
from tests._faro_falsos import FalsoPool
from tests._faro_utils import _git, cliente_por_rele, corre, ejecucion, repo_de_juguete


def _reg(i):
    return {"evento": "llamada", "momento": 1000.0 + i, "run_id": "r", "id_correlacion": "c", "decision": "permitido", "n": i}


async def _emitir(pool, n, **kw):
    e = EmisorTabla(pool, **kw)
    for i in range(n):
        await e(_reg(i))
    return e


# --------------------------------------------------------------------------- #
# ancla externa                                                               #
# --------------------------------------------------------------------------- #

def test_el_emisor_expone_su_ancla_el_ultimo_hash_de_su_cadena():
    async def caso():
        pool = FalsoPool()
        e = EmisorTabla(pool)
        assert e.ancla() is None
        await e(_reg(0))
        await e(_reg(1))
        return e.ancla(), pool.filas
    ancla, filas = corre(caso())
    assert ancla == {"cadena_id": filas[-1]["cadena_id"], "seq": filas[-1]["seq"], "hash": filas[-1]["hash"]}


def test_con_el_ancla_publicada_truncar_la_cola_se_nota():
    async def caso():
        pool = FalsoPool()
        e = await _emitir(pool, 6)
        return e.ancla(), pool.filas
    ancla, filas = corre(caso())
    assert verificar_cadena(filas, anclas=[ancla]) == []
    # quien puede borrar la cola de la cadena la deja "valida" por si sola...
    truncada = [f for f in filas if f["seq"] <= ancla["seq"] - 2]
    assert verificar_cadena(truncada) == []
    # ...pero no contra el ancla publicada
    assert any(p.codigo == "cola_truncada" for p in verificar_cadena(truncada, anclas=[ancla]))


def test_un_ancla_que_no_coincide_o_de_una_cadena_ausente_se_detecta():
    async def caso():
        pool = FalsoPool()
        e = await _emitir(pool, 4)
        return e.ancla(), pool.filas
    ancla, filas = corre(caso())
    assert any(p.codigo == "ancla_no_coincide" for p in verificar_cadena(filas, anclas=[{**ancla, "hash": "f" * 64}]))
    assert any(p.codigo == "cadena_ausente" for p in verificar_cadena(filas, anclas=[{**ancla, "cadena_id": "9" * 32}]))


def test_las_anclas_se_publican_periodicamente_solo_cuando_cambian():
    publicadas = []

    async def caso():
        pool = FalsoPool()
        e = EmisorTabla(pool)
        tarea = asyncio.create_task(publicar_anclas_periodicamente(e, lambda a: publicadas.append(a), intervalo_s=0.05))
        await asyncio.sleep(0.12)
        assert publicadas == []                         # sin cadena no hay ancla
        await e(_reg(0))
        await asyncio.sleep(0.2)
        n1 = len(publicadas)
        await asyncio.sleep(0.2)
        assert len(publicadas) == n1 == 1               # no repite el mismo hash
        await e(_reg(1))
        await asyncio.sleep(0.2)
        tarea.cancel()
        await asyncio.gather(tarea, return_exceptions=True)
        return e.ancla()
    ultima = corre(caso())
    assert publicadas[-1] == ultima and len(publicadas) == 2


def test_un_publicador_que_falla_no_mata_la_publicacion():
    llamadas = []

    async def publicar(a):
        llamadas.append(a)
        if len(llamadas) == 1:
            raise OSError("destino de la ancla caido")

    async def caso():
        e = EmisorTabla(FalsoPool())
        tarea = asyncio.create_task(publicar_anclas_periodicamente(e, publicar, intervalo_s=0.05))
        await e(_reg(0))
        await asyncio.sleep(0.4)
        tarea.cancel()
        await asyncio.gather(tarea, return_exceptions=True)
    corre(caso())
    assert len(llamadas) >= 2                             # reintento tras el fallo


# --------------------------------------------------------------------------- #
# plazo al INSERT                                                             #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("modo", ["colgar", "colgar_acquire"])
def test_un_insert_colgado_falla_con_plazo_y_no_deja_a_los_demas_detras_del_lock(modo):
    async def caso():
        pool = FalsoPool(modo)
        e = EmisorTabla(pool, plazo_s=0.3)
        t0 = time.monotonic()
        resultados = await asyncio.gather(*(e(_reg(i)) for i in range(5)), return_exceptions=True)
        return time.monotonic() - t0, resultados, e, pool
    dt, resultados, e, pool = corre(caso())
    assert all(isinstance(r, TimeoutError) for r in resultados)
    assert dt < 2.0, f"los 5 esperaron {dt:.1f}s detras del Lock"          # cada uno con su plazo, no en fila de plazos


def test_tras_un_plazo_vencido_el_emisor_se_recupera_con_una_cadena_nueva():
    async def caso():
        pool = FalsoPool("colgar")
        e = EmisorTabla(pool, plazo_s=0.2)
        with pytest.raises(TimeoutError):
            await e(_reg(0))
        pool.modo = "ok"
        await e(_reg(1))
        return pool.filas
    filas = corre(caso())
    assert verificar_cadena(filas) == []


# --------------------------------------------------------------------------- #
# el punto de entrada del servicio falla cerrado                              #
# --------------------------------------------------------------------------- #

@pytest.fixture
def entorno(tmp_path):
    repo = repo_de_juguete(tmp_path)
    sha = _git(repo, "rev-parse", "HEAD")
    cfg = ConfigFaro(repo=repo, sha=sha, destino=tmp_path / "eco", uid_duenio=os.getuid())
    paquete.construir_paquete(cfg)
    d = tmp_path / "run"
    d.mkdir(mode=0o700)
    return {"JAX_FARO_REPO": str(repo), "JAX_FARO_SHA": sha, "JAX_FARO_ECOSISTEMA_DIR": str(tmp_path / "eco"),
            "JAX_FARO_DUENIO_UID": str(os.getuid()), "JAX_FARO_SOCKET_DIR": str(d),
            "JAX_FARO_BITACORA_DB_HOST": "127.0.0.1", "JAX_FARO_BITACORA_DB_PORT": "3306", "JAX_FARO_BITACORA_DB_USER": "u",
            "JAX_FARO_BITACORA_DB_PASSWORD": "p", "JAX_FARO_BITACORA_DB_NAME": "b"}


def _fabrica(pool=None, falla=None):
    llamadas = []

    async def crear_pool(cfg):
        llamadas.append(cfg)
        if falla:
            raise falla
        return pool or FalsoPool()
    crear_pool.llamadas = llamadas
    return crear_pool


@pytest.mark.parametrize("falta", [
    "JAX_FARO_BITACORA_DB_HOST", "JAX_FARO_BITACORA_DB_PORT", "JAX_FARO_BITACORA_DB_USER",
    "JAX_FARO_BITACORA_DB_PASSWORD", "JAX_FARO_BITACORA_DB_NAME"])
def test_sin_la_configuracion_de_la_bitacora_durable_el_servicio_no_arranca(entorno, falta):
    entorno.pop(falta)
    crear = _fabrica()
    with pytest.raises(ConfigFaroInvalida, match=falta):
        corre(arrancar(entorno, crear_pool=crear, solo_pruebas_mismo_uid=True))
    assert crear.llamadas == []


def test_sin_la_configuracion_del_resto_tampoco_arranca(entorno):
    for falta in ("JAX_FARO_REPO", "JAX_FARO_SHA", "JAX_FARO_ECOSISTEMA_DIR", "JAX_FARO_SOCKET_DIR"):
        e = {k: v for k, v in entorno.items() if k != falta}
        with pytest.raises(ConfigFaroInvalida, match=falta):
            corre(arrancar(e, crear_pool=_fabrica(), solo_pruebas_mismo_uid=True))


def test_con_la_base_inalcanzable_el_servicio_no_arranca(entorno):
    with pytest.raises(OSError):
        corre(arrancar(entorno, crear_pool=_fabrica(falla=OSError("sin ruta a la base")), solo_pruebas_mismo_uid=True))


def test_si_la_sonda_de_insercion_falla_el_servicio_no_arranca_y_cierra_el_pool(entorno):
    pool = FalsoPool("fallar")
    with pytest.raises(OSError):
        corre(arrancar(entorno, crear_pool=_fabrica(pool), solo_pruebas_mismo_uid=True))
    assert pool.cerrado


def test_el_servicio_no_arranca_como_root(entorno, monkeypatch):
    monkeypatch.setattr(os, "geteuid", lambda: 0)
    crear = _fabrica()
    with pytest.raises(ConfigFaroInvalida, match="root"):
        corre(arrancar(entorno, crear_pool=crear, solo_pruebas_mismo_uid=True))
    assert crear.llamadas == []


def test_un_paquete_que_no_verifica_impide_arrancar_antes_de_tocar_la_base(entorno, tmp_path):
    raiz = tmp_path / "eco" / entorno["JAX_FARO_SHA"]
    (raiz / "skills/alfa/SKILL.md").write_text("alterado")
    crear = _fabrica()
    with pytest.raises(paquete.PaqueteNoVerifica):
        corre(arrancar(entorno, crear_pool=crear, solo_pruebas_mismo_uid=True))
    assert crear.llamadas == []


def test_arrancado_el_servicio_su_bitacora_va_a_la_tabla_y_sus_puertos_la_usan(entorno):
    async def caso():
        pool = FalsoPool()
        s = await arrancar(entorno, crear_pool=_fabrica(pool), solo_pruebas_mismo_uid=True)
        assert isinstance(s, Servicio) and isinstance(s.emisor, EmisorTabla)
        assert isinstance(s.bitacora, Bitacora) and s.bitacora.emisores[0] is s.emisor
        assert [f["evento"] for f in pool.filas][:2] == ["inicio_cadena", "servicio_iniciado"]     # la sonda
        async with s.crear_puerto(ejecucion()) as srv, cliente_por_rele(srv) as c:
            await c.call_tool("skills.leer", {"nombre": "alfa"})
        await s.cerrar()
        assert pool.cerrado
        return pool.filas
    filas = corre(caso())
    assert sum(1 for f in filas if f["evento"] == "llamada") >= 3 and verificar_cadena(filas) == []


def test_main_sale_con_codigo_2_y_sin_traza_si_falta_la_bitacora(entorno, capsys):
    entorno.pop("JAX_FARO_BITACORA_DB_HOST")
    assert main([], env=entorno) == 2
    err = capsys.readouterr().err
    assert "JAX_FARO_BITACORA_DB_HOST" in err and "Traceback" not in err
