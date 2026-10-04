from types import SimpleNamespace

import pytest

from jax.faro.config import ConfigFaroInvalida, ConfigMemoria
from jax.faro.herramientas.memoria import AdaptadorMemoria, MemoriaNoDisponible
from tests._faro_utils import ejecucion
from jax.faro.identidad import Identidad


def identidad(usuario="usuario-socket", tenant="tenant-socket"):
    return Identidad(ejecucion(usuario=usuario, tenant=tenant), "conn", 1, 2, 3)


def envelope(memory_id, payload):
    return SimpleNamespace(
        identity=SimpleNamespace(memory_id=memory_id),
        revision=SimpleNamespace(revision_id=f"rev-{memory_id}", payload=payload,
                                 provenance_status="LEGACY_UNVERIFIED"),
    )


class Reader:
    def __init__(self, entries):
        self.entries = entries
        self.scope = None
        self.limit = None

    async def retrieve(self, scope, *, limit):
        self.scope, self.limit = scope, limit
        return self.entries


@pytest.mark.asyncio
async def test_buscar_usa_tenant_y_usuario_fijados_por_el_socket_y_marca_fuente_no_confiable():
    reader = Reader([envelope("m1", "dato de prueba"), envelope("m2", "otra memoria")])
    adapter = AdaptadorMemoria(reader)

    result = await adapter.buscar(identidad(), "dato", 10)

    assert reader.scope.tenant_id == "tenant-socket"
    assert reader.scope.subject_user_id == "usuario-socket"
    assert reader.scope.actor_type == "USER"
    assert reader.limit == 100
    assert result == [{
        "memory_id": "m1", "revision_id": "rev-m1",
        "trust": "untrusted_source", "content": "dato de prueba",
    }]


@pytest.mark.asyncio
async def test_no_hay_lector_devuelve_memoria_no_disponible():
    with pytest.raises(MemoriaNoDisponible, match="memoria no disponible"):
        await AdaptadorMemoria().buscar(identidad(), "consulta", 10)


@pytest.mark.asyncio
async def test_no_hay_identidad_devuelve_memoria_no_disponible():
    with pytest.raises(MemoriaNoDisponible, match="memoria no disponible"):
        await AdaptadorMemoria(Reader([])).buscar(None, "consulta", 10)


@pytest.mark.asyncio
@pytest.mark.parametrize(("consulta", "limite"), [("", 10), ("x" * 201, 10), ("x", 0), ("x", 101)])
async def test_rechaza_consulta_vacia_o_fuera_de_limites(consulta, limite):
    with pytest.raises(ValueError):
        await AdaptadorMemoria(Reader([])).buscar(identidad(), consulta, limite)


@pytest.mark.asyncio
async def test_falla_cerrado_si_el_lector_no_responde():
    class LectorCaido:
        async def retrieve(self, *_args, **_kwargs):
            raise ConnectionError("test only")

    with pytest.raises(MemoriaNoDisponible, match="memoria no disponible"):
        await AdaptadorMemoria(LectorCaido()).buscar(identidad(), "consulta", 10)


def test_configuracion_de_memoria_apagada_por_defecto_y_solo_acepta_base_de_prueba():
    assert ConfigMemoria.desde_entorno({}).habilitada is False
    env = {
        "JAX_FARO_MEMORIA_HABILITADA": "true",
        "JAX_FARO_MEMORIA_TEST_DB_HOST": "127.0.0.1",
        "JAX_FARO_MEMORIA_TEST_DB_PORT": "3308",
        "JAX_FARO_MEMORIA_TEST_DB_USER": "jax_test",
        "JAX_FARO_MEMORIA_TEST_DB_NAME": "jax_memory_test",
        "JAX_FARO_MEMORIA_TEST_DB_PASSWORD": "test-secret",
    }
    cfg = ConfigMemoria.desde_entorno(env)
    assert cfg.habilitada is True
    assert (cfg.host, cfg.port, cfg.usuario, cfg.base, cfg.clave) == (
        "127.0.0.1", 3308, "jax_test", "jax_memory_test", "test-secret")
    for key, value in (("JAX_FARO_MEMORIA_TEST_DB_HOST", "10.0.0.1"),
                       ("JAX_FARO_MEMORIA_TEST_DB_NAME", "jax_memory"),
                       ("JAX_FARO_MEMORIA_TEST_DB_USER", "root")):
        invalid = {**env, key: value}
        with pytest.raises(ConfigFaroInvalida, match="solo permite la base de prueba"):
            ConfigMemoria.desde_entorno(invalid)
