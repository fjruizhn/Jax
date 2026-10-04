import asyncio
from types import SimpleNamespace

import pytest

from jax.faro.config import ConfigFaroInvalida, ConfigMemoria
from jax.faro.herramientas.memoria import AdaptadorMemoria, MemoriaNoDisponible
from jax.faro.config import ConfigPuerto
from tests._faro_utils import cliente_por_rele, ejecucion, paquete_listo, puerto
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


def test_buscar_usa_tenant_y_usuario_fijados_por_el_socket_y_marca_fuente_no_confiable():
    reader = Reader([envelope("m1", "dato de prueba"), envelope("m2", "otra memoria")])
    adapter = AdaptadorMemoria(reader)

    result = asyncio.run(adapter.buscar(identidad(), "dato", 10))

    assert reader.scope.tenant_id == "tenant-socket"
    assert reader.scope.subject_user_id == "usuario-socket"
    assert reader.scope.actor_type == "USER"
    assert reader.limit == 100
    assert result == [{
        "memory_id": "m1", "revision_id": "rev-m1",
        "trust": "untrusted_source", "content": "dato de prueba",
    }]


def test_no_hay_lector_devuelve_memoria_no_disponible():
    with pytest.raises(MemoriaNoDisponible, match="memoria no disponible"):
        asyncio.run(AdaptadorMemoria().buscar(identidad(), "consulta", 10))


def test_no_hay_identidad_devuelve_memoria_no_disponible():
    with pytest.raises(MemoriaNoDisponible, match="memoria no disponible"):
        asyncio.run(AdaptadorMemoria(Reader([])).buscar(None, "consulta", 10))


@pytest.mark.parametrize(("consulta", "limite"), [("", 10), ("x" * 201, 10), ("x", 0), ("x", 101)])
def test_rechaza_consulta_vacia_o_fuera_de_limites(consulta, limite):
    with pytest.raises(ValueError):
        asyncio.run(AdaptadorMemoria(Reader([])).buscar(identidad(), consulta, limite))


def test_falla_cerrado_si_el_lector_no_responde():
    class LectorCaido:
        async def retrieve(self, *_args, **_kwargs):
            raise ConnectionError("test only")

    with pytest.raises(MemoriaNoDisponible, match="memoria no disponible"):
        asyncio.run(AdaptadorMemoria(LectorCaido()).buscar(identidad(), "consulta", 10))


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


def test_memoria_buscar_pasa_por_guardia_bitacora_y_no_acepta_scope_en_argumentos(tmp_path):
    async def caso():
        _, paquete = paquete_listo(tmp_path)
        reader = Reader([envelope("m1", "dato sensible de test")])
        socket_dir = tmp_path / "sockets"
        socket_dir.mkdir(mode=0o700)
        cfg = ConfigPuerto(socket_dir=socket_dir, costo_conexion_bytes=0)
        async with puerto(cfg, paquete, adaptador_memoria=AdaptadorMemoria(reader)) as srv:
            async with cliente_por_rele(srv) as client:
                tools = {tool.name: tool for tool in (await client.list_tools()).tools}
                props = tools["memoria.buscar"].input_schema["properties"]
                assert set(props) == {"consulta", "limite"}
                result = await client.call_tool("memoria.buscar", {
                    "consulta": "sensible", "limite": 5,
                    "tenant": "tenant-atacante", "usuario": "usuario-atacante",
                })
                assert not result.is_error
                assert reader.scope.tenant_id == "t-real"
                assert reader.scope.subject_user_id == "u-real"
                assert "untrusted_source" in str(result.structured_content or result.content)
            llamadas = [r for r in srv.registros if r.get("evento") == "llamada"]
            assert any(r.get("objetivo") == "memoria.buscar" and r.get("decision") == "permitido" for r in llamadas)

    asyncio.run(caso())


def test_memoria_buscar_sin_adaptador_falla_cerrado_y_queda_en_bitacora(tmp_path):
    async def caso():
        _, paquete = paquete_listo(tmp_path)
        socket_dir = tmp_path / "sockets"
        socket_dir.mkdir(mode=0o700)
        cfg = ConfigPuerto(socket_dir=socket_dir, costo_conexion_bytes=0)
        async with puerto(cfg, paquete) as srv:
            async with cliente_por_rele(srv) as client:
                result = await client.call_tool("memoria.buscar", {"consulta": "cualquier cosa"})
                assert result.is_error
                assert "memoria no disponible" in str(result.content).lower()
            llamadas = [r for r in srv.registros if r.get("evento") == "llamada"]
            assert any(r.get("objetivo") == "memoria.buscar" for r in llamadas)

    asyncio.run(caso())
