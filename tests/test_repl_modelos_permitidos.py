"""
El REPL estaba roto para las 7 facetas (medido 2026-09-14, DEUDA.md "Bloquea
trabajo"): main() pisaba `model_default` con el modelo de facet_binding, pero
`models_allowed` seguía siendo la lista de config.toml, y ninguno de los modelos
vigentes estaba en ella -> ModelNotAllowedError antes de llamar al proveedor.

Decisión de Fernando: la lista permitida sale del catálogo `model` de la DB --
los modelos del MISMO proveedor de la faceta en estado available/degraded -- y
no del TOML. Si la DB no responde al arrancar, se usa el TOML completo (modelo
Y lista), para que las dos cosas salgan siempre de la misma fuente.

Estos tests son puros: `aplicar_registro` recibe el registro ya leído. La
consulta a la DB se verifica en vivo al desplegar.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import copy

from jax.core.registro_facetas import aplicar_registro

_CFG = {
    "personalities": {
        "hipatia": {"type": "http", "model_default": "gemini-2.5-flash",
                    "models_allowed": ["gemini-2.5-flash", "gemini-2.5-pro"]},
        "jekyll": {"type": "http", "model_default": "deepseek-v4-flash",
                   "models_allowed": ["deepseek-v4-flash", "deepseek-v4-pro"]},
        "hyde": {"type": "subprocess", "model_default": "sonnet",
                 "models_allowed": ["sonnet", "opus", "haiku"]},
    }
}

_REGISTRO = {
    "hipatia": {"model": "gemini-3.8-flash", "models_allowed": ["gemini-3.8-flash", "gemini-2.5-pro"]},
    "jekyll": {"model": "deepseek-flash", "models_allowed": ["deepseek-flash", "deepseek-v4-pro"]},
    "hyde": {"model": "claude-opus-5", "models_allowed": ["claude-opus-5", "claude-sonnet-5"]},
}


def test_el_modelo_asignado_queda_dentro_de_su_lista_en_todas_las_facetas():
    cfg = copy.deepcopy(_CFG)
    aplicar_registro(cfg, _REGISTRO)
    for clave, p in cfg["personalities"].items():
        assert p["model_default"] == _REGISTRO[clave]["model"], clave
        assert p["model_default"] in p["models_allowed"], (clave, p)


def test_la_lista_sale_del_catalogo_y_no_del_toml():
    cfg = copy.deepcopy(_CFG)
    aplicar_registro(cfg, _REGISTRO)
    assert cfg["personalities"]["hipatia"]["models_allowed"] == ["gemini-3.8-flash", "gemini-2.5-pro"]
    assert "gemini-2.5-flash" not in cfg["personalities"]["hipatia"]["models_allowed"]


def test_el_modo_pesado_de_jekyll_sigue_permitido():
    # MODELO_PESADO = "deepseek-v4-pro" (main.py): mismo proveedor que jekyll.
    cfg = copy.deepcopy(_CFG)
    aplicar_registro(cfg, _REGISTRO)
    assert "deepseek-v4-pro" in cfg["personalities"]["jekyll"]["models_allowed"]


def test_sin_registro_se_usa_el_toml_completo():
    cfg = copy.deepcopy(_CFG)
    aplicar_registro(cfg, {})
    assert cfg == _CFG


def test_una_faceta_del_registro_que_no_esta_en_el_toml_no_se_inventa():
    cfg = copy.deepcopy(_CFG)
    aplicar_registro(cfg, {**_REGISTRO, "nueva": {"model": "x", "models_allowed": ["x"]}})
    assert "nueva" not in cfg["personalities"]


class _Cursor:
    """Devuelve `filas` para la consulta de la lista permitida y `camino` para
    la del proveedor/URL del modelo (PR-K ronda 2), según el SQL ejecutado."""

    def __init__(self, filas, registro, camino=()):
        self._filas, self._registro, self._camino = filas, registro, list(camino)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def execute(self, sql, params=()):
        self._registro.append((sql, params))

    async def fetchall(self):
        if "f.transport" in self._registro[-1][0]:
            return self._camino
        return self._filas


class _Conexion:
    def __init__(self, filas, registro, camino=()):
        self._filas, self._registro, self._camino, self.cerrada = filas, registro, camino, False

    def cursor(self):
        return _Cursor(self._filas, self._registro, self._camino)

    def close(self):
        self.cerrada = True


def test_cargar_registro_toma_el_proveedor_del_modelo_asignado_y_filtra_por_estado(monkeypatch):
    """Revisión de jax#153: la lista tiene que salir del proveedor del MODELO
    ASIGNADO (model_ref), no de facet_binding.provider_id -- el endpoint de
    aprobación cambia model_ref sin tocar provider_id, así que pueden quedar
    desalineados. Y solo estados invocables."""
    import asyncio
    from jax.core import registro_facetas

    async def registro_falso():
        # provider_id del binding desalineado a propósito
        return {"hipatia": {"model": "gemini-3.8-flash", "provider_id": "proveedor-viejo",
                            "display_name": "Hipatia", "icon": "", "auto_selectable": True},
                "jekyll": {"model": "deepseek-flash", "provider_id": "deepseek",
                           "display_name": "Jekyll", "icon": "", "auto_selectable": True}}

    consultas = []
    filas = [("hipatia", "gemini-2.5-pro"), ("hipatia", "gemini-3.8-flash"), ("jekyll", "deepseek-v4-pro")]
    conexion = _Conexion(filas, consultas)

    async def conexion_falsa():
        return conexion

    monkeypatch.setattr(registro_facetas, "load_facet_registry", registro_falso)
    monkeypatch.setattr(registro_facetas, "_db_conn", conexion_falsa)
    registro = asyncio.run(registro_facetas.cargar_registro())

    assert registro["hipatia"]["models_allowed"] == ["gemini-2.5-pro", "gemini-3.8-flash"]
    assert registro["jekyll"]["models_allowed"] == ["deepseek-v4-pro"]
    assert registro["hipatia"]["display_name"] == "Hipatia", "los campos del router se conservan"
    # PR-K ronda 2: una 2da consulta trae transporte, proveedor y URL del modelo.
    assert len(consultas) == 2
    sql, params = consultas[0]
    assert "model_ref" in sql, "el proveedor tiene que salir del modelo asignado"
    assert "provider_id IN" not in sql, "no se filtra por el provider_id del binding"
    assert set(params) >= {"available", "degraded"} and "deprecated" not in params
    assert conexion.cerrada


def test_cargar_registro_una_faceta_sin_filas_queda_con_lista_vacia(monkeypatch):
    import asyncio
    from jax.core import registro_facetas

    async def registro_falso():
        return {"ada": {"model": "glm-5.3", "provider_id": "zhipu",
                        "display_name": None, "icon": None, "auto_selectable": False}}

    async def conexion_falsa():
        return _Conexion([], [])

    monkeypatch.setattr(registro_facetas, "load_facet_registry", registro_falso)
    monkeypatch.setattr(registro_facetas, "_db_conn", conexion_falsa)
    registro = asyncio.run(registro_facetas.cargar_registro())
    assert registro["ada"]["models_allowed"] == []
    cfg = {"personalities": {"ada": {"model_default": "glm-5.2", "models_allowed": ["glm-5.2"]}}}
    aplicar_registro(cfg, registro)
    assert cfg["personalities"]["ada"]["models_allowed"] == ["glm-5.3"], "el asignado queda permitido igual"


def test_si_el_catalogo_no_lista_el_asignado_igual_queda_permitido():
    # El asignado es la autoridad (Mesa y Jacobs lo usan sin lista): nunca
    # puede quedar fuera de la lista del REPL, aunque el catálogo lo marque
    # deprecated o no lo liste.
    cfg = copy.deepcopy(_CFG)
    aplicar_registro(cfg, {"hipatia": {"model": "gemini-3.8-flash", "models_allowed": ["gemini-2.5-pro"]}})
    assert "gemini-3.8-flash" in cfg["personalities"]["hipatia"]["models_allowed"]


# ---------------------------------------------------------------------------
# PR-K ronda 2 (I1): el proveedor, la URL y el camino salen del MODELO del
# binding, no de config.toml.
# ---------------------------------------------------------------------------

_HTTP = {"type": "http", "provider": "deepseek", "model_default": "deepseek-flash",
         "models_allowed": ["deepseek-flash"]}


def _registro(**kw):
    base = {"model": "m-x", "models_allowed": ["m-x"], "transport": "http_openai_compat",
            "provider_modelo": "openai", "base_url_modelo": "https://api.openai.example/v1"}
    base.update(kw)
    return {"jekyll": base}


def test_el_proveedor_y_la_url_salen_del_modelo_del_binding():
    cfg = {"personalities": {"jekyll": dict(_HTTP)}}
    aplicar_registro(cfg, _registro())
    p = cfg["personalities"]["jekyll"]
    assert p["provider"] == "openai", "el proveedor del TOML no puede ganarle al del modelo"
    assert p["api_url"] == "https://api.openai.example/v1/chat/completions"
    assert "dispatch_bloqueado" not in p


def test_moonshot_se_mapea_a_la_clave_http_kimi():
    cfg = {"personalities": {"jekyll": dict(_HTTP)}}
    aplicar_registro(cfg, _registro(provider_modelo="moonshot", base_url_modelo="https://m.example/v1"))
    assert cfg["personalities"]["jekyll"]["provider"] == "kimi"


def test_gemini_recibe_la_url_base_del_catalogo():
    cfg = {"personalities": {"jekyll": dict(_HTTP)}}
    aplicar_registro(cfg, _registro(transport="http_gemini", provider_modelo="gemini",
                                    base_url_modelo="https://g.example/v1beta"))
    assert cfg["personalities"]["jekyll"]["provider"] == "gemini"
    assert cfg["personalities"]["jekyll"]["api_url"] == "https://g.example/v1beta"


def test_transporte_distinto_del_type_del_toml_bloquea():
    cfg = {"personalities": {"jekyll": dict(_HTTP)}}
    aplicar_registro(cfg, _registro(transport="subprocess", provider_modelo="anthropic", base_url_modelo=None))
    p = cfg["personalities"]["jekyll"]
    assert "subprocess" in p["dispatch_bloqueado"]
    assert p["provider"] == "deepseek", "bloqueada: no se toca el proveedor"


def test_proveedor_sin_camino_http_o_gemini_por_openai_compat_bloquea():
    cfg = {"personalities": {"jekyll": dict(_HTTP)}}
    aplicar_registro(cfg, _registro(provider_modelo="gemini", base_url_modelo="https://g.example"))
    assert "no sabe despachar" in cfg["personalities"]["jekyll"]["dispatch_bloqueado"]


def test_proveedor_sin_base_url_bloquea():
    cfg = {"personalities": {"jekyll": dict(_HTTP)}}
    aplicar_registro(cfg, _registro(base_url_modelo=None))
    assert "sin URL" in cfg["personalities"]["jekyll"]["dispatch_bloqueado"]


def test_ollama_recibe_el_provider_id_del_modelo():
    cfg = {"personalities": {"jax_local": {"type": "ollama", "provider": "ollama",
                                           "model_default": "q", "models_allowed": ["q"]}}}
    aplicar_registro(cfg, {"jax_local": {"model": "q2", "models_allowed": ["q2"], "transport": "ollama",
                                         "provider_modelo": "ollama", "base_url_modelo": "http://l/v1"}})
    assert cfg["personalities"]["jax_local"]["provider_id"] == "ollama"
    assert "dispatch_bloqueado" not in cfg["personalities"]["jax_local"]


def test_cargar_registro_trae_transporte_proveedor_y_url_del_modelo(monkeypatch):
    import asyncio
    from jax.core import registro_facetas

    async def registro_falso():
        return {"jekyll": {"model": "deepseek-flash", "provider_id": "otro",
                           "display_name": "J", "icon": "", "auto_selectable": True}}

    consultas = []
    conexion = _Conexion([("jekyll", "deepseek-flash")], consultas,
                         camino=[("jekyll", "http_openai_compat", "deepseek", "https://api.deepseek.com/v1")])

    async def conexion_falsa():
        return conexion

    monkeypatch.setattr(registro_facetas, "load_facet_registry", registro_falso)
    monkeypatch.setattr(registro_facetas, "_db_conn", conexion_falsa)
    registro = asyncio.run(registro_facetas.cargar_registro())
    assert registro["jekyll"]["transport"] == "http_openai_compat"
    assert registro["jekyll"]["provider_modelo"] == "deepseek"
    assert registro["jekyll"]["base_url_modelo"] == "https://api.deepseek.com/v1"
    sql = consultas[1][0]
    assert "asignado.provider_id" in sql and "b.provider_id" not in sql, "el proveedor sale del MODELO"
