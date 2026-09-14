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


def test_si_el_catalogo_no_lista_el_asignado_igual_queda_permitido():
    # El asignado es la autoridad (Mesa y Jacobs lo usan sin lista): nunca
    # puede quedar fuera de la lista del REPL, aunque el catálogo lo marque
    # deprecated o no lo liste.
    cfg = copy.deepcopy(_CFG)
    aplicar_registro(cfg, {"hipatia": {"model": "gemini-3.8-flash", "models_allowed": ["gemini-2.5-pro"]}})
    assert "gemini-3.8-flash" in cfg["personalities"]["hipatia"]["models_allowed"]
