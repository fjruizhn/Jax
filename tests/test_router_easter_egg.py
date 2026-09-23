#!/usr/bin/env python3
"""Easter egg IDE1990: el criterio de disparo del REPL (y de la Mesa web, que
lo copia; lo vigila la familia `router_keywords` de check_mirror_sync).

POR QUE EXISTE (2026-09-23, auditoría adversarial de jax-platform#152). El
criterio viejo era una subcadena tras quitar TODOS los espacios: "el cliente
pide 1990 unidades" -> "pide1990" -> contiene "ide1990" -> disparaba. En el
REPL solo lo usa Fernando; en la Mesa, multiusuario, eso le quita la
respuesta a quien escribió y le muestra el mensaje de Fernando.

Suite pytest pura, sin DB ni red.
"""
from __future__ import annotations

import asyncio

import pytest

from jax.core.router import EASTER_EGG_TEXT, Router, es_easter_egg


@pytest.mark.parametrize("texto", [
    "IDE1990", "ide1990", "IDE 1990", "IDE  1990", "  Ide1990  ", "Íde1990",
    "hola IDE1990 que tal", "IDE1990!", "(ide1990)", "ＩＤＥ１９９０",
])
def test_dispara(texto):
    assert es_easter_egg(texto)


@pytest.mark.parametrize("texto", [
    # Los falsos positivos del criterio viejo (medidos en la auditoría).
    "el cliente pide 1990 unidades", "Please provide 1990 census data",
    "decide 1990 o 1991", "cuánto mide 1990 mm", "side 1990",
    # Los casos reales del 2026-08-09 y el inventado.
    "IDE|990", "IDE2024",
    # Bordes numéricos y de palabra.
    "IDE19901", "IDE199", "tengo un IDE de 1990", "xide1990", "ide-1990",
])
def test_no_dispara(texto):
    assert not es_easter_egg(texto)


def test_el_repl_usa_el_mismo_criterio():
    router = Router()
    decision = asyncio.run(router.route("IDE1990"))
    assert decision.kind == "easter_egg"
    assert decision.text == EASTER_EGG_TEXT

    decision = asyncio.run(router.route("el cliente pide 1990 unidades"))
    assert decision.kind != "easter_egg"
