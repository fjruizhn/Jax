#!/usr/bin/env python3
"""`_quitar_vocativo_faceta`: el nombre de la faceta, cuando ENCABEZA el
mensaje como vocativo, no es contenido de la pregunta.

POR QUE EXISTE (2026-09-20, decision de Fernando sobre una observacion real,
misma ronda que `search_similar_facts` y `detect_completeness_intent`). El
nombre de la faceta va a aparecer en CASI TODOS los mensajes -- para
dirigirse a una hay que nombrarla. La contaminacion medida en el caso real
("jax sabes a que me dedico?" enterrado bajo 25 facts que solo comparten la
palabra "jax" con la consulta, no el contenido) no es un caso raro, es el
caso normal.

El matiz que importa: el nombre suele ser un VOCATIVO, no parte de la
pregunta -- "jax, sabes a que me dedico?" LLAMA a la faceta (el nombre es
ruido para la busqueda); "¿que modelo usa JAX?" el nombre ES la pregunta
(sacarlo la rompe). Por eso esta funcion NO saca el nombre siempre: solo el
vocativo cuando ENCABEZA el mensaje.

Suite pytest pura, sin DB ni red: `_quitar_vocativo_faceta` es una funcion
de texto sobre `jax.core.router.ALIASES` (import, no una lista inventada
aca). La medicion de que esto MEJORA la recuperacion real (no que
"deberia") esta documentada con numeros de produccion en el docstring de
la funcion (jax/memory/db.py) -- no se puede reproducir en esta suite
porque necesita el modelo de embeddings real (bge-m3/Ollama), que esta
deliberadamente inalcanzable aca (ver conftest.py raiz).
"""
from __future__ import annotations

from jax.memory.db import _quitar_vocativo_faceta


# ---------------------------------------------------------------------------
# 1. El vocativo se saca cuando encabeza el mensaje, en sus formas naturales
# ---------------------------------------------------------------------------

def test_el_caso_real_jax_sabes_a_que_me_dedico():
    assert _quitar_vocativo_faceta("jax sabes a que me dedico?") == "sabes a que me dedico?"


def test_vocativo_con_coma():
    assert _quitar_vocativo_faceta("jax, sabes a que me dedico?") == "sabes a que me dedico?"


def test_vocativo_con_dos_puntos():
    assert _quitar_vocativo_faceta("hyde: como configuro nginx?") == "como configuro nginx?"


def test_vocativo_hipatia():
    assert _quitar_vocativo_faceta("hipatia busca info de tributacion en Honduras") == \
        "busca info de tributacion en Honduras"


def test_vocativo_ada():
    assert _quitar_vocativo_faceta("ada ayudame con esta cuenta") == "ayudame con esta cuenta"


def test_vocativo_case_insensitive():
    assert _quitar_vocativo_faceta("JAX sabes a que me dedico?") == "sabes a que me dedico?"


def test_vocativo_con_tilde_en_el_resto_intacta():
    # El vocativo se pela; el resto del texto -- tildes incluidas -- viaja
    # SIN normalizar (esta funcion no toca el texto que se embebe mas alla
    # del vocativo).
    assert _quitar_vocativo_faceta("jax ¿a qué me dedico?") == "¿a qué me dedico?"


def test_vocativo_variante_fonetica_de_la_tabla_existente():
    # "jaid" (fonetico de "hyde") ya vive en jax.core.router.ALIASES -- esta
    # funcion reutiliza esa tabla, no inventa la suya.
    assert _quitar_vocativo_faceta("jaid como estas?") == "como estas?"


def test_vocativo_de_dos_palabras_se_prueba_antes_que_el_de_una():
    # "jax local" es un alias de dos palabras en la misma tabla -- tiene que
    # intentarse ANTES que "jax" solo, si no "jax local, ..." pierde solo
    # "jax " y deja "local" colgando.
    assert _quitar_vocativo_faceta("jax local, decime algo") == "decime algo"


# ---------------------------------------------------------------------------
# 2. Lo que NO se toca: la mencion legitima, y el mensaje sin nada despues
# ---------------------------------------------------------------------------

def test_mencion_legitima_no_al_frente_se_conserva():
    """El caso adversarial de Fernando: cuando el nombre ES la pregunta, no
    se saca. "jax" no encabeza el mensaje -- esta en el medio."""
    assert _quitar_vocativo_faceta("¿que modelo usa jax?") == "¿que modelo usa jax?"


def test_mencion_legitima_al_final_se_conserva():
    assert _quitar_vocativo_faceta("quiero saber que hace jax") == "quiero saber que hace jax"


def test_solo_el_nombre_sin_nada_mas_no_se_toca():
    """El mensaje ES el vocativo -- no hay pregunta que buscar con una
    cadena vacia, asi que se devuelve intacto."""
    assert _quitar_vocativo_faceta("jax") == "jax"


def test_nombre_seguido_de_signo_sin_espacio_no_se_toca():
    # "jax?" no tiene la forma vocativo-con-espacio ("jax ...", "jax, ...");
    # se deja intacto en vez de adivinar.
    assert _quitar_vocativo_faceta("jax?") == "jax?"


def test_palabra_que_empieza_igual_no_es_el_nombre():
    # "jaxsomething" no es "jax" seguido de espacio/puntuacion -- no hay
    # limite de palabra, no se toca.
    assert _quitar_vocativo_faceta("jaxsomething raro") == "jaxsomething raro"


def test_texto_sin_ninguna_faceta_no_se_toca():
    assert _quitar_vocativo_faceta("que clima hace hoy?") == "que clima hace hoy?"
