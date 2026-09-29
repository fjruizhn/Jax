"""Barrido de la entrega (spec 2026-09-28 §3.3.2), en dos funciones puras: secretos sobre las
líneas agregadas (del diff neto y de CADA commit intermedio) y tope sobre tamaños ya medidos en
el espejo (`git cat-file -s`), nunca con `stat` en el árbol del clon."""
from jax.ejecutor.codigo.barrido import lineas_con_secretos, tamanos_excedidos, tapar_secretos
from jax.ejecutor.codigo.diff import Cambio

TOKENS = ["t = 'github_pat_" + "A" * 30 + "'", "k='sk-ant-oat01-" + "x" * 40 + "'",
          "AKIAABCDEFGHIJKLMNOP", "-----BEGIN OPENSSH PRIVATE KEY-----", "ghp_" + "b" * 36]


def C(ruta, mas=(), estado="M", menos=()):
    return Cambio(ruta, estado, None, tuple(mas), tuple(menos))


def test_detecta_tokens_y_llaves():
    for linea in TOKENS:
        assert [(v.regla, v.ruta) for v in lineas_con_secretos((C("a.py", ["x", linea]),))] == [("secretos", "a.py")]


def test_una_violacion_por_archivo_y_el_detalle_no_lleva_el_secreto():
    (v,) = lineas_con_secretos((C("a.py", TOKENS),))
    assert all(t not in v.detalle for t in TOKENS)


def test_lo_quitado_no_es_un_secreto_nuevo():
    assert lineas_con_secretos((C("a.py", ["x = 1"], menos=[TOKENS[0]]),)) == ()


def test_tope_sobre_tamanos_ya_medidos():
    vs = tamanos_excedidos({"grande.bin": 11, "justo.bin": 10, "chico.txt": 1}, tope_bytes=10)
    assert [(v.regla, v.ruta, v.detalle) for v in vs] == [("tamano", "grande.bin", "11 > 10 bytes")]


def test_tapar_secretos():
    assert tapar_secretos("a " + TOKENS[0] + " b") == "a t = '***' b"


def test_hay_secreto_y_rutas_con_secretos():
    from jax.ejecutor.codigo.barrido import hay_secreto, rutas_con_secretos
    assert hay_secreto("x " + TOKENS[0]) and not hay_secreto("") and not hay_secreto("limpio")
    vs = rutas_con_secretos((C("ok.py"), C(TOKENS[4] + ".txt"),
                             Cambio("nuevo.py", "R", "viejo_" + TOKENS[4], (), ())))
    assert [(v.regla, v.ruta) for v in vs] == [("secretos", TOKENS[4] + ".txt"), ("secretos", "nuevo.py")]
