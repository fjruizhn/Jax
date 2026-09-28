from jax.ejecutor.codigo.diff import Cambio
from jax.ejecutor.codigo.barrido import barrer

def C(ruta, mas=(), estado="M"):
    return Cambio(ruta, estado, None, tuple(mas), ())

def test_detecta_tokens_y_llaves(tmp_path):
    for linea in ["t = 'github_pat_" + "A" * 30 + "'", "k='sk-ant-oat01-" + "x" * 40 + "'",
                  "AKIAABCDEFGHIJKLMNOP", "-----BEGIN OPENSSH PRIVATE KEY-----"]:
        assert {v.regla for v in barrer((C("a.py", [linea]),), tmp_path, tope_bytes=10)} == {"secretos"}, linea

def test_tope_de_tamano(tmp_path):
    (tmp_path / "grande.bin").write_bytes(b"0" * 11)
    assert [v.regla for v in barrer((C("grande.bin", estado="A"),), tmp_path, tope_bytes=10)] == ["tamano"]

def test_borrado_no_se_mide(tmp_path):
    assert barrer((C("ya-no-esta.bin", estado="D"),), tmp_path, tope_bytes=10) == ()
