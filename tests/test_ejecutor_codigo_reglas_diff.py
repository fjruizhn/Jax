from jax.ejecutor.codigo.diff import Cambio
from jax.ejecutor.codigo.reglas_diff import revisar

def C(ruta, estado="M", mas=(), menos=(), anterior=None):
    return Cambio(ruta, estado, anterior, tuple(mas), tuple(menos))

def reglas(cambios, repo="fjruizhn/jax-platform"):
    return {v.regla for v in revisar(tuple(cambios), repo)}

def test_workflows_prohibidos():
    assert "flujos_ci" in reglas([C(".github/workflows/policy.yml", mas=["x"])])

def test_archivo_de_fernando_solo_en_claude_skills():
    assert "archivo_de_fernando" in reglas([C("bin/go-fernando")], "fjruizhn/claude-skills")
    assert "archivo_de_fernando" in reglas([C("common/hooks/x.sh")], "fjruizhn/claude-skills")
    assert "archivo_de_fernando" not in reglas([C("bin/go-fernando")], "fjruizhn/jax")

def test_borrar_prueba_y_renombrar_fuera_de_tests():
    assert "prueba_debilitada" in reglas([C("tests/test_a.py", "D")])
    assert "prueba_debilitada" in reglas([C("otros/a.py", "R", anterior="tests/test_a.py")])
    assert "prueba_debilitada" in reglas([C("src/x.test.jsx", "D")])

def test_skip_agregado():
    for linea in ["@pytest.mark.skip", "@pytest.mark.xfail(reason='x')", "it.skip('a', () => {})",
                  "self.skipTest('x')", "$this->markTestSkipped('x');", "@unittest.skip('x')"]:
        assert "prueba_debilitada" in reglas([C("tests/test_a.py", mas=[linea])]), linea

def test_aserciones_netas_quitadas():
    assert "prueba_debilitada" in reglas([C("tests/test_a.py", mas=["x = 1"], menos=["assert x == 1", "assert y"])])
    assert "prueba_debilitada" not in reglas([C("tests/test_a.py", mas=["assert x == 2"], menos=["assert x == 1"])])

def test_ganchos_y_env():
    assert "ganchos" in reglas([C("Makefile", mas=["git commit --no-verify"])])
    assert "ganchos" in reglas([C("scripts/x.sh", mas=["git config core.hooksPath /dev/null"])])
    assert "secretos" in reglas([C(".env", "A", mas=["CLAVE=abc"])])
    assert "secretos" in reglas([C("backend/.env.local", "A", mas=["X=1"])])
    assert "secretos" not in reglas([C(".env.example", "A", mas=["CLAVE="])])

def test_codigo_normal_pasa():
    assert revisar((C("src/a.py", mas=["x = 2"], menos=["x = 1"]),), "fjruizhn/jax") == ()
