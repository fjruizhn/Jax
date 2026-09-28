from jax.ejecutor.codigo.diff import parsear, Cambio

DIFF = """diff --git a/src/a.py b/src/a.py
index 1..2 100644
--- a/src/a.py
+++ b/src/a.py
@@ -1 +1,2 @@
-x = 1
+x = 2
+y = 3
diff --git a/tests/test_b.py b/tests/test_b.py
deleted file mode 100644
index 3..0
--- a/tests/test_b.py
+++ /dev/null
@@ -1 +0,0 @@
-def test_b(): pass
diff --git a/tests/test_c.py b/otros/c.py
similarity index 100%
rename from tests/test_c.py
rename to otros/c.py
diff --git a/.env b/.env
new file mode 100644
--- /dev/null
+++ b/.env
@@ -0,0 +1 @@
+CLAVE=abc
"""

def test_parsea_estados_y_lineas():
    c = {x.ruta: x for x in parsear(DIFF)}
    assert c["src/a.py"] == Cambio("src/a.py", "M", None, ("x = 2", "y = 3"), ("x = 1",))
    assert c["tests/test_b.py"].estado == "D"
    assert c["otros/c.py"].estado == "R" and c["otros/c.py"].ruta_anterior == "tests/test_c.py"
    assert c[".env"].estado == "A" and c[".env"].agregadas == ("CLAVE=abc",)

def test_diff_vacio_es_tupla_vacia():
    assert parsear("") == ()


# --- ruling del controlador (Tarea 9): solo "\n" parte líneas; rutas UTF-8, con espacios y entre comillas ---

from jax.ejecutor.codigo.barrido import lineas_con_secretos  # noqa: E402

import pytest  # noqa: E402

PAT = "github_pat_" + "Z" * 30


def _un_archivo(linea_agregada: str, ruta: str = "a.py") -> str:
    return (f"diff --git a/{ruta} b/{ruta}\n--- a/{ruta}\n+++ b/{ruta}\n@@ -0,0 +1 @@\n+{linea_agregada}\n")


@pytest.mark.parametrize("separador", ["\x1c", "\x1d", "\x1e", "\r", "\x0b", "\x0c", "\x85", "\u2028", "\u2029"])
def test_un_separador_que_no_es_salto_de_linea_no_esconde_un_secreto(separador):
    """`str.splitlines()` corta en todos estos: la segunda mitad no empezaba con '+' y se perdía."""
    (c,) = parsear(_un_archivo(f"x = 1{separador}{PAT}"))
    assert c.agregadas == (f"x = 1{separador}{PAT}",)
    assert [v.regla for v in lineas_con_secretos((c,))] == ["secretos"]


def test_una_linea_agregada_que_empieza_con_mas_mas_es_contenido():
    """Dentro de un hunk, `+++ …` es una línea agregada cuyo texto empieza con «++», no una cabecera."""
    (c,) = parsear(_un_archivo(f"++ {PAT}"))
    assert c.agregadas == (f"++ {PAT}",)


def test_una_linea_quitada_que_empieza_con_menos_menos_es_contenido():
    texto = "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +0,0 @@\n--- x\n"
    (c,) = parsear(texto)
    assert c.quitadas == ("-- x",) and c.agregadas == ()


@pytest.mark.parametrize("ruta", ["año.py", "dir con espacio/x.py", "x b/y.py", "ñandú/ç a/b.txt"])
def test_rutas_utf8_y_con_espacios_sin_comillas(ruta):
    (c,) = parsear(_un_archivo("x", ruta))
    assert c.ruta == ruta and c.agregadas == ("x",)


def test_git_agrega_un_tab_tras_una_ruta_con_espacios_en_mas_mas_mas():
    texto = ("diff --git a/dir con espacio/x.py b/dir con espacio/x.py\nnew file mode 100644\n"
             "--- /dev/null\n+++ b/dir con espacio/x.py\t\n@@ -0,0 +1 @@\n+x\n")
    (c,) = parsear(texto)
    assert c.ruta == "dir con espacio/x.py"


def test_rutas_entre_comillas_con_escapes_y_octal():
    texto = ('diff --git "a/a\\303\\261o\\t.py" "b/a\\303\\261o\\t.py"\n'
             'new file mode 100644\n--- /dev/null\n+++ "b/a\\303\\261o\\t.py"\n@@ -0,0 +1 @@\n+x\n')
    (c,) = parsear(texto)
    assert (c.ruta, c.estado, c.agregadas) == ("año\t.py", "A", ("x",))


def test_borrado_y_modo_sin_contenido_con_espacios():
    texto = ("diff --git a/dir con espacio/v.py b/dir con espacio/v.py\ndeleted file mode 100644\n"
             "--- a/dir con espacio/v.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-x\n"
             "diff --git a/s p.sh b/s p.sh\nold mode 100644\nnew mode 100755\n")
    c = {x.ruta: x for x in parsear(texto)}
    assert c["dir con espacio/v.py"].estado == "D" and c["dir con espacio/v.py"].quitadas == ("x",)
    assert c["s p.sh"].estado == "M"


def test_renombre_con_espacios():
    texto = ("diff --git a/tests/test a.py b/otro dir/c.py\nsimilarity index 90%\n"
             "rename from tests/test a.py\nrename to otro dir/c.py\n--- a/tests/test a.py\n+++ b/otro dir/c.py\n"
             "@@ -1 +1 @@\n-x\n+y\n")
    (c,) = parsear(texto)
    assert (c.ruta, c.estado, c.ruta_anterior, c.agregadas) == ("otro dir/c.py", "R", "tests/test a.py", ("y",))
