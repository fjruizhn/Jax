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
