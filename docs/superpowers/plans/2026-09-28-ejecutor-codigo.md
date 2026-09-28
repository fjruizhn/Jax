# El Ejecutor programa — Plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Que Fernando pida un cambio de código desde el modo Ejecutor de la Mesa y Qwen, solo, lo haga en una rama `axioma/<misión>`, corra las pruebas y abra un PR con informe firmado por C5.

**Architecture:** Un tipo de misión nuevo, `codigo`, dentro del Ejecutor existente. `mision_servicio` (proceso de `jaxsvc`, FUERA de la jaula) prepara el clon y entrega el PR; el cerebro (Qwen vía proxy del carril) trabaja como la cuenta `axioma` dentro de la jaula de `cuenta_axioma.py`, con el cerco de red que ya le impide llegar a GitHub. C1 se aplica dos veces: gancho en la jaula y revisión del diff en la entrega.

**Tech Stack:** Python 3.12 + asyncio + httpx (jax); FastAPI + MariaDB 12.3.3 (jax-platform, migraciones en `backend/db/migrations.py`); React 19 + Zustand + vitest (frontend); git; bubblewrap; nftables (cerco).

**Spec:** `docs/superpowers/specs/2026-09-28-ejecutor-codigo-design.md` (jax PR #292), con la enmienda v1.1 de la Tarea 1.

## Global Constraints

- Cerebro de código = el mismo del carril (`JAX_PROXY_CARRIL_MODELO`); nunca Anthropic (DC1).
- El token vive solo en `/etc/jax/.env` como `JAX_GITHUB_TOKEN` (`root:jaxsvc 640`); nunca en argv, nunca en `.git/config`, nunca en la jaula (§4).
- La entrega solo empuja `refs/heads/axioma/<mision_id>`; nunca la rama por omisión (§3.3.4).
- PR listo para revisión (no borrador), etiqueta `axioma`, pie `Hecho-por: Axioma (Ejecutor, misión <id>, cerebro <modelo>)` (DC6, §3.3.5).
- Autor de commits del clon: `Axioma (Ejecutor) <axioma@axioma-ia.io>`, configurable en `axioma_config` clave `ejecutor.codigo.autor` (§4.3).
- Sin informe legible de C5, no hay PR (DC8).
- Una misión a la vez (el latido global del vigía ya lo impone).
- i18n es/en con paridad (`i18n/paridad.test.js`); temas con tokens de `src/tema/tokens.css`; confirmaciones en ventana propia.
- Sin `except Exception` que tape un fallo sin comentario `# fail-soft:`/`# fail-closed:` (job `no-fail-open-except`).
- Todo archivo de test nuevo se lista en `.github/workflows/policy.yml` (job `archivos-de-test-en-ci`) y se re-miden los pisos exactos que toque (regla 5 de SESIONES EN PARALELO).
- Ningún subproceso que contenga el literal `claude` fuera de los exentos de `test_claude_subprocess_solo_via_sandbox.py`.

## Review Focus

1. **Repo cuya rama por omisión no es `main`** (p. ej. `master` en jax): la preparación la lee de la API; la entrega la rechaza igual. Test en Tarea 6 y Tarea 5.
2. **Segundo turno de la misma misión** (`--resume`): reusa el clon y la rama, actualiza el MISMO PR en vez de abrir otro. Test en Tarea 5 (PR existente → PATCH) y Tarea 9.
3. **Qwen no hizo ningún commit** (rama sin cambios): no hay push ni PR; la misión termina `sin_cambios`, visible. Test en Tarea 9.
4. **Diff que renombra un archivo de prueba fuera de `tests/`** (borrado disfrazado): `reglas_diff` lo trata como borrado de prueba. Test en Tarea 3.
5. **Token ausente o vacío en `/etc/jax/.env`**: la misión de código ni arranca (`arranque_rechazado`, contrato `codigo`), sin tocar la red. Test en Tarea 12.

---

### Task 0: El Ejecutor vuelve a arrancar como `jaxsvc` (prerrequisito)

**Hallazgo (2026-09-28, Hyde):** `mision_servicio.py:220` llama a `cuenta_axioma.preparar_directorio_projects`, que corre `sudo install -d -o axioma`. `jaxsvc` no tiene sudo (`sudo -n -u jaxsvc sudo -n true` → «a password is required»). Último turno completado: 2026-09-20 17:22; ninguna misión creó `claude-projects` desde que ese código se desplegó. El directorio de misiones ya trae ACL `user:axioma:--x` y `default:user:fruiz:rwx` (`getfacl /var/lib/jax-ejecutor-misiones`).

**Files:**
- Modify: `jax/ejecutor/contratos/cuenta_axioma.py` (`preparar_directorio_projects`, ~L210-222)
- Test: `tests/test_ejecutor_contratos_cuenta_axioma.py`

**Interfaces:**
- Produces: `async preparar_directorio_de_la_cuenta(c: Cuenta, ruta: Path, *, correr=None) -> None` — crea `ruta` (0770, dueño el proceso), le pone ACL `u:<c.nombre>:rwx` y `d:u:<c.nombre>:rwx`, sin sudo. `preparar_directorio_projects` pasa a ser un alias que la llama (mismo contrato para los llamadores).

- [ ] **Step 1: Test en rojo**

```python
# tests/test_ejecutor_contratos_cuenta_axioma.py
import asyncio
from pathlib import Path
from jax.ejecutor.contratos import cuenta_axioma as CA

def _cuenta(tmp_path):
    return CA.Cuenta("axioma", 22, tmp_path / "k", tmp_path / "n", tmp_path / "l", tmp_path / "p", tmp_path / "h")

def test_preparar_directorio_no_usa_sudo_y_pone_acl(tmp_path):
    llamadas = []
    class _Proc:
        returncode = 0
        async def communicate(self):
            return b"", b""
    async def correr(*argv, **kw):
        llamadas.append(argv)
        return _Proc()
    ruta = tmp_path / "m" / "claude-projects"
    asyncio.run(CA.preparar_directorio_de_la_cuenta(_cuenta(tmp_path), ruta, correr=correr))
    assert ruta.is_dir()
    assert all(a[0] != "sudo" for a in llamadas)
    assert ("setfacl", "-m", "u:axioma:rwx,d:u:axioma:rwx", str(ruta)) in llamadas

def test_preparar_directorio_falla_cerrado_si_setfacl_falla(tmp_path):
    class _Proc:
        returncode = 1
        async def communicate(self):
            return b"", b"setfacl: Operation not permitted"
    async def correr(*argv, **kw):
        return _Proc()
    import pytest
    with pytest.raises(RuntimeError, match="preparar_directorio_fallo"):
        asyncio.run(CA.preparar_directorio_de_la_cuenta(_cuenta(tmp_path), tmp_path / "x", correr=correr))
```

- [ ] **Step 2: Correr y ver el rojo**

Run: `cd ~/worktrees/jax-ejecutor-codigo && PYTHONPATH=.:las_manos .venv/bin/python -m pytest tests/test_ejecutor_contratos_cuenta_axioma.py -k preparar_directorio -v`
Expected: FAIL con `AttributeError: module ... has no attribute 'preparar_directorio_de_la_cuenta'`

- [ ] **Step 3: Implementación**

```python
async def preparar_directorio_de_la_cuenta(c: Cuenta, ruta: Path, *, correr=None) -> None:
    """Crea `ruta` para que la cuenta escriba DESDE LA JAULA, sin sudo (2026-09-28: `jaxsvc`
    no tiene sudo; la versión con `sudo install -o axioma` dejó al Ejecutor sin arrancar desde
    el 2026-09-20). El dueño es el proceso (jaxsvc); `axioma` entra por ACL, también en lo que
    se cree adentro (ACL por omisión). Falla cerrado: sin ACL no hay directorio utilizable."""
    correr = correr or asyncio.create_subprocess_exec
    await asyncio.to_thread(ruta.mkdir, mode=0o770, parents=True, exist_ok=True)
    proc = await correr("setfacl", "-m", f"u:{c.nombre}:rwx,d:u:{c.nombre}:rwx", str(ruta),
                        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    _, errores = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"preparar_directorio_fallo: {errores.decode(errors='replace')}")


async def preparar_directorio_projects(c: Cuenta, ruta: Path, *, correr=None) -> None:
    await preparar_directorio_de_la_cuenta(c, ruta, correr=correr)
```

Borra la implementación vieja con `sudo install` y actualiza su docstring (queda como alias). Actualiza el docstring de `ruta_projects_de_la_mision` donde mencione `sudo install`.

- [ ] **Step 4: Verde** — mismo comando que el Step 2. Expected: PASS. Luego la suite del archivo entero: `... -m pytest tests/test_ejecutor_contratos_cuenta_axioma.py -q` (los `test_bwrap_real_*` se saltan sin bwrap; en hall9000 corren).

- [ ] **Step 5: Prueba real de permisos como `jaxsvc` y `axioma`**

```bash
D=$(sudo -n -u jaxsvc mktemp -d /var/lib/jax-ejecutor-misiones/prueba-acl-XXXX)
sudo -n -u jaxsvc bash -c "cd ~ && PYTHONPATH=$PWD:$PWD/las_manos $PWD/.venv/bin/python - <<EOF
import asyncio; from pathlib import Path; from jax.ejecutor.contratos import cuenta_axioma as CA
c = CA.Cuenta('axioma', 22, Path('/'), Path('/'), Path('/'), Path('/'), Path('/'))
asyncio.run(CA.preparar_directorio_de_la_cuenta(c, Path('$D/claude-projects')))
EOF"
sudo -n -u axioma touch "$D/claude-projects/escribe-axioma" && echo "axioma escribe: OK"
sudo -n rm -rf "$D"
```
Expected: `axioma escribe: OK`. Si `jaxsvc` no puede leer el venv del worktree, usa `/srv/jax-prod/jax/.venv` con este módulo copiado a un tmp — lo que importa es el efecto con los usuarios reales.

- [ ] **Step 6: Commit**

```bash
git add jax/ejecutor/contratos/cuenta_axioma.py tests/test_ejecutor_contratos_cuenta_axioma.py
git commit -m "fix(ejecutor): directorio de la cuenta por ACL, sin sudo — jaxsvc no tiene sudo"
```

- [ ] **Step 7: Despliegue y misión real de solo lectura (GO de Fernando dado 2026-09-28 para el frente)**

Tras PR + integración de Fernando + despliegue según `docs/runbooks/` del Ejecutor: lanzar desde la Mesa una misión de solo lectura contra `ejecutor-prueba` («uptime y df -h»). Expected: turno `completado` con ≥1 afirmación respaldada, y `claude-projects` creado con ACL de `axioma`. Registrar mision_id en la Biblioteca.

---

### Task 1: Enmienda v1.1 del spec

**Files:**
- Modify: `docs/superpowers/specs/2026-09-28-ejecutor-codigo-design.md` (§0, §3, §3.2, §3.3, §11)

- [ ] **Step 1: Editar** — en §0 agrega la fila `v1.1 | (este commit) | Correcciones medidas contra el código vivo antes de planificar`. En §3: «LAS MANOS» → «`mision_servicio` (proceso de `jaxsvc`, fuera de la jaula)» en PREPARAR y ENTREGAR, con la nota: *el Ejecutor no pasa por LAS MANOS (medido 2026-09-28: es un subproceso de jax-platform); la propiedad que el diseño exige —fuera de la jaula, el modelo nunca ve el token— se conserva*. En §3.2: «Perfil nuevo en `hyde_sandbox.py`» → «Variante de la jaula de `jax/ejecutor/contratos/cuenta_axioma.py` (`_jaula` + `remoto_claude`), la misma del Ejecutor: `--dev-bind / /` como `axioma`; la red la limita el cerco `inet ejecutor_cerco` (uid de `axioma`: solo proxy del carril en loopback y SSH del inventario; GitHub bloqueado)». Tabla de montajes: sustituir por «el clon: ACL `u:axioma:rwX` (Tarea 0); `/etc/jax/.env`: `root:jaxsvc 640`, `axioma` no lo lee (medido)». En §11 marca los cuatro puntos como verificados con su dato.

- [ ] **Step 2: Commit** — `git commit -am "docs(ejecutor): spec v1.1 — jaula de cuenta_axioma y entrega en mision_servicio (medido)"`

---

### Task 2: `diff.py` — el diff como estructura

**Files:**
- Create: `jax/ejecutor/codigo/__init__.py` (vacío), `jax/ejecutor/codigo/diff.py`
- Test: `tests/test_ejecutor_codigo_diff.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class Cambio: ruta: str; estado: str  # "A"|"M"|"D"|"R"; ruta_anterior: str | None; agregadas: tuple[str, ...]; quitadas: tuple[str, ...]`
  - `def parsear(texto_diff: str) -> tuple[Cambio, ...]` — sobre `git diff --no-color --find-renames -U0 <base>...HEAD`
  - `async def diff_de_la_rama(clon: Path, base: str) -> tuple[Cambio, ...]`

- [ ] **Step 1: Test en rojo**

```python
# tests/test_ejecutor_codigo_diff.py
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
```

- [ ] **Step 2: Rojo** — `PYTHONPATH=.:las_manos .venv/bin/python -m pytest tests/test_ejecutor_codigo_diff.py -v` → FAIL `ModuleNotFoundError: jax.ejecutor.codigo`.

- [ ] **Step 3: Implementación**

```python
# jax/ejecutor/codigo/diff.py
"""El diff de una misión de código, como estructura (spec 2026-09-28 §3.3.1).
C1 de la entrega decide sobre ESTO, no sobre texto plano con regex."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Cambio:
    ruta: str
    estado: str
    ruta_anterior: str | None
    agregadas: tuple[str, ...]
    quitadas: tuple[str, ...]


def _cerrar(actual: dict | None, salida: list) -> None:
    if actual is not None:
        salida.append(Cambio(actual["ruta"], actual["estado"], actual["anterior"],
                             tuple(actual["mas"]), tuple(actual["menos"])))


def parsear(texto_diff: str) -> tuple[Cambio, ...]:
    salida: list[Cambio] = []
    actual: dict | None = None
    for linea in texto_diff.splitlines():
        if linea.startswith("diff --git "):
            _cerrar(actual, salida)
            b = linea.split(" b/", 1)[1]
            actual = {"ruta": b, "estado": "M", "anterior": None, "mas": [], "menos": []}
        elif actual is None:
            continue
        elif linea.startswith("new file mode"):
            actual["estado"] = "A"
        elif linea.startswith("deleted file mode"):
            actual["estado"] = "D"
        elif linea.startswith("rename from "):
            actual["estado"], actual["anterior"] = "R", linea[len("rename from "):]
        elif linea.startswith("rename to "):
            actual["ruta"] = linea[len("rename to "):]
        elif linea.startswith("+++ ") or linea.startswith("--- "):
            continue
        elif linea.startswith("+"):
            actual["mas"].append(linea[1:])
        elif linea.startswith("-"):
            actual["menos"].append(linea[1:])
    _cerrar(actual, salida)
    return tuple(salida)


async def diff_de_la_rama(clon: Path, base: str) -> tuple[Cambio, ...]:
    proc = await asyncio.create_subprocess_exec(
        "git", "-C", str(clon), "diff", "--no-color", "--find-renames", "-U0", f"{base}...HEAD",
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    salida, errores = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"diff_fallo: {errores.decode(errors='replace')[-500:]}")
    return parsear(salida.decode("utf-8", errors="replace"))
```

- [ ] **Step 4: Verde** — mismo comando. Expected: 2 passed.
- [ ] **Step 5: Commit** — `git add jax/ejecutor/codigo tests/test_ejecutor_codigo_diff.py && git commit -m "feat(ejecutor): diff de la misión de código como estructura"`

---

### Task 3: `reglas_diff.py` — C1 sobre el diff (DC7)

**Files:**
- Create: `jax/ejecutor/codigo/reglas_diff.py`
- Test: `tests/test_ejecutor_codigo_reglas_diff.py`

**Interfaces:**
- Consumes: `Cambio` (Tarea 2)
- Produces: `@dataclass(frozen=True) class Violacion: regla: str; ruta: str; detalle: str` y `def revisar(cambios: tuple[Cambio, ...], repo: str) -> tuple[Violacion, ...]` (`repo` = `owner/nombre`)

- [ ] **Step 1: Test en rojo**

```python
# tests/test_ejecutor_codigo_reglas_diff.py
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
```

- [ ] **Step 2: Rojo** — `... -m pytest tests/test_ejecutor_codigo_reglas_diff.py -v` → FAIL `ModuleNotFoundError`.

- [ ] **Step 3: Implementación**

```python
# jax/ejecutor/codigo/reglas_diff.py
"""C1 de código aplicado al DIFF en la entrega (spec 2026-09-28 §5.1, DC7). Es la que manda:
el gancho de la jaula ataja errores honestos; esto revisa lo que de verdad saldría."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath

from jax.ejecutor.codigo.diff import Cambio

ARCHIVOS_DE_FERNANDO = {
    "fjruizhn/claude-skills": ("common/firmantes-fernando", "bin/go-fernando", "common/CLAUDE.md.core",
                               "common/secciones-constitucionales.txt", "common/hooks/", "bin/codex-bridge.py",
                               "lib/assemble.py", "lib/hostid.py", "bin/codex-auto"),
}
_SKIP = re.compile(r"(@pytest\.mark\.(skip|xfail)|@unittest\.skip|\bskipTest\(|\bmarkTestSkipped\(|"
                   r"\b(it|test|describe)\.(skip|todo)\(|\bxit\(|\bxdescribe\()")
_ASERCION = re.compile(r"^\s*(assert\b|self\.assert|expect\(|\$this->assert|assert[A-Z]\w*\()")
_GANCHOS = re.compile(r"(--no-verify\b|core\.hooksPath|HUSKY=0|SKIP_HOOKS)")


@dataclass(frozen=True)
class Violacion:
    regla: str
    ruta: str
    detalle: str


def _es_prueba(ruta: str | None) -> bool:
    if not ruta:
        return False
    p = PurePosixPath(ruta)
    return ("tests" in p.parts or "test" in p.parts or p.name.startswith("test_")
            or re.search(r"\.(test|spec)\.[jt]sx?$", p.name) is not None or p.name.endswith("Test.php"))


def _es_env_con_valores(c: Cambio) -> bool:
    nombre = PurePosixPath(c.ruta).name
    if not nombre.startswith(".env") or nombre == ".env.example":
        return False
    return any("=" in l and l.split("=", 1)[1].strip() for l in c.agregadas)


def revisar(cambios: tuple[Cambio, ...], repo: str) -> tuple[Violacion, ...]:
    v: list[Violacion] = []
    for c in cambios:
        if c.ruta.startswith(".github/workflows/") or (c.ruta_anterior or "").startswith(".github/workflows/"):
            v.append(Violacion("flujos_ci", c.ruta, "cambio bajo .github/workflows/"))
        for prot in ARCHIVOS_DE_FERNANDO.get(repo, ()):
            if c.ruta == prot or (prot.endswith("/") and c.ruta.startswith(prot)):
                v.append(Violacion("archivo_de_fernando", c.ruta, prot))
        if _es_prueba(c.ruta_anterior or c.ruta) and (c.estado == "D" or (c.estado == "R" and not _es_prueba(c.ruta))):
            v.append(Violacion("prueba_debilitada", c.ruta, "prueba borrada o sacada de las pruebas"))
        for linea in c.agregadas:
            if _SKIP.search(linea):
                v.append(Violacion("prueba_debilitada", c.ruta, linea.strip()[:200]))
            if _GANCHOS.search(linea):
                v.append(Violacion("ganchos", c.ruta, linea.strip()[:200]))
        if _es_prueba(c.ruta):
            netas = sum(bool(_ASERCION.match(l)) for l in c.quitadas) - sum(bool(_ASERCION.match(l)) for l in c.agregadas)
            if netas > 0:
                v.append(Violacion("prueba_debilitada", c.ruta, f"{netas} aserciones netas menos"))
        if _es_env_con_valores(c):
            v.append(Violacion("secretos", c.ruta, ".env con valores"))
    return tuple(v)
```

- [ ] **Step 4: Verde** — Expected: 7 passed.
- [ ] **Step 5: Commit** — `git commit -m "feat(ejecutor): C1 de código sobre el diff de la entrega"` (con los dos archivos).

---

### Task 4: `barrido.py` — secretos y tamaño en la entrega

**Files:**
- Create: `jax/ejecutor/codigo/barrido.py`
- Test: `tests/test_ejecutor_codigo_barrido.py`

**Interfaces:**
- Consumes: `Cambio`, `Violacion`
- Produces: `def barrer(cambios: tuple[Cambio, ...], clon: Path, *, tope_bytes: int) -> tuple[Violacion, ...]`

- [ ] **Step 1: Test en rojo**

```python
# tests/test_ejecutor_codigo_barrido.py
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
```

- [ ] **Step 2: Rojo**. **Step 3: Implementación**

```python
# jax/ejecutor/codigo/barrido.py
"""Barrido de la entrega (spec 2026-09-28 §3.3.2): secretos en lo agregado y tope por archivo."""
from __future__ import annotations

import re
from pathlib import Path

from jax.ejecutor.codigo.diff import Cambio
from jax.ejecutor.codigo.reglas_diff import Violacion

_SECRETOS = re.compile(r"(github_pat_[A-Za-z0-9_]{20,}|ghp_[A-Za-z0-9]{30,}|sk-ant-[a-z0-9]+-[A-Za-z0-9_-]{20,}|"
                       r"AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----)")


def barrer(cambios: tuple[Cambio, ...], clon: Path, *, tope_bytes: int) -> tuple[Violacion, ...]:
    v: list[Violacion] = []
    for c in cambios:
        if c.estado == "D":
            continue
        for linea in c.agregadas:
            if _SECRETOS.search(linea):
                v.append(Violacion("secretos", c.ruta, "patrón de credencial en una línea agregada"))
                break
        ruta = clon / c.ruta
        if ruta.is_file() and ruta.stat().st_size > tope_bytes:
            v.append(Violacion("tamano", c.ruta, f"{ruta.stat().st_size} > {tope_bytes} bytes"))
    return tuple(v)
```

(El detalle nunca incluye la línea con el secreto.)

- [ ] **Step 4: Verde** (3 passed). **Step 5: Commit** — `feat(ejecutor): barrido de secretos y tamaño en la entrega`.

---

### Task 5: `entrega.py` — empujar solo `axioma/*` y abrir/actualizar el PR

**Files:**
- Create: `jax/ejecutor/codigo/entrega.py`, `jax/ejecutor/codigo/git_token.py`
- Test: `tests/test_ejecutor_codigo_entrega.py`

**Interfaces:**
- Produces:
  - `def rama_de_la_mision(mision_id: str) -> str` → `"axioma/<mision_id>"` (valida UUID canónico)
  - `def referencia_permitida(rama: str, rama_por_omision: str, mision_id: str) -> bool`
  - `async def empujar(clon: Path, *, mision_id: str, rama_por_omision: str, remoto_url: str, token: str) -> None`
  - `async def abrir_o_actualizar_pr(cliente: httpx.AsyncClient, *, repo: str, rama: str, base: str, titulo: str, cuerpo: str) -> str` (URL del PR)
  - `git_token.entorno_git(token: str, directorio: Path) -> dict[str, str]` — crea un `GIT_ASKPASS` 0700 que lee `JAX_GIT_TOKEN_EFIMERO` del entorno del subproceso git; devuelve el env para ese subproceso.

- [ ] **Step 1: Tests en rojo** (repo remoto = bare local; API = `httpx.MockTransport`)

```python
# tests/test_ejecutor_codigo_entrega.py
import asyncio, json, subprocess
import httpx, pytest
from jax.ejecutor.codigo import entrega as E

MID = "3f1c9a2e-7b4d-4e8a-9c1f-2a3b4c5d6e7f"

def test_referencia_permitida():
    assert E.referencia_permitida(f"axioma/{MID}", "main", MID)
    assert not E.referencia_permitida("main", "main", MID)
    assert not E.referencia_permitida(f"axioma/{MID}", f"axioma/{MID}", MID)  # por omisión aunque coincida
    assert not E.referencia_permitida("axioma/otra", "main", MID)
    with pytest.raises(ValueError):
        E.rama_de_la_mision("no-es-uuid")

def _git(*a, cwd):
    subprocess.run(["git", *a], cwd=cwd, check=True, capture_output=True)

def test_empujar_solo_la_rama_de_la_mision(tmp_path):
    remoto = tmp_path / "remoto.git"; _git("init", "--bare", "-b", "main", str(remoto), cwd=tmp_path)
    clon = tmp_path / "clon"; _git("clone", str(remoto), str(clon), cwd=tmp_path)
    for k, v in (("user.name", "t"), ("user.email", "t@t")): _git("config", k, v, cwd=clon)
    (clon / "a").write_text("1"); _git("add", "a", cwd=clon); _git("commit", "-m", "base", cwd=clon)
    _git("push", "origin", "main", cwd=clon)
    _git("checkout", "-b", f"axioma/{MID}", cwd=clon)
    (clon / "a").write_text("2"); _git("commit", "-am", "cambio", cwd=clon)
    asyncio.run(E.empujar(clon, mision_id=MID, rama_por_omision="main", remoto_url=str(remoto), token="falso"))
    ramas = subprocess.run(["git", "branch", "--list"], cwd=remoto, capture_output=True, text=True).stdout
    assert f"axioma/{MID}" in ramas
    main = subprocess.run(["git", "log", "--oneline", "main"], cwd=remoto, capture_output=True, text=True).stdout
    assert "cambio" not in main

def test_empujar_niega_si_head_no_es_la_rama(tmp_path):
    clon = tmp_path / "c"; _git("init", "-b", "main", str(clon), cwd=tmp_path)
    with pytest.raises(E.EntregaRechazada, match="rama_no_permitida"):
        asyncio.run(E.empujar(clon, mision_id=MID, rama_por_omision="main", remoto_url="x", token="t"))

def test_pr_nuevo_y_existente():
    pedidos = []
    def manejar(req: httpx.Request):
        pedidos.append((req.method, req.url.path))
        if req.method == "GET" and req.url.path.endswith("/pulls"):
            return httpx.Response(200, json=estado["abiertos"])
        if req.method == "POST" and req.url.path.endswith("/pulls"):
            return httpx.Response(201, json={"number": 7, "html_url": "https://gh/pr/7"})
        if req.method == "PATCH":
            return httpx.Response(200, json={"number": 7, "html_url": "https://gh/pr/7"})
        return httpx.Response(200, json=[])
    estado = {"abiertos": []}
    cli = httpx.AsyncClient(transport=httpx.MockTransport(manejar), base_url="https://api.github.com")
    url = asyncio.run(E.abrir_o_actualizar_pr(cli, repo="o/r", rama=f"axioma/{MID}", base="main", titulo="t", cuerpo="c"))
    assert url == "https://gh/pr/7" and ("POST", "/repos/o/r/pulls") in pedidos
    assert ("POST", "/repos/o/r/issues/7/labels") in pedidos
    estado["abiertos"] = [{"number": 7, "html_url": "https://gh/pr/7"}]; pedidos.clear()
    asyncio.run(E.abrir_o_actualizar_pr(cli, repo="o/r", rama=f"axioma/{MID}", base="main", titulo="t", cuerpo="c2"))
    assert ("PATCH", "/repos/o/r/pulls/7") in pedidos and ("POST", "/repos/o/r/pulls") not in pedidos
```

- [ ] **Step 2: Rojo**. **Step 3: Implementación**

```python
# jax/ejecutor/codigo/git_token.py
"""El token llega a git por GIT_ASKPASS leyendo una variable del entorno DEL SUBPROCESO:
nunca en argv, nunca en la URL, nunca en .git/config (el clon entra a la jaula)."""
from __future__ import annotations

import os
from pathlib import Path

_ASKPASS = '#!/bin/sh\ncase "$1" in *Username*) echo x-access-token;; *) printf %s "$JAX_GIT_TOKEN_EFIMERO";; esac\n'


def entorno_git(token: str, directorio: Path) -> dict[str, str]:
    script = directorio / "askpass.sh"
    script.write_text(_ASKPASS)
    script.chmod(0o700)
    env = {k: v for k, v in os.environ.items() if not k.startswith("JAX_GITHUB")}
    env.update(GIT_ASKPASS=str(script), GIT_TERMINAL_PROMPT="0", JAX_GIT_TOKEN_EFIMERO=token)
    return env
```

```python
# jax/ejecutor/codigo/entrega.py
"""Entrega de la misión de código (spec 2026-09-28 §3.3). Única pieza con el token."""
from __future__ import annotations

import asyncio
import re
import tempfile
from pathlib import Path

import httpx

from jax.ejecutor.codigo.git_token import entorno_git

_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
ETIQUETA = "axioma"


class EntregaRechazada(RuntimeError):
    pass


def rama_de_la_mision(mision_id: str) -> str:
    if not _UUID.match(mision_id):
        raise ValueError("mision_id_invalido")
    return f"axioma/{mision_id}"


def referencia_permitida(rama: str, rama_por_omision: str, mision_id: str) -> bool:
    return rama == rama_de_la_mision(mision_id) and rama != rama_por_omision


async def _git(clon: Path, *args: str, env: dict | None = None) -> str:
    proc = await asyncio.create_subprocess_exec("git", "-C", str(clon), *args, env=env,
                                                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    salida, errores = await proc.communicate()
    if proc.returncode != 0:
        raise EntregaRechazada(f"git_fallo: {' '.join(args[:2])}: {errores.decode(errors='replace')[-300:]}")
    return salida.decode().strip()


async def empujar(clon: Path, *, mision_id: str, rama_por_omision: str, remoto_url: str, token: str) -> None:
    rama = await _git(clon, "rev-parse", "--abbrev-ref", "HEAD")
    if not referencia_permitida(rama, rama_por_omision, mision_id):
        raise EntregaRechazada(f"rama_no_permitida: {rama}")
    with tempfile.TemporaryDirectory(prefix="jax-askpass-") as d:
        env = entorno_git(token, Path(d))
        await _git(clon, "push", "--force-with-lease", remoto_url, f"HEAD:refs/heads/{rama}", env=env)


async def abrir_o_actualizar_pr(cliente: httpx.AsyncClient, *, repo: str, rama: str, base: str,
                                titulo: str, cuerpo: str) -> str:
    duenio = repo.split("/", 1)[0]
    r = await cliente.get(f"/repos/{repo}/pulls", params={"head": f"{duenio}:{rama}", "state": "open"})
    r.raise_for_status()
    abiertos = r.json()
    if abiertos:
        n = abiertos[0]["number"]
        r = await cliente.patch(f"/repos/{repo}/pulls/{n}", json={"title": titulo, "body": cuerpo})
    else:
        r = await cliente.post(f"/repos/{repo}/pulls",
                               json={"title": titulo, "body": cuerpo, "head": rama, "base": base, "draft": False})
    r.raise_for_status()
    pr = r.json()
    et = await cliente.post(f"/repos/{repo}/issues/{pr['number']}/labels", json={"labels": [ETIQUETA]})
    et.raise_for_status()
    return pr["html_url"]
```

- [ ] **Step 4: Verde** (4 passed). **Step 5: Commit** — `feat(ejecutor): entrega de código — solo axioma/<misión> y PR con etiqueta`.

---

### Task 6: `preparar.py` — clon, rama, dependencias, ACL, identidad

**Files:**
- Create: `jax/ejecutor/codigo/preparar.py`
- Test: `tests/test_ejecutor_codigo_preparar.py`

**Interfaces:**
- Consumes: `entorno_git` (Tarea 5), `preparar_directorio_de_la_cuenta` (Tarea 0)
- Produces:
  - `@dataclass(frozen=True) class Repo: owner_repo: str; remoto_url: str; comandos_prueba: tuple[str, ...]`
  - `@dataclass(frozen=True) class Clon: ruta: Path; rama: str; rama_por_omision: str; dependencias: tuple[str, ...]  # lo instalado o "sin_lockfile:<x>"`
  - `async def preparar(repo: Repo, *, mision_id: str, raiz: Path, rama_por_omision: str, token: str, autor: str, dar_acceso) -> Clon` — idempotente: si `raiz/<mision_id>/repo` existe, hace `fetch` y conserva la rama (turno ≥2). `dar_acceso(ruta) -> Awaitable[None]` es `preparar_directorio_de_la_cuenta` ligado a la cuenta (inyectable en tests).
  - `async def rama_por_omision(cliente: httpx.AsyncClient, repo: str) -> str`

- [ ] **Step 1: Tests en rojo**

```python
# tests/test_ejecutor_codigo_preparar.py
import asyncio, subprocess
import httpx
from jax.ejecutor.codigo import preparar as P

MID = "3f1c9a2e-7b4d-4e8a-9c1f-2a3b4c5d6e7f"

def _git(*a, cwd): subprocess.run(["git", *a], cwd=cwd, check=True, capture_output=True)

def _remoto(tmp_path, rama="master"):
    r = tmp_path / "r.git"; _git("init", "--bare", "-b", rama, str(r), cwd=tmp_path)
    w = tmp_path / "w"; _git("clone", str(r), str(w), cwd=tmp_path)
    for k, v in (("user.name", "t"), ("user.email", "t@t")): _git("config", k, v, cwd=w)
    (w / "a").write_text("1"); _git("add", "a", cwd=w); _git("commit", "-m", "x", cwd=w); _git("push", "origin", rama, cwd=w)
    return r

async def _acceso(ruta): pass

def test_clona_rama_de_mision_desde_master_y_autor(tmp_path):
    r = _remoto(tmp_path)
    c = asyncio.run(P.preparar(P.Repo("o/r", str(r), ()), mision_id=MID, raiz=tmp_path / "m",
                               rama_por_omision="master", token="t", autor="Axioma (Ejecutor) <axioma@axioma-ia.io>",
                               dar_acceso=_acceso))
    head = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=c.ruta, capture_output=True, text=True).stdout.strip()
    assert head == f"axioma/{MID}" and c.rama_por_omision == "master"
    cfg = (c.ruta / ".git" / "config").read_text()
    assert "Axioma (Ejecutor)" in cfg and "t@" not in cfg and "token" not in cfg.lower()
    assert c.dependencias == ("sin_lockfile",)

def test_turno_dos_reusa_el_clon(tmp_path):
    r = _remoto(tmp_path)
    args = dict(mision_id=MID, raiz=tmp_path / "m", rama_por_omision="master", token="t", autor="A <a@a>", dar_acceso=_acceso)
    c1 = asyncio.run(P.preparar(P.Repo("o/r", str(r), ()), **args))
    (c1.ruta / "nuevo").write_text("x"); _git("add", "nuevo", cwd=c1.ruta); _git("commit", "-m", "qwen", cwd=c1.ruta)
    c2 = asyncio.run(P.preparar(P.Repo("o/r", str(r), ()), **args))
    assert (c2.ruta / "nuevo").exists()

def test_rama_por_omision_de_la_api():
    cli = httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(200, json={"default_branch": "master"})),
                            base_url="https://api.github.com")
    assert asyncio.run(P.rama_por_omision(cli, "o/r")) == "master"
```

- [ ] **Step 2: Rojo**. **Step 3: Implementación**

```python
# jax/ejecutor/codigo/preparar.py
"""Preparar la misión de código (spec 2026-09-28 §3.1). Corre como jaxsvc, FUERA de la jaula."""
from __future__ import annotations

import asyncio
import tempfile
from dataclasses import dataclass
from pathlib import Path

import httpx

from jax.ejecutor.codigo.entrega import rama_de_la_mision
from jax.ejecutor.codigo.git_token import entorno_git

# lockfile -> comando de instalación (sin red en la jaula: se instala aquí).
LOCKFILES = (
    ("requirements.txt", ("python3", "-m", "venv", ".venv"), (".venv/bin/pip", "install", "-r", "requirements.txt")),
    ("backend/requirements.txt", ("python3", "-m", "venv", "backend/.venv"),
     ("backend/.venv/bin/pip", "install", "-r", "backend/requirements.txt")),
    ("package-lock.json", None, ("npm", "ci", "--no-audit", "--no-fund")),
    ("frontend/package-lock.json", None, ("npm", "--prefix", "frontend", "ci", "--no-audit", "--no-fund")),
    ("composer.lock", None, ("composer", "install", "--no-interaction", "--no-scripts")),
)


@dataclass(frozen=True)
class Repo:
    owner_repo: str
    remoto_url: str
    comandos_prueba: tuple[str, ...]


@dataclass(frozen=True)
class Clon:
    ruta: Path
    rama: str
    rama_por_omision: str
    dependencias: tuple[str, ...]


async def _correr(*argv, cwd: Path, env=None) -> None:
    proc = await asyncio.create_subprocess_exec(*argv, cwd=str(cwd), env=env,
                                                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    _, errores = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"preparar_fallo: {argv[0]} {argv[1] if len(argv) > 1 else ''}: "
                           f"{errores.decode(errors='replace')[-400:]}")


async def rama_por_omision(cliente: httpx.AsyncClient, repo: str) -> str:
    r = await cliente.get(f"/repos/{repo}")
    r.raise_for_status()
    return r.json()["default_branch"]


async def preparar(repo: Repo, *, mision_id: str, raiz: Path, rama_por_omision: str, token: str, autor: str,
                   dar_acceso) -> Clon:
    rama = rama_de_la_mision(mision_id)
    base = raiz / mision_id
    ruta = base / "repo"
    with tempfile.TemporaryDirectory(prefix="jax-askpass-") as d:
        env = entorno_git(token, Path(d))
        if not ruta.exists():
            await dar_acceso(base)
            await _correr("git", "clone", "--branch", rama_por_omision, repo.remoto_url, str(ruta), cwd=base, env=env)
            await _correr("git", "checkout", "-b", rama, cwd=ruta)
        else:
            await _correr("git", "fetch", "origin", cwd=ruta, env=env)
    nombre, correo = autor.rsplit(" <", 1)
    await _correr("git", "config", "user.name", nombre, cwd=ruta)
    await _correr("git", "config", "user.email", correo.rstrip(">"), cwd=ruta)
    instaladas = []
    for lock, previo, comando in LOCKFILES:
        if (ruta / lock).is_file():
            if previo:
                await _correr(*previo, cwd=ruta)
            await _correr(*comando, cwd=ruta)
            instaladas.append(lock)
    await dar_acceso(ruta)
    return Clon(ruta, rama, rama_por_omision, tuple(instaladas) or ("sin_lockfile",))
```

Nota: `dar_acceso(ruta)` al final re-aplica la ACL al árbol recién creado; en la Tarea 0 `preparar_directorio_de_la_cuenta` usa `setfacl -m` sin `-R`: añade aquí `setfacl -R -m u:axioma:rwX,d:u:axioma:rwX` en una función `dar_acceso_recursivo(c, ruta)` en `cuenta_axioma.py` con su test (mismo patrón que la Tarea 0, argv esperado `("setfacl", "-R", "-m", "u:axioma:rwX,d:u:axioma:rwX", str(ruta))`).

- [ ] **Step 4: Verde** (3 passed + el de `dar_acceso_recursivo`). **Step 5: Commit** — `feat(ejecutor): preparar el clon de la misión de código`.

---

### Task 7: C1 dentro de la jaula — reglas de código en `ejecutor_regla`

**Files:**
- Create: `jax-platform/backend/db/semilla_ejecutor_reglas_codigo.json`
- Modify: `jax-platform/backend/db/migrations.py` (bloque del Ejecutor en `run_migrations()`, ~L3581-3587)
- Test: `jax-platform/backend/tests/test_ejecutor_reglas_codigo.py`

- [ ] **Step 1: Semilla** (formato de `semilla_ejecutor_reglas.json`; ninguna es canario):

```json
[
  {"codigo": "codigo_git_push", "tipo": "prohibido", "herramientas": "Bash", "campo": "command",
   "patron": "\\bgit\\b[^\\n]*\\bpush\\b", "ambito_host": null, "ambito_roles": [],
   "origen": "spec 2026-09-28 §3.2: la jaula no empuja; la entrega sí",
   "ejemplos_coincide": [{"tool_name": "Bash", "tool_input": {"command": "git push origin HEAD"}}],
   "ejemplos_no_coincide": [{"tool_name": "Bash", "tool_input": {"command": "git status"}}]},
  {"codigo": "codigo_no_verify", "tipo": "prohibido", "herramientas": "Bash", "campo": "command",
   "patron": "--no-verify\\b|core\\.hooksPath", "ambito_host": null, "ambito_roles": [],
   "origen": "spec 2026-09-28 §5.1 (ganchos)",
   "ejemplos_coincide": [{"tool_name": "Bash", "tool_input": {"command": "git commit --no-verify -m x"}}],
   "ejemplos_no_coincide": [{"tool_name": "Bash", "tool_input": {"command": "git commit -m x"}}]},
  {"codigo": "codigo_workflows", "tipo": "prohibido", "herramientas": "Edit|Write", "campo": "file_path",
   "patron": "/\\.github/workflows/", "ambito_host": null, "ambito_roles": [],
   "origen": "spec 2026-09-28 §5.1 (flujos de CI)",
   "ejemplos_coincide": [{"tool_name": "Write", "tool_input": {"file_path": "/x/repo/.github/workflows/a.yml"}}],
   "ejemplos_no_coincide": [{"tool_name": "Write", "tool_input": {"file_path": "/x/repo/src/a.py"}}]}
]
```

- [ ] **Step 2: Test en rojo**

```python
# backend/tests/test_ejecutor_reglas_codigo.py
import pytest

pytestmark = pytest.mark.asyncio

async def test_reglas_de_codigo_sembradas(db_conn):
    async with db_conn.cursor() as cur:
        await cur.execute("SELECT codigo, tipo, es_canario FROM ejecutor_regla WHERE codigo LIKE 'codigo\\_%' ORDER BY codigo")
        filas = await cur.fetchall()
    assert [f[0] for f in filas] == ["codigo_git_push", "codigo_no_verify", "codigo_workflows"]
    assert all(f[1] == "prohibido" and not f[2] for f in filas)
```

(Usa la fixture de conexión que ya usa `test_ejecutor_tablas.py`; si se llama distinto, adopta ESE nombre — no crees una fixture nueva.)

- [ ] **Step 3: Migración** — en `run_migrations()`, junto a la siembra de reglas existente:

```python
    await _sembrar_reglas_una_vez(cur, "ejecutor_reglas_codigo_v1",
                                  Path(__file__).with_name("semilla_ejecutor_reglas_codigo.json"))
```

- [ ] **Step 4: Verde** con la base de prueba: `cd backend && python -m pytest tests/test_ejecutor_reglas_codigo.py -v`. Luego verifica que la política exportada valida: los ejemplos de cada regla deben pasar `politica.validar` (corre `tests/test_ejecutor_contratos_politica*.py` de jax con un JSON exportado que incluya las 3 reglas).
- [ ] **Step 5: Commit** (en jax-platform, rama `feat/ejecutor-codigo`) — `feat(ejecutor): reglas C1 de código sembradas`.

---

### Task 8: Datos — `ejecutor_repo` y columnas de la misión

**Files:**
- Modify: `jax-platform/backend/db/migrations.py` (`CREATE_EJECUTOR_REPO`, `_TABLES` tras `ejecutor_mision`, `_COLUMNS`)
- Test: `jax-platform/backend/tests/test_ejecutor_tablas.py` (extender)

- [ ] **Step 1: Test en rojo** — agrega al test de tablas:

```python
async def test_ejecutor_repo_y_columnas_de_codigo(db_conn):
    async with db_conn.cursor() as cur:
        await cur.execute("SHOW COLUMNS FROM ejecutor_repo")
        assert {r[0] for r in await cur.fetchall()} >= {"id", "owner_repo", "remoto_url", "comandos_prueba", "activo"}
        await cur.execute("SHOW COLUMNS FROM ejecutor_mision")
        assert {r[0] for r in await cur.fetchall()} >= {"tipo", "repo_id", "rama", "pr_url", "estado_entrega"}
```

- [ ] **Step 2: Rojo**. **Step 3: DDL**

```python
CREATE_EJECUTOR_REPO = """
CREATE TABLE IF NOT EXISTS ejecutor_repo (
  id INT AUTO_INCREMENT PRIMARY KEY,
  owner_repo VARCHAR(140) NOT NULL,
  remoto_url VARCHAR(300) NOT NULL,
  comandos_prueba JSON NOT NULL,
  activo BOOLEAN NOT NULL DEFAULT TRUE,
  created_at DATETIME DEFAULT NOW(),
  UNIQUE KEY uk_ejecutor_repo_owner_repo (owner_repo),
  CHECK (JSON_TYPE(comandos_prueba) = 'ARRAY')
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""
```

En `_COLUMNS`:

```python
    ("ejecutor_mision", "tipo", "ALTER TABLE ejecutor_mision ADD COLUMN tipo ENUM('servidor','codigo') NOT NULL DEFAULT 'servidor'"),
    ("ejecutor_mision", "repo_id", "ALTER TABLE ejecutor_mision ADD COLUMN repo_id INT NULL, ADD CONSTRAINT fk_ejecutor_mision_repo FOREIGN KEY (repo_id) REFERENCES ejecutor_repo(id)"),
    ("ejecutor_mision", "rama", "ALTER TABLE ejecutor_mision ADD COLUMN rama VARCHAR(80) NULL"),
    ("ejecutor_mision", "pr_url", "ALTER TABLE ejecutor_mision ADD COLUMN pr_url VARCHAR(300) NULL"),
    ("ejecutor_mision", "estado_entrega", "ALTER TABLE ejecutor_mision ADD COLUMN estado_entrega ENUM('abierto','rechazada_por_contrato','sin_informe_c5','fallo_entrega','sin_cambios') NULL"),
```

Semilla (datos, una vez, con `_marcada`): la fila de `jax-platform` (DC10):

```python
async def _sembrar_repo_jax_platform_v1(cur):
    if await _marcada(cur, "ejecutor_repo_jax_platform_v1"):
        return
    await cur.execute(
        "INSERT IGNORE INTO ejecutor_repo (owner_repo, remoto_url, comandos_prueba) VALUES (%s, %s, %s)",
        ("fjruizhn/jax-platform", "https://github.com/fjruizhn/jax-platform.git",
         json.dumps(["cd frontend && npx vitest run", "cd backend && JAX_CI_NO_DB=1 .venv/bin/python -m pytest -q"])))
    await cur.execute("INSERT INTO axioma_migracion_de_datos (nombre) VALUES (%s)", ("ejecutor_repo_jax_platform_v1",))
```

(Revisa en `migrations.py` el nombre real de la columna de `axioma_migracion_de_datos` que usa `_marcada` y úsalo igual.) Índice para la consulta del selector: `("ejecutor_repo", "idx_ejecutor_repo_activo", "CREATE INDEX idx_ejecutor_repo_activo ON ejecutor_repo (activo)")` en `_INDEXES`.

- [ ] **Step 4: Verde** + `EXPLAIN SELECT id, owner_repo FROM ejecutor_repo WHERE activo = 1 ORDER BY owner_repo` sin `Using filesort` relevante (tabla chica; anota el plan en el commit).
- [ ] **Step 5: Commit** — `feat(ejecutor): ejecutor_repo y columnas de la misión de código`.

---

### Task 9: El turno de código en `mision.py` / `mision_servicio.py`

**Files:**
- Modify: `jax/ejecutor/mision.py` (`Turno`, `turno_desde_json`, `Dependencias`, `correr_turno`, `prompt_del_turno`)
- Modify: `jax/ejecutor/mision_servicio.py` (`dependencias_reales`)
- Modify: `jax/ejecutor/contratos/cuenta_axioma.py` (`remoto_claude`: parámetro `directorio_trabajo`)
- Create: `jax/ejecutor/codigo/mision_codigo.py` (orquesta preparar → entregar; puro salvo lo inyectado)
- Test: `tests/test_ejecutor_codigo_mision.py`, `tests/test_ejecutor_contratos_cuenta_axioma.py`

**Interfaces:**
- Consumes: Tareas 2–6.
- Produces:
  - `Turno` gana `tipo: str = "servidor"` y `repo: dict | None = None` (`{"owner_repo","remoto_url","comandos_prueba"}`); `turno_desde_json` los lee opcionales y rechaza `tipo` desconocido con `TurnoIlegible`.
  - `Dependencias` gana `preparar_codigo: Callable | None = None` y `entregar_codigo: Callable | None = None`.
  - `remoto_claude(..., directorio_trabajo: Path | None = None)` → `cd {shlex.quote(str(dir))}` en lugar de `cd ~`.
  - `mision_codigo.entregar(clon, *, mision_id, repo, revision_legible, informe, token, cliente, tope_bytes, modelo) -> dict` → `{"estado_entrega": ..., "pr_url": ..., "violaciones": [...]}`

- [ ] **Step 1: Tests en rojo** (mision_codigo puro con dobles)

```python
# tests/test_ejecutor_codigo_mision.py
import asyncio
from pathlib import Path
from jax.ejecutor.codigo import mision_codigo as MC
from jax.ejecutor.codigo.diff import Cambio

MID = "3f1c9a2e-7b4d-4e8a-9c1f-2a3b4c5d6e7f"

class Dobles:
    def __init__(self, cambios=(), falla_push=False):
        self.cambios, self.falla_push, self.empujado, self.pr = cambios, falla_push, False, None
    async def diff(self, clon, base): return self.cambios
    async def empujar(self, clon, **kw):
        if self.falla_push: raise MC.EntregaRechazada("git_fallo")
        self.empujado = True
    async def pr(self, cliente, **kw):
        self.pr = kw; return "https://gh/pr/1"

def _entregar(d, **kw):
    base = dict(clon=MC.Clon(Path("/x"), f"axioma/{MID}", "main", ()), mision_id=MID, repo="o/r",
                revision_legible=True, informe="informe", token="t", cliente=None, tope_bytes=10, modelo="qwen",
                diff=d.diff, empujar=d.empujar, abrir_pr=d.pr)
    base.update(kw)
    return asyncio.run(MC.entregar(**base))

def test_sin_cambios_no_empuja():
    d = Dobles(); r = _entregar(d)
    assert r["estado_entrega"] == "sin_cambios" and not d.empujado

def test_sin_informe_c5_no_abre_pr():
    d = Dobles((Cambio("a.py", "M", None, ("x",), ()),)); r = _entregar(d, revision_legible=False)
    assert r["estado_entrega"] == "sin_informe_c5" and not d.empujado

def test_violacion_rechaza_por_contrato():
    d = Dobles((Cambio(".github/workflows/x.yml", "M", None, ("x",), ()),)); r = _entregar(d)
    assert r["estado_entrega"] == "rechazada_por_contrato" and r["violaciones"][0]["regla"] == "flujos_ci"
    assert not d.empujado

def test_camino_feliz_pie_y_url():
    d = Dobles((Cambio("a.py", "M", None, ("x",), ()),)); r = _entregar(d)
    assert r == {"estado_entrega": "abierto", "pr_url": "https://gh/pr/1", "violaciones": []}
    assert f"Hecho-por: Axioma (Ejecutor, misión {MID}, cerebro qwen)" in d.pr["cuerpo"]

def test_fallo_de_push_es_fallo_entrega():
    d = Dobles((Cambio("a.py", "M", None, ("x",), ()),), falla_push=True)
    assert _entregar(d)["estado_entrega"] == "fallo_entrega"
```

Y en `tests/test_ejecutor_contratos_cuenta_axioma.py`:

```python
def test_remoto_claude_directorio_de_trabajo(tmp_path):
    c = _cuenta(tmp_path)
    r = CA.remoto_claude(c, base_url="http://127.0.0.1:1", modelo="m", prompt="p", directorio_trabajo=Path("/var/lib/x/repo"))
    assert "cd /var/lib/x/repo && env " in r and "cd ~ &&" not in r
    assert "cd ~ && env " in CA.remoto_claude(c, base_url="http://127.0.0.1:1", modelo="m", prompt="p")
```

- [ ] **Step 2: Rojo**. **Step 3: Implementación de `mision_codigo.py`**

```python
# jax/ejecutor/codigo/mision_codigo.py
"""Orquestación de la entrega de una misión de código (spec 2026-09-28 §3.3, DC8)."""
from __future__ import annotations

from jax.ejecutor.codigo.barrido import barrer
from jax.ejecutor.codigo.entrega import EntregaRechazada
from jax.ejecutor.codigo.preparar import Clon
from jax.ejecutor.codigo.reglas_diff import revisar

__all__ = ["entregar", "Clon", "EntregaRechazada"]


def _pie(mision_id: str, modelo: str) -> str:
    return f"\n\n---\nHecho-por: Axioma (Ejecutor, misión {mision_id}, cerebro {modelo})"


async def entregar(*, clon: Clon, mision_id: str, repo: str, revision_legible: bool, informe: str, token: str,
                   cliente, tope_bytes: int, modelo: str, diff, empujar, abrir_pr) -> dict:
    cambios = await diff(clon.ruta, f"origin/{clon.rama_por_omision}")
    if not cambios:
        return {"estado_entrega": "sin_cambios", "pr_url": None, "violaciones": []}
    if not revision_legible:
        return {"estado_entrega": "sin_informe_c5", "pr_url": None, "violaciones": []}
    violaciones = revisar(cambios, repo) + barrer(cambios, clon.ruta, tope_bytes=tope_bytes)
    if violaciones:
        return {"estado_entrega": "rechazada_por_contrato", "pr_url": None,
                "violaciones": [{"regla": v.regla, "ruta": v.ruta, "detalle": v.detalle} for v in violaciones]}
    try:
        await empujar(clon.ruta, mision_id=mision_id, rama_por_omision=clon.rama_por_omision,
                      remoto_url=f"https://github.com/{repo}.git", token=token)
        url = await abrir_pr(cliente, repo=repo, rama=clon.rama, base=clon.rama_por_omision,
                             titulo=f"Axioma: misión {mision_id[:8]}", cuerpo=informe + _pie(mision_id, modelo))
    except EntregaRechazada as exc:  # fail-closed: sin push ni PR; el motivo queda en la bitácora
        return {"estado_entrega": "fallo_entrega", "pr_url": None, "violaciones": [{"regla": "entrega", "ruta": "", "detalle": str(exc)[:300]}]}
    return {"estado_entrega": "abierto", "pr_url": url, "violaciones": []}
```

(El `abrir_pr` real envuelve `httpx.HTTPStatusError` en `EntregaRechazada` en `dependencias_reales`.)

- [ ] **Step 4: `remoto_claude`** — añade `directorio_trabajo: Path | None = None` y cambia la última línea:

```python
    ir = "cd ~" if directorio_trabajo is None else f"cd {q(str(directorio_trabajo))}"
    return f"read -r K; {ir} && env {entorno} {_jaula(c, directorio_projects=directorio_projects)} {claude}"
```

- [ ] **Step 5: `mision.py`** — `Turno` + `turno_desde_json`:

```python
TIPOS = ("servidor", "codigo")
# en Turno (frozen), al final:
    tipo: str = "servidor"
    repo: dict | None = None
# en turno_desde_json, tras leer los campos existentes:
    tipo = datos.get("tipo", "servidor")
    if tipo not in TIPOS or (tipo == "codigo") != isinstance(datos.get("repo"), dict):
        raise TurnoIlegible("tipo_o_repo")
```

`prompt_del_turno` para código añade al final (i18n no aplica: es prompt del modelo, no UI):

```python
def instrucciones_de_codigo(repo: dict) -> str:
    comandos = "\n".join(f"- `{c}`" for c in repo["comandos_prueba"]) or "- (el repo no declara comandos)"
    return ("\n\nMISIÓN DE CÓDIGO. Trabajas en el directorio actual, un clon de "
            f"{repo['owner_repo']} en la rama de la misión. Haz commits locales (git add/commit). "
            "NO empujes: la entrega la hace el sistema. Pruebas del repo:\n" + comandos +
            "\nCita la salida real de cada prueba que corras; lo que no corriste, dilo como no corrido.")
```

`correr_turno`: tras el bloque del auditor y ANTES del `finally`, si `turno.tipo == "codigo" and codigo is None and not auditor_pauso and deps.entregar_codigo is not None`:

```python
                resultado_entrega = await deps.entregar_codigo(ctx, entrega, auditor_legible)
                dice("entrega_codigo", **resultado_entrega)
                if resultado_entrega["estado_entrega"] not in ("abierto", "sin_cambios"):
                    codigo = resultado_entrega["estado_entrega"]
```

y antes de `deps.correr_cerebro`, si `turno.tipo == "codigo"`: `clon = await deps.preparar_codigo(ctx)` y `dice("codigo_preparado", rama=clon.rama, base=clon.rama_por_omision, dependencias=list(clon.dependencias))`. El `_resultado` final incluye `entrega_codigo=resultado_entrega` cuando exista. Tests de `correr_turno` para código con `Dependencias` falsas (patrón de `tests/test_ejecutor_mision*.py` existente): (a) camino feliz emite `codigo_preparado` y `entrega_codigo` con `abierto`; (b) auditor ilegible → `entrega_codigo` con `sin_informe_c5` y código de turno `sin_informe_c5`; (c) tipo servidor NO llama a `preparar_codigo` ni `entregar_codigo`.

- [ ] **Step 6: `mision_servicio.dependencias_reales`** — para `turno.tipo == "codigo"`:
  - `preparar_codigo(ctx)`: `httpx.AsyncClient(base_url="https://api.github.com", headers={"Authorization": f"Bearer {env['JAX_GITHUB_TOKEN']}", "Accept": "application/vnd.github+json"}, timeout=30)`; `rama_por_omision(...)`; `preparar(P.Repo(**turno.repo), mision_id=turno.mision_id, raiz=Path(env["JAX_EJECUTOR_MISIONES"]), ..., autor=<axioma_config 'ejecutor.codigo.autor' o 'Axioma (Ejecutor) <axioma@axioma-ia.io>'>, dar_acceso=lambda r: cuenta_axioma.dar_acceso_recursivo(ctx.cuenta, r))`.
  - `cerebro(...)`: con `directorio_trabajo=clon.ruta` y `herramientas="Bash,Read,Edit,Write,Glob,Grep,Skill"` solo para código.
  - `entregar_codigo(ctx, entrega, auditor_legible)`: informe = `cita.presentar` de `entrega.respaldadas` + faceta del auditor; llama a `mision_codigo.entregar(..., diff=diff_de_la_rama, empujar=entrega_mod.empujar, abrir_pr=<abrir_o_actualizar_pr envuelto>)`; `tope_bytes` de `axioma_config` `ejecutor.codigo.tope_bytes` (default 5242880).
  - El token se lee UNA vez de `env` y nunca se emite en eventos.

- [ ] **Step 7: Verde** de todo `tests/test_ejecutor_*` + `policy/tests/test_ejecutor_lanza_solo_con_contratos.py` (piso `^5 passed` intacto) + `policy/tests/test_claude_subprocess_solo_via_sandbox.py` (los módulos nuevos no contienen el literal `claude`: verifícalo con `grep -n "claude" jax/ejecutor/codigo/*.py` → vacío).
- [ ] **Step 8: Commit** — `feat(ejecutor): turno de código — preparar, trabajar en el clon, entregar`.

---

### Task 10: API de jax-platform

**Files:**
- Modify: `jax-platform/backend/ejecutor/misiones.py` (`crear()`, ~L337; payload del subproceso)
- Modify: `jax-platform/backend/api/ejecutor.py` (nueva ruta `GET /repos`; `POST /misiones` acepta `tipo` y `repo_id`)
- Test: `jax-platform/backend/tests/test_ejecutor_misiones.py` (extender; fixture `runner` existente)

- [ ] **Step 1: Tests en rojo**

```python
async def test_crear_mision_de_codigo_pasa_repo_al_runner(client_superadmin, runner, repo_jax_platform):
    r = await client_superadmin.post("/api/ejecutor/misiones", json={"tipo": "codigo", "repo_id": repo_jax_platform, "objetivo": "arregla X"})
    assert r.status_code == 202
    pedido = runner.ultimo_pedido()
    assert pedido["tipo"] == "codigo" and pedido["repo"]["owner_repo"] == "fjruizhn/jax-platform"
    assert pedido["hosts"] == [runner.host_local]

async def test_codigo_sin_repo_es_422(client_superadmin):
    r = await client_superadmin.post("/api/ejecutor/misiones", json={"tipo": "codigo", "objetivo": "x"})
    assert r.status_code == 422

async def test_repos_solo_superadmin(client, client_superadmin):
    assert (await client.get("/api/ejecutor/repos")).status_code in (401, 403)
    r = await client_superadmin.get("/api/ejecutor/repos")
    assert r.status_code == 200 and any(x["owner_repo"] == "fjruizhn/jax-platform" for x in r.json())
```

(`repo_jax_platform` = fixture que devuelve el id sembrado; `runner.ultimo_pedido()`/`host_local` se añaden al doble existente de `test_ejecutor_misiones.py` — lee su implementación y extiende ESE doble.)

- [ ] **Step 2: Rojo**. **Step 3: Implementación** — en `crear()`: si `tipo == "codigo"`, exige `repo_id` activo (`SELECT owner_repo, remoto_url, comandos_prueba FROM ejecutor_repo WHERE id=%s AND activo=1`; si no → `ErrorDelEjecutor(422, "repo_invalido")`), fija `maquinas=[<nombre de ejecutor_host WHERE es_local=1>]` (si no hay exactamente una → `ErrorDelEjecutor(409, "sin_host_local")`), guarda `tipo, repo_id, rama='axioma/<id>'` en `ejecutor_mision`, y agrega `tipo`/`repo` al JSON del subproceso. Al recibir el evento `entrega_codigo`, `UPDATE ejecutor_mision SET pr_url=%s, estado_entrega=%s WHERE id=%s`. Ruta nueva:

```python
@router.get("/repos")
async def repos(_=Depends(require_superadmin)):
    return await misiones.repos_activos()
```

con `repos_activos()` → `SELECT id, owner_repo FROM ejecutor_repo WHERE activo = 1 ORDER BY owner_repo`.

- [ ] **Step 4: Verde** (`cd backend && python -m pytest tests/test_ejecutor_misiones.py -q`). **Step 5: Commit** — `feat(ejecutor): API de misiones de código y lista de repos`.

---

### Task 11: Frontend — tipo Código en el modo Ejecutor

**Files:**
- Modify: `frontend/src/components/Ejecutor/PanelEjecutor.jsx`, `DetalleMision.jsx`, `textos.js`
- Modify: `frontend/src/store/useEjecutor.js` (`cargarRepos`, `crearMision({tipo, repoId, objetivo})`)
- Modify: `frontend/src/i18n/es.js` (bajo `ejecutor:` L1131), `frontend/src/i18n/en.js` (L1067)
- Test: `PanelEjecutor.test.jsx`, `useEjecutor.test.js`

- [ ] **Step 1: Tests en rojo** (patrón de `PanelEjecutor.test.jsx` existente)

```jsx
it('en tipo Código muestra el selector de repos y exige uno', async () => {
  mockApi({ repos: [{ id: 1, owner_repo: 'fjruizhn/jax-platform' }] });
  render(<PanelEjecutor />);
  await user.click(screen.getByRole('radio', { name: t('ejecutor.tipoCodigo') }));
  expect(await screen.findByRole('combobox', { name: t('ejecutor.repo') })).toBeInTheDocument();
  expect(screen.getByRole('button', { name: t('ejecutor.lanzar') })).toBeDisabled();
});

it('DetalleMision enlaza el PR y traduce el estado de entrega', () => {
  render(<DetalleMision mision={{ tipo: 'codigo', pr_url: 'https://github.com/o/r/pull/7', estado_entrega: 'abierto' }} />);
  expect(screen.getByRole('link', { name: t('ejecutor.verPr') })).toHaveAttribute('href', 'https://github.com/o/r/pull/7');
  expect(screen.getByText(t('ejecutor.entrega.abierto'))).toBeInTheDocument();
});
```

- [ ] **Step 2: Rojo** — `cd frontend && npx vitest run src/components/Ejecutor`.
- [ ] **Step 3: Implementación** — radio `servidor|codigo` (default servidor); en `codigo` se ocultan las máquinas y aparece `<select aria-label={t('ejecutor.repo')}>` con `cargarRepos()`; `DetalleMision` muestra `pr_url` como `<a target="_blank" rel="noopener noreferrer">` y `estado_entrega` vía `ejecutor.entrega.<estado>`; violaciones listadas con `ejecutor.regla.<regla>`. Colores solo con clases de tokens (`text-modo-ejecutor` etc.). Claves nuevas en **es y en**: `tipoServidor, tipoCodigo, repo, elegirRepo, verPr, entrega.{abierto,rechazada_por_contrato,sin_informe_c5,fallo_entrega,sin_cambios}, regla.{flujos_ci,archivo_de_fernando,prueba_debilitada,ganchos,secretos,tamano,entrega}`.
- [ ] **Step 4: Verde** + `npx vitest run src/i18n/paridad.test.js src/tema/contraste.test.js` + `npm run build`. Probar a mano claro y oscuro (captura de ambos en el PR).
- [ ] **Step 5: Commit** — `feat(ejecutor): modo Código en la Mesa (repo, PR y estado de entrega)`.

---

### Task 12: Token de GitHub — guion de guardado y contrato de arranque

**Files:**
- Create: `jax-platform/ops/guardar-token-github.sh` (clon de `ops/guardar-token-claude.sh`)
- Create: `jax-platform/backend/tests/test_guardar_token_github_script.py` (clon de `test_guardar_token_claude_script.py`)
- Modify: `jax/ejecutor/contratos/arranque.py` (prueba `codigo` solo para turnos de código)
- Test: `tests/test_ejecutor_contratos_arranque.py`

- [ ] **Step 1: Guion** — copia exacta del de Claude cambiando: `VARIABLE="JAX_GITHUB_TOKEN"`, `PATRON_TOKEN='^github_pat_[A-Za-z0-9_]+$'`, `LARGO_MINIMO=40`, `LARGO_MAXIMO=255`, y los textos. Test: copia del de Claude con `TOKEN_VALIDO = "github_pat_" + "A" * 60` y un caso más: un token `ghp_…` (clásico) se RECHAZA (DC3 exige grano fino).
- [ ] **Step 2: Contrato de arranque** — test en rojo: con `tipo="codigo"` y `JAX_GITHUB_TOKEN` ausente o vacío, `verificar_contratos` devuelve un fallo `Fallo("codigo", "sin_token_github", {})` **sin** llamar a la red; con token presente, la prueba pasa (no valida contra GitHub: eso lo hace la preparación). Implementación: `pruebas_reales()` añade `"codigo"` solo cuando el contexto trae `tipo == "codigo"`; `_ORDEN` añade `"codigo"` al final.
- [ ] **Step 3: Verde + Commit** (dos repos, dos commits).

---

### Task 13: Canarios de código (Principio VII)

**Files:**
- Create: `jax/ejecutor/contratos/canario_codigo.py`
- Modify: `jax/ejecutor/contratos/arranque.py` (`pruebas_reales()["codigo"]` corre también el canario)
- Test: `tests/test_ejecutor_contratos_canario_codigo.py`

**Interfaces:** `async verificar_codigo(c: cuenta_axioma.Cuenta, *, correr=cuenta_axioma.correr_en_la_cuenta) -> tuple` (fallos vacíos = OK)

- [ ] **Step 1: Test en rojo** — con `correr` falso que devuelve rc según el comando: (a) el evento directo `git push origin HEAD` por el gancho debe dar rc 2 con `regla="codigo_git_push"`; (b) `git ls-remote https://github.com/fjruizhn/jax-platform.git` ejecutado como la cuenta debe FALLAR por red (rc≠0) — si da 0, fallo `cerco_deja_salir_a_github`; (c) `reglas_diff.revisar` sobre un `Cambio(".github/workflows/x.yml","M",...)` debe devolver `flujos_ci` — si no, fallo `c1_diff_no_bloquea`.
- [ ] **Step 2: Implementación** siguiendo `canario_c1.py` (mismo estilo de evento directo al gancho con `correr_en_la_cuenta` y `tope_s` corto).
- [ ] **Step 3: Prueba real en hall9000** — `sudo -n -u axioma git ls-remote https://github.com/fjruizhn/jax-platform.git; echo rc=$?` → `rc≠0` (cerco). Anotar la salida en el commit.
- [ ] **Step 4: Verde + Commit** — `feat(ejecutor): canarios de código — push desde la jaula y workflows bloqueados`.

---

### Task 14: CI — listar los tests nuevos y re-medir pisos

**Files:**
- Modify: `jax/.github/workflows/policy.yml` (job `tests-puros`: listas `-v` ~L1704 y `-q` ~L2032; piso `^2812 passed, 45 skipped` L4463)
- Modify: `jax-platform/.github/workflows/policy.yml` (`frontend-tests` piso 1054; `backend-tests-con-db` `PISO_PASSED=3047`; `backend-tests-no-db` `JAX_CI_MIN_PASSED`)

- [ ] **Step 1:** Añade cada `tests/test_ejecutor_codigo_*.py`, `tests/test_ejecutor_contratos_canario_codigo.py` a las DOS listas de `tests-puros` (el job `archivos-de-test-en-ci` falla si falta alguno).
- [ ] **Step 2:** Re-mide los pisos **sobre la rama rebasada en `origin/master` del día** (no sobre master local): corre exactamente el comando del job y copia el número. No bajes ningún piso.
- [ ] **Step 3:** Commit por repo — `ci: pisos re-medidos con los tests del modo Código`. (Nota: este archivo es de los que integra Fernando cuando lo toca Codex; aquí va en el PR normal para su revisión.)

---

### Task 15: Primera misión real (DC10) y prueba de carga

- [ ] **Step 1:** Fernando crea el token de grano fino (*Contents* y *Pull requests* en escritura; todos sus repos) y corre `sudo /srv/jax-prod/jax-platform/ops/guardar-token-github.sh`; reinicia `jax-platform`. Verificar SIN imprimir el valor: `sudo awk -F= '/^JAX_GITHUB_TOKEN=/{print length($2)}' /etc/jax/.env` > 40.
- [ ] **Step 2:** Misión real desde la Mesa en `fjruizhn/jax-platform`, pedido acotado y verificable (ejemplo: «agrega la clave i18n que falte en en.js respecto de es.js, si falta alguna, con su test de paridad en verde»). Expected: PR abierto con etiqueta `axioma`, pie `Hecho-por`, informe de C5; `git log` del PR con autor `Axioma (Ejecutor)`.
- [ ] **Step 3:** Carga (Cuatro del Rendimiento, §8.8 del spec): con la misión de código corriendo, `loadtest/` existente contra el chat de la Mesa — p95 con y sin misión; registrar en la Biblioteca (`docs/historia/2026-09-XX-ejecutor-codigo-primera-mision.md`).

---

### Task 16: Retiro de `/command` (DC9)

**Files:**
- Delete: `jax-platform/backend/api/command.py`, `backend/tests/test_command_{ownership,path_traversal,stderr}.py`
- Modify: `backend/main.py` (L94 import, L241 `ROUTERS`), `backend/tests/conftest.py` (L289-292), `test_mesa_codigos.py` (L147-380), `test_rutas_de_entorno.py` (L34-35), los demás tests que lo importan (lista del mapa: `test_kill_switch_mesa.py`, `test_owner_cleanup.py`, `test_pipeline_ownership.py`, `test_shadow_validation.py`, `test_memoria_api.py`, `test_websocket_isolation.py`, `test_ejecutor_reglas_envoltorios.py`, `test_usuario_solo_insert_punto_restauracion.py` — revisa cada uso; quita solo lo de /command)
- Modify: `frontend/src/components/BottomBar/BottomBar.jsx` (`MODES` L129, L137, `handleComando` L261-289, L359-360/397-402/462/478), `store/useJaxStore.js` (L15-18, L409-437, L556-573, L594), i18n es/en (claves `modeComando`, `placeholderComando`, `hydeHint`, `hyde_usa_modo_comando`, `commandFailed*`, `commandDryRun`, `taskInitializing`, `taskStarted`, `commandNoResult`), `tema/tokens.css`+`tokens.js` (`--modo-comando`) si ya nadie lo usa
- Modify: `loadtest/{historial_orquestar,descartados_orquestar,memoria_levantar_entorno,catalogo_sync_estado_http_orquestar}.py` (quitar `JAX_BIN`/`JAX_MISSIONS_DIR`)

- [ ] **Step 1: Test en rojo** — `test_command_retirado.py`: `POST /api/command` → 404; y un test de frontend: `MODES` no contiene `comando`.
- [ ] **Step 2:** Borrar/modificar lo listado. `grep -rn "api/command\|JAX_BIN\|JAX_MISSIONS_DIR\|modeComando\|handleComando" backend frontend/src loadtest` → vacío (probar primero que ese grep SÍ encuentra antes de borrar).
- [ ] **Step 3:** Suites completas + pisos re-medidos (bajan por los tests borrados: se anota en el commit cuántos y cuáles, y el piso nuevo se mide, no se resta a ojo).
- [ ] **Step 4:** Tras integrar y desplegar: Fernando borra `~/.local/bin/jax` (es de su home) y quita `JAX_BIN`/`JAX_MISSIONS_DIR` de `/etc/jax/.env`; se cierra sin integrar la rama `ops/lanzador-misiones` (`~/worktrees/jax-lanzador-misiones`).
- [ ] **Step 5: Commit** — `chore(mesa): retirar /command — el modo Código del Ejecutor lo reemplaza`.

---

## Auditorías

Tras las Tareas 0, 5+6, 9, 11 y 16: auditoría de escalón 3 (`arquitecto-adversarial`) sobre el SHA exacto, con el spec y este plan como entrada. Ninguna tarea siguiente empieza con un BLOCK abierto.
