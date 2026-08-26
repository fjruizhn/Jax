# Hyde — Verificación Real del Sandbox y Precisión de DEUDA.md Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** El ítem de `DEUDA.md` sobre Hyde mezcla 3 sub-problemas de estado muy distinto: (1) escritura a repos reales — ya bloqueada hoy, mal redactada como "pendiente"; (2) concurrencia de `HYDE_SEMAPHORE` — bug real, cerrado por el plan de gobernanza de sub-agentes; (3) red sin acotar — genuinamente abierto, deferido por decisión explícita de esta ronda. Convertir la única verificación adversarial que existía (T5, manual/narrativa en `CONTEXT.md`, nunca quedó como test repetible) en cobertura de regresión real contra bwrap de verdad — y actualizar `DEUDA.md` únicamente en la parte de red (política explícita del usuario, 2026-08-25: es la única edición de `DEUDA.md` en toda la ronda), con el enunciado exacto que dio: `--share-net` da acceso irrestricto, el sandbox cubre filesystem/entorno pero no red, un allowlist requiere enumerar destinos primero (no hecho), es decisión consciente. Escritura y semáforo quedan resueltos en código/test sin narración nueva en `DEUDA.md`.

**Architecture:** Nueva clase de test de integración en `_hyde_sandbox_test.py` (el archivo que crea el plan de gobernanza de sub-agentes) que invoca bwrap REAL — no mockeado — para confirmar en cada corrida que Hyde no puede escribir a los repos reales, sí puede escribir a su workspace, y no hereda variables de entorno del proceso padre. Se salta automáticamente si `bwrap` no está disponible en el host (`unittest.skipUnless`).

**Tech Stack:** Python 3.12, `unittest`, `bwrap` real (confirmado instalado: `/usr/bin/bwrap`, sin privilegios especiales para `fruiz`).

**Spec:** `DEUDA.md` (raíz de `/home/fruiz/jax`), sección "Bloquea trabajo", ítem "Hyde: red sin acotar por dominio/IP, escritura directa...".

## Global Constraints

- **Depende de** `docs/superpowers/plans/2026-08-25-claude-subprocess-gobernanza.md` (Task 1-2 ya aplicados: `_hyde_sandbox_test.py` debe existir con `run_sandboxed_claude` ya migrado) — ejecutar ESE plan primero.
- La red de Hyde (`--share-net` sin allowlist) queda explícitamente FUERA de este plan — decisión tomada: requiere un proxy de egress aparte, proyecto propio, no se construye acá. Este plan solo corrige cómo se describe en `DEUDA.md`, no cambia el comportamiento.
- Los tests de integración de Task 1 usan `sh -c` trivial, nunca invocan `claude` de verdad — corren rápido, sin costo de tokens/API.

---

### Task 1: Test de integración real contra bwrap — verificar las 3 garantías del sandbox

**Files:**
- Modify: `_hyde_sandbox_test.py` (agregar una clase nueva al archivo que crea el plan de gobernanza de sub-agentes)

**Interfaces:**
- Consume: `hyde_sandbox.wrap_hyde_command`, `hyde_sandbox.REAL_JAX_REPO`.

- [ ] **Step 1: Confirmar que el plan previo ya corrió (precondición)**

Run: `test -f /home/fruiz/jax/_hyde_sandbox_test.py && grep -q "run_sandboxed_claude" /home/fruiz/jax/hyde_sandbox.py && echo "OK: precondicion cumplida"`
Expected: `OK: precondicion cumplida` — si falla, ejecutar primero `docs/superpowers/plans/2026-08-25-claude-subprocess-gobernanza.md`.

- [ ] **Step 2: Agregar la clase de test de integración real (al final de `_hyde_sandbox_test.py`)**

```python
import os
import shutil
import subprocess
import tempfile


@unittest.skipUnless(shutil.which("bwrap"), "bwrap no disponible en este host")
class HydeSandboxRealBwrapTest(unittest.TestCase):
    """Integra bwrap REAL (no mock) -- convierte la verificacion adversarial
    manual de T5 (PR jax#18, narrativa en CONTEXT.md, nunca quedó como test
    repetible en el repo) en cobertura de regresion real. Comandos `sh -c`
    triviales, no invoca `claude` de verdad -- corre en <1s por caso."""

    def setUp(self):
        self.workspace_dir = tempfile.mkdtemp(prefix="hyde-sandbox-test-")

    def tearDown(self):
        shutil.rmtree(self.workspace_dir, ignore_errors=True)

    def _run_in_sandbox(self, shell_cmd: str) -> subprocess.CompletedProcess:
        argv = hyde_sandbox.wrap_hyde_command(["sh", "-c", shell_cmd], self.workspace_dir)
        return subprocess.run(argv, capture_output=True, text=True, timeout=15)

    def test_no_puede_escribir_en_jax_real(self):
        probe_path = f"{hyde_sandbox.REAL_JAX_REPO}/_hyde_write_probe_delete_me"
        result = self._run_in_sandbox(f"echo pwned > {probe_path}")
        self.assertNotEqual(result.returncode, 0, result.stderr)
        self.assertFalse(
            os.path.exists(probe_path),
            "Hyde pudo escribir en el repo real -- regresion de sandbox",
        )

    def test_puede_leer_jax_real(self):
        result = self._run_in_sandbox(f"cat {hyde_sandbox.REAL_JAX_REPO}/DEUDA.md | head -1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Deuda técnica", result.stdout)

    def test_puede_escribir_en_workspace(self):
        result = self._run_in_sandbox(f"echo ok > {self.workspace_dir}/probe.txt")
        self.assertEqual(result.returncode, 0, result.stderr)
        with open(f"{self.workspace_dir}/probe.txt") as fh:
            self.assertEqual(fh.read().strip(), "ok")

    def test_entorno_no_hereda_secretos_del_proceso_padre(self):
        os.environ["_HYDE_TEST_SECRET_PROBE"] = "no-deberia-verse-nunca"
        try:
            result = self._run_in_sandbox("echo [${_HYDE_TEST_SECRET_PROBE}]")
        finally:
            del os.environ["_HYDE_TEST_SECRET_PROBE"]
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("no-deberia-verse-nunca", result.stdout)

    def test_home_es_virtual_no_home_real(self):
        result = self._run_in_sandbox("echo $HOME")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), hyde_sandbox.SANDBOX_HOME)
        self.assertNotIn("/home/fruiz", result.stdout)
```

Asegurarse de que `import hyde_sandbox` (no solo `from hyde_sandbox import ...`) esté disponible en el archivo — si el plan previo lo importó distinto, agregar `import hyde_sandbox` junto a los demás imports del archivo.

- [ ] **Step 3: Correr los tests de integración real — con `-rs` para que un skip quede explícito en el resumen, nunca oculto dentro de un "passed" agregado**

Run: `cd /home/fruiz/jax && .venv/bin/python -m pytest _hyde_sandbox_test.py::HydeSandboxRealBwrapTest -v -rs`
Expected: `5 passed` — y CERO líneas `SKIPPED` en el resumen `-rs` (dado que `bwrap` está confirmado instalado en hall9000, `/usr/bin/bwrap`). Si alguno de los 5 aparece como `SKIPPED`, es una anomalía a reportar explícitamente (¿por qué no está `bwrap` donde se esperaba?), NUNCA a tratar como "la suite pasó". Si `test_no_puede_escribir_en_jax_real` falla (no skip, FAIL real), es una regresión real de seguridad — detener y escalar, no ajustar el test para que pase.

- [ ] **Step 4: Correr la suite completa del archivo (unitarios de Task 1 del plan previo + integración de este plan), con `-rs`**

Run: `cd /home/fruiz/jax && .venv/bin/python -m pytest _hyde_sandbox_test.py -v -rs`
Expected: todos en verde (3 de gobernanza de sub-agentes + 5 de este task = 8), resumen `-rs` sin ninguna línea `SKIPPED`. Pegar la salida completa del resumen en el disclosure del task — no solo el número final de "passed".

- [ ] **Step 5: Commit**

```bash
cd /home/fruiz/jax
git add _hyde_sandbox_test.py
git commit -m "test(hyde): convierte la verificacion adversarial manual de T5 en cobertura de regresion real contra bwrap"
```

---

### Task 2: Actualizar el ítem de `DEUDA.md` — SOLO la red de Hyde, con el enunciado exacto del usuario

**Files:**
- Modify: `DEUDA.md`

**Interfaces:**
- Consume: Task 1 de este plan (evidencia real de que escritura/semáforo ya no son problema — usada para JUSTIFICAR sacarlos del bullet, no para agregar prosa de "cerrado" a `DEUDA.md`).

**Política explícita del usuario (2026-08-25): esta es la ÚNICA edición de `DEUDA.md` en toda la ronda de estos 4 planes.** No se agrega ninguna entrada nueva a "Anotado, no bloquea" para escritura/semáforo — esos dos sub-problemas simplemente dejan de mencionarse acá (quedan cerrados en código y verificados por test, sin narración en este archivo). El enunciado de la red va EXACTO como lo dio el usuario, sin parafrasear.

- [ ] **Step 1: Reemplazar el ítem actual**

Reemplazar (sección "Bloquea trabajo"):
```
- **Hyde: red sin acotar por dominio/IP, escritura directa a los repos
  reales fuera de alcance, concurrencia de `HYDE_SEMAPHORE` con el
  sandbox no reverificada.** Declarado explícitamente como no resuelto al
  cerrar el sandbox de bubblewrap (2026-08-23) — el contenimiento
  principal (secretos, filesystem, hooks) sí está cerrado, estos son
  refinamientos de defensa en profundidad pendientes.
```

Por (texto exacto dado por el usuario, sin modificar):
```
- **Hyde: red sin acotar por dominio/IP.**
  - `--share-net` da a Hyde acceso de red IRRESTRICTO desde dentro del
    sandbox.
  - El sandbox contiene filesystem y entorno, NO red.
  - Un allowlist requiere primero enumerar qué destinos necesita Hyde en
    operación real. Esa enumeración NO está hecha.
  - Es una decisión explícita y consciente, no un olvido.
```

- [ ] **Step 2: Commit**

```bash
cd /home/fruiz/jax
git add DEUDA.md
git commit -m "docs(deuda): reescribe el bullet de red de Hyde con enunciado preciso, sin mezclar con escritura/semaforo (ya resueltos en codigo)"
```

---

## Self-Review

- **Cobertura del spec:** el ítem original mezclaba 3 estados distintos — Task 1 produce la evidencia real (tests contra bwrap de verdad) que Task 2 usa para documentar cada sub-problema con precisión, sin inflar ni minimizar ninguno.
- **Decisión del usuario respetada:** la red de Hyde no se toca (código sin cambios) — el fix de alcance es puramente el redactado de `DEUDA.md`, como se acordó.
- **Sin placeholders:** todo el código de Task 1 es el test completo a agregar; Task 2 tiene el texto final exacto para `DEUDA.md`.
- **Orden de ejecución:** este plan requiere que `2026-08-25-claude-subprocess-gobernanza.md` ya esté aplicado — señalado en Global Constraints y verificado como precondición explícita en Task 1 Step 1.
