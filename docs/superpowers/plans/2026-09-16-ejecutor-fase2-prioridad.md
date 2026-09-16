# Ejecutor Fase 2 · la cola con prioridad para la Mesa — plan de implementación

> **Para agentes ejecutores:** SUB-SKILL REQUERIDA: usar `superpowers:subagent-driven-development`
> (recomendado) o `superpowers:executing-plans`. Los pasos usan casillas (`- [ ]`).

**Objetivo:** que un turno de la Mesa no espere detrás del Ejecutor en la cola de Ollama.

**Arquitectura:** un semáforo **entre procesos** (fichero + `flock`) con dos carriles. La Mesa
entra siempre. El Ejecutor toma su carril sólo si no hay una petición de Mesa esperando, y lo
**libera entre pasos** en vez de retener el turno toda la misión.

**Stack:** Python 3.12, sólo biblioteca estándar (`fcntl.flock`). Sin dependencias nuevas.

**Spec:** `docs/superpowers/specs/2026-09-16-ejecutor-fase2-design.md` §3.4

**Alcance:** este plan es independiente del de las citas
(`2026-09-16-ejecutor-fase2-citas.md`). No comparten un archivo.

## Restricciones globales

- **Entre PROCESOS, no dentro de uno.** La Mesa (`jax-platform`) y el Ejecutor (el arnés del
  usuario `axioma`) son procesos distintos. Un `asyncio.Lock` no coordina nada entre ellos —
  es exactamente el defecto que costó el arreglo de `encolar` el 2026-09-16, y antes de eso ya
  estaba medido que sin semáforo cross-proceso **Ollama serializa**.
- **Prohibido medir contra `/api/chat` de la Mesa**: ensucia la memoria de verdad. Se usa el
  arnés de U5, que mide la cola de Ollama directamente.
- **`/etc/jax/.env` apunta a PRODUCCIÓN fuera de pytest.**
- **El piso de `tests-puros` se declara en DOS listas** del job en `.github/workflows/policy.yml`.
  Se fija con lo que cuenta el runner, nunca con una resta a mano.
- **Nunca cambiar de rama en `/home/fruiz/jax`.**

---

### Task 1: `prioridad.py` — el semáforo de dos carriles

**Archivos:**
- Crear: `jax/ejecutor/prioridad.py`
- Test: `tests/test_ejecutor_prioridad.py`

**Interfaces:**
- Consume: nada.
- Produce: `carril_mesa(raiz)` y `carril_ejecutor(raiz, tope_s)` — dos context managers — y
  `hay_mesa_esperando(raiz) -> bool`. `tope_s` vencido lanza `EsperaAgotada`.

- [ ] **Paso 1: escribir los tests que fallan**

```python
# tests/test_ejecutor_prioridad.py
"""Semáforo de dos carriles para el acceso a Ollama (Fase 2, §3.4)."""
import multiprocessing as mp
import time
import pytest

from jax.ejecutor.prioridad import (
    EsperaAgotada, carril_ejecutor, carril_mesa, hay_mesa_esperando,
)


def test_la_mesa_entra_aunque_el_ejecutor_este_trabajando(tmp_path):
    with carril_ejecutor(tmp_path, tope_s=1):
        with carril_mesa(tmp_path):
            pass  # si esto bloquea, el diseño está mal: la Mesa entra SIEMPRE


def test_el_ejecutor_espera_si_hay_mesa_esperando(tmp_path):
    def mesa(listo, suelte):
        with carril_mesa(tmp_path):
            listo.set()
            suelte.wait(5)
    listo, suelte = mp.Event(), mp.Event()
    p = mp.Process(target=mesa, args=(listo, suelte)); p.start()
    try:
        listo.wait(5)
        assert hay_mesa_esperando(tmp_path) is True
        with pytest.raises(EsperaAgotada):
            with carril_ejecutor(tmp_path, tope_s=0.5):
                pass
    finally:
        suelte.set(); p.join(5)


def test_sin_mesa_el_ejecutor_entra_enseguida(tmp_path):
    t0 = time.monotonic()
    with carril_ejecutor(tmp_path, tope_s=5):
        pass
    assert time.monotonic() - t0 < 1


def test_el_tope_vencido_FALLA_y_no_se_cuela(tmp_path):
    """Si colarse fuera una opción, la prioridad no existiría (§5 del spec)."""
    def mesa(listo, suelte):
        with carril_mesa(tmp_path):
            listo.set(); suelte.wait(5)
    listo, suelte = mp.Event(), mp.Event()
    p = mp.Process(target=mesa, args=(listo, suelte)); p.start()
    try:
        listo.wait(5)
        with pytest.raises(EsperaAgotada):
            with carril_ejecutor(tmp_path, tope_s=0.2):
                pytest.fail("se coló: el carril no debió concederse")
    finally:
        suelte.set(); p.join(5)


def test_el_carril_se_suelta_aunque_el_cuerpo_lance(tmp_path):
    with pytest.raises(ValueError):
        with carril_ejecutor(tmp_path, tope_s=1):
            raise ValueError("boom")
    with carril_ejecutor(tmp_path, tope_s=1):
        pass  # si el anterior no soltó, esto se cuelga
```

- [ ] **Paso 2: correr y ver el rojo**

Correr: `PYTHONPATH=. /home/fruiz/jax/.venv/bin/python -m pytest tests/test_ejecutor_prioridad.py -q`
Esperado: `ModuleNotFoundError: No module named 'jax.ejecutor.prioridad'`.
**Anotá la razón exacta.**

- [ ] **Paso 3: implementar**

```python
# jax/ejecutor/prioridad.py
"""Dos carriles para el acceso a Ollama. La Mesa primero.

Spec: §3.4. Consecuencia pre-registrada de U5: p95 de espera en cola
62,71 s contra un umbral de 60.

ENTRE PROCESOS a propósito: la Mesa y el Ejecutor son procesos distintos,
así que un lock por proceso no coordina nada. Se usa `flock` sobre dos
ficheros, como `backup-hall9000.sh`.

`mesa.lock` lo toma la Mesa mientras usa la GPU. El Ejecutor no lo toma
nunca: lo SONDEA. Si está tomado, hay una petición de persona en curso y
el trabajo de máquina espera.
"""
from __future__ import annotations

import fcntl
import time
from contextlib import contextmanager
from pathlib import Path


class EsperaAgotada(RuntimeError):
    """El Ejecutor no consiguió carril antes del tope. La misión FALLA: si
    colarse fuera una opción, la prioridad no existiría."""


def _fichero(raiz, nombre: str) -> Path:
    p = Path(raiz) / nombre
    p.parent.mkdir(parents=True, exist_ok=True)
    p.touch(exist_ok=True)
    return p


@contextmanager
def carril_mesa(raiz):
    """La Mesa entra SIEMPRE. Bloquea sólo contra otra petición de Mesa."""
    with open(_fichero(raiz, "mesa.lock"), "r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def hay_mesa_esperando(raiz) -> bool:
    """¿Hay una petición de Mesa usando la GPU ahora?"""
    with open(_fichero(raiz, "mesa.lock"), "r+") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return True
        fcntl.flock(f, fcntl.LOCK_UN)
        return False


@contextmanager
def carril_ejecutor(raiz, tope_s: float):
    """El Ejecutor entra sólo si no hay Mesa. Se suelta ENTRE PASOS."""
    limite = time.monotonic() + tope_s
    while hay_mesa_esperando(raiz):
        if time.monotonic() >= limite:
            raise EsperaAgotada(
                f"no se consiguió carril en {tope_s}s; la Mesa tiene prioridad")
        time.sleep(0.05)
    with open(_fichero(raiz, "ejecutor.lock"), "r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
```

- [ ] **Paso 4: correr y ver verde.** Esperado: **5 passed**.

- [ ] **Paso 5: ejercitar el freno (un control que no falla no valida)**

Cambiá `carril_ejecutor` para que **no** sondee (sacá el `while`), corré los tests y comprobá
que **`test_el_tope_vencido_FALLA_y_no_se_cuela` se pone rojo** con el `pytest.fail("se coló")`.
Restaurá. **Reportá la salida.** Sin ese paso, el test no prueba que la prioridad exista.

- [ ] **Paso 6: meter el test en las DOS listas del piso** y medir el número con el runner.

- [ ] **Paso 7: commit**

```bash
git add jax/ejecutor/prioridad.py tests/test_ejecutor_prioridad.py .github/workflows/policy.yml
git commit -m "feat(ejecutor): dos carriles para la GPU, la Mesa primero"
```

---

### Task 2: V3 — medir que la Mesa deja de esperar

**Archivos:**
- Modificar: `scripts/ejecutor_fase0/sonda_cola.py` (sólo si hace falta parametrizar la raíz)
- Crear: `docs/superpowers/specs/2026-09-16-ejecutor-fase2-v3-medicion.md` (el resultado)

**Esta tarea mide, no cambia el diseño.** Si el número obliga a cambiar código, **parar y
avisar**.

- [ ] **Paso 1: línea base en reposo**

Correr la sonda de U5 con el sistema quieto. Guardar en `~/ejecutor-fase2/resultados/`.
**La sonda ya se niega a escribir sobre una salida existente** (arreglado en `ab8341d` tras
mezclar dos corridas en un mismo archivo y dar un falso verde de 25,9 s contra un umbral de 60).
Si se queja, **es correcto**: usá otro nombre, no la fuerces.

- [ ] **Paso 2: medir bajo carga, con el Ejecutor tomando su carril**

Lanzar una misión del Ejecutor que use `carril_ejecutor` y correr la sonda **mientras** corre.
**Verificá el proceso real antes y después** — no con `pgrep -f`, que se encuentra a sí mismo:
ése fue el defecto que hizo que el primer k6 «bajo carga» se corriera con el examen ya terminado
y no midiera nada.

- [ ] **Paso 3: comparar contra 62,71 s**

**V3 pasa si el p95 de espera es ≤ 60 s.** Si da entre 60 y 62,71 mejoró pero **no pasa**: hay
que decirlo así y no redondear a favor.

- [ ] **Paso 4: escribir el resultado con fecha**

Número, cuántas muestras, el comando que lo produjo, y **el ruido** (dispersión entre
repeticiones). Sin ruido declarado, «mejoró» no significa nada. Mínimo 3 repeticiones.

- [ ] **Paso 5: commit**

```bash
git add docs/superpowers/specs/2026-09-16-ejecutor-fase2-v3-medicion.md
git commit -m "docs(ejecutor): V3 medido — la Mesa con el Ejecutor trabajando"
```

---

## Auto-revisión de este plan

- **Cobertura del spec:** §3.4 → Task 1 · §5 fila «no se consigue carril» → Task 1
  (`test_el_tope_vencido_FALLA_y_no_se_cuela`) · §6 V3 → Task 2.
- **Sin placeholders:** todos los pasos de código llevan el código.
- **Tipos consistentes:** `EsperaAgotada`, `carril_mesa`, `carril_ejecutor` y
  `hay_mesa_esperando` se usan con los mismos nombres en tests e implementación.
- **Límite conocido, del spec §3.4:** el Ejecutor puede esperar indefinidamente si la Mesa está
  saturada. Es la intención; el tope lo acota y, al vencerse, la misión falla en vez de colarse.
- **Lo que este plan NO resuelve:** mover el Ejecutor a otra GPU (Red Queen, Q3 2026). Acá se
  reparte la que hay.
