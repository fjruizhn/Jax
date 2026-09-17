# Ejecutor Fase 2 · la tubería de citas — plan de implementación

> **Para agentes ejecutores:** SUB-SKILL REQUERIDA: usar `superpowers:subagent-driven-development`
> (recomendado) o `superpowers:executing-plans` para implementar tarea por tarea. Los pasos usan
> casillas (`- [ ]`) para seguimiento.

**Objetivo:** que el Ejecutor no pueda entregar una afirmación que no esté literal en una salida
de comando capturada.

**Arquitectura:** tres módulos. `captura.py` corre comandos y guarda la salida **completa** con
su procedencia y una marca de truncado. `hechos.py` deriva hechos del sistema y los inyecta en
cada turno, con TTL. `cita.py` es puro: recibe una afirmación y las capturas, y dice si la línea
citada está literal ahí. El transporte deja caer toda afirmación que no salga `respaldada`, y
las salidas crudas se entregan siempre.

**Stack:** Python 3.12 (el CI corre 3.12; el venv local es 3.14), sólo biblioteca estándar en
`cita.py`. Sin dependencias nuevas.

**Spec:** `docs/superpowers/specs/2026-09-16-ejecutor-fase2-design.md`

**Alcance:** este plan **no** incluye `prioridad.py` (la cola con prioridad de U5). Es un
subsistema independiente y tiene plan propio:
`2026-09-16-ejecutor-fase2-prioridad.md`.

## Restricciones globales

- **`cita.py` es puro:** sin red, sin E/S, sin reloj. Lo corre `tests-puros` en CI, igual que
  `scripts/ejecutor_fase0/medicion.py`.
- **Sólo biblioteca estándar** en `cita.py`. `captura.py` y `hechos.py` pueden usar lo que ya
  está en `requirements.txt`; no se agregan dependencias.
- **Ningún dato de clientes entra al repo.** Las capturas viven en `~/ejecutor/capturas/`, fuera
  del árbol, con permisos 0600. Los tests usan `tmp_path`.
- **Marca fail-soft en la línea del `except`**, no dentro del bloque. `policy/tests/test_no_fail_open_except.py`
  lo exige y hoy está en cero violaciones.
- **El piso de tests de `tests-puros` se declara en DOS listas** del job en
  `.github/workflows/policy.yml`: la de `python -m pytest -v` y la del contador. Un archivo de
  test que entre sólo a la primera **no queda vigilado por el piso**. Van en las dos.
- **El piso se fija con lo que cuenta el runner, nunca con una resta a mano.** Hoy:
  **494 passed, 1 skipped**.
- **Nunca cambiar de rama en `/home/fruiz/jax`** — es el checkout que sirve producción. Usar el
  worktree.

---

### Task 1: `cita.py` — el verificador puro

**Archivos:**
- Crear: `jax/ejecutor/__init__.py` (vacío)
- Crear: `jax/ejecutor/cita.py`
- Test: `tests/test_ejecutor_cita.py`

**Interfaces:**
- Consume: nada. Es el primero.
- Produce: `Captura(maquina, comando, salida, stderr, truncada)`,
  `Afirmacion(maquina, texto, comando, linea)`,
  *(firmas corregidas el 2026-09-16 tras el cambio de contrato: stderr citable y máquina
  obligatoria — ver `2026-09-16-ejecutor-fase2-design.md` §3.3)*
  `Veredicto(estado, motivo)`, `verificar(afirmacion, capturas) -> Veredicto`,
  `normalizar(linea) -> str`, y las constantes `RESPALDADA`, `SIN_RESPALDO`,
  `FUENTE_TRUNCADA`, `FUENTE_INEXISTENTE`.

- [ ] **Paso 1: escribir los tests que fallan**

```python
# tests/test_ejecutor_cita.py
"""Verificador de citas del Ejecutor (Fase 2).

Spec: docs/superpowers/specs/2026-09-16-ejecutor-fase2-design.md §3.3.
Puro: sin red, sin E/S, sin reloj. Lo corre tests-puros en CI.
"""
import pytest

from jax.ejecutor.cita import (
    FUENTE_INEXISTENTE, FUENTE_TRUNCADA, RESPALDADA, SIN_RESPALDO,
    Afirmacion, Captura, normalizar, verificar,
)

SALIDA_FREE = "               total        used        free\nMem:            89Gi        12Gi        70Gi"
CAPTURAS = [Captura(comando="free -h", salida=SALIDA_FREE, truncada=False)]


def test_una_linea_literal_esta_respaldada():
    a = Afirmacion(texto="hay 89Gi de RAM", comando="free -h",
                   linea="Mem:            89Gi        12Gi        70Gi")
    assert verificar(a, CAPTURAS).estado == RESPALDADA


def test_una_linea_que_no_esta_en_la_salida_no_tiene_respaldo():
    a = Afirmacion(texto="hay 128Gi de RAM", comando="free -h",
                   linea="Mem:           128Gi        12Gi        70Gi")
    assert verificar(a, CAPTURAS).estado == SIN_RESPALDO


def test_citar_un_comando_que_no_se_corrio_es_fuente_inexistente():
    a = Afirmacion(texto="algo", comando="lsblk", linea="sda")
    assert verificar(a, CAPTURAS).estado == FUENTE_INEXISTENTE


def test_una_captura_truncada_no_respalda_NADA_aunque_la_linea_este():
    """§2.4 y tarea 9 de U3: afirmó que todos los paquetes eran de `noble`
    habiendo visto 2 KB de una salida de 85,9 KB que nunca abrió. Si la
    salida vino cortada, no se mira el contenido: se rechaza antes."""
    capturas = [Captura(comando="apt list", salida="paquete/noble 1.0", truncada=True)]
    a = Afirmacion(texto="es de noble", comando="apt list", linea="paquete/noble 1.0")
    assert verificar(a, capturas).estado == FUENTE_TRUNCADA


def test_los_espacios_no_deciden():
    a = Afirmacion(texto="hay 89Gi", comando="free -h", linea="Mem: 89Gi 12Gi 70Gi")
    assert verificar(a, CAPTURAS).estado == RESPALDADA


def test_un_numero_parecido_NO_cuenta_como_respaldo():
    """Invención real de U3 (tarea 3): dijo «contexto 131.074» cuando su
    propia salida decía 131072. Si esto pasara, el verificador no sirve."""
    capturas = [Captura(comando="ollama show", salida="context length 131072", truncada=False)]
    a = Afirmacion(texto="el contexto es 131074", comando="ollama show",
                   linea="context length 131074")
    assert verificar(a, capturas).estado == SIN_RESPALDO


def test_las_mayusculas_SI_deciden():
    """No se normaliza mayúsculas: `Docker` y `docker` son datos distintos."""
    capturas = [Captura(comando="ss -ltnp", salida="LISTEN 0 4096 *:8188 users:((\"node\"))", truncada=False)]
    a = Afirmacion(texto="8188 es Docker", comando="ss -ltnp",
                   linea="LISTEN 0 4096 *:8188 users:((\"Docker\"))")
    assert verificar(a, capturas).estado == SIN_RESPALDO


def test_normalizar_colapsa_espacios_pero_no_toca_el_resto():
    assert normalizar("  Mem:   89Gi  ") == "Mem: 89Gi"
    assert normalizar("89 GiB") != normalizar("91 GB")
```

- [ ] **Paso 2: correr los tests y verlos fallar**

Correr: `PYTHONPATH=. /home/fruiz/jax/.venv/bin/python -m pytest tests/test_ejecutor_cita.py -q`
Esperado: FALLA con `ModuleNotFoundError: No module named 'jax.ejecutor'`.
**Anotá la razón exacta del rojo.** Si falla por otra cosa, el test está mal.

- [ ] **Paso 3: implementar lo mínimo**

```python
# jax/ejecutor/cita.py
"""Verificador de citas del Ejecutor (Fase 2).

Spec: docs/superpowers/specs/2026-09-16-ejecutor-fase2-design.md §3.3.

Puro a propósito: sin red, sin E/S, sin reloj, sólo biblioteca estándar.
Verificar una cita es una BÚSQUEDA DE SUBCADENA, no un juicio sobre la
verdad -- por eso no hay nada que calibrar y no existe el falso positivo
por umbral mal puesto.

Garantiza PROCEDENCIA, no CORRECCIÓN: una afirmación puede citar una línea
real y aun así concluir mal a partir de ella (riesgo 2 del spec).
"""
from __future__ import annotations

from dataclasses import dataclass

RESPALDADA = "respaldada"
SIN_RESPALDO = "sin_respaldo"
FUENTE_TRUNCADA = "fuente_truncada"
FUENTE_INEXISTENTE = "fuente_inexistente"


@dataclass(frozen=True)
class Captura:
    comando: str
    salida: str
    truncada: bool


@dataclass(frozen=True)
class Afirmacion:
    texto: str
    comando: str
    linea: str


@dataclass(frozen=True)
class Veredicto:
    estado: str
    motivo: str


def normalizar(linea: str) -> str:
    """Recorta los laterales y colapsa espacios internos. NADA MÁS.

    No toca mayúsculas, ni puntuación, ni números: `131.074` no puede
    coincidir con `131.072` (invención real de U3, tarea 3) y `Docker` no
    puede coincidir con `docker`.
    """
    return " ".join(linea.split())


def verificar(afirmacion: Afirmacion, capturas) -> Veredicto:
    """¿La línea citada está literal en la salida de ese comando?"""
    for captura in capturas:
        if captura.comando != afirmacion.comando:
            continue
        # El truncado se mira ANTES que el contenido: si la salida vino
        # cortada, dar por bueno lo que sí llegó es exactamente el error de
        # la tarea 9 (2 KB leídos de 85,9 KB).
        if captura.truncada:
            return Veredicto(FUENTE_TRUNCADA,
                             f"la salida de {afirmacion.comando!r} vino truncada")
        aguja = normalizar(afirmacion.linea)
        for linea in captura.salida.splitlines():
            if normalizar(linea) == aguja:
                return Veredicto(RESPALDADA, "")
        return Veredicto(SIN_RESPALDO,
                         f"la línea citada no está en la salida de {afirmacion.comando!r}")
    return Veredicto(FUENTE_INEXISTENTE,
                     f"no se corrió el comando {afirmacion.comando!r}")
```

- [ ] **Paso 4: correr los tests y verlos pasar**

Correr: `PYTHONPATH=. /home/fruiz/jax/.venv/bin/python -m pytest tests/test_ejecutor_cita.py -q`
Esperado: **8 passed**.

- [ ] **Paso 5: ejercitar el verificador contra sí mismo (un control que no falla no valida)**

Cambiá `normalizar` para que devuelva `linea.lower()` además de colapsar, corré los tests, y
comprobá que **`test_las_mayusculas_SI_deciden` se pone rojo**. Restaurá.
Después cambiá el orden en `verificar` para mirar el contenido antes que `truncada`, y
comprobá que **`test_una_captura_truncada_no_respalda_NADA` se pone rojo**. Restaurá.
**Reportá las dos salidas.** Si alguna no se pone roja, el test no prueba nada y hay que
rehacerlo.

- [ ] **Paso 6: meter el archivo en las DOS listas del piso**

En `.github/workflows/policy.yml`, job `tests-puros`: agregá `tests/test_ejecutor_cita.py`
a la lista de `python -m pytest -v` **y** a la lista del contador. Medí el piso nuevo con el
runner (no lo calcules) y dejalo escrito con la fecha y el porqué.

- [ ] **Paso 7: commit**

```bash
git add jax/ejecutor/__init__.py jax/ejecutor/cita.py tests/test_ejecutor_cita.py .github/workflows/policy.yml
git commit -m "feat(ejecutor): el verificador de citas, puro y sin nada que calibrar"
```

---

### Task 2: `captura.py` — la salida cruda y su procedencia

**Archivos:**
- Crear: `jax/ejecutor/captura.py`
- Test: `tests/test_ejecutor_captura.py`

**Interfaces:**
- Consume: `Captura` de `jax.ejecutor.cita` (el mismo tipo, no uno paralelo).
- Produce: `correr(comando, maquina, tope_bytes=...) -> CapturaCompleta`, donde
  `CapturaCompleta` agrega `maquina`, `codigo`, `stderr`, `bytes_totales` y `momento` a los
  campos de `Captura`. Y `a_captura(completa) -> Captura` para pasársela a `cita.verificar`.

- [ ] **Paso 1: escribir los tests que fallan**

```python
# tests/test_ejecutor_captura.py
import pytest
from jax.ejecutor.captura import correr, a_captura


def test_captura_la_salida_y_el_codigo():
    c = correr("echo hola", maquina="local")
    assert c.codigo == 0
    assert "hola" in c.salida
    assert c.truncada is False
    assert c.maquina == "local"


def test_un_comando_que_falla_no_lanza_y_queda_registrado():
    c = correr("exit 3", maquina="local")
    assert c.codigo == 3
    assert c.truncada is False


def test_pasado_el_tope_la_captura_queda_MARCADA_como_truncada():
    """El truncado se marca, no se esconde: es el dato del que depende la
    regla de §2.4 del spec."""
    c = correr("seq 1 100000", maquina="local", tope_bytes=1024)
    assert c.truncada is True
    assert c.bytes_totales > 1024
    assert len(c.salida.encode()) <= 1024


def test_a_captura_conserva_la_marca_de_truncado():
    c = correr("seq 1 100000", maquina="local", tope_bytes=1024)
    assert a_captura(c).truncada is True
```

- [ ] **Paso 2: correr y ver el rojo**

Correr: `PYTHONPATH=. /home/fruiz/jax/.venv/bin/python -m pytest tests/test_ejecutor_captura.py -q`
Esperado: `ModuleNotFoundError: No module named 'jax.ejecutor.captura'`.

- [ ] **Paso 3: implementar**

```python
# jax/ejecutor/captura.py
"""Corre un comando y guarda su salida COMPLETA con procedencia.

Spec: §3.2. El truncado se marca, nunca se esconde: `cita.verificar`
rechaza toda afirmación que cite una captura truncada.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone

from jax.ejecutor.cita import Captura

TOPE_BYTES_POR_DEFECTO = 1_000_000


@dataclass(frozen=True)
class CapturaCompleta:
    maquina: str
    comando: str
    codigo: int
    salida: str
    stderr: str
    truncada: bool
    bytes_totales: int
    momento: str


def _ahora_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def correr(comando: str, maquina: str, tope_bytes: int = TOPE_BYTES_POR_DEFECTO) -> CapturaCompleta:
    p = subprocess.run(comando, shell=True, capture_output=True, text=True)
    crudo = p.stdout
    total = len(crudo.encode())
    truncada = total > tope_bytes
    salida = crudo.encode()[:tope_bytes].decode(errors="ignore") if truncada else crudo
    return CapturaCompleta(
        maquina=maquina, comando=comando, codigo=p.returncode, salida=salida,
        stderr=p.stderr, truncada=truncada, bytes_totales=total, momento=_ahora_iso())


def a_captura(completa: CapturaCompleta) -> Captura:
    return Captura(comando=completa.comando, salida=completa.salida,
                   truncada=completa.truncada)
```

- [ ] **Paso 4: correr y ver verde**

Esperado: **4 passed**.

- [ ] **Paso 5: commit** (y agregar el test a las DOS listas del piso, como en la Task 1 paso 6)

```bash
git add jax/ejecutor/captura.py tests/test_ejecutor_captura.py .github/workflows/policy.yml
git commit -m "feat(ejecutor): captura de salidas con procedencia y marca de truncado"
```

---

### Task 3: V1 y V2 — medir contra el corpus de U3 que ya existe

**Archivos:**
- Crear: `scripts/ejecutor_fase2/reproducir_u3.py`
- Test: `tests/test_ejecutor_reproduccion_u3.py`

**Interfaces:**
- Consume: `cita.verificar`, `cita.Captura`, `cita.Afirmacion`.
- Produce: un JSON con, por cada una de las 11 invenciones, el veredicto del verificador.

**Esta tarea no corre el examen de nuevo.** El corpus está en disco desde el 2026-09-15 en
`~/ejecutor-fase0/resultados/examen/qwen/` y la calificación vigente en
`~/ejecutor-fase0/resultados/calificacion_tres_capas.json`. **No hace falta el gate G1 ni cortar
`jax_local`.**

- [ ] **Paso 1: escribir el test que fija V1 y V2**

```python
# tests/test_ejecutor_reproduccion_u3.py
"""V1 y V2 del spec §6, contra el corpus real de U3.

Se saltea si el corpus no está en disco: vive fuera del repo porque tiene
datos de clientes (baranda de Fase 0 §3).
"""
import json
import os
import pathlib
import pytest

CORPUS = pathlib.Path(os.path.expanduser("~/ejecutor-fase0/resultados"))
pytestmark = pytest.mark.skipif(not CORPUS.exists(), reason="corpus de U3 no está en disco")


def test_v1_las_once_invenciones_se_rechazan():
    from scripts.ejecutor_fase2.reproducir_u3 import reproducir
    r = reproducir(CORPUS)
    assert r["invenciones_totales"] == 11, "la calificación vigente cuenta 11"
    assert r["invenciones_rechazadas"] == 11, (
        f"el verificador dejó pasar {11 - r['invenciones_rechazadas']}: "
        f"{r['invenciones_que_pasaron']}")


def test_v2_cero_falsos_positivos_en_las_tareas_limpias():
    from scripts.ejecutor_fase2.reproducir_u3 import reproducir
    r = reproducir(CORPUS)
    assert r["tareas_limpias"] == [1, 6, 7, 8, 10], "las cinco con 0 invenciones"
    assert r["falsos_positivos"] == 0, (
        f"rechazó trabajo bueno: {r['afirmaciones_buenas_rechazadas']}")
```

- [ ] **Paso 2: correr y ver el rojo**

Esperado: `ModuleNotFoundError` o `skipped` si el corpus no está. **Si sale `skipped`, parar y
avisar**: sin corpus esta tarea no se puede hacer y no hay que inventar un sustituto.

- [ ] **Paso 3: escribir `reproducir_u3.py`**

Lee `calificacion_tres_capas.json`, toma el campo `detalle` de cada tarea (cada invención trae
`que` y `capa`), ubica la transcripción de esa tarea en `examen/qwen/`, arma las `Captura` de
los comandos que el modelo corrió y las `Afirmacion` de lo que dijo, y corre `verificar` sobre
cada una. Devuelve el dict que los tests esperan.

**Las 11 invenciones a reproducir**, de `calificacion_tres_capas.json`:
tarea 2 (1) «~91 GB» por 89 GiB · tarea 3 (1) «131.074» por 131072 · tarea 4 (1) · tarea 5 (5)
8188→Docker, 11332→DNS local, 24842→socket efímero, 15222→puente SSH, 3001 como público
cuando la salida decía `172.16.20.11:3001` · tarea 9 (3).

- [ ] **Paso 4: correr y ver verde**

Esperado: **2 passed**. Si V1 falla, el verificador tiene un hueco: **arreglá el verificador,
no el umbral.** Si V2 falla, hay un falso positivo y es un problema real — reportalo antes de
tocar nada.

- [ ] **Paso 5: commit**

```bash
git add scripts/ejecutor_fase2/ tests/test_ejecutor_reproduccion_u3.py
git commit -m "test(ejecutor): V1 y V2 contra el corpus real de U3"
```

---

### Task 4: `hechos.py` — hechos del sistema, derivados e inyectados

**Archivos:**
- Crear: `jax/ejecutor/hechos.py`
- Test: `tests/test_ejecutor_hechos.py`

**Interfaces:**
- Consume: `captura.correr`.
- Produce: `derivar(inventario, ahora) -> list[Hecho]` y `bloque(hechos, ahora) -> str`, con
  `Hecho(nombre, valor, comando, momento)`.

- [ ] **Paso 1: escribir los tests que fallan**

```python
# tests/test_ejecutor_hechos.py
from jax.ejecutor.hechos import Hecho, bloque, vigente


def test_un_hecho_vencido_no_entra_al_bloque():
    """TTL: un hecho de hace una hora presentado como actual es una mentira
    nueva (riesgo 4 del spec)."""
    h = Hecho(nombre="uptime", valor="38 min", comando="uptime -p",
              momento="2026-09-16T10:00:00+00:00")
    assert vigente(h, ahora="2026-09-16T10:00:30+00:00", ttl_s=60) is True
    assert vigente(h, ahora="2026-09-16T11:00:00+00:00", ttl_s=60) is False


def test_el_bloque_muestra_el_comando_y_la_hora_de_cada_hecho():
    h = Hecho(nombre="uptime", valor="38 min", comando="uptime -p",
              momento="2026-09-16T10:00:00+00:00")
    texto = bloque([h], ahora="2026-09-16T10:00:10+00:00")
    assert "uptime -p" in texto and "2026-09-16T10:00:00+00:00" in texto


def test_un_hecho_que_no_se_pudo_derivar_se_omite_y_se_dice():
    texto = bloque([], ahora="2026-09-16T10:00:00+00:00", no_derivados=["disco"])
    assert "disco" in texto and "no se pudo derivar" in texto
```

- [ ] **Paso 2: correr y ver el rojo.** Esperado: `ModuleNotFoundError`.

- [ ] **Paso 3: implementar** `Hecho`, `vigente` (compara ISO-8601 con zona) y `bloque`
  (renderiza `nombre = valor  (comando, momento)` por línea, más una línea por cada no derivado).

- [ ] **Paso 4: correr y ver verde.** Esperado: **3 passed**.

- [ ] **Paso 5: commit** (y las DOS listas del piso)

```bash
git add jax/ejecutor/hechos.py tests/test_ejecutor_hechos.py .github/workflows/policy.yml
git commit -m "feat(ejecutor): hechos del sistema derivados, con TTL y procedencia"
```

---

### Task 5: el transporte, y V4 (fail-closed)

**Archivos:**
- Crear: `jax/ejecutor/transporte.py`
- Test: `tests/test_ejecutor_transporte.py`

**Interfaces:**
- Consume: `cita.verificar`, `captura.a_captura`, `hechos.bloque`.
- Produce: `entregar(afirmaciones, capturas) -> Entrega`, con
  `Entrega(crudas, afirmaciones_respaldadas, descartadas)`.

- [ ] **Paso 1: escribir los tests que fallan**

```python
# tests/test_ejecutor_transporte.py
import pytest
from jax.ejecutor.cita import Afirmacion, Captura
from jax.ejecutor import transporte

CAPTURAS = [Captura(comando="uptime -p", salida="up 38 minutes", truncada=False)]


def test_solo_salen_las_afirmaciones_respaldadas():
    buena = Afirmacion(texto="lleva 38 minutos", comando="uptime -p", linea="up 38 minutes")
    mala = Afirmacion(texto="lleva casi un día", comando="uptime -p", linea="up 23 hours")
    e = transporte.entregar([buena, mala], CAPTURAS)
    assert [a.texto for a in e.afirmaciones_respaldadas] == ["lleva 38 minutos"]
    assert len(e.descartadas) == 1


def test_las_salidas_crudas_se_entregan_SIEMPRE():
    """El piso de §2.3: si no puede citar nada, no queda mudo."""
    mala = Afirmacion(texto="inventado", comando="uptime -p", linea="up 23 hours")
    e = transporte.entregar([mala], CAPTURAS)
    assert e.afirmaciones_respaldadas == []
    assert e.crudas == CAPTURAS


def test_v4_si_el_verificador_falla_NO_sale_ninguna_afirmacion(monkeypatch):
    """Fail-closed: un verificador caído no puede volverse un pase libre."""
    def explota(*a, **k):
        raise RuntimeError("verificador caído")
    monkeypatch.setattr(transporte.cita, "verificar", explota)
    buena = Afirmacion(texto="lleva 38 minutos", comando="uptime -p", linea="up 38 minutes")
    e = transporte.entregar([buena], CAPTURAS)
    assert e.afirmaciones_respaldadas == []
    assert e.crudas == CAPTURAS
```

- [ ] **Paso 2: correr y ver el rojo.** Esperado: `ModuleNotFoundError`.

- [ ] **Paso 3: implementar.** El `except` del fail-closed lleva la marca **en su línea**:

```python
        except Exception as error:  # fail-soft: el turno entrega las crudas; fail-CLOSED para las afirmaciones
```

- [ ] **Paso 4: correr y ver verde.** Esperado: **3 passed**.

- [ ] **Paso 5: ejercitar V4 de verdad.** Quitá el `try/except` y comprobá que
  `test_v4_...` se pone rojo por `RuntimeError` y no por otra cosa. Restaurá. Reportá la salida.

- [ ] **Paso 6: commit** (y las DOS listas del piso)

```bash
git add jax/ejecutor/transporte.py tests/test_ejecutor_transporte.py .github/workflows/policy.yml
git commit -m "feat(ejecutor): el transporte deja caer lo que no cita, y es fail-closed"
```

---

## Auto-revisión de este plan

- **Cobertura del spec:** §2.1 → Task 5 · §2.2 → Task 1 (no hay detector que calibrar) ·
  §2.3 → Task 5 (`test_las_salidas_crudas_se_entregan_SIEMPRE`) · §2.4 → Task 1
  (`test_una_captura_truncada...`) · §3.1 → Task 4 · §3.2 → Task 2 · §3.3 → Task 1 ·
  §3.4 → **plan aparte** · §5 → Tasks 1 y 5 · §6 V1/V2 → Task 3 · V3 → plan aparte ·
  V4 → Task 5.
- **Sin placeholders:** todos los pasos de código llevan el código.
- **Tipos consistentes:** `Captura` se define una sola vez, en `cita.py`, y `captura.py` la
  importa en vez de declarar una paralela. `a_captura` es el único puente.
- **Riesgo 1 del spec (tasa de rechazo por formato)** no tiene tarea propia: se mide recién
  cuando el modelo produzca afirmaciones de verdad, que es después de este plan. Queda
  anotado, no olvidado.
