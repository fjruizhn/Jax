# Ejecutor SP1 · Plan 2 · C3 registro intocable y cerco de red de la jaula

> **Para agentes ejecutores:** SUB-SKILL REQUERIDA: usar `superpowers:subagent-driven-development`
> (recomendado) o `superpowers:executing-plans`. Los pasos usan casillas (`- [ ]`).

**Objetivo:** que cada herramienta que el cerebro del Ejecutor pide, y cada resultado que la jaula devuelve,
quede en un registro encadenado **antes** de que la herramienta pueda correr; que desde la jaula ese registro
no se pueda escribir ni por archivo ni por red; y que un tercero pueda comprobarlo con un script.

**Arquitectura:** el proxy con carril (`jax/ejecutor/proxy_carril.py`, cuenta `fruiz`) es el único camino del
cerebro. Lee al pasar el stream de la API de mensajes (`lectura.py`), anota cada `tool_use` en
`registro.jsonl` (cadena de sha256, `O_APPEND`, `fsync`, `chattr +a`) **antes** de reenviar el trozo que lo
completa, y cada `tool_result` nuevo de la petición siguiente. Si no puede anotar, corta: la herramienta nunca
llega. Un cerco `nftables` por `meta skuid` deja a la cuenta `axioma` hablar sólo con el proxy, el puerto del
canario y los SSH del inventario.

**Tech stack:** Python 3.12, `h11` + `httpx` (ya en el proxy), `fcntl`/`os` para `O_APPEND`, `fsync` y
`FS_IOC_GETFLAGS`; `nftables`; systemd.

**Spec:** `docs/superpowers/specs/2026-09-15-ejecutor-design.md` §4 C3; `2026-09-16-ejecutor-fase2-design.md`
§3.4 bis (el proxy) y §6.2.2 (tope ≥ 60 s). Decisión D-SP1-2 del índice
`docs/superpowers/plans/2026-09-17-ejecutor-contratos-0-indice.md` (por qué el proxy y no un endpoint de LAS MANOS).

## Global Constraints

- **Nunca cambiar de rama en `/home/fruiz/jax`.** Worktree: `/home/fruiz/worktrees/jax-sp1-c3`, rama
  `feat/ejecutor-c3` desde `master` **con el plan 1 mergeado**. Siempre `git -C <ruta absoluta>`.
- **`/etc/jax/.env` apunta a PRODUCCIÓN.** Ningún test la carga; los scripts de la Task 7 la leen a propósito.
- **P10:** todo `except` amplio sin `raise`, o con cuerpo `pass`, lleva `# fail-soft: <razón específica>` en la
  MISMA línea. `python3 policy/tests/test_no_fail_open_except.py` verde.
- **Piso de `tests-puros`:** tests nuevos en **las dos** listas del job (`pytest -v` y «Piso exacto»); el `grep`
  del piso se fija con **lo que cuenta el runner**, con su línea de historia `N -> M`.
- **Mutaciones:** `PYTHONDONTWRITEBYTECODE=1`; `cp <archivo> <archivo>.mut-bak` antes; `cmp` después; nunca
  `git checkout` para restaurar.
- **Tests con procesos:** `multiprocessing.get_context("fork")` explícito; vida de procesos con
  `ps -o pid= -p <pid>`, nunca `pgrep -f`.
- **Cero strings visibles hardcodeados:** códigos (`Motivo`, `formato.campos`), nunca frases.
- **Sin hardcoding:** rutas, puertos, cuenta y sondas desde `JAX_EJECUTOR_*` / `JAX_PROXY_CARRIL_*`.
- **Guardia `no-naked-claude-subprocess`:** ningún archivo nuevo de este plan menciona `claude` en un literal
  **y** lanza un subproceso. Los módulos de C3 no lanzan subprocesos; entran a la cuenta con
  `cuenta_axioma.correr_en_la_cuenta` (plan 1).
- **Nada destructivo sobre .10/.11/.20.** Este plan no toca esas máquinas.
- **Cuatro del rendimiento:** el registro escribe con `write`+`fsync` **fuera del event loop**
  (`asyncio.to_thread`); `tests/test_no_blocking_in_async.py` sigue verde. La sobrecarga del proxy se mide con
  carga en la Task 7 (criterio pre-registrado) antes de dar C3 por cerrado.
- **Datos de clientes:** el registro guarda comandos y el tamaño/sha256 de los resultados, **no** el contenido de
  los resultados. Directorio `0750` de `fruiz`, archivo `0640`.
- Commits en castellano, terminados con `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`.

## Mapa de archivos

| Archivo | Acción | Responsabilidad |
|---|---|---|
| `jax/ejecutor/contratos/registro.py` | Crear | `Registro` (append encadenado), `verificar_cadena`, `RegistroCorrupto` |
| `jax/ejecutor/contratos/lectura.py` | Crear | `LectorSSE`, `herramientas_de_mensaje`, `resultados_de_peticion`, eventos para el registro |
| `jax/ejecutor/proxy_carril.py` | Modificar | `Config.registro`, anotar antes de reenviar, `accept-encoding: identity`, 502 si no puede |
| `jax/ejecutor/contratos/cerco.py` | Crear | render del ruleset `nftables` desde la política |
| `jax/ejecutor/contratos/canario_c3.py` | Crear | `verificar_c3`: desde la jaula no se escribe el registro ni se alcanza lo prohibido |
| `ops/ejecutor/ejecutor-cerco.service`, `ops/ejecutor/jax-ejecutor-proxy.service`, `ops/ejecutor/instalar_registro_y_cerco.sh` | Crear | instalación root en hall9000 |
| `scripts/ejecutor_contratos/probar_c3.py` | Crear | prueba real re-ejecutable |
| `tests/test_ejecutor_contratos_registro.py`, `…_lectura.py`, `…_cerco.py`, `…_canario_c3.py`, `tests/test_ejecutor_proxy_registro.py`, `tests/test_ejecutor_carril_solo_en_el_proxy.py` | Crear | tests puros |
| `tests/test_ejecutor_proxy_carril.py` | Modificar | fixture `Proxy` y tests de config con `JAX_EJECUTOR_REGISTRO` |
| `.github/workflows/policy.yml` | Modificar | listas y piso de `tests-puros` |
| `DEUDA.md` | Modificar | Hyde alcanza LAS MANOS sin autenticación (fecha 2026-09-24) |

## Interfaces

```python
# jax/ejecutor/contratos/registro.py
GENESIS = "0" * 64
class RegistroCorrupto(RuntimeError): codigo: str
@dataclass(frozen=True)
class Verificacion: ok: bool; lineas: int; primer_error: int | None; codigo: str | None
class Registro:
    def __init__(self, ruta)                     # abre O_APPEND; lee la cola; RegistroCorrupto si no cuadra
    def anotar(self, evento: dict) -> int        # BLOQUEANTE (write+fsync); devuelve n
    def cerrar(self) -> None
def verificar_cadena(ruta) -> Verificacion

# jax/ejecutor/contratos/lectura.py
TOPE_ENTRADA_BYTES = 65536
@dataclass(frozen=True) class HerramientaPedida: tool_use_id: str | None; nombre: str | None; entrada: object; entrada_legible: bool
@dataclass(frozen=True) class ResultadoDevuelto: tool_use_id: str | None; es_error: bool; bytes: int; sha256: str
class LectorSSE: def alimentar(self, trozo: bytes) -> list[HerramientaPedida]
def herramientas_de_mensaje(cuerpo: bytes) -> list[HerramientaPedida] | None
def resultados_de_peticion(cuerpo: bytes) -> list[ResultadoDevuelto] | None
def evento_de_pedida(p: HerramientaPedida, ruta: str) -> dict
def evento_de_resultado(r: ResultadoDevuelto, ruta: str) -> dict

# jax/ejecutor/proxy_carril.py (cambios)
REGISTRO_FALLO = "registro_fallo"; REGISTRO_ILEGIBLE = "registro_ilegible"
@dataclass(frozen=True) class Config: upstream; raiz; tope_s; host; puerto; registro: Path
# config_desde_entorno exige JAX_EJECUTOR_REGISTRO (absoluta)

# jax/ejecutor/contratos/cerco.py
TABLA = "ejecutor_cerco"
def renderizar(uid: int, puertos_locales: tuple[int, ...], destinos_ssh: tuple[tuple[str, int], ...]) -> str
def desde_politica(p: politica.Politica, uid: int, puertos_locales: tuple[int, ...]) -> str

# jax/ejecutor/contratos/canario_c3.py
FS_IOC_GETFLAGS = 0x80086601; FS_APPEND_FL = 0x20
def banderas(ruta) -> int
async def verificar_c3(c, *, registro: Path, puerto_proxy: int, sondas: tuple[int, ...],
                       correr=..., leer_banderas=banderas) -> tuple[Fallo, ...]
```

Formato de un evento del registro (una línea, JSON ASCII con claves ordenadas):
`{"evento": "herramienta_pedida"|"resultado_devuelto"|"registro_abierto"|"peticion_ilegible", "n": int, "prev": sha256hex, "momento": ISO-8601 UTC, …}`
- `herramienta_pedida`: `tool_use_id`, `herramienta`, `ruta`, y `entrada` (si su JSON cabe en `TOPE_ENTRADA_BYTES`) o
  `entrada_sha256` + `entrada_bytes` + `entrada_inicio` (los primeros 16 KiB del JSON); `entrada_legible`.
- `resultado_devuelto`: `tool_use_id`, `es_error`, `bytes`, `sha256` (del JSON canónico del `content`), `ruta`.

---

### Task 1: el registro encadenado

**Files:**
- Create: `/home/fruiz/worktrees/jax-sp1-c3/jax/ejecutor/contratos/registro.py`
- Test: `/home/fruiz/worktrees/jax-sp1-c3/tests/test_ejecutor_contratos_registro.py`

**Interfaces:** Consume nada. Produce `Registro`, `verificar_cadena`, `Verificacion`, `RegistroCorrupto`, `GENESIS`.

- [ ] **Step 1: Escribir los tests que fallan**

```python
# tests/test_ejecutor_contratos_registro.py
"""Registro intocable del Ejecutor (C3): cadena de sha256, O_APPEND, fsync.
Cada forma de tocarlo por detrás se detecta; un registro que no cuadra no se abre."""
import fcntl
import json
import os
import threading

import pytest

from jax.ejecutor.contratos import registro as R


def _lineas(ruta):
    return ruta.read_bytes().splitlines()


def test_anota_y_la_cadena_cuadra(tmp_path):
    ruta = tmp_path / "registro.jsonl"
    reg = R.Registro(ruta)
    assert [reg.anotar({"evento": "x", "i": i}) for i in range(3)] == [1, 2, 3]
    reg.cerrar()
    assert R.verificar_cadena(ruta) == R.Verificacion(True, 3, None, None)
    primera = json.loads(_lineas(ruta)[0])
    assert primera["prev"] == R.GENESIS and primera["n"] == 1 and primera["momento"].endswith("+00:00")


def test_reabrir_continua_la_cadena(tmp_path):
    ruta = tmp_path / "registro.jsonl"
    reg = R.Registro(ruta)
    reg.anotar({"evento": "a"})
    reg.cerrar()
    reg = R.Registro(ruta)
    assert reg.anotar({"evento": "b"}) == 2
    reg.cerrar()
    assert R.verificar_cadena(ruta).ok is True


def test_abre_en_modo_append(tmp_path):
    reg = R.Registro(tmp_path / "registro.jsonl")
    try:
        assert fcntl.fcntl(reg._fd, fcntl.F_GETFL) & os.O_APPEND
    finally:
        reg.cerrar()


def _tres(tmp_path):
    ruta = tmp_path / "registro.jsonl"
    reg = R.Registro(ruta)
    for i in range(3):
        reg.anotar({"evento": "x", "comando": f"cmd-{i}"})
    reg.cerrar()
    return ruta


def test_editar_una_linea_se_detecta_en_la_siguiente(tmp_path):
    ruta = _tres(tmp_path)
    lineas = _lineas(ruta)
    lineas[1] = lineas[1].replace(b"cmd-1", b"cmd-X")
    ruta.write_bytes(b"\n".join(lineas) + b"\n")
    assert R.verificar_cadena(ruta) == R.Verificacion(False, 3, 3, "prev_no_cuadra")


def test_borrar_una_linea_se_detecta(tmp_path):
    ruta = _tres(tmp_path)
    lineas = _lineas(ruta)
    ruta.write_bytes(lineas[0] + b"\n" + lineas[2] + b"\n")
    assert R.verificar_cadena(ruta) == R.Verificacion(False, 2, 2, "n_no_cuadra")


def test_linea_incompleta_no_abre_y_se_reporta(tmp_path):
    ruta = _tres(tmp_path)
    with open(ruta, "ab") as f:
        f.write(b'{"evento":"a medias"')
    with pytest.raises(R.RegistroCorrupto) as e:
        R.Registro(ruta)
    assert e.value.codigo == "ultima_linea_incompleta"
    assert R.verificar_cadena(ruta) == R.Verificacion(False, 4, 4, "ultima_linea_incompleta")


def test_cola_ilegible_no_abre(tmp_path):
    ruta = tmp_path / "registro.jsonl"
    ruta.write_bytes(b"no es json\n")
    with pytest.raises(R.RegistroCorrupto) as e:
        R.Registro(ruta)
    assert e.value.codigo == "ultima_linea_ilegible"


def test_escritores_concurrentes_no_rompen_la_cadena(tmp_path):
    ruta = tmp_path / "registro.jsonl"
    reg = R.Registro(ruta)
    hilos = [threading.Thread(target=lambda: [reg.anotar({"evento": "x"}) for _ in range(200)]) for _ in range(4)]
    for h in hilos:
        h.start()
    for h in hilos:
        h.join(30)
    reg.cerrar()
    assert R.verificar_cadena(ruta) == R.Verificacion(True, 800, None, None)


def test_escritura_parcial_deja_el_registro_roto_y_lo_dice(tmp_path, monkeypatch):
    reg = R.Registro(tmp_path / "registro.jsonl")
    monkeypatch.setattr(R.os, "write", lambda fd, datos: len(datos) - 1)
    with pytest.raises(OSError):
        reg.anotar({"evento": "x"})
    monkeypatch.undo()
    with pytest.raises(OSError):
        reg.anotar({"evento": "y"})
    reg.cerrar()


def test_un_evento_no_puede_pisar_la_cadena(tmp_path):
    reg = R.Registro(tmp_path / "registro.jsonl")
    reg.anotar({"evento": "x", "n": 999, "prev": "falso"})
    reg.cerrar()
    assert R.verificar_cadena(tmp_path / "registro.jsonl").ok is True
```

- [ ] **Step 2: Correr y ver que fallan**

Run: `cd /home/fruiz/worktrees/jax-sp1-c3 && pwd && git branch --show-current && PYTHONPATH=.:las_manos python -m pytest tests/test_ejecutor_contratos_registro.py -v`
Expected: FAIL en la colección (`cannot import name 'registro'`).

- [ ] **Step 3: Implementar**

```python
# jax/ejecutor/contratos/registro.py
"""Registro intocable del Ejecutor (C3). Lo escribe el PROXY, como `fruiz`, fuera de la jaula.

Spec 2026-09-15 §4 C3; decisión D-SP1-2 del índice de SP1: no lo alimenta la jaula
(podría no llamarlo, o llamarlo con lo que quisiera); lo alimenta lo que el cerebro
pide y la jaula devuelve, visto al pasar por el proxy.

Intocable por tres capas:
1. otra cuenta: directorio 0750 de `fruiz`; `axioma` ni lo lee;
2. `chattr +a` (root, al instalar): ni `fruiz` lo trunca ni lo reescribe;
3. cadena: cada línea lleva `n` (1, 2, 3…) y `prev`, el sha256 de los bytes de la
   línea anterior (GENESIS para la primera). `verificar_cadena` da la primera línea
   donde se rompe: una línea editada se ve en la SIGUIENTE (`prev_no_cuadra`); una
   borrada o insertada, donde el número salta (`n_no_cuadra`).

`anotar` es BLOQUEANTE (write + fsync): desde async, con `asyncio.to_thread`. Un lock
lo serializa: la cadena exige un escritor ordenado. Una escritura parcial deja el
registro marcado como roto y toda escritura posterior falla: un registro con un hueco
no sigue aceptando como si nada.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

GENESIS = "0" * 64
_LEER_COLA = 1 << 20


class RegistroCorrupto(RuntimeError):
    def __init__(self, codigo: str):
        super().__init__(codigo)
        self.codigo = codigo


@dataclass(frozen=True)
class Verificacion:
    ok: bool
    lineas: int
    primer_error: int | None
    codigo: str | None


def _hash(linea: bytes) -> str:
    return hashlib.sha256(linea).hexdigest()


def _ultima_linea(fd: int) -> bytes | None:
    tamanio = os.fstat(fd).st_size
    if tamanio == 0:
        return None
    desde = max(0, tamanio - _LEER_COLA)
    datos = os.pread(fd, tamanio - desde, desde)
    if not datos.endswith(b"\n"):
        raise RegistroCorrupto("ultima_linea_incompleta")
    cuerpo = datos[:-1]
    corte = cuerpo.rfind(b"\n")
    if corte < 0 and desde > 0:
        raise RegistroCorrupto("linea_demasiado_larga")
    return cuerpo[corte + 1:]


class Registro:
    def __init__(self, ruta):
        self.ruta = Path(ruta)
        self._lock = threading.Lock()
        self._roto = False
        self._fd = os.open(self.ruta, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_CLOEXEC, 0o640)
        try:
            lector = os.open(self.ruta, os.O_RDONLY | os.O_CLOEXEC)
            try:
                ultima = _ultima_linea(lector)
            finally:
                os.close(lector)
            if ultima is None:
                self._n, self._prev = 0, GENESIS
            else:
                try:
                    doc = json.loads(ultima)
                except ValueError:
                    raise RegistroCorrupto("ultima_linea_ilegible") from None
                if not isinstance(doc, dict) or isinstance(doc.get("n"), bool) or not isinstance(doc.get("n"), int):
                    raise RegistroCorrupto("ultima_linea_ilegible")
                self._n, self._prev = doc["n"], _hash(ultima)
        except BaseException:
            os.close(self._fd)
            raise

    def anotar(self, evento: dict) -> int:
        with self._lock:
            if self._roto:
                raise OSError("registro_roto")
            n = self._n + 1
            doc = {**evento, "n": n, "prev": self._prev, "momento": datetime.now(timezone.utc).isoformat()}
            linea = json.dumps(doc, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
            datos = linea + b"\n"
            if os.write(self._fd, datos) != len(datos):
                self._roto = True
                raise OSError("escritura_parcial")
            os.fsync(self._fd)
            self._n, self._prev = n, _hash(linea)
            return n

    def cerrar(self) -> None:
        os.close(self._fd)


def verificar_cadena(ruta) -> Verificacion:
    prev, n = GENESIS, 0
    with open(ruta, "rb") as f:
        for n, cruda in enumerate(f, start=1):
            if not cruda.endswith(b"\n"):
                return Verificacion(False, n, n, "ultima_linea_incompleta")
            linea = cruda[:-1]
            try:
                doc = json.loads(linea)
            except ValueError:
                return Verificacion(False, n, n, "linea_ilegible")
            if not isinstance(doc, dict) or doc.get("n") != n:
                return Verificacion(False, n, n, "n_no_cuadra")
            if doc.get("prev") != prev:
                return Verificacion(False, n, n, "prev_no_cuadra")
            prev = _hash(linea)
    return Verificacion(True, n, None, None)
```

Nota para quien revisa `test_borrar_una_linea_se_detecta`: al borrar la línea 2, la que queda en segundo lugar
tiene `n=3` → `n_no_cuadra` en la línea 2, antes de mirar `prev`. Y en `test_un_evento_no_puede_pisar_la_cadena`
los campos del evento van **antes** en el dict: `n`, `prev` y `momento` del registro los pisan.

- [ ] **Step 4: Correr y ver que pasan**

Run: `cd /home/fruiz/worktrees/jax-sp1-c3 && PYTHONPATH=.:las_manos python -m pytest tests/test_ejecutor_contratos_registro.py -v && python3 policy/tests/test_no_fail_open_except.py`
Expected: 10 PASS; P10 verde (el `except BaseException` relanza).

- [ ] **Step 5: Mutación (neutralizar dejando presente)**

Con backup y `cmp`: en `verificar_cadena`, `if doc.get("prev") != prev:` → `if doc.get("prev") != prev and False:`.
Expected: FAIL `test_editar_una_linea_se_detecta_en_la_siguiente`. Restaurar.

- [ ] **Step 6: Commit**

```bash
git -C /home/fruiz/worktrees/jax-sp1-c3 add jax/ejecutor/contratos/registro.py tests/test_ejecutor_contratos_registro.py
git -C /home/fruiz/worktrees/jax-sp1-c3 commit -m "feat(ejecutor): registro encadenado de C3, append-only con fsync y verificación de cadena

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: leer lo que el cerebro pide y lo que la jaula devuelve

**Files:**
- Create: `/home/fruiz/worktrees/jax-sp1-c3/jax/ejecutor/contratos/lectura.py`
- Test: `/home/fruiz/worktrees/jax-sp1-c3/tests/test_ejecutor_contratos_lectura.py`

**Interfaces:** Consume `canario_upstream._stream_tool_use` (plan 1) sólo en tests, como SSE real de referencia.
Produce `LectorSSE`, `HerramientaPedida`, `ResultadoDevuelto`, `herramientas_de_mensaje`,
`resultados_de_peticion`, `evento_de_pedida`, `evento_de_resultado`, `TOPE_ENTRADA_BYTES`.

**Medido (2026-09-17, arnés 2.1.273 contra el upstream falso del plan 1):** los `tool_result` NO van en el último
mensaje (el último es `role: system`), y la historia se repite entera en cada petición. Por eso se recorren
todos los mensajes y el proxy anota cada `tool_use_id` una sola vez.

- [ ] **Step 1: Escribir los tests que fallan**

```python
# tests/test_ejecutor_contratos_lectura.py
"""Lectura de la API de mensajes para C3: un tool_use se reconoce COMPLETO en el
trozo que trae su content_block_stop, sin importar cómo llegue partido."""
import hashlib
import json

from jax.ejecutor.contratos import lectura as L
from jax.ejecutor.contratos.canario_upstream import _stream_tool_use, guion_bash

SSE = _stream_tool_use(guion_bash("toolu_1", "df -h /"))
ESPERADA = [L.HerramientaPedida("toolu_1", "Bash", {"command": "df -h /"}, True)]


def test_stream_entero():
    assert L.LectorSSE().alimentar(SSE) == ESPERADA


def test_byte_a_byte_da_lo_mismo_y_solo_al_final():
    lector, vistas, momento = L.LectorSSE(), [], None
    for i in range(len(SSE)):
        nuevas = lector.alimentar(SSE[i:i + 1])
        if nuevas and momento is None:
            momento = i
        vistas += nuevas
    assert vistas == ESPERADA
    assert momento >= SSE.index(b"content_block_stop"), "se anotó antes de que el bloque estuviera completo"


def test_crlf():
    assert L.LectorSSE().alimentar(SSE.replace(b"\n", b"\r\n")) == ESPERADA


def test_datos_que_no_son_json_se_ignoran():
    assert L.LectorSSE().alimentar(b"data: trozo-0\n\ndata: trozo-1\n\n") == []


def test_entrada_rota_se_anota_como_ilegible():
    sse = SSE.replace(b'{\\"command\\": \\"df -h /\\"}', b'{\\"command\\": ')
    (pedida,) = L.LectorSSE().alimentar(sse)
    assert (pedida.tool_use_id, pedida.entrada_legible, pedida.entrada) == ("toolu_1", False, '{"command": ')


def test_entrada_en_el_bloque_de_inicio():
    ev = {"type": "content_block_start", "index": 0,
          "content_block": {"type": "tool_use", "id": "t", "name": "Read", "input": {"file_path": "/etc/hosts"}}}
    sse = f"data: {json.dumps(ev)}\n\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n".encode()
    assert L.LectorSSE().alimentar(sse) == [L.HerramientaPedida("t", "Read", {"file_path": "/etc/hosts"}, True)]


def test_mensaje_no_stream():
    cuerpo = json.dumps({"content": [{"type": "text", "text": "x"},
                                     {"type": "tool_use", "id": "t2", "name": "Bash", "input": {"command": "ls"}}]})
    assert L.herramientas_de_mensaje(cuerpo.encode()) == [L.HerramientaPedida("t2", "Bash", {"command": "ls"}, True)]
    assert L.herramientas_de_mensaje(b"no json") is None


def test_resultados_en_cualquier_mensaje():
    contenido = [{"type": "text", "text": "12G libres"}]
    cuerpo = json.dumps({"messages": [
        {"role": "user", "content": "hola"},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": contenido}]},
        {"role": "system", "content": [{"type": "text", "text": "recordatorio"}]},
    ]}).encode()
    canonico = json.dumps(contenido, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    assert L.resultados_de_peticion(cuerpo) == [
        L.ResultadoDevuelto("t1", False, len(canonico), hashlib.sha256(canonico).hexdigest())]
    assert L.resultados_de_peticion(b"[1") is None


def test_evento_de_pedida_con_entrada_enorme_guarda_huella():
    enorme = L.HerramientaPedida("t", "Write", {"file_path": "/x", "content": "a" * (L.TOPE_ENTRADA_BYTES + 1)}, True)
    ev = L.evento_de_pedida(enorme, "/v1/messages")
    assert "entrada" not in ev
    assert ev["entrada_bytes"] > L.TOPE_ENTRADA_BYTES and len(ev["entrada_sha256"]) == 64
    assert len(ev["entrada_inicio"]) == 16384
    chica = L.evento_de_pedida(ESPERADA[0], "/v1/messages")
    assert chica == {"evento": "herramienta_pedida", "tool_use_id": "toolu_1", "herramienta": "Bash",
                     "entrada": {"command": "df -h /"}, "entrada_legible": True, "ruta": "/v1/messages"}
```

- [ ] **Step 2: Correr y ver que fallan**

Run: `cd /home/fruiz/worktrees/jax-sp1-c3 && PYTHONPATH=.:las_manos python -m pytest tests/test_ejecutor_contratos_lectura.py -v`
Expected: FAIL en la colección (`cannot import name 'lectura'`).

- [ ] **Step 3: Implementar**

```python
# jax/ejecutor/contratos/lectura.py
"""Qué pidió el cerebro y qué devolvió la jaula, leído de la API de mensajes (C3).

- `LectorSSE.alimentar(trozo)` devuelve cada `tool_use` COMPLETO en el trozo que trae
  su `content_block_stop`. El proxy anota ANTES de reenviar ese trozo: el arnés no
  puede ejecutar un bloque que todavía no terminó de recibir.
- Un `data:` que no es JSON se ignora: si el proxy no lo puede leer, el arnés tampoco,
  y no puede ejecutar nada a partir de él.
- `resultados_de_peticion` recorre TODOS los mensajes (medido: el último es `system`).

El contenido de los resultados NO se guarda: tamaño y sha256 (datos de clientes).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

TOPE_ENTRADA_BYTES = 65536
_INICIO_BYTES = 16384


@dataclass(frozen=True)
class HerramientaPedida:
    tool_use_id: str | None
    nombre: str | None
    entrada: object
    entrada_legible: bool


@dataclass(frozen=True)
class ResultadoDevuelto:
    tool_use_id: str | None
    es_error: bool
    bytes: int
    sha256: str


def _canonico(valor) -> bytes:
    return json.dumps(valor, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _pedida(bloque: dict, parciales: list[str]) -> HerramientaPedida:
    texto = "".join(parciales)
    if not texto:
        entrada = bloque.get("input")
        return HerramientaPedida(bloque.get("id"), bloque.get("name"), entrada, isinstance(entrada, dict))
    try:
        entrada = json.loads(texto)
    except ValueError:
        return HerramientaPedida(bloque.get("id"), bloque.get("name"), texto, False)
    return HerramientaPedida(bloque.get("id"), bloque.get("name"), entrada, isinstance(entrada, dict))


class LectorSSE:
    def __init__(self) -> None:
        self._resto = b""
        self._abiertos: dict = {}

    def alimentar(self, trozo: bytes) -> list[HerramientaPedida]:
        self._resto = (self._resto + trozo).replace(b"\r\n", b"\n")
        completas = []
        while True:
            fin = self._resto.find(b"\n\n")
            if fin < 0:
                return completas
            crudo, self._resto = self._resto[:fin], self._resto[fin + 2:]
            datos = b"\n".join(l[5:].lstrip(b" ") for l in crudo.split(b"\n") if l.startswith(b"data:"))
            if not datos:
                continue
            try:
                evento = json.loads(datos)
            except ValueError:
                continue  # ni el proxy ni el arnés pueden leerlo: no puede originar una herramienta
            completas.extend(self._evento(evento))

    def _evento(self, ev) -> list[HerramientaPedida]:
        if not isinstance(ev, dict):
            return []
        tipo, indice = ev.get("type"), ev.get("index")
        if tipo == "content_block_start" and isinstance(ev.get("content_block"), dict) \
                and ev["content_block"].get("type") == "tool_use":
            self._abiertos[indice] = (ev["content_block"], [])
        elif tipo == "content_block_delta" and indice in self._abiertos and isinstance(ev.get("delta"), dict) \
                and ev["delta"].get("type") == "input_json_delta":
            self._abiertos[indice][1].append(str(ev["delta"].get("partial_json", "")))
        elif tipo == "content_block_stop" and indice in self._abiertos:
            bloque, parciales = self._abiertos.pop(indice)
            return [_pedida(bloque, parciales)]
        return []


def herramientas_de_mensaje(cuerpo: bytes) -> list[HerramientaPedida] | None:
    try:
        doc = json.loads(cuerpo)
    except ValueError:
        return None
    if not isinstance(doc, dict) or not isinstance(doc.get("content"), list):
        return []
    return [_pedida(b, []) for b in doc["content"] if isinstance(b, dict) and b.get("type") == "tool_use"]


def resultados_de_peticion(cuerpo: bytes) -> list[ResultadoDevuelto] | None:
    try:
        doc = json.loads(cuerpo)
    except ValueError:
        return None
    if not isinstance(doc, dict):
        return None
    salida = []
    for mensaje in doc.get("messages") or []:
        if not isinstance(mensaje, dict) or not isinstance(mensaje.get("content"), list):
            continue
        for b in mensaje["content"]:
            if isinstance(b, dict) and b.get("type") == "tool_result":
                crudo = _canonico(b.get("content"))
                salida.append(ResultadoDevuelto(b.get("tool_use_id"), bool(b.get("is_error", False)), len(crudo),
                                                hashlib.sha256(crudo).hexdigest()))
    return salida


def evento_de_pedida(p: HerramientaPedida, ruta: str) -> dict:
    ev = {"evento": "herramienta_pedida", "tool_use_id": p.tool_use_id, "herramienta": p.nombre,
          "entrada_legible": p.entrada_legible, "ruta": ruta}
    crudo = _canonico(p.entrada)
    if len(crudo) <= TOPE_ENTRADA_BYTES:
        ev["entrada"] = p.entrada
    else:
        ev.update(entrada_sha256=hashlib.sha256(crudo).hexdigest(), entrada_bytes=len(crudo),
                  entrada_inicio=crudo[:_INICIO_BYTES].decode("utf-8", errors="replace"))
    return ev


def evento_de_resultado(r: ResultadoDevuelto, ruta: str) -> dict:
    return {"evento": "resultado_devuelto", "tool_use_id": r.tool_use_id, "es_error": r.es_error,
            "bytes": r.bytes, "sha256": r.sha256, "ruta": ruta}
```

- [ ] **Step 4: Correr y ver que pasan**

Run: `cd /home/fruiz/worktrees/jax-sp1-c3 && PYTHONPATH=.:las_manos python -m pytest tests/test_ejecutor_contratos_lectura.py -v`
Expected: 9 PASS. `test_evento_de_pedida_con_entrada_enorme_guarda_huella` exige `len(entrada_inicio) == 16384`:
el contenido es ASCII, así que bytes y caracteres coinciden.

- [ ] **Step 5: Commit**

```bash
git -C /home/fruiz/worktrees/jax-sp1-c3 add jax/ejecutor/contratos/lectura.py tests/test_ejecutor_contratos_lectura.py
git -C /home/fruiz/worktrees/jax-sp1-c3 commit -m "feat(ejecutor): lectura de tool_use y tool_result para C3, completos o nada

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: el proxy anota antes de reenviar

**Files:**
- Modify: `/home/fruiz/worktrees/jax-sp1-c3/jax/ejecutor/proxy_carril.py` (imports, constantes, `_NO_REENVIAR`, `Config`, `config_desde_entorno`, `_Proxy.__init__`, `_reenviar`, `Servidor.wait_closed`, `arrancar`, docstring de configuración)
- Modify: `/home/fruiz/worktrees/jax-sp1-c3/tests/test_ejecutor_proxy_carril.py` (clase `Proxy`, dict `_ENTORNO`, parámetros de `test_config_invalida_falla_cerrado`)
- Test: `/home/fruiz/worktrees/jax-sp1-c3/tests/test_ejecutor_proxy_registro.py`

**Interfaces:** Consume `Registro`, `RegistroCorrupto` (Task 1), `lectura` (Task 2), `UpstreamCanario` (plan 1, en
tests). El código de esta task está escrito contra `proxy_carril.py` de `master` en `de6964e` (frente E: el cliente
sale de `crear_cliente_http()` y el tope va por petición en `_TIMEOUT_UPSTREAM`). **Antes de empezar**,
`git -C /home/fruiz/jax log --oneline -1 -- jax/ejecutor/proxy_carril.py`: si cambió después de `de6964e`, se
relee el archivo y se adapta el diff conservando las dos intenciones (lección: los specs envejecen). Produce `Config.registro`, `REGISTRO_FALLO`, `REGISTRO_ILEGIBLE`; `arrancar` lanza `RegistroCorrupto`.

- [ ] **Step 1: Escribir los tests que fallan**

```python
# tests/test_ejecutor_proxy_registro.py
"""C3 en el proxy: lo que el cerebro pide queda anotado ANTES de que el arnés reciba el
final del bloque; si no se puede anotar, el bloque no llega. Sin registro no hay acción."""
import asyncio
import gzip
import json

import h11
import httpx
import pytest

from jax.ejecutor import proxy_carril
from jax.ejecutor.contratos import registro as R
from jax.ejecutor.contratos.canario_upstream import UpstreamCanario, guion_bash
from tests.test_ejecutor_proxy_carril import Proxy, Upstream, _correr

_BASE = {"model": "m", "stream": True, "max_tokens": 5, "tools": [{"name": "Bash", "input_schema": {"type": "object"}}]}


def _resultado(tool_use_id, contenido):
    return {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_use_id, "content": contenido}]}


def _registro(tmp_path):
    return [json.loads(l) for l in (tmp_path / "registro.jsonl").read_text().splitlines()]


def test_tool_use_y_resultado_quedan_encadenados_y_una_sola_vez(tmp_path):
    async def escenario():
        async with UpstreamCanario([guion_bash("toolu_1", "uptime")], "127.0.0.1", 0) as up, \
                Proxy(f"http://127.0.0.1:{up.puerto}", tmp_path, 2) as px, httpx.AsyncClient() as cli:
            url = px.url + "/v1/messages"
            e1 = (await cli.post(url, json={**_BASE, "messages": [{"role": "user", "content": "x"}]})).status_code
            recordatorio = {"role": "system", "content": [{"type": "text", "text": "r"}]}
            e2 = (await cli.post(url, json={**_BASE, "messages": [_resultado("toolu_1", "arriba"), recordatorio]})).status_code
            e3 = (await cli.post(url, json={**_BASE, "messages": [_resultado("toolu_1", "arriba")]})).status_code
            return e1, e2, e3

    assert _correr(escenario()) == (200, 200, 200)
    eventos = _registro(tmp_path)
    assert [e["evento"] for e in eventos] == ["registro_abierto", "herramienta_pedida", "resultado_devuelto"]
    assert (eventos[1]["tool_use_id"], eventos[1]["herramienta"], eventos[1]["entrada"]) == ("toolu_1", "Bash", {"command": "uptime"})
    assert eventos[2]["tool_use_id"] == "toolu_1" and eventos[2]["es_error"] is False
    assert R.verificar_cadena(tmp_path / "registro.jsonl") == R.Verificacion(True, 3, None, None)


def test_respuesta_no_stream_se_anota_antes_de_entregarse(tmp_path):
    async def escenario():
        async with UpstreamCanario([guion_bash("toolu_2", "ls")], "127.0.0.1", 0) as up, \
                Proxy(f"http://127.0.0.1:{up.puerto}", tmp_path, 2) as px, httpx.AsyncClient() as cli:
            r = await cli.post(px.url + "/v1/messages", json={**_BASE, "stream": False, "messages": []})
            return r.status_code, r.json()["content"][0]["id"]

    assert _correr(escenario()) == (200, "toolu_2")
    assert [e["evento"] for e in _registro(tmp_path)] == ["registro_abierto", "herramienta_pedida"]


def test_sin_registro_el_bloque_no_llega(tmp_path, monkeypatch):
    original = R.Registro.anotar

    def roto(self, evento):
        if evento.get("evento") == "herramienta_pedida":
            raise OSError("disco_lleno")
        return original(self, evento)

    async def escenario():
        async with UpstreamCanario([guion_bash("toolu_3", "rm -rf /tmp/x")], "127.0.0.1", 0) as up, \
                Proxy(f"http://127.0.0.1:{up.puerto}", tmp_path, 2) as px, httpx.AsyncClient() as cli:
            monkeypatch.setattr(R.Registro, "anotar", roto)
            recibido, cortado = b"", False
            try:
                async with cli.stream("POST", px.url + "/v1/messages", json={**_BASE, "messages": []}) as r:
                    async for trozo in r.aiter_raw():
                        recibido += trozo
            except httpx.RemoteProtocolError:
                cortado = True
            return recibido, cortado

    recibido, cortado = _correr(escenario())
    assert cortado is True
    assert b"content_block_stop" not in recibido and b"toolu_3" not in recibido


def test_un_resultado_que_no_se_puede_anotar_no_sube_al_cerebro(tmp_path, monkeypatch):
    def roto(self, evento):
        raise OSError("disco_lleno")

    async def escenario():
        async with Upstream(n_trozos=1) as up, Proxy(up.url, tmp_path, 2) as px, httpx.AsyncClient() as cli:
            monkeypatch.setattr(R.Registro, "anotar", roto)
            r = await cli.post(px.url + "/v1/messages", json={**_BASE, "messages": [_resultado("t", "x")]})
            return r.status_code, r.json(), len(up.recibidas)

    assert _correr(escenario()) == (502, {"type": "error", "error": {"type": proxy_carril.REGISTRO_FALLO}}, 0)


def test_pide_identidad_y_rechaza_respuesta_comprimida(tmp_path):
    async def atender(reader, writer):
        conn = h11.Connection(h11.SERVER)
        while not isinstance(ev := conn.next_event(), (h11.EndOfMessage, h11.ConnectionClosed)):
            if ev is h11.NEED_DATA:
                conn.receive_data(await reader.read(65536))
        cuerpo = gzip.compress(json.dumps({"content": [{"type": "tool_use", "id": "t", "name": "Bash",
                                                        "input": {"command": "x"}}]}).encode())
        writer.write(conn.send(h11.Response(status_code=200, headers=[
            ("content-type", "application/json"), ("content-encoding", "gzip"), ("content-length", str(len(cuerpo)))])))
        writer.write(conn.send(h11.Data(data=cuerpo)))
        writer.write(conn.send(h11.EndOfMessage()))
        await writer.drain()
        writer.close()

    async def escenario():
        srv = await asyncio.start_server(atender, "127.0.0.1", 0)
        try:
            async with Proxy(f"http://127.0.0.1:{srv.sockets[0].getsockname()[1]}", tmp_path, 2) as px, \
                    httpx.AsyncClient() as cli:
                r = await cli.post(px.url + "/v1/messages", headers={"accept-encoding": "gzip"},
                                   json={**_BASE, "stream": False, "messages": []})
            async with Upstream(n_trozos=1) as up, Proxy(up.url, tmp_path, 2) as px, httpx.AsyncClient() as cli:
                await cli.post(px.url + "/v1/messages", headers={"accept-encoding": "gzip, br"}, content=b"{}")
                pedida = up.recibidas[0][2]
            return r.status_code, r.json()["error"]["type"], pedida
        finally:
            srv.close()
            await srv.wait_closed()

    estado, codigo, cabeceras = _correr(escenario())
    assert (estado, codigo) == (502, proxy_carril.REGISTRO_ILEGIBLE)
    assert cabeceras[b"accept-encoding"] == b"identity"


def test_registro_corrupto_no_arranca(tmp_path):
    (tmp_path / "registro.jsonl").write_bytes(b'{"n":1,"prev":"0"}\n{"n":2')
    cfg = proxy_carril.Config(upstream="http://127.0.0.1:9", raiz=tmp_path, tope_s=1, host="127.0.0.1", puerto=0,
                              registro=tmp_path / "registro.jsonl")
    with pytest.raises(R.RegistroCorrupto):
        _correr(proxy_carril.arrancar(cfg))
```

En `tests/test_ejecutor_proxy_carril.py`:

```python
class Proxy:
    def __init__(self, upstream_url, raiz, tope_s):
        self.cfg = Config(upstream=upstream_url, raiz=raiz, tope_s=tope_s,
                          host="127.0.0.1", puerto=0, registro=raiz / "registro.jsonl")
```

y en `_ENTORNO` agregar `"JAX_EJECUTOR_REGISTRO": "/var/log/jax-ejecutor/registro.jsonl",`; en los parámetros de
`test_config_invalida_falla_cerrado` agregar `("JAX_EJECUTOR_REGISTRO", "relativa/registro.jsonl"),`; y en
`test_config_sale_del_entorno_sin_upstream_hardcodeado` agregar
`assert str(cfg.registro) == "/var/log/jax-ejecutor/registro.jsonl"`.

- [ ] **Step 2: Correr y ver que fallan**

Run: `cd /home/fruiz/worktrees/jax-sp1-c3 && PYTHONPATH=.:las_manos python -m pytest tests/test_ejecutor_proxy_registro.py tests/test_ejecutor_proxy_carril.py -v`
Expected: FAIL (`TypeError: Config.__init__() got an unexpected keyword argument 'registro'` y
`AttributeError: ... REGISTRO_FALLO`).

- [ ] **Step 3: Implementar**

En `jax/ejecutor/proxy_carril.py`, imports y constantes:

```python
import collections

from jax.ejecutor.contratos import lectura
from jax.ejecutor.contratos.registro import Registro

REGISTRO_FALLO = "registro_fallo"
REGISTRO_ILEGIBLE = "registro_ilegible"
_RESULTADOS_RECORDADOS = 10000
```

Cabeceras: el proxy pide SIEMPRE el cuerpo sin comprimir (si no, lo que pasa no se puede leer):

```python
_NO_REENVIAR = frozenset({
    b"connection", b"keep-alive", b"proxy-connection", b"proxy-authenticate",
    b"proxy-authorization", b"te", b"trailer", b"transfer-encoding", b"upgrade",
    b"host", b"content-length", b"accept-encoding",
})
```

`Config` y su lectura:

```python
@dataclass(frozen=True)
class Config:
    upstream: str
    raiz: Path
    tope_s: float
    host: str
    puerto: int
    registro: Path
```

y al final de `config_desde_entorno`, antes del `return`:

```python
    registro = Path(obligatoria("JAX_EJECUTOR_REGISTRO"))
    if not registro.is_absolute():
        raise ConfigInvalida(Motivo(CONFIG_INVALIDA, (("variable", "JAX_EJECUTOR_REGISTRO"),)))
```

con `registro=registro,` en el `Config(...)`. Agregar al docstring del módulo, en «Configuración»:
`JAX_EJECUTOR_REGISTRO   registro de C3 (obligatoria): sin registro no hay proxy, sin proxy no hay cerebro`.

`_Proxy`:

```python
class _Proxy:
    def __init__(self, cfg: Config, registro: Registro) -> None:
        self.cfg = cfg
        self.registro = registro
        self._resultados_anotados: collections.OrderedDict = collections.OrderedDict()
        # Sin cambios respecto de master (E-24): crear_cliente_http() es el único constructor.
        self.cliente = crear_cliente_http()

    async def _anotar(self, evento: dict) -> None:
        await asyncio.to_thread(self.registro.anotar, evento)

    async def _anotar_resultados(self, ruta: str, cuerpo: bytes) -> None:
        resultados = lectura.resultados_de_peticion(cuerpo)
        if resultados is None:
            await self._anotar({"evento": "peticion_ilegible", "ruta": ruta})
            return
        for r in resultados:
            if r.tool_use_id in self._resultados_anotados:
                continue
            await self._anotar(lectura.evento_de_resultado(r, ruta))
            self._resultados_anotados[r.tool_use_id] = None
            while len(self._resultados_anotados) > _RESULTADOS_RECORDADOS:
                self._resultados_anotados.popitem(last=False)
```

`_reenviar` completo:

```python
    async def _reenviar(self, conn, writer, peticion: h11.Request, cuerpo: bytes) -> None:
        metodo = peticion.method.decode("latin-1")
        ruta = _ruta_sin_query(peticion.target)
        de_mensajes = metodo == "POST" and ruta.endswith("/messages")
        if de_mensajes:
            try:
                await self._anotar_resultados(ruta, cuerpo)
            except OSError as exc:
                log.error("proxy_carril %s metodo=%s ruta=%s tipo=%s", REGISTRO_FALLO, metodo, ruta, type(exc).__name__)
                await _responder_error(conn, writer, 502, Motivo(REGISTRO_FALLO))
                return
        try:
            async with carril_ejecutor_async(self.cfg.raiz, self.cfg.tope_s):
                cabeceras = [(k, v) for k, v in peticion.headers if k.lower() not in _NO_REENVIAR]
                cabeceras.append((b"accept-encoding", b"identity"))
                solicitud = self.cliente.build_request(
                    metodo, self.cfg.upstream + peticion.target.decode("latin-1"),
                    headers=cabeceras, content=cuerpo, timeout=_TIMEOUT_UPSTREAM)
                try:
                    respuesta = await self.cliente.send(solicitud, stream=True)
                except httpx.HTTPError as exc:
                    log.warning("proxy_carril %s metodo=%s ruta=%s tipo=%s",
                                UPSTREAM_INALCANZABLE, metodo, ruta, type(exc).__name__)
                    await _responder_error(conn, writer, 502, Motivo(UPSTREAM_INALCANZABLE))
                    return
                try:
                    await self._devolver(conn, writer, respuesta, metodo, ruta, de_mensajes)
                finally:
                    await respuesta.aclose()
        except EsperaAgotada as exc:
            (motivo,) = exc.args
            log.info("proxy_carril %s metodo=%s ruta=%s", ESPERA_AGOTADA, metodo, ruta)
            await _responder_error(conn, writer, 503, motivo,
                                   extra=((b"x-should-retry", b"false"),))

    async def _devolver(self, conn, writer, respuesta, metodo: str, ruta: str, de_mensajes: bool) -> None:
        codificacion = respuesta.headers.get("content-encoding", "identity").strip().lower()
        if codificacion not in ("", "identity"):
            log.error("proxy_carril %s metodo=%s ruta=%s codificacion=%s", REGISTRO_ILEGIBLE, metodo, ruta, codificacion)
            await _responder_error(conn, writer, 502, Motivo(REGISTRO_ILEGIBLE))
            return
        devolver = [(k, v) for k, v in respuesta.headers.raw if k.lower() not in _NO_DEVOLVER]
        devolver.append((b"connection", b"close"))
        inicio = h11.Response(status_code=respuesta.status_code, headers=devolver,
                              reason=respuesta.reason_phrase.encode("latin-1"))
        es_sse = respuesta.headers.get("content-type", "").startswith("text/event-stream")
        if not es_sse:
            # Un JSON entero se lee ANTES de entregarlo: un tool_use dentro no sale sin anotar.
            try:
                crudo = b"".join([trozo async for trozo in respuesta.aiter_raw()])
            except httpx.HTTPError as exc:
                log.warning("proxy_carril upstream_cortado metodo=%s ruta=%s tipo=%s", metodo, ruta, type(exc).__name__)
                await _responder_error(conn, writer, 502, Motivo(UPSTREAM_INALCANZABLE))
                return
            try:
                for pedida in (lectura.herramientas_de_mensaje(crudo) or []) if de_mensajes else []:
                    await self._anotar(lectura.evento_de_pedida(pedida, ruta))
            except OSError as exc:
                log.error("proxy_carril %s metodo=%s ruta=%s tipo=%s", REGISTRO_FALLO, metodo, ruta, type(exc).__name__)
                await _responder_error(conn, writer, 502, Motivo(REGISTRO_FALLO))
                return
            await _enviar(conn, writer, inicio)
            log.info("proxy_carril peticion metodo=%s ruta=%s estado=%d", metodo, ruta, respuesta.status_code)
            await _enviar(conn, writer, h11.Data(data=crudo))
            await _enviar(conn, writer, h11.EndOfMessage())
            return
        await _enviar(conn, writer, inicio)
        log.info("proxy_carril peticion metodo=%s ruta=%s estado=%d", metodo, ruta, respuesta.status_code)
        lector = lectura.LectorSSE()
        try:
            async for trozo in respuesta.aiter_raw():
                try:
                    for pedida in lector.alimentar(trozo):
                        await self._anotar(lectura.evento_de_pedida(pedida, ruta))
                except OSError as exc:
                    # El trozo que completa el tool_use NO sale: el arnés ve un stream cortado.
                    log.error("proxy_carril %s metodo=%s ruta=%s tipo=%s", REGISTRO_FALLO, metodo, ruta, type(exc).__name__)
                    writer.transport.abort()
                    return
                await _enviar(conn, writer, h11.Data(data=trozo))
        except httpx.HTTPError as exc:
            log.warning("proxy_carril upstream_cortado metodo=%s ruta=%s tipo=%s", metodo, ruta, type(exc).__name__)
            writer.transport.abort()
            return
        await _enviar(conn, writer, h11.EndOfMessage())
```

`Servidor.wait_closed` y `arrancar`:

```python
    async def wait_closed(self) -> None:
        await self._servidor.wait_closed()
        await self._proxy.cerrar()
        self._proxy.registro.cerrar()


async def arrancar(cfg: Config) -> Servidor:
    # Un registro que no cuadra NO se abre (RegistroCorrupto): sin registro no hay cerebro.
    registro = await asyncio.to_thread(Registro, cfg.registro)
    try:
        await asyncio.to_thread(registro.anotar, {"evento": "registro_abierto", "pid": os.getpid()})
        proxy = _Proxy(cfg, registro)
        servidor = await asyncio.start_server(proxy.atender, cfg.host, cfg.puerto)
    except BaseException:
        registro.cerrar()
        raise
    return Servidor(servidor, proxy)
```

- [ ] **Step 4: Correr y ver que pasan (los nuevos y TODOS los del proxy)**

Run: `cd /home/fruiz/worktrees/jax-sp1-c3 && PYTHONPATH=.:las_manos python -m pytest tests/test_ejecutor_proxy_registro.py tests/test_ejecutor_proxy_carril.py -v && python3 policy/tests/test_no_fail_open_except.py && PYTHONPATH=.:las_manos python -m pytest tests/test_no_blocking_in_async.py -q`
Expected: todos PASS (6 nuevos + los existentes, que no cambian de estado); P10 y la guardia de bloqueo en async verdes.

- [ ] **Step 5: Mutaciones — el orden importa (suprimir y neutralizar)**

Con backup y `cmp`, una por vez:
1. **Invertir el orden** en el loop SSE: mover `await _enviar(conn, writer, h11.Data(data=trozo))` **antes** del
   `try` que anota. Expected: FAIL `test_sin_registro_el_bloque_no_llega` (el bloque llega aunque no se anotó).
2. **Neutralizar** el chequeo de codificación: `if codificacion not in ("", "identity"):` → `if False:`.
   Expected: FAIL `test_pide_identidad_y_rechaza_respuesta_comprimida`.
3. **Suprimir** la anotación de resultados: cuerpo de `_anotar_resultados` → `return`.
   Expected: FAIL `test_tool_use_y_resultado_quedan_encadenados_y_una_sola_vez` y
   `test_un_resultado_que_no_se_puede_anotar_no_sube_al_cerebro`.

- [ ] **Step 6: Commit**

```bash
git -C /home/fruiz/worktrees/jax-sp1-c3 add jax/ejecutor/proxy_carril.py tests/test_ejecutor_proxy_carril.py tests/test_ejecutor_proxy_registro.py
git -C /home/fruiz/worktrees/jax-sp1-c3 commit -m "feat(ejecutor): el proxy anota cada tool_use antes de entregarlo; sin registro no hay acción

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: el cerco de red

**Files:**
- Create: `/home/fruiz/worktrees/jax-sp1-c3/jax/ejecutor/contratos/cerco.py`
- Test: `/home/fruiz/worktrees/jax-sp1-c3/tests/test_ejecutor_contratos_cerco.py`

**Interfaces:** Consume `politica.Politica`, `politica.validar` (plan 1). Produce `TABLA`, `renderizar`,
`desde_politica`, `principal(argv, env)` (`python -m jax.ejecutor.contratos.cerco <politica.json> <salida.nft>`).

**Por qué `meta skuid` y no un namespace de red:** la jaula comparte la red del host (el arnés necesita llegar al
proxy y el Ejecutor a los servidores). `nftables` en `output` con `meta skuid` filtra por la cuenta que abrió el
socket, sin tocar a nadie más. `ct state established,related` deja responder a la sesión SSH con la que `fruiz`
entra a la cuenta (la inicia el otro lado).

- [ ] **Step 1: Escribir los tests que fallan**

```python
# tests/test_ejecutor_contratos_cerco.py
"""Cerco de red de la cuenta del Ejecutor (C3): sólo proxy, canario y SSH del inventario."""
import pytest

from jax.ejecutor.contratos import cerco
from tests.test_ejecutor_contratos_politica import OTRO_UID, doc_base, escribir

ESPERADO = """# Generado por jax/ejecutor/contratos/cerco.py: no se edita a mano.
table inet ejecutor_cerco
delete table inet ejecutor_cerco
table inet ejecutor_cerco {
\tset destinos_ssh {
\t\ttype ipv4_addr . inet_service
\t\telements = { 127.0.0.1 . 58291, 192.0.2.20 . 58291 }
\t}
\tchain salida {
\t\ttype filter hook output priority filter; policy accept;
\t\tmeta skuid != 1001 accept
\t\tct state established,related accept
\t\toifname "lo" ip daddr 127.0.0.1 tcp dport { 18435, 18436 } accept
\t\tip daddr . tcp dport @destinos_ssh accept
\t\tmeta l4proto tcp counter reject with tcp reset
\t\tcounter reject with icmpx admin-prohibited
\t}
}
"""


def test_render_exacto():
    assert cerco.renderizar(1001, (18436, 18435), (("192.0.2.20", 58291), ("127.0.0.1", 58291))) == ESPERADO


@pytest.mark.parametrize("uid, puertos, destinos", [
    (True, (1,), (("127.0.0.1", 22),)),
    (0, (1,), (("127.0.0.1", 22),)),
    (1001, (), (("127.0.0.1", 22),)),
    (1001, (70000,), (("127.0.0.1", 22),)),
    (1001, (1,), ()),
    (1001, (1,), (("::1", 22),)),
    (1001, (1,), (("no-es-ip", 22),)),
])
def test_rechaza_lo_que_no_puede_ser(uid, puertos, destinos):
    with pytest.raises(ValueError):
        cerco.renderizar(uid, puertos, destinos)


def test_desde_la_politica(tmp_path):
    from jax.ejecutor.contratos import politica
    p = politica.cargar(escribir(tmp_path, doc_base()), uid_de_la_cuenta=OTRO_UID)
    texto = cerco.desde_politica(p, 1001, (18435,))
    assert "elements = { 192.0.2.5 . 58291, 192.0.2.11 . 58291, 192.0.2.20 . 58291 }" in texto
```

- [ ] **Step 2: Correr y ver que fallan**

Run: `cd /home/fruiz/worktrees/jax-sp1-c3 && PYTHONPATH=.:las_manos python -m pytest tests/test_ejecutor_contratos_cerco.py -v`
Expected: FAIL (`cannot import name 'cerco'`).

- [ ] **Step 3: Implementar**

```python
# jax/ejecutor/contratos/cerco.py
"""Cerco de red de la cuenta del Ejecutor (C3). Decisión D-SP1-2.

Medido 2026-09-17 como `axioma`, sin cerco: alcanza LAS MANOS (7777, sin
autenticación, emite tokens de human gate), Ollama (11434), MariaDB (3308) y
jax-platform (8080). Con el cerco, la cuenta sólo abre conexiones a:
- 127.0.0.1 en los puertos locales dados (proxy con carril, upstream del canario);
- cada ip:puerto SSH del inventario.
Todo lo demás se rechaza (TCP con reset, el resto con icmp): falla rápido, no cuelga.

`table …` + `delete table …` + definición = recarga idempotente y atómica con `nft -f`.
Sólo IPv4: una IP v6 en el inventario es un error, no un hueco silencioso.

Uso: python -m jax.ejecutor.contratos.cerco <politica.json> <salida.nft>
     (lee JAX_EJECUTOR_CUENTA, JAX_PROXY_CARRIL_PUERTO, JAX_EJECUTOR_CANARIO_PUERTO)
"""
from __future__ import annotations

import ipaddress
import json
import os
import pwd
import sys
from pathlib import Path

from jax.ejecutor.contratos import politica

TABLA = "ejecutor_cerco"


def _puerto(p) -> int:
    if isinstance(p, bool) or not isinstance(p, int) or not 0 < p < 65536:
        raise ValueError("puerto_invalido")
    return p


def renderizar(uid: int, puertos_locales, destinos_ssh) -> str:
    if isinstance(uid, bool) or not isinstance(uid, int) or uid <= 0:
        raise ValueError("uid_invalido")
    locales = sorted({_puerto(p) for p in puertos_locales})
    if not locales:
        raise ValueError("sin_puertos_locales")
    destinos = []
    for ip, puerto in destinos_ssh:
        if not isinstance(ipaddress.ip_address(ip), ipaddress.IPv4Address):
            raise ValueError("solo_ipv4")
        destinos.append((ipaddress.IPv4Address(ip), _puerto(puerto)))
    if not destinos:
        raise ValueError("sin_destinos")
    elementos = ", ".join(f"{ip} . {p}" for ip, p in sorted(set(destinos)))
    return (
        "# Generado por jax/ejecutor/contratos/cerco.py: no se edita a mano.\n"
        f"table inet {TABLA}\n"
        f"delete table inet {TABLA}\n"
        f"table inet {TABLA} {{\n"
        "\tset destinos_ssh {\n"
        "\t\ttype ipv4_addr . inet_service\n"
        f"\t\telements = {{ {elementos} }}\n"
        "\t}\n"
        "\tchain salida {\n"
        "\t\ttype filter hook output priority filter; policy accept;\n"
        f"\t\tmeta skuid != {uid} accept\n"
        "\t\tct state established,related accept\n"
        f"\t\toifname \"lo\" ip daddr 127.0.0.1 tcp dport {{ {', '.join(map(str, locales))} }} accept\n"
        "\t\tip daddr . tcp dport @destinos_ssh accept\n"
        "\t\tmeta l4proto tcp counter reject with tcp reset\n"
        "\t\tcounter reject with icmpx admin-prohibited\n"
        "\t}\n"
        "}\n"
    )


def desde_politica(p: politica.Politica, uid: int, puertos_locales) -> str:
    return renderizar(uid, puertos_locales, tuple((h.ip, h.puerto) for h in p.hosts))


def principal(argv, env=None) -> int:
    env = os.environ if env is None else env
    doc = json.loads(Path(argv[0]).read_bytes())
    p = politica.validar(doc)
    uid = pwd.getpwnam(env["JAX_EJECUTOR_CUENTA"]).pw_uid
    puertos = (int(env["JAX_PROXY_CARRIL_PUERTO"]), int(env["JAX_EJECUTOR_CANARIO_PUERTO"]))
    Path(argv[1]).write_text(desde_politica(p, uid, puertos), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(principal(sys.argv[1:]))
```

Nota: `principal` usa `politica.validar` (no `cargar`) a propósito: lo corre `fruiz`, que es el dueño del archivo
—`cargar` lo rechazaría por `duenio_es_la_cuenta`, que es la regla para la jaula, no para el escritor.

- [ ] **Step 4: Correr y ver que pasan**

Run: `cd /home/fruiz/worktrees/jax-sp1-c3 && PYTHONPATH=.:las_manos python -m pytest tests/test_ejecutor_contratos_cerco.py -v`
Expected: 9 PASS.

- [ ] **Step 5: Commit**

```bash
git -C /home/fruiz/worktrees/jax-sp1-c3 add jax/ejecutor/contratos/cerco.py tests/test_ejecutor_contratos_cerco.py
git -C /home/fruiz/worktrees/jax-sp1-c3 commit -m "feat(ejecutor): cerco de red de la cuenta axioma derivado del inventario

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: `verificar_c3` — desde la jaula no se escribe el registro

**Files:**
- Create: `/home/fruiz/worktrees/jax-sp1-c3/jax/ejecutor/contratos/canario_c3.py`
- Test: `/home/fruiz/worktrees/jax-sp1-c3/tests/test_ejecutor_contratos_canario_c3.py`

**Interfaces:** Consume `cuenta_axioma.correr_en_la_cuenta`, `Fallo` (plan 1), `verificar_cadena` (Task 1).
Produce `verificar_c3(c, *, registro, puerto_proxy, sondas, correr, leer_banderas)` y `banderas(ruta)`. Códigos de
`Fallo("c3", …)`: `cuenta_inalcanzable`, `registro_escribible_desde_la_jaula`, `registro_cambio_desde_la_jaula`,
`cerco_abierto` (con `puerto`), `proxy_inalcanzable`, `registro_sin_append_only`, `cadena_rota`.

**Por qué el control positivo (el proxy SÍ se alcanza):** sin él, una cuenta sin red pasaría las sondas
«cerradas» y el canario diría que el cerco funciona cuando lo único que hay es un Ejecutor que no puede trabajar.

- [ ] **Step 1: Escribir los tests que fallan**

```python
# tests/test_ejecutor_contratos_canario_c3.py
"""verificar_c3 con la cuenta falsa: cada forma en que C3 puede estar roto da su código."""
import asyncio
from pathlib import Path

import pytest

from jax.ejecutor.contratos import canario_c3
from jax.ejecutor.contratos import registro as R
from jax.ejecutor.contratos.cuenta_axioma import Cuenta
from jax.ejecutor.contratos.fallo import Fallo

C = Cuenta("axioma", 58291, Path("/k"), Path("/n"), Path("/l"), Path("/p"))


def _registro(tmp_path):
    ruta = tmp_path / "registro.jsonl"
    reg = R.Registro(ruta)
    reg.anotar({"evento": "x"})
    reg.cerrar()
    return ruta


def _cuenta(abiertas=(), proxy=True, escribe=False, rc=0):
    async def correr(cuenta, remoto, *, entrada=b"", tope_s):
        if rc:
            return rc, b"", b"ssh: connect refused"
        lineas = [f"sonda={p} {'abierta' if p in abiertas else 'cerrada'}" for p in (7777, 11434)]
        lineas.append(f"proxy={'abierto' if proxy else 'cerrado'}")
        lineas.append(f"registro_escrito={'si' if escribe else 'no'}")
        return 0, ("\n".join(lineas) + "\n").encode(), b""
    return correr


def _verificar(tmp_path, correr, banderas=canario_c3.FS_APPEND_FL):
    return asyncio.run(canario_c3.verificar_c3(C, registro=_registro(tmp_path), puerto_proxy=18435,
                                               sondas=(7777, 11434), correr=correr, leer_banderas=lambda r: banderas))


def test_todo_vivo(tmp_path):
    assert _verificar(tmp_path, _cuenta()) == ()


@pytest.mark.parametrize("correr, banderas, esperado", [
    (_cuenta(abiertas=(7777,)), canario_c3.FS_APPEND_FL, Fallo("c3", "cerco_abierto", (("puerto", 7777),))),
    (_cuenta(proxy=False), canario_c3.FS_APPEND_FL, Fallo("c3", "proxy_inalcanzable")),
    (_cuenta(escribe=True), canario_c3.FS_APPEND_FL, Fallo("c3", "registro_escribible_desde_la_jaula")),
    (_cuenta(), 0, Fallo("c3", "registro_sin_append_only")),
])
def test_cada_rotura(tmp_path, correr, banderas, esperado):
    assert esperado in _verificar(tmp_path, correr, banderas)


def test_cuenta_inalcanzable(tmp_path):
    assert _verificar(tmp_path, _cuenta(rc=255)) == (Fallo("c3", "cuenta_inalcanzable", (("rc", 255),)),)


def test_registro_que_cambia_durante_la_sonda(tmp_path):
    ruta = _registro(tmp_path)

    async def correr(cuenta, remoto, *, entrada=b"", tope_s):
        with open(ruta, "ab") as f:
            f.write(b"x")
        return 0, b"sonda=7777 cerrada\nsonda=11434 cerrada\nproxy=abierto\nregistro_escrito=no\n", b""

    fallos = asyncio.run(canario_c3.verificar_c3(C, registro=ruta, puerto_proxy=18435, sondas=(7777, 11434),
                                                 correr=correr, leer_banderas=lambda r: canario_c3.FS_APPEND_FL))
    assert Fallo("c3", "registro_cambio_desde_la_jaula") in fallos
    assert any(f.codigo == "cadena_rota" for f in fallos)


def test_banderas_lee_un_archivo_real(tmp_path):
    ruta = tmp_path / "f"
    ruta.write_text("x")
    assert canario_c3.banderas(ruta) & canario_c3.FS_APPEND_FL == 0
```

- [ ] **Step 2: Correr y ver que fallan**

Run: `cd /home/fruiz/worktrees/jax-sp1-c3 && PYTHONPATH=.:las_manos python -m pytest tests/test_ejecutor_contratos_canario_c3.py -v`
Expected: FAIL (`cannot import name 'canario_c3'`).

- [ ] **Step 3: Implementar**

```python
# jax/ejecutor/contratos/canario_c3.py
"""Prueba viva de C3 (spec 2026-09-15 §4: «desde la jaula, escribir el registro falla»).

Como la cuenta del Ejecutor: intenta anexar al registro y abrir conexiones a los
servicios que el cerco tiene que tapar (`sondas`), y como control abre la del proxy,
que tiene que funcionar. Desde fuera: el tamaño del registro no cambió, tiene la
bandera append-only del sistema de archivos, y su cadena cuadra.

Si C3 ya estaba roto, el intento de escritura deja un byte suelto al final del
registro: el proxy no vuelve a abrirlo (RegistroCorrupto) y el Ejecutor queda
bloqueado hasta que alguien lo mire. Es a propósito: fallar ruidoso y cerrado.
"""
from __future__ import annotations

import fcntl
import os
import shlex
import struct
from pathlib import Path

from jax.ejecutor.contratos import cuenta_axioma
from jax.ejecutor.contratos.fallo import Fallo
from jax.ejecutor.contratos.registro import verificar_cadena

FS_IOC_GETFLAGS = 0x80086601
FS_APPEND_FL = 0x00000020
_TOPE_S = 60


def banderas(ruta) -> int:
    fd = os.open(ruta, os.O_RDONLY | os.O_CLOEXEC)
    try:
        return struct.unpack("<i", fcntl.ioctl(fd, FS_IOC_GETFLAGS, b"\0" * 8)[:4])[0]
    finally:
        os.close(fd)


def _guion(registro: Path, puerto_proxy: int, sondas) -> str:
    q = shlex.quote
    partes = []
    for p in sondas:
        partes.append(f'if timeout 3 bash -c "exec 3<>/dev/tcp/127.0.0.1/{int(p)}" 2>/dev/null; '
                      f'then echo "sonda={int(p)} abierta"; else echo "sonda={int(p)} cerrada"; fi')
    partes.append(f'if timeout 3 bash -c "exec 3<>/dev/tcp/127.0.0.1/{int(puerto_proxy)}" 2>/dev/null; '
                  'then echo proxy=abierto; else echo proxy=cerrado; fi')
    partes.append(f"if printf x 2>/dev/null >> {q(str(registro))}; "
                  "then echo registro_escrito=si; else echo registro_escrito=no; fi")
    return "; ".join(partes)


async def verificar_c3(c, *, registro: Path, puerto_proxy: int, sondas, correr=cuenta_axioma.correr_en_la_cuenta,
                       leer_banderas=banderas) -> tuple:
    antes = os.stat(registro).st_size
    rc, salida, _ = await correr(c, _guion(registro, puerto_proxy, sondas), tope_s=_TOPE_S)
    if rc == 255:
        return (Fallo("c3", "cuenta_inalcanzable", (("rc", rc),)),)
    vistas = dict(linea.split("=", 1) for linea in salida.decode(errors="replace").split("\n") if "=" in linea)
    fallos = []
    if vistas.get("registro_escrito") != "no":
        fallos.append(Fallo("c3", "registro_escribible_desde_la_jaula"))
    if os.stat(registro).st_size != antes:
        fallos.append(Fallo("c3", "registro_cambio_desde_la_jaula"))
    for p in sondas:
        if not salida.decode(errors="replace").count(f"sonda={int(p)} cerrada"):
            fallos.append(Fallo("c3", "cerco_abierto", (("puerto", int(p)),)))
    if vistas.get("proxy") != "abierto":
        fallos.append(Fallo("c3", "proxy_inalcanzable"))
    if not leer_banderas(registro) & FS_APPEND_FL:
        fallos.append(Fallo("c3", "registro_sin_append_only"))
    cadena = verificar_cadena(registro)
    if not cadena.ok:
        fallos.append(Fallo("c3", "cadena_rota", (("linea", cadena.primer_error), ("codigo", cadena.codigo))))
    return tuple(fallos)
```

- [ ] **Step 4: Correr y ver que pasan**

Run: `cd /home/fruiz/worktrees/jax-sp1-c3 && PYTHONPATH=.:las_manos python -m pytest tests/test_ejecutor_contratos_canario_c3.py -v && python3 policy/tests/test_claude_subprocess_solo_via_sandbox.py`
Expected: 8 PASS; guardia verde (`canario_c3.py` no lanza subprocesos ni menciona el arnés en literales).

- [ ] **Step 5: Commit**

```bash
git -C /home/fruiz/worktrees/jax-sp1-c3 add jax/ejecutor/contratos/canario_c3.py tests/test_ejecutor_contratos_canario_c3.py
git -C /home/fruiz/worktrees/jax-sp1-c3 commit -m "feat(ejecutor): prueba viva de C3 con control positivo del proxy

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: el carril sólo lo toma el proxy; CI y canario rojo

**Files:**
- Create: `/home/fruiz/worktrees/jax-sp1-c3/tests/test_ejecutor_carril_solo_en_el_proxy.py`
- Modify: `/home/fruiz/worktrees/jax-sp1-c3/.github/workflows/policy.yml`

**Por qué este test (hallazgo 7 del LEDGER):** `prioridad._fichero` crea los locks con el umask del primero que
llega. Mientras el único que toma `carril_ejecutor*` sea el proxy (cuenta `fruiz`), `axioma` nunca abre un lock y
no hace falta un directorio de grupo común. Si alguien lo llama desde otro lado, ese supuesto se rompe en
silencio: el test lo hace ruidoso.

- [ ] **Step 1: Escribir el test (falla si el supuesto se rompe)**

```python
# tests/test_ejecutor_carril_solo_en_el_proxy.py
"""El carril del Ejecutor sólo lo toma el proxy (cuenta fruiz). Si otro módulo lo usa,
los locks pueden nacer con el umask de otra cuenta (hallazgo 7, LEDGER 2026-09-17)."""
import ast
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
PERMITIDOS = {"jax/ejecutor/proxy_carril.py", "jax/ejecutor/prioridad.py"}
NOMBRES = {"carril_ejecutor", "carril_ejecutor_async"}


def test_solo_el_proxy_usa_el_carril_del_ejecutor():
    usos = []
    for ruta in RAIZ.rglob("*.py"):
        rel = ruta.relative_to(RAIZ).as_posix()
        if rel.startswith(("tests/", ".venv/", "las_manos/.venv/")) or "/site-packages/" in rel or rel in PERMITIDOS:
            continue
        try:
            arbol = ast.parse(ruta.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):  # fail-soft: archivos que no parsean (p. ej. _director_patch) no pueden importar nada al correr
            continue
        for nodo in ast.walk(arbol):
            if isinstance(nodo, ast.Name) and nodo.id in NOMBRES or isinstance(nodo, ast.Attribute) and nodo.attr in NOMBRES \
                    or isinstance(nodo, ast.alias) and nodo.name in NOMBRES:
                usos.append(rel)
    assert usos == [], usos
```

- [ ] **Step 2: Verlo fallar y pasar**

Con backup: agregar a `jax/ejecutor/transporte.py` la línea `from jax.ejecutor.prioridad import carril_ejecutor`.
Run: `PYTHONPATH=.:las_manos python -m pytest tests/test_ejecutor_carril_solo_en_el_proxy.py -q` → FAIL nombrando
`jax/ejecutor/transporte.py`. Restaurar con `cmp` → PASS.

- [ ] **Step 3: Listas y piso**

Agregar en **las dos** listas de `tests-puros`:
```
          tests/test_ejecutor_contratos_registro.py
          tests/test_ejecutor_contratos_lectura.py
          tests/test_ejecutor_contratos_cerco.py
          tests/test_ejecutor_contratos_canario_c3.py
          tests/test_ejecutor_proxy_registro.py
          tests/test_ejecutor_carril_solo_en_el_proxy.py
```
Verificar igualdad de las dos listas con el script del plan 1 (Task 8, Step 2). Push; fijar el piso con el número
del runner y su línea `# N -> M el 2026-09-17 (Ejecutor SP1 plan 2): C3 — registro encadenado, lectura, proxy que
anota antes de entregar, cerco, prueba viva, carril sólo en el proxy. CONFIRMADO POR EL RUNNER (3.12, PR #<número del PR>).`

- [ ] **Step 4: Canario rojo por API**

Commit `test(canario): C3 — el proxy entrega el trozo antes de anotar; se revierte en el commit siguiente` con la
mutación 1 de la Task 3. Push; `gh api repos/fjruizhn/jax/commits/<sha>/check-runs --jq '.check_runs[] | select(.name=="tests-puros") | .conclusion'` → `failure`. Revert, push, mismo comando → `success`.

- [ ] **Step 5: Commit**

```bash
git -C /home/fruiz/worktrees/jax-sp1-c3 add tests/test_ejecutor_carril_solo_en_el_proxy.py .github/workflows/policy.yml
git -C /home/fruiz/worktrees/jax-sp1-c3 commit -m "ci(ejecutor): C3 en tests-puros; el carril del Ejecutor sólo lo toma el proxy

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: hall9000 · instalar proxy, registro y cerco; VERLO FALLAR; medir con carga

> Operación sobre hall9000. No toca .10/.11/.20. La Mesa no usa todavía el proxy ni el carril (§6.2 es SP3): el
> proxy de este plan sólo lo usa el Ejecutor. Requiere el plan 1 instalado y este PR mergeado en `/home/fruiz/jax`.

**Files:**
- Create: `/home/fruiz/worktrees/jax-sp1-c3/ops/ejecutor/jax-ejecutor-proxy.service`
- Create: `/home/fruiz/worktrees/jax-sp1-c3/ops/ejecutor/ejecutor-cerco.service`
- Create: `/home/fruiz/worktrees/jax-sp1-c3/ops/ejecutor/instalar_registro_y_cerco.sh`
- Create: `/home/fruiz/worktrees/jax-sp1-c3/scripts/ejecutor_contratos/probar_c3.py`

- [ ] **Step 1: Unidades e instalador**

```ini
# ops/ejecutor/jax-ejecutor-proxy.service
[Unit]
Description=JAX · proxy con carril del Ejecutor (C3: registro intocable)
After=network.target ollama.service
StartLimitIntervalSec=300
StartLimitBurst=5

[Service]
Type=simple
User=fruiz
WorkingDirectory=/home/fruiz/jax
EnvironmentFile=/etc/jax/.env
Environment=PYTHONPATH=/home/fruiz/jax:/home/fruiz/jax/las_manos
ExecStart=/home/fruiz/jax/.venv/bin/python -m jax.ejecutor.proxy_carril
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```ini
# ops/ejecutor/ejecutor-cerco.service
[Unit]
Description=Ejecutor · cerco de red de la cuenta del Ejecutor (C3)
After=nftables.service
Before=ssh.service

[Service]
Type=oneshot
RemainAfterExit=yes
# Sin ExecStop a propósito: detener la unidad NO abre el cerco. Se quita a mano, con GO.
ExecStart=/usr/sbin/nft -f /etc/jax-ejecutor-cerco/cerco.nft

[Install]
WantedBy=multi-user.target
```

```bash
#!/usr/bin/env bash
# ops/ejecutor/instalar_registro_y_cerco.sh — C3 en hall9000. Corre como fruiz; sudo para lo de root.
set -euo pipefail
: "${JAX_EJECUTOR_REGISTRO:?}" "${JAX_EJECUTOR_POLITICA:?}" "${JAX_EJECUTOR_CUENTA:?}"
: "${JAX_PROXY_CARRIL_PUERTO:?}" "${JAX_EJECUTOR_CANARIO_PUERTO:?}" "${JAX_PROXY_CARRIL_RAIZ:?}"
REPO="$(git -C "$(dirname "$(readlink -f "$0")")" rev-parse --show-toplevel)"
test "$(git -C "$REPO" branch --show-current)" = master

# Registro: directorio de fruiz 0750 (la cuenta ni lo lista), archivo 0640, append-only.
DIR="$(dirname "$JAX_EJECUTOR_REGISTRO")"
sudo install -d -o fruiz -g fruiz -m 0750 "$DIR"
test -e "$JAX_EJECUTOR_REGISTRO" || install -m 0640 /dev/null "$JAX_EJECUTOR_REGISTRO"
sudo chattr +a "$JAX_EJECUTOR_REGISTRO"
lsattr "$JAX_EJECUTOR_REGISTRO" | cut -c1-22 | grep -q a

# Locks del carril: de fruiz; la cuenta del Ejecutor nunca los toca (tests/test_ejecutor_carril_solo_en_el_proxy.py).
sudo install -d -o fruiz -g fruiz -m 0750 "$JAX_PROXY_CARRIL_RAIZ"

# Proxy.
sudo install -o root -g root -m 0644 "$REPO/ops/ejecutor/jax-ejecutor-proxy.service" /etc/systemd/system/
# Cerco: render como fruiz desde la política, instalación root.
ETAPA="$(mktemp -d)"; trap 'rm -rf "$ETAPA"' EXIT
( cd "$REPO" && PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:las_manos python3 -m jax.ejecutor.contratos.cerco "$JAX_EJECUTOR_POLITICA" "$ETAPA/cerco.nft" )
sudo nft -c -f "$ETAPA/cerco.nft"
sudo install -d -o root -g root -m 0755 /etc/jax-ejecutor-cerco
sudo install -o root -g root -m 0644 "$ETAPA/cerco.nft" /etc/jax-ejecutor-cerco/cerco.nft
sudo install -o root -g root -m 0644 "$REPO/ops/ejecutor/ejecutor-cerco.service" /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now ejecutor-cerco.service
sudo systemctl restart ejecutor-cerco.service
sudo nft list table inet ejecutor_cerco >/dev/null
sudo systemctl enable --now jax-ejecutor-proxy.service
echo "c3_instalado=true"
```

- [ ] **Step 2: `.env` e instalación**

Backup `sudo cp -a /etc/jax/.env /etc/jax/.env.backup-pre-ejecutor-c3-$(date +%Y%m%d-%H%M%S)`; con `sudoedit`
agregar `JAX_EJECUTOR_REGISTRO=/var/log/jax-ejecutor/registro.jsonl`, `JAX_EJECUTOR_CERCO_SONDAS=7777,11434,3308,8080`,
`JAX_PROXY_CARRIL_UPSTREAM=http://127.0.0.1:11434`, `JAX_PROXY_CARRIL_RAIZ=/var/lib/jax-carril`,
`JAX_PROXY_CARRIL_TOPE_S=90` (≥ 60, §6.2.2), `JAX_PROXY_CARRIL_PUERTO=18435`, `JAX_PROXY_CARRIL_HOST=127.0.0.1`.
Leer cada una con `bash` (`set -a; . /etc/jax/.env`) **y** con `systemd-run --pipe --wait -p EnvironmentFile=…
printenv` y comparar (lección del frente A). Después: `cd /home/fruiz/jax && set -a && . /etc/jax/.env && set +a &&
ops/ejecutor/instalar_registro_y_cerco.sh`.
Expected: `c3_instalado=true`; `systemctl show jax-ejecutor-proxy -p NRestarts,ActiveState` → `0`, `active`;
`sudo nft list table inet ejecutor_cerco` imprime la tabla.

- [ ] **Step 3: El script de prueba real**

```python
#!/usr/bin/env python3
# scripts/ejecutor_contratos/probar_c3.py
"""Prueba real de C3 en hall9000, re-ejecutable por un tercero.
Uso: set -a; . <(sudo -n cat /etc/jax/.env); set +a; PYTHONPATH=.:las_manos python3 scripts/ejecutor_contratos/probar_c3.py
Lee /etc/jax/.env (producción) sólo para JAX_EJECUTOR_* y JAX_PROXY_CARRIL_*; no toca la DB."""
import asyncio
import os
import sys
from pathlib import Path

from jax.ejecutor.contratos import canario_c3, cuenta_axioma, formato


def main() -> int:
    fallos = asyncio.run(canario_c3.verificar_c3(
        cuenta_axioma.cuenta_desde_entorno(), registro=Path(os.environ["JAX_EJECUTOR_REGISTRO"]),
        puerto_proxy=int(os.environ["JAX_PROXY_CARRIL_PUERTO"]),
        sondas=tuple(int(p) for p in os.environ["JAX_EJECUTOR_CERCO_SONDAS"].split(","))))
    for f in fallos:
        print(formato.campos((("contrato", f.contrato), ("codigo", f.codigo)) + tuple(f.datos)))
    print(formato.campos((("c3_vivo", not fallos),)))
    return 0 if not fallos else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Verlo vivo, y con una herramienta real anotada**

1. `probar_c3.py` → `c3_vivo=true`.
2. `probar_c1.py` del plan 1 **con `JAX_EJECUTOR_CANARIO_PUERTO` apuntado a través del proxy no aplica** (el canario
   habla directo con su upstream). Para ver C3 con el arnés real: una misión mínima por el proxy con Qwen, como la
   cuenta del Ejecutor, `remoto_claude(c, base_url="http://127.0.0.1:18435", modelo=<modelo de facet_binding de jax_local, leído en vivo, no escrito>, prompt="Ejecuta uptime y nada más.")`.
   Sólo si `jax_local` no tiene uso real en ese momento (journal de jax-platform sin `/api/chat` en los últimos 5
   minutos): si aparece uso, se aborta y se reintenta después. Expected: en el registro, una `herramienta_pedida`
   con `{"command": "uptime"}` y su `resultado_devuelto`; `verificar_cadena` ok.

- [ ] **Step 5: VERLO FALLAR — tres roturas, una por capa**

Cada una se deshace y `probar_c3.py` vuelve a `c3_vivo=true` antes de la siguiente:
1. **Red:** `sudo nft delete table inet ejecutor_cerco` → Expected: `cerco_abierto` con `puerto=7777`, `11434`,
   `3308` y `8080`. Restaurar: `sudo systemctl restart ejecutor-cerco.service`.
2. **Append-only:** `sudo chattr -a "$JAX_EJECUTOR_REGISTRO"` → Expected: `registro_sin_append_only`.
   Restaurar: `sudo chattr +a …`.
3. **Cadena, sobre una COPIA** (nunca el registro real): `cp "$JAX_EJECUTOR_REGISTRO" /tmp/reg-copia.jsonl`, editar
   un byte de la segunda línea con `python3`, `python3 -c "from jax.ejecutor.contratos.registro import verificar_cadena as v; print(v('/tmp/reg-copia.jsonl'))"`
   → Expected: `ok=False, primer_error=3, codigo='prev_no_cuadra'`. `rm /tmp/reg-copia.jsonl`.
4. **Proxy abajo:** `sudo systemctl stop jax-ejecutor-proxy` → Expected: `proxy_inalcanzable` y **ningún**
   `cerco_abierto` (el control positivo distingue «cerco cerrado» de «no hay red»). `sudo systemctl start …`.

- [ ] **Step 6: Carga (Cuatro del rendimiento) con criterio pre-registrado**

Con el upstream falso del plan 1 en un puerto libre y **una instancia de prueba** del proxy (registro en `/tmp`, no el
de producción): 200 peticiones con `tool_use`, concurrencia 2 (lo que Claude Code manda en paralelo, §6.1), midiendo
latencia hasta el último byte con y sin registro (instancia sin registro = `master` anterior en un worktree).
**Criterio escrito antes de medir:** p95 agregado por el registro ≤ 10 ms por petición. Si no pasa, se mide
`fsync` por separado antes de tocar nada (el `fsync` no se quita: un registro sin `fsync` puede perder la última
línea en un corte de luz, que ya pasó en este ecosistema el 2026-09-03).

- [ ] **Step 7: Commit**

```bash
git -C /home/fruiz/worktrees/jax-sp1-c3 add ops/ejecutor/jax-ejecutor-proxy.service ops/ejecutor/ejecutor-cerco.service ops/ejecutor/instalar_registro_y_cerco.sh scripts/ejecutor_contratos/probar_c3.py
git -C /home/fruiz/worktrees/jax-sp1-c3 commit -m "ops(ejecutor): proxy, registro append-only y cerco de C3; prueba real re-ejecutable

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: Biblioteca y la deuda que no es del Ejecutor

**Files:**
- Modify: `/home/fruiz/worktrees/jax-sp1-c3/DEUDA.md`
- Modify: `/home/fruiz/worktrees/jax-sp1-c3/CONTEXT.md` (§9, entrada nueva al final)

- [ ] **Step 1: DEUDA con fecha**

Entrada nueva en la sección de abiertos de `DEUDA.md`:

```markdown
### Hyde y cualquier proceso de `fruiz` alcanzan LAS MANOS sin autenticación — fecha: 2026-09-24

- **Hecho (medido 2026-09-17, Mr. Hyde, al planificar C3):** `127.0.0.1:7777` no pide credencial;
  `POST /human_gate/token` emite un token de aprobación a quien lo pida. `hyde_sandbox.py` usa `--share-net`:
  el `claude` de Hyde puede pedir un token y llamar `/execute`. El cerco de C3 lo cierra para la cuenta
  `axioma`, **no** para Hyde (corre como `fruiz`).
- **Por qué no se cierra en SP1:** es la misma propiedad (una jaula no puede autoaprobarse acciones), pero de
  otra faceta con otro dueño de proceso; el cerco por `meta skuid` no aplica. Opciones a decidir: token de
  servicio en LAS MANOS leído de `/etc/jax/.env` (que la jaula de Hyde no monta), o `--unshare-net` + proxy de
  salida para Hyde.
- **Verificación de cierre:** desde un `claude` sandboxeado de Hyde, `curl -X POST 127.0.0.1:7777/human_gate/token`
  falla; el mismo pedido desde LAS MANOS/Jacobs funciona.
```

- [ ] **Step 2: CONTEXT.md §9**

Entrada del 2026-09-17 con: qué se hizo (C3), por qué el proxy y no LAS MANOS (medición de puertos), las cuatro
roturas del Step 5 con su salida literal, el número de la carga, y la lección: **«un registro que alimenta el
observado no registra: registra lo que el observado quiere; el testigo tiene que estar en un camino que el
observado no puede evitar».**

- [ ] **Step 3: Commit**

```bash
git -C /home/fruiz/worktrees/jax-sp1-c3 add DEUDA.md CONTEXT.md
git -C /home/fruiz/worktrees/jax-sp1-c3 commit -m "docs(biblioteca): C3 cerrado con sus roturas vistas; deuda fechada de LAS MANOS sin autenticación

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Auto-revisión (hecha al escribir)

- **Spec §4 C3:** cada tool call → registro fuera de la jaula ✓ (Tasks 1–3; por el proxy, con el porqué en el
  índice); desde la jaula escribir el registro falla ✓ (Task 5 y Task 7 Step 5, por archivo y por red);
  log de `sudo` de `axioma` en cada servidor → **plan 5, Parte B** (hoy `axioma` no tiene sudo en ninguna máquina:
  se instala antes de dárselo, Principio IX).
- **Placeholders:** ninguno. El modelo de la misión mínima (Task 7 Step 4) se lee de `facet_binding` a propósito:
  escribirlo en el plan sería hardcodear un dato que cambia.
- **Tipos:** `Config(…, registro)`, `Registro.anotar(evento) -> int`, `verificar_c3(c, *, registro, puerto_proxy,
  sondas, correr, leer_banderas)` iguales en Interfaces, tasks y tests.
