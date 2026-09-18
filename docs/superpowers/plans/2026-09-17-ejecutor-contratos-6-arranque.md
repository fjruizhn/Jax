# Ejecutor SP1 · Plan 6 · El arranque que se niega: los seis contratos como condición de lanzamiento

> **Para agentes ejecutores:** SUB-SKILL REQUERIDA: usar `superpowers:subagent-driven-development`
> (recomendado) o `superpowers:executing-plans`. Los pasos usan casillas (`- [ ]`).

**Objetivo:** que ningún camino del código pueda lanzar el cerebro del Ejecutor sin antes pasar

> **Enmienda 2026-09-17 (plan 4 ejecutado):** antes de cada misión el arranque corre
> `eleccion_c5.verificar_eleccion(conn, cfg=…, proveedor_cerebro=…, proveedor_auditor=…, hosts_mision=…)` (con la
> compuerta cerrada, una misión sobre máquinas con datos de clientes NO arranca), exige la pausa del Ejecutor
> ausente, lanza el vigía (`vigia.ConfigVigia` lleva `pausa`, `latido` y `latido_cada_s`, que tiene que ser menor que
> `JAX_EJECUTOR_VIGIA_LATIDO_MAX_S`) y sólo después abre el proxy al arnés. `aplicar_revision(entrega, revision)` ya no
> recibe el mapa de ids: usa `auditor.afirmaciones_auditables(entrega)`.
`exigir_contratos`, y que `exigir_contratos` lance `ContratosNoVerificados` si **cualquiera** de C1–C6 no está vivo
—con los canarios de C1 y C5 corriendo de verdad en cada arranque—. Y retirar el lanzador de la Fase 0, que hoy
lanza sin jaula ni gancho.

**Arquitectura:** `arranque.py` compone las pruebas de los planes 1–5 (instalación idéntica al repo; exportar la
política; `verificar_c1`; `verificar_c3`; C4 estático: latido del freno, unidad habilitada, cron y linger cerrados;
`verificar_c5` con el auditor real y la elección cerebro/auditor; C6 estático: llaves de root, llave del freno y
revocador presentes en cada máquina). Una guardia de CI (`policy/tests/test_ejecutor_lanza_solo_con_contratos.py`)
falla si un módulo usa `cuenta_axioma.remoto_claude` sin `exigir_contratos`. `scripts/ejecutor_fase0/harness.py`
deja de lanzar.

**Tech stack:** Python 3.12, `asyncio`, `ast`.

**Spec:** `docs/superpowers/specs/2026-09-15-ejecutor-design.md` §4 («C1 y C5 llevan una entrada canario permanente; si
no se dispara, el Ejecutor no arranca»; «gate verificable por un tercero»); Principio IX. Índice:
`docs/superpowers/plans/2026-09-17-ejecutor-contratos-0-indice.md`.

## Global Constraints

- **Nunca cambiar de rama en `/home/fruiz/jax`.** Worktree `/home/fruiz/worktrees/jax-sp1-arranque`, rama
  `feat/ejecutor-arranque` desde `master` con los planes 1–5 mergeados.
- **`/etc/jax/.env` apunta a PRODUCCIÓN.** Ningún test la carga.
- **P10:** marca `# fail-soft: <razón específica>` en la línea del `except` amplio sin `raise`.
- **Piso de `tests-puros`** en las dos listas, número del runner; la guardia nueva con su piso propio.
- **Mutaciones:** `PYTHONDONTWRITEBYTECODE=1`; backup `.mut-bak` y `cmp`.
- **Tests con procesos:** `fork` explícito; `ps`, nunca `pgrep -f`. (Este plan no lanza procesos en tests.)
- **Cero strings visibles hardcodeados:** códigos. **Sin hardcoding:** todo desde `JAX_EJECUTOR_*` y la DB.
- **Nada destructivo sobre .10/.11/.20:** el C6 del arranque sólo LEE (`stat`, `grep`) en cada máquina.
- **Costo por arranque, declarado:** canario C1 (~2 s, medido al escribir el plan 1), canario C5 (tres llamadas al
  auditor). Se mide en la Task 4 y queda en la Biblioteca.
- Commits en castellano, terminados con `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`.

## Mapa de archivos

| Archivo | Acción | Responsabilidad |
|---|---|---|
| `jax/ejecutor/contratos/arranque.py` | Crear | `Contexto`, pruebas estáticas de C4/C6 e instalación, composición, `exigir_contratos` |
| `policy/tests/test_ejecutor_lanza_solo_con_contratos.py` | Crear | guardia: `remoto_claude` sin `exigir_contratos` es violación |
| `policy/tests/test_claude_subprocess_solo_via_sandbox.py` | Modificar | quitar `scripts/ejecutor_fase0/harness.py` de `_AISLADO_POR_CUENTA` |
| `scripts/ejecutor_fase0/harness.py` | Modificar | `correr` deja de lanzar (`Fase0Retirada`); `preparar` queda (su test) |
| `scripts/ejecutor_contratos/probar_arranque.py` | Crear | arranque real en hall9000 y sus roturas |
| `tests/test_ejecutor_contratos_arranque.py`, `tests/test_ejecutor_fase0_retirada.py` | Crear | tests puros |
| `.github/workflows/policy.yml` | Modificar | listas, pisos, guardia en `no-naked-claude-subprocess` |
| `DEUDA.md`, `CONTEXT.md` | Modificar | obligaciones de SP2 con fecha; cierre de SP1 |

## Interfaces

Consume: `verificar_c1` (plan 1), `exportar.exportar`, `instalacion.INSTALABLES`/`renderizar_*` (planes 1 y 3),
`verificar_c3` (plan 2), latido del freno `{"momento","activo","remotos_cargados","uid_resuelto"}` (plan 3),
`verificar_c5`, `eleccion_c5.*`, `auditor_cliente.auditar` (plan 4), `revocacion` marcas (plan 5), `Fallo`,
`cuenta_axioma.Cuenta`/`correr_en_la_cuenta`/`cuenta_desde_entorno`, `politica.validar`, `facet_resolver.resolve_facet`.

Produce:
```python
# jax/ejecutor/contratos/arranque.py
@dataclass(frozen=True)
class Contexto:
    cuenta: Cuenta; repo: Path; puerto_canario: int; registro: Path; puerto_proxy: int; sondas: tuple
    estado_freno: Path; llaves_root: Path; tope_gancho_s: int; hosts_mision: frozenset
    cron_deny: Path = Path("/etc/cron.deny"); linger_dir: Path = Path("/var/lib/systemd/linger")
    unidad_freno: Path = Path("/etc/systemd/system/ejecutor-freno.service")
    unidad_freno_habilitada: Path = Path("/etc/systemd/system/multi-user.target.wants/ejecutor-freno.service")
class ContratosNoVerificados(RuntimeError): fallos: tuple[Fallo, ...]
LATIDO_MAX_S = 2.0
def contexto_desde_entorno(env, hosts_mision: frozenset) -> Contexto
def verificar_instalacion(ctx: Contexto) -> tuple[Fallo, ...]
def verificar_c4_estatico(ctx: Contexto, *, ahora=time.time) -> tuple[Fallo, ...]
def remoto_c6(llaves_root: str) -> str
def leer_c6(salida: bytes) -> tuple[str, ...]                  # códigos de fallo de una máquina
async def verificar_c6_estatico(ctx: Contexto, hosts, *, correr=correr_en_la_cuenta) -> tuple[Fallo, ...]
def pruebas_reales(ctx: Contexto) -> dict                       # nombre -> async () -> tuple[Fallo, ...]
async def verificar_contratos(ctx: Contexto, pruebas: dict | None = None) -> tuple[Fallo, ...]
async def exigir_contratos(ctx: Contexto, pruebas: dict | None = None) -> None
```

**Contrato para SP2 (el transporte `harness`):** antes de CADA misión, `await exigir_contratos(ctx)` con los
`hosts_mision` de esa misión; si lanza, la misión no empieza y los `fallos` van a la UI como códigos.

---

### Task 1: `arranque.py`

**Files:**
- Create: `/home/fruiz/worktrees/jax-sp1-arranque/jax/ejecutor/contratos/arranque.py`
- Test: `/home/fruiz/worktrees/jax-sp1-arranque/tests/test_ejecutor_contratos_arranque.py`

**Por qué corre todas las pruebas y no se corta en la primera:** quien tiene que arreglar un arranque negado necesita
la lista completa; y cortar en la primera deja sin ejercitar las demás en el arranque en que sí pasan todas menos una.
La decisión es igual de cerrada: con un solo fallo, no arranca.

**Por qué una prueba que revienta cuenta como fallo:** una prueba que lanza una excepción no dijo «vivo». Se convierte
en `Fallo(<contrato>, "prueba_reventada")` — nunca en silencio.

- [ ] **Step 1: Test que falla**

```python
# tests/test_ejecutor_contratos_arranque.py
"""El arranque del Ejecutor se niega si un contrato no está vivo. Pruebas falsas,
archivos reales en tmp_path."""
import asyncio
import json
import os
import time
from pathlib import Path

import pytest

from jax.ejecutor.contratos import arranque as AR
from jax.ejecutor.contratos import instalacion
from jax.ejecutor.contratos.cuenta_axioma import Cuenta
from jax.ejecutor.contratos.fallo import Fallo


def _ctx(tmp_path, **cambios):
    base = dict(cuenta=Cuenta("axioma", 58291, Path("/k"), Path("/n"), tmp_path / "lib", tmp_path / "politica.json"),
                repo=Path(__file__).resolve().parents[1], puerto_canario=18436, registro=tmp_path / "registro.jsonl",
                puerto_proxy=18435, sondas=(7777,), estado_freno=tmp_path / "estado.json",
                llaves_root=Path("/etc/ssh/authorized_keys.d/axioma"), tope_gancho_s=10, hosts_mision=frozenset({"hall9000"}),
                cron_deny=tmp_path / "cron.deny", linger_dir=tmp_path / "linger", unidad_freno=tmp_path / "ejecutor-freno.service",
                unidad_freno_habilitada=tmp_path / "wants" / "ejecutor-freno.service")
    base.update(cambios)
    return AR.Contexto(**base)


def _vivas(**rotas):
    async def ok():
        return ()

    pruebas = {n: ok for n in ("instalacion", "exportar", "c1", "c3", "c4", "c5", "c6")}
    pruebas.update(rotas)
    return pruebas


def test_todo_vivo_arranca(tmp_path):
    asyncio.run(AR.exigir_contratos(_ctx(tmp_path), _vivas()))


def test_un_contrato_muerto_no_arranca_y_se_listan_todos(tmp_path):
    async def c1():
        return (Fallo("c1", "canario_no_bloqueado"),)

    async def c5():
        return (Fallo("c5", "canario_no_disparado"),)

    with pytest.raises(AR.ContratosNoVerificados) as e:
        asyncio.run(AR.exigir_contratos(_ctx(tmp_path), _vivas(c1=c1, c5=c5)))
    assert {f.codigo for f in e.value.fallos} == {"canario_no_bloqueado", "canario_no_disparado"}


def test_una_prueba_que_revienta_es_un_fallo(tmp_path):
    async def c3():
        raise OSError("sin red")

    fallos = asyncio.run(AR.verificar_contratos(_ctx(tmp_path), _vivas(c3=c3)))
    assert fallos == (Fallo("c3", "prueba_reventada", (("tipo", "OSError"),)),)


def test_faltar_una_prueba_es_un_fallo(tmp_path):
    pruebas = _vivas()
    del pruebas["c6"]
    assert asyncio.run(AR.verificar_contratos(_ctx(tmp_path), pruebas)) == (Fallo("c6", "prueba_ausente"),)


# --- C4 estático -------------------------------------------------------------

def _c4_sano(ctx):
    ctx.estado_freno.write_text(json.dumps({"momento": time.time(), "activo": False, "remotos_cargados": True,
                                            "uid_resuelto": True}))
    ctx.unidad_freno_habilitada.parent.mkdir(parents=True, exist_ok=True)
    ctx.unidad_freno_habilitada.write_text("")
    ctx.cron_deny.write_text("otro\naxioma\n")
    ctx.linger_dir.mkdir(exist_ok=True)


def test_c4_estatico_sano(tmp_path):
    ctx = _ctx(tmp_path)
    _c4_sano(ctx)
    assert AR.verificar_c4_estatico(ctx) == ()


@pytest.mark.parametrize("romper, codigo", [
    (lambda c: c.estado_freno.unlink(), "freno_sin_latido"),
    (lambda c: c.estado_freno.write_text(json.dumps({"momento": time.time() - 10, "activo": False,
                                                     "remotos_cargados": True, "uid_resuelto": True})), "freno_sin_latido"),
    (lambda c: c.estado_freno.write_text(json.dumps({"momento": time.time(), "activo": True,
                                                     "remotos_cargados": True, "uid_resuelto": True})), "interruptor_puesto"),
    (lambda c: c.estado_freno.write_text(json.dumps({"momento": time.time(), "activo": False,
                                                     "remotos_cargados": False, "uid_resuelto": True})), "freno_sin_remotos"),
    (lambda c: c.estado_freno.write_text(json.dumps({"momento": time.time(), "activo": False,
                                                     "remotos_cargados": True, "uid_resuelto": False})), "freno_sin_cuenta"),
    (lambda c: c.unidad_freno_habilitada.unlink(), "freno_no_habilitado"),
    (lambda c: c.cron_deny.write_text("axiomas\n"), "cron_abierto_para_la_cuenta"),
    (lambda c: (c.linger_dir / "axioma").write_text(""), "linger_activo"),
])
def test_c4_estatico_roto(tmp_path, romper, codigo):
    ctx = _ctx(tmp_path)
    _c4_sano(ctx)
    romper(ctx)
    assert codigo in [f.codigo for f in AR.verificar_c4_estatico(ctx)]


# --- C6 estático -------------------------------------------------------------

@pytest.mark.parametrize("salida, codigos", [
    (b"llaves=root 644\nfreno=1\nrevocador=root 755\n", ()),
    (b"llaves=axioma 600\nfreno=1\nrevocador=root 755\n", ("llaves_no_son_de_root",)),
    (b"llaves=root 666\nfreno=1\nrevocador=root 755\n", ("llaves_no_son_de_root",)),
    (b"llaves=root 644\nfreno=0\nrevocador=root 755\n", ("sin_llave_del_freno",)),
    (b"llaves=root 644\nfreno=1\nrevocador=\n", ("sin_revocador",)),
    (b"", ("llaves_no_son_de_root", "sin_llave_del_freno", "sin_revocador")),
])
def test_leer_c6(salida, codigos):
    assert AR.leer_c6(salida) == codigos


def test_c6_estatico_por_maquina_local_y_remota(tmp_path):
    from jax.ejecutor.contratos.destinos import Host
    hosts = (Host("hall9000", "127.0.0.1", 58291, "hypervisor", True), Host("bridge", "192.0.2.20", 58291, "clientes", False))
    vistos = []

    async def correr(c, remoto, *, entrada=b"", tope_s):
        vistos.append(remoto)
        if "192.0.2.20" in remoto:
            return 255, b"", b"Connection timed out"
        return 0, b"llaves=root 644\nfreno=1\nrevocador=root 755\n", b""

    fallos = asyncio.run(AR.verificar_c6_estatico(_ctx(tmp_path), hosts, correr=correr))
    assert fallos == (Fallo("c6", "maquina_inalcanzable", (("host", "bridge"),)),)
    assert vistos[0] == AR.remoto_c6("/etc/ssh/authorized_keys.d/axioma")
    assert vistos[1].startswith("ssh -o BatchMode=yes") and "axioma@192.0.2.20" in vistos[1]


# --- instalación ---------------------------------------------------------------

def _instalar_copia(ctx):
    for rel in instalacion.INSTALABLES:
        destino = ctx.cuenta.lib / rel
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_bytes((ctx.repo / rel).read_bytes())
    lib = str(ctx.cuenta.lib)
    (ctx.cuenta.lib / "gancho.sh").write_text(instalacion.renderizar_gancho(lib, ctx.tope_gancho_s))
    (ctx.cuenta.lib / "managed-settings.json").write_text(
        instalacion.renderizar_managed_settings(lib, str(ctx.cuenta.politica), ctx.tope_gancho_s))
    (ctx.cuenta.lib / "settings-usuario.json").write_text(instalacion.SETTINGS_USUARIO)
    ctx.unidad_freno.write_text(instalacion.renderizar_unidad_freno(lib))


def test_instalacion_identica_al_repo(tmp_path):
    ctx = _ctx(tmp_path)
    _instalar_copia(ctx)
    assert AR.verificar_instalacion(ctx) == ()


def test_instalacion_con_un_byte_distinto(tmp_path):
    ctx = _ctx(tmp_path)
    _instalar_copia(ctx)
    with open(ctx.cuenta.lib / "jax/ejecutor/contratos/politica.py", "a") as f:
        f.write("\n")
    (ctx.cuenta.lib / "gancho.sh").unlink()
    assert AR.verificar_instalacion(ctx) == (
        Fallo("arranque", "instalado_distinto_del_repo", (("archivo", "jax/ejecutor/contratos/politica.py"),)),
        Fallo("arranque", "instalado_distinto_del_repo", (("archivo", "gancho.sh"),)),
    )
```

- [ ] **Step 2: Correr y ver que falla** — `PYTHONPATH=.:las_manos python -m pytest tests/test_ejecutor_contratos_arranque.py -v` → `ImportError`.

- [ ] **Step 3: Implementar**

```python
# jax/ejecutor/contratos/arranque.py
"""El Ejecutor NO arranca si un contrato no está vivo.

Spec 2026-09-15 §4: «cada contrato existe y se vio fallar antes de que el Ejecutor
toque un servidor»; «C1 y C5 llevan una entrada canario permanente; si no se dispara,
el Ejecutor no arranca». Principio IX: el contrato antes que la capacidad.

`exigir_contratos` corre TODAS las pruebas (lista completa para quien tenga que
arreglar) y lanza ContratosNoVerificados con un solo fallo. Una prueba que revienta o
que falta es un fallo. La guardia policy/tests/test_ejecutor_lanza_solo_con_contratos.py
impide lanzar el cerebro del Ejecutor sin pasar por acá.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shlex
import time
from dataclasses import dataclass, field
from pathlib import Path

from jax.ejecutor.contratos import cuenta_axioma, instalacion, politica
from jax.ejecutor.contratos.cuenta_axioma import Cuenta
from jax.ejecutor.contratos.fallo import Fallo

LATIDO_MAX_S = 2.0
_ORDEN = ("instalacion", "exportar", "c1", "c3", "c4", "c5", "c6")
_CONTRATO_DE = {"instalacion": "arranque", "exportar": "c1"}


class ContratosNoVerificados(RuntimeError):
    def __init__(self, fallos):
        super().__init__(len(fallos))
        self.fallos = tuple(fallos)


@dataclass(frozen=True)
class Contexto:
    cuenta: Cuenta
    repo: Path
    puerto_canario: int
    registro: Path
    puerto_proxy: int
    sondas: tuple
    estado_freno: Path
    llaves_root: Path
    tope_gancho_s: int
    hosts_mision: frozenset
    cron_deny: Path = field(default=Path("/etc/cron.deny"))
    linger_dir: Path = field(default=Path("/var/lib/systemd/linger"))
    unidad_freno: Path = field(default=Path("/etc/systemd/system/ejecutor-freno.service"))
    unidad_freno_habilitada: Path = field(default=Path("/etc/systemd/system/multi-user.target.wants/ejecutor-freno.service"))


def contexto_desde_entorno(env, hosts_mision) -> Contexto:
    return Contexto(
        cuenta=cuenta_axioma.cuenta_desde_entorno(env), repo=Path(__file__).resolve().parents[3],
        puerto_canario=int(env["JAX_EJECUTOR_CANARIO_PUERTO"]), registro=Path(env["JAX_EJECUTOR_REGISTRO"]),
        puerto_proxy=int(env["JAX_PROXY_CARRIL_PUERTO"]),
        sondas=tuple(int(p) for p in env["JAX_EJECUTOR_CERCO_SONDAS"].split(",")),
        estado_freno=Path(env["JAX_EJECUTOR_FRENO_ESTADO"]), llaves_root=Path(env["JAX_EJECUTOR_LLAVES_ROOT"]),
        tope_gancho_s=int(env["JAX_EJECUTOR_GANCHO_TOPE_S"]), hosts_mision=frozenset(hosts_mision))


def _sha(datos: bytes) -> str:
    return hashlib.sha256(datos).hexdigest()


def verificar_instalacion(ctx: Contexto) -> tuple:
    lib = str(ctx.cuenta.lib)
    esperados = {rel: (ctx.repo / rel).read_bytes() for rel in instalacion.INSTALABLES}
    esperados["gancho.sh"] = instalacion.renderizar_gancho(lib, ctx.tope_gancho_s).encode()
    esperados["managed-settings.json"] = instalacion.renderizar_managed_settings(
        lib, str(ctx.cuenta.politica), ctx.tope_gancho_s).encode()
    esperados["settings-usuario.json"] = instalacion.SETTINGS_USUARIO.encode()
    fallos = []
    for rel, datos in esperados.items():
        try:
            igual = _sha((ctx.cuenta.lib / rel).read_bytes()) == _sha(datos)
        except OSError:
            igual = False
        if not igual:
            fallos.append(Fallo("arranque", "instalado_distinto_del_repo", (("archivo", rel),)))
    try:
        unidad_igual = ctx.unidad_freno.read_text() == instalacion.renderizar_unidad_freno(lib)
    except OSError:
        unidad_igual = False
    if not unidad_igual:
        fallos.append(Fallo("arranque", "instalado_distinto_del_repo", (("archivo", "ejecutor-freno.service"),)))
    return tuple(fallos)


def verificar_c4_estatico(ctx: Contexto, *, ahora=time.time) -> tuple:
    fallos = []
    try:
        latido = json.loads(ctx.estado_freno.read_text())
        fresco = ahora() - float(latido["momento"]) <= LATIDO_MAX_S
    except (OSError, ValueError, KeyError, TypeError):
        latido, fresco = {}, False
    if not fresco:
        fallos.append(Fallo("c4", "freno_sin_latido"))
    else:
        if latido.get("activo") is not False:
            fallos.append(Fallo("c4", "interruptor_puesto"))
        if latido.get("remotos_cargados") is not True:
            fallos.append(Fallo("c4", "freno_sin_remotos"))
        if latido.get("uid_resuelto") is not True:
            fallos.append(Fallo("c4", "freno_sin_cuenta"))
    if not ctx.unidad_freno_habilitada.exists():
        fallos.append(Fallo("c4", "freno_no_habilitado"))
    try:
        cerrado = ctx.cuenta.nombre in ctx.cron_deny.read_text().split()
    except OSError:
        cerrado = False
    if not cerrado:
        fallos.append(Fallo("c4", "cron_abierto_para_la_cuenta"))
    if (ctx.linger_dir / ctx.cuenta.nombre).exists():
        fallos.append(Fallo("c4", "linger_activo"))
    return tuple(fallos)


def remoto_c6(llaves_root: str) -> str:
    q = shlex.quote(llaves_root)
    return (f'echo "llaves=$(stat -c \'%U %a\' {q} 2>/dev/null)"; '
            f'echo "freno=$(grep -c \' ejecutor-freno$\' {q} 2>/dev/null || true)"; '
            'echo "revocador=$(stat -c \'%U %a\' /usr/local/sbin/ejecutor-revocar 2>/dev/null)"')


def leer_c6(salida: bytes) -> tuple:
    vistas = dict(l.split("=", 1) for l in salida.decode(errors="replace").splitlines() if "=" in l)
    codigos = []
    if vistas.get("llaves") != "root 644":
        codigos.append("llaves_no_son_de_root")
    if vistas.get("freno", "0").strip() in ("", "0"):
        codigos.append("sin_llave_del_freno")
    if vistas.get("revocador") != "root 755":
        codigos.append("sin_revocador")
    return tuple(codigos)


async def verificar_c6_estatico(ctx: Contexto, hosts, *, correr=cuenta_axioma.correr_en_la_cuenta) -> tuple:
    async def una(h):
        remoto = remoto_c6(str(ctx.llaves_root))
        if not h.es_local:
            remoto = (f"ssh -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=yes -p {int(h.puerto)} "
                      f"{ctx.cuenta.nombre}@{shlex.quote(h.ip)} {shlex.quote(remoto)}")
        rc, salida, _ = await correr(ctx.cuenta, remoto, tope_s=30)
        if rc != 0:
            return (Fallo("c6", "maquina_inalcanzable", (("host", h.nombre),)),)
        return tuple(Fallo("c6", c, (("host", h.nombre),)) for c in leer_c6(salida))

    resultados = await asyncio.gather(*(una(h) for h in hosts))
    return tuple(f for r in resultados for f in r)


def pruebas_reales(ctx: Contexto) -> dict:
    from facet_resolver import resolve_facet
    from jacobs.store import get_conn
    from jax.ejecutor.contratos import (auditor_cliente, canario_c1, canario_c3, canario_c5, eleccion_c5, exportar)

    def _politica():
        return politica.validar(json.loads(ctx.cuenta.politica.read_bytes()))

    async def p_instalacion():
        return verificar_instalacion(ctx)

    async def p_exportar():
        try:
            await exportar.exportar(ctx.cuenta.politica, get_conn)
        except exportar.ExportacionImposible as exc:
            return (Fallo("c1", "exportacion_imposible", (("codigo", exc.codigo),)),)
        return ()

    async def p_c1():
        return await canario_c1.verificar_c1(ctx.cuenta, puerto_canario=ctx.puerto_canario)

    async def p_c3():
        return await canario_c3.verificar_c3(ctx.cuenta, registro=ctx.registro, puerto_proxy=ctx.puerto_proxy,
                                             sondas=ctx.sondas)

    async def p_c4():
        return verificar_c4_estatico(ctx)

    async def p_c5():
        conn = await get_conn()
        try:
            cfg = await eleccion_c5.leer_config(conn)
            cerebro, auditor_f = await resolve_facet(cfg.cerebro_faceta), await resolve_facet(cfg.auditor_faceta)
            local = await eleccion_c5.es_local(conn, auditor_f.provider_id)
            con_clientes, conocidos = await eleccion_c5.hosts_con_clientes(conn)
        finally:
            conn.close()
        fallos = eleccion_c5.validar_eleccion(
            proveedor_cerebro=cerebro.provider_id, proveedor_auditor=auditor_f.provider_id, auditor_es_local=local,
            admite_datos_de_clientes=cfg.admite_datos_de_clientes, hosts_mision=ctx.hosts_mision,
            hosts_con_clientes=con_clientes, hosts_conocidos=conocidos)

        async def auditar(lote):
            return await auditor_cliente.auditar(lote, faceta=auditor_f, max_tokens=cfg.max_tokens)

        return tuple(fallos) + await canario_c5.verificar_c5(auditar)

    async def p_c6():
        return await verificar_c6_estatico(ctx, _politica().hosts)

    return {"instalacion": p_instalacion, "exportar": p_exportar, "c1": p_c1, "c3": p_c3, "c4": p_c4,
            "c5": p_c5, "c6": p_c6}


async def verificar_contratos(ctx: Contexto, pruebas: dict | None = None) -> tuple:
    pruebas = pruebas if pruebas is not None else pruebas_reales(ctx)
    fallos = []
    for nombre in _ORDEN:
        contrato = _CONTRATO_DE.get(nombre, nombre)
        prueba = pruebas.get(nombre)
        if prueba is None:
            fallos.append(Fallo(contrato, "prueba_ausente"))
            continue
        try:
            fallos.extend(await prueba())
        except Exception as exc:  # fail-soft: una prueba que revienta se reporta como Fallo y el arranque se niega (cerrado)
            fallos.append(Fallo(contrato, "prueba_reventada", (("tipo", type(exc).__name__),)))
    return tuple(fallos)


async def exigir_contratos(ctx: Contexto, pruebas: dict | None = None) -> None:
    fallos = await verificar_contratos(ctx, pruebas)
    if fallos:
        raise ContratosNoVerificados(fallos)
```

Nota: las pruebas corren en orden y no en paralelo a propósito: la exportación va antes que el canario de C1 (que usa
la política recién exportada), y el canario de C1 usa un puerto fijo. El costo se mide en la Task 4.

- [ ] **Step 4: Pasa** — 22 PASS; P10 verde; guardia de subprocesos verde (`arranque.py` no lanza procesos).

- [ ] **Step 5: Mutaciones** (backup y `cmp`): (1) `fallos.append(Fallo(contrato, "prueba_reventada", …))` → `pass` en
  ese `except` (quitando la marca: el propio escáner P10 tiene que dar ROJO, y además) → FAIL
  `test_una_prueba_que_revienta_es_un_fallo`; (2) en `exigir_contratos`, `if fallos:` → `if False:` → FAIL
  `test_un_contrato_muerto_no_arranca_y_se_listan_todos`.

- [ ] **Step 6: Commit** `feat(ejecutor): arranque que exige los seis contratos vivos, con lista completa de fallos`.

---

### Task 2: la guardia de CI y el retiro del lanzador de la Fase 0

**Files:**
- Create: `/home/fruiz/worktrees/jax-sp1-arranque/policy/tests/test_ejecutor_lanza_solo_con_contratos.py`
- Modify: `/home/fruiz/worktrees/jax-sp1-arranque/scripts/ejecutor_fase0/harness.py` (función `correr`)
- Modify: `/home/fruiz/worktrees/jax-sp1-arranque/policy/tests/test_claude_subprocess_solo_via_sandbox.py` (quitar la entrada de `harness.py` de `_AISLADO_POR_CUENTA`)
- Test: `/home/fruiz/worktrees/jax-sp1-arranque/tests/test_ejecutor_fase0_retirada.py`

**Por qué retirar `harness.correr`:** lanza `claude` como la cuenta **sin jaula y sin gancho** (C1 no aplica) y
apuntando directo a Ollama (C3 no lo ve; el cerco del plan 2 ya lo corta). Después de SP1 es un camino que esquiva
cuatro contratos. `preparar` queda: su test documenta que la llave viaja por stdin, y no lanza nada.

- [ ] **Step 1: Tests que fallan**

```python
# policy/tests/test_ejecutor_lanza_solo_con_contratos.py
"""Nadie lanza el cerebro del Ejecutor sin exigir los contratos (spec 2026-09-15 §4;
Principio IX). Guardia de CI: un módulo que usa `remoto_claude` (el lanzamiento del
arnés dentro de la jaula) sin `exigir_contratos` es violación. Los archivos declarados
como aislados por cuenta en la guardia de subprocesos también tienen que exigirlos.

Permitidos, con su razón:
- cuenta_axioma.py: define el lanzador;
- canario_c1.py: ES el canario de C1, corre dentro de exigir_contratos;
- arranque.py: es quien exige.
Corre con: python -m pytest policy/tests/test_ejecutor_lanza_solo_con_contratos.py -v
"""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
LANZADOR = "remoto_claude"
EXIGIR = "exigir_contratos"
PERMITIDOS = {
    "jax/ejecutor/contratos/cuenta_axioma.py": "define el lanzador",
    "jax/ejecutor/contratos/canario_c1.py": "es el canario de C1: corre dentro de exigir_contratos",
    "jax/ejecutor/contratos/arranque.py": "es quien exige",
}


def _nombres(arbol: ast.AST) -> set[str]:
    vistos = set()
    for n in ast.walk(arbol):
        if isinstance(n, ast.Name):
            vistos.add(n.id)
        elif isinstance(n, ast.Attribute):
            vistos.add(n.attr)
        elif isinstance(n, ast.alias):
            vistos.add(n.name.rsplit(".", 1)[-1])
    return vistos


def viola(fuente: str) -> bool:
    nombres = _nombres(ast.parse(fuente))
    return LANZADOR in nombres and EXIGIR not in nombres


def _aislados_por_cuenta() -> dict:
    ruta = RAIZ / "policy" / "tests" / "test_claude_subprocess_solo_via_sandbox.py"
    spec = importlib.util.spec_from_file_location("guardia_subprocesos", ruta)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo._AISLADO_POR_CUENTA


def _archivos():
    for ruta in RAIZ.rglob("*.py"):
        rel = ruta.relative_to(RAIZ).as_posix()
        if rel.startswith(("tests/", "policy/", ".venv/", "las_manos/.venv/")) or "/site-packages/" in rel:
            continue
        yield rel, ruta


def test_nadie_lanza_sin_exigir():
    violaciones = []
    for rel, ruta in _archivos():
        if rel in PERMITIDOS:
            continue
        try:
            fuente = ruta.read_text(encoding="utf-8")
            if LANZADOR in fuente and viola(fuente):
                violaciones.append(rel)
        except (SyntaxError, UnicodeDecodeError):  # fail-soft: un archivo que no parsea (p. ej. _director_patch) no puede importarse ni lanzar nada
            continue
    assert violaciones == [], violaciones


def test_los_aislados_por_cuenta_exigen_o_son_el_lanzador():
    for rel in _aislados_por_cuenta():
        if rel in PERMITIDOS:
            continue
        assert EXIGIR in _nombres(ast.parse((RAIZ / rel).read_text(encoding="utf-8"))), rel


def test_detecta_el_uso_directo_y_por_alias():
    assert viola("from jax.ejecutor.contratos.cuenta_axioma import remoto_claude\nremoto_claude(c)\n")
    assert viola("from jax.ejecutor.contratos import cuenta_axioma as ca\nca.remoto_claude(c)\n")


def test_acepta_el_que_exige():
    assert not viola("from jax.ejecutor.contratos import arranque, cuenta_axioma\n"
                     "async def f(ctx):\n    await arranque.exigir_contratos(ctx)\n    cuenta_axioma.remoto_claude(ctx.cuenta)\n")


def test_los_permitidos_existen():
    for rel in PERMITIDOS:
        assert (RAIZ / rel).is_file(), rel
```

```python
# tests/test_ejecutor_fase0_retirada.py
"""El lanzador de la Fase 0 ya no lanza: esquivaba la jaula, el gancho y el registro."""
import importlib.util
from pathlib import Path

import pytest

RUTA = Path(__file__).resolve().parents[1] / "scripts" / "ejecutor_fase0" / "harness.py"


def _harness():
    spec = importlib.util.spec_from_file_location("fase0_harness_retirada", RUTA)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_correr_ya_no_lanza():
    h = _harness()
    with pytest.raises(h.Fase0Retirada):
        h.correr("http://127.0.0.1:18435", "m", "hola")


def test_el_modulo_no_llama_subprocesos():
    fuente = RUTA.read_text(encoding="utf-8")
    assert "subprocess.run(" not in fuente and "import subprocess" not in fuente
```

- [ ] **Step 2: Correr y ver que fallan** — `PYTHONPATH=.:las_manos python -m pytest policy/tests/test_ejecutor_lanza_solo_con_contratos.py tests/test_ejecutor_fase0_retirada.py -v`
  → FAIL: `test_los_aislados_por_cuenta_exigen_o_son_el_lanzador` nombra `scripts/ejecutor_fase0/harness.py`, y los dos de la retirada.

- [ ] **Step 3: Implementar**

En `scripts/ejecutor_fase0/harness.py`, quitar `import subprocess` y reemplazar `correr`:

```python
class Fase0Retirada(RuntimeError):
    """Retirado el 2026-09-17 (Ejecutor SP1, plan 6): lanzaba el arnés como la cuenta sin
    jaula ni gancho (C1) y directo a Ollama (C3). El lanzamiento vive en el transporte de SP2,
    detrás de jax/ejecutor/contratos/arranque.py::exigir_contratos."""


def correr(base_url, modelo, prompt, llave="ollama", **kw):
    raise Fase0Retirada("fase0_retirada")
```

y actualizar el docstring del módulo con una línea: `RETIRADO para lanzar (2026-09-17): ver Fase0Retirada.`
En `policy/tests/test_claude_subprocess_solo_via_sandbox.py`, borrar la entrada `"scripts/ejecutor_fase0/harness.py": …`
de `_AISLADO_POR_CUENTA` (una exención de un archivo que ya no lanza es un hueco a la espera).

- [ ] **Step 4: Pasan** los 5 + 2; `python -m pytest policy/tests/test_claude_subprocess_solo_via_sandbox.py -q` sigue verde
  (su test de snippet «un archivo declarado que pierde el ssh vuelve a ser violación» usa snippets, no el archivo real:
  confirmarlo leyendo el test antes de dar por bueno el verde).

- [ ] **Step 5: Verla fallar** — con backup: en un archivo nuevo `jax/ejecutor/lanzador_de_prueba.py` escribir
  `from jax.ejecutor.contratos.cuenta_axioma import remoto_claude` y una llamada → FAIL `test_nadie_lanza_sin_exigir`
  nombrándolo. Borrar el archivo → verde.

- [ ] **Step 6: CI** — en el job `no-naked-claude-subprocess` de `.github/workflows/policy.yml`:

```yaml
      - run: python -m pytest policy/tests/test_claude_subprocess_solo_via_sandbox.py policy/tests/test_ejecutor_lanza_solo_con_contratos.py -v
      - name: Piso exacto de la guardia del arranque del Ejecutor
        # 5 tests (Ejecutor SP1 plan 6, 2026-09-17). Exacto: uno que desaparezca deja pasar un lanzador sin contratos.
        run: |
          python -m pytest -q policy/tests/test_ejecutor_lanza_solo_con_contratos.py 2>&1 | tee /tmp/lanza
          grep -qE "^5 passed" /tmp/lanza || { echo "PISO ROTO: se esperaban 5 tests CORRIDOS."; exit 1; }
```

(reemplazando el `run` único que había). `tests/test_ejecutor_contratos_arranque.py` y
`tests/test_ejecutor_fase0_retirada.py` en las dos listas de `tests-puros`; piso con el número del runner. **Canario
rojo por API** sobre el sha con el archivo lanzador de prueba del Step 5 → `no-naked-claude-subprocess` = `failure`;
revert → `success`.

- [ ] **Step 7: Commit** `feat(ejecutor): guardia de CI — nadie lanza el cerebro del Ejecutor sin exigir los contratos; se retira el lanzador de la Fase 0`.

---

### Task 3: hall9000 · el arranque real; y VERLO NEGARSE, contrato por contrato

**Files:**
- Create: `/home/fruiz/worktrees/jax-sp1-arranque/scripts/ejecutor_contratos/probar_arranque.py`

- [ ] **Step 1: El script**

```python
#!/usr/bin/env python3
# scripts/ejecutor_contratos/probar_arranque.py
"""Arranque real del Ejecutor en hall9000 SIN lanzar ninguna misión: corre exigir_contratos
para una misión sobre las máquinas dadas y dice si arrancaría, con cada fallo y el tiempo.

Uso: set -a; . <(sudo -n cat /etc/jax/.env); set +a
     PYTHONPATH=.:las_manos python3 scripts/ejecutor_contratos/probar_arranque.py [--instrucciones-auditor ARCHIVO] hall9000 [otra …]
`--instrucciones-auditor` existe sólo para VER FALLAR a C5 sin escribir en la DB (Task 3): cambia, en este proceso,
el archivo de instrucciones del auditor. El arranque de producción no tiene esa opción.
"""
import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

from jax.ejecutor.contratos import arranque, auditor_cliente, formato


async def principal(args) -> int:
    ctx = arranque.contexto_desde_entorno(os.environ, frozenset(args.hosts))
    if args.instrucciones_auditor:
        auditor_cliente._INSTRUCCIONES = Path(args.instrucciones_auditor)
    inicio = time.monotonic()
    try:
        await arranque.exigir_contratos(ctx)
    except arranque.ContratosNoVerificados as exc:
        for f in exc.fallos:
            print(formato.campos((("contrato", f.contrato), ("codigo", f.codigo)) + tuple(f.datos)))
        print(formato.campos((("arrancaria", False), ("segundos", round(time.monotonic() - inicio, 1)))))
        return 1
    print(formato.campos((("arrancaria", True), ("segundos", round(time.monotonic() - inicio, 1)))))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--instrucciones-auditor")
    ap.add_argument("hosts", nargs="+")
    sys.exit(asyncio.run(principal(ap.parse_args())))
```

- [ ] **Step 2: Verlo arrancar** — Run: `cd /home/fruiz/jax && set -a && . /etc/jax/.env && set +a && PYTHONPATH=.:las_manos python3 scripts/ejecutor_contratos/probar_arranque.py hall9000`
Expected: `arrancaria=true` y el tiempo (se registra). **Hoy, sin la faceta `ejecutor` (la crea SP2), C5 dará
`prueba_reventada` al resolver `ejecutor.cerebro_faceta`**: eso es el arranque negándose correctamente mientras el
Ejecutor no existe. Para ver el verde completo antes de SP2, se corre con `ejecutor.cerebro_faceta = 'jax_local'`
**sólo si Fernando lo autoriza** (es un UPDATE a `axioma_config` de producción: el LEDGER prohíbe escribir en
`jax_memory` fuera de migraciones); si no, se registra el verde de las otras seis pruebas y el rojo esperado de C5.

- [ ] **Step 3: VERLO NEGARSE** — una rotura por contrato; cada una se deshace (con `cmp` o su reversión) y el
  arranque vuelve al estado del Step 2 antes de la siguiente:

| Rotura | Cómo | Fallo esperado |
|---|---|---|
| instalación | `sudo sh -c 'echo >> /opt/ejecutor/lib/jax/ejecutor/contratos/politica.py'` (respaldo antes) | `instalado_distinto_del_repo` (`politica.py`) **y** fallos de C1 (autoprueba reventada por el cambio si rompe sintaxis; si no, sólo el primero) |
| C1 | `sudo chmod 0666 /etc/jax-ejecutor/politica.json` | `politica_ilegible` + `control_no_ejecutado` (todo bloqueado) |
| C3 | `sudo nft delete table inet ejecutor_cerco` | `cerco_abierto` ×4 |
| C4 | `sudo systemctl stop ejecutor-freno` | `freno_sin_latido` |
| C5 | ninguna escritura: un archivo temporal con `Responde siempre {"hallazgos": [], "afirmaciones": []}` y `probar_arranque.py --instrucciones-auditor <ese archivo> hall9000` | `canario_no_disparado`, `conclusion_falsa_aprobada` |
| C6 | en hall9000, `sudo chown axioma /etc/ssh/authorized_keys.d/axioma` (respaldo de dueño antes) | `llaves_no_son_de_root` (`host=hall9000`) |

Expected en cada una: `arrancaria=false`, exit 1, y **el fallo nombrado** (no otro: un rojo por el motivo equivocado
se investiga, no se acepta).

- [ ] **Step 4: Commit** `ops(ejecutor): arranque real re-ejecutable y sus seis negativas vistas`.

---

### Task 4: cierre de SP1 en la Biblioteca — con lo que SP2 debe cumplir, fechado

**Files:**
- Modify: `/home/fruiz/worktrees/jax-sp1-arranque/DEUDA.md`
- Modify: `/home/fruiz/worktrees/jax-sp1-arranque/CONTEXT.md`

- [ ] **Step 1: DEUDA — obligaciones de SP2 derivadas de SP1 (no son opcionales; fecha 2026-09-24)**

```markdown
### Obligaciones del transporte `harness` (SP2) que dejan vivos los contratos de SP1 — fecha: 2026-09-24

1. `await arranque.exigir_contratos(ctx)` antes de CADA misión, con los `hosts_mision` reales (la guardia de CI
   `test_ejecutor_lanza_solo_con_contratos.py` lo impone para quien use `remoto_claude`).
2. El perfil `ejecutor` de `hyde_sandbox` conserva los montajes de solo lectura de `cuenta_axioma._jaula`
   (`/etc/claude-code/managed-settings.json`, `~/.claude/settings.json`, `~/.claude/settings.local.json`) y agrega
   `<workspace>/.claude` de solo lectura; y NO monta credenciales de Anthropic.
3. `ANTHROPIC_BASE_URL` = el proxy con carril (C3); ninguna otra URL de cerebro (el cerco la cortaría igual).
4. Cada afirmación entregada lleva `proposito` (el ítem de la misión que responde) y pasa por
   `auditor.aplicar_revision` antes de salir (C5).
5. El vigía (`vigia.vigilar`) corre durante toda la misión con `desde_byte` = tamaño del registro al empezar; `fin` se
   pone SÓLO al terminar la misión normalmente.
6. Informe final afirmación ↔ evidencia (spec §4 C5) con las `Descartada` de citas y auditor.
7. Kimi/GLM como cerebro: el proxy hoy tiene UN upstream (Ollama). Un cerebro de nube necesita su propio proxy
   anotado o un proxy con upstream por misión; sin eso, C3 no lo ve y el cerco lo corta.
```

y la **DECISIÓN pendiente** (del plan 4): compuerta `ejecutor.c5_auditor_admite_datos_de_clientes`, a preguntar a
Fernando el 2026-09-18, con las tres opciones del índice.

- [ ] **Step 2: CONTEXT.md §9** — entrada de cierre de SP1: los seis planes con sus PR y sha; cada «verlo fallar» con su
  salida literal; los números medidos (latencia del gancho p95, sobrecarga del registro, corte del freno, tasa del
  canario de C5, tokens por lote, duración del arranque); las cuatro decisiones que se apartaron del spec (índice,
  D-SP1-2 a D-SP1-5) y por qué; y la lección de la sesión.

- [ ] **Step 3: Commit** `docs(biblioteca): cierre de SP1 — seis contratos vistos fallar; obligaciones de SP2 con fecha`.

---

## Auto-revisión (hecha al escribir)

- **Spec §4:** «si el canario no se dispara, el Ejecutor no arranca» ✓ (`exigir_contratos` con C1 y C5 reales en cada
  arranque); «gate verificable por un tercero» ✓ (guardia de CI con piso y canario rojo; `probar_arranque.py`
  re-ejecutable); cada contrato se vio fallar ✓ (Task 3, uno por uno, además de los de cada plan).
- **Principio IX:** el lanzador sin contratos de la Fase 0 se retira ✓.
- **Placeholders:** ninguno. La única condición abierta (C5 sin faceta `ejecutor` hasta SP2) está escrita con su
  resultado esperado y con qué hacer, sin escribir en producción sin permiso.
- **Tipos:** `Contexto`, `exigir_contratos(ctx, pruebas=None)`, `verificar_c6_estatico(ctx, hosts, *, correr)` iguales
  en Interfaces, código y tests.
