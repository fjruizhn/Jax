# Ejecutor SP1 · Plan 5 · C6 revocación de un golpe

> **Para agentes ejecutores:** SUB-SKILL REQUERIDA: usar `superpowers:subagent-driven-development`
> (recomendado) o `superpowers:executing-plans`. Los pasos usan casillas (`- [ ]`).

**Objetivo:** que un solo script quite el acceso de la cuenta del Ejecutor en todas las máquinas del inventario,
**corte también las sesiones ya abiertas**, y compruebe máquina por máquina que no entra — distinguiendo «no entra
porque se revocó» de «no pude comprobarlo».

**Arquitectura:** en cada máquina, las llaves de `axioma` pasan a un archivo **de root**
(`Match User axioma` → `AuthorizedKeysFile /etc/ssh/authorized_keys.d/%u`): la cuenta ya no puede agregarse una llave
de repuesto antes de que la revoquen. Cada línea lleva una marca: `ejecutor-axioma` (la llave del Ejecutor),
`ejecutor-controlador` (sólo en hall9000: la de `fruiz` para lanzar), `ejecutor-freno` (la del freno de C4, con
comando forzado; **no** se revoca). `ejecutor-revocar` (root) quita las dos primeras, las guarda aparte y mata los
procesos de la cuenta. `revocacion.py` orquesta: primero las remotas en paralelo (comprobando desde la cuenta, que
todavía entra a hall9000), después hall9000.

**Tech stack:** Python 3.12, `asyncio`; `sh`; OpenSSH (`sshd -t`, `sshd -T -C`); sudo (`visudo -cf`).

**Spec:** `docs/superpowers/specs/2026-09-15-ejecutor-design.md` §4 C6, §3.4 (inventario), D5 (usuario `axioma`, GO por
máquina); C3 (log de `sudo` de `axioma` en cada servidor). Decisión D-SP1-5 del índice
`docs/superpowers/plans/2026-09-17-ejecutor-contratos-0-indice.md`.

## Dos partes, a propósito separadas

- **Parte A (Tasks 1–4):** código, tests en CI, y la prueba sobre **hall9000**. No toca otras máquinas.
- **Parte B (Tasks 5–7):** instalación y **prueba real de revocación contra .11, .10 y .20**, en ese orden.
  **NO DESTRUCTIVA:** en cada máquina se respalda el archivo de llaves, se revoca, se comprueba que no entra, se
  **repone**, se comprueba con `cmp` que quedó idéntico y que vuelve a entrar. No se toca ningún dato ni servicio de
  clientes. Sí cambia configuración de `sshd` y agrega un `sudoers.d` en servidores de clientes: el spec (D5) pone el
  alta de la cuenta en cada máquina bajo **GO de Fernando por máquina**. **El coordinador decide si la orden de
  autonomía del 2026-09-17 lo cubre; si no, la Parte B espera ese GO, máquina por máquina.**

## Global Constraints

- **Nunca cambiar de rama en `/home/fruiz/jax`.** Worktree `/home/fruiz/worktrees/jax-sp1-c6`, rama `feat/ejecutor-c6`
  desde `master` con los planes 1 y 3 mergeados (usa el inventario y el comando forzado del freno).
- **`/etc/jax/.env` apunta a PRODUCCIÓN.** Ningún test la carga.
- **P10:** marca `# fail-soft: <razón específica>` en la línea del `except` amplio sin `raise`.
- **Piso de `tests-puros`** en las dos listas, número del runner.
- **Mutaciones:** `PYTHONDONTWRITEBYTECODE=1`; backup `.mut-bak` y `cmp`.
- **Tests con procesos:** `fork` explícito; `ps`, nunca `pgrep -f`. **Ningún test corre `pkill` de verdad**: se prueba
  con un `pkill` falso en `PATH`.
- **Cero strings visibles hardcodeados:** códigos. **Sin hardcoding:** usuario administrador remoto, cuenta y rutas
  desde `JAX_EJECUTOR_*`; IPs y puertos desde el inventario (DB → política).
- **Nada destructivo sobre .10/.11/.20.** Toda operación de la Parte B tiene su reversión escrita **antes** del paso y
  una sesión de administrador abierta mientras se recarga `sshd`.
- **Una máquina caída no prueba una revocación:** sólo `Permission denied (publickey)` cuenta como «no entra».
- Commits en castellano, terminados con `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`.

## Mapa de archivos

| Archivo | Acción | Responsabilidad |
|---|---|---|
| `ops/ejecutor/ejecutor-revocar` | Crear | root en cada máquina: quita las marcas de acceso, guarda, mata sesiones |
| `jax/ejecutor/contratos/revocacion.py` | Crear | clasificar intentos, orquestar, comandos por máquina |
| `scripts/ejecutor_contratos/revocar.py` | Crear | el botón: revoca todo (con confirmación explícita) |
| `scripts/ejecutor_contratos/probar_c6.py` | Crear | prueba no destructiva por máquina: revocar, comprobar, reponer, comprobar |
| `ops/ejecutor/instalar_en_maquina.sh` | Crear | Parte B: archivo root de llaves, drop-in de sshd, `sudoers.d` de registro, cron/at/linger, scripts |
| `tests/test_ejecutor_revocar_sh.py`, `tests/test_ejecutor_contratos_revocacion.py` | Crear | tests puros |
| `scripts/ejecutor_contratos/probar_c4.py` | Modificar | `--remoto <nombre>`: el `ssh -tt` del simulacro contra otra máquina |
| `.github/workflows/policy.yml` | Modificar | listas y piso |

## Interfaces

Consume: `cuenta_axioma.Cuenta`, `correr_en_la_cuenta` (plan 1); `politica.validar`, `destinos.Host` (plan 1);
`Fallo`, `formato.campos` (plan 1).

Produce:
```python
# jax/ejecutor/contratos/revocacion.py
MARCAS_DE_ACCESO = ("ejecutor-axioma", "ejecutor-controlador")
MARCA_FRENO = "ejecutor-freno"
ENTRA, DENEGADO, INALCANZABLE = "entra", "denegado", "inalcanzable"
REVOCADA, SIGUE_ENTRANDO, NO_VERIFICABLE, ERROR_AL_REVOCAR = "revocada", "sigue_entrando", "no_verificable", "error_al_revocar"
@dataclass(frozen=True) class Resultado: host: str; estado: str; detalle: tuple
def clasificar_intento(rc: int, stderr: bytes) -> str
def remoto_probar_entrada(h: Host, cuenta: str) -> str              # comando que corre DENTRO de la cuenta en hall9000
def argv_admin(h: Host, usuario_admin: str, remoto: str) -> list[str]
async def revocar_todas(hosts, *, revocar_en, probar_entrada) -> tuple[Resultado, ...]
```

`ejecutor-revocar <archivo> <cuenta>` → stdout `revocar=ok quitadas=<n> quedan=0` (exit 0), o `revocar=error codigo=<c>` (exit 2).

---

### Task 1: `ejecutor-revocar`

**Files:**
- Create: `/home/fruiz/worktrees/jax-sp1-c6/ops/ejecutor/ejecutor-revocar`
- Test: `/home/fruiz/worktrees/jax-sp1-c6/tests/test_ejecutor_revocar_sh.py`

- [ ] **Step 1: Test que falla**

```python
# tests/test_ejecutor_revocar_sh.py
"""ejecutor-revocar (C6): quita las llaves de acceso de la cuenta, deja la del freno,
guarda lo quitado y mata las sesiones vivas. pkill es FALSO: anota y sale."""
import subprocess
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
GUION = RAIZ / "ops" / "ejecutor" / "ejecutor-revocar"
LLAVES = (
    "ssh-ed25519 AAAAejecutor ejecutor-axioma\n"
    'command="/usr/local/sbin/ejecutor-freno-remoto",restrict ssh-ed25519 AAAAfreno ejecutor-freno\n'
    "ssh-ed25519 AAAAcontrolador ejecutor-controlador\n"
)


def _entorno(tmp_path, rc_pkill=1):
    bin_ = tmp_path / "bin"
    bin_.mkdir(exist_ok=True)
    (bin_ / "pkill").write_text(f"#!/bin/sh\necho \"$@\" >> {tmp_path / 'pkill.log'}\nexit {rc_pkill}\n")
    (bin_ / "pkill").chmod(0o755)
    return {"PATH": f"{bin_}:/usr/bin:/bin"}


def _correr(tmp_path, *args, rc_pkill=1):
    return subprocess.run([str(GUION), *args], env=_entorno(tmp_path, rc_pkill), capture_output=True, timeout=20)


def test_quita_acceso_deja_freno_guarda_y_mata(tmp_path):
    archivo = tmp_path / "axioma"
    archivo.write_text(LLAVES)
    r = _correr(tmp_path, str(archivo), "axioma")
    assert (r.returncode, r.stdout) == (0, b"revocar=ok quitadas=2 quedan=0\n")
    assert archivo.read_text() == 'command="/usr/local/sbin/ejecutor-freno-remoto",restrict ssh-ed25519 AAAAfreno ejecutor-freno\n'
    (guardado,) = tmp_path.glob("axioma.revocadas-*")
    assert guardado.read_text() == "ssh-ed25519 AAAAejecutor ejecutor-axioma\nssh-ed25519 AAAAcontrolador ejecutor-controlador\n"
    assert (tmp_path / "pkill.log").read_text() == "-KILL -u axioma\n"


def test_sin_procesos_vivos_tambien_es_ok(tmp_path):
    archivo = tmp_path / "axioma"
    archivo.write_text(LLAVES)
    assert _correr(tmp_path, str(archivo), "axioma", rc_pkill=0).returncode == 0


def test_pkill_que_falla_de_verdad_es_error(tmp_path):
    archivo = tmp_path / "axioma"
    archivo.write_text(LLAVES)
    r = _correr(tmp_path, str(archivo), "axioma", rc_pkill=3)
    assert r.returncode == 2 and r.stdout == b"revocar=error codigo=pkill\n"


def test_revocar_dos_veces_es_idempotente(tmp_path):
    archivo = tmp_path / "axioma"
    archivo.write_text(LLAVES)
    _correr(tmp_path, str(archivo), "axioma")
    r = _correr(tmp_path, str(archivo), "axioma")
    assert r.stdout == b"revocar=ok quitadas=0 quedan=0\n"


def test_argumentos_y_archivo(tmp_path):
    assert _correr(tmp_path).stdout == b"revocar=error codigo=argumentos\n"
    assert _correr(tmp_path, str(tmp_path / "no-existe"), "axioma").stdout == b"revocar=error codigo=sin_archivo\n"
```

- [ ] **Step 2: Correr y ver que falla** — `PYTHONPATH=.:las_manos python -m pytest tests/test_ejecutor_revocar_sh.py -v` → FAIL (no existe).

- [ ] **Step 3: Implementar**

```sh
#!/bin/sh
# ops/ejecutor/ejecutor-revocar — C6, root, en cada máquina del inventario (/usr/local/sbin, 0755).
# Uso: ejecutor-revocar <archivo de llaves de la cuenta> <cuenta>
# Quita toda línea marcada ejecutor-axioma o ejecutor-controlador, la guarda en
# <archivo>.revocadas-<momento> y mata todo proceso de la cuenta: quitar una llave NO
# corta una conexión ya abierta. La llave del freno (ejecutor-freno) se queda: después
# de revocar, el freno tiene que seguir pudiendo entrar a matar.
set -u
test $# -eq 2 || { echo "revocar=error codigo=argumentos"; exit 2; }
archivo=$1
cuenta=$2
test -f "$archivo" || { echo "revocar=error codigo=sin_archivo"; exit 2; }
marcas=' ejecutor-(axioma|controlador)$'
# Con nanosegundos: dos revocaciones en el mismo segundo no pisan lo guardado.
momento=$(date -u +%Y%m%dT%H%M%S.%NZ)
grep -E "$marcas" "$archivo" > "$archivo.revocadas-$momento"
grep -vE "$marcas" "$archivo" > "$archivo.nuevo"
chmod 0644 "$archivo.nuevo"
mv "$archivo.nuevo" "$archivo" || { echo "revocar=error codigo=reemplazo"; exit 2; }
pkill -KILL -u "$cuenta"
rc=$?
# pkill: 0 = mató algo, 1 = no había nada; cualquier otro código es un fallo real.
if [ "$rc" -ne 0 ] && [ "$rc" -ne 1 ]; then echo "revocar=error codigo=pkill"; exit 2; fi
quitadas=$(wc -l < "$archivo.revocadas-$momento" | tr -d ' ')
quedan=$(grep -cE "$marcas" "$archivo")
echo "revocar=ok quitadas=$quitadas quedan=$quedan"
```

- [ ] **Step 4: Correr y ver que pasan**

Run: `cd /home/fruiz/worktrees/jax-sp1-c6 && chmod +x ops/ejecutor/ejecutor-revocar && git -C /home/fruiz/worktrees/jax-sp1-c6 update-index --add --chmod=+x ops/ejecutor/ejecutor-revocar; PYTHONPATH=.:las_manos python -m pytest tests/test_ejecutor_revocar_sh.py -v`
Expected: 5 PASS.

- [ ] **Step 5: Mutación** — `if [ "$rc" -ne 0 ] && [ "$rc" -ne 1 ]` → `if false` → FAIL `test_pkill_que_falla_de_verdad_es_error`.

- [ ] **Step 6: Commit** `feat(ejecutor): ejecutor-revocar — quita el acceso, deja el freno y corta las sesiones vivas`.

---

### Task 2: `revocacion.py` — clasificar y orquestar

**Files:**
- Create: `/home/fruiz/worktrees/jax-sp1-c6/jax/ejecutor/contratos/revocacion.py`
- Test: `/home/fruiz/worktrees/jax-sp1-c6/tests/test_ejecutor_contratos_revocacion.py`

**Por qué el orden (remotas primero, hall9000 al final):** la comprobación «no entra» a una remota se hace **desde la
cuenta** (con su propia llave, que vive en su home de hall9000 y `fruiz` no puede leer). Si se revocara hall9000
primero, ya no habría cómo entrar a la cuenta para comprobar las remotas. hall9000 se comprueba al final desde `fruiz`
con la llave del controlador.

- [ ] **Step 1: Test que falla**

```python
# tests/test_ejecutor_contratos_revocacion.py
"""Orquestación de C6: el orden, la clasificación y que «no pude comprobar» nunca
cuente como «revocada»."""
import asyncio

import pytest

from jax.ejecutor.contratos import revocacion as R
from jax.ejecutor.contratos.destinos import Host

HOSTS = (Host("hall9000", "127.0.0.1", 58291, "hypervisor", True), Host("atemai", "192.0.2.11", 58291, "desarrollo", False),
         Host("bridge", "192.0.2.20", 58291, "clientes", False))


@pytest.mark.parametrize("rc, stderr, esperado", [
    (0, b"", R.ENTRA),
    (255, b"axioma@192.0.2.11: Permission denied (publickey).\r\n", R.DENEGADO),
    (255, b"ssh: connect to host 192.0.2.11 port 58291: Connection timed out\n", R.INALCANZABLE),
    (255, b"ssh: connect to host 192.0.2.11 port 58291: Connection refused\n", R.INALCANZABLE),
    (1, b"", R.INALCANZABLE),
])
def test_clasificar(rc, stderr, esperado):
    assert R.clasificar_intento(rc, stderr) == esperado


def test_orden_remotas_primero_y_local_al_final():
    orden = []

    async def revocar_en(h):
        orden.append(("revocar", h.nombre))
        return 0, b"revocar=ok quitadas=1 quedan=0\n", b""

    async def probar(h):
        orden.append(("probar", h.nombre))
        return 255, b"", b"Permission denied (publickey)."

    res = asyncio.run(R.revocar_todas(HOSTS, revocar_en=revocar_en, probar_entrada=probar))
    assert [r.estado for r in res] == [R.REVOCADA] * 3
    locales = [i for i, (_, n) in enumerate(orden) if n == "hall9000"]
    assert min(locales) > max(i for i, (_, n) in enumerate(orden) if n != "hall9000")
    assert [r.host for r in res] == ["atemai", "bridge", "hall9000"]


def test_inalcanzable_no_cuenta_como_revocada_y_sigue_entrando_se_dice():
    async def revocar_en(h):
        return 0, b"revocar=ok quitadas=1 quedan=0\n", b""

    async def probar(h):
        return {"atemai": (255, b"", b"Connection timed out"), "bridge": (0, b"", b""),
                "hall9000": (255, b"", b"Permission denied (publickey).")}[h.nombre]

    res = {r.host: r.estado for r in asyncio.run(R.revocar_todas(HOSTS, revocar_en=revocar_en, probar_entrada=probar))}
    assert res == {"atemai": R.NO_VERIFICABLE, "bridge": R.SIGUE_ENTRANDO, "hall9000": R.REVOCADA}


def test_error_al_revocar_no_se_da_por_revocada():
    async def revocar_en(h):
        return (2, b"revocar=error codigo=sin_archivo\n", b"") if h.nombre == "bridge" else (0, b"revocar=ok quitadas=1 quedan=0\n", b"")

    async def probar(h):
        return 255, b"", b"Permission denied (publickey)."

    res = {r.host: r for r in asyncio.run(R.revocar_todas(HOSTS, revocar_en=revocar_en, probar_entrada=probar))}
    assert res["bridge"].estado == R.ERROR_AL_REVOCAR
    assert res["bridge"].detalle == (("salida", "revocar=error codigo=sin_archivo"),)


def test_una_excepcion_al_revocar_es_error_no_silencio():
    async def revocar_en(h):
        if h.nombre == "atemai":
            raise OSError("ssh no está")
        return 0, b"revocar=ok quitadas=1 quedan=0\n", b""

    async def probar(h):
        return 255, b"", b"Permission denied (publickey)."

    res = {r.host: r.estado for r in asyncio.run(R.revocar_todas(HOSTS, revocar_en=revocar_en, probar_entrada=probar))}
    assert res["atemai"] == R.ERROR_AL_REVOCAR


def test_comandos():
    h = HOSTS[2]
    assert R.remoto_probar_entrada(h, "axioma") == (
        "LC_ALL=C ssh -o BatchMode=yes -o PreferredAuthentications=publickey -o ConnectTimeout=5 "
        "-o StrictHostKeyChecking=yes -p 58291 axioma@192.0.2.20 true")
    assert R.argv_admin(h, "admin", "sudo -n true") == [
        "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=yes",
        "-p", "58291", "admin@192.0.2.20", "sudo -n true"]
```

- [ ] **Step 2: Correr y ver que falla** — `ImportError`.

- [ ] **Step 3: Implementar**

```python
# jax/ejecutor/contratos/revocacion.py
"""Revocación de un golpe (C6). Spec 2026-09-15 §4: «un script quita la llave de
axioma en todas las máquinas del inventario; revocar y comprobar que no entra a ninguna».

- Primero las remotas, en paralelo; hall9000 al final (desde la cuenta se comprueban
  las remotas; revocada hall9000, ya no se entra a la cuenta).
- «No entra» es SÓLO `Permission denied (publickey)`. Una máquina que no contesta es
  NO_VERIFICABLE: no se sabe si se revocó, y se dice.
- Un fallo al revocar (salida, código o excepción) es ERROR_AL_REVOCAR, nunca silencio.
"""
from __future__ import annotations

import asyncio
import shlex
from dataclasses import dataclass

MARCAS_DE_ACCESO = ("ejecutor-axioma", "ejecutor-controlador")
MARCA_FRENO = "ejecutor-freno"

ENTRA, DENEGADO, INALCANZABLE = "entra", "denegado", "inalcanzable"
REVOCADA, SIGUE_ENTRANDO, NO_VERIFICABLE, ERROR_AL_REVOCAR = (
    "revocada", "sigue_entrando", "no_verificable", "error_al_revocar")


@dataclass(frozen=True)
class Resultado:
    host: str
    estado: str
    detalle: tuple = ()


def clasificar_intento(rc: int, stderr: bytes) -> str:
    if rc == 0:
        return ENTRA
    if rc == 255 and b"Permission denied (publickey" in stderr:
        return DENEGADO
    return INALCANZABLE


def remoto_probar_entrada(h, cuenta: str) -> str:
    return (f"LC_ALL=C ssh -o BatchMode=yes -o PreferredAuthentications=publickey -o ConnectTimeout=5 "
            f"-o StrictHostKeyChecking=yes -p {int(h.puerto)} {shlex.quote(cuenta)}@{shlex.quote(h.ip)} true")


def argv_admin(h, usuario_admin: str, remoto: str) -> list[str]:
    return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=yes",
            "-p", str(h.puerto), f"{usuario_admin}@{h.ip}", remoto]


async def _uno(h, revocar_en, probar_entrada) -> Resultado:
    try:
        rc, salida, _ = await revocar_en(h)
    except Exception as exc:  # fail-soft: se reporta ERROR_AL_REVOCAR para esta máquina y el resto sigue; nunca cuenta como revocada
        return Resultado(h.nombre, ERROR_AL_REVOCAR, (("tipo", type(exc).__name__),))
    texto = salida.decode(errors="replace").strip()
    if rc != 0 or not texto.startswith("revocar=ok") or not texto.endswith("quedan=0"):
        return Resultado(h.nombre, ERROR_AL_REVOCAR, (("salida", texto),))
    rc, _, errores = await probar_entrada(h)
    intento = clasificar_intento(rc, errores)
    estado = {DENEGADO: REVOCADA, ENTRA: SIGUE_ENTRANDO}.get(intento, NO_VERIFICABLE)
    return Resultado(h.nombre, estado, (("stderr", errores.decode(errors="replace").strip()[:300]),) if estado != REVOCADA else ())


async def revocar_todas(hosts, *, revocar_en, probar_entrada) -> tuple:
    remotas = [h for h in hosts if not h.es_local]
    locales = [h for h in hosts if h.es_local]
    primero = await asyncio.gather(*(_uno(h, revocar_en, probar_entrada) for h in remotas))
    despues = [await _uno(h, revocar_en, probar_entrada) for h in locales]
    return tuple(primero) + tuple(despues)
```

Nota para `test_una_excepcion_al_revocar_es_error_no_silencio`: la excepción es de la máquina `atemai`; `bridge` y
`hall9000` siguen y quedan `revocada`.

- [ ] **Step 4: Pasa** — 10 PASS; P10 verde.
- [ ] **Step 5: Mutación** — `{DENEGADO: REVOCADA, ENTRA: SIGUE_ENTRANDO}.get(intento, NO_VERIFICABLE)` → `.get(intento, REVOCADA)`
  → FAIL `test_inalcanzable_no_cuenta_como_revocada_y_sigue_entrando_se_dice`.
- [ ] **Step 6: Commit** `feat(ejecutor): orquestación de C6 — remotas primero, y sin comprobación no hay revocación`.

---

### Task 3: los scripts — el botón y la prueba no destructiva

**Files:**
- Create: `/home/fruiz/worktrees/jax-sp1-c6/scripts/ejecutor_contratos/revocar.py`
- Create: `/home/fruiz/worktrees/jax-sp1-c6/scripts/ejecutor_contratos/probar_c6.py`

- [ ] **Step 1: El botón**

```python
#!/usr/bin/env python3
# scripts/ejecutor_contratos/revocar.py
"""C6 — REVOCA el acceso del Ejecutor en TODAS las máquinas del inventario y corta sus sesiones.

Uso de emergencia:  set -a; . /etc/jax/.env; set +a
                    PYTHONPATH=.:las_manos python3 scripts/ejecutor_contratos/revocar.py --confirmo-revocar-todo
Sin la bandera no hace nada. Reponer el acceso es a mano, con los archivos .revocadas-* de cada máquina.
Lee /etc/jax/.env (producción) para JAX_EJECUTOR_* y la política exportada; no toca la DB.
"""
import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from jax.ejecutor.contratos import cuenta_axioma, formato, politica, revocacion


def comandos(env):
    c = cuenta_axioma.cuenta_desde_entorno(env)
    admin = env["JAX_EJECUTOR_ADMIN_USUARIO"]
    archivo = env["JAX_EJECUTOR_LLAVES_ROOT"]

    async def revocar_en(h):
        remoto = f"sudo -n /usr/local/sbin/ejecutor-revocar {archivo} {c.nombre}"
        if h.es_local:
            argv = ["sudo", "-n", "/usr/local/sbin/ejecutor-revocar", archivo, c.nombre]
        else:
            argv = revocacion.argv_admin(h, admin, remoto)
        p = await asyncio.create_subprocess_exec(*argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        salida, errores = await asyncio.wait_for(p.communicate(), 60)
        return p.returncode, salida, errores

    async def probar_entrada(h):
        if h.es_local:
            argv = cuenta_axioma.ssh_a_la_cuenta(c, "true")
            p = await asyncio.create_subprocess_exec(*argv, env={**os.environ, "LC_ALL": "C"},
                                                     stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            salida, errores = await asyncio.wait_for(p.communicate(), 30)
            return p.returncode, salida, errores
        return await cuenta_axioma.correr_en_la_cuenta(c, revocacion.remoto_probar_entrada(h, c.nombre), tope_s=30)

    return revocar_en, probar_entrada


async def principal(args, env) -> int:
    p = politica.validar(json.loads(Path(env["JAX_EJECUTOR_POLITICA"]).read_bytes()))
    hosts = [h for h in p.hosts if not args.solo or h.nombre == args.solo]
    revocar_en, probar_entrada = comandos(env)
    resultados = await revocacion.revocar_todas(hosts, revocar_en=revocar_en, probar_entrada=probar_entrada)
    for r in resultados:
        print(formato.campos((("host", r.host), ("estado", r.estado)) + tuple(r.detalle)))
    ok = all(r.estado == revocacion.REVOCADA for r in resultados)
    print(formato.campos((("c6_revocado_en_todas", ok),)))
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirmo-revocar-todo", action="store_true")
    ap.add_argument("--solo", help="una sola máquina (la usa probar_c6.py)")
    a = ap.parse_args()
    if not a.confirmo_revocar_todo:
        print(formato.campos((("codigo", "falta_confirmacion"),)))
        sys.exit(2)
    sys.exit(asyncio.run(principal(a, os.environ)))
```

**Guardia de subprocesos:** `revocar.py` lanza `ssh`/`sudo` y **no** menciona el arnés en literales: no es violación.
Verificar con `python3 policy/tests/test_claude_subprocess_solo_via_sandbox.py`.

- [ ] **Step 2: La prueba no destructiva por máquina**

```python
#!/usr/bin/env python3
# scripts/ejecutor_contratos/probar_c6.py
"""Prueba REAL y NO DESTRUCTIVA de C6 en UNA máquina: respaldar el archivo de llaves,
revocar, comprobar que no entra (Permission denied), REPONER, comprobar con cmp que
quedó idéntico, comprobar que vuelve a entrar. Si algo falla después de revocar, repone
igual (finally) y lo dice.

Uso: set -a; . /etc/jax/.env; set +a
     PYTHONPATH=.:las_manos python3 scripts/ejecutor_contratos/probar_c6.py <nombre-de-máquina>
Precondición que el script comprueba: ningún proceso de la cuenta vivo en esa máquina
(revocar mata sus sesiones).
"""
import asyncio
import json
import os
import sys
import time
from pathlib import Path

from jax.ejecutor.contratos import cuenta_axioma, formato, politica, revocacion
sys.path.insert(0, str(Path(__file__).resolve().parent))
from revocar import comandos  # noqa: E402


async def _admin(h, c, env, remoto_admin):
    if h.es_local:
        argv = ["sudo", "-n", "sh", "-c", remoto_admin]
    else:
        argv = revocacion.argv_admin(h, env["JAX_EJECUTOR_ADMIN_USUARIO"], f"sudo -n sh -c '{remoto_admin}'")
    p = await asyncio.create_subprocess_exec(*argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    salida, errores = await asyncio.wait_for(p.communicate(), 60)
    return p.returncode, salida.decode(errors="replace").strip(), errores.decode(errors="replace").strip()


async def principal(nombre: str, env) -> int:
    p = politica.validar(json.loads(Path(env["JAX_EJECUTOR_POLITICA"]).read_bytes()))
    (h,) = [x for x in p.hosts if x.nombre == nombre]
    c = cuenta_axioma.cuenta_desde_entorno(env)
    archivo = env["JAX_EJECUTOR_LLAVES_ROOT"]
    respaldo = f"{archivo}.prueba-c6-{int(time.time())}"
    revocar_en, probar_entrada = comandos(env)
    pasos = []

    rc, vivos, _ = await _admin(h, c, env, f"ps -u {c.nombre} -o pid= | wc -l")
    if rc != 0 or vivos != "0":
        print(formato.campos((("codigo", "cuenta_ocupada_o_admin_inaccesible"), ("rc", rc), ("vivos", vivos))))
        return 2
    rc, _, err = await _admin(h, c, env, f"cp -a {archivo} {respaldo}")
    if rc != 0:
        print(formato.campos((("codigo", "respaldo_imposible"), ("stderr", err))))
        return 2
    try:
        (resultado,) = await revocacion.revocar_todas([h], revocar_en=revocar_en, probar_entrada=probar_entrada)
        pasos.append(("revocacion", resultado.estado))
    finally:
        rc, _, err = await _admin(h, c, env, f"install -m 0644 -o root -g root {respaldo} {archivo} && cmp {respaldo} {archivo} && rm {respaldo}")
        pasos.append(("repuesta_identica", rc == 0))
    intento = revocacion.clasificar_intento(*(await probar_entrada(h))[::2])
    pasos.append(("vuelve_a_entrar", intento == revocacion.ENTRA))
    for clave, valor in pasos:
        print(formato.campos((("host", nombre), (clave, valor))))
    ok = pasos == [("revocacion", revocacion.REVOCADA), ("repuesta_identica", True), ("vuelve_a_entrar", True)]
    print(formato.campos((("c6_probado", ok), ("host", nombre))))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(principal(sys.argv[1], os.environ)))
```

Nota: `(await probar_entrada(h))[::2]` toma `(rc, stderr)` de `(rc, stdout, stderr)`.

- [ ] **Step 3: Guardias y commit**

Run: `cd /home/fruiz/worktrees/jax-sp1-c6 && python3 policy/tests/test_claude_subprocess_solo_via_sandbox.py && python3 policy/tests/test_no_fail_open_except.py`

```bash
git -C /home/fruiz/worktrees/jax-sp1-c6 add scripts/ejecutor_contratos/revocar.py scripts/ejecutor_contratos/probar_c6.py
git -C /home/fruiz/worktrees/jax-sp1-c6 commit -m "feat(ejecutor): botón de revocación y prueba no destructiva por máquina

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: CI y la prueba sobre hall9000

- [ ] **Step 1: Listas y piso** — `tests/test_ejecutor_revocar_sh.py` y `tests/test_ejecutor_contratos_revocacion.py` en
  las dos listas de `tests-puros`; piso con el número del runner; canario rojo por API con la mutación de la Task 2.
- [ ] **Step 2: Instalar en hall9000** — con `ops/ejecutor/instalar_en_maquina.sh hall9000` (Task 5; en la máquina local
  el script usa `sudo` directo en vez de ssh de administrador). `.env`: `JAX_EJECUTOR_ADMIN_USUARIO` (el usuario
  administrador de las máquinas; se lee, no se adivina) y `JAX_EJECUTOR_LLAVES_ROOT=/etc/ssh/authorized_keys.d/axioma`,
  con respaldo del `.env` y lectura doble (bash y systemd).
- [ ] **Step 3: Verlo funcionar y VERLO FALLAR en hall9000**
  1. `probar_c6.py hall9000` → `revocacion="revocada"`, `repuesta_identica=true`, `vuelve_a_entrar=true`.
  2. **Rotura:** con respaldo y `cmp`, quitar la marca ` ejecutor-controlador` del final de la línea del controlador en
     `/etc/ssh/authorized_keys.d/axioma` (así `ejecutor-revocar` no la reconoce) → `probar_c6.py hall9000` → Expected:
     `revocacion="sigue_entrando"`, exit 1. Restaurar y volver a correr → verde. Es la prueba de que la comprobación
     mira si entra de verdad y no sólo la salida del script.
- [ ] **Step 4: Commit** `ci(ejecutor): C6 en tests-puros; probado en hall9000 con su rotura`.

---

### Task 5 (Parte B): `instalar_en_maquina.sh`

**Files:**
- Create: `/home/fruiz/worktrees/jax-sp1-c6/ops/ejecutor/instalar_en_maquina.sh`

**Qué instala en cada máquina, y con qué reversión escrita antes:**

| Pieza | Reversión |
|---|---|
| `/usr/local/sbin/ejecutor-revocar`, `/usr/local/sbin/ejecutor-freno-remoto` (root 0755) | `rm` de los dos |
| `/etc/ssh/authorized_keys.d/axioma` (root 0644): líneas actuales de `~axioma/.ssh/authorized_keys` con la marca `ejecutor-axioma` (en hall9000 la del controlador con `ejecutor-controlador`) + la pública del freno con `command=…,restrict` y `ejecutor-freno` | `rm` del archivo |
| `/etc/ssh/sshd_config.d/50-ejecutor-axioma.conf`: `Match User axioma` / `AuthorizedKeysFile /etc/ssh/authorized_keys.d/%u` | `rm` + `sshd -t` + `systemctl reload ssh` |
| `/etc/sudoers.d/50-ejecutor-axioma-registro`: `Defaults:axioma log_output, iolog_dir=/var/log/sudo-io/%{user}, logfile=/var/log/sudo-axioma.log` (C3: log de `sudo` antes de que la cuenta tenga sudo) | `rm` |
| `axioma` en `/etc/cron.deny` y `/etc/at.deny`; `loginctl disable-linger axioma` | quitar la línea |
| Entrada de la máquina en `/etc/jax-ejecutor/freno/known_hosts` de hall9000 (copiada del `known_hosts` de `fruiz`, ya confiado; nunca `ssh-keyscan` a ciegas) | quitar la línea |

- [ ] **Step 1: El script**

```bash
#!/usr/bin/env bash
# ops/ejecutor/instalar_en_maquina.sh <nombre> — Parte B de C6 (y remoto de C3/C4) en UNA máquina del inventario.
# Corre como fruiz en hall9000. Remoto: ssh de administrador + sudo -n. Local (hall9000): sudo directo.
# Cada paso verifica; ante el primer fallo se detiene SIN seguir (set -e). La reversión está en la tabla del plan.
set -euo pipefail
NOMBRE="$1"
: "${JAX_EJECUTOR_POLITICA:?}" "${JAX_EJECUTOR_CUENTA:?}" "${JAX_EJECUTOR_ADMIN_USUARIO:?}" "${JAX_EJECUTOR_LLAVES_ROOT:?}"
: "${JAX_EJECUTOR_FRENO_LLAVE:?}" "${JAX_EJECUTOR_FRENO_KNOWN_HOSTS:?}"
REPO="$(git -C "$(dirname "$(readlink -f "$0")")" rev-parse --show-toplevel)"
read -r IP PUERTO LOCAL < <(python3 - "$JAX_EJECUTOR_POLITICA" "$NOMBRE" <<'PY'
import json, sys
(h,) = [h for h in json.load(open(sys.argv[1]))["hosts"] if h["nombre"] == sys.argv[2]]
print(h["ip"], h["puerto"], "si" if h["es_local"] else "no")
PY
)
if [ "$LOCAL" = si ]; then
  corre() { sudo -n sh -c "$1"; }
  sube() { sudo -n install -o root -g root -m "$3" "$1" "$2"; }
else
  ADMIN=(ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=yes -p "$PUERTO" "$JAX_EJECUTOR_ADMIN_USUARIO@$IP")
  corre() { "${ADMIN[@]}" "sudo -n sh -c $(printf %q "$1")"; }
  sube() { scp -q -P "$PUERTO" -o BatchMode=yes "$1" "$JAX_EJECUTOR_ADMIN_USUARIO@$IP:/tmp/ejecutor-subida" && corre "install -o root -g root -m $3 /tmp/ejecutor-subida $2 && rm /tmp/ejecutor-subida"; }
fi
C="$JAX_EJECUTOR_CUENTA"
ETAPA="$(mktemp -d)"; trap 'rm -rf "$ETAPA"' EXIT

# 0. Precondiciones: administrador con sudo sin contraseña, la cuenta existe, sin procesos vivos, foto de sshd -T.
corre "true"
corre "id $C >/dev/null"
test "$(corre "ps -u $C -o pid= | wc -l" | tr -d ' ')" = 0
corre "sshd -T -C user=root,host=x,addr=127.0.0.1" | sort > "$ETAPA/sshd-root-antes.txt"

# 1. Scripts root.
sube "$REPO/ops/ejecutor/ejecutor-revocar" /usr/local/sbin/ejecutor-revocar 0755
sube "$REPO/ops/ejecutor/ejecutor-freno-remoto" /usr/local/sbin/ejecutor-freno-remoto 0755

# 2. Archivo root de llaves: las actuales de la cuenta, marcadas; más la del freno.
corre "cat ~$C/.ssh/authorized_keys" > "$ETAPA/actuales"
MARCA=ejecutor-axioma
python3 - "$ETAPA/actuales" "$MARCA" "$LOCAL" "$(sudo cat "$JAX_EJECUTOR_FRENO_LLAVE.pub")" > "$ETAPA/llaves" <<'PY'
import sys
actuales, marca, local, freno = open(sys.argv[1]).read().splitlines(), sys.argv[2], sys.argv[3] == "si", sys.argv[4].split()
salida = []
for linea in actuales:
    partes = linea.split()
    if len(partes) < 2 or linea.lstrip().startswith("#"):
        continue
    tipo, clave = partes[0], partes[1]
    comentario = " ".join(partes[2:])
    m = "ejecutor-controlador" if local and "controlador" in comentario else marca
    salida.append(f"{tipo} {clave} {m}")
if not salida:
    sys.exit("sin_llaves_actuales")
salida.append(f'command="/usr/local/sbin/ejecutor-freno-remoto",restrict {freno[0]} {freno[1]} ejecutor-freno')
print("\n".join(salida))
PY
if [ "$LOCAL" = no ]; then corre "install -d -o root -g root -m 0755 $(dirname "$JAX_EJECUTOR_LLAVES_ROOT")"; else sudo install -d -o root -g root -m 0755 "$(dirname "$JAX_EJECUTOR_LLAVES_ROOT")"; fi
sube "$ETAPA/llaves" "$JAX_EJECUTOR_LLAVES_ROOT" 0644

# 3. sshd: sólo para la cuenta; validar; comprobar que para otro usuario no cambió NADA; recargar.
printf 'Match User %s\n    AuthorizedKeysFile %s\n' "$C" "$(dirname "$JAX_EJECUTOR_LLAVES_ROOT")/%u" > "$ETAPA/50-ejecutor.conf"
sube "$ETAPA/50-ejecutor.conf" /etc/ssh/sshd_config.d/50-ejecutor-axioma.conf 0644
corre "sshd -t"
corre "sshd -T -C user=root,host=x,addr=127.0.0.1" | sort > "$ETAPA/sshd-root-despues.txt"
diff "$ETAPA/sshd-root-antes.txt" "$ETAPA/sshd-root-despues.txt"
corre "sshd -T -C user=$C,host=x,addr=127.0.0.1" | grep -qx "authorizedkeysfile $(dirname "$JAX_EJECUTOR_LLAVES_ROOT")/%u"
corre "systemctl reload ssh"

# 4. Registro de sudo de la cuenta (C3), validado ANTES de instalar.
printf 'Defaults:%s log_output, iolog_dir=/var/log/sudo-io/%%{user}, logfile=/var/log/sudo-%s.log\n' "$C" "$C" > "$ETAPA/sudoers"
sube "$ETAPA/sudoers" /tmp/ejecutor-sudoers-prueba 0440
corre "visudo -cf /tmp/ejecutor-sudoers-prueba && install -o root -g root -m 0440 /tmp/ejecutor-sudoers-prueba /etc/sudoers.d/50-ejecutor-axioma-registro && rm /tmp/ejecutor-sudoers-prueba && visudo -c"

# 5. cron, at, linger.
corre "for f in /etc/cron.deny /etc/at.deny; do touch \$f; grep -qx $C \$f || echo $C >> \$f; done; test ! -e /etc/cron.allow; loginctl disable-linger $C"

# 6. known_hosts del freno (en hall9000), desde el known_hosts de fruiz ya confiado.
if [ "$LOCAL" = no ]; then
  ENTRADA="$(ssh-keygen -F "[$IP]:$PUERTO" -f ~/.ssh/known_hosts | grep -v '^#')"
  test -n "$ENTRADA"
  sudo grep -qxF "$ENTRADA" "$JAX_EJECUTOR_FRENO_KNOWN_HOSTS" || echo "$ENTRADA" | sudo tee -a "$JAX_EJECUTOR_FRENO_KNOWN_HOSTS" >/dev/null
fi
echo "maquina_instalada=\"$NOMBRE\""
```

- [ ] **Step 2: Verificaciones que el script no puede hacer solo (una sesión de administrador ABIERTA en otra terminal
  mientras se recarga `sshd`)**
  1. Una conexión **nueva** de administrador entra.
  2. La cuenta entra con su llave (desde hall9000, vía `correr_en_la_cuenta`: `LC_ALL=C ssh -o BatchMode=yes -p <puerto> axioma@<ip> true` → rc 0).
  3. La llave del freno **no** abre una shell: `sudo ssh -i $JAX_EJECUTOR_FRENO_LLAVE -o UserKnownHostsFile=$JAX_EJECUTOR_FRENO_KNOWN_HOSTS -p <puerto> axioma@<ip> 'id'`
     → imprime `freno_remoto=ok quedan=0` y **no** la salida de `id` (el comando forzado manda).
  4. **La cuenta ya no puede agregarse una llave:** como la cuenta, en la máquina: `echo 'ssh-ed25519 AAAAprueba prueba' >> ~/.ssh/authorized_keys`
     y verificar con `sshd -T -C user=axioma,…` que ese archivo no se consulta; borrar la línea agregada.

- [ ] **Step 3: Commit** `ops(ejecutor): instalación por máquina — llaves de root, sshd sólo para la cuenta, log de sudo, cron/at/linger`.

---

### Task 6 (Parte B): instalar y probar C6 en .11, .10 y .20 — NO DESTRUCTIVO, una por vez

> **GO:** ver «Dos partes, a propósito separadas». Orden: `atemai` (.11, desarrollo), `prod` (.10), `bridge` (.20,
> clientes). Nunca dos a la vez. Antes de cada una: `ps -u axioma` vacío en esa máquina (lo comprueba el script).

Para cada máquina `M` en ese orden:

- [ ] **Step 1:** `cd /home/fruiz/jax && set -a && . /etc/jax/.env && set +a && ops/ejecutor/instalar_en_maquina.sh M`
  → `maquina_instalada="M"`. Las cuatro verificaciones de la Task 5 Step 2. Si una falla: reversión de la tabla,
  **en orden inverso**, y se detiene la Parte B.
- [ ] **Step 2:** `PYTHONPATH=.:las_manos python3 scripts/ejecutor_contratos/probar_c6.py M` → Expected:
  `revocacion="revocada"`, `repuesta_identica=true`, `vuelve_a_entrar=true`, `c6_probado=true`.
- [ ] **Step 3: Comprobación independiente de que se repuso** (no la del script): `sha256sum /etc/ssh/authorized_keys.d/axioma`
  por ssh de administrador, comparado con el sha256 anotado en el Step 1; y la cuenta entra (Task 5 Step 2.2).
- [ ] **Step 4:** anotar en el registro de la sesión la salida literal de los Steps 1–3 de `M`.

Después de las tres:

- [ ] **Step 5: Revocación de un golpe, simulada sin revocar:** `revocar.py` **sin** la bandera → Expected:
  `codigo="falta_confirmacion"`, exit 2, y `sha256sum` de los archivos de llaves de las cuatro máquinas sin cambios.
  (La revocación real de todas a la vez no se ejecuta: quedaría el Ejecutor sin acceso; su mecanismo es el mismo que
  `probar_c6.py` ejercitó máquina por máquina.)

---

### Task 7 (Parte B): C4 remoto real — `ssh -tt` y barrido con la llave del freno

**Files:**
- Modify: `/home/fruiz/worktrees/jax-sp1-c6/scripts/ejecutor_contratos/probar_c4.py` (plan 3)

- [ ] **Step 1: `--remoto`** — agregar a `probar_c4.py`:

```python
import argparse
```

y cambiar `_escenario` y `principal`:

```python
def _escenario(c, nonce: str, remoto=None) -> str:
    destino_ip, destino_puerto = (remoto.ip, remoto.puerto) if remoto else ("127.0.0.1", c.puerto)
    return (f'setsid nohup sh -c \'sleep 5 && touch "$HOME/.c4-desprendido-{nonce}"\' >/dev/null 2>&1 </dev/null & '
            f'ssh -tt -o BatchMode=yes -p {destino_puerto} {c.nombre}@{destino_ip} '
            f'\'sleep 5 && touch "$HOME/.c4-anidado-{nonce}"\'')
```

`_marcas` lista también las del remoto: si `remoto`, un segundo `correr_en_la_cuenta` con
`ssh -o BatchMode=yes -p <puerto> axioma@<ip> 'ls -1 "$HOME"/.c4-*-<nonce> 2>/dev/null; rm -f "$HOME"/.c4-*-<nonce>'`
y se concatenan. `principal` lee `--remoto <nombre>` con `argparse`, resuelve la máquina en la política
(`politica.validar` sobre `JAX_EJECUTOR_POLITICA`) y la pasa a `ronda`. Además de `ps -u` local, en cada lectura con
freno se consulta por ssh de administrador `ps -u axioma -o pid= | wc -l` en la remota → tiene que dar `0`
(código `procesos_remotos_vivos` si no).

- [ ] **Step 2:** con la Mesa sin uso, `probar_c4.py --remoto atemai` → `c4_vivo=true` (control con las dos marcas;
  dos rondas; a +1 y +6 s nada vivo ni local ni en atemai; ninguna marca).
- [ ] **Step 3: VERLO FALLAR (remoto):** con respaldo y `cmp`, quitar de `/etc/jax-ejecutor/freno/known_hosts` la línea
  de atemai (el barrido no puede entrar) **y** reemplazar en el escenario `ssh -tt` por `ssh -T` (sin pty: no hay
  SIGHUP) en una copia local del script → Expected: `procesos_remotos_vivos a_los_s=1` y `marca_creada` con la
  anidada. Restaurar ambos y volver a correr → verde. Así se ve que las DOS capas remotas hacen falta y que la prueba
  las distingue.
- [ ] **Step 4: Commit y Biblioteca** — `ops(ejecutor): simulacro de C4 contra una máquina remota real` y en
  `CONTEXT.md` §9 las salidas literales de las Tasks 6 y 7, máquina por máquina, con la lección: **«quitar una llave
  no saca a quien ya entró; y una llave que la cuenta puede reescribir no es una llave que se pueda revocar»**.

---

## Auto-revisión (hecha al escribir)

- **Spec §4 C6:** un script quita la llave en todas las máquinas ✓ (`revocar.py` + `ejecutor-revocar`); revocar y
  comprobar que no entra a ninguna ✓ (con `Permission denied`, por máquina; la de todas a la vez se simula sin
  revocar, Task 6 Step 5). **Separado y no destructivo contra .10/.11/.20** ✓ (Parte B: respaldar, revocar,
  comprobar, reponer, `cmp`, comprobar).
- **C3 (log de sudo de `axioma` en cada servidor):** ✓ Task 5, antes de que la cuenta tenga sudo.
- **C4 remoto real:** ✓ Task 7.
- **Placeholders:** ninguno. `JAX_EJECUTOR_ADMIN_USUARIO` se lee del entorno a propósito: no se adivina el usuario
  administrador de servidores de clientes.
- **Tipos:** `Resultado(host, estado, detalle)`, `revocar_todas(hosts, *, revocar_en, probar_entrada)` iguales en
  código, tests y scripts.
