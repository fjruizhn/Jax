# Ejecutor SP1 · Plan 1 · C1 prohibiciones duras y C2 respaldo antes de destruir

> **Para agentes ejecutores:** SUB-SKILL REQUERIDA: usar `superpowers:subagent-driven-development`
> (recomendado) o `superpowers:executing-plans`. Los pasos usan casillas (`- [ ]`).

**Objetivo:** que ninguna llamada de herramienta del Ejecutor que coincida con una prohibición (C1), ni una
destructiva sin punto de restauración verificado de esa máquina (C2), llegue a correr — y que con la lista
ilegible no corra NADA.

**Arquitectura:** reglas en la DB (`ejecutor_regla`, con sus ejemplos) → un exportador (como `fruiz`) las
vuelca firmadas por sha256 a `/etc/jax-ejecutor/politica.json` → un gancho `PreToolUse` de Claude Code,
instalado root en `/opt/ejecutor/lib` y activado por un `managed-settings.json` montado **sólo dentro de la
jaula** (bwrap superpuesto), evalúa cada llamada como `axioma`. Todo fallo del gancho sale como exit 2.
El canario C1 corre el `claude` real contra un upstream Anthropic falso: sin modelo, sin GPU, determinista.

**Tech stack:** Python 3.12 (CI) / 3.14 (hall9000, `/usr/bin/python3` de `axioma`), sólo biblioteca estándar en
lo instalado; `h11` + `asyncio` para el upstream falso (ya usados por `proxy_carril.py`); MariaDB (migraciones
de jax-platform); `bwrap`; `sh`.

**Spec:** `docs/superpowers/specs/2026-09-15-ejecutor-design.md` §4 (C1, C2, canario, semilla, verdad incómoda),
§3.2. Índice y decisiones: `docs/superpowers/plans/2026-09-17-ejecutor-contratos-0-indice.md` (D-SP1-1).

## Global Constraints

- **Nunca cambiar de rama en `/home/fruiz/jax` ni en `/home/fruiz/jax-platform`.** Worktrees:
  jax `/home/fruiz/worktrees/jax-sp1-c1` rama `feat/ejecutor-c1-c2` desde `master`;
  jax-platform `/home/fruiz/worktrees/jax-platform-sp1-c1` rama `feat/ejecutor-tablas-c1-c2` desde `master`.
  Siempre `git -C <ruta absoluta>`; si un comando necesita cwd, `pwd && git branch --show-current` en el MISMO comando.
- **`/etc/jax/.env` apunta a PRODUCCIÓN.** Ningún test la carga. Los tests con DB corren contra
  `jax_memory_test` (conftest de jax-platform; job `jacobs-gobernanza-db` en jax). Los scripts de las Tasks 9 y
  10 la leen a propósito y lo dicen.
- **P10:** todo `except` amplio sin `raise`, o con cuerpo `pass`, lleva `# fail-soft: <razón específica>` en la
  MISMA línea. `python3 policy/tests/test_no_fail_open_except.py` verde en los dos repos.
- **Piso de `tests-puros`:** cada test nuevo se agrega en **las dos** listas del job en
  `.github/workflows/policy.yml` (la del `python -m pytest -v` y la del paso «Piso exacto de tests CORRIDOS»), y
  el `grep -qE "^N passed, 1 skipped"` se fija con **lo que cuenta el runner** en el run del PR, con su línea de
  historia `N -> M`. Lo mismo para el job `jacobs-gobernanza-db`.
- **Mutaciones:** `PYTHONDONTWRITEBYTECODE=1`; antes de mutar, `cp <archivo> <archivo>.mut-bak`; después de
  restaurar, `cmp <archivo> <archivo>.mut-bak && rm <archivo>.mut-bak`. Nunca `git checkout` para restaurar.
- **Tests con procesos:** `multiprocessing.get_context("fork")` explícito; comprobar vida con
  `ps -o pid= -p <pid>` (o `ps -u axioma -o pid=`), **nunca** `pgrep -f`.
- **Cero strings visibles hardcodeados:** los contratos emiten **códigos** en formato de máquina neutro
  (`clave=valor`, valores JSON ASCII) o `Motivo`; la frase la pone el frontend con i18n (SP2).
- **Sin hardcoding:** rutas, puertos y cuenta salen de `JAX_EJECUTOR_*` (tabla del índice); reglas, inventario y
  umbrales, de la DB. Sin defaults para lo que decide dónde se ejecuta algo.
- **Lo instalado en `/opt/ejecutor/lib` es sólo biblioteca estándar** (lo corre `axioma` con `python3 -I`).
  `tests/test_ejecutor_contratos_instalacion.py` lo impone.
- **Guardia `no-naked-claude-subprocess`:** un archivo que menciona `claude` en un literal (docstrings incluidos)
  y lanza un subproceso es violación. El ÚNICO módulo de este plan que lanza subprocesos hacia la cuenta es
  `jax/ejecutor/contratos/cuenta_axioma.py`, declarado en `_AISLADO_POR_CUENTA`. Los tests que lanzan
  subprocesos no escriben esa palabra en ningún literal.
- **Nada destructivo sobre .10/.11/.20.** Este plan no toca esas máquinas.
- **Checklist de cierre de cada task con tests:** pasan local → un job de CI los nombra (mirar el YAML) → al final
  del plan, romper a propósito y ver ROJO en el runner, por API sobre el sha, y después verde tras el revert.
- **Cuatro del rendimiento:** la única tabla que crece (`ejecutor_punto_restauracion`) lleva índice para su
  consulta y un test con `EXPLAIN`; el gancho corre en proceso propio, fuera de todo event loop (sin async); su
  latencia se mide en la Task 9 (p95 con 200 llamadas).
- Commits en castellano, terminados con `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`.

## Mapa de archivos

| Repo | Archivo | Acción | Responsabilidad |
|---|---|---|---|
| jax-platform | `backend/db/migrations.py` | Modificar | tablas `ejecutor_host`, `ejecutor_regla`, `ejecutor_punto_restauracion`; migraciones de datos `ejecutor_reglas_v1` e `ejecutor_inventario_v1` |
| jax-platform | `backend/db/semilla_ejecutor_reglas.json` | Crear | las 14 reglas semilla con sus ejemplos |
| jax-platform | `backend/tests/test_ejecutor_tablas.py` | Crear | tablas, semilla una sola vez, inventario desde el entorno, índice |
| jax | `jax/ejecutor/contratos/__init__.py` | Crear | paquete |
| jax | `jax/ejecutor/contratos/formato.py` | Crear | `clave=valor` neutro (stdlib) |
| jax | `jax/ejecutor/contratos/fallo.py` | Crear | `Fallo(contrato, codigo, datos)` para el arranque |
| jax | `jax/ejecutor/contratos/destinos.py` | Crear | a qué máquinas toca un comando (stdlib) |
| jax | `jax/ejecutor/contratos/politica.py` | Crear | cargar fail-closed, evaluar, autoprueba (stdlib) |
| jax | `jax/ejecutor/contratos/gancho.py` | Crear | entrada del gancho PreToolUse (stdlib) |
| jax | `ops/ejecutor/gancho.sh.plantilla` | Crear | envoltorio fail-closed |
| jax | `jax/ejecutor/contratos/instalacion.py` | Crear | `INSTALABLES`, render de `gancho.sh`, `managed-settings.json`, manifiesto |
| jax | `ops/ejecutor/instalar_contratos.sh` | Crear | instalación root en hall9000 |
| jax | `jax/ejecutor/contratos/exportar.py` | Crear | DB → `politica.json` firmado y atómico |
| jax | `jax/ejecutor/contratos/cuenta_axioma.py` | Crear | entrar a la cuenta; jaula superpuesta; lanzar `claude` |
| jax | `jax/ejecutor/contratos/canario_upstream.py` | Crear | upstream Anthropic falso con guion |
| jax | `jax/ejecutor/contratos/canario_c1.py` | Crear | `verificar_c1`: autoprueba + canario directo + canario por Claude Code real |
| jax | `policy/tests/test_claude_subprocess_solo_via_sandbox.py` | Modificar | declarar `cuenta_axioma.py` |
| jax | `tests/test_ejecutor_contratos_{destinos,politica,gancho,instalacion,exportar,cuenta_axioma,canario_upstream,canario_c1}.py` | Crear | tests puros |
| jax | `tests/test_ejecutor_politica_db.py` | Crear | la política de la DB real se exporta legible y pasa su autoprueba |
| jax | `.github/workflows/policy.yml` | Modificar | listas y pisos de `tests-puros` y `jacobs-gobernanza-db` |

## Interfaces (lo que producen estas tasks y consumen los planes 2–6)

```python
# jax/ejecutor/contratos/formato.py
def campos(pares) -> str                     # "clave=valor clave=valor", valores JSON ASCII

# jax/ejecutor/contratos/fallo.py
@dataclass(frozen=True)
class Fallo:
    contrato: str       # "c1".."c6"
    codigo: str
    datos: tuple = ()   # pares (clave, valor)

# jax/ejecutor/contratos/destinos.py
@dataclass(frozen=True)
class Host: nombre: str; ip: str; puerto: int; rol: str; es_local: bool
class ComandoIlegible(ValueError)
class HostDesconocido(ValueError)
def destinos(comando: str, hosts: tuple[Host, ...]) -> frozenset[str]

# jax/ejecutor/contratos/politica.py
VERSION = 1
PERMITIDO, PROHIBIDO, DESTRUCTIVO_SIN_RESPALDO, HOST_DESCONOCIDO, COMANDO_ILEGIBLE, ENTRADA_ILEGIBLE, POLITICA_ILEGIBLE
class PoliticaIlegible(ValueError): codigo: str; datos: tuple
@dataclass(frozen=True) class Regla: ...
@dataclass(frozen=True) class Politica: generada_at: str; hosts: tuple[Host, ...]; reglas: tuple[Regla, ...]; respaldos: dict; c2_edad_max_s: int
@dataclass(frozen=True) class Decision: permitir: bool; codigo: str; regla: str | None; hosts: tuple[str, ...]
@dataclass(frozen=True) class FalloDeEjemplo: regla: str; indice: int; esperado: str; obtenido: tuple
def contenido_canonico(doc: dict) -> bytes
def firmar(doc: dict) -> dict
def validar(doc: dict) -> Politica            # sin E/S: lo usa el exportador antes de publicar
def cargar(ruta, *, uid_de_la_cuenta: int) -> Politica
def evaluar(p: Politica, tool_name, tool_input, ahora: datetime) -> Decision
def autoprueba(p: Politica) -> tuple[FalloDeEjemplo, ...]

# jax/ejecutor/contratos/gancho.py
def principal(argv, *, entrada=None, salida=None, errores=None, ahora=None) -> int   # 0 permite, 2 bloquea

# jax/ejecutor/contratos/instalacion.py
INSTALABLES: tuple[str, ...]
def renderizar_gancho(lib: str, tope_s: int) -> str
def renderizar_managed_settings(lib: str, politica: str, tope_s: int) -> str
def manifiesto(archivos: dict[str, bytes]) -> str

# jax/ejecutor/contratos/exportar.py
class ExportacionImposible(RuntimeError): codigo: str
def documento(hosts, reglas, respaldos, edad_c2, generada_at: str) -> dict       # firmado y validado
async def leer(conn) -> tuple[list, list, list, str | None]
def escribir_atomico(ruta: Path, doc: dict) -> None
async def exportar(ruta: Path, conectar) -> dict

# jax/ejecutor/contratos/cuenta_axioma.py
@dataclass(frozen=True) class Cuenta: nombre: str; puerto: int; llave: Path; node_bin: Path; lib: Path; politica: Path
class CuentaSinConfigurar(RuntimeError)
def cuenta_desde_entorno(env=None) -> Cuenta
def ssh_a_la_cuenta(c: Cuenta, remoto: str) -> list[str]
def remoto_claude(c: Cuenta, *, base_url: str, modelo: str, prompt: str, herramientas: str = "Bash,Read") -> str
async def correr_en_la_cuenta(c: Cuenta, remoto: str, *, entrada: bytes = b"", tope_s: float) -> tuple[int, bytes, bytes]

# jax/ejecutor/contratos/canario_upstream.py
@dataclass(frozen=True) class Resultado: tool_use_id: str; es_error: bool; contenido: str
class UpstreamCanario:  # async context manager
    def __init__(self, guion: list[dict], host: str, puerto: int)
    resultados: dict[str, Resultado]; peticiones: list[tuple[str, str, str | None, bool]]
def guion_bash(tool_use_id: str, comando: str) -> dict

# jax/ejecutor/contratos/canario_c1.py
async def verificar_c1(c: Cuenta, *, puerto_canario: int, correr=..., upstream=..., nonce: str | None = None) -> tuple[Fallo, ...]
```

---

### Task 1: jax-platform · las tres tablas, la semilla de reglas y el inventario desde el entorno

**Files:**
- Modify: `/home/fruiz/worktrees/jax-platform-sp1-c1/backend/db/migrations.py` (DDL junto a `CREATE_USER_ADMIN_AUDIT`, registro en `_TABLES` después de `("user_admin_audit", ...)`, y dos llamadas en `run_migrations()` después de `await _ajustes_que_mandan_v1(cur)`)
- Create: `/home/fruiz/worktrees/jax-platform-sp1-c1/backend/db/semilla_ejecutor_reglas.json`
- Test: `/home/fruiz/worktrees/jax-platform-sp1-c1/backend/tests/test_ejecutor_tablas.py`

**Interfaces:**
- Consume: `axioma_migracion_de_datos`, `axioma_config` (existen).
- Produce: tablas `ejecutor_host(nombre PK, ip, puerto, rol, es_local, machine_id, con_datos_de_clientes, sudo, api_only, activo)`, `ejecutor_regla(id, codigo UNIQUE, tipo, herramientas, campo, patron, ambito_host, ambito_roles SET, es_canario, activa, origen, ejemplos_coincide JSON, ejemplos_no_coincide JSON)`, `ejecutor_punto_restauracion(id, host_nombre, referencia, metodo, restaurado_y_verificado_at UTC, verificado_por, evidencia)` con índice `idx_ejecutor_punto_host_fecha(host_nombre, restaurado_y_verificado_at)`; `axioma_config['ejecutor.c2_edad_max_s'] = '86400'` (INSERT IGNORE); funciones `_ejecutor_reglas_v1(cur)`, `_ejecutor_inventario_v1(cur)`, `parsear_inventario(texto) -> list[dict]`; constantes `MIGRACION_EJECUTOR_REGLAS_V1 = "ejecutor_reglas_v1"`, `MIGRACION_EJECUTOR_INVENTARIO_V1 = "ejecutor_inventario_v1"`.

**Por qué el inventario sale del entorno:** los dos repos son públicos (`gh repo view … visibility` = `PUBLIC`,
2026-09-17) y la ronda 9 sacó las IPs del código. `JAX_EJECUTOR_INVENTARIO` vive en `/etc/jax/.env` con un formato
sin espacios ni comillas (`nombre:ip:puerto:rol[:opcion+opcion]`), para que systemd y `bash` lo lean igual
(lección del frente A). Un valor mal formado **no** tumba la plataforma: se registra y no se marca la migración
(sin inventario no hay política exportable y el Ejecutor no arranca — cerrado).

- [ ] **Step 1: Escribir los tests que fallan**

```python
# backend/tests/test_ejecutor_tablas.py
"""Tablas del Ejecutor (SP1, plan 1): prohibiciones, inventario y puntos de
restauración. Contra jax_memory_test. La semilla de reglas corre UNA vez: una
regla que el admin desactiva después no revive en el próximo arranque."""
import json

import pytest

from db.migrations import (
    MIGRACION_EJECUTOR_INVENTARIO_V1, MIGRACION_EJECUTOR_REGLAS_V1, _ejecutor_inventario_v1,
    _ejecutor_reglas_v1, parsear_inventario,
)
from tests.identidades import sql

_SEMILLA_CODIGOS = {
    "canario_c1", "ssh_sin_tt", "apt_full_upgrade_bridge", "migrate_fresh_produccion", "sed_i_env",
    "pure_ftpd_parar_atemai", "respaldos_borrar", "ajustes_claude_code", "borrar_archivos",
    "sql_destructivo", "dns_correo", "parar_servicio", "quitar_paquetes", "disco",
}


async def _correr(funcion):
    from db.connection import get_pool
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await funcion(cur)
        await conn.commit()


def _vaciar(client):
    for nombre in (MIGRACION_EJECUTOR_REGLAS_V1, MIGRACION_EJECUTOR_INVENTARIO_V1):
        client.portal.call(sql, "DELETE FROM axioma_migracion_de_datos WHERE nombre = %s", (nombre,))
    client.portal.call(sql, "DELETE FROM ejecutor_punto_restauracion")
    client.portal.call(sql, "DELETE FROM ejecutor_regla")
    client.portal.call(sql, "DELETE FROM ejecutor_host")


@pytest.fixture
def sin_marcas(client):
    """jax_memory_test es compartida entre frentes: se deja como se encontró el
    esquema (vacío de filas del Ejecutor) y la semilla vuelve a correr en el
    próximo arranque de la suite."""
    _vaciar(client)
    yield
    _vaciar(client)


def test_la_semilla_trae_las_catorce_reglas_con_un_solo_canario(client, sin_marcas):
    client.portal.call(_correr, _ejecutor_reglas_v1)
    filas = client.portal.call(sql, "SELECT codigo, es_canario, ejemplos_coincide FROM ejecutor_regla", None, True)
    assert {f[0] for f in filas} == _SEMILLA_CODIGOS
    assert sum(1 for f in filas if f[1]) == 1
    assert all(json.loads(f[2]) for f in filas), "una regla sin ejemplo que coincida no se vio bloquear"


def test_la_semilla_corre_una_sola_vez(client, sin_marcas):
    client.portal.call(_correr, _ejecutor_reglas_v1)
    client.portal.call(sql, "UPDATE ejecutor_regla SET activa = 0 WHERE codigo = 'sed_i_env'")
    client.portal.call(_correr, _ejecutor_reglas_v1)
    assert client.portal.call(sql, "SELECT activa FROM ejecutor_regla WHERE codigo = 'sed_i_env'", None, True) == ((0,),)


def test_la_edad_maxima_de_c2_se_siembra_sin_pisar(client, sin_marcas):
    client.portal.call(sql, "DELETE FROM axioma_config WHERE config_key = 'ejecutor.c2_edad_max_s'")
    client.portal.call(_correr, _ejecutor_reglas_v1)
    assert client.portal.call(sql, "SELECT config_value FROM axioma_config WHERE config_key = 'ejecutor.c2_edad_max_s'",
                              None, True) == (("86400",),)


def test_inventario_desde_el_entorno(client, sin_marcas, monkeypatch):
    monkeypatch.setenv("JAX_EJECUTOR_INVENTARIO",
                       "hall9000:192.0.2.5:58291:hypervisor:local+sin_clientes,bridge:192.0.2.20:58291:clientes")
    client.portal.call(_correr, _ejecutor_inventario_v1)
    filas = client.portal.call(sql, "SELECT nombre, ip, puerto, rol, es_local, con_datos_de_clientes FROM ejecutor_host "
                                    "ORDER BY nombre", None, True)
    assert filas == (("bridge", "192.0.2.20", 58291, "clientes", 0, 1),
                     ("hall9000", "192.0.2.5", 58291, "hypervisor", 1, 0))


def test_inventario_ausente_no_marca_la_migracion(client, sin_marcas, monkeypatch):
    monkeypatch.delenv("JAX_EJECUTOR_INVENTARIO", raising=False)
    client.portal.call(_correr, _ejecutor_inventario_v1)
    assert client.portal.call(sql, "SELECT COUNT(*) FROM axioma_migracion_de_datos WHERE nombre = %s",
                              (MIGRACION_EJECUTOR_INVENTARIO_V1,), True) == ((0,),)


def test_inventario_mal_formado_no_tumba_ni_marca(client, sin_marcas, monkeypatch):
    monkeypatch.setenv("JAX_EJECUTOR_INVENTARIO", "hall9000:192.0.2.5:no-es-puerto:hypervisor")
    client.portal.call(_correr, _ejecutor_inventario_v1)
    assert client.portal.call(sql, "SELECT COUNT(*) FROM ejecutor_host", None, True) == ((0,),)


@pytest.mark.parametrize("texto", [
    "", "a:1.2.3.4:22", "a:1.2.3.4:22:rol_raro", "a:1.2.3.4:0:clientes", "a:1.2.3.4:22:clientes:opcion_rara",
    "a:1.2.3.4:22:clientes,a:1.2.3.5:22:clientes",
])
def test_parsear_inventario_rechaza(texto):
    with pytest.raises(ValueError):
        parsear_inventario(texto)


def test_la_consulta_del_exportador_usa_el_indice(client, sin_marcas, monkeypatch):
    monkeypatch.setenv("JAX_EJECUTOR_INVENTARIO", "a:192.0.2.1:22:clientes,b:192.0.2.2:22:clientes")
    client.portal.call(_correr, _ejecutor_inventario_v1)
    for k in range(200):
        client.portal.call(sql, "INSERT INTO ejecutor_punto_restauracion (host_nombre, referencia, metodo, "
                                "restaurado_y_verificado_at, verificado_por, evidencia) VALUES (%s, %s, 'prueba', "
                                "UTC_TIMESTAMP(), 'test', 'test')", ("ab"[k % 2], f"prueba-{k}"))
    filas = client.portal.call(sql, "EXPLAIN SELECT host_nombre, MAX(restaurado_y_verificado_at) "
                                    "FROM ejecutor_punto_restauracion GROUP BY host_nombre", None, True)
    assert any("idx_ejecutor_punto_host_fecha" in str(f) for f in filas), filas
```

- [ ] **Step 2: Correr y ver que fallan**

Run: `cd /home/fruiz/worktrees/jax-platform-sp1-c1/backend && pwd && git branch --show-current && python -m pytest tests/test_ejecutor_tablas.py -v`
Expected: FAIL en la colección con `ImportError: cannot import name 'MIGRACION_EJECUTOR_INVENTARIO_V1'`.

- [ ] **Step 3: Escribir la semilla**

`backend/db/semilla_ejecutor_reglas.json` (JSON válido; los `\\` son backslashes de regex escapados para JSON):

```json
[
  {"codigo": "canario_c1", "tipo": "prohibido", "herramientas": "Bash", "campo": "command",
   "patron": "ejecutor-canario-c1", "ambito_host": null, "ambito_roles": [], "es_canario": true,
   "origen": "spec 2026-09-15 §4: canario permanente de C1",
   "ejemplos_coincide": [{"tool_name": "Bash", "tool_input": {"command": "echo ejecutor-canario-c1"}}],
   "ejemplos_no_coincide": [{"tool_name": "Bash", "tool_input": {"command": "echo ejecutor-canario"}}]},
  {"codigo": "ssh_sin_tt", "tipo": "prohibido", "herramientas": "Bash", "campo": "command",
   "patron": "(?:^|[;&|(]|\\bsudo|\\bnohup|\\bexec|\\benv|\\btimeout\\s+\\S+)\\s*(?:\\S*/)?ssh(?=\\s)(?![^;&|\\n]*\\s-tt(?=\\s|$))",
   "ambito_host": null, "ambito_roles": [], "es_canario": false,
   "origen": "C4: sin pty, matar el cliente ssh no manda SIGHUP al remoto (LAS MANOS, 2026-06-14)",
   "ejemplos_coincide": [
     {"tool_name": "Bash", "tool_input": {"command": "ssh -p 58291 axioma@atemai uptime"}},
     {"tool_name": "Bash", "tool_input": {"command": "uptime; ssh axioma@prod df -h"}},
     {"tool_name": "Bash", "tool_input": {"command": "sudo ssh axioma@bridge uptime"}}],
   "ejemplos_no_coincide": [
     {"tool_name": "Bash", "tool_input": {"command": "ssh -tt -p 58291 axioma@atemai uptime"}},
     {"tool_name": "Bash", "tool_input": {"command": "grep ssh /var/log/auth.log"}},
     {"tool_name": "Bash", "tool_input": {"command": "systemctl status ssh"}},
     {"tool_name": "Bash", "tool_input": {"command": "ls /etc/ssh -la"}}]},
  {"codigo": "apt_full_upgrade_bridge", "tipo": "prohibido", "herramientas": "Bash", "campo": "command",
   "patron": "\\bapt(?:-get)?\\s+(?:-\\S+\\s+)*(?:full-upgrade|dist-upgrade)\\b",
   "ambito_host": "bridge", "ambito_roles": [], "es_canario": false,
   "origen": "spec 2026-09-15 §4 semilla: apt full-upgrade en .20",
   "ejemplos_coincide": [
     {"tool_name": "Bash", "tool_input": {"command": "ssh -tt axioma@bridge 'sudo apt full-upgrade -y'"}},
     {"tool_name": "Bash", "tool_input": {"command": "ssh -tt axioma@bridge 'sudo apt-get -y dist-upgrade'"}}],
   "ejemplos_no_coincide": [
     {"tool_name": "Bash", "tool_input": {"command": "ssh -tt axioma@bridge 'apt list --upgradable'"}},
     {"tool_name": "Bash", "tool_input": {"command": "ssh -tt axioma@atemai 'sudo apt full-upgrade -y'"}}]},
  {"codigo": "migrate_fresh_produccion", "tipo": "prohibido", "herramientas": "Bash", "campo": "command",
   "patron": "\\bmigrate:fresh\\b", "ambito_host": null, "ambito_roles": ["produccion", "clientes"], "es_canario": false,
   "origen": "spec 2026-09-15 §4 semilla; Principio III: nunca migrate:fresh en producción",
   "ejemplos_coincide": [
     {"tool_name": "Bash", "tool_input": {"command": "ssh -tt axioma@prod 'cd /www/app && php artisan migrate:fresh --force'"}},
     {"tool_name": "Bash", "tool_input": {"command": "ssh -tt axioma@bridge 'php artisan migrate:fresh'"}}],
   "ejemplos_no_coincide": [
     {"tool_name": "Bash", "tool_input": {"command": "ssh -tt axioma@atemai 'php artisan migrate:fresh'"}},
     {"tool_name": "Bash", "tool_input": {"command": "ssh -tt axioma@prod 'php artisan migrate:status'"}}]},
  {"codigo": "sed_i_env", "tipo": "prohibido", "herramientas": "Bash", "campo": "command",
   "patron": "\\bsed\\b(?:'[^']*'|\"[^\"]*\"|[^;&|\\n'\"])*?\\s(?:-[a-zA-Z]*i\\S*|--in-place\\S*)(?:'[^']*'|\"[^\"]*\"|[^;&|\\n'\"])*?\\.env\\b",
   "ambito_host": null, "ambito_roles": [], "es_canario": false,
   "origen": "spec 2026-09-15 §4 semilla: sed -i sobre un .env",
   "ejemplos_coincide": [
     {"tool_name": "Bash", "tool_input": {"command": "sed -i 's/^DEBUG=.*/DEBUG=false/' /www/app/.env"}},
     {"tool_name": "Bash", "tool_input": {"command": "sed -i 's/a/b/;s/c/d/' .env"}},
     {"tool_name": "Bash", "tool_input": {"command": "sed --in-place 's/a/b/' .env"}}],
   "ejemplos_no_coincide": [
     {"tool_name": "Bash", "tool_input": {"command": "sed -n '1,5p' /www/app/.env"}},
     {"tool_name": "Bash", "tool_input": {"command": "sed -i 's/a/b/' /etc/nginx/nginx.conf"}},
     {"tool_name": "Bash", "tool_input": {"command": "sed -i 's/a/b/' /www/app/.environment"}}]},
  {"codigo": "pure_ftpd_parar_atemai", "tipo": "prohibido", "herramientas": "Bash", "campo": "command",
   "patron": "\\b(?:systemctl\\s+(?:-\\S+\\s+)*(?:stop|disable|mask|kill)\\s+(?:\\S+\\s+)*pure-ftpd|service\\s+pure-ftpd\\s+stop)\\b",
   "ambito_host": "atemai", "ambito_roles": [], "es_canario": false,
   "origen": "spec 2026-09-15 §4 semilla: parar, deshabilitar o enmascarar pure-ftpd en .11",
   "ejemplos_coincide": [
     {"tool_name": "Bash", "tool_input": {"command": "ssh -tt axioma@atemai 'sudo systemctl stop pure-ftpd'"}},
     {"tool_name": "Bash", "tool_input": {"command": "ssh -tt axioma@atemai 'sudo systemctl disable --now pure-ftpd.service'"}},
     {"tool_name": "Bash", "tool_input": {"command": "ssh -tt axioma@atemai 'sudo service pure-ftpd stop'"}}],
   "ejemplos_no_coincide": [
     {"tool_name": "Bash", "tool_input": {"command": "ssh -tt axioma@atemai 'systemctl status pure-ftpd'"}},
     {"tool_name": "Bash", "tool_input": {"command": "ssh -tt axioma@prod 'sudo systemctl stop pure-ftpd'"}}]},
  {"codigo": "respaldos_borrar", "tipo": "prohibido", "herramientas": "Bash", "campo": "command",
   "patron": "\\brestic\\b[^\\n]*\\s(?:forget|prune)\\b|\\bzfs\\s+destroy\\b|\\bmidclt\\s+call\\s+(?:pool\\.snapshottask|replication|pool\\.dataset)\\.(?:update|delete)\\b|\\brm\\s+[^\\n]*/srv/backup-adata",
   "ambito_host": null, "ambito_roles": [], "es_canario": false,
   "origen": "spec 2026-09-15 §4 y §7: borrar respaldos o acortar retenciones",
   "ejemplos_coincide": [
     {"tool_name": "Bash", "tool_input": {"command": "restic -r /srv/backup-adata/restic forget --keep-last 1 --prune"}},
     {"tool_name": "Bash", "tool_input": {"command": "sudo zfs destroy tank/backups@auto-2026-09-01"}},
     {"tool_name": "Bash", "tool_input": {"command": "rm -rf /srv/backup-adata/staging"}},
     {"tool_name": "Bash", "tool_input": {"command": "midclt call pool.snapshottask.update 3 '{\"lifetime_value\": 1}'"}}],
   "ejemplos_no_coincide": [
     {"tool_name": "Bash", "tool_input": {"command": "restic -r /srv/backup-adata/restic snapshots"}},
     {"tool_name": "Bash", "tool_input": {"command": "zfs list -t snapshot"}},
     {"tool_name": "Bash", "tool_input": {"command": "ls -la /srv/backup-adata"}}]},
  {"codigo": "ajustes_claude_code", "tipo": "prohibido", "herramientas": "Bash|Write|Edit|MultiEdit|NotebookEdit",
   "campo": "cualquiera", "patron": "\\.claude/settings(?:\\.local)?\\.json|managed-settings\\.json|disableAllHooks|\\.claude\\.json",
   "ambito_host": null, "ambito_roles": [], "es_canario": false,
   "origen": "C1: disableAllHooks en cualquier nivel apaga los ganchos (documentación oficial, 2026-09-17)",
   "ejemplos_coincide": [
     {"tool_name": "Write", "tool_input": {"file_path": "/home/axioma/.claude/settings.json", "content": "{}"}},
     {"tool_name": "Bash", "tool_input": {"command": "echo '{\"disableAllHooks\": true}' > .claude/settings.local.json"}}],
   "ejemplos_no_coincide": [
     {"tool_name": "Read", "tool_input": {"file_path": "/home/axioma/.claude/settings.json"}},
     {"tool_name": "Write", "tool_input": {"file_path": "/home/axioma/notas.md", "content": "x"}}]},
  {"codigo": "borrar_archivos", "tipo": "destructivo", "herramientas": "Bash", "campo": "command",
   "patron": "\\b(?:rm|unlink|shred)\\s|\\bfind\\b[^\\n]*\\s-delete\\b|\\btruncate\\s+(?:-\\S+\\s+)*-s\\b",
   "ambito_host": null, "ambito_roles": [], "es_canario": false, "origen": "spec 2026-09-15 §4 C2: borrar datos",
   "ejemplos_coincide": [
     {"tool_name": "Bash", "tool_input": {"command": "rm -rf /tmp/build"}},
     {"tool_name": "Bash", "tool_input": {"command": "ssh -tt axioma@prod 'rm /www/app/storage/logs/laravel.log'"}},
     {"tool_name": "Bash", "tool_input": {"command": "find /var/log/app -name '*.gz' -delete"}},
     {"tool_name": "Bash", "tool_input": {"command": "truncate -s 0 /var/log/syslog"}}],
   "ejemplos_no_coincide": [
     {"tool_name": "Bash", "tool_input": {"command": "ls -la /tmp"}},
     {"tool_name": "Bash", "tool_input": {"command": "echo confirm yes"}}]},
  {"codigo": "sql_destructivo", "tipo": "destructivo", "herramientas": "Bash", "campo": "command",
   "patron": "(?i)\\b(?:drop\\s+(?:table|database|schema|user)|truncate\\s+table|delete\\s+from)\\b",
   "ambito_host": null, "ambito_roles": [], "es_canario": false, "origen": "spec 2026-09-15 §4 C2: DROP",
   "ejemplos_coincide": [
     {"tool_name": "Bash", "tool_input": {"command": "mysql -e 'DROP TABLE clientes'"}},
     {"tool_name": "Bash", "tool_input": {"command": "ssh -tt axioma@bridge \"mysql app -e 'delete from users where id=3'\""}}],
   "ejemplos_no_coincide": [
     {"tool_name": "Bash", "tool_input": {"command": "mysql -e 'SELECT COUNT(*) FROM users'"}},
     {"tool_name": "Bash", "tool_input": {"command": "mysql -e 'SHOW TABLES'"}}]},
  {"codigo": "dns_correo", "tipo": "destructivo", "herramientas": "Bash", "campo": "command",
   "patron": "\\bnsupdate\\b|\\bpdnsutil\\s+(?:delete-rrset|replace-rrset|add-record|delete-zone|edit-zone)\\b|\\brndc\\s+(?:freeze|thaw|reload|delzone|modzone)\\b|dns_records[^\\n]*-X\\s*(?:DELETE|PUT|PATCH|POST)|-X\\s*(?:DELETE|PUT|PATCH|POST)[^\\n]*dns_records|\\bpostconf\\s+-e\\b|\\bpostsuper\\s+-d\\b|\\bexim\\s+-Mrm\\b",
   "ambito_host": null, "ambito_roles": [], "es_canario": false, "origen": "spec 2026-09-15 §4 C2: DNS/MX",
   "ejemplos_coincide": [
     {"tool_name": "Bash", "tool_input": {"command": "nsupdate -k /etc/bind/k.key cambios.txt"}},
     {"tool_name": "Bash", "tool_input": {"command": "curl -X DELETE https://api.cloudflare.com/client/v4/zones/Z/dns_records/R"}},
     {"tool_name": "Bash", "tool_input": {"command": "ssh -tt axioma@atemai 'sudo postconf -e relayhost=smtp:587'"}}],
   "ejemplos_no_coincide": [
     {"tool_name": "Bash", "tool_input": {"command": "dig MX axioma-ia.io"}},
     {"tool_name": "Bash", "tool_input": {"command": "curl https://api.cloudflare.com/client/v4/zones/Z/dns_records"}}]},
  {"codigo": "parar_servicio", "tipo": "destructivo", "herramientas": "Bash", "campo": "command",
   "patron": "\\bsystemctl\\s+(?:-\\S+\\s+)*(?:stop|disable|mask|kill|reboot|poweroff|halt)\\b|\\bservice\\s+\\S+\\s+stop\\b|(?:^|[;&|(]|\\bsudo)\\s*(?:shutdown|poweroff|halt|reboot)\\b",
   "ambito_host": null, "ambito_roles": [], "es_canario": false, "origen": "spec 2026-09-15 §4 C2: parar servicio",
   "ejemplos_coincide": [
     {"tool_name": "Bash", "tool_input": {"command": "sudo systemctl stop nginx"}},
     {"tool_name": "Bash", "tool_input": {"command": "ssh -tt axioma@prod 'sudo systemctl disable --now php8.3-fpm'"}},
     {"tool_name": "Bash", "tool_input": {"command": "sudo reboot"}}],
   "ejemplos_no_coincide": [
     {"tool_name": "Bash", "tool_input": {"command": "systemctl status nginx"}},
     {"tool_name": "Bash", "tool_input": {"command": "grep -i shutdown /var/log/syslog"}}]},
  {"codigo": "quitar_paquetes", "tipo": "destructivo", "herramientas": "Bash", "campo": "command",
   "patron": "\\b(?:apt|apt-get|aptitude)\\s+(?:-\\S+\\s+)*(?:remove|purge|autoremove|full-upgrade|dist-upgrade)\\b|\\bdpkg\\s+(?:-\\S+\\s+)*(?:-r|-P|--remove|--purge)\\b|\\bsnap\\s+remove\\b|\\bpip3?\\s+uninstall\\b",
   "ambito_host": null, "ambito_roles": [], "es_canario": false, "origen": "spec 2026-09-15 §4 C2: quitar paquetes",
   "ejemplos_coincide": [
     {"tool_name": "Bash", "tool_input": {"command": "sudo apt-get -y purge exim4"}},
     {"tool_name": "Bash", "tool_input": {"command": "ssh -tt axioma@bridge 'sudo apt autoremove'"}},
     {"tool_name": "Bash", "tool_input": {"command": "sudo dpkg -P paquete"}}],
   "ejemplos_no_coincide": [
     {"tool_name": "Bash", "tool_input": {"command": "apt list --installed"}},
     {"tool_name": "Bash", "tool_input": {"command": "sudo apt install htop"}}]},
  {"codigo": "disco", "tipo": "destructivo", "herramientas": "Bash", "campo": "command",
   "patron": "\\bmkfs(?:\\.\\w+)?\\b|\\bdd\\s+[^\\n]*\\bof=/dev/|\\bwipefs\\b|\\b(?:lvremove|vgremove|pvremove)\\b|\\bvirsh\\s+(?:undefine|destroy|vol-delete)\\b",
   "ambito_host": null, "ambito_roles": [], "es_canario": false, "origen": "spec 2026-09-15 §4 C2: borrar datos (discos y VMs)",
   "ejemplos_coincide": [
     {"tool_name": "Bash", "tool_input": {"command": "sudo mkfs.ext4 /dev/sdb1"}},
     {"tool_name": "Bash", "tool_input": {"command": "sudo dd if=imagen.img of=/dev/sdb bs=4M"}},
     {"tool_name": "Bash", "tool_input": {"command": "sudo lvremove -f vg_vms/dev_root_snap"}},
     {"tool_name": "Bash", "tool_input": {"command": "virsh destroy dev"}}],
   "ejemplos_no_coincide": [
     {"tool_name": "Bash", "tool_input": {"command": "lsblk"}},
     {"tool_name": "Bash", "tool_input": {"command": "sudo dd if=/dev/vg_vms/prod_root_snap bs=4M status=none"}},
     {"tool_name": "Bash", "tool_input": {"command": "virsh list --all"}}]}
]
```

Los ejemplos usan **nombres** del inventario (`axioma@bridge`), nunca IPs: `destinos.py` resuelve por nombre o
por IP. Que todos den lo que dicen lo prueba jax contra esta misma semilla (Task 8), no esta task.

- [ ] **Step 4: Escribir la migración**

En `backend/db/migrations.py`, junto a `CREATE_USER_ADMIN_AUDIT`:

```python
# Ejecutor SP1 (plan 1, 2026-09-17). Catálogos que no crecen (inventario y reglas:
# decenas de filas) y una tabla que sí (puntos de restauración, con índice para la
# única consulta que la lee: el último verificado por máquina).
CREATE_EJECUTOR_HOST = """
CREATE TABLE IF NOT EXISTS ejecutor_host (
  nombre VARCHAR(50) NOT NULL PRIMARY KEY,
  ip VARCHAR(45) NOT NULL,
  puerto INT NOT NULL,
  rol ENUM('hypervisor','desarrollo','produccion','clientes','respaldo') NOT NULL,
  es_local BOOLEAN NOT NULL DEFAULT FALSE,
  machine_id CHAR(32) NULL,
  con_datos_de_clientes BOOLEAN NOT NULL DEFAULT TRUE,
  sudo BOOLEAN NOT NULL DEFAULT FALSE,
  api_only BOOLEAN NOT NULL DEFAULT FALSE,
  activo BOOLEAN NOT NULL DEFAULT TRUE,
  created_at DATETIME DEFAULT NOW(),
  UNIQUE KEY uk_ejecutor_host_ip_puerto (ip, puerto)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""

CREATE_EJECUTOR_REGLA = """
CREATE TABLE IF NOT EXISTS ejecutor_regla (
  id INT AUTO_INCREMENT PRIMARY KEY,
  codigo VARCHAR(80) NOT NULL,
  tipo ENUM('prohibido','destructivo') NOT NULL,
  herramientas VARCHAR(200) NOT NULL,
  campo ENUM('command','file_path','cualquiera') NOT NULL,
  patron VARCHAR(1000) NOT NULL,
  ambito_host VARCHAR(50) NULL,
  ambito_roles SET('hypervisor','desarrollo','produccion','clientes','respaldo') NULL,
  es_canario BOOLEAN NOT NULL DEFAULT FALSE,
  activa BOOLEAN NOT NULL DEFAULT TRUE,
  origen VARCHAR(300) NOT NULL,
  ejemplos_coincide JSON NOT NULL,
  ejemplos_no_coincide JSON NOT NULL,
  created_at DATETIME DEFAULT NOW(),
  UNIQUE KEY uk_ejecutor_regla_codigo (codigo)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""

CREATE_EJECUTOR_PUNTO_RESTAURACION = """
CREATE TABLE IF NOT EXISTS ejecutor_punto_restauracion (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  host_nombre VARCHAR(50) NOT NULL,
  referencia VARCHAR(255) NOT NULL,
  metodo VARCHAR(50) NOT NULL,
  restaurado_y_verificado_at DATETIME NOT NULL COMMENT 'UTC: momento en que se RESTAURÓ y verificó, no en que se respaldó',
  verificado_por VARCHAR(100) NOT NULL,
  evidencia VARCHAR(500) NOT NULL,
  created_at DATETIME DEFAULT NOW(),
  INDEX idx_ejecutor_punto_host_fecha (host_nombre, restaurado_y_verificado_at),
  FOREIGN KEY (host_nombre) REFERENCES ejecutor_host(nombre)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""
```

En `_TABLES`, después de `("user_admin_audit", CREATE_USER_ADMIN_AUDIT),`:

```python
    ("ejecutor_host", CREATE_EJECUTOR_HOST),                            # antes de punto_restauracion (FK)
    ("ejecutor_regla", CREATE_EJECUTOR_REGLA),
    ("ejecutor_punto_restauracion", CREATE_EJECUTOR_PUNTO_RESTAURACION),
```

Después de `_ajustes_que_mandan_v1` (arriba del archivo, `import json`, `import logging`, `import os` y
`from pathlib import Path` si no están):

```python
MIGRACION_EJECUTOR_REGLAS_V1 = "ejecutor_reglas_v1"
MIGRACION_EJECUTOR_INVENTARIO_V1 = "ejecutor_inventario_v1"
_SEMILLA_EJECUTOR_REGLAS = Path(__file__).with_name("semilla_ejecutor_reglas.json")
_ROLES_EJECUTOR = ("hypervisor", "desarrollo", "produccion", "clientes", "respaldo")
_OPCIONES_INVENTARIO = frozenset({"local", "sin_clientes"})
_log_migraciones = logging.getLogger("migrations")


async def _marcada(cur, nombre: str) -> bool:
    await cur.execute("SELECT 1 FROM axioma_migracion_de_datos WHERE nombre = %s", (nombre,))
    return await cur.fetchone() is not None


async def _ejecutor_reglas_v1(cur) -> None:
    """Siembra las reglas del Ejecutor UNA vez (una desactivada por el admin no
    revive) y la edad máxima de un punto de restauración para C2 (sin pisar)."""
    await cur.execute(
        "INSERT IGNORE INTO axioma_config (config_key, config_value) VALUES ('ejecutor.c2_edad_max_s', '86400')")
    if await _marcada(cur, MIGRACION_EJECUTOR_REGLAS_V1):
        return
    for r in json.loads(_SEMILLA_EJECUTOR_REGLAS.read_text(encoding="utf-8")):
        await cur.execute(
            "INSERT IGNORE INTO ejecutor_regla (codigo, tipo, herramientas, campo, patron, ambito_host, "
            "ambito_roles, es_canario, origen, ejemplos_coincide, ejemplos_no_coincide) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (r["codigo"], r["tipo"], r["herramientas"], r["campo"], r["patron"], r["ambito_host"],
             ",".join(r["ambito_roles"]) or None, r["es_canario"], r["origen"],
             json.dumps(r["ejemplos_coincide"], ensure_ascii=False),
             json.dumps(r["ejemplos_no_coincide"], ensure_ascii=False)))
    await cur.execute("INSERT INTO axioma_migracion_de_datos (nombre) VALUES (%s)", (MIGRACION_EJECUTOR_REGLAS_V1,))


def parsear_inventario(texto: str) -> list[dict]:
    """`nombre:ip:puerto:rol[:opcion+opcion]` separados por coma. Sin espacios ni
    comillas: systemd (EnvironmentFile) y bash lo leen igual."""
    filas, nombres = [], set()
    for entrada in texto.split(","):
        partes = entrada.split(":")
        if len(partes) not in (4, 5) or not all(partes[:4]):
            raise ValueError("inventario_mal_formado")
        nombre, ip, puerto_txt, rol = partes[:4]
        opciones = set(partes[4].split("+")) if len(partes) == 5 else set()
        if not puerto_txt.isdigit() or not 0 < int(puerto_txt) < 65536:
            raise ValueError("inventario_puerto_invalido")
        if rol not in _ROLES_EJECUTOR or not opciones <= _OPCIONES_INVENTARIO or nombre in nombres:
            raise ValueError("inventario_valor_invalido")
        nombres.add(nombre)
        filas.append({"nombre": nombre, "ip": ip, "puerto": int(puerto_txt), "rol": rol,
                      "es_local": "local" in opciones, "con_datos_de_clientes": "sin_clientes" not in opciones})
    return filas


async def _ejecutor_inventario_v1(cur) -> None:
    """Las máquinas registradas en la Fase 0 (GO de Fernando por máquina, 2026-09-15),
    desde JAX_EJECUTOR_INVENTARIO. Sin la variable o mal formada, NO se marca: se
    reintenta en el próximo arranque."""
    if await _marcada(cur, MIGRACION_EJECUTOR_INVENTARIO_V1):
        return
    texto = os.environ.get("JAX_EJECUTOR_INVENTARIO", "").strip()
    if not texto:
        return
    try:
        filas = parsear_inventario(texto)
    except ValueError as exc:  # fail-soft: sin inventario la política no se exporta y el Ejecutor no arranca (cerrado); tumbar la plataforma por esto dejaría a la Mesa sin servicio
        _log_migraciones.error("ejecutor_inventario_invalido codigo=%s", exc)
        return
    for f in filas:
        await cur.execute(
            "INSERT IGNORE INTO ejecutor_host (nombre, ip, puerto, rol, es_local, con_datos_de_clientes) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (f["nombre"], f["ip"], f["puerto"], f["rol"], f["es_local"], f["con_datos_de_clientes"]))
    await cur.execute("INSERT INTO axioma_migracion_de_datos (nombre) VALUES (%s)", (MIGRACION_EJECUTOR_INVENTARIO_V1,))
```

En `run_migrations()`, después de `await _ajustes_que_mandan_v1(cur)`:

```python
            await _ejecutor_reglas_v1(cur)
            await _ejecutor_inventario_v1(cur)
```

- [ ] **Step 5: Correr los tests y ver que pasan**

Run: `cd /home/fruiz/worktrees/jax-platform-sp1-c1/backend && pwd && python -m pytest tests/test_ejecutor_tablas.py -v && python3 tests/test_no_fail_open_except.py`
Expected: `13 passed` y el escáner de P10 en verde.

- [ ] **Step 6: Suite completa de jax-platform y commit**

Run: `cd /home/fruiz/worktrees/jax-platform-sp1-c1/backend && python -m pytest -q 2>&1 | tail -3`
Expected: el piso con DB sube exactamente en los tests nuevos; ningún test previo cambia de estado.

```bash
git -C /home/fruiz/worktrees/jax-platform-sp1-c1 add backend/db/migrations.py backend/db/semilla_ejecutor_reglas.json backend/tests/test_ejecutor_tablas.py
git -C /home/fruiz/worktrees/jax-platform-sp1-c1 commit -m "feat(ejecutor): tablas de C1/C2, semilla de reglas con ejemplos e inventario desde el entorno

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

Actualizar el piso del job con DB en `jax-platform/.github/workflows` con el número del runner, en el mismo PR.

---

### Task 2: jax · `formato`, `fallo` y `destinos` — a qué máquinas toca un comando

**Files:**
- Create: `/home/fruiz/worktrees/jax-sp1-c1/jax/ejecutor/contratos/__init__.py`
- Create: `/home/fruiz/worktrees/jax-sp1-c1/jax/ejecutor/contratos/formato.py`
- Create: `/home/fruiz/worktrees/jax-sp1-c1/jax/ejecutor/contratos/fallo.py`
- Create: `/home/fruiz/worktrees/jax-sp1-c1/jax/ejecutor/contratos/destinos.py`
- Test: `/home/fruiz/worktrees/jax-sp1-c1/tests/test_ejecutor_contratos_destinos.py`

**Interfaces:**
- Consume: nada.
- Produce: `formato.campos(pares) -> str`; `Fallo(contrato, codigo, datos=())`; `Host`, `ComandoIlegible`,
  `HostDesconocido`, `destinos(comando, hosts) -> frozenset[str]`.

- [ ] **Step 1: Escribir los tests que fallan**

```python
# tests/test_ejecutor_contratos_destinos.py
"""¿A qué máquinas del inventario toca un comando? (C1/C2, plan 1).

Honesto a propósito: lee ssh/scp/rsync/sftp escritos a la vista. Lo que tiene
que garantizar es fallar CERRADO: destino fuera del inventario o comando que no
se puede partir -> excepción, nunca "no toca nada"."""
import pytest

from jax.ejecutor.contratos import formato
from jax.ejecutor.contratos.destinos import ComandoIlegible, Host, HostDesconocido, destinos

HOSTS = (
    Host("hall9000", "192.0.2.5", 58291, "hypervisor", True),
    Host("atemai", "192.0.2.11", 58291, "desarrollo", False),
    Host("prod", "192.0.2.10", 58291, "produccion", False),
    Host("bridge", "192.0.2.20", 58291, "clientes", False),
)


@pytest.mark.parametrize("comando, esperado", [
    ("uptime", {"hall9000"}),
    ("", {"hall9000"}),
    ("ssh -tt -p 58291 axioma@192.0.2.20 uptime", {"bridge"}),
    ("ssh -tt axioma@bridge 'sudo apt list'", {"bridge"}),
    ("ssh -tt -l axioma prod df -h", {"prod"}),
    ("ssh -tt -p58291 -o BatchMode=yes axioma@atemai", {"atemai"}),
    ("uptime && ssh -tt axioma@prod df", {"hall9000", "prod"}),
    ("ssh -tt axioma@bridge 'ssh -tt axioma@prod uptime'", {"bridge", "prod"}),
    ("sudo -u root ssh -tt axioma@prod uptime", {"prod"}),
    ("FOO=1 timeout 5 ssh -tt axioma@prod uptime", {"prod"}),
    ("/usr/bin/ssh -tt axioma@prod uptime", {"prod"}),
    ("ssh -tt ssh://axioma@prod:58291 uptime", {"prod"}),
    ("scp -P 58291 informe.txt axioma@bridge:/tmp/", {"hall9000", "bridge"}),
    ("rsync -a -e 'ssh -p 58291' axioma@192.0.2.11:/srv/x/ ./x/", {"hall9000", "atemai"}),
    ("grep ssh /var/log/auth.log", {"hall9000"}),
    ("echo a:b", {"hall9000"}),
])
def test_destinos(comando, esperado):
    assert destinos(comando, HOSTS) == frozenset(esperado)


@pytest.mark.parametrize("comando", [
    "ssh -tt axioma@192.0.2.99 uptime",
    "ssh -tt axioma@desconocido uptime",
    "ssh -tt localhost uptime",
    "scp x axioma@otra:/tmp/",
    "ssh -tt axioma@bridge 'ssh -tt axioma@192.0.2.77 uptime'",
])
def test_destino_fuera_del_inventario_falla_cerrado(comando):
    with pytest.raises(HostDesconocido):
        destinos(comando, HOSTS)


@pytest.mark.parametrize("comando", ["echo 'sin cerrar", "ssh -tt", "ssh -p 58291"])
def test_comando_que_no_se_puede_partir_falla_cerrado(comando):
    with pytest.raises(ComandoIlegible):
        destinos(comando, HOSTS)


def test_anidamiento_excesivo_falla_cerrado():
    comando = "uptime"
    for _ in range(5):
        comando = f"ssh -tt axioma@prod {formato.valor(comando)}"
    with pytest.raises(ComandoIlegible):
        destinos(comando, HOSTS)


def test_inventario_sin_una_sola_local_falla_cerrado():
    sin_local = tuple(Host(h.nombre, h.ip, h.puerto, h.rol, False) for h in HOSTS)
    with pytest.raises(HostDesconocido):
        destinos("uptime", sin_local)
    with pytest.raises(HostDesconocido):
        destinos("uptime", HOSTS + (Host("otra", "192.0.2.6", 22, "hypervisor", True),))


def test_formato_neutro():
    assert formato.campos((("codigo", "prohibido"), ("hosts", ("bridge",)), ("regla", None), ("n", 3), ("ok", True))) \
        == 'codigo="prohibido" hosts=["bridge"] regla=null n=3 ok=true'
    assert formato.campos((("x", "ñ \"y\""),)) == 'x="\\u00f1 \\"y\\""'
```

- [ ] **Step 2: Correr y ver que fallan**

Run: `cd /home/fruiz/worktrees/jax-sp1-c1 && pwd && git branch --show-current && PYTHONPATH=.:las_manos python -m pytest tests/test_ejecutor_contratos_destinos.py -v`
Expected: FAIL en la colección, `ModuleNotFoundError: No module named 'jax.ejecutor.contratos'`.

- [ ] **Step 3: Implementar**

```python
# jax/ejecutor/contratos/__init__.py
"""Los seis contratos del Ejecutor (Fase 1, SP1).

Spec: docs/superpowers/specs/2026-09-15-ejecutor-design.md §4.
Plan: docs/superpowers/plans/2026-09-17-ejecutor-contratos-0-indice.md
"""
```

```python
# jax/ejecutor/contratos/formato.py
"""Formato de máquina neutro de los contratos.

Misma gramática que `jax/ejecutor/herramientas.py`: `clave=valor` separados por un
espacio; el valor es un literal JSON ASCII. Sin idioma: lo lee una máquina (o el
modelo) y lo rotula el frontend con i18n. Sólo biblioteca estándar (se instala en
/opt/ejecutor/lib).
"""
from __future__ import annotations

import json


def valor(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "null"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=True)
    if isinstance(v, (tuple, list)):
        return "[" + ",".join(valor(x) for x in v) + "]"
    return json.dumps(repr(v), ensure_ascii=True)


def campos(pares) -> str:
    return " ".join(f"{clave}={valor(v)}" for clave, v in pares)
```

```python
# jax/ejecutor/contratos/fallo.py
"""Un contrato que no está vivo. Lo que junta el arranque (plan 6) para negarse."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Fallo:
    contrato: str
    codigo: str
    datos: tuple = ()
```

```python
# jax/ejecutor/contratos/destinos.py
"""¿A qué máquinas del inventario toca una llamada de herramienta? (C1 y C2)

Spec 2026-09-15 §4, «verdad incómoda»: C1 y C2 operan por patrones y atajan
ERRORES HONESTOS. Este módulo lee lo escrito a la vista —ssh, scp, rsync, sftp—
y nada más: `python3 -c` con un socket no se ve, `bash -c 'ssh …'` tampoco.

Lo que SÍ garantiza, para fallar cerrado:
- destino escrito que no está en el inventario → HostDesconocido;
- comando que no se puede partir, ssh sin destino, anidamiento > 3 → ComandoIlegible;
- inventario sin exactamente UNA máquina local → HostDesconocido.

Segmentos: se parte por operadores de shell. Un segmento cuyo programa efectivo
es `ssh` toca SÓLO su destino, y su comando remoto se lee recursivo con ese
destino como «local». Cualquier otro segmento toca la local; scp/rsync/sftp tocan
la local y cada remoto que nombran.

Sólo biblioteca estándar: lo corre `axioma` desde /opt/ejecutor/lib.
"""
from __future__ import annotations

import os
import re
import shlex
from dataclasses import dataclass

PROFUNDIDAD_MAX = 3

_PUNTUACION = ";&|()"
_ASIGNACION = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_SSH_CON_VALOR = frozenset("BbcDEeFIiJLlmOoPpQRSWw")
_PREFIJOS = {  # programa -> letras de opción que consumen el token siguiente
    "sudo": frozenset("ugphCDRTU"), "nohup": frozenset(), "exec": frozenset(), "time": frozenset(),
    "command": frozenset(), "nice": frozenset("n"), "ionice": frozenset("cnp"), "stdbuf": frozenset("ioe"),
    "env": frozenset("uCS"),
}
_REMOTO = re.compile(r"^(?:[^@/:\s]+@)?(\[[^\]]+\]|[^/:@\s\[\]]+):")
_USUARIO = re.compile(r"^[^@]*@")


@dataclass(frozen=True)
class Host:
    nombre: str
    ip: str
    puerto: int
    rol: str
    es_local: bool


class ComandoIlegible(ValueError):
    """`args[0]` es un código estable."""


class HostDesconocido(ValueError):
    """`args[0]` es el destino tal como está escrito."""


def _palabras(comando: str) -> list[str]:
    lexer = shlex.shlex(comando, posix=True, punctuation_chars=_PUNTUACION)
    lexer.whitespace_split = True
    lexer.commenters = ""
    try:
        return list(lexer)
    except ValueError:
        raise ComandoIlegible("comillas_sin_cerrar") from None


def _segmentos(palabras):
    actual: list[str] = []
    for p in palabras:
        if p and set(p) <= set(_PUNTUACION):
            if actual:
                yield actual
            actual = []
        else:
            actual.append(p)
    if actual:
        yield actual


def _saltar_opciones(segmento, i, con_valor):
    while i < len(segmento) and segmento[i].startswith("-") and segmento[i] != "-":
        opcion = segmento[i]
        i += 1
        if len(opcion) == 2 and opcion[1] in con_valor:
            i += 1
    return i


def _programa(segmento):
    i = 0
    while i < len(segmento):
        palabra = segmento[i]
        base = os.path.basename(palabra)
        if _ASIGNACION.match(palabra):
            i += 1
        elif base == "timeout":
            i = _saltar_opciones(segmento, i + 1, frozenset("sk")) + 1
        elif base in _PREFIJOS:
            i = _saltar_opciones(segmento, i + 1, _PREFIJOS[base])
        else:
            return base, segmento[i + 1:]
    return "", []


def _destino_ssh(args):
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--":
            i += 1
            break
        if len(a) > 1 and a.startswith("-"):
            for j in range(1, len(a)):
                if a[j] in _SSH_CON_VALOR:
                    if j == len(a) - 1:
                        i += 1
                    break
            i += 1
            continue
        break
    if i >= len(args):
        raise ComandoIlegible("ssh_sin_destino")
    return args[i], args[i + 1:]


def _host_de_destino(destino: str) -> str:
    if destino.startswith("ssh://"):
        destino = _USUARIO.sub("", destino[len("ssh://"):], count=1)
        if destino.count(":") == 1:
            destino = destino.split(":", 1)[0]
        return destino
    return _USUARIO.sub("", destino, count=1)


def _resolver(escrito: str, hosts) -> str:
    limpio = escrito.strip("[]")
    for h in hosts:
        if limpio in (h.nombre, h.ip):
            return h.nombre
    raise HostDesconocido(escrito)


def _destinos(comando, hosts, local, profundidad):
    if profundidad > PROFUNDIDAD_MAX:
        raise ComandoIlegible("anidamiento_excesivo")
    tocados: set[str] = set()
    for segmento in _segmentos(_palabras(comando)):
        programa, args = _programa(segmento)
        if programa == "ssh":
            destino, remoto = _destino_ssh(args)
            host = _resolver(_host_de_destino(destino), hosts)
            tocados.add(host)
            if remoto:
                tocados |= _destinos(" ".join(remoto), hosts, host, profundidad + 1)
        elif programa == "sftp":
            tocados.add(local)
            destino, _ = _destino_ssh(args)
            tocados.add(_resolver(_host_de_destino(destino.split(":", 1)[0]), hosts))
        elif programa in ("scp", "rsync"):
            tocados.add(local)
            for a in args:
                m = None if a.startswith("-") else _REMOTO.match(a)
                if m:
                    tocados.add(_resolver(m.group(1), hosts))
        else:
            tocados.add(local)
    return tocados or {local}


def destinos(comando: str, hosts) -> frozenset[str]:
    hosts = tuple(hosts)
    locales = [h.nombre for h in hosts if h.es_local]
    if len(locales) != 1:
        raise HostDesconocido("local")
    return frozenset(_destinos(comando, hosts, locales[0], 0))
```

- [ ] **Step 4: Correr y ver que pasan**

Run: `cd /home/fruiz/worktrees/jax-sp1-c1 && PYTHONPATH=.:las_manos python -m pytest tests/test_ejecutor_contratos_destinos.py -v`
Expected: todos PASS (27). Si un caso falla, se corrige el código, **no** el caso: cada caso es una forma real de
escribir un comando.

- [ ] **Step 5: Mutación del fail-closed (verla fallar)**

```bash
cd /home/fruiz/worktrees/jax-sp1-c1 && export PYTHONDONTWRITEBYTECODE=1
cp jax/ejecutor/contratos/destinos.py jax/ejecutor/contratos/destinos.py.mut-bak
python - <<'PY'
import pathlib
p = pathlib.Path("jax/ejecutor/contratos/destinos.py")
p.write_text(p.read_text().replace('    raise HostDesconocido(escrito)\n', '    return escrito\n', 1))
PY
PYTHONPATH=.:las_manos python -m pytest tests/test_ejecutor_contratos_destinos.py -q 2>&1 | tail -1
cp jax/ejecutor/contratos/destinos.py.mut-bak jax/ejecutor/contratos/destinos.py
cmp jax/ejecutor/contratos/destinos.py jax/ejecutor/contratos/destinos.py.mut-bak && rm jax/ejecutor/contratos/destinos.py.mut-bak
```
Expected: con la mutación, `5 failed` (los de `test_destino_fuera_del_inventario_falla_cerrado`); restaurado, `cmp` sin salida.

- [ ] **Step 6: Commit**

```bash
git -C /home/fruiz/worktrees/jax-sp1-c1 add jax/ejecutor/contratos/__init__.py jax/ejecutor/contratos/formato.py jax/ejecutor/contratos/fallo.py jax/ejecutor/contratos/destinos.py tests/test_ejecutor_contratos_destinos.py
git -C /home/fruiz/worktrees/jax-sp1-c1 commit -m "feat(ejecutor): destinos de un comando para C1/C2, cerrado ante lo que no se reconoce

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: jax · `politica` — cargar fail-closed, evaluar y autoprueba

**Files:**
- Create: `/home/fruiz/worktrees/jax-sp1-c1/jax/ejecutor/contratos/politica.py`
- Test: `/home/fruiz/worktrees/jax-sp1-c1/tests/test_ejecutor_contratos_politica.py`

**Interfaces:**
- Consume: `destinos.Host`, `destinos.destinos`, `ComandoIlegible`, `HostDesconocido` (Task 2).
- Produce: todo `politica.py` (ver Interfaces del plan). Documento JSON versión 1:
  `{"version":1,"generada_at":str,"hosts":[{nombre,ip,puerto,rol,es_local}],"reglas":[{id,codigo,tipo,herramientas,campo,patron,ambito_hosts:[str],ambito_roles:[str],es_canario,ejemplos_coincide:[{tool_name,tool_input}],ejemplos_no_coincide:[…]}],"respaldos":{nombre: ISO-8601 con zona},"c2_edad_max_s":int,"sha256":hex}`.

- [ ] **Step 1: Escribir los tests que fallan**

```python
# tests/test_ejecutor_contratos_politica.py
"""Política del Ejecutor (C1 prohibiciones, C2 respaldo antes de destruir).

Lo que DEBE bloquear se prueba bloqueando; lo ilegible bloquea todo."""
import json
import os
from datetime import datetime, timedelta, timezone

import pytest

from jax.ejecutor.contratos import politica as P

AHORA = datetime(2026, 9, 17, 12, 0, tzinfo=timezone.utc)
OTRO_UID = os.getuid() + 1


def _bash(comando):
    return {"tool_name": "Bash", "tool_input": {"command": comando}}


def doc_base(**cambios):
    doc = {
        "version": 1, "generada_at": "2026-09-17T11:59:00+00:00",
        "hosts": [
            {"nombre": "hall9000", "ip": "192.0.2.5", "puerto": 58291, "rol": "hypervisor", "es_local": True},
            {"nombre": "bridge", "ip": "192.0.2.20", "puerto": 58291, "rol": "clientes", "es_local": False},
            {"nombre": "atemai", "ip": "192.0.2.11", "puerto": 58291, "rol": "desarrollo", "es_local": False},
        ],
        "reglas": [
            {"id": 1, "codigo": "canario_c1", "tipo": "prohibido", "herramientas": "Bash", "campo": "command",
             "patron": "ejecutor-canario-c1", "ambito_hosts": [], "ambito_roles": [], "es_canario": True,
             "ejemplos_coincide": [_bash("echo ejecutor-canario-c1")], "ejemplos_no_coincide": [_bash("echo x")]},
            {"id": 2, "codigo": "apt_full_upgrade_bridge", "tipo": "prohibido", "herramientas": "Bash",
             "campo": "command", "patron": r"\bapt\s+full-upgrade\b", "ambito_hosts": ["bridge"], "ambito_roles": [],
             "es_canario": False, "ejemplos_coincide": [_bash("ssh -tt axioma@bridge 'apt full-upgrade'")],
             "ejemplos_no_coincide": [_bash("ssh -tt axioma@atemai 'apt full-upgrade'")]},
            {"id": 3, "codigo": "borrar_archivos", "tipo": "destructivo", "herramientas": "Bash", "campo": "command",
             "patron": r"\brm\s", "ambito_hosts": [], "ambito_roles": [], "es_canario": False,
             "ejemplos_coincide": [_bash("rm -rf /tmp/x")], "ejemplos_no_coincide": [_bash("ls /tmp")]},
            {"id": 4, "codigo": "env_por_ruta", "tipo": "prohibido", "herramientas": "Write|Edit", "campo": "file_path",
             "patron": r"\.env$", "ambito_hosts": [], "ambito_roles": ["clientes", "hypervisor"], "es_canario": False,
             "ejemplos_coincide": [{"tool_name": "Write", "tool_input": {"file_path": "/a/.env", "content": ""}}],
             "ejemplos_no_coincide": [{"tool_name": "Read", "tool_input": {"file_path": "/a/.env"}}]},
        ],
        "respaldos": {"bridge": "2026-09-17T06:00:00+00:00"},
        "c2_edad_max_s": 86400,
    }
    doc.update(cambios)
    return P.firmar(doc)


def escribir(tmp_path, doc, modo=0o644):
    ruta = tmp_path / "politica.json"
    ruta.write_text(json.dumps(doc))
    ruta.chmod(modo)
    return ruta


def cargada(tmp_path, **cambios):
    return P.cargar(escribir(tmp_path, doc_base(**cambios)), uid_de_la_cuenta=OTRO_UID)


# --- cargar: fail-closed ------------------------------------------------------

def test_carga_una_politica_sana(tmp_path):
    p = cargada(tmp_path)
    assert [r.codigo for r in p.reglas] == ["canario_c1", "apt_full_upgrade_bridge", "borrar_archivos", "env_por_ruta"]


def test_archivo_ausente(tmp_path):
    with pytest.raises(P.PoliticaIlegible) as e:
        P.cargar(tmp_path / "no-existe.json", uid_de_la_cuenta=OTRO_UID)
    assert e.value.codigo == "no_se_puede_leer"


def test_archivo_de_la_propia_cuenta(tmp_path):
    with pytest.raises(P.PoliticaIlegible) as e:
        P.cargar(escribir(tmp_path, doc_base()), uid_de_la_cuenta=os.getuid())
    assert e.value.codigo == "duenio_es_la_cuenta"


def test_archivo_escribible_por_el_grupo(tmp_path):
    with pytest.raises(P.PoliticaIlegible) as e:
        P.cargar(escribir(tmp_path, doc_base(), modo=0o664), uid_de_la_cuenta=OTRO_UID)
    assert e.value.codigo == "escribible_por_otros"


def test_un_byte_cambiado_rompe_el_sha256(tmp_path):
    doc = doc_base()
    doc["c2_edad_max_s"] = 999999
    with pytest.raises(P.PoliticaIlegible) as e:
        P.cargar(escribir(tmp_path, doc), uid_de_la_cuenta=OTRO_UID)
    assert e.value.codigo == "sha256_no_cuadra"


@pytest.mark.parametrize("contenido, codigo", [(b"{no json", "json_invalido"), (b"[]", "json_invalido")])
def test_json_roto(tmp_path, contenido, codigo):
    ruta = tmp_path / "politica.json"
    ruta.write_bytes(contenido)
    ruta.chmod(0o644)  # con umask 002 nacería escribible por el grupo y caería antes, por otra causa
    with pytest.raises(P.PoliticaIlegible) as e:
        P.cargar(ruta, uid_de_la_cuenta=OTRO_UID)
    assert e.value.codigo == codigo


def _sin(doc, codigo):
    return [r for r in doc["reglas"] if r["codigo"] != codigo]


@pytest.mark.parametrize("mutar, codigo", [
    (lambda d: d.update(version=2), "version_desconocida"),
    (lambda d: d.update(reglas=_sin(d, "canario_c1")), "canario_ausente_o_multiple"),
    (lambda d: d["reglas"][1].update(es_canario=True), "canario_ausente_o_multiple"),
    (lambda d: d["reglas"][2].update(ejemplos_coincide=[]), "regla_sin_ejemplo_que_coincida"),
    (lambda d: d["reglas"][2].update(patron="(sin cerrar"), "regex_invalida"),
    (lambda d: d["reglas"][1].update(ambito_hosts=["no-esta"]), "ambito_desconocido"),
    (lambda d: d["reglas"][1].update(codigo="canario_c1"), "regla_duplicada"),
    (lambda d: d["hosts"][1].update(es_local=True), "inventario_sin_una_local"),
    (lambda d: d.update(c2_edad_max_s=0), "campo_invalido"),
    (lambda d: d.update(respaldos={"bridge": "2026-09-17T06:00:00"}), "respaldo_invalido"),
    (lambda d: d.update(respaldos={"no-esta": "2026-09-17T06:00:00+00:00"}), "respaldo_invalido"),
])
def test_cualquier_parte_ilegible_invalida_todo(tmp_path, mutar, codigo):
    doc = {k: v for k, v in doc_base().items() if k != "sha256"}
    doc = json.loads(json.dumps(doc))
    mutar(doc)
    with pytest.raises(P.PoliticaIlegible) as e:
        P.cargar(escribir(tmp_path, P.firmar(doc)), uid_de_la_cuenta=OTRO_UID)
    assert e.value.codigo == codigo


# --- evaluar ------------------------------------------------------------------

def test_canario_se_bloquea(tmp_path):
    d = P.evaluar(cargada(tmp_path), "Bash", {"command": "touch x # ejecutor-canario-c1"}, AHORA)
    assert (d.permitir, d.codigo, d.regla, d.hosts) == (False, P.PROHIBIDO, "canario_c1", ("hall9000",))


def test_prohibido_con_ambito_de_host(tmp_path):
    p = cargada(tmp_path)
    assert P.evaluar(p, "Bash", {"command": "ssh -tt axioma@bridge 'apt full-upgrade'"}, AHORA).codigo == P.PROHIBIDO
    assert P.evaluar(p, "Bash", {"command": "ssh -tt axioma@atemai 'apt full-upgrade'"}, AHORA).permitir is True


def test_prohibido_por_ruta_con_ambito_de_rol(tmp_path):
    p = cargada(tmp_path)
    assert P.evaluar(p, "Write", {"file_path": "/www/.env", "content": ""}, AHORA).regla == "env_por_ruta"
    assert P.evaluar(p, "Read", {"file_path": "/www/.env"}, AHORA).permitir is True


def test_destructivo_sin_respaldo_se_bloquea_por_maquina(tmp_path):
    p = cargada(tmp_path)
    local = P.evaluar(p, "Bash", {"command": "rm -rf /tmp/x"}, AHORA)
    assert (local.permitir, local.codigo, local.hosts) == (False, P.DESTRUCTIVO_SIN_RESPALDO, ("hall9000",))
    remoto = P.evaluar(p, "Bash", {"command": "ssh -tt axioma@bridge 'rm -rf /tmp/x'"}, AHORA)
    assert remoto.permitir is True


def test_respaldo_viejo_o_del_futuro_no_cuenta(tmp_path):
    comando = {"command": "ssh -tt axioma@bridge 'rm -rf /tmp/x'"}
    viejo = cargada(tmp_path, respaldos={"bridge": (AHORA - timedelta(days=2)).isoformat()})
    assert P.evaluar(viejo, "Bash", comando, AHORA).codigo == P.DESTRUCTIVO_SIN_RESPALDO
    futuro = cargada(tmp_path, respaldos={"bridge": (AHORA + timedelta(minutes=1)).isoformat()})
    assert P.evaluar(futuro, "Bash", comando, AHORA).codigo == P.DESTRUCTIVO_SIN_RESPALDO


@pytest.mark.parametrize("tool_name, tool_input, codigo", [
    ("Bash", {"command": "ssh -tt axioma@192.0.2.99 uptime"}, P.HOST_DESCONOCIDO),
    ("Bash", {"command": "echo 'sin cerrar"}, P.COMANDO_ILEGIBLE),
    ("Bash", {}, P.ENTRADA_ILEGIBLE),
    ("Bash", "no-es-dict", P.ENTRADA_ILEGIBLE),
    (None, {"command": "uptime"}, P.ENTRADA_ILEGIBLE),
])
def test_lo_que_no_se_entiende_se_bloquea(tmp_path, tool_name, tool_input, codigo):
    d = P.evaluar(cargada(tmp_path), tool_name, tool_input, AHORA)
    assert (d.permitir, d.codigo) == (False, codigo)


def test_lo_inocuo_pasa(tmp_path):
    assert P.evaluar(cargada(tmp_path), "Bash", {"command": "uptime"}, AHORA) == \
        P.Decision(True, P.PERMITIDO, None, ("hall9000",))


# --- autoprueba ---------------------------------------------------------------

def test_autoprueba_sana_no_devuelve_fallos(tmp_path):
    assert P.autoprueba(cargada(tmp_path)) == ()


def test_autoprueba_ve_un_ejemplo_que_miente(tmp_path):
    doc = {k: v for k, v in doc_base().items() if k != "sha256"}
    doc["reglas"][2]["ejemplos_coincide"] = [_bash("ls /tmp")]
    p = P.cargar(escribir(tmp_path, P.firmar(doc)), uid_de_la_cuenta=OTRO_UID)
    assert P.autoprueba(p) == (P.FalloDeEjemplo("borrar_archivos", 0, "coincide", ()),)


def test_autoprueba_cuenta_un_error_de_destino_como_fallo(tmp_path):
    doc = {k: v for k, v in doc_base().items() if k != "sha256"}
    doc["reglas"][2]["ejemplos_no_coincide"] = [_bash("ssh -tt axioma@192.0.2.99 ls")]
    p = P.cargar(escribir(tmp_path, P.firmar(doc)), uid_de_la_cuenta=OTRO_UID)
    assert P.autoprueba(p) == (P.FalloDeEjemplo("borrar_archivos", 0, "no_coincide", (P.HOST_DESCONOCIDO,)),)
```

- [ ] **Step 2: Correr y ver que fallan**

Run: `cd /home/fruiz/worktrees/jax-sp1-c1 && PYTHONPATH=.:las_manos python -m pytest tests/test_ejecutor_contratos_politica.py -v`
Expected: FAIL en la colección, `ImportError: cannot import name 'politica'`.

- [ ] **Step 3: Implementar**

```python
# jax/ejecutor/contratos/politica.py
"""Política del Ejecutor: prohibiciones (C1) y respaldo antes de destruir (C2).

Spec 2026-09-15 §4. Plan 2026-09-17-ejecutor-contratos-1.

- `cargar(ruta, uid_de_la_cuenta)` FALLA CERRADO: ausente, ilegible, de la propia
  cuenta, escribible por grupo u otros (archivo o directorio), JSON roto, versión
  desconocida, sha256 que no cuadra, regla sin ejemplo que coincida, regex que no
  compila, ámbito desconocido, sin canario o con más de uno. Una política que no se
  puede creer entera no se cree en parte: el gancho bloquea TODO.
- `evaluar`: prohibido gana; después, destructivo sin punto de restauración vigente
  en CADA máquina tocada. Destino desconocido o comando ilegible → bloqueado.
- `autoprueba`: cada ejemplo de cada regla da lo que dice. Es la prueba del spec
  («cada prohibido se intenta y sale bloqueado») repetida en cada arranque.

El sha256 detecta corrupción y escrituras a medias; la autenticidad la da el
dueño del archivo (otra cuenta) y los permisos, que se comprueban.

Sólo biblioteca estándar (lo corre `axioma` desde /opt/ejecutor/lib).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from jax.ejecutor.contratos import destinos as _destinos
from jax.ejecutor.contratos.destinos import Host

VERSION = 1
TIPOS = ("prohibido", "destructivo")
CAMPOS = ("command", "file_path", "cualquiera")
ROLES = ("hypervisor", "desarrollo", "produccion", "clientes", "respaldo")

PERMITIDO = "permitido"
PROHIBIDO = "prohibido"
DESTRUCTIVO_SIN_RESPALDO = "destructivo_sin_respaldo"
HOST_DESCONOCIDO = "host_desconocido"
COMANDO_ILEGIBLE = "comando_ilegible"
ENTRADA_ILEGIBLE = "entrada_ilegible"
POLITICA_ILEGIBLE = "politica_ilegible"

_CAMPOS_DE_RUTA = ("file_path", "notebook_path", "path")


class PoliticaIlegible(ValueError):
    def __init__(self, codigo: str, datos: tuple = ()):
        super().__init__(codigo)
        self.codigo = codigo
        self.datos = datos


@dataclass(frozen=True)
class Regla:
    id: int
    codigo: str
    tipo: str
    herramientas: re.Pattern
    campo: str
    patron: re.Pattern
    ambito_hosts: frozenset
    ambito_roles: frozenset
    es_canario: bool
    ejemplos_coincide: tuple
    ejemplos_no_coincide: tuple


@dataclass(frozen=True)
class Politica:
    generada_at: str
    hosts: tuple
    reglas: tuple
    respaldos: dict
    c2_edad_max_s: int


@dataclass(frozen=True)
class Decision:
    permitir: bool
    codigo: str
    regla: str | None
    hosts: tuple


@dataclass(frozen=True)
class FalloDeEjemplo:
    regla: str
    indice: int
    esperado: str
    obtenido: tuple


def contenido_canonico(doc: dict) -> bytes:
    sin_sha = {k: v for k, v in doc.items() if k != "sha256"}
    return json.dumps(sin_sha, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode()


def firmar(doc: dict) -> dict:
    return {**doc, "sha256": hashlib.sha256(contenido_canonico(doc)).hexdigest()}


# --- validación ---------------------------------------------------------------

def _lista(d: dict, clave: str) -> list:
    v = d.get(clave)
    if not isinstance(v, list):
        raise PoliticaIlegible("campo_invalido", (("campo", clave),))
    return v


def _texto(v, campo: str) -> str:
    if not isinstance(v, str) or not v:
        raise PoliticaIlegible("campo_invalido", (("campo", campo),))
    return v


def _entero(v, campo: str) -> int:
    if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
        raise PoliticaIlegible("campo_invalido", (("campo", campo),))
    return v


def _host(h) -> Host:
    if not isinstance(h, dict) or h.get("rol") not in ROLES or not isinstance(h.get("es_local"), bool):
        raise PoliticaIlegible("inventario_invalido")
    return Host(_texto(h.get("nombre"), "nombre"), _texto(h.get("ip"), "ip"), _entero(h.get("puerto"), "puerto"),
                h["rol"], h["es_local"])


def _ejemplo(e, codigo: str) -> dict:
    if not isinstance(e, dict) or not isinstance(e.get("tool_name"), str) or not isinstance(e.get("tool_input"), dict):
        raise PoliticaIlegible("ejemplo_invalido", (("regla", codigo),))
    return e


def _regla(r, nombres: set) -> Regla:
    if not isinstance(r, dict):
        raise PoliticaIlegible("regla_invalida")
    codigo = _texto(r.get("codigo"), "codigo")
    datos = (("regla", codigo),)
    if r.get("tipo") not in TIPOS or r.get("campo") not in CAMPOS or not isinstance(r.get("es_canario"), bool):
        raise PoliticaIlegible("regla_invalida", datos)
    try:
        herramientas = re.compile(_texto(r.get("herramientas"), "herramientas"))
        patron = re.compile(_texto(r.get("patron"), "patron"))
    except re.error:
        raise PoliticaIlegible("regex_invalida", datos) from None
    ambito_hosts = frozenset(_texto(x, "ambito_hosts") for x in _lista(r, "ambito_hosts"))
    ambito_roles = frozenset(_texto(x, "ambito_roles") for x in _lista(r, "ambito_roles"))
    if not ambito_hosts <= nombres or not ambito_roles <= set(ROLES):
        raise PoliticaIlegible("ambito_desconocido", datos)
    coincide = tuple(_ejemplo(e, codigo) for e in _lista(r, "ejemplos_coincide"))
    no_coincide = tuple(_ejemplo(e, codigo) for e in _lista(r, "ejemplos_no_coincide"))
    if not coincide:
        raise PoliticaIlegible("regla_sin_ejemplo_que_coincida", datos)
    return Regla(_entero(r.get("id"), "id"), codigo, r["tipo"], herramientas, r["campo"], patron,
                 ambito_hosts, ambito_roles, r["es_canario"], coincide, no_coincide)


def _respaldos(v, nombres: set) -> dict:
    if not isinstance(v, dict):
        raise PoliticaIlegible("respaldo_invalido")
    salida = {}
    for nombre, momento in v.items():
        try:
            t = datetime.fromisoformat(momento)
        except (TypeError, ValueError):
            raise PoliticaIlegible("respaldo_invalido", (("host", nombre),)) from None
        if nombre not in nombres or t.tzinfo is None:
            raise PoliticaIlegible("respaldo_invalido", (("host", nombre),))
        salida[nombre] = t
    return salida


def validar(doc) -> Politica:
    if not isinstance(doc, dict):
        raise PoliticaIlegible("json_invalido")
    if doc.get("version") != VERSION:
        raise PoliticaIlegible("version_desconocida", (("version", repr(doc.get("version"))),))
    firma = doc.get("sha256")
    if not isinstance(firma, str) or hashlib.sha256(contenido_canonico(doc)).hexdigest() != firma:
        raise PoliticaIlegible("sha256_no_cuadra")
    hosts = tuple(_host(h) for h in _lista(doc, "hosts"))
    nombres = [h.nombre for h in hosts]
    if not hosts or len(set(nombres)) != len(nombres):
        raise PoliticaIlegible("inventario_invalido")
    if sum(h.es_local for h in hosts) != 1:
        raise PoliticaIlegible("inventario_sin_una_local")
    reglas = tuple(_regla(r, set(nombres)) for r in _lista(doc, "reglas"))
    codigos = [r.codigo for r in reglas]
    if len(set(codigos)) != len(codigos):
        raise PoliticaIlegible("regla_duplicada")
    canarios = [r for r in reglas if r.es_canario]
    if len(canarios) != 1 or canarios[0].tipo != "prohibido":
        raise PoliticaIlegible("canario_ausente_o_multiple", (("canarios", len(canarios)),))
    return Politica(_texto(doc.get("generada_at"), "generada_at"), hosts, reglas,
                    _respaldos(doc.get("respaldos"), set(nombres)), _entero(doc.get("c2_edad_max_s"), "c2_edad_max_s"))


def _exigir_ajeno(ruta: Path, uid: int) -> None:
    for objetivo in (ruta, ruta.parent):
        st = os.stat(objetivo)
        if st.st_uid == uid:
            raise PoliticaIlegible("duenio_es_la_cuenta", (("ruta", str(objetivo)),))
        if st.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
            raise PoliticaIlegible("escribible_por_otros", (("ruta", str(objetivo)),))


def cargar(ruta, *, uid_de_la_cuenta: int) -> Politica:
    ruta = Path(ruta)
    try:
        _exigir_ajeno(ruta, uid_de_la_cuenta)
        crudo = ruta.read_bytes()
    except OSError as exc:
        raise PoliticaIlegible("no_se_puede_leer", (("error", type(exc).__name__),)) from None
    try:
        doc = json.loads(crudo)
    except ValueError:
        raise PoliticaIlegible("json_invalido") from None
    return validar(doc)


# --- evaluación ---------------------------------------------------------------

class _EntradaIlegible(ValueError):
    pass


def _hosts_de(p: Politica, tool_name: str, tool_input: dict) -> frozenset:
    if tool_name == "Bash":
        comando = tool_input.get("command")
        if not isinstance(comando, str):
            raise _EntradaIlegible()
        return _destinos.destinos(comando, p.hosts)
    return frozenset(h.nombre for h in p.hosts if h.es_local)


def _texto_de(regla: Regla, tool_input: dict):
    if regla.campo == "command":
        v = tool_input.get("command")
        return v if isinstance(v, str) else None
    if regla.campo == "file_path":
        for c in _CAMPOS_DE_RUTA:
            if isinstance(tool_input.get(c), str):
                return tool_input[c]
        return None
    return json.dumps(tool_input, sort_keys=True, ensure_ascii=False)


def _que_aplica(p: Politica, tool_name, tool_input):
    if not isinstance(tool_name, str) or not isinstance(tool_input, dict):
        return ENTRADA_ILEGIBLE, frozenset(), ()
    try:
        hosts = _hosts_de(p, tool_name, tool_input)
    except _EntradaIlegible:
        return ENTRADA_ILEGIBLE, frozenset(), ()
    except _destinos.HostDesconocido:
        return HOST_DESCONOCIDO, frozenset(), ()
    except _destinos.ComandoIlegible:
        return COMANDO_ILEGIBLE, frozenset(), ()
    por_nombre = {h.nombre: h for h in p.hosts}
    aplican = []
    for regla in p.reglas:
        donde = frozenset(n for n in hosts
                          if (not regla.ambito_hosts or n in regla.ambito_hosts)
                          and (not regla.ambito_roles or por_nombre[n].rol in regla.ambito_roles))
        if not donde or not regla.herramientas.fullmatch(tool_name):
            continue
        texto = _texto_de(regla, tool_input)
        if texto is not None and regla.patron.search(texto):
            aplican.append((regla, donde))
    return None, hosts, tuple(aplican)


def _respaldo_vigente(p: Politica, host: str, ahora: datetime) -> bool:
    t = p.respaldos.get(host)
    return t is not None and timedelta(0) <= ahora - t <= timedelta(seconds=p.c2_edad_max_s)


def evaluar(p: Politica, tool_name, tool_input, ahora: datetime) -> Decision:
    if ahora.tzinfo is None:
        raise ValueError("ahora_sin_zona")
    error, hosts, aplican = _que_aplica(p, tool_name, tool_input)
    if error:
        return Decision(False, error, None, ())
    for regla, donde in aplican:
        if regla.tipo == "prohibido":
            return Decision(False, PROHIBIDO, regla.codigo, tuple(sorted(donde)))
    for regla, donde in aplican:
        sin = tuple(n for n in sorted(donde) if not _respaldo_vigente(p, n, ahora))
        if sin:
            return Decision(False, DESTRUCTIVO_SIN_RESPALDO, regla.codigo, sin)
    return Decision(True, PERMITIDO, None, tuple(sorted(hosts)))


def autoprueba(p: Politica) -> tuple:
    fallos = []
    for regla in p.reglas:
        for esperado, ejemplos in (("coincide", regla.ejemplos_coincide), ("no_coincide", regla.ejemplos_no_coincide)):
            for i, e in enumerate(ejemplos):
                error, _, aplican = _que_aplica(p, e["tool_name"], e["tool_input"])
                obtenido = (error,) if error else tuple(r.codigo for r, _ in aplican)
                coincide = error is None and regla.codigo in obtenido
                if error is not None or coincide != (esperado == "coincide"):
                    fallos.append(FalloDeEjemplo(regla.codigo, i, esperado, obtenido))
    return tuple(fallos)
```

Nota: `test_autoprueba_ve_un_ejemplo_que_miente` espera `obtenido=()` porque `ls /tmp` no coincide con ninguna regla.

- [ ] **Step 4: Correr y ver que pasan**

Run: `cd /home/fruiz/worktrees/jax-sp1-c1 && PYTHONPATH=.:las_manos python -m pytest tests/test_ejecutor_contratos_politica.py -v && python3 policy/tests/test_no_fail_open_except.py`
Expected: todos PASS; P10 verde (no hay `except` amplio en el módulo).

- [ ] **Step 5: Mutaciones — suprimir y neutralizar (taxonomía, no conteo)**

Tres mutaciones, una por clase; cada una con backup y `cmp`, como en la Task 2:
1. **Suprimir** el chequeo de sha256: reemplazar `raise PoliticaIlegible("sha256_no_cuadra")` por `pass`.
   Expected: FAIL sólo `test_un_byte_cambiado_rompe_el_sha256` (los parametrizados re-firman su documento, así
   que no dependen del chequeo: que no caigan es correcto, no un hueco).
2. **Neutralizar dejando presente:** en `_exigir_ajeno`, `if st.st_uid == uid:` → `if st.st_uid == uid and False:`.
   Expected: FAIL `test_archivo_de_la_propia_cuenta`.
3. **Invertir la precedencia:** en `evaluar`, `if error:` → `if error and False:`.
   Expected: FAIL los 5 de `test_lo_que_no_se_entiende_se_bloquea`.

- [ ] **Step 6: Commit**

```bash
git -C /home/fruiz/worktrees/jax-sp1-c1 add jax/ejecutor/contratos/politica.py tests/test_ejecutor_contratos_politica.py
git -C /home/fruiz/worktrees/jax-sp1-c1 commit -m "feat(ejecutor): política de C1/C2 — ilegible bloquea todo, cada regla con su autoprueba

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: jax · el gancho y su envoltorio fail-closed

**Files:**
- Create: `/home/fruiz/worktrees/jax-sp1-c1/jax/ejecutor/contratos/gancho.py`
- Create: `/home/fruiz/worktrees/jax-sp1-c1/ops/ejecutor/gancho.sh.plantilla`
- Create: `/home/fruiz/worktrees/jax-sp1-c1/jax/ejecutor/contratos/instalacion.py`
- Test: `/home/fruiz/worktrees/jax-sp1-c1/tests/test_ejecutor_contratos_gancho.py`
- Test: `/home/fruiz/worktrees/jax-sp1-c1/tests/test_ejecutor_contratos_instalacion.py`

**Interfaces:**
- Consume: `politica.cargar/evaluar/autoprueba`, `formato.campos`.
- Produce: `gancho.principal(argv, *, entrada, salida, errores, ahora) -> int`; `instalacion.INSTALABLES`,
  `renderizar_gancho(lib, tope_s)`, `renderizar_managed_settings(lib, politica, tope_s)`,
  `SETTINGS_USUARIO`, `manifiesto(archivos)`, `principal(argv, env)` (escribe la etapa de instalación).

**Contrato con Claude Code** (documentación oficial, verificado 2026-09-17): stdin JSON con `tool_name`,
`tool_input`, `tool_use_id`, `session_id`, `agent_id` (subagentes); **exit 2 bloquea**; exit 1, crash, 127 o
timeout **no bloquean**; `permissionDecision: deny` y exit 2 bloquean incluso con `bypassPermissions`.

- [ ] **Step 1: Escribir los tests que fallan**

```python
# tests/test_ejecutor_contratos_gancho.py
"""Gancho PreToolUse del Ejecutor y su envoltorio fail-closed (C1/C2).

El programa que llama al gancho NO bloquea si el gancho sale con 1, 127, se cae o
vence su tiempo. Estos tests prueban que el envoltorio convierte TODO eso en 2."""
import io
import json
import os
import subprocess
import time
from datetime import datetime, timezone

import pytest

from jax.ejecutor.contratos import gancho, instalacion
from tests.test_ejecutor_contratos_politica import OTRO_UID, doc_base, escribir

AHORA = datetime(2026, 9, 17, 12, 0, tzinfo=timezone.utc)


def _llamar(monkeypatch, tmp_path, evento, argv_extra=(), doc=None, uid=OTRO_UID):
    ruta = escribir(tmp_path, doc or doc_base())
    monkeypatch.setattr(gancho.os, "getuid", lambda: uid)
    salida, errores = io.StringIO(), io.StringIO()
    rc = gancho.principal([str(ruta), *argv_extra], entrada=io.StringIO(json.dumps(evento)),
                          salida=salida, errores=errores, ahora=AHORA)
    return rc, salida.getvalue(), errores.getvalue()


def test_permitido_sale_0_sin_ruido(monkeypatch, tmp_path):
    assert _llamar(monkeypatch, tmp_path, {"tool_name": "Bash", "tool_input": {"command": "uptime"}}) == (0, "", "")


def test_canario_sale_2_con_su_codigo(monkeypatch, tmp_path):
    rc, _, err = _llamar(monkeypatch, tmp_path, {"tool_name": "Bash", "tool_input": {"command": "echo ejecutor-canario-c1"}})
    assert rc == 2
    assert err == 'contrato="c1" codigo="prohibido" regla="canario_c1" hosts=["hall9000"]\n'


def test_destructivo_sin_respaldo_dice_c2(monkeypatch, tmp_path):
    rc, _, err = _llamar(monkeypatch, tmp_path, {"tool_name": "Bash", "tool_input": {"command": "rm -rf /tmp/x"}})
    assert rc == 2 and err.startswith('contrato="c2" codigo="destructivo_sin_respaldo"')


def test_politica_de_la_propia_cuenta_bloquea_todo(monkeypatch, tmp_path):
    rc, _, err = _llamar(monkeypatch, tmp_path, {"tool_name": "Bash", "tool_input": {"command": "uptime"}},
                         uid=os.getuid())
    assert rc == 2 and 'codigo="politica_ilegible" motivo="duenio_es_la_cuenta"' in err


@pytest.mark.parametrize("crudo", ["", "no json", "[]"])
def test_entrada_ilegible_bloquea(monkeypatch, tmp_path, crudo):
    ruta = escribir(tmp_path, doc_base())
    monkeypatch.setattr(gancho.os, "getuid", lambda: OTRO_UID)
    errores = io.StringIO()
    assert gancho.principal([str(ruta)], entrada=io.StringIO(crudo), salida=io.StringIO(), errores=errores,
                            ahora=AHORA) == 2
    assert 'codigo="entrada_ilegible"' in errores.getvalue()


def test_autoprueba_sana_sale_0_y_lista_vacia(monkeypatch, tmp_path):
    assert _llamar(monkeypatch, tmp_path, {}, argv_extra=("autoprueba",))[:2] == (0, "[]")


@pytest.mark.parametrize("argv", [[], ["a", "b"], ["a", "autoprueba", "c"]])
def test_argumentos_invalidos_bloquean(argv):
    assert gancho.principal(argv, entrada=io.StringIO("{}"), salida=io.StringIO(), errores=io.StringIO()) == 2


# --- el envoltorio sh, con una biblioteca falsa ---------------------------------

def _lib_falsa(tmp_path, cuerpo):
    lib = tmp_path / "lib"
    paquete = lib / "jax" / "ejecutor" / "contratos"
    paquete.mkdir(parents=True)
    for d in (lib / "jax", lib / "jax" / "ejecutor", paquete):
        (d / "__init__.py").write_text("")
    (paquete / "gancho.py").write_text(f"def principal(argv):\n{cuerpo}\n")
    guion = tmp_path / "gancho.sh"
    guion.write_text(instalacion.renderizar_gancho(str(lib), 1))
    guion.chmod(0o755)
    return guion


@pytest.mark.parametrize("cuerpo, esperado", [
    ("    return 0", 0),
    ("    return 2", 2),
    ("    return 1", 2),
    ("    return 3", 2),
    ("    raise RuntimeError('x')", 2),
    ("    import sys; sys.exit(127)", 2),
])
def test_el_envoltorio_convierte_todo_lo_que_no_es_0_en_2(tmp_path, cuerpo, esperado):
    guion = _lib_falsa(tmp_path, cuerpo)
    r = subprocess.run([str(guion), "x"], input=b"{}", capture_output=True, timeout=20)
    assert r.returncode == esperado


def test_el_envoltorio_corta_por_tiempo_y_bloquea(tmp_path):
    guion = _lib_falsa(tmp_path, "    import time; time.sleep(30); return 0")
    inicio = time.monotonic()
    r = subprocess.run([str(guion), "x"], input=b"{}", capture_output=True, timeout=20)
    assert r.returncode == 2
    assert time.monotonic() - inicio < 10


def test_el_envoltorio_ignora_el_arranque_que_controla_la_cuenta(tmp_path):
    """La cuenta controla su entorno. Un usercustomize que sale con 0 antes de
    evaluar nada sería un permiso universal; `python3 -I` no lo carga."""
    guion = _lib_falsa(tmp_path, "    return 2")
    version = subprocess.run(["/usr/bin/python3", "-c", "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')"],
                             capture_output=True, text=True, timeout=20).stdout.strip()
    sitio = tmp_path / "base" / "lib" / f"python{version}" / "site-packages"
    sitio.mkdir(parents=True)
    (sitio / "usercustomize.py").write_text("import os\nos._exit(0)\n")
    env = {**os.environ, "PYTHONUSERBASE": str(tmp_path / "base"), "PYTHONPATH": str(sitio)}
    control = subprocess.run(["/usr/bin/python3", "-c", "print('vivo')"], capture_output=True, timeout=20, env=env)
    assert (control.returncode, control.stdout) == (0, b""), "la trampa no se dispara: este test no mediría nada"
    r = subprocess.run([str(guion), "x"], input=b"{}", capture_output=True, timeout=20, env=env)
    assert r.returncode == 2
```

```python
# tests/test_ejecutor_contratos_instalacion.py
"""Lo que se instala en /opt/ejecutor/lib: sólo biblioteca estándar, render
determinista del gancho y del managed-settings de la jaula."""
import ast
import json
import sys
from pathlib import Path

import pytest

from jax.ejecutor.contratos import instalacion

RAIZ = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("rel", instalacion.INSTALABLES)
def test_lo_instalable_es_solo_biblioteca_estandar(rel):
    arbol = ast.parse((RAIZ / rel).read_text(encoding="utf-8"))
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Import):
            modulos = [a.name for a in nodo.names]
        elif isinstance(nodo, ast.ImportFrom) and nodo.module == "jax.ejecutor.contratos":
            for alias in nodo.names:
                assert f"jax/ejecutor/contratos/{alias.name}.py" in instalacion.INSTALABLES, (rel, alias.name)
            continue
        elif isinstance(nodo, ast.ImportFrom):
            modulos = [nodo.module or ""]
        else:
            continue
        for m in modulos:
            if m.startswith("jax.ejecutor.contratos."):
                assert f"jax/ejecutor/contratos/{m.rsplit('.', 1)[1]}.py" in instalacion.INSTALABLES, (rel, m)
            else:
                assert m == "__future__" or m.split(".")[0] in sys.stdlib_module_names, (rel, m)


def test_render_del_gancho():
    texto = instalacion.renderizar_gancho("/opt/ejecutor/lib", 10)
    assert "LIB='/opt/ejecutor/lib'" in texto and "timeout -s KILL '10'" in texto
    assert "@LIB@" not in texto and "@TOPE_S@" not in texto


@pytest.mark.parametrize("lib, tope", [("relativa/lib", 10), ("/opt/con espacio", 10), ("/opt/x'y", 10),
                                       ("/opt/ejecutor/lib", 0), ("/opt/ejecutor/lib", 61)])
def test_render_rechaza_valores_peligrosos(lib, tope):
    with pytest.raises(ValueError):
        instalacion.renderizar_gancho(lib, tope)


def test_managed_settings_de_la_jaula():
    doc = json.loads(instalacion.renderizar_managed_settings("/opt/ejecutor/lib", "/etc/jax-ejecutor/politica.json", 10))
    (grupo,) = doc["hooks"]["PreToolUse"]
    assert grupo["matcher"] == "*"
    assert grupo["hooks"] == [{"type": "command", "timeout": 15,
                               "command": "/opt/ejecutor/lib/gancho.sh /etc/jax-ejecutor/politica.json"}]
    assert doc["allowManagedHooksOnly"] is True and doc["disableAllHooks"] is False


def test_manifiesto_es_estable():
    assert instalacion.manifiesto({"b": b"2", "a": b"1"}) == (
        "6b86b273ff34fce19d6b804eff5a3f5747ada4eaa22f1d49c01e52ddb7875b4b  a\n"
        "d4735e3a265e16eee03f59718b9b5d03019c07d8b6c51f90da3a666eec13ab35  b\n")
```

- [ ] **Step 2: Correr y ver que fallan**

Run: `cd /home/fruiz/worktrees/jax-sp1-c1 && PYTHONPATH=.:las_manos python -m pytest tests/test_ejecutor_contratos_gancho.py tests/test_ejecutor_contratos_instalacion.py -v`
Expected: FAIL en la colección (`cannot import name 'gancho'`).

- [ ] **Step 3: Implementar**

```python
# jax/ejecutor/contratos/gancho.py
"""Gancho PreToolUse del Ejecutor (C1 y C2). Lo corre `axioma` dentro de la jaula,
en cada llamada de herramienta, a través de `gancho.sh`.

Contrato con Claude Code (documentación oficial, 2026-09-17): exit 2 BLOQUEA y
stderr vuelve al modelo; exit 1, crash, 127 o timeout NO bloquean. Por eso este
módulo sólo devuelve 0 cuando `evaluar` dice PERMITIDO, y `gancho.sh` convierte
todo lo demás en 2.

Modos:
  gancho.py <politica>             el gancho (stdin: el evento)
  gancho.py <politica> autoprueba  ejemplos de todas las reglas; stdout JSON; 0 si sana

stderr en formato de máquina neutro: el modelo lo lee, el frontend lo rotula.
Sólo biblioteca estándar.
"""
from __future__ import annotations

import dataclasses
import json
import os
import sys
from datetime import datetime, timezone

from jax.ejecutor.contratos import formato, politica


def _contrato(codigo: str) -> str:
    return "c2" if codigo == politica.DESTRUCTIVO_SIN_RESPALDO else "c1"


def principal(argv, *, entrada=None, salida=None, errores=None, ahora=None) -> int:
    entrada = sys.stdin if entrada is None else entrada
    salida = sys.stdout if salida is None else salida
    errores = sys.stderr if errores is None else errores
    if not 1 <= len(argv) <= 2 or (len(argv) == 2 and argv[1] != "autoprueba"):
        errores.write(formato.campos((("contrato", "c1"), ("codigo", "argumentos_invalidos"))) + "\n")
        return 2
    evento = None
    if len(argv) == 1:
        # stdin ANTES que la política: un stdin que no cierra cae en el tope de gancho.sh.
        try:
            evento = json.loads(entrada.read())
        except ValueError:
            evento = None
        if not isinstance(evento, dict):
            errores.write(formato.campos((("contrato", "c1"), ("codigo", politica.ENTRADA_ILEGIBLE))) + "\n")
            return 2
    try:
        p = politica.cargar(argv[0], uid_de_la_cuenta=os.getuid())
    except politica.PoliticaIlegible as exc:
        errores.write(formato.campos((("contrato", "c1"), ("codigo", politica.POLITICA_ILEGIBLE),
                                      ("motivo", exc.codigo)) + tuple(exc.datos)) + "\n")
        return 2
    if evento is None:
        fallos = politica.autoprueba(p)
        salida.write(json.dumps([dataclasses.asdict(f) for f in fallos], ensure_ascii=True, separators=(",", ":")))
        return 0 if not fallos else 2
    decision = politica.evaluar(p, evento.get("tool_name"), evento.get("tool_input"),
                                ahora or datetime.now(timezone.utc))
    if decision.permitir:
        return 0
    errores.write(formato.campos((("contrato", _contrato(decision.codigo)), ("codigo", decision.codigo),
                                  ("regla", decision.regla), ("hosts", decision.hosts))) + "\n")
    return 2
```

```sh
#!/bin/sh
# ops/ejecutor/gancho.sh.plantilla — gancho PreToolUse del Ejecutor (C1 y C2).
# Lo renderiza jax/ejecutor/contratos/instalacion.py; no se edita a mano.
#
# FAIL-CLOSED. Claude Code NO bloquea si el gancho sale con 1 o 127, se cae o vence
# su timeout: la herramienta sigue (documentación oficial, 2026-09-17). Por eso todo
# lo que no sea un 0 explícito sale como 2, y `timeout -s KILL` corta ANTES que el
# timeout del gancho en managed-settings.json (tope + 5 s).
# `-I`: sin PYTHONPATH, sin site de usuario, sin el directorio actual en sys.path:
# la cuenta no puede poner delante un módulo propio con el mismo nombre.
LIB='@LIB@'
/usr/bin/timeout -s KILL '@TOPE_S@' /usr/bin/python3 -I -c 'import sys; sys.path.insert(0, sys.argv[1]); from jax.ejecutor.contratos.gancho import principal; sys.exit(principal(sys.argv[2:]))' "$LIB" "$@"
rc=$?
if [ "$rc" -eq 0 ]; then exit 0; fi
exit 2
```

```python
# jax/ejecutor/contratos/instalacion.py
"""Qué se instala en /opt/ejecutor/lib y cómo se renderiza (C1/C2; C4 agrega el freno).

`INSTALABLES` es la única lista: la usan el instalador, el test de biblioteca
estándar y el arranque (plan 6) para comparar sha256 instalado contra repo.

`python -m jax.ejecutor.contratos.instalacion <etapa>` escribe en <etapa>:
gancho.sh, managed-settings.json, settings-usuario.json, instalables.txt y
manifiesto.sha256, leyendo JAX_EJECUTOR_LIB, JAX_EJECUTOR_POLITICA y
JAX_EJECUTOR_GANCHO_TOPE_S (sin defaults).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[3]
PLANTILLA_GANCHO = RAIZ / "ops" / "ejecutor" / "gancho.sh.plantilla"

INSTALABLES = (
    "jax/__init__.py",
    "jax/ejecutor/__init__.py",
    "jax/ejecutor/contratos/__init__.py",
    "jax/ejecutor/contratos/formato.py",
    "jax/ejecutor/contratos/destinos.py",
    "jax/ejecutor/contratos/politica.py",
    "jax/ejecutor/contratos/gancho.py",
)

# settings de usuario de la cuenta DENTRO de la jaula: vacíos y de solo lectura.
SETTINGS_USUARIO = "{}\n"

_RUTA_SEGURA = re.compile(r"^/[A-Za-z0-9_./-]+$")
_MARGEN_TIMEOUT_S = 5


def _ruta(valor: str) -> str:
    if not _RUTA_SEGURA.match(valor or "") or "/../" in valor:
        raise ValueError("ruta_insegura")
    return valor


def _tope(valor: int) -> int:
    if isinstance(valor, bool) or not isinstance(valor, int) or not 1 <= valor <= 60:
        raise ValueError("tope_invalido")
    return valor


def renderizar_gancho(lib: str, tope_s: int) -> str:
    return (PLANTILLA_GANCHO.read_text(encoding="utf-8")
            .replace("@LIB@", _ruta(lib)).replace("@TOPE_S@", str(_tope(tope_s))))


def renderizar_managed_settings(lib: str, politica: str, tope_s: int) -> str:
    doc = {
        "allowManagedHooksOnly": True,
        "disableAllHooks": False,
        "hooks": {"PreToolUse": [{"matcher": "*", "hooks": [{
            "type": "command",
            "command": f"{_ruta(lib)}/gancho.sh {_ruta(politica)}",
            "timeout": _tope(tope_s) + _MARGEN_TIMEOUT_S,
        }]}]},
    }
    return json.dumps(doc, indent=2, sort_keys=True) + "\n"


def manifiesto(archivos: dict) -> str:
    return "".join(f"{hashlib.sha256(archivos[n]).hexdigest()}  {n}\n" for n in sorted(archivos))


def principal(argv, env=None) -> int:
    env = os.environ if env is None else env
    etapa = Path(argv[0])
    lib = env["JAX_EJECUTOR_LIB"]
    tope = int(env["JAX_EJECUTOR_GANCHO_TOPE_S"])
    renderizados = {
        "gancho.sh": renderizar_gancho(lib, tope).encode(),
        "managed-settings.json": renderizar_managed_settings(lib, env["JAX_EJECUTOR_POLITICA"], tope).encode(),
        "settings-usuario.json": SETTINGS_USUARIO.encode(),
    }
    for nombre, contenido in renderizados.items():
        (etapa / nombre).write_bytes(contenido)
    (etapa / "instalables.txt").write_text("".join(f"{r}\n" for r in INSTALABLES), encoding="utf-8")
    todos = {**renderizados, **{r: (RAIZ / r).read_bytes() for r in INSTALABLES}}
    (etapa / "manifiesto.sha256").write_text(manifiesto(todos), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(principal(sys.argv[1:]))
```

- [ ] **Step 4: Correr y ver que pasan**

Run: `cd /home/fruiz/worktrees/jax-sp1-c1 && PYTHONPATH=.:las_manos python -m pytest tests/test_ejecutor_contratos_gancho.py tests/test_ejecutor_contratos_instalacion.py -v && python3 policy/tests/test_claude_subprocess_solo_via_sandbox.py`
Expected: todos PASS. La guardia de subprocesos sigue verde: `test_ejecutor_contratos_gancho.py` lanza
subprocesos pero **no** menciona la palabra prohibida en ningún literal (revisar el docstring si da violación).

- [ ] **Step 5: Mutación del envoltorio (verlo fallar)**

Con backup y `cmp`: en `ops/ejecutor/gancho.sh.plantilla`, cambiar `exit 2` (la última línea) por `exit $rc`.
Expected: FAIL `test_el_envoltorio_convierte_todo_lo_que_no_es_0_en_2[    return 1-2]`, `[    return 3-2]`,
`[    raise RuntimeError('x')-2]`, `[    import sys; sys.exit(127)-2]` y `test_el_envoltorio_corta_por_tiempo_y_bloquea`
(rc 137). Restaurar y `cmp`.

- [ ] **Step 6: Commit**

```bash
git -C /home/fruiz/worktrees/jax-sp1-c1 add jax/ejecutor/contratos/gancho.py jax/ejecutor/contratos/instalacion.py ops/ejecutor/gancho.sh.plantilla tests/test_ejecutor_contratos_gancho.py tests/test_ejecutor_contratos_instalacion.py
git -C /home/fruiz/worktrees/jax-sp1-c1 commit -m "feat(ejecutor): gancho PreToolUse de C1/C2 y envoltorio que convierte todo fallo en bloqueo

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: jax · el exportador — de la DB a `politica.json`

**Files:**
- Create: `/home/fruiz/worktrees/jax-sp1-c1/jax/ejecutor/contratos/exportar.py`
- Test: `/home/fruiz/worktrees/jax-sp1-c1/tests/test_ejecutor_contratos_exportar.py`
- Test (con DB): `/home/fruiz/worktrees/jax-sp1-c1/tests/test_ejecutor_politica_db.py`

**Interfaces:**
- Consume: tablas de la Task 1; `politica.firmar`, `politica.validar`.
- Produce: `ExportacionImposible(codigo)`, `documento(hosts, reglas, respaldos, edad_c2, generada_at) -> dict`,
  `async leer(conn)`, `escribir_atomico(ruta, doc)`, `async exportar(ruta, conectar) -> dict`,
  `python -m jax.ejecutor.contratos.exportar` (lee `JAX_EJECUTOR_POLITICA` y las `JAX_DB_*` del entorno).

**Por qué valida antes de publicar:** un exportador que escribe una política que el gancho no puede leer deja al
Ejecutor bloqueado sin explicación; uno que escribe sin validar podría publicar una regla con regex rota. Se
valida con el MISMO `politica.validar` del gancho y, si no pasa, no se toca el archivo publicado y se sale ≠ 0.

- [ ] **Step 1: Escribir los tests que fallan**

```python
# tests/test_ejecutor_contratos_exportar.py
"""Exportador de la política (C1/C2): filas de la DB -> documento firmado y validado
-> escritura atómica. Sin DB: las filas se construyen a mano."""
import json
import os
import stat
from datetime import datetime

import pytest

from jax.ejecutor.contratos import exportar, politica

HOSTS = [("hall9000", "192.0.2.5", 58291, "hypervisor", 1), ("bridge", "192.0.2.20", 58291, "clientes", 0)]
_EJ = json.dumps([{"tool_name": "Bash", "tool_input": {"command": "echo ejecutor-canario-c1"}}])
_NO = json.dumps([{"tool_name": "Bash", "tool_input": {"command": "echo x"}}])
REGLAS = [
    (1, "canario_c1", "prohibido", "Bash", "command", "ejecutor-canario-c1", None, None, 1, _EJ, _NO),
    (2, "migrate", "prohibido", "Bash", "command", r"\bmigrate:fresh\b", None, "produccion,clientes", 0,
     json.dumps([{"tool_name": "Bash", "tool_input": {"command": "ssh -tt axioma@bridge 'php artisan migrate:fresh'"}}]),
     "[]"),
]
RESPALDOS = [("bridge", datetime(2026, 9, 17, 6, 0))]


def test_documento_firmado_y_legible():
    doc = exportar.documento(HOSTS, REGLAS, RESPALDOS, "86400", "2026-09-17T12:00:00+00:00")
    p = politica.validar(doc)
    assert [r.codigo for r in p.reglas] == ["canario_c1", "migrate"]
    assert p.reglas[1].ambito_roles == frozenset({"produccion", "clientes"})
    assert doc["respaldos"] == {"bridge": "2026-09-17T06:00:00+00:00"}
    assert politica.autoprueba(p) == ()


@pytest.mark.parametrize("edad", [None, "", "cero", "0", "-5"])
def test_sin_edad_de_c2_no_se_exporta(edad):
    with pytest.raises(exportar.ExportacionImposible) as e:
        exportar.documento(HOSTS, REGLAS, RESPALDOS, edad, "2026-09-17T12:00:00+00:00")
    assert e.value.codigo == "c2_edad_max_invalida"


def test_una_regla_rota_no_se_publica():
    rota = REGLAS + [(3, "rota", "destructivo", "Bash", "command", "(sin cerrar", None, None, 0, _EJ, "[]")]
    with pytest.raises(exportar.ExportacionImposible) as e:
        exportar.documento(HOSTS, rota, RESPALDOS, "86400", "2026-09-17T12:00:00+00:00")
    assert e.value.codigo == "politica_invalida"


def test_escritura_atomica_con_permisos(tmp_path):
    ruta = tmp_path / "politica.json"
    ruta.write_text("vieja")
    doc = exportar.documento(HOSTS, REGLAS, RESPALDOS, "86400", "2026-09-17T12:00:00+00:00")
    exportar.escribir_atomico(ruta, doc)
    assert json.loads(ruta.read_text()) == doc
    assert stat.S_IMODE(os.stat(ruta).st_mode) == 0o640
    assert [p.name for p in tmp_path.iterdir()] == ["politica.json"]
```

```python
# tests/test_ejecutor_politica_db.py
"""La política que sale de la DB REAL (esquema y semilla de las migraciones de
jax-platform, que este job corre antes) es legible y TODAS sus reglas pasan su
autoprueba. Es donde se prueba la semilla de jax-platform con el evaluador de jax,
sin copiar ninguno de los dos."""
import asyncio

import pytest

from jacobs.store import get_conn
from jax.ejecutor.contratos import exportar, politica

INVENTARIO = [("hall9000", "192.0.2.5", "hypervisor", 1), ("atemai", "192.0.2.11", "desarrollo", 0),
              ("prod", "192.0.2.10", "produccion", 0), ("bridge", "192.0.2.20", "clientes", 0)]


async def _con_inventario(accion):
    conn = await get_conn()
    try:
        async with conn.cursor() as cur:
            for nombre, ip, rol, local in INVENTARIO:
                await cur.execute("INSERT IGNORE INTO ejecutor_host (nombre, ip, puerto, rol, es_local) "
                                  "VALUES (%s, %s, 58291, %s, %s)", (nombre, ip, rol, local))
            # Con la tabla vacía el optimizador puede no mostrar el índice: se mide con filas.
            for k in range(200):
                await cur.execute(
                    "INSERT INTO ejecutor_punto_restauracion (host_nombre, referencia, metodo, "
                    "restaurado_y_verificado_at, verificado_por, evidencia) VALUES (%s, %s, 'prueba', "
                    "UTC_TIMESTAMP() - INTERVAL %s MINUTE, 'test', 'test')",
                    (INVENTARIO[k % 4][0], f"prueba-{k}", k))
        await conn.commit()
        return await accion(conn)
    finally:
        async with conn.cursor() as cur:
            await cur.execute("DELETE FROM ejecutor_punto_restauracion WHERE referencia LIKE %s", ("prueba-%",))
            await cur.execute("DELETE FROM ejecutor_host WHERE ip LIKE %s", ("192.0.2.%",))
        await conn.commit()
        conn.close()


def test_la_semilla_real_exporta_legible_y_pasa_su_autoprueba():
    async def accion(conn):
        filas = await exportar.leer(conn)
        return exportar.documento(*filas, "2026-09-17T12:00:00+00:00")

    doc = asyncio.run(_con_inventario(accion))
    p = politica.validar(doc)
    assert len(p.reglas) == 14
    assert politica.autoprueba(p) == ()


def test_la_consulta_de_respaldos_usa_su_indice():
    async def accion(conn):
        async with conn.cursor() as cur:
            await cur.execute("EXPLAIN " + exportar.SQL_RESPALDOS)
            return await cur.fetchall()

    filas = asyncio.run(_con_inventario(accion))
    assert any("idx_ejecutor_punto_host_fecha" in str(f) for f in filas), filas
```

- [ ] **Step 2: Correr y ver que fallan**

Run: `cd /home/fruiz/worktrees/jax-sp1-c1 && PYTHONPATH=.:las_manos python -m pytest tests/test_ejecutor_contratos_exportar.py -v`
Expected: FAIL (`cannot import name 'exportar'`). El test con DB se corre en CI (job `jacobs-gobernanza-db`), o
local contra `jax_memory_test` con `JAX_DB_NAME=jax_memory_test` **explícito en el mismo comando**.

- [ ] **Step 3: Implementar**

```python
# jax/ejecutor/contratos/exportar.py
"""De la DB a /etc/jax-ejecutor/politica.json (C1/C2). Corre como `fruiz`.

Valida con el MISMO `politica.validar` del gancho antes de publicar: si no pasa,
no toca el archivo publicado y sale ≠ 0. Escritura atómica: temporal en el mismo
directorio (hereda el grupo `axioma` por el setgid del directorio), fsync, rename,
fsync del directorio.

Lee la DB de PRODUCCIÓN cuando corre con /etc/jax/.env: sólo SELECT.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from jax.ejecutor.contratos import politica

SQL_HOSTS = "SELECT nombre, ip, puerto, rol, es_local FROM ejecutor_host WHERE activo = 1 ORDER BY nombre"
SQL_REGLAS = ("SELECT id, codigo, tipo, herramientas, campo, patron, ambito_host, ambito_roles, es_canario, "
              "ejemplos_coincide, ejemplos_no_coincide FROM ejecutor_regla WHERE activa = 1 ORDER BY id")
SQL_RESPALDOS = ("SELECT host_nombre, MAX(restaurado_y_verificado_at) FROM ejecutor_punto_restauracion "
                 "GROUP BY host_nombre")
SQL_EDAD = "SELECT config_value FROM axioma_config WHERE config_key = 'ejecutor.c2_edad_max_s'"


class ExportacionImposible(RuntimeError):
    def __init__(self, codigo: str):
        super().__init__(codigo)
        self.codigo = codigo


def _roles(v) -> list:
    if not v:
        return []
    return sorted(v) if isinstance(v, (set, frozenset)) else sorted(v.split(","))


def documento(hosts, reglas, respaldos, edad_c2, generada_at: str) -> dict:
    try:
        edad = int(edad_c2)
    except (TypeError, ValueError):
        raise ExportacionImposible("c2_edad_max_invalida") from None
    if edad <= 0:
        raise ExportacionImposible("c2_edad_max_invalida")
    doc = {
        "version": politica.VERSION,
        "generada_at": generada_at,
        "hosts": [{"nombre": n, "ip": ip, "puerto": int(pt), "rol": rol, "es_local": bool(loc)}
                  for n, ip, pt, rol, loc in hosts],
        "reglas": [{"id": int(i), "codigo": c, "tipo": t, "herramientas": h, "campo": ca, "patron": pa,
                    "ambito_hosts": [ah] if ah else [], "ambito_roles": _roles(ar), "es_canario": bool(ec),
                    "ejemplos_coincide": json.loads(ej), "ejemplos_no_coincide": json.loads(no)}
                   for i, c, t, h, ca, pa, ah, ar, ec, ej, no in reglas],
        "respaldos": {n: m.replace(tzinfo=timezone.utc).isoformat() for n, m in respaldos if m is not None},
        "c2_edad_max_s": edad,
    }
    firmado = politica.firmar(doc)
    try:
        politica.validar(firmado)
    except politica.PoliticaIlegible as exc:
        raise ExportacionImposible("politica_invalida") from exc
    return firmado


async def leer(conn):
    async with conn.cursor() as cur:
        await cur.execute(SQL_HOSTS)
        hosts = await cur.fetchall()
        await cur.execute(SQL_REGLAS)
        reglas = await cur.fetchall()
        await cur.execute(SQL_RESPALDOS)
        respaldos = await cur.fetchall()
        await cur.execute(SQL_EDAD)
        fila = await cur.fetchone()
    return hosts, reglas, respaldos, (fila[0] if fila else None)


def escribir_atomico(ruta: Path, doc: dict) -> None:
    ruta = Path(ruta)
    datos = json.dumps(doc, ensure_ascii=True, sort_keys=True, indent=1).encode() + b"\n"
    fd, temporal = tempfile.mkstemp(dir=ruta.parent, prefix=".politica-", suffix=".tmp")
    try:
        os.fchmod(fd, 0o640)
        os.write(fd, datos)
        os.fsync(fd)
    finally:
        os.close(fd)
    try:
        os.replace(temporal, ruta)
    except OSError:
        os.unlink(temporal)
        raise
    dir_fd = os.open(ruta.parent, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


async def exportar(ruta: Path, conectar) -> dict:
    conn = await conectar()
    try:
        filas = await leer(conn)
    finally:
        conn.close()
    doc = documento(*filas, datetime.now(timezone.utc).isoformat())
    await asyncio.to_thread(escribir_atomico, ruta, doc)
    return doc


def principal() -> int:
    from jacobs.store import get_conn
    ruta = os.environ.get("JAX_EJECUTOR_POLITICA", "").strip()
    if not ruta.startswith("/"):
        sys.stderr.write("codigo=\"config_falta\" variable=\"JAX_EJECUTOR_POLITICA\"\n")
        return 2
    try:
        doc = asyncio.run(exportar(Path(ruta), get_conn))
    except ExportacionImposible as exc:
        sys.stderr.write(f"codigo=\"{exc.codigo}\"\n")
        return 2
    sys.stdout.write(f"sha256=\"{doc['sha256']}\" reglas={len(doc['reglas'])} hosts={len(doc['hosts'])}\n")
    return 0


if __name__ == "__main__":
    sys.exit(principal())
```

- [ ] **Step 4: Correr y ver que pasan**

Run: `cd /home/fruiz/worktrees/jax-sp1-c1 && PYTHONPATH=.:las_manos python -m pytest tests/test_ejecutor_contratos_exportar.py -v`
Expected: 8 PASS.

- [ ] **Step 5: El test con DB en su job**

En `.github/workflows/policy.yml`, job `jacobs-gobernanza-db`, agregar `tests/test_ejecutor_politica_db.py` en
**las dos** listas (la del `pytest -v` y la del piso). El piso `27` pasa a lo que cuente el runner (esperado
`29`), con su línea de historia: `# 27 -> 29 el 2026-09-17 (Ejecutor SP1 plan 1): test_ejecutor_politica_db.py
(+2) -- la semilla de reglas de jax-platform exporta legible y pasa su autoprueba con el evaluador de jax; EXPLAIN
de respaldos. Requiere jax-platform con las tablas del Ejecutor en master.` **Orden:** el PR de jax-platform
(Task 1) se mergea antes que este.

- [ ] **Step 6: Commit**

```bash
git -C /home/fruiz/worktrees/jax-sp1-c1 add jax/ejecutor/contratos/exportar.py tests/test_ejecutor_contratos_exportar.py tests/test_ejecutor_politica_db.py .github/workflows/policy.yml
git -C /home/fruiz/worktrees/jax-sp1-c1 commit -m "feat(ejecutor): exportador de la política, valida con el evaluador del gancho antes de publicar

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: jax · entrar a la cuenta y la jaula superpuesta

**Files:**
- Create: `/home/fruiz/worktrees/jax-sp1-c1/jax/ejecutor/contratos/cuenta_axioma.py`
- Modify: `/home/fruiz/worktrees/jax-sp1-c1/policy/tests/test_claude_subprocess_solo_via_sandbox.py` (dict `_AISLADO_POR_CUENTA`)
- Test: `/home/fruiz/worktrees/jax-sp1-c1/tests/test_ejecutor_contratos_cuenta_axioma.py`

**Interfaces:**
- Consume: nada de tasks anteriores.
- Produce: `Cuenta`, `CuentaSinConfigurar`, `cuenta_desde_entorno`, `ssh_a_la_cuenta`, `remoto_claude`,
  `correr_en_la_cuenta` (ver Interfaces del plan). **SP2 reemplaza `_jaula` por el perfil `ejecutor` de
  `hyde_sandbox` y conserva sus dos montajes de solo lectura** (managed-settings y settings de la cuenta).

**Por qué `correr_en_la_cuenta` no acepta un argv libre:** la guardia `no-naked-claude-subprocess` exime a los
archivos declarados que lanzan por `ssh` a `usuario@host`. Si esta función aceptara cualquier argv, otro módulo
podría pasarle `["claude", ...]` sin que la guardia lo vea. Sólo acepta un comando **remoto**: siempre sale por la
cuenta.

- [ ] **Step 1: Escribir los tests que fallan**

```python
# tests/test_ejecutor_contratos_cuenta_axioma.py
"""Cómo se entra a la cuenta del Ejecutor: todo desde el entorno, sin defaults; la
llave del cerebro nunca en argv; el lanzamiento siempre dentro de la jaula
superpuesta con los settings de solo lectura."""
import shlex
from pathlib import Path

import pytest

from jax.ejecutor.contratos import cuenta_axioma as CA

ENV = {
    "JAX_EJECUTOR_CUENTA": "axioma", "JAX_EJECUTOR_SSH_PUERTO": "58291",
    "JAX_EJECUTOR_CONTROLADOR_LLAVE": "/home/fruiz/.ssh/id_ejecutor_controlador",
    "JAX_EJECUTOR_NODE_BIN": "/opt/ejecutor/node-v24.16.0/bin", "JAX_EJECUTOR_LIB": "/opt/ejecutor/lib",
    "JAX_EJECUTOR_POLITICA": "/etc/jax-ejecutor/politica.json",
}


def test_cuenta_desde_el_entorno():
    c = CA.cuenta_desde_entorno(ENV)
    assert c == CA.Cuenta("axioma", 58291, Path("/home/fruiz/.ssh/id_ejecutor_controlador"),
                          Path("/opt/ejecutor/node-v24.16.0/bin"), Path("/opt/ejecutor/lib"),
                          Path("/etc/jax-ejecutor/politica.json"))


@pytest.mark.parametrize("variable", sorted(ENV))
def test_sin_una_variable_no_se_entra(variable):
    with pytest.raises(CA.CuentaSinConfigurar):
        CA.cuenta_desde_entorno({k: v for k, v in ENV.items() if k != variable})


@pytest.mark.parametrize("variable, valor", [("JAX_EJECUTOR_SSH_PUERTO", "x"), ("JAX_EJECUTOR_LIB", "relativa"),
                                             ("JAX_EJECUTOR_CUENTA", "root; rm")])
def test_valores_invalidos_no_entran(variable, valor):
    with pytest.raises(CA.CuentaSinConfigurar):
        CA.cuenta_desde_entorno({**ENV, variable: valor})


def test_ssh_a_la_cuenta():
    argv = CA.ssh_a_la_cuenta(CA.cuenta_desde_entorno(ENV), "uptime")
    assert argv == ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", "-p", "58291",
                    "-i", "/home/fruiz/.ssh/id_ejecutor_controlador", "axioma@127.0.0.1", "uptime"]


def test_remoto_claude_va_en_la_jaula_y_sin_la_llave_en_argv():
    remoto = CA.remoto_claude(CA.cuenta_desde_entorno(ENV), base_url="http://127.0.0.1:18436", modelo="canario",
                              prompt="hola 'mundo'")
    assert remoto.startswith("read -r K; cd ~ && env ")
    assert 'ANTHROPIC_AUTH_TOKEN="$K"' in remoto
    palabras = shlex.split(remoto.split(" && ", 1)[1].replace('"$K"', "K"))
    i = palabras.index("bwrap")
    jaula = palabras[i:palabras.index("--", i)]
    assert ["--tmpfs", "/etc/claude-code"] == jaula[jaula.index("--tmpfs"):jaula.index("--tmpfs") + 2]
    assert ["--ro-bind", "/opt/ejecutor/lib/managed-settings.json", "/etc/claude-code/managed-settings.json"] in \
        [jaula[k:k + 3] for k in range(len(jaula))]
    assert "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1" in palabras
    assert palabras[palabras.index("--", i) + 1] == "/opt/ejecutor/node-v24.16.0/bin/claude"
    assert "hola 'mundo'" in palabras
```

- [ ] **Step 2: Correr y ver que fallan**

Run: `cd /home/fruiz/worktrees/jax-sp1-c1 && PYTHONPATH=.:las_manos python -m pytest tests/test_ejecutor_contratos_cuenta_axioma.py -v`
Expected: FAIL (`cannot import name 'cuenta_axioma'`).

- [ ] **Step 3: Implementar**

```python
# jax/ejecutor/contratos/cuenta_axioma.py
"""Entrar a la cuenta del Ejecutor desde fuera (C1, C3, C4, C6).

`fruiz` entra a `axioma@127.0.0.1` con la llave del controlador: el kernel acota
las dos puntas (Fase 0, decisión de Fernando 2026-09-15). Todo sale de
JAX_EJECUTOR_*; sin una variable, no se entra.

El `claude` del Ejecutor se lanza SIEMPRE dentro de una jaula bwrap SUPERPUESTA
(`--dev-bind / /`): el mismo sistema que ve `axioma`, más montajes de solo lectura
que la cuenta no puede quitar:
- /etc/claude-code/managed-settings.json → el gancho de C1/C2. En el host ese
  directorio está vacío: el gancho no alcanza a nadie más (las sesiones de Fernando
  en hall9000 no lo ven).
- ~/.claude/settings.json y settings.local.json → `{}`: la cuenta no puede apagar
  los ganchos con `disableAllHooks` (documentación oficial: en cualquier nivel).
SP2 reemplaza esta jaula por el perfil `ejecutor` de hyde_sandbox y conserva estos
montajes.

La llave del cerebro viaja por stdin (`read -r K`), nunca en argv: `ps` la vería.
"""
from __future__ import annotations

import asyncio
import os
import re
import shlex
from dataclasses import dataclass
from pathlib import Path

_NOMBRE = re.compile(r"^[a-z_][a-z0-9_-]{0,31}$")


class CuentaSinConfigurar(RuntimeError):
    """`args[0]` es la variable que falta o no vale."""


@dataclass(frozen=True)
class Cuenta:
    nombre: str
    puerto: int
    llave: Path
    node_bin: Path
    lib: Path
    politica: Path


def cuenta_desde_entorno(env=None) -> Cuenta:
    env = os.environ if env is None else env

    def obligatoria(nombre):
        valor = env.get(nombre, "").strip()
        if not valor:
            raise CuentaSinConfigurar(nombre)
        return valor

    def ruta(nombre):
        valor = obligatoria(nombre)
        if not valor.startswith("/"):
            raise CuentaSinConfigurar(nombre)
        return Path(valor)

    nombre = obligatoria("JAX_EJECUTOR_CUENTA")
    if not _NOMBRE.match(nombre):
        raise CuentaSinConfigurar("JAX_EJECUTOR_CUENTA")
    puerto = obligatoria("JAX_EJECUTOR_SSH_PUERTO")
    if not puerto.isdigit() or not 0 < int(puerto) < 65536:
        raise CuentaSinConfigurar("JAX_EJECUTOR_SSH_PUERTO")
    return Cuenta(nombre, int(puerto), ruta("JAX_EJECUTOR_CONTROLADOR_LLAVE"), ruta("JAX_EJECUTOR_NODE_BIN"),
                  ruta("JAX_EJECUTOR_LIB"), ruta("JAX_EJECUTOR_POLITICA"))


def ssh_a_la_cuenta(c: Cuenta, remoto: str) -> list[str]:
    return ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", "-p", str(c.puerto),
            "-i", str(c.llave), f"{c.nombre}@127.0.0.1", remoto]


def _jaula(c: Cuenta) -> str:
    q = shlex.quote
    return " ".join([
        "bwrap", "--dev-bind", "/", "/", "--die-with-parent",
        "--tmpfs", "/etc/claude-code",
        "--ro-bind", q(str(c.lib / "managed-settings.json")), "/etc/claude-code/managed-settings.json",
        "--ro-bind", q(str(c.lib / "settings-usuario.json")), '"$HOME/.claude/settings.json"',
        "--ro-bind", q(str(c.lib / "settings-usuario.json")), '"$HOME/.claude/settings.local.json"',
        "--",
    ])


def remoto_claude(c: Cuenta, *, base_url: str, modelo: str, prompt: str, herramientas: str = "Bash,Read") -> str:
    q = shlex.quote
    entorno = " ".join([
        f"ANTHROPIC_BASE_URL={q(base_url)}", 'ANTHROPIC_AUTH_TOKEN="$K"', f"PATH={q(str(c.node_bin))}:/usr/bin:/bin",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1", "DISABLE_AUTOUPDATER=1", "DISABLE_TELEMETRY=1",
        "DISABLE_ERROR_REPORTING=1",
    ])
    claude = " ".join(q(a) for a in [str(c.node_bin / "claude"), "-p", prompt, "--output-format", "stream-json",
                                      "--verbose", "--model", modelo, "--allowedTools", herramientas])
    return f"read -r K; cd ~ && env {entorno} {_jaula(c)} {claude}"


async def correr_en_la_cuenta(c: Cuenta, remoto: str, *, entrada: bytes = b"", tope_s: float):
    proc = await asyncio.create_subprocess_exec(
        *ssh_a_la_cuenta(c, remoto), stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE, start_new_session=True)
    try:
        salida, errores = await asyncio.wait_for(proc.communicate(entrada), tope_s)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        proc.kill()
        await proc.wait()
        raise
    return proc.returncode, salida, errores
```

En `policy/tests/test_claude_subprocess_solo_via_sandbox.py`, dentro de `_AISLADO_POR_CUENTA`:

```python
    "jax/ejecutor/contratos/cuenta_axioma.py":
        "entra por ssh a la cuenta axioma (sin sudo, HOME propio) y lanza el claude del "
        "Ejecutor dentro de una jaula bwrap superpuesta con el gancho de C1/C2 "
        "(Ejecutor SP1 plan 1, 2026-09-17)",
```

- [ ] **Step 4: Correr y ver que pasan**

Run: `cd /home/fruiz/worktrees/jax-sp1-c1 && PYTHONPATH=.:las_manos python -m pytest tests/test_ejecutor_contratos_cuenta_axioma.py -v && python3 policy/tests/test_claude_subprocess_solo_via_sandbox.py && python3 policy/tests/test_no_fail_open_except.py`
Expected: PASS; las dos guardias verdes. **Verlas fallar:** borrar la entrada nueva de `_AISLADO_POR_CUENTA` (con
backup) → la guardia de subprocesos da ROJO nombrando `cuenta_axioma.py`; restaurar y `cmp`.

- [ ] **Step 5: Commit**

```bash
git -C /home/fruiz/worktrees/jax-sp1-c1 add jax/ejecutor/contratos/cuenta_axioma.py policy/tests/test_claude_subprocess_solo_via_sandbox.py tests/test_ejecutor_contratos_cuenta_axioma.py
git -C /home/fruiz/worktrees/jax-sp1-c1 commit -m "feat(ejecutor): entrar a la cuenta axioma y lanzar dentro de la jaula superpuesta

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: jax · el upstream falso y `verificar_c1`

**Files:**
- Create: `/home/fruiz/worktrees/jax-sp1-c1/jax/ejecutor/contratos/canario_upstream.py`
- Create: `/home/fruiz/worktrees/jax-sp1-c1/jax/ejecutor/contratos/canario_c1.py`
- Test: `/home/fruiz/worktrees/jax-sp1-c1/tests/test_ejecutor_contratos_canario_upstream.py`
- Test: `/home/fruiz/worktrees/jax-sp1-c1/tests/test_ejecutor_contratos_canario_c1.py`

**Interfaces:**
- Consume: `Fallo` (Task 2), `Cuenta`, `correr_en_la_cuenta`, `remoto_claude` (Task 6).
- Produce: `UpstreamCanario(guion, host, puerto)` (async context manager con `.resultados` y `.peticiones`),
  `guion_bash(tool_use_id, comando)`, `Resultado`; `verificar_c1(c, *, puerto_canario, correr, upstream, nonce)`.
  Códigos de `Fallo("c1", …)`: `autoprueba_fallida`, `politica_ilegible`, `canario_directo_no_bloqueado`,
  `canario_directo_sin_su_regla`, `control_directo_bloqueado`, `claude_no_llego_al_canario`,
  `canario_no_bloqueado`, `canario_sin_su_regla`, `canario_ejecutado`, `control_no_ejecutado`, `cuenta_inalcanzable`.

**Por qué el `claude` real y no sólo el gancho a mano:** el gancho a mano prueba el gancho. Lo que tiene que estar
vivo es la cadena: el binario lee el `managed-settings.json` montado en la jaula, llama al gancho, respeta el exit
2. El control (un segundo comando que DEBE correr) distingue «bloqueó el gancho» de «bloqueó cualquier otra cosa»
(permisos de Claude Code, jaula rota, todo bloqueado).

- [ ] **Step 1: Escribir los tests que fallan**

```python
# tests/test_ejecutor_contratos_canario_upstream.py
"""Upstream Anthropic falso para el canario de C1: sigue un guion de tool_use y
anota los tool_result que le devuelve el arnés. Sin modelo."""
import asyncio
import json

import httpx

from jax.ejecutor.contratos.canario_upstream import Resultado, UpstreamCanario, guion_bash

_HERRAMIENTAS = [{"name": "Bash", "input_schema": {"type": "object"}}]


def _eventos(texto):
    return [json.loads(l[len("data: "):]) for l in texto.splitlines() if l.startswith("data: ")]


def test_sigue_el_guion_y_anota_los_resultados():
    async def escenario():
        guion = [guion_bash("toolu_a", "echo a"), guion_bash("toolu_b", "echo b")]
        async with UpstreamCanario(guion, "127.0.0.1", 0) as up, httpx.AsyncClient() as cli:
            url = f"http://127.0.0.1:{up.puerto}/v1/messages"
            base = {"model": "canario", "stream": True, "tools": _HERRAMIENTAS, "max_tokens": 10}
            r1 = await cli.post(url, json={**base, "messages": [{"role": "user", "content": "x"}]})
            # Forma medida del arnés real (2026-09-17): el último mensaje es `system`.
            recordatorio = {"role": "system", "content": [{"type": "text", "text": "recordatorio"}]}
            r2 = await cli.post(url, json={**base, "messages": [{"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_a", "is_error": True,
                 "content": [{"type": "text", "text": "bloqueado"}]}]}, recordatorio]})
            r3 = await cli.post(url, json={**base, "messages": [{"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_b", "content": "b"}]}, recordatorio]})
            otra = await cli.post(url, json={"model": "chico", "messages": [], "max_tokens": 5})
            hola = await cli.head(f"http://127.0.0.1:{up.puerto}/api/hello")
            return r1.text, r2.text, r3.text, otra.json(), hola.status_code, dict(up.resultados), list(up.peticiones)

    t1, t2, t3, otra, hola, resultados, peticiones = asyncio.run(asyncio.wait_for(escenario(), 20))
    assert hola == 200
    inicio = [e for e in _eventos(t1) if e["type"] == "content_block_start"][0]
    assert inicio["content_block"] == {"type": "tool_use", "id": "toolu_a", "name": "Bash", "input": {}}
    deltas = "".join(e["delta"]["partial_json"] for e in _eventos(t1) if e["type"] == "content_block_delta")
    assert json.loads(deltas) == {"command": "echo a"}
    assert [e for e in _eventos(t2) if e["type"] == "content_block_start"][0]["content_block"]["id"] == "toolu_b"
    assert [e for e in _eventos(t3) if e["type"] == "message_delta"][0]["delta"]["stop_reason"] == "end_turn"
    assert otra["stop_reason"] == "end_turn"
    assert resultados == {"toolu_a": Resultado("toolu_a", True, "bloqueado"), "toolu_b": Resultado("toolu_b", False, "b")}
    assert [(p[0], p[3]) for p in peticiones] == [("POST", True), ("POST", True), ("POST", True), ("POST", False), ("HEAD", False)]
```

```python
# tests/test_ejecutor_contratos_canario_c1.py
"""verificar_c1 con la cuenta y el arnés falsos: qué observación da cada Fallo.
Un control que no falla no valida: cada código tiene su caso."""
import asyncio
import json
from pathlib import Path

import pytest

from jax.ejecutor.contratos import canario_c1
from jax.ejecutor.contratos.canario_upstream import Resultado
from jax.ejecutor.contratos.cuenta_axioma import Cuenta
from jax.ejecutor.contratos.fallo import Fallo

C = Cuenta("axioma", 58291, Path("/k"), Path("/opt/node/bin"), Path("/opt/ejecutor/lib"), Path("/etc/p.json"))
NONCE = "abc123"
BLOQUEO = b'contrato="c1" codigo="prohibido" regla="canario_c1" hosts=["hall9000"]\n'


class Arnes:
    """Cuenta falsa. `rotura` elige qué observación sale mal."""

    def __init__(self, rotura=None):
        self.rotura = rotura
        self.upstream = None

    async def correr(self, cuenta, remoto, *, entrada=b"", tope_s):
        if remoto.endswith("autoprueba"):
            if self.rotura == "autoprueba":
                return 2, b'[{"regla":"x","indice":0,"esperado":"coincide","obtenido":[]}]', b""
            if self.rotura == "ilegible":
                return 2, b"", b'contrato="c1" codigo="politica_ilegible" motivo="sha256_no_cuadra"\n'
            return 0, b"[]", b""
        if "gancho.sh" in remoto:
            canario = b"ejecutor-canario-c1" in entrada
            if canario:
                if self.rotura == "directo_sin_regla":
                    return 2, b"", b'contrato="c1" codigo="politica_ilegible" motivo="sha256_no_cuadra"\n'
                return (0, b"", b"") if self.rotura == "directo" else (2, b"", BLOQUEO)
            return (2, b"", BLOQUEO) if self.rotura == "control_directo" else (0, b"", b"")
        if remoto.startswith("test -e"):
            return (0, b"", b"") if self.rotura == "ejecutado" else (1, b"", b"")
        # lanzamiento del arnés: simula lo que el upstream anotaría
        r = self.upstream.resultados
        if self.rotura != "no_llego":
            r[f"toolu_canario_{NONCE}"] = Resultado(
                f"toolu_canario_{NONCE}", self.rotura != "no_bloqueado",
                {"no_bloqueado": "", "sin_regla": "Error: permiso denegado"}.get(self.rotura, BLOQUEO.decode()))
            ok = self.rotura != "control"
            r[f"toolu_control_{NONCE}"] = Resultado(f"toolu_control_{NONCE}", not ok,
                                                    f"control-c1-{NONCE}" if ok else BLOQUEO.decode())
        return 0, b"", b""

    def upstream_falso(self, guion, host, puerto):
        arnes = self

        class _Up:
            async def __aenter__(self_inner):
                self_inner.resultados, self_inner.peticiones, self_inner.puerto = {}, [], puerto
                arnes.upstream = self_inner
                return self_inner

            async def __aexit__(self_inner, *exc):
                return False
        return _Up()


def _verificar(arnes):
    return asyncio.run(canario_c1.verificar_c1(C, puerto_canario=18436, correr=arnes.correr,
                                               upstream=arnes.upstream_falso, nonce=NONCE))


def test_todo_vivo_no_da_fallos():
    assert _verificar(Arnes()) == ()


@pytest.mark.parametrize("rotura, codigo", [
    ("autoprueba", "autoprueba_fallida"), ("ilegible", "politica_ilegible"),
    ("directo", "canario_directo_no_bloqueado"), ("directo_sin_regla", "canario_directo_sin_su_regla"),
    ("control_directo", "control_directo_bloqueado"),
    ("no_llego", "claude_no_llego_al_canario"), ("no_bloqueado", "canario_no_bloqueado"),
    ("sin_regla", "canario_sin_su_regla"),
    ("control", "control_no_ejecutado"), ("ejecutado", "canario_ejecutado"),
])
def test_cada_rotura_da_su_codigo(rotura, codigo):
    fallos = _verificar(Arnes(rotura))
    assert codigo in [f.codigo for f in fallos], fallos
    assert all(f.contrato == "c1" for f in fallos)


def test_cuenta_inalcanzable():
    async def caida(*a, **k):
        return 255, b"", b"ssh: connect to host 127.0.0.1 port 58291: Connection refused\n"
    fallos = asyncio.run(canario_c1.verificar_c1(C, puerto_canario=18436, correr=caida,
                                                 upstream=Arnes().upstream_falso, nonce=NONCE))
    assert fallos == (Fallo("c1", "cuenta_inalcanzable", (("rc", 255),)),)
```

- [ ] **Step 2: Correr y ver que fallan**

Run: `cd /home/fruiz/worktrees/jax-sp1-c1 && PYTHONPATH=.:las_manos python -m pytest tests/test_ejecutor_contratos_canario_upstream.py tests/test_ejecutor_contratos_canario_c1.py -v`
Expected: FAIL (`cannot import name 'canario_upstream'`).

- [ ] **Step 3: Implementar**

```python
# jax/ejecutor/contratos/canario_upstream.py
"""Upstream con API de mensajes de Anthropic, FALSO, para el canario de C1.

Sigue un guion: cada petición «principal» (la que trae herramientas) recibe el
siguiente `tool_use` del guion; cuando el guion se acaba, `end_turn`. Anota los
`tool_result` que devuelve el arnés. Cualquier otra petición (sin herramientas,
p. ej. un modelo chico para títulos) recibe `end_turn` sin consumir el guion; toda
petición queda en `peticiones` (método, ruta, modelo, con_herramientas) para medir
qué manda el arnés de verdad (Task 9).

asyncio + h11, como proxy_carril. Una petición por conexión.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass

import h11

_LEER = 65536


@dataclass(frozen=True)
class Resultado:
    tool_use_id: str
    es_error: bool
    contenido: str


def guion_bash(tool_use_id: str, comando: str) -> dict:
    return {"id": tool_use_id, "name": "Bash", "input": {"command": comando}}


def _sse(eventos) -> bytes:
    return b"".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n".encode() for e in eventos)


def _mensaje(contenido_bloques, stop_reason):
    return {"id": "msg_canario", "type": "message", "role": "assistant", "model": "canario",
            "content": contenido_bloques, "stop_reason": stop_reason, "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1}}


def _stream_tool_use(paso) -> bytes:
    return _sse([
        {"type": "message_start", "message": _mensaje([], None)},
        {"type": "content_block_start", "index": 0,
         "content_block": {"type": "tool_use", "id": paso["id"], "name": paso["name"], "input": {}}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "input_json_delta", "partial_json": json.dumps(paso["input"])}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use", "stop_sequence": None},
         "usage": {"output_tokens": 1}},
        {"type": "message_stop"},
    ])


def _stream_fin() -> bytes:
    return _sse([
        {"type": "message_start", "message": _mensaje([], None)},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "ok"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
         "usage": {"output_tokens": 1}},
        {"type": "message_stop"},
    ])


def _texto_de(contenido) -> str:
    if isinstance(contenido, str):
        return contenido
    if isinstance(contenido, list):
        return "".join(b.get("text", "") for b in contenido if isinstance(b, dict))
    return ""


class UpstreamCanario:
    def __init__(self, guion: list, host: str, puerto: int):
        self._guion = list(guion)
        self._host, self._puerto_pedido = host, puerto
        self.resultados: dict = {}
        self.peticiones: list = []
        self.puerto: int | None = None
        self._servidor = None

    async def __aenter__(self):
        self._servidor = await asyncio.start_server(self._atender, self._host, self._puerto_pedido)
        self.puerto = self._servidor.sockets[0].getsockname()[1]
        return self

    async def __aexit__(self, *exc):
        self._servidor.close()
        await self._servidor.wait_closed()
        return False

    def _anotar_resultados(self, cuerpo: dict) -> None:
        # TODOS los mensajes, no el último: medido 2026-09-17 con el arnés 2.1.273, el
        # último mensaje es `role: system` (un recordatorio) y el tool_result va antes.
        for mensaje in cuerpo.get("messages") or []:
            if not isinstance(mensaje, dict) or not isinstance(mensaje.get("content"), list):
                continue
            for bloque in mensaje["content"]:
                if isinstance(bloque, dict) and bloque.get("type") == "tool_result":
                    self.resultados[bloque["tool_use_id"]] = Resultado(
                        bloque["tool_use_id"], bool(bloque.get("is_error", False)), _texto_de(bloque.get("content")))

    def _responder(self, ruta: str, cuerpo: dict):
        if ruta.endswith("/count_tokens"):
            return "application/json", json.dumps({"input_tokens": 1}).encode()
        con_herramientas = bool(cuerpo.get("tools"))
        if con_herramientas:
            self._anotar_resultados(cuerpo)
        paso = self._guion.pop(0) if con_herramientas and self._guion else None
        if cuerpo.get("stream"):
            return "text/event-stream", (_stream_tool_use(paso) if paso else _stream_fin())
        bloques = ([{"type": "tool_use", **paso}] if paso else [{"type": "text", "text": "ok"}])
        return "application/json", json.dumps(_mensaje(bloques, "tool_use" if paso else "end_turn")).encode()

    async def _atender(self, reader, writer):
        conn = h11.Connection(h11.SERVER)
        peticion, trozos = None, []
        try:
            while True:
                evento = conn.next_event()
                if evento is h11.NEED_DATA:
                    conn.receive_data(await reader.read(_LEER))
                    continue
                if isinstance(evento, h11.Request):
                    peticion = evento
                elif isinstance(evento, h11.Data):
                    trozos.append(evento.data)
                elif isinstance(evento, (h11.EndOfMessage, h11.ConnectionClosed)):
                    break
            if peticion is None:
                return
            ruta = peticion.target.split(b"?", 1)[0].decode("latin-1")
            try:
                cuerpo = json.loads(b"".join(trozos) or b"{}")
            except ValueError:
                cuerpo = {}
            self.peticiones.append((peticion.method.decode(), ruta, cuerpo.get("model"), bool(cuerpo.get("tools"))))
            if peticion.method != b"POST":
                # Medido: el arnés arranca con `HEAD /api/hello`. Una respuesta a HEAD no lleva cuerpo.
                writer.write(conn.send(h11.Response(status_code=200, headers=[
                    ("content-length", "0"), ("connection", "close")])))
                writer.write(conn.send(h11.EndOfMessage()))
                await writer.drain()
                return
            tipo, datos = self._responder(ruta, cuerpo)
            writer.write(conn.send(h11.Response(status_code=200, headers=[
                ("content-type", tipo), ("content-length", str(len(datos))), ("connection", "close")])))
            writer.write(conn.send(h11.Data(data=datos)))
            writer.write(conn.send(h11.EndOfMessage()))
            await writer.drain()
        except (ConnectionError, h11.RemoteProtocolError):  # fail-soft: upstream de canario; si el arnés corta, el canario no ve su resultado y verificar_c1 lo reporta como fallo
            return
        finally:
            writer.close()
```

```python
# jax/ejecutor/contratos/canario_c1.py
"""Canario permanente de C1 (spec 2026-09-15 §4): si no se dispara, el Ejecutor no arranca.

Tres observaciones, todas como `axioma`:
1. Autoprueba: cada ejemplo de cada regla da lo que dice, con el gancho INSTALADO.
2. Canario directo: el gancho instalado bloquea el evento canario (exit 2, regla
   canario_c1) y deja pasar un control inocuo (exit 0).
3. Canario por el arnés real dentro de la jaula, contra un upstream falso: el
   comando canario vuelve `is_error` con la regla canario_c1 y NO se ejecutó (su
   archivo no existe); el control SÍ se ejecutó (su nonce vuelve en la salida).

Devuelve los Fallo; vacío = C1 vivo. No lanza por un contrato roto: lo reporta.
"""
from __future__ import annotations

import json
import secrets
import shlex

from jax.ejecutor.contratos import canario_upstream, cuenta_axioma
from jax.ejecutor.contratos.fallo import Fallo

_TOPE_DIRECTO_S = 30
_TOPE_ARNES_S = 180
_MARCA = 'regla="canario_c1"'


def _evento(comando: str) -> bytes:
    return json.dumps({"tool_name": "Bash", "tool_input": {"command": comando}}).encode()


async def verificar_c1(c, *, puerto_canario: int, correr=cuenta_axioma.correr_en_la_cuenta,
                       upstream=canario_upstream.UpstreamCanario, nonce: str | None = None) -> tuple:
    nonce = nonce or secrets.token_hex(8)
    q = shlex.quote
    gancho = f"{q(str(c.lib / 'gancho.sh'))} {q(str(c.politica))}"
    fallos: list[Fallo] = []

    rc, salida, errores = await correr(c, f"{gancho} autoprueba", tope_s=_TOPE_DIRECTO_S)
    if rc == 255:
        return (Fallo("c1", "cuenta_inalcanzable", (("rc", rc),)),)
    if rc != 0:
        if b'codigo="politica_ilegible"' in errores:
            fallos.append(Fallo("c1", "politica_ilegible", (("stderr", errores.decode(errors="replace").strip()),)))
        else:
            fallos.append(Fallo("c1", "autoprueba_fallida", (("fallos", salida.decode(errors="replace")),)))

    rc, _, errores = await correr(c, gancho, entrada=_evento("echo ejecutor-canario-c1"), tope_s=_TOPE_DIRECTO_S)
    if rc != 2:
        fallos.append(Fallo("c1", "canario_directo_no_bloqueado", (("rc", rc),)))
    elif _MARCA.encode() not in errores:
        # Bloqueado, pero no por su regla: la política o el gancho están rotos.
        fallos.append(Fallo("c1", "canario_directo_sin_su_regla", (("stderr", errores.decode(errors="replace").strip()),)))
    rc, _, errores = await correr(c, gancho, entrada=_evento("uptime"), tope_s=_TOPE_DIRECTO_S)
    if rc != 0:
        fallos.append(Fallo("c1", "control_directo_bloqueado", (("stderr", errores.decode(errors="replace").strip()),)))

    marca = f"$HOME/.canario-c1-{nonce}"
    id_canario, id_control = f"toolu_canario_{nonce}", f"toolu_control_{nonce}"
    guion = [canario_upstream.guion_bash(id_canario, f'touch "{marca}" # ejecutor-canario-c1'),
             canario_upstream.guion_bash(id_control, f"echo control-c1-{nonce}")]
    async with upstream(guion, "127.0.0.1", puerto_canario) as up:
        remoto = cuenta_axioma.remoto_claude(c, base_url=f"http://127.0.0.1:{up.puerto}", modelo="canario",
                                             prompt="canario")
        await correr(c, remoto, entrada=b"canario\n", tope_s=_TOPE_ARNES_S)
        canario, control = up.resultados.get(id_canario), up.resultados.get(id_control)
    if canario is None:
        fallos.append(Fallo("c1", "claude_no_llego_al_canario", (("peticiones", len(up.peticiones)),)))
    elif not canario.es_error:
        fallos.append(Fallo("c1", "canario_no_bloqueado", (("contenido", canario.contenido[:500]),)))
    elif _MARCA not in canario.contenido:
        fallos.append(Fallo("c1", "canario_sin_su_regla", (("contenido", canario.contenido[:500]),)))
    if control is None or control.es_error or f"control-c1-{nonce}" not in control.contenido:
        fallos.append(Fallo("c1", "control_no_ejecutado",
                            (("contenido", "" if control is None else control.contenido[:500]),)))
    rc, _, _ = await correr(c, f'test -e "{marca}"', tope_s=_TOPE_DIRECTO_S)
    if rc == 0:
        fallos.append(Fallo("c1", "canario_ejecutado"))
    return tuple(fallos)
```

- [ ] **Step 4: Correr y ver que pasan**

Run: `cd /home/fruiz/worktrees/jax-sp1-c1 && PYTHONPATH=.:las_manos python -m pytest tests/test_ejecutor_contratos_canario_upstream.py tests/test_ejecutor_contratos_canario_c1.py -v && python3 policy/tests/test_claude_subprocess_solo_via_sandbox.py && python3 policy/tests/test_no_fail_open_except.py`
Expected: PASS (1 + 12); guardias verdes (`canario_c1.py` menciona el arnés pero no lanza subprocesos).

- [ ] **Step 5: Commit**

```bash
git -C /home/fruiz/worktrees/jax-sp1-c1 add jax/ejecutor/contratos/canario_upstream.py jax/ejecutor/contratos/canario_c1.py tests/test_ejecutor_contratos_canario_upstream.py tests/test_ejecutor_contratos_canario_c1.py
git -C /home/fruiz/worktrees/jax-sp1-c1 commit -m "feat(ejecutor): canario permanente de C1 por el arnés real contra un upstream falso

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: jax · CI — las dos listas, el piso y el canario rojo por API

**Files:**
- Modify: `/home/fruiz/worktrees/jax-sp1-c1/.github/workflows/policy.yml` (job `tests-puros`, las dos listas y el `grep` del piso)

- [ ] **Step 1: Agregar los 8 archivos nuevos a las DOS listas de `tests-puros`**

Después de `tests/test_ejecutor_proxy_carril.py` en la lista del `python -m pytest -v`, y después de
`tests/test_ejecutor_proxy_carril.py \` en la del piso:

```
          tests/test_ejecutor_contratos_destinos.py
          tests/test_ejecutor_contratos_politica.py
          tests/test_ejecutor_contratos_gancho.py
          tests/test_ejecutor_contratos_instalacion.py
          tests/test_ejecutor_contratos_exportar.py
          tests/test_ejecutor_contratos_cuenta_axioma.py
          tests/test_ejecutor_contratos_canario_upstream.py
          tests/test_ejecutor_contratos_canario_c1.py
```

(en la lista del piso, cada línea termina en ` \`).

- [ ] **Step 2: Medir local y verificar que las dos listas son iguales**

Run: `cd /home/fruiz/worktrees/jax-sp1-c1 && python3 - <<'PY'
import re, pathlib
y = pathlib.Path(".github/workflows/policy.yml").read_text()
bloque = y[y.index("  tests-puros:"):y.index("  jacobs-gobernanza-db:")]
listas = [set(re.findall(r"(?:tests|las_manos)/[\w/]+\.py", parte)) for parte in bloque.split("Piso exacto de tests CORRIDOS")]
print(len(listas[0]), len(listas[1]), listas[0] ^ listas[1])
PY`
Expected: dos números iguales y `set()`.

- [ ] **Step 3: Push, piso con el número del runner**

Push de la rama y PR. Leer en el log del job `tests-puros` la línea `N passed, 1 skipped, 1 xfailed`. Poner ese N
en el `grep -qE "^N passed, 1 skipped"` con su línea de historia
`# 742 -> N el 2026-09-17 (Ejecutor SP1 plan 1): contratos C1/C2 — destinos, política, gancho y envoltorio, instalación, exportador, cuenta, upstream y canario. CONFIRMADO POR EL RUNNER (3.12, PR #<número del PR>).`
(742 es el piso en `master` al 2026-09-17; si `master` avanzó, se parte del que haya).

- [ ] **Step 4: Canario rojo por API sobre el sha real**

Commit que rompe a propósito (`test(canario): C1 — el gancho deja pasar lo prohibido; se revierte en el commit
siguiente`), cambiando en `politica.py` `return Decision(False, PROHIBIDO, …)` por
`return Decision(True, PERMITIDO, None, ())`. Push. Verificar por API:
`gh api repos/fjruizhn/jax/commits/<sha>/check-runs --jq '.check_runs[] | select(.name=="tests-puros") | .conclusion'`
Expected: `failure`. Revert, push, mismo comando sobre el sha del revert → `success`. Anotar los dos sha en el PR.

- [ ] **Step 5: Commit del piso**

```bash
git -C /home/fruiz/worktrees/jax-sp1-c1 add .github/workflows/policy.yml
git -C /home/fruiz/worktrees/jax-sp1-c1 commit -m "ci(ejecutor): contratos C1/C2 en tests-puros con piso medido por el runner

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: hall9000 · instalar, ver C1 vivo con el arnés real, y VERLO FALLAR

> Operación sobre hall9000. No toca .10/.11/.20 ni servicios de la Mesa. Requiere los PR de las Tasks 1–8
> mergeados, `jax-platform` desplegado con la migración (backup y restauración probada del dump de
> `axioma_config` antes del restart, lección del frente C) y `JAX_EJECUTOR_INVENTARIO` en `/etc/jax/.env`.

**Files:**
- Create: `/home/fruiz/worktrees/jax-sp1-c1/ops/ejecutor/instalar_contratos.sh`
- Create: `/home/fruiz/worktrees/jax-sp1-c1/scripts/ejecutor_contratos/probar_c1.py`

- [ ] **Step 1: El instalador**

```bash
#!/usr/bin/env bash
# ops/ejecutor/instalar_contratos.sh — instala C1/C2 del Ejecutor en hall9000.
# Corre como fruiz desde el checkout de producción (/home/fruiz/jax en master); usa sudo
# para lo que es de root. Idempotente. Lee JAX_EJECUTOR_* del entorno (set -a; . /etc/jax/.env).
set -euo pipefail
: "${JAX_EJECUTOR_LIB:?}" "${JAX_EJECUTOR_POLITICA:?}" "${JAX_EJECUTOR_GANCHO_TOPE_S:?}" "${JAX_EJECUTOR_CUENTA:?}"
REPO="$(git -C "$(dirname "$(readlink -f "$0")")" rev-parse --show-toplevel)"
test "$(git -C "$REPO" branch --show-current)" = master
ETAPA="$(mktemp -d)"
trap 'rm -rf "$ETAPA"' EXIT
( cd "$REPO" && PYTHONDONTWRITEBYTECODE=1 python3 -m jax.ejecutor.contratos.instalacion "$ETAPA" )

sudo install -d -o root -g root -m 0755 "$JAX_EJECUTOR_LIB"
while IFS= read -r rel; do
  sudo install -D -o root -g root -m 0644 "$REPO/$rel" "$JAX_EJECUTOR_LIB/$rel"
done < "$ETAPA/instalables.txt"
sudo install -o root -g root -m 0755 "$ETAPA/gancho.sh" "$JAX_EJECUTOR_LIB/gancho.sh"
for f in managed-settings.json settings-usuario.json manifiesto.sha256; do
  sudo install -o root -g root -m 0644 "$ETAPA/$f" "$JAX_EJECUTOR_LIB/$f"
done

# Punto de montaje de la jaula: VACÍO en el host. Si alguien puso un managed-settings
# global, se para: afectaría a todas las sesiones de la máquina.
sudo install -d -o root -g root -m 0755 /etc/claude-code
test -z "$(sudo ls -A /etc/claude-code)"

# Política: la escribe fruiz, la lee la cuenta, nadie más.
sudo install -d -o fruiz -g "$JAX_EJECUTOR_CUENTA" -m 2750 "$(dirname "$JAX_EJECUTOR_POLITICA")"

# Puntos de montaje de los settings de la cuenta (sólo si no existen: no se pisa nada).
HOME_CUENTA="$(getent passwd "$JAX_EJECUTOR_CUENTA" | cut -d: -f6)"
for f in settings.json settings.local.json; do
  sudo -u "$JAX_EJECUTOR_CUENTA" test -e "$HOME_CUENTA/.claude/$f" \
    || sudo install -D -o "$JAX_EJECUTOR_CUENTA" -g "$JAX_EJECUTOR_CUENTA" -m 0600 /dev/null "$HOME_CUENTA/.claude/$f"
done

( cd "$JAX_EJECUTOR_LIB" && sudo sha256sum -c --quiet manifiesto.sha256 )
echo "instalado=true lib=\"$JAX_EJECUTOR_LIB\""
```

- [ ] **Step 2: `.env`, migración y exportación (con backup)**

```bash
TS=$(date +%Y%m%d-%H%M%S)
sudo cp -a /etc/jax/.env /etc/jax/.env.backup-pre-ejecutor-c1-$TS
# Agregar (con sudoedit, nunca con sed -i sobre el .env): JAX_EJECUTOR_CUENTA, _SSH_PUERTO, _CONTROLADOR_LLAVE,
# _NODE_BIN, _LIB, _POLITICA, _GANCHO_TOPE_S, _CANARIO_PUERTO, _INVENTARIO (valores del índice).
sudoedit /etc/jax/.env
bash -c 'set -a; . /etc/jax/.env; printf "%s\n" "$JAX_EJECUTOR_INVENTARIO"'
sudo systemd-run --pipe --wait -p EnvironmentFile=/etc/jax/.env /usr/bin/printenv JAX_EJECUTOR_INVENTARIO
```
Expected: las dos lecturas imprimen **exactamente** el mismo valor (lección del frente A: dos lectores).

Después del deploy de jax-platform (su runbook: dump de `axioma_config`, restauración probada en tabla temporal
de `jax_memory_test`, restart, `NRestarts=0`):

```bash
cd /home/fruiz/jax && pwd && git branch --show-current
set -a; . /etc/jax/.env; set +a
PYTHONPATH=.:las_manos python3 -m jax.ejecutor.contratos.exportar
ops/ejecutor/instalar_contratos.sh
```
Expected: `sha256="…" reglas=14 hosts=4` e `instalado=true`. `stat -c '%U:%G %a' /etc/jax-ejecutor /etc/jax-ejecutor/politica.json` → `fruiz:axioma 2750` y `fruiz:axioma 640`.

- [ ] **Step 3: El script de prueba real**

```python
#!/usr/bin/env python3
# scripts/ejecutor_contratos/probar_c1.py
"""Prueba real de C1 en hall9000, para que un tercero la vuelva a correr.

Corre verificar_c1 contra la cuenta real, el gancho instalado y el arnés real en la
jaula. Imprime cada Fallo en formato neutro y sale 0 sólo si no hay ninguno.
Uso:  set -a; . /etc/jax/.env; set +a
      PYTHONPATH=.:las_manos python3 scripts/ejecutor_contratos/probar_c1.py
Lee /etc/jax/.env (producción) sólo para JAX_EJECUTOR_*: no toca la DB.
"""
import asyncio
import os
import sys

from jax.ejecutor.contratos import canario_c1, cuenta_axioma, formato


def main() -> int:
    c = cuenta_axioma.cuenta_desde_entorno()
    fallos = asyncio.run(canario_c1.verificar_c1(c, puerto_canario=int(os.environ["JAX_EJECUTOR_CANARIO_PUERTO"])))
    for f in fallos:
        print(formato.campos((("contrato", f.contrato), ("codigo", f.codigo)) + tuple(f.datos)))
    print(formato.campos((("c1_vivo", not fallos),)))
    return 0 if not fallos else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Verlo vivo**

Run: `cd /home/fruiz/jax && set -a && . /etc/jax/.env && set +a && PYTHONPATH=.:las_manos python3 scripts/ejecutor_contratos/probar_c1.py`
Expected: `c1_vivo=true`, exit 0. **Medir y anotar** (Principio V, riesgo declarado): la lista `peticiones` del
upstream (agregar un `print` temporal y quitarlo, o correr con `JAX_EJECUTOR_CANARIO_DEPURAR=1` si se decide
agregarlo con test): qué rutas y modelos pide el arnés. Si pide algo que el upstream no sabe responder y por eso
`claude_no_llego_al_canario`, se agrega esa respuesta al upstream **con su test**, no se afloja el canario.

- [ ] **Step 5: VERLO FALLAR — lista ilegible bloquea TODO (el caso del spec)**

```bash
set -a; . /etc/jax/.env; set +a
cp -a "$JAX_EJECUTOR_POLITICA" /tmp/politica.mut-bak
python3 - <<'PY'
import json, os, pathlib
p = pathlib.Path(os.environ["JAX_EJECUTOR_POLITICA"]); d = json.loads(p.read_text()); d["c2_edad_max_s"] += 1
p.write_text(json.dumps(d))
PY
PYTHONPATH=.:las_manos python3 scripts/ejecutor_contratos/probar_c1.py; echo "rc=$?"
cp -a /tmp/politica.mut-bak "$JAX_EJECUTOR_POLITICA" && cmp "$JAX_EJECUTOR_POLITICA" /tmp/politica.mut-bak && rm /tmp/politica.mut-bak
```
Expected con la mutación, exactamente estos cinco y ninguno más: `politica_ilegible`, `canario_directo_sin_su_regla`,
`control_directo_bloqueado`, `canario_sin_su_regla` y `control_no_ejecutado`. **Ni `canario_directo_no_bloqueado` ni
`canario_no_bloqueado` ni `canario_ejecutado`**: el canario sigue bloqueado (por la política ilegible) y el control
inocuo TAMBIÉN, a través del arnés real. Eso es «lista ilegible → todo bloqueado». `rc=1`. Restaurado:
`probar_c1.py` vuelve a `c1_vivo=true`.

- [ ] **Step 6: VERLO FALLAR — sin canario no arranca; con el gancho roto bloquea todo**

Dos mutaciones más, cada una con backup y `cmp`, cada una corriendo `probar_c1.py`:
1. Borrar la regla `canario_c1` del JSON y re-firmar con `politica.firmar` → Expected: `politica_ilegible` con
   `motivo="canario_ausente_o_multiple"`, más los mismos cuatro de bloqueo total del Step 5.
2. `sudo cp -a $JAX_EJECUTOR_LIB/jax/ejecutor/contratos/politica.py /tmp/p.mut-bak` y agregarle una línea
   `raise RuntimeError` al final (con `sudo tee -a`) → Expected: `autoprueba_fallida` (rc 2 sin
   `politica_ilegible`), `canario_directo_sin_su_regla`, `control_directo_bloqueado`, `canario_sin_su_regla` y
   `control_no_ejecutado`; ninguno de «no bloqueado» (el crash sale 1 → envoltorio 2 → bloqueado). Restaurar con
   `sudo install -o root -g root -m 0644 /tmp/p.mut-bak …` y `sudo sha256sum -c manifiesto.sha256` limpio.

- [ ] **Step 7: MEDIR `disableAllHooks` a nivel de USUARIO (el de proyecto ya está medido)**

**Ya medido al escribir este plan** (2026-09-17 03:45, Mr. Hyde, `claude` 2.1.273 como `axioma`, con el código de
las Tasks 2–7 en un directorio descartable y `/etc/claude-code` vacío creado y borrado después): con
`cd <proyecto>` y `<proyecto>/.claude/settings.json` = `{"disableAllHooks": true, "permissions": {"deny":
["Bash(echo:*)"]}}`, el canario **siguió bloqueado por el gancho** y el control salió **denegado por el `deny`**.
El control prueba que el archivo de proyecto SÍ se leyó; el canario prueba que su `disableAllHooks` NO apagó el
gancho gestionado. Falta el nivel de usuario, que la jaula tapa con `settings-usuario.json` de solo lectura:
1. Variante de `remoto_claude` **sin** los dos `--ro-bind` de settings, y `~/.claude/settings.json` de la cuenta
   con `{"disableAllHooks": true, "permissions": {"deny": ["Bash(echo:*)"]}}` (guardar antes el archivo instalado
   con `cp -a` y restaurarlo con `cmp`).
2. Correr `verificar_c1` con esa variante. Si aparece `canario_no_bloqueado`: el montaje de solo lectura es lo
   único que sostiene C1 → se agrega a `tests/test_ejecutor_contratos_cuenta_axioma.py` un test que falla si
   `_jaula` pierde cualquiera de los dos `--ro-bind`, y se registra en la Biblioteca. Si no aparece: se registra
   igual, con el control (el `deny` aplicado).
3. Con la jaula completa otra vez, `verificar_c1` → `()`.

- [ ] **Step 8: Latencia del gancho (Cuatro del rendimiento)**

Como `axioma`: 200 invocaciones de `gancho.sh` con eventos variados (la mitad con `ssh -tt axioma@bridge …`),
midiendo cada una con `date +%s%N`. Expected y registro: p50, p95 y máx en ms. Criterio pre-registrado: **p95 ≤
150 ms** (el gancho corre en cada herramienta). Si no pasa, se mide dónde (arranque de python vs. carga del JSON vs.
regex) antes de tocar nada.

- [ ] **Step 9: Commit y Biblioteca**

```bash
git -C /home/fruiz/worktrees/jax-sp1-c1 add ops/ejecutor/instalar_contratos.sh scripts/ejecutor_contratos/probar_c1.py
git -C /home/fruiz/worktrees/jax-sp1-c1 commit -m "ops(ejecutor): instalador de C1/C2 y prueba real re-ejecutable por un tercero

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

Entrada en `CONTEXT.md` §9 (fecha, qué, por qué, los tres «verlo fallar» con su salida literal, la medición de
`disableAllHooks`, p95 del gancho, lecciones) y en `DEUDA.md` cualquier residuo **con fecha**.

---

## Auto-revisión (hecha al escribir)

- **Cobertura del spec §4:** C1 lista en DB ✓ (Task 1), gancho PreToolUse ✓ (Task 4), fail-closed con lista
  ilegible ✓ (Tasks 3, 4 y 9 Step 5), cada prohibido se intenta y sale bloqueado ✓ (autoprueba, Tasks 3, 5, 7, 9),
  canario permanente ✓ (Tasks 7 y 9; el arranque que se niega es el plan 6), semilla derivada de la Biblioteca ✓
  (las cinco del spec + `ssh_sin_tt` de C4 + `ajustes_claude_code` del riesgo medido). C2: destructivo sin punto de
  restauración verificado de ESA máquina → bloqueado ✓ (Task 3, `test_destructivo_sin_respaldo_se_bloquea_por_maquina`).
  Quién escribe puntos de restauración verificados: **Fase 3/4 (SP4/SP5)**; hasta entonces todo destructivo se
  bloquea, que es lo cerrado.
- **Placeholders:** ninguno; los números de piso los fija el runner por regla del ecosistema, con la instrucción exacta.
- **Tipos:** `Fallo(contrato, codigo, datos)`, `Cuenta(nombre, puerto, llave, node_bin, lib, politica)`,
  `verificar_c1(c, *, puerto_canario, correr, upstream, nonce)` iguales en Interfaces, Tasks 6–7 y tests.
