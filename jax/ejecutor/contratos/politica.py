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
