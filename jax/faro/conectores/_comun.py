"""Validadores y helpers compartidos de los conectores del Faro (spec §0.8/P6).

Generadores PUROS: entran solo datos publicos (nombre/modelo/perfil, URL de puente
loopback validada, rutas absolutas del hogar efimero) y salen bytes deterministas
(JSON/TOML/argv). Nada de IO, nada de proceso, nada de red. Falla cerrada: cualquier
dato faltante o invalido levanta TypeError/ValueError, nunca se "adivina".

Invariantes comunes:
- El token de la ejecucion NUNCA viaja por argv ni por entorno: siempre por
  `--token-file` hacia el rele (`jax.faro.relay`), que es quien lo lee y lo manda
  como primera linea del socket. Los conectores solo montan esa argv.
- Las credenciales que si hay que escribir en config (Kimi `api_key`, Claude
  `ANTHROPIC_AUTH_TOKEN`, Codex `env_key`) llevan la MARCA_FICTICIA: un marcador
  explicito que el provisionador real reemplaza en el hogar efimero. Un valor que
  pareciera real seria un bug.
"""
from __future__ import annotations

import json
import string
from urllib.parse import urlsplit

MARCA_FICTICIA = "FARO-CREDENCIAL-FICTICIA"

_ALFABETO_NOMBRE = set(string.ascii_lowercase + string.digits + "._-")
_ALFABETO_MODELO = set(string.ascii_letters + string.digits + "._-/")
_ALFABETO_URL = set(string.ascii_lowercase + string.digits + "-.:[]")


def _revisar_tipo(valor: object, campo: str) -> str:
    if not isinstance(valor, str):
        raise TypeError(f"{campo}: se esperaba str, llego {type(valor).__name__}")
    return valor


def texto_plano(valor: object, campo: str) -> str:
    """Texto sin caracteres de control ni NUL; falla cerrada si hay inyeccion."""
    valor = _revisar_tipo(valor, campo)
    if not valor:
        raise ValueError(f"{campo}: texto vacio")
    for ch in valor:
        o = ord(ch)
        if o < 0x20 or o == 0x7F:
            raise ValueError(f"{campo}: caracter de control prohibido (U+{o:04X})")
    return valor


def nombre_sano(valor: object, campo: str = "nombre") -> str:
    """Identificador de motor/perfil: [a-z0-9._-]+, sin '..' (anti path traversal)."""
    valor = texto_plano(valor, campo)
    if any(ch not in _ALFABETO_NOMBRE for ch in valor):
        raise ValueError(f"{campo}: solo [a-z0-9._-]")
    if ".." in valor or valor.startswith((".", "-")):
        raise ValueError(f"{campo}: no puede empezar por '.'/'-' ni contener '..'")
    return valor


def modelo_sano(valor: object, campo: str = "modelo") -> str:
    """Nombre de modelo publico: letras, digitos, '._-/'; sin '..' ni saltos."""
    valor = texto_plano(valor, campo)
    if any(ch not in _ALFABETO_MODELO for ch in valor):
        raise ValueError(f"{campo}: caracter fuera de [A-Za-z0-9._/-]")
    if ".." in valor or valor.startswith("/"):
        raise ValueError(f"{campo}: '..' o ruta absoluta prohibida")
    return valor


def ruta_absoluta(valor: object, campo: str) -> str:
    """Ruta absoluta sin control chars, sin NUL y sin componente '..'."""
    valor = texto_plano(valor, campo)
    if not valor.startswith("/"):
        raise ValueError(f"{campo}: debe ser ruta absoluta (empezar por '/')")
    partes = [p for p in valor.split("/") if p not in ("", ".")]
    if any(p == ".." for p in partes):
        raise ValueError(f"{campo}: componente '..' prohibido")
    return valor


def url_loopback(valor: object, campo: str = "base_url") -> str:
    """Solo http://127.0.0.1[:puerto] o http://localhost[:puerto]; sin userinfo,
    sin query, sin fragment. Cualquier otra cosa (host externo, https con ruta
    rara, credenciales embebidas) es rechazo cerrado."""
    valor = texto_plano(valor, campo)
    partes = urlsplit(valor)
    if partes.scheme != "http":
        raise ValueError(f"{campo}: solo esquema http (puente loopback)")
    if partes.hostname not in ("127.0.0.1", "localhost"):
        raise ValueError(f"{campo}: solo 127.0.0.1 o localhost")
    if partes.username is not None or partes.password is not None:
        raise ValueError(f"{campo}: userinfo prohibido en la URL")
    if partes.query or partes.fragment:
        raise ValueError(f"{campo}: query/fragment prohibidos")
    if partes.port is not None and not (1 <= partes.port <= 65535):
        raise ValueError(f"{campo}: puerto fuera de rango")
    if any(ch not in _ALFABETO_URL for ch in partes.netloc):
        raise ValueError(f"{campo}: netloc con caracteres prohibidos")
    if partes.path not in ("", "/"):
        raise ValueError(f"{campo}: path prohibido en el puente loopback")
    return valor


def credencial_ficticia(nombre: str) -> str:
    """Marcador explicito de credencial: el provisionador real lo reemplaza."""
    return f"{MARCA_FICTICIA}-{nombre_sano(nombre)}"


def argv_rele(python: object, socket: object, token_file: object) -> list[str]:
    """argv del rele: el token viaja SOLO por --token-file, jamas en argv/env."""
    python = ruta_absoluta(python, "python")
    socket = ruta_absoluta(socket, "socket")
    token_file = ruta_absoluta(token_file, "token_file")
    return [python, "-m", "jax.faro.relay", "--socket", socket, "--token-file", token_file]


def json_ordenado(objeto: object) -> str:
    """JSON determinista: claves ordenadas, indentacion fija, newline final."""
    return json.dumps(objeto, indent=2, sort_keys=True, ensure_ascii=True) + "\n"


def toml_texto(valor: object, campo: str = "valor") -> str:
    """String basico TOML escapado (backslash, comilla, control). Entradas ya
    validadas como texto_plano; esto es defensa en profundidad."""
    valor = texto_plano(valor, campo)
    escapado = (
        valor.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\t", "\\t")
    )
    return f'"{escapado}"'


def toml_arreglo_textos(valores: list[str]) -> str:
    """Arreglo TOML de strings, una linea, orden dado (ya validados)."""
    return "[" + ", ".join(toml_texto(v) for v in valores) + "]"
