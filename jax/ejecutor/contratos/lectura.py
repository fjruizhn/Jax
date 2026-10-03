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


def plegar(nombre: str) -> str:
    """Plegado de nombres de clave como el de `encoding/json` de Go (≥1.21, `appendFoldedName`):
    ASCII a mayúsculas y, por cada rune no ASCII, `ToUpper(ToLower(r))` con mapeos SIMPLES (un
    rune a un rune; `str.lower()`/`str.upper()` de Python a veces devuelven varios y se dejan
    como están). Así `K` (Kelvin, U+212A), `ſ` (U+017F), `ı` (U+0131) e `İ` (U+0130) se pliegan
    como en Go. Es una emulación CONSERVADORA: puede unir de más (rechaza), nunca de menos."""
    def minuscula(c: str) -> str:
        if c == "\u0130":  # Go: ToLower(İ) = 'i'; Python daría 'i' + U+0307
            return "i"
        m = c.lower()
        return m if len(m) == 1 else c

    def mayuscula(c: str) -> str:
        m = c.upper()
        return m if len(m) == 1 else c

    return "".join(mayuscula(minuscula(c)) for c in nombre)


#: Campos conocidos por nivel. Ollama (Go, `encoding/json`) empareja las claves SIN distinguir
#: mayúsculas: `"Model"`, `"Max_Tokens"`, `{"Type":"tool_result"}` o un `"Content"` a nivel de
#: mensaje se leerían allá como el campo conocido y no acá (ni en C3). Un objeto que escribe uno
#: de sus campos conocidos con otra forma, o que repite una clave tras el plegado, es ambiguo.
CAMPOS_CONOCIDOS = frozenset({
    "model", "max_tokens", "messages", "system", "tools", "stream", "thinking", "think",
    "reasoning_effort", "metadata", "stop_sequences", "temperature", "top_p", "top_k", "tool_choice",
    "output_config",  # MessagesRequest.OutputConfig de Ollama v0.34.3 (anthropic.go:82); Claude Code lo manda
})
CAMPOS_DE_MENSAJE = frozenset({"role", "content"})
CAMPOS_DE_BLOQUE = frozenset({
    "type", "text", "id", "name", "input", "tool_use_id", "content", "is_error", "source",
    "thinking", "signature", "cache_control", "citations",
})
CAMPOS_DE_HERRAMIENTA = frozenset({"type", "name", "description", "input_schema", "cache_control"})


class PedidoAmbiguo(ValueError):
    """Un objeto del pedido repite una clave tras el plegado de Go, o escribe un campo conocido
    con otra capitalización."""


def _revisar(pares: list, conocidos: frozenset) -> None:
    plegadas = [plegar(k) for k, _ in pares]
    if len(set(plegadas)) != len(plegadas):
        raise PedidoAmbiguo("clave repetida tras el plegado")
    conocidas_plegadas = {plegar(c) for c in conocidos}
    for k, _ in pares:
        if plegar(k) in conocidas_plegadas and k not in conocidos:
            raise PedidoAmbiguo("campo conocido con otra forma")


def _revisar_bloques(contenido, pares_de: dict) -> None:
    if not isinstance(contenido, list):
        return
    for bloque in contenido:
        if not isinstance(bloque, dict):
            continue
        _revisar(pares_de[id(bloque)], CAMPOS_DE_BLOQUE)
        # Recursivo en el `content` cuando es lista (los tool_result). Es CONSERVADOR, no
        # necesario: Ollama decodifica ese `content` como any/map y `convertToolResultContent` lo
        # lee distinguiendo mayúsculas; aquí se rechaza igual por simetría con el resto.
        _revisar_bloques(bloque.get("content"), pares_de)


def bloque_de_servidor(doc) -> bool:
    """¿Algún bloque de `messages[].content[]` es de herramienta de SERVIDOR? Ollama 0.34.3
    (anthropic.go:525-541) convierte `web_search_tool_result` en un mensaje de rol tool y
    `server_tool_use` en una llamada a herramienta; C3 solo cuenta `tool_result` y `tool_use`.
    Esos bloques no tienen origen legítimo (las herramientas de servidor ya dan 403): `type`
    plegado `server_tool_use`, o que empiece por `web_search` / `web_fetch`, o que termine en
    `_tool_result` sin ser `tool_result`."""
    if not isinstance(doc, dict) or not isinstance(doc.get("messages"), list):
        return False
    for mensaje in doc["messages"]:
        contenido = mensaje.get("content") if isinstance(mensaje, dict) else None
        if not isinstance(contenido, list):
            continue
        for bloque in contenido:
            tipo = bloque.get("type") if isinstance(bloque, dict) else None
            if not isinstance(tipo, str):
                continue
            t = plegar(tipo)
            if (t == "SERVER_TOOL_USE" or t.startswith(("WEB_SEARCH", "WEB_FETCH"))
                    or (t.endswith("_TOOL_RESULT") and t != "TOOL_RESULT")):
                return True
    return False


def cargar_pedido(cuerpo: bytes):
    """`json.loads` del pedido, rechazando lo que Go leería distinto que Python.
    Lanza `PedidoAmbiguo` (un `ValueError`) si repite una clave tras el plegado de Go o escribe
    un campo conocido con otra capitalización, en CUALQUIERA de estos objetos:
      - el primer nivel (`CAMPOS_CONOCIDOS`);
      - cada objeto de `messages[]` (`CAMPOS_DE_MENSAJE`);
      - cada bloque de `content` cuando es una lista (`CAMPOS_DE_BLOQUE`), y, recursivamente, el
        `content` de un bloque cuando también es una lista (los tool_result);
      - cada objeto de `tools[]` (`CAMPOS_DE_HERRAMIENTA`), porque el proxy decide por su `type`.
    NO mira `system`, `metadata` ni lo que hay dentro de `input`/`source`/`cache_control`/
    `input_schema`: C3 y el proxy no leen nada ahí. La validación del proxy y la lectura de C3 usan ESTA función."""
    pares_de: dict = {}

    def gancho(pares):
        objeto = dict(pares)
        pares_de[id(objeto)] = pares  # los pares crudos: dict() esconde los repetidos
        return objeto

    doc = json.loads(cuerpo, object_pairs_hook=gancho)
    if isinstance(doc, dict):
        _revisar(pares_de[id(doc)], CAMPOS_CONOCIDOS)
        mensajes = doc.get("messages")
        if isinstance(mensajes, list):
            for mensaje in mensajes:
                if isinstance(mensaje, dict):
                    _revisar(pares_de[id(mensaje)], CAMPOS_DE_MENSAJE)
                    _revisar_bloques(mensaje.get("content"), pares_de)
        herramientas = doc.get("tools")
        if isinstance(herramientas, list):
            for herramienta in herramientas:
                if isinstance(herramienta, dict):
                    _revisar(pares_de[id(herramienta)], CAMPOS_DE_HERRAMIENTA)
    return doc


def _canonico(valor) -> bytes:
    # surrogatepass: idéntico para todo texto válido; un surrogate suelto (JSON con escape ud83d)
    # no tumba la lectura de C3.
    return json.dumps(valor, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8", "surrogatepass")


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
        self._cr_pendiente = False
        self._abiertos: dict = {}

    def alimentar(self, trozo: bytes) -> list[HerramientaPedida]:
        # SSE admite CR, LF y CRLF como fin de línea. Un CR se toma como fin de línea EN
        # CUANTO llega (lo antes posible: nunca después que el arnés); si el byte siguiente,
        # aunque venga en otro trozo, es el LF de un CRLF, ese LF se descarta.
        if self._cr_pendiente and trozo.startswith(b"\n"):
            trozo = trozo[1:]
        self._cr_pendiente = trozo.endswith(b"\r")
        self._resto += trozo.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
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
        doc = cargar_pedido(cuerpo)
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
