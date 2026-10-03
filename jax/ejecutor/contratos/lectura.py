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


#: Campos de primer nivel de la API de mensajes que el proxy o Ollama interpretan. Ollama (Go,
#: `encoding/json`) empareja las claves SIN distinguir mayúsculas: `"Model"` o `"Max_Tokens"` se
#: leerían como `model` / `max_tokens` allá y no acá. Un pedido que escribe uno de estos campos
#: con otra capitalización, o repite una clave por casefold, es ambiguo y no pasa (MAJOR-5).
CAMPOS_CONOCIDOS = frozenset({
    "model", "max_tokens", "messages", "system", "tools", "stream", "thinking", "think",
    "reasoning_effort", "metadata", "stop_sequences", "temperature", "top_p", "top_k", "tool_choice",
})


class PedidoAmbiguo(ValueError):
    """Dos claves de primer nivel iguales por casefold, o un campo conocido escrito distinto."""


def cargar_pedido(cuerpo: bytes):
    """`json.loads` del pedido, rechazando lo que Go leería distinto que Python.
    Lanza `PedidoAmbiguo` (un `ValueError`) si el primer nivel repite una clave por casefold o
    escribe un campo conocido con otra capitalización. La validación del proxy y la lectura de C3
    usan ESTA función: ven lo mismo que verá Ollama."""
    ultimo: list = []

    def gancho(pares):
        ultimo[:] = [pares]  # el objeto que termina de leerse al final es el de primer nivel
        return dict(pares)

    doc = json.loads(cuerpo, object_pairs_hook=gancho)
    if isinstance(doc, dict) and ultimo:
        plegadas = [k.casefold() for k, _ in ultimo[0]]
        if len(set(plegadas)) != len(plegadas):
            raise PedidoAmbiguo("clave repetida por casefold")
        for k in doc:
            if k.casefold() in CAMPOS_CONOCIDOS and k not in CAMPOS_CONOCIDOS:
                raise PedidoAmbiguo("campo conocido con otra capitalización")
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
