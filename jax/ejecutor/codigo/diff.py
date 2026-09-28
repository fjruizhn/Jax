"""El diff de una misión de código, como estructura (spec 2026-09-28 §3.3.1).
C1 de la entrega decide sobre ESTO, no sobre texto plano con regex.

Ruling del controlador (Tarea 9, 2026-09-28), tres defectos reproducidos del parser anterior:
- Las líneas se parten SOLO por "\\n". `str.splitlines()` corta también en \\r, \\x0b, \\x0c,
  \\x1c-\\x1e, \\x85, \\u2028 y \\u2029: lo que quedaba detrás no empezaba con '+' y se perdía, y un
  secreto pasaba el barrido.
- Cabecera y contenido se distinguen por ESTADO: `---`/`+++` son cabecera solo antes del primer
  `@@` de cada archivo. Antes, una línea agregada cuyo texto empieza con «++» salía como `+++ …`
  y se tiraba como cabecera.
- Rutas: UTF-8 y con espacios sin comillas (la entrega corre git con `core.quotePath=false`), y
  entre comillas con los escapes de git (\\t, \\", \\\\, octal) para lo que git sigue citando. La
  ruta de `diff --git` se lee por simetría (`a/P b/P`); `rename from/to` y `---`/`+++` la fijan
  cuando existen."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Cambio:
    ruta: str
    estado: str
    ruta_anterior: str | None
    agregadas: tuple[str, ...]
    quitadas: tuple[str, ...]


_ESCAPES = {"a": 7, "b": 8, "t": 9, "n": 10, "v": 11, "f": 12, "r": 13, '"': 34, "\\": 92}


def _cita(texto: str, i: int) -> tuple[str, int]:
    """Lee una ruta entre comillas de git que empieza en `texto[i] == '"'`. Devuelve (ruta, índice
    después de la comilla de cierre). Los octales son BYTES (UTF-8 escapado)."""
    salida = bytearray()
    i += 1
    while i < len(texto):
        ch = texto[i]
        if ch == '"':
            return salida.decode("utf-8", errors="replace"), i + 1
        if ch == "\\" and i + 1 < len(texto):
            sig = texto[i + 1]
            octal = texto[i + 1:i + 4]
            if len(octal) == 3 and all(c in "01234567" for c in octal):
                salida.append(int(octal, 8) & 0xFF)
                i += 4
                continue
            if sig not in _ESCAPES:  # git no emite otro: fail-closed antes que adivinar una ruta
                raise ValueError("ruta_entre_comillas_ilegible")
            salida.append(_ESCAPES[sig])
            i += 2
            continue
        salida.extend(ch.encode("utf-8"))
        i += 1
    raise ValueError("ruta_entre_comillas_sin_cerrar")


def _ruta(token: str, prefijo: str) -> str | None:
    """`a/x`, `"a/x"` o `/dev/null` (→ None)."""
    if token.startswith('"'):
        token = _cita(token, 0)[0]
    if token == "/dev/null":
        return None
    return token[len(prefijo):] if token.startswith(prefijo) else token


def _cabecera(resto: str) -> str:
    """La ruta de `diff --git <resto>`. Con comillas, se leen los dos tokens; sin comillas,
    `a/P b/P` es simétrico aunque P tenga espacios o « b/» adentro."""
    if resto.startswith('"'):
        a, j = _cita(resto, 0)
        b = resto[j + 1:]
        b = _cita(b, 0)[0] if b.startswith('"') else b
        return b[2:] if b.startswith("b/") else a[2:]
    if resto.startswith("a/"):
        n = (len(resto) - 5) // 2  # "a/" + P + " b/" + P
        p = resto[2:2 + n]
        if resto == f"a/{p} b/{p}":
            return p
    # Asimétrico (renombre): la fijan después `rename to` o `+++`.
    return resto.split(" b/", 1)[1] if " b/" in resto else resto


def _cerrar(actual: dict | None, salida: list) -> None:
    if actual is not None:
        salida.append(Cambio(actual["ruta"], actual["estado"], actual["anterior"],
                             tuple(actual["mas"]), tuple(actual["menos"])))


def parsear(texto_diff: str) -> tuple[Cambio, ...]:
    salida: list[Cambio] = []
    actual: dict | None = None
    en_hunk = False
    for linea in texto_diff.split("\n"):
        if linea.startswith("diff --git "):
            _cerrar(actual, salida)
            actual = {"ruta": _cabecera(linea[len("diff --git "):]), "estado": "M", "anterior": None,
                      "mas": [], "menos": []}
            en_hunk = False
        elif actual is None:
            continue
        elif en_hunk:
            if linea.startswith("+"):
                actual["mas"].append(linea[1:])
            elif linea.startswith("-"):
                actual["menos"].append(linea[1:])
            elif linea.startswith("@@"):
                continue
        elif linea.startswith("@@"):
            en_hunk = True
        elif linea.startswith("new file mode"):
            actual["estado"] = "A"
        elif linea.startswith("deleted file mode"):
            actual["estado"] = "D"
        elif linea.startswith("rename from "):
            actual["estado"], actual["anterior"] = "R", _ruta(linea[len("rename from "):], "")
        elif linea.startswith("rename to "):
            actual["ruta"] = _ruta(linea[len("rename to "):], "")
        elif linea.startswith("+++ "):
            ruta = _ruta(linea[4:], "b/")
            if ruta is not None:
                actual["ruta"] = ruta
        elif linea.startswith("--- "):
            ruta = _ruta(linea[4:], "a/")
            if ruta is not None and actual["estado"] == "D":
                actual["ruta"] = ruta
    _cerrar(actual, salida)
    return tuple(salida)
