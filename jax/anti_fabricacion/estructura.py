"""Detector de estructura determinista (anti-fabricación, capa 2).

Contrasta lo que una salida AFIRMA sobre estructuras existentes (archivos,
tablas, columnas, símbolos, rutas de la app) con la whitelist derivada del
sistema. Sin LLM, sin heurísticas de verbos: lo que se revisa lo decide el CANAL.

CANALES.
  * Salida con schema: el mapa `CANALES_POR_SCHEMA` (junto a `SCHEMAS`, en
    `motor_registry/output_validator.py`) declara qué campos AFIRMAN, cuáles
    PROPONEN, cuáles son listas de rutas y cuáles son diffs. Lo que propone no se
    revisa; un campo que no figura no se revisa (límite declarado).
  * Diff: `--- a/X` afirma X; `--- /dev/null` + `+++ b/Y` propone Y. Lo que el
    diff define (`+def`, `+class`, `+CREATE TABLE`, `+CONSTANTE =`) es propio.
  * Texto libre y campos que afirman: SOLO 4 formas (Enmienda 1, regla 2): rutas con
    «/» (y extensión conocida, o primera componente igual a una carpeta del repo, o
    absoluta), `tabla.columna` con la izquierda tabla real, `*.service|timer` (systemd:
    fuera de v1, se declara) y las rutas de `files_modified` y «--- a/». La prosa nunca
    es candidata. Un identificador suelto en backticks (sin «/» ni «.») y un nombre de
    archivo sin directorio NO son candidatos.

PROPUESTO / HIPÓTESIS. Propuesto = lo que el job escribió (`propias`), lo que su
diff define, o lo marcado `nuevo:` en backticks. Hipótesis = candidato dentro de
una oración que termina en «?».

CORPUS CONFIABLE. Solo texto de persona o del sistema (objetivo, definición de la
capability). La salida de otro modelo NUNCA respalda: un candidato que solo se ve
en `previos` (pasos anteriores) es `heredado`. Enmienda 1, regla 3: si el job de origen
tiene veredicto propio guardado (`Previo.veredicto`), el heredado es `no_verificable`
(visible, nunca limpio, no falla en duro); si no hay origen o no tiene veredicto, es
`marcado`.

LÍMITES DECLARADOS (lo que este detector NO ve; no son olvidos).
  * Una palabra o identificador suelto, con o sin backticks, no se revisa: «`bodegas` es una
    tabla» no se atrapa (los casos de ese tipo de U4 quedan «fuera de alcance»).
  * Una ruta absoluta cuya primera componente existe en `/` (`/etc`, `/srv`, `/home`...)
    se SALTA: `/srv/jax-data/jax-workspace/services/bodegas.py` pasa limpio. Este
    detector responde por el repo y la app, no por el sistema de archivos.
  * Los campos de un schema que no figuran en `CANALES_POR_SCHEMA` no se revisan.
  * systemd y las rutas de jax-platform están fuera de v1 (`fuera_de_alcance`).
  * `corpus_confiable` solo respalda lo que la persona o el sistema dijeron. En el
    worker, hoy, eso es el nombre de la capability, sus `forbidden_paths` y
    `context["objective"]` si viene; `/motor/dispatch` no manda `objective`, así que
    en la práctica el prompt entero entra como `previos`: un operador que despacha
    directo y nombra un archivo real en su prompt no lo respalda por eso (lo respalda
    la whitelist, si existe).
  * La whitelist de archivos es `git ls-files`: un archivo que existe en disco pero
    no está versionado cuenta como inexistente.

ESTADOS: limpio / marcado / no_verificable (una fuente opcional cayó para esa
categoría: visible, no pasa limpio, no acusa) / no_corrio.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import PurePosixPath
from typing import Collection, Iterable, Sequence

from .whitelist import Whitelist

try:
    from motor_registry.output_validator import CANALES_POR_SCHEMA  # LAS MANOS (cwd las_manos/)
except ImportError:  # pragma: no cover - según desde dónde se importe
    from las_manos.motor_registry.output_validator import CANALES_POR_SCHEMA  # scripts (raíz del repo)

LIMPIO = "limpio"
MARCADO = "marcado"
NO_VERIFICABLE = "no_verificable"
NO_CORRIO = "no_corrio"

SIN_RESPALDO = "sin_respaldo"
HEREDADO = "heredado"

#: Cota del largo de un candidato y del número de ítems que se guardan por veredicto.
_MAX_CANDIDATO = 200
_MAX_ITEMS = 100

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_SPAN = re.compile(r"`([^`\n]+)`")
_URL = re.compile(r"\b[A-Za-z][A-Za-z0-9+.-]*://\S+")
_RUTA_BARE = re.compile(r"(?<![\w/.:@~$-])((?:/|\./|\.\./)?[\w.@+-]+(?:/[\w.@+-]+)+/?)")
_PAR_BARE = re.compile(r"(?<![\w/.@-])([A-Za-z_]\w*)\.([A-Za-z_]\w*)\b")
_SERVICIO_BARE = re.compile(r"(?<![\w/.@-])[\w@-]+\.(?:service|timer)\b")
_RUTA_CHARS = re.compile(r"[\w.@+-]+(?:/[\w.@+-]+)*/?\Z")
_VALLA = re.compile(r"```([A-Za-z0-9_+-]*)[^\n]*\n(.*?)```", re.DOTALL)
_DIFF_A = re.compile(r"^--- a/(\S+)", re.MULTILINE)
_DIFF_NUEVO = re.compile(r"^--- /dev/null[^\n]*\n\+\+\+ b/(\S+)", re.MULTILINE)
_DEF = re.compile(r"^\+\s*(?:async\s+def|def|class)\s+([A-Za-z_]\w*)", re.MULTILINE)
_CREATE = re.compile(r"^\+\s*CREATE\s+(?:OR\s+REPLACE\s+)?(?:TEMPORARY\s+)?TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?`?([A-Za-z_]\w*)`?",
                     re.MULTILINE | re.IGNORECASE)
_CONST = re.compile(r"^\+([A-Z_][A-Z0-9_]+)\s*(?::[^=\n]+)?=", re.MULTILINE)
_ORACIONES = re.compile(r"(?<=[.!?])\s+|\n+")
_REF_LINEA = re.compile(r"(?::\d+(?:-\d+)?|::\S*)\Z")


@dataclass(frozen=True)
class Previo:
    """Salida de un paso anterior (otro modelo). `veredicto` es el estado que el JOB DE
    ORIGEN guardó para esa salida, o None si no hay origen o no tiene veredicto."""
    texto: str
    veredicto: str | None = None


@dataclass(frozen=True)
class Item:
    candidato: str
    tipo: str      # ruta | ruta_abs | tabla_columna
    canal: str     # afirma | texto | diff
    clase: str     # sin_respaldo | heredado
    motivo: str


@dataclass(frozen=True)
class Veredicto:
    estado: str
    items: tuple[Item, ...] = ()
    propuestos: tuple[str, ...] = ()
    hipotesis: tuple[str, ...] = ()
    no_verificables: tuple[str, ...] = ()
    fuera_de_alcance: tuple[str, ...] = ()
    motivo: str = ""

    def como_dict(self) -> dict:
        return {
            "estado": self.estado,
            "items": [asdict(i) for i in self.items[:_MAX_ITEMS]],
            "items_total": len(self.items),
            "propuestos": list(self.propuestos[:_MAX_ITEMS]),
            "hipotesis": list(self.hipotesis[:_MAX_ITEMS]),
            "no_verificables": list(self.no_verificables),
            "fuera_de_alcance": list(self.fuera_de_alcance),
            "motivo": self.motivo,
            # el detector no llegó a mirar: esta salida no cuenta entre las 30 de la fase sombra
            "cuenta_para_sombra": self.estado != NO_CORRIO,
        }


def no_corrio(motivo: str) -> Veredicto:
    """El detector no llegó a mirar la salida. No es limpio y no acusa."""
    return Veredicto(estado=NO_CORRIO, motivo=motivo)


@dataclass(frozen=True)
class _Cand:
    texto: str
    tipo: str
    canal: str
    hipotesis: bool = False
    nuevo: bool = False


# --------------------------------------------------------------------------
# extracción
# --------------------------------------------------------------------------

def _strings(valor) -> list[str]:
    if isinstance(valor, str):
        return [valor]
    if isinstance(valor, dict):
        return [s for v in valor.values() for s in _strings(v)]
    if isinstance(valor, (list, tuple)):
        return [s for v in valor for s in _strings(v)]
    return []


def _segmentos(salida: str, schema: str) -> list[tuple[str, str]]:
    """[(texto, canal)] con canal en afirma | rutas | texto | diff. Los campos que
    PROPONEN no salen: no se revisan."""
    canales = CANALES_POR_SCHEMA.get(schema)
    try:
        parseado = json.loads(salida)
    except (json.JSONDecodeError, ValueError):
        return [(salida, "texto")]
    if not isinstance(parseado, dict):
        return [(salida, "texto")]
    if canales is None:
        return [(s, "texto") for s in _strings(parseado)]
    segs: list[tuple[str, str]] = []
    for campo, valor in parseado.items():
        if campo in canales["propone"]:
            continue
        if campo in canales["diff"]:
            segs += [(s, "diff") for s in _strings(valor)]
        elif campo in canales["rutas"]:
            segs += [(s, "rutas") for s in _strings(valor)]
        elif campo in canales["afirma"]:
            segs += [(s, "afirma") for s in _strings(valor)]
    return segs


def _diff(texto: str) -> tuple[list[str], list[str]]:
    """(afirmados, propuestos) de las cabeceras y definiciones de un diff."""
    afirmados = [m for m in _DIFF_A.findall(texto)]
    propuestos = list(_DIFF_NUEVO.findall(texto))
    propuestos += _DEF.findall(texto) + _CREATE.findall(texto) + _CONST.findall(texto)
    return afirmados, propuestos


def _sin_hunks(texto: str) -> str:
    """El texto sin las líneas de cuerpo de un diff (código, no prosa)."""
    salida, en_hunk = [], False
    for linea in texto.split("\n"):
        if linea.startswith("@@"):
            en_hunk = True
            continue
        if en_hunk and linea[:1] in (" ", "+", "-", "\\") and linea != "":
            if linea.startswith("--- a/") or linea.startswith("--- /dev/null"):
                en_hunk = False  # cabecera del archivo siguiente
            else:
                continue
        else:
            en_hunk = False
        if linea.startswith(("--- a/", "--- /dev/null", "+++ b/", "diff --git ", "index ", "new file mode",
                             "deleted file mode")):
            continue
        salida.append(linea)
    return "\n".join(salida)


def _normalizar_ruta(r: str) -> str:
    r = _REF_LINEA.sub("", r.strip().strip("\"'"))
    while r.startswith("./"):
        r = r[2:]
    return r.rstrip(",;:)")


def _es_forma_de_ruta(p: str, exts: frozenset[str], raices: frozenset[str]) -> bool:
    """Una cadena con barra es ruta si termina en extensión que el repo usa, si su
    primera componente es una carpeta del repo, o si es absoluta / ./ / ../."""
    if "*" in p or "{" in p or "<" in p or "$" in p or "~" in p:
        return False
    if p.startswith(("/", "./", "../")):
        return True
    partes = [x for x in p.split("/") if x]
    if len(partes) < 2:
        return False
    ultimo = PurePosixPath(partes[-1]).suffix.lower()
    return (ultimo in exts) or (partes[0] in raices)


def _clasificar_span(s: str, wl: Whitelist, exts: frozenset[str], raices: frozenset[str]) -> tuple[str, str] | None:
    """(tipo, candidato) de un trozo entre backticks, o None si no es revisable.
    Los comandos y los fragmentos con espacios no se revisan."""
    s = s.strip()
    if not s or len(s) > _MAX_CANDIDATO or re.search(r"\s", s):
        return None
    if re.search(r"\.(?:service|timer)\Z", s):
        return ("systemd", s)
    if s.startswith("/") and _RUTA_CHARS.match(s.lstrip("/") or "x"):
        r = _normalizar_ruta(s)
        return ("ruta_abs", r) if len(r) > 1 and "*" not in r and "{" not in r else None
    if "/" in s:
        r = _normalizar_ruta(s)
        if _RUTA_CHARS.match(r) and _es_forma_de_ruta(r, exts, raices):
            return ("ruta", r)
        return None
    if "." in s:
        izq, _, der = s.partition(".")
        if izq in wl.tablas and _IDENT.match(der):
            return ("tabla_columna", s)
    # Enmienda 1, regla 2: un identificador suelto (sin «/») o un nombre de archivo sin
    # directorio NO es candidato. Límite declarado: «`bodegas` es una tabla» no se atrapa.
    return None


def _candidatos(segs, wl: Whitelist) -> tuple[list[_Cand], list[str], set[str]]:
    exts = frozenset(PurePosixPath(a).suffix.lower() for a in wl.archivos if PurePosixPath(a).suffix)
    raices = frozenset(d for d in wl.directorios if "/" not in d)
    cands: list[_Cand] = []
    propuestos: list[str] = []
    fuera: set[str] = set()

    for texto, canal in segs:
        if canal == "rutas":
            r = _normalizar_ruta(texto)
            if r and "*" not in r:
                cands.append(_Cand(r, "ruta_abs" if r.startswith("/") else "ruta", "afirma"))
            continue
        # fences: el código de ejemplo no es prosa; los de diff/patch se leen como diff
        bloques_diff = []
        for lenguaje, cuerpo in _VALLA.findall(texto):
            if lenguaje.lower() in ("diff", "patch"):
                bloques_diff.append(cuerpo)
        texto_sin_vallas = _VALLA.sub("\n", texto)
        for cuerpo in bloques_diff + [texto_sin_vallas]:
            afirmados, nuevos = _diff(cuerpo)
            cands += [_Cand(_normalizar_ruta(a), "ruta", "diff") for a in afirmados]
            propuestos += nuevos
        if canal == "diff":
            continue
        prosa = _sin_hunks(texto_sin_vallas)
        for oracion in _ORACIONES.split(prosa):
            if not oracion.strip():
                continue
            hipotesis = oracion.rstrip().rstrip("\"')]»”").endswith("?")
            for span in _SPAN.findall(oracion):
                nuevo = False
                if span.strip().lower().startswith("nuevo:"):
                    span, nuevo = span.strip()[6:].strip(), True
                    if span:
                        propuestos.append(span)
                clase = _clasificar_span(span, wl, exts, raices)
                if clase is None:
                    continue
                tipo, cand = clase
                if tipo == "systemd":
                    fuera.add("systemd")
                    continue
                cands.append(_Cand(cand, tipo, canal if canal != "texto" else "texto", hipotesis, nuevo))
            fuera_de_spans = _URL.sub(" ", _SPAN.sub(" ", oracion))
            if _SERVICIO_BARE.search(fuera_de_spans):
                fuera.add("systemd")
            for m in _RUTA_BARE.finditer(fuera_de_spans):
                r = _normalizar_ruta(m.group(1))
                if _RUTA_CHARS.match(r) and _es_forma_de_ruta(r, exts, raices):
                    cands.append(_Cand(r, "ruta_abs" if r.startswith("/") else "ruta", "texto", hipotesis))
            sin_rutas = _RUTA_BARE.sub(" ", fuera_de_spans)
            for m in _PAR_BARE.finditer(sin_rutas):
                if m.group(1) in wl.tablas:
                    cands.append(_Cand(f"{m.group(1)}.{m.group(2)}", "tabla_columna", "texto", hipotesis))
    return cands, propuestos, fuera


# --------------------------------------------------------------------------
# contraste con la whitelist
# --------------------------------------------------------------------------

def _existe_ruta(r: str, wl: Whitelist) -> bool:
    r = r.rstrip("/")
    if r in wl.archivos or r in wl.directorios:
        return True
    sufijo = "/" + r
    return any(a.endswith(sufijo) for a in wl.archivos) or any(d.endswith(sufijo) for d in wl.directorios)


def _existe_ruta_app(r: str, wl: Whitelist) -> bool:
    if r in wl.rutas_app:
        return True
    for patron in wl.rutas_app:
        if "{" in patron:
            trozos = re.split(r"\{[^/}]+\}", patron)
            if re.fullmatch("[^/]+".join(re.escape(t) for t in trozos), r):
                return True
    return False


def _parece_ruta(x: str) -> bool:
    return "/" in x or "." in x


def _es_propia(c: str, propias: Collection[str]) -> bool:
    c = c.rstrip("/")
    for p in propias:
        p = _normalizar_ruta(str(p)).rstrip("/")
        if p == c:
            return True
        if _parece_ruta(p) and _parece_ruta(c) and (p.endswith("/" + c) or c.endswith("/" + p)):
            return True
    return False


def _en_texto(candidato: str, textos: Iterable[str]) -> bool:
    return any(candidato in t for t in textos)


def verificar(
    salida: str,
    wl: Whitelist,
    corpus_confiable: str | Sequence[str],
    propias: Collection[str],
    *,
    previos: Sequence[str | Previo] = (),
    schema: str = "",
) -> Veredicto:
    """Veredicto de estructura de `salida`.

    `corpus_confiable`: texto de persona o del sistema (objetivo, definición de la
    capability). `propias`: lo que el job escribió. `previos`: salidas de pasos
    anteriores (otros modelos): NUNCA respaldan; un candidato que solo está ahí es
    `heredado`. `schema`: el output_schema de la capability, si hay."""
    confiable = [corpus_confiable] if isinstance(corpus_confiable, str) else list(corpus_confiable)
    segs = _segmentos(salida or "", schema)
    cands, propuestos_diff, fuera = _candidatos(segs, wl)
    propios_total = set(propuestos_diff) | {str(p) for p in propias}
    previos_p = [x if isinstance(x, Previo) else Previo(str(x)) for x in previos]

    items: list[Item] = []
    hipotesis: list[str] = []
    propuestos: list[str] = list(dict.fromkeys(propuestos_diff))
    no_verificables: set[str] = set()
    vistos: set[tuple[str, str]] = set()

    for c in cands:
        clave = (c.texto, c.tipo)
        if clave in vistos:
            continue
        if c.hipotesis:
            vistos.add(clave)
            hipotesis.append(c.texto)
            continue
        if c.nuevo or c.texto in propios_total or _es_propia(c.texto, propios_total):
            vistos.add(clave)
            if c.texto not in propuestos:
                propuestos.append(c.texto)
            continue
        if c.tipo == "ruta":
            existe = _existe_ruta(c.texto, wl)
        elif c.tipo == "ruta_abs":
            primera = c.texto.lstrip("/").split("/")[0]
            if "raices_fs" in wl.no_disponibles:
                no_verificables.add("raices_fs")
                vistos.add(clave)
                continue
            if primera in wl.raices_fs:
                vistos.add(clave)  # ruta del sistema (/etc, /srv...): no es de este detector
                continue
            existe = _existe_ruta(c.texto.lstrip("/"), wl) or _existe_ruta_app(c.texto, wl)
            if not existe and "rutas_app" in wl.no_disponibles:
                no_verificables.add("rutas_app")
                vistos.add(clave)
                continue
        elif c.tipo == "tabla_columna":
            existe = c.texto in wl.pares
        vistos.add(clave)
        if existe or _en_texto(c.texto, confiable):
            continue
        if c.tipo == "ruta":
            # el árbol en disco o el checkout de jax no se pudieron ver enteros: no se acusa
            falta = [k for k in ("archivos", "jax_checkout") if k in wl.no_disponibles]
            if falta:
                no_verificables.update(falta)
                continue
        origen = [pr for pr in previos_p if c.texto in pr.texto]
        if origen and all(pr.veredicto for pr in origen):
            no_verificables.add("heredado")   # regla 3: el origen tiene veredicto propio guardado
            continue
        clase = HEREDADO if origen else SIN_RESPALDO
        motivo = {
            "ruta": "no existe en git ls-files del repo de trabajo",
            "ruta_abs": "no es una ruta del repo ni de la app LAS MANOS",
            "tabla_columna": "no existe en information_schema de la base",
        }[c.tipo]
        if clase == HEREDADO:
            motivo += "; solo aparece en la salida de un paso previo sin veredicto propio (otro modelo no respalda)"
        items.append(Item(c.texto[:_MAX_CANDIDATO], c.tipo, c.canal, clase, motivo))

    estado = MARCADO if items else (NO_VERIFICABLE if no_verificables else LIMPIO)
    return Veredicto(
        estado=estado, items=tuple(items), propuestos=tuple(propuestos),
        hipotesis=tuple(dict.fromkeys(hipotesis)), no_verificables=tuple(sorted(no_verificables)),
        fuera_de_alcance=tuple(sorted(fuera)),
    )
