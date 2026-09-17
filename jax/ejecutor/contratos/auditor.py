# jax/ejecutor/contratos/auditor.py
"""Auditor en vivo del Ejecutor (C5), parte pura.

Spec 2026-09-15 §4 C5 y §6.3; spec de Fase 2 §2.0, avisos de §2.2 y §7. Plan 4 de SP1.
Tres cosas por lote:
- un paso fuera de misión o prohibido → PAUSA (con el número del paso);
- una afirmación que cita una línea real pero no responde lo que su `proposito`
  pregunta → se RETIENE (no sale). Es el límite medido de las citas: «el 8188 es
  Docker multi-hilo» citando la línea real del puerto salía respaldada;
- checklist de la constitución (parche, hardcoding, cierre sin verificar) → se anota.

`interpretar` es FAIL-CLOSED: lo que el auditor dice sólo vale si cita el lote. Un
hallazgo que pausa sin un paso real, un paso o una afirmación citados que no existen, un
tipo o veredicto desconocido, un duplicado, o algo que no es JSON → AuditorIlegible, y
quien llama frena. Una afirmación sin veredicto se retiene.

Las afirmaciones se identifican por POSICIÓN en `Entrega.respaldadas` (`a1`, `a2`…), no
por igualdad ni por identidad de objeto: dos afirmaciones idénticas se juzgan cada una, y
`transporte.entregar` ya reemplaza los objetos (la línea como la imprimió la máquina).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace

from jax.ejecutor import cita, transporte
from jax.ejecutor.cita import Motivo

TIPOS = ("fuera_de_mision", "prohibido", "solucion_temporal", "hardcoding", "cierre_sin_verificacion")
PAUSAN = frozenset({"fuera_de_mision", "prohibido"})
VEREDICTOS = ("responde", "no_responde")
RETENIDA_POR_AUDITOR = "retenida_por_auditor"
#: Líneas de la salida que acompañan a la citada: la primera (suele ser la cabecera de
#: columnas, que decide qué significa cada campo) y las vecinas.
CONTEXTO_VECINAS = 1
_BLOQUE = re.compile(r"^\s*`{3}(?:json)?\s*\n(.*)\n\s*`{3}\s*$", re.S)  # un bloque de código JSON


@dataclass(frozen=True)
class Paso:
    n: int
    herramienta: str | None
    entrada: object
    es_error: bool | None


@dataclass(frozen=True)
class AfirmacionAuditable:
    id: str
    proposito: str
    dato: str
    maquina: str
    comando: str
    linea: str
    contexto: tuple


@dataclass(frozen=True)
class Lote:
    mision: str
    pasos: tuple
    afirmaciones: tuple


@dataclass(frozen=True)
class Hallazgo:
    tipo: str
    paso: int | None
    afirmacion: str | None


@dataclass(frozen=True)
class Revision:
    pausar: bool
    motivo: str | None
    paso: int | None
    hallazgos: tuple
    aprobadas: frozenset
    retenidas: frozenset


class AuditorIlegible(ValueError):
    def __init__(self, codigo: str):
        super().__init__(codigo)
        self.codigo = codigo


def id_de(posicion: int) -> str:
    return f"a{posicion + 1}"


def lote_desde_dict(d: dict) -> Lote:
    return Lote(
        mision=d["mision"],
        pasos=tuple(Paso(p["n"], p.get("herramienta"), p.get("entrada"), p.get("es_error")) for p in d["pasos"]),
        afirmaciones=tuple(AfirmacionAuditable(a["id"], a["proposito"], a["dato"], a["maquina"], a["comando"],
                                               a["linea"], tuple(a.get("contexto", ()))) for a in d["afirmaciones"]),
    )


def _contexto(afirmacion, capturas) -> tuple:
    aguja = cita.normalizar(afirmacion.linea)
    for captura in capturas:
        if (captura.maquina, captura.comando) != (afirmacion.maquina, afirmacion.comando) or captura.truncada:
            continue
        for flujo in (captura.salida, captura.stderr):
            lineas = flujo.splitlines()
            for i, linea in enumerate(lineas):
                if cita.normalizar(linea) == aguja:
                    indices = sorted({0, *range(max(0, i - CONTEXTO_VECINAS), min(len(lineas), i + CONTEXTO_VECINAS + 1))})
                    return tuple(lineas[k] for k in indices)
    return ()


def afirmaciones_auditables(entrega) -> tuple:
    """Las respaldadas de una entrega, numeradas por posición, con su contexto."""
    return tuple(AfirmacionAuditable(id_de(i), a.proposito, a.dato, a.maquina, a.comando, a.linea,
                                     _contexto(a, entrega.crudas))
                 for i, a in enumerate(entrega.respaldadas))


def mensajes(lote: Lote, instrucciones: str) -> list[dict]:
    cuerpo = {
        "mision": lote.mision,
        "pasos": [{"n": p.n, "herramienta": p.herramienta, "entrada": p.entrada, "es_error": p.es_error}
                  for p in lote.pasos],
        "afirmaciones": [{"id": a.id, "proposito": a.proposito, "dato": a.dato, "maquina": a.maquina,
                          "comando": a.comando, "linea": a.linea, "contexto": list(a.contexto)}
                         for a in lote.afirmaciones],
    }
    return [{"role": "system", "content": instrucciones},
            {"role": "user", "content": json.dumps(cuerpo, ensure_ascii=False)}]


def _es_numero_de_paso(valor) -> bool:
    return isinstance(valor, int) and not isinstance(valor, bool)


def interpretar(lote: Lote, texto) -> Revision:
    m = _BLOQUE.match(texto) if isinstance(texto, str) else None
    try:
        doc = json.loads(m.group(1) if m else texto)
    except (TypeError, ValueError):
        raise AuditorIlegible("json_invalido") from None
    if not isinstance(doc, dict):
        raise AuditorIlegible("json_invalido")
    if not isinstance(doc.get("hallazgos"), list) or not isinstance(doc.get("afirmaciones"), list):
        raise AuditorIlegible("forma_invalida")
    pasos = {p.n for p in lote.pasos}
    ids = {a.id for a in lote.afirmaciones}
    hallazgos = []
    for h in doc["hallazgos"]:
        if not isinstance(h, dict) or h.get("tipo") not in TIPOS:
            raise AuditorIlegible("tipo_desconocido")
        paso = h.get("paso")
        if h["tipo"] in PAUSAN and not (_es_numero_de_paso(paso) and paso in pasos):
            raise AuditorIlegible("cita_paso_inexistente")
        if paso is not None and not (_es_numero_de_paso(paso) and paso in pasos):
            raise AuditorIlegible("cita_paso_inexistente")
        afirmacion = h.get("afirmacion")
        if afirmacion is not None and afirmacion not in ids:
            raise AuditorIlegible("cita_afirmacion_inexistente")
        hallazgos.append(Hallazgo(h["tipo"], paso, afirmacion))
    aprobadas, vistas = set(), set()
    for v in doc["afirmaciones"]:
        if not isinstance(v, dict) or v.get("id") not in ids:
            raise AuditorIlegible("cita_afirmacion_inexistente")
        if v.get("veredicto") not in VEREDICTOS:
            raise AuditorIlegible("veredicto_desconocido")
        if v["id"] in vistas:
            raise AuditorIlegible("veredicto_duplicado")
        vistas.add(v["id"])
        if v["veredicto"] == "responde":
            aprobadas.add(v["id"])
    que_pausan = [h for h in hallazgos if h.tipo in PAUSAN]
    primero = que_pausan[0] if que_pausan else None
    return Revision(bool(que_pausan), primero.tipo if primero else None, primero.paso if primero else None,
                    tuple(hallazgos), frozenset(aprobadas), frozenset(ids - aprobadas))


def aplicar_revision(entrega, revision: Revision):
    """Toda respaldada cuyo id de posición el auditor no aprobó pasa a descartadas.
    Las crudas no se tocan (piso de Fase 2 §2.3)."""
    motivo = Motivo(RETENIDA_POR_AUDITOR)
    salen, retenidas = [], []
    for i, a in enumerate(entrega.respaldadas):
        if id_de(i) in revision.aprobadas:
            salen.append(a)
        else:
            retenidas.append(transporte.Descartada(a, RETENIDA_POR_AUDITOR, motivo))
    return replace(entrega, respaldadas=tuple(salen), descartadas=entrega.descartadas + tuple(retenidas))
