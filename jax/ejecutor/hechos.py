"""Hechos del sistema, derivados de comandos reales e inyectados en cada turno.

Spec: docs/superpowers/specs/2026-09-16-ejecutor-fase2-design.md §3.1 y §8
riesgo 4. Es la capa 1 del resultado de Fase 0: cierra por adelantado los
huecos que U3 midió («llevan casi un día encendidos» con un timestamp que
decía 38 minutos).

Contrato:
1. Cada hecho lleva su comando y su momento, y el bloque los muestra.
2. Lo que no se pudo derivar se OMITE y se dice cuál, con qué comando y por
   qué. Nunca se rellena.
3. Los hechos caducan (TTL, arranca en 60 s). El TTL compara INSTANTES con
   zona, no cadenas ISO: Tegucigalpa es UTC−6.
4. Una captura TRUNCADA, o con código distinto de 0, o sin código (el proceso
   no murió en plazo), NO es un hecho. Dar por bueno lo que llegó cortado es
   el error de la tarea 9 de U3.

Desviaciones del plan (2026-09-16), cada una con su test:
- `derivar(inventario, maquina, ...)` devuelve `Derivacion(hechos,
  no_derivados)`, no `list[Hecho]`: una lista de hechos sola pierde las
  omisiones, y la regla 2 exige decirlas. No recibe `ahora`: el momento de
  cada hecho es el de SU captura, no el del turno.
- `no_derivados` son `NoDerivado(nombre, comando, motivo)`, no nombres
  sueltos: un nombre sin motivo es «desconocido sin decir por qué».
- Un hecho vencido que llega a `bloque` también se omite y se dice.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

from jax.ejecutor import captura

TTL_S_POR_DEFECTO = 60
# Los hechos se derivan antes de CADA turno: comandos cortos, salida corta.
TOPE_BYTES_POR_DEFECTO = 65_536
TIMEOUT_S_POR_DEFECTO = 10.0


@dataclass(frozen=True)
class Fuente:
    nombre: str
    comando: str


@dataclass(frozen=True)
class Hecho:
    nombre: str
    valor: str
    comando: str
    # ISO-8601 CON zona: momento en que empezó la captura que lo respalda.
    momento: str


@dataclass(frozen=True)
class NoDerivado:
    nombre: str
    comando: str
    motivo: str


@dataclass(frozen=True)
class Derivacion:
    hechos: tuple[Hecho, ...]
    no_derivados: tuple[NoDerivado, ...]


INVENTARIO_BASE: tuple[Fuente, ...] = (
    Fuente("hostname", "hostname"),
    Fuente("uptime", "uptime -p"),
    Fuente("arranque", "uptime -s"),
    Fuente("so", '. /etc/os-release && printf "%s\\n" "$PRETTY_NAME"'),
    Fuente("kernel", "uname -r"),
    Fuente("disco", "df -h /"),
)

_NOMBRE_UNIDAD = re.compile(r"^[A-Za-z0-9@._:-]+$")


def fuente_servicio(unidad: str) -> Fuente:
    """Estado del servicio y desde cuándo. Se pide ActiveState junto al
    timestamp porque ActiveEnterTimestamp sobrevive a un stop: sólo no diría
    si el servicio sigue activo. Y LoadState porque una unidad que no existe
    da código 0 e `inactive` (medido en hall9000 el 2026-09-16): sin él se
    leería «existe y está parada»."""
    if not _NOMBRE_UNIDAD.fullmatch(unidad):
        raise ValueError(f"nombre de unidad inválido: {unidad!r}")
    return Fuente(f"servicio {unidad}",
                  f"systemctl show -p LoadState -p ActiveState -p ActiveEnterTimestamp -- {unidad}")


def _instante(iso: str) -> datetime | None:
    """El instante, o None si no tiene zona: sin zona no se sabe cuál es."""
    momento = datetime.fromisoformat(iso)
    return momento if momento.tzinfo is not None else None


def _motivo_de_invalidez(hecho: Hecho, ahora: datetime, ttl_s: float) -> str | None:
    try:
        momento = _instante(hecho.momento)
    except ValueError:  # fail-soft: un momento ilegible deja el hecho fuera del bloque y se dice por qué; nunca entra como vigente
        return f"momento ilegible {hecho.momento!r}"
    if momento is None:
        return f"momento sin zona horaria {hecho.momento!r}"
    edad = (ahora - momento).total_seconds()
    if edad < 0:
        return f"momento en el futuro ({-edad:.0f} s después de ahora)"
    if edad > ttl_s:
        return f"vencido: tiene {edad:.0f} s y el TTL es {ttl_s:g} s"
    return None


def _ahora(ahora: str) -> datetime:
    instante = _instante(ahora)
    if instante is None:
        raise ValueError(f"`ahora` necesita zona horaria: {ahora!r}")
    return instante


def vigente(hecho: Hecho, ahora: str, ttl_s: float = TTL_S_POR_DEFECTO) -> bool:
    return _motivo_de_invalidez(hecho, _ahora(ahora), ttl_s) is None


def _motivo_de_rechazo(completa: captura.CapturaCompleta) -> str | None:
    # El truncado se mira ANTES que el código y el contenido: lo que llegó
    # cortado no se lee (tarea 9 de U3).
    if completa.truncada:
        motivos = ", ".join(completa.motivos_truncado) or "sin motivo registrado"
        return f"la salida vino truncada ({motivos})"
    if completa.codigo is None:
        return "sin código de salida: el proceso no terminó dentro del plazo"
    if completa.codigo != 0:
        return f"código de salida {completa.codigo}"
    if not completa.salida.strip():
        return "la salida vino vacía"
    return None


def derivar(inventario, maquina: str, *, correr=captura.correr,
            tope_bytes: int = TOPE_BYTES_POR_DEFECTO,
            timeout_s: float = TIMEOUT_S_POR_DEFECTO) -> Derivacion:
    hechos: list[Hecho] = []
    omitidos: list[NoDerivado] = []
    for fuente in inventario:
        try:
            completa = correr(fuente.comando, maquina=maquina,
                              tope_bytes=tope_bytes, timeout_s=timeout_s)
        except Exception as error:  # fail-soft: un comando que no se pudo lanzar deja ESE hecho omitido y dicho; los demás hechos del turno siguen
            omitidos.append(NoDerivado(fuente.nombre, fuente.comando,
                                       f"no se pudo correr: {error!r}"))
            continue
        motivo = _motivo_de_rechazo(completa)
        if motivo is not None:
            omitidos.append(NoDerivado(fuente.nombre, fuente.comando, motivo))
            continue
        hechos.append(Hecho(nombre=fuente.nombre, valor=completa.salida.strip(),
                            comando=fuente.comando, momento=completa.momento))
    return Derivacion(tuple(hechos), tuple(omitidos))


def bloque(hechos, ahora: str, no_derivados=(), ttl_s: float = TTL_S_POR_DEFECTO) -> str:
    """El texto que se inyecta en el turno. Una línea `- nombre = valor` por
    hecho, con su comando y su momento; las líneas extra de un valor van
    sangradas para que no parezcan otro hecho."""
    instante = _ahora(ahora)
    lineas = [f"Hechos del sistema (derivados de comandos; ahora {ahora}; TTL {ttl_s:g} s):"]
    omitidos: list[str] = []
    for hecho in hechos:
        motivo = _motivo_de_invalidez(hecho, instante, ttl_s)
        if motivo is not None:
            omitidos.append(f"- {hecho.nombre}: omitido, {motivo} "
                            f"(comando: `{hecho.comando}`, momento: {hecho.momento})")
            continue
        primera, *resto = hecho.valor.splitlines() or [""]
        lineas.append(f"- {hecho.nombre} = {primera}")
        lineas.extend(f"    {linea}" for linea in resto)
        lineas.append(f"    (comando: `{hecho.comando}`, momento: {hecho.momento})")
    for omitido in no_derivados:
        omitidos.append(f"- {omitido.nombre}: no se pudo derivar, {omitido.motivo} "
                        f"(comando: `{omitido.comando}`)")
    if omitidos:
        lineas.append("Omitidos (no usar como hechos):")
        lineas.extend(omitidos)
    return "\n".join(lineas)
