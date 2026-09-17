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

SIN TEXTOS VISIBLES (política del ecosistema, 2026-09-16): ningún string que
vea la persona se escribe aquí. El backend de jax no tiene i18n; lo muestra el
frontend de jax-platform con `react-i18next`. Por eso `bloque` devuelve un
`Bloque` (estructura) y cada omisión lleva un `Motivo`: un CÓDIGO estable y
sus datos, que el frontend traduce. Los valores de los hechos y los comandos
son datos de las capturas, no rótulos.
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


# Códigos de `Motivo`: claves estables del contrato con el frontend, que pone
# el texto traducido. Cambiar uno rompe esa traducción.
MOMENTO_ILEGIBLE = "momento_ilegible"
MOMENTO_SIN_ZONA = "momento_sin_zona"
MOMENTO_FUTURO = "momento_futuro"
VENCIDO = "vencido"
TRUNCADA = "truncada"
SIN_CODIGO = "sin_codigo"
CODIGO_DISTINTO_DE_CERO = "codigo_distinto_de_cero"
SALIDA_VACIA = "salida_vacia"
NO_SE_PUDO_CORRER = "no_se_pudo_correr"


@dataclass(frozen=True)
class Motivo:
    """Por qué algo no es un hecho: un código de arriba y sus datos, como
    pares `(clave, valor)` (inmutables; `dict(motivo.datos)` para serializar).
    Sin prosa: la frase la pone el frontend."""
    codigo: str
    datos: tuple[tuple[str, object], ...] = ()


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
    motivo: Motivo


@dataclass(frozen=True)
class Derivacion:
    hechos: tuple[Hecho, ...]
    no_derivados: tuple[NoDerivado, ...]


@dataclass(frozen=True)
class Invalido:
    """Un hecho que se derivó pero ya no vale (vencido, del futuro, momento
    sin zona o ilegible). Va con su momento a la vista, dentro de `hecho`."""
    hecho: Hecho
    motivo: Motivo


@dataclass(frozen=True)
class Bloque:
    """Lo que se inyecta en el turno, como estructura. `vigentes` son los
    únicos que se pueden usar como hechos; `invalidos` y `no_derivados` se
    muestran como omitidos, cada uno con su motivo."""
    ahora: str
    ttl_s: float
    vigentes: tuple[Hecho, ...]
    invalidos: tuple[Invalido, ...]
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


def _motivo_de_invalidez(hecho: Hecho, ahora: datetime, ttl_s: float) -> Motivo | None:
    try:
        momento = _instante(hecho.momento)
    except ValueError:  # fail-soft: un momento ilegible deja el hecho fuera del bloque y se dice por qué; nunca entra como vigente
        return Motivo(MOMENTO_ILEGIBLE, (("momento", hecho.momento),))
    if momento is None:
        return Motivo(MOMENTO_SIN_ZONA, (("momento", hecho.momento),))
    edad = (ahora - momento).total_seconds()
    if edad < 0:
        return Motivo(MOMENTO_FUTURO, (("segundos", -edad),))
    if edad > ttl_s:
        return Motivo(VENCIDO, (("edad_s", edad), ("ttl_s", ttl_s)))
    return None


def _ahora(ahora: str) -> datetime:
    instante = _instante(ahora)
    if instante is None:
        raise ValueError(f"`ahora` necesita zona horaria: {ahora!r}")
    return instante


def vigente(hecho: Hecho, ahora: str, ttl_s: float = TTL_S_POR_DEFECTO) -> bool:
    return _motivo_de_invalidez(hecho, _ahora(ahora), ttl_s) is None


def _motivo_de_rechazo(completa: captura.CapturaCompleta) -> Motivo | None:
    # El truncado se mira ANTES que el código y el contenido: lo que llegó
    # cortado no se lee (tarea 9 de U3).
    if completa.truncada:
        # `motivos_truncado` ya son códigos de captura.py; vacío = sin motivo
        # registrado, y lo dice la tupla vacía.
        return Motivo(TRUNCADA, (("motivos", tuple(completa.motivos_truncado)),))
    if completa.codigo is None:
        return Motivo(SIN_CODIGO)
    if completa.codigo != 0:
        return Motivo(CODIGO_DISTINTO_DE_CERO, (("codigo", completa.codigo),))
    if not completa.salida.strip():
        return Motivo(SALIDA_VACIA)
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
                                       Motivo(NO_SE_PUDO_CORRER, (("error", repr(error)),))))
            continue
        motivo = _motivo_de_rechazo(completa)
        if motivo is not None:
            omitidos.append(NoDerivado(fuente.nombre, fuente.comando, motivo))
            continue
        hechos.append(Hecho(nombre=fuente.nombre, valor=completa.salida.strip(),
                            comando=fuente.comando, momento=completa.momento))
    return Derivacion(tuple(hechos), tuple(omitidos))


def bloque(hechos, ahora: str, no_derivados=(), ttl_s: float = TTL_S_POR_DEFECTO) -> Bloque:
    """Lo que se inyecta en el turno, como ESTRUCTURA: cada hecho con su
    comando y su momento; los que no valen, aparte y con su `Motivo`.

    No rotula (ningún «Hechos del sistema», «omitido», «comando:»): los textos
    visibles los pone el frontend con sus traducciones. Un valor de varias
    líneas es UN campo de UN hecho: con estructura no hay líneas de texto que
    un valor pueda imitar."""
    instante = _ahora(ahora)
    vigentes: list[Hecho] = []
    invalidos: list[Invalido] = []
    for hecho in hechos:
        motivo = _motivo_de_invalidez(hecho, instante, ttl_s)
        if motivo is None:
            vigentes.append(hecho)
        else:
            invalidos.append(Invalido(hecho, motivo))
    return Bloque(ahora=ahora, ttl_s=ttl_s, vigentes=tuple(vigentes),
                  invalidos=tuple(invalidos), no_derivados=tuple(no_derivados))
