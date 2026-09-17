"""Herramientas cuya salida se captura (Ejecutor Fase 2).

DECISIÓN de Fernando (2026-09-16, con GO), a partir de la medición contra el
corpus de U3 (spec §2.2 «REFUTADO POR MEDICIÓN» y §8 riesgo 3): hay datos
verdaderos que ninguna línea de ninguna salida imprime -- los conteos de la
tarea 8 («14 servicios con certificado SSL») y las conversiones de unidades
(«89 GiB» escrito de cabeza como «~91 GB» en la tarea 2, una invención real).

La respuesta NO es aflojar el verificador: es que la operación la haga una
herramienta auditable y deje su resultado en una `CapturaCompleta`, el mismo
tipo que produce un comando. Así «14» o «95.6 GB» quedan literal en una línea
de una captura y se citan como cualquier otro dato.

Contrato de cada herramienta:
- `comando` describe la operación y TODAS sus entradas, para reproducirla.
- `salida` es UNA línea que imprime el resultado junto con sus entradas.
- Ante una entrada que no permite un resultado cierto, lanza
  `HerramientaRechazada`. Nunca devuelve un número dudoso: la salida cruda de
  una herramienta también se entrega (piso §2.3), y un número falso ahí sería
  una invención con apariencia de captura.

Sólo biblioteca estándar, sin E/S, sin reloj.
"""
from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal, Inexact, InvalidOperation, localcontext

from jax.ejecutor.captura import CapturaCompleta


class HerramientaRechazada(ValueError):
    """La herramienta se niega a producir un resultado. El mensaje dice por qué."""


def _una_linea(salida: str) -> str:
    # Defensa final del contrato: la salida tiene que ser exactamente una
    # línea para `str.splitlines`, que es lo que usa `cita.verificar`. Un
    # `raise` y no un `assert`: con `python -O` el assert desaparece.
    if salida.splitlines() != [salida]:
        raise HerramientaRechazada(f"la salida no cabe en una línea citable: {salida!r}")
    return salida + "\n"


def _captura(maquina: str, comando: str, salida: str, momento: str) -> CapturaCompleta:
    salida = _una_linea(salida)
    return CapturaCompleta(
        maquina=maquina, comando=comando, codigo=0, salida=salida, stderr="",
        truncada=False, bytes_totales=len(salida.encode()), momento=momento)


# --- contar -----------------------------------------------------------------

# Modos de coincidencia de `contar`. Sólo los que el corpus justifica:
# - `subcadena`: el patrón aparece en cualquier lugar de la línea.
# - `termina_en`: la línea TERMINA en el patrón. Medido contra U3, tarea 8: el
#   criterio del modelo fue «los archivos `.ssl.conf`», y por subcadena da 42
#   de 112 y no 14, porque el nombre sale también en la cabecera `=== … ===` y
#   en el `cat: …: Permission denied`.
SUBCADENA = "subcadena"
TERMINA_EN = "termina_en"
_COINCIDE = {
    SUBCADENA: lambda linea, patron: patron in linea,
    TERMINA_EN: lambda linea, patron: linea.endswith(patron),
}


def contar(captura_origen: CapturaCompleta, patron: str, *, modo: str) -> CapturaCompleta:
    """Cuántas líneas del stdout de `captura_origen` coinciden con `patron`
    según `modo` (`SUBCADENA` o `TERMINA_EN`).

    Literal (no regex), distingue mayúsculas, sin recortar espacios: una línea
    con un espacio al final no «termina en» el patrón. Las líneas se parten
    con `str.splitlines`, igual que el verificador de citas.

    `modo` es obligatorio y queda ESCRITO en el comando y en la salida: la
    cita dice exactamente qué se contó. No hay modo por defecto que se elija
    en silencio; un modo desconocido se rechaza.

    Se niega si la captura de origen está truncada (tarea 9 de U3: un conteo
    sobre 2 KB de 85,9 KB) o si el comando no terminó con código 0 (un ssh
    caído deja stdout vacío y el conteo diría «0»).

    `momento` es el de la captura de origen: el dato tiene la edad de la
    salida que se contó, no la del conteo.
    """
    if not isinstance(modo, str) or modo not in _COINCIDE:
        raise HerramientaRechazada(
            f"modo de coincidencia desconocido {modo!r}; se aceptan exactamente: "
            f"{', '.join(_COINCIDE)}")
    if captura_origen.truncada:
        motivos = ", ".join(captura_origen.motivos_truncado) or "sin motivo registrado"
        raise HerramientaRechazada(
            f"no se cuenta sobre una captura truncada ({motivos}): "
            f"{captura_origen.comando!r} en {captura_origen.maquina!r}")
    if captura_origen.codigo != 0:
        raise HerramientaRechazada(
            f"no se cuenta sobre un comando que terminó con código "
            f"{captura_origen.codigo}: {captura_origen.comando!r} en "
            f"{captura_origen.maquina!r}")
    if not isinstance(patron, str) or patron == "":
        raise HerramientaRechazada("patrón vacío: toda línea lo contiene")
    if patron.splitlines() != [patron]:
        raise HerramientaRechazada(
            f"el patrón {patron!r} tiene un salto de línea: ninguna línea puede contenerlo")

    coincide = _COINCIDE[modo]
    lineas = captura_origen.salida.splitlines()
    coinciden = sum(1 for linea in lineas if coincide(linea, patron))
    # `!r` escapa saltos de línea y separadores Unicode del patrón y del
    # comando de origen: la línea citable no se puede partir.
    origen = (f"la salida de {captura_origen.comando!r} en "
              f"{captura_origen.maquina!r}")
    criterio = f"{patron!r} (modo={modo}, literal, distingue mayusculas)"
    comando = f"contar lineas que coinciden con {criterio} en {origen} capturada {captura_origen.momento}"
    salida = f"{coinciden} lineas de {len(lineas)} coinciden con {criterio} en {origen}"
    return _captura(captura_origen.maquina, comando, salida, captura_origen.momento)


# --- convertir --------------------------------------------------------------

_PRECISION = 200

# Nombres EXACTOS. «G» (df -h, free -h) no está a propósito: según la
# herramienta significa GiB o GB, y adivinarlo es el error que esto evita.
_BYTES_POR_UNIDAD: dict[str, tuple[int, str]] = {
    "B": (1, "1"),
    "KB": (1000, "1000"), "MB": (1000 ** 2, "1000^2"), "GB": (1000 ** 3, "1000^3"),
    "TB": (1000 ** 4, "1000^4"), "PB": (1000 ** 5, "1000^5"),
    "KiB": (1024, "1024"), "MiB": (1024 ** 2, "1024^2"), "GiB": (1024 ** 3, "1024^3"),
    "TiB": (1024 ** 4, "1024^4"), "PiB": (1024 ** 5, "1024^5"),
}


def _unidad(nombre) -> tuple[int, str]:
    if not isinstance(nombre, str) or nombre not in _BYTES_POR_UNIDAD:
        raise HerramientaRechazada(
            f"unidad desconocida {nombre!r}; se aceptan exactamente: "
            f"{', '.join(_BYTES_POR_UNIDAD)}")
    return _BYTES_POR_UNIDAD[nombre]


def _valor(valor) -> Decimal:
    # bool es int en Python: True no es un tamaño.
    if isinstance(valor, bool) or not isinstance(valor, (int, float, str, Decimal)):
        raise HerramientaRechazada(f"valor no numérico: {valor!r}")
    try:
        # float pasa por str para no arrastrar el error binario (0.1 -> 0.1).
        numero = Decimal(str(valor) if isinstance(valor, float) else valor)
    except InvalidOperation:
        # «89,5» cae aquí: la coma es ambigua (¿decimal o miles?), no se adivina.
        raise HerramientaRechazada(
            f"valor no numérico: {valor!r} (decimal con punto, sin separador de miles)"
        ) from None
    if not numero.is_finite() or numero < 0:
        raise HerramientaRechazada(f"valor fuera de rango para un tamaño: {valor!r}")
    return numero


def _texto(numero: Decimal) -> str:
    # Sin notación científica y sin ceros de relleno: 89 y no 89.000 ni 8.9E+1.
    texto = format(numero, "f")
    if "." in texto:
        texto = texto.rstrip("0").rstrip(".")
    return texto


def convertir(valor, desde: str, hacia: str, *, maquina: str,
              decimales: int = 1) -> CapturaCompleta:
    """Convierte un tamaño de almacenamiento entre unidades.

    GiB = 1024^3 bytes y GB = 1000^3: 89 GiB son 95.6 GB, no «~91 GB».
    Redondeo mitad hacia arriba a `decimales`, escrito en la salida junto con
    los factores y el valor exacto.

    `maquina` es de dónde viene el valor: el verificador sólo acepta una cita
    si coincide la máquina. La herramienta garantiza la ARITMÉTICA, no la
    procedencia del valor de entrada: ése se cita aparte.
    """
    if not isinstance(maquina, str) or not maquina.strip():
        raise HerramientaRechazada("sin máquina: la captura no tendría procedencia")
    if isinstance(decimales, bool) or not isinstance(decimales, int) or decimales < 0:
        raise HerramientaRechazada(f"decimales inválidos: {decimales!r}")
    numero = _valor(valor)
    bytes_desde, factor_desde = _unidad(desde)
    bytes_hacia, factor_hacia = _unidad(hacia)

    # Factores potencias de 2 y de 10: la división siempre termina. Con
    # precisión holgada y `Inexact` atrapado, un resultado que no quepa lanza
    # en vez de redondearse en silencio (el contexto por defecto tiene 28
    # dígitos).
    with localcontext() as contexto:
        contexto.prec = _PRECISION
        contexto.traps[Inexact] = True
        try:
            exacto = numero * bytes_desde / bytes_hacia
            # El redondeo pedido SÍ es inexacto por definición: sin la trampa.
            contexto.traps[Inexact] = False
            redondeado = exacto.quantize(Decimal(1).scaleb(-decimales),
                                         rounding=ROUND_HALF_UP)
        except (Inexact, InvalidOperation):
            raise HerramientaRechazada(
                f"valor {valor!r} con {decimales} decimales excede la precisión "
                f"exacta ({_PRECISION} dígitos)") from None
    plural = "decimal" if decimales == 1 else "decimales"
    entrada = f"{_texto(numero)} {desde}"
    comando = f"convertir {entrada} a {hacia} ({decimales} {plural})"
    salida = (f"{entrada} = {format(redondeado, 'f')} {hacia} "
              f"(1 {desde} = {factor_desde} bytes, 1 {hacia} = {factor_hacia} bytes; "
              f"redondeo a {decimales} {plural}, mitad hacia arriba; "
              f"exacto: {_texto(exacto)})")
    # Sin momento propio: una conversión no envejece, envejece el valor de
    # entrada, que se cita con su propia captura. Vacío = ilegible para
    # `hechos.py`, que lo deja fuera (fallo cerrado), nunca como vigente.
    return _captura(maquina, comando, salida, momento="")
