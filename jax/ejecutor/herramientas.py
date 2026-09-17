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
  `HerramientaRechazada` con un `Motivo` (código y datos). Nunca devuelve un
  número dudoso: la salida cruda de una herramienta también se entrega (piso
  §2.3), y un número falso ahí sería una invención con apariencia de captura.

FORMATO DE MÁQUINA NEUTRO (DECISIÓN 2026-09-16)
------------------------------------------------
La línea de salida es la captura que se cita LITERAL: no es interfaz, es
evidencia, como la salida en inglés de `ss`. Traducirla rompería la cita y
escribirla en castellano chocaría con la política de cero textos visibles
hardcodeados. Por eso no lleva idioma:

    linea  := campo (" " campo)*          -- un solo espacio entre campos
    campo  := clave "=" valor
    clave  := [a-z_]+                     -- fija, en orden fijo, por herramienta
    valor  := literal JSON en ASCII:
              cadena  -> json.dumps(ensure_ascii=True): entre comillas dobles,
                         con `"` `\\` y todo carácter no ASCII o de control
                         escritos como escape (`\\n`, `\\u2028`, `\\u202e`, `\\u00f1`)
              entero / decimal -> dígitos sin comillas, sin notación científica
              booleano -> true | false;  ausente -> null
              tupla   -> arreglo JSON de valores

Con eso: cada valor se recupera exacto con `json.loads`; un espacio, un `=` o
un `count=99` dentro de una cadena no fabrica otro campo (va entre comillas);
la línea es ASCII imprimible, así que no se parte (`str.splitlines`) ni se
dibuja distinto de lo que dice; y todo número sale como TOKEN ENTERO según
`cita._TOKEN` (`=` y espacio separan), así que `14` o `95.6` se citan como
`dato`.

`contar`:
    salida:  count=14 total=112 mode="termina_en" pattern=".ssl.conf"
             literal=true case_sensitive=true source_machine="atemai"
             source_command="ls /etc/nginx/conf.d/domains/"
    comando: contar mode=… pattern=… literal=true case_sensitive=true
             source_machine=… source_command=… source_moment="2026-…"
`convertir`:
    salida:  value=89 from="GiB" to="GB" result=95.6 exact=95.563022336
             factor_from=1073741824 factor_to=1000000000 rounding="half_up"
             decimals=1
    comando: convertir value=89 from="GiB" to="GB" rounding="half_up" decimals=1
(cada una, en UNA línea). `factor_*` son bytes por unidad. `result` lleva
exactamente `decimals` decimales; `value` y `exact` van sin ceros de relleno.
El nombre de la herramienta al frente del comando es un identificador, como
`ss` o `df`.

`str(HerramientaRechazada)` usa la misma gramática: el código y después sus
datos como campos (`origen_truncado motivos=["tope_bytes"] comando="…" …`).

Sólo biblioteca estándar, sin E/S, sin reloj.
"""
from __future__ import annotations

import json
from decimal import ROUND_HALF_UP, Decimal, Inexact, InvalidOperation, localcontext

from jax.ejecutor.captura import CapturaCompleta
from jax.ejecutor.cita import Motivo

# Códigos de `HerramientaRechazada.motivo`: claves estables, como los de
# `hechos` y `cita`. Los datos de entrada que pueden ser de cualquier tipo
# (modo, unidad, valor, decimales) van como `repr`: el tipo equivocado también
# es la causa.


MODO_DESCONOCIDO = "modo_desconocido"
ORIGEN_TRUNCADO = "origen_truncado"
ORIGEN_CON_CODIGO_NO_CERO = "origen_con_codigo_no_cero"
PATRON_VACIO = "patron_vacio"
PATRON_MULTILINEA = "patron_multilinea"
SALIDA_MULTILINEA = "salida_multilinea"
UNIDAD_DESCONOCIDA = "unidad_desconocida"
VALOR_NO_NUMERICO = "valor_no_numerico"
VALOR_FUERA_DE_RANGO = "valor_fuera_de_rango"
SIN_MAQUINA = "sin_maquina"
DECIMALES_INVALIDOS = "decimales_invalidos"
PRECISION_EXCEDIDA = "precision_excedida"


def _valor_neutro(valor) -> str:
    # bool antes que int: en Python True es un int.
    if isinstance(valor, bool):
        return "true" if valor else "false"
    if valor is None:
        return "null"
    if isinstance(valor, int):
        return str(valor)
    if isinstance(valor, Decimal):
        # `f`: sin notación científica. Quien llama decide los ceros (ver
        # `_texto`): `result` conserva los decimales pedidos.
        return format(valor, "f")
    if isinstance(valor, str):
        return json.dumps(valor, ensure_ascii=True)
    if isinstance(valor, (tuple, list)):
        return "[" + ",".join(_valor_neutro(v) for v in valor) + "]"
    # Nada más entra hoy; si entrara, como cadena y no como texto suelto.
    return json.dumps(repr(valor), ensure_ascii=True)


def _campos(pares) -> str:
    """`clave=valor` separados por un espacio. Gramática: docstring del módulo."""
    return " ".join(f"{clave}={_valor_neutro(valor)}" for clave, valor in pares)


class HerramientaRechazada(ValueError):
    """La herramienta se niega a producir un resultado. `motivo` dice por qué,
    como código y datos; `str()` es ese motivo en formato de máquina."""

    def __init__(self, motivo: Motivo):
        self.motivo = motivo
        super().__init__(" ".join(filter(None, (motivo.codigo, _campos(motivo.datos)))))


def _una_linea(salida: str) -> str:
    # Defensa final del contrato: la salida tiene que ser exactamente una
    # línea para `str.splitlines`, que es lo que usa `cita.verificar`. Un
    # `raise` y no un `assert`: con `python -O` el assert desaparece.
    if salida.splitlines() != [salida]:
        raise HerramientaRechazada(Motivo(SALIDA_MULTILINEA, (("salida", salida),)))
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

    Salida y comando en el formato neutro del docstring del módulo: `count`
    y `total` como números citables, y todas las entradas.

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
        raise HerramientaRechazada(Motivo(MODO_DESCONOCIDO, (
            ("modo", repr(modo)), ("aceptados", tuple(_COINCIDE)))))
    donde = (("comando", captura_origen.comando), ("maquina", captura_origen.maquina))
    if captura_origen.truncada:
        # Tupla vacía = sin motivo registrado, y lo dice la tupla vacía.
        raise HerramientaRechazada(Motivo(ORIGEN_TRUNCADO, (
            ("motivos", tuple(captura_origen.motivos_truncado)),) + donde))
    # `None` (el proceso no terminó) también es distinto de 0.
    if captura_origen.codigo != 0:
        raise HerramientaRechazada(Motivo(ORIGEN_CON_CODIGO_NO_CERO, (
            ("codigo", captura_origen.codigo),) + donde))
    if not isinstance(patron, str) or patron == "":
        # Toda línea contiene el patrón vacío.
        raise HerramientaRechazada(Motivo(PATRON_VACIO))
    if patron.splitlines() != [patron]:
        # Ninguna línea puede contener un salto de línea.
        raise HerramientaRechazada(Motivo(PATRON_MULTILINEA, (("patron", patron),)))

    coincide = _COINCIDE[modo]
    lineas = captura_origen.salida.splitlines()
    coinciden = sum(1 for linea in lineas if coincide(linea, patron))
    criterio = (("mode", modo), ("pattern", patron), ("literal", True),
                ("case_sensitive", True))
    origen = (("source_machine", captura_origen.maquina),
              ("source_command", captura_origen.comando))
    comando = "contar " + _campos(criterio + origen + (("source_moment", captura_origen.momento),))
    salida = _campos((("count", coinciden), ("total", len(lineas))) + criterio + origen)
    return _captura(captura_origen.maquina, comando, salida, captura_origen.momento)


# --- convertir --------------------------------------------------------------

_PRECISION = 200

# Nombres EXACTOS. «G» (df -h, free -h) no está a propósito: según la
# herramienta significa GiB o GB, y adivinarlo es el error que esto evita.
_BYTES_POR_UNIDAD: dict[str, int] = {
    "B": 1,
    "KB": 1000, "MB": 1000 ** 2, "GB": 1000 ** 3, "TB": 1000 ** 4, "PB": 1000 ** 5,
    "KiB": 1024, "MiB": 1024 ** 2, "GiB": 1024 ** 3, "TiB": 1024 ** 4, "PiB": 1024 ** 5,
}


def _unidad(nombre) -> int:
    if not isinstance(nombre, str) or nombre not in _BYTES_POR_UNIDAD:
        raise HerramientaRechazada(Motivo(UNIDAD_DESCONOCIDA, (
            ("unidad", repr(nombre)), ("aceptadas", tuple(_BYTES_POR_UNIDAD)))))
    return _BYTES_POR_UNIDAD[nombre]


def _valor(valor) -> Decimal:
    # bool es int en Python: True no es un tamaño.
    if isinstance(valor, bool) or not isinstance(valor, (int, float, str, Decimal)):
        raise HerramientaRechazada(Motivo(VALOR_NO_NUMERICO, (("valor", repr(valor)),)))
    try:
        # float pasa por str para no arrastrar el error binario (0.1 -> 0.1).
        numero = Decimal(str(valor) if isinstance(valor, float) else valor)
    except InvalidOperation:
        # «89,5» cae aquí: la coma es ambigua (¿decimal o miles?), no se
        # adivina. Se acepta decimal con punto, sin separador de miles.
        raise HerramientaRechazada(Motivo(VALOR_NO_NUMERICO, (("valor", repr(valor)),))) from None
    if not numero.is_finite() or numero < 0:
        raise HerramientaRechazada(Motivo(VALOR_FUERA_DE_RANGO, (("valor", repr(valor)),)))
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
    los factores y el valor exacto, en el formato neutro del docstring del
    módulo.

    `maquina` es de dónde viene el valor: el verificador sólo acepta una cita
    si coincide la máquina. La herramienta garantiza la ARITMÉTICA, no la
    procedencia del valor de entrada: ése se cita aparte.
    """
    if not isinstance(maquina, str) or not maquina.strip():
        # Sin máquina la captura no tendría procedencia.
        raise HerramientaRechazada(Motivo(SIN_MAQUINA))
    if isinstance(decimales, bool) or not isinstance(decimales, int) or decimales < 0:
        raise HerramientaRechazada(Motivo(DECIMALES_INVALIDOS, (("decimales", repr(decimales)),)))
    numero = _valor(valor)
    bytes_desde = _unidad(desde)
    bytes_hacia = _unidad(hacia)

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
            raise HerramientaRechazada(Motivo(PRECISION_EXCEDIDA, (
                ("valor", repr(valor)), ("decimales", decimales),
                ("precision", _PRECISION)))) from None
    entrada = (("value", Decimal(_texto(numero))), ("from", desde), ("to", hacia))
    redondeo = (("rounding", "half_up"), ("decimals", decimales))
    comando = "convertir " + _campos(entrada + redondeo)
    salida = _campos(entrada + (
        ("result", redondeado), ("exact", Decimal(_texto(exacto))),
        ("factor_from", bytes_desde), ("factor_to", bytes_hacia)) + redondeo)
    # Sin momento propio: una conversión no envejece, envejece el valor de
    # entrada, que se cita con su propia captura. Vacío = ilegible para
    # `hechos.py`, que lo deja fuera (fallo cerrado), nunca como vigente.
    return _captura(maquina, comando, salida, momento="")
