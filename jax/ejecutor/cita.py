"""Verificador de citas del Ejecutor (Fase 2).

Spec: docs/superpowers/specs/2026-09-16-ejecutor-fase2-design.md §2.0 y §3.3.

Puro a propósito: sin red, sin E/S, sin reloj, sólo biblioteca estándar.

Garantiza PROCEDENCIA, no CORRECCIÓN: un dato puede estar literal en una línea
real y aun así esa línea no decir lo que alguien concluye de ella (riesgo 2).

EL EJECUTOR NO ESCRIBE PROSA (DECISIÓN de Fernando, 2026-09-16, §2.0)
---------------------------------------------------------------------
Dos mediciones contra el corpus de U3 mostraron que una regla literal ata
NÚMEROS y no PALABRAS: «el 8188 es Docker multi-hilo», citando la línea real
del puerto, salía respaldada. Se quitó la superficie de ataque en vez de
vigilarla: la `Afirmacion` ya no tiene `texto`. Es `(maquina, comando, linea,
dato)`, y lo que ve la persona lo arma `presentar`, no el modelo: estructura
sin rótulos, que el frontend rotula con i18n.

Con la prosa se fueron las dos reglas que sólo la vigilaban («el dato está en
el texto» y «todo número del texto está en la línea») y sus veredictos
`dato_fuera_de_texto` y `numero_sin_respaldo`.

Qué se exige, en orden:
1. cita no vacía; 2. máquina no vacía; 3. dato no vacío (los tres bypass por
   vacío hallados al implementar); 3 bis. propósito no vacío (C5, plan 4 de SP1:
   el auditor juzga si la línea contesta la pregunta que dice contestar).
4. `dato` literal dentro de la `linea` citada, SIN CORTAR UN TOKEN
   → si no: `dato_fuera_de_linea`. Ver `_esta_entero`.
5. máquina y comando coinciden con una captura; el truncado se mira ANTES que
   el contenido; stdout y stderr se recorren por separado.

Cada rechazo lleva un `Motivo` (código estable y datos), no una frase: la
frase la pone el frontend con sus traducciones (política del ecosistema, sin
textos visibles hardcodeados). Es el mismo tipo que usa `hechos`.

La regla 4 sin bordes (subcadena pura, como estaba) dejaba salir `active` de
`inactive` y `3107` de `131072`: con la prosa ya fuera, el dato es lo ÚNICO que
escribe el modelo, y una subcadena mal cortada es un dato falso sobre una línea
verdadera. Hallado 2026-09-16 al quitar la prosa.
"""
from __future__ import annotations

import re
import shlex
from dataclasses import dataclass

RESPALDADA = "respaldada"
SIN_RESPALDO = "sin_respaldo"
FUENTE_TRUNCADA = "fuente_truncada"
FUENTE_INEXISTENTE = "fuente_inexistente"
DATO_FUERA_DE_LINEA = "dato_fuera_de_linea"

# Códigos de `Veredicto.motivo`: claves estables del contrato con el frontend,
# que pone el texto traducido. Cambiar uno rompe esa traducción. Son más finos
# que el estado (`sin_respaldo` tiene cuatro causas distintas).
LINEA_VACIA = "linea_vacia"
MAQUINA_VACIA = "maquina_vacia"
DATO_VACIO = "dato_vacio"
PROPOSITO_VACIO = "proposito_vacio"
DATO_NO_ENTERO = "dato_no_entero"
LINEA_NO_ESTA = "linea_no_esta"
COMANDO_NO_CORRIDO = "comando_no_corrido"
# `fuente_truncada` es a la vez estado y código: tiene una sola causa.


@dataclass(frozen=True)
class Motivo:
    """Por qué algo no vale: un CÓDIGO estable y sus datos, como pares
    `(clave, valor)` (inmutables; `dict(motivo.datos)` para serializar). Sin
    prosa: la frase la pone el frontend con sus traducciones.

    Vive aquí, en el módulo más bajo del Ejecutor (no importa a nadie), para
    que lo compartan `hechos`, `herramientas` y `transporte` sin ciclos:
    `hechos` importa `captura`, que importa `cita`."""
    codigo: str
    datos: tuple[tuple[str, object], ...] = ()


@dataclass(frozen=True)
class Captura:
    maquina: str
    comando: str
    salida: str
    stderr: str
    truncada: bool


@dataclass(frozen=True)
class Afirmacion:
    maquina: str
    comando: str
    linea: str
    dato: str
    # Qué pregunta de la misión dice responder `dato` (plan 4 de SP1, C5). Lo
    # escribe el modelo y NUNCA se presenta a la persona (no está en
    # `CAMPOS_PRESENTACION`): es lo que el auditor de C5 juzga. Una cita
    # verdadera no prueba que la línea conteste la pregunta («el 8188 es Docker
    # multi-hilo» citando la línea real del puerto); sin el propósito, nadie
    # puede decir que no la contesta. Obligatorio: sin él, no sale.
    proposito: str


@dataclass(frozen=True)
class Veredicto:
    estado: str
    # `None` sólo si `respaldada`; si no, el código de arriba y sus datos.
    motivo: Motivo | None
    # Sólo si `respaldada`: la línea TAL COMO LA IMPRIMIÓ la máquina (la cita
    # se compara con espacios colapsados; lo que se muestra es la original).
    linea_capturada: str = ""


def normalizar(linea: str) -> str:
    """Recorta los laterales y colapsa espacios internos. NADA MÁS.

    No toca mayúsculas, ni puntuación, ni números: `131.074` no puede
    coincidir con `131.072` (invención real de U3, tarea 3) y `Docker` no
    puede coincidir con `docker`.
    """
    return " ".join(linea.split())


# Un token es una corrida de letras, o un número que puede llevar `.` o `,`
# ENTRE dígitos (miles, decimales, versión, IP: una sola cantidad). Todo lo
# demás separa: espacios, `:` `-` `/` `_`, y el paso de letra a dígito.
_TOKEN = re.compile(r"\d+(?:[.,]\d+)*|[^\W\d_]+")


def _esta_entero(dato: str, linea: str) -> bool:
    """¿`dato` aparece en `linea` sin que sus bordes caigan DENTRO de un token?

    - `8188` en `0.0.0.0:8188` sí; `6.8.0` en `6.8.0-139` sí; `89` en `89Gi` sí.
    - `active` en `inactive` no; `3107` en `131072` no; `24.04` en `24.04.5`
      no; `M` en `Mem:` no.

    Basta una aparición con bordes limpios (`inactive active` respalda
    `active`). Se aplica sobre las formas normalizadas de las dos cadenas.
    """
    tokens = [m.span() for m in _TOKEN.finditer(linea)]

    def corta(posicion: int) -> bool:
        return any(inicio < posicion < fin for inicio, fin in tokens)

    desde = 0
    while (i := linea.find(dato, desde)) != -1:
        if not corta(i) and not corta(i + len(dato)):
            return True
        desde = i + 1
    return False


def comando_canonico(comando: str) -> str:
    """Los mismos tokens, separados por un espacio. `ssh h "df -h /"` y `ssh h df -h /` corren lo
    mismo (ssh pega sus argumentos con espacios) y el modelo escribe una u otra al citar: la real
    de prod, 2026-09-17, salió `comando_no_corrido` con el dato correcto. Si no se puede partir
    (comilla sin cerrar), no se adivina: se devuelve tal cual y se compara carácter por carácter."""
    try:
        return " ".join(shlex.split(comando))
    except ValueError:
        return comando


def mismo_comando(corrido: str, citado: str) -> bool:
    """El comando sólo ELIGE la captura; la máquina la decidió el gancho sobre el comando CORRIDO
    y la línea se busca literal en esa salida. Por eso alcanza con que sean el mismo comando."""
    return corrido == citado or comando_canonico(corrido) == comando_canonico(citado)


def verificar(afirmacion: Afirmacion, capturas) -> Veredicto:
    """¿El dato está entero en la línea citada, y la línea está literal en la
    salida de ese comando, en esa máquina?

    Una captura respalda sólo si coinciden MÁQUINA y COMANDO: un `free -h` de
    otra máquina no dice nada de ésta. La línea puede estar en stdout o en
    stderr, cada flujo por separado.

    Si el comando se corrió más de una vez, respalda cualquier captura
    COMPLETA que tenga la línea. Una captura truncada no respalda nada.
    """
    aguja = normalizar(afirmacion.linea)
    # Una cita vacía no cita nada: `""` es igual a cualquier línea en blanco
    # de la salida y dejaría pasar cualquier afirmación inventada.
    if not aguja:
        return Veredicto(SIN_RESPALDO, Motivo(LINEA_VACIA))
    # Lo mismo con la máquina: vacía no identifica nada, y `"" == ""` dejaría
    # que una captura sin procedencia respalde una afirmación sin procedencia.
    if not afirmacion.maquina.strip():
        return Veredicto(SIN_RESPALDO, Motivo(MAQUINA_VACIA))
    # La ligadura se mira ANTES que las capturas: una afirmación incoherente
    # consigo misma se rechaza por eso, sea cual sea la fuente.
    dato = normalizar(afirmacion.dato)
    if not dato:
        return Veredicto(SIN_RESPALDO, Motivo(DATO_VACIO))
    # Sin la pregunta que dice responder, C5 no la puede juzgar: no sale.
    if not afirmacion.proposito.strip():
        return Veredicto(SIN_RESPALDO, Motivo(PROPOSITO_VACIO))
    if not _esta_entero(dato, aguja):
        return Veredicto(DATO_FUERA_DE_LINEA, Motivo(DATO_NO_ENTERO, (("dato", dato),)))
    se_corrio = False
    alguna_truncada = False
    for captura in capturas:
        if (captura.maquina != afirmacion.maquina
                or not mismo_comando(captura.comando, afirmacion.comando)):
            continue
        se_corrio = True
        # El truncado se mira ANTES que el contenido: si la salida vino
        # cortada, dar por bueno lo que sí llegó es exactamente el error de
        # la tarea 9 (2 KB leídos de 85,9 KB).
        if captura.truncada:
            alguna_truncada = True
            continue
        # stdout y stderr se recorren POR SEPARADO. Pegarlos en un solo texto
        # fabrica, si stdout no termina en salto de línea, una línea que la
        # máquina nunca imprimió en ningún flujo.
        for flujo in (captura.salida, captura.stderr):
            for linea in flujo.splitlines():
                if normalizar(linea) == aguja:
                    return Veredicto(RESPALDADA, None, linea_capturada=linea)
    donde = (("comando", afirmacion.comando), ("maquina", afirmacion.maquina))
    if alguna_truncada:
        return Veredicto(FUENTE_TRUNCADA, Motivo(FUENTE_TRUNCADA, donde))
    if se_corrio:
        return Veredicto(SIN_RESPALDO, Motivo(LINEA_NO_ESTA, donde))
    return Veredicto(FUENTE_INEXISTENTE, Motivo(COMANDO_NO_CORRIDO, donde))


# Claves estables del contrato con el frontend: los rótulos visibles («dato»,
# «máquina», «comando», «línea») viven en las traducciones de jax-platform.
CAMPOS_PRESENTACION = ("dato", "maquina", "comando", "linea")


@dataclass(frozen=True)
class Presentacion:
    """Lo que ve la persona de una afirmación respaldada: SÓLO VALORES.

    Cada valor es el `repr` del campo: la cadena literal entre comillas, con
    todo carácter no imprimible escrito como escape. Se recupera exacta con
    `ast.literal_eval`.
    """
    dato: str
    maquina: str
    comando: str
    linea: str


def presentar(afirmacion: Afirmacion) -> Presentacion:
    """Lo que ve la persona. Lo arma el sistema; el modelo no escribe nada aquí.

    Devuelve ESTRUCTURA, no texto rotulado (política del ecosistema: ningún
    string visible hardcodeado; el backend de jax no tiene i18n). Las claves
    son `CAMPOS_PRESENTACION` y los rótulos los pone el frontend con sus
    traducciones. Nunca resume, nunca traduce, nunca interpreta: la línea
    citada va COMPLETA.

    Por qué la inyección sigue cerrada, con estructura en vez de texto:
    - Fabricar OTRO campo (un `dato: 'falso'` colado en la línea) ya no tiene
      forma: no hay separador que parsear. Cada valor está en su clave, y lo
      que haya adentro es contenido de ESE campo, nunca un campo nuevo.
    - Lo que queda es lo VISUAL dentro de un campo: un escape de terminal
      (`\x1b[2K`) o un override bidireccional (U+202E, U+2066) puede borrar o
      dar vuelta el `No` de la línea al dibujarse, en una terminal o en un
      navegador. Por eso cada valor sigue saliendo con `repr`: todo carácter
      de control, de formato (categoría Cf, que incluye los bidi y el espacio
      de ancho cero) o separador (Zl, Zp, Zs salvo el espacio) queda escrito
      como `\n`, `\u202e`, `\x1b`. El valor es imprimible entero
      (`str.isprintable`) y cabe en una línea, sin depender de cómo lo pinte
      cada consumidor. Las comillas delimitan: un espacio al final se ve.

    Sólo para afirmaciones ya `respaldadas`: `transporte.entregar` le pasa la
    línea tal como la imprimió la máquina (`Veredicto.linea_capturada`).
    """
    return Presentacion(
        dato=repr(afirmacion.dato),
        maquina=repr(afirmacion.maquina),
        comando=repr(afirmacion.comando),
        linea=repr(afirmacion.linea),
    )
