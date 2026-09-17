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
   vacío hallados al implementar).
4. `dato` literal dentro de la `linea` citada, SIN CORTAR UN TOKEN
   → si no: `dato_fuera_de_linea`. Ver `_esta_entero`.
5. máquina y comando coinciden con una captura; el truncado se mira ANTES que
   el contenido; stdout y stderr se recorren por separado.

La regla 4 sin bordes (subcadena pura, como estaba) dejaba salir `active` de
`inactive` y `3107` de `131072`: con la prosa ya fuera, el dato es lo ÚNICO que
escribe el modelo, y una subcadena mal cortada es un dato falso sobre una línea
verdadera. Hallado 2026-09-16 al quitar la prosa.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

RESPALDADA = "respaldada"
SIN_RESPALDO = "sin_respaldo"
FUENTE_TRUNCADA = "fuente_truncada"
FUENTE_INEXISTENTE = "fuente_inexistente"
DATO_FUERA_DE_LINEA = "dato_fuera_de_linea"


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


@dataclass(frozen=True)
class Veredicto:
    estado: str
    motivo: str
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
        return Veredicto(SIN_RESPALDO, "la afirmación no cita ninguna línea")
    # Lo mismo con la máquina: vacía no identifica nada, y `"" == ""` dejaría
    # que una captura sin procedencia respalde una afirmación sin procedencia.
    if not afirmacion.maquina.strip():
        return Veredicto(SIN_RESPALDO, "la afirmación no dice de qué máquina viene")
    # La ligadura se mira ANTES que las capturas: una afirmación incoherente
    # consigo misma se rechaza por eso, sea cual sea la fuente.
    dato = normalizar(afirmacion.dato)
    if not dato:
        return Veredicto(SIN_RESPALDO, "la afirmación no dice qué dato afirma")
    if not _esta_entero(dato, aguja):
        return Veredicto(DATO_FUERA_DE_LINEA,
                         f"el dato {dato!r} no está entero en la línea citada")
    se_corrio = False
    alguna_truncada = False
    for captura in capturas:
        if (captura.maquina, captura.comando) != (afirmacion.maquina, afirmacion.comando):
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
                    return Veredicto(RESPALDADA, "", linea_capturada=linea)
    if alguna_truncada:
        return Veredicto(FUENTE_TRUNCADA,
                         f"la salida de {afirmacion.comando!r} en "
                         f"{afirmacion.maquina!r} vino truncada")
    if se_corrio:
        return Veredicto(SIN_RESPALDO,
                         f"la línea citada no está en la salida de "
                         f"{afirmacion.comando!r} en {afirmacion.maquina!r}")
    return Veredicto(FUENTE_INEXISTENTE,
                     f"no se corrió el comando {afirmacion.comando!r} "
                     f"en {afirmacion.maquina!r}")


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
