"""Verificador de citas del Ejecutor (Fase 2).

Spec: docs/superpowers/specs/2026-09-16-ejecutor-fase2-design.md §3.3.

Puro a propósito: sin red, sin E/S, sin reloj, sólo biblioteca estándar.
Verificar una cita es una BÚSQUEDA DE SUBCADENA, no un juicio sobre la
verdad -- por eso no hay nada que calibrar y no existe el falso positivo
por umbral mal puesto.

Garantiza PROCEDENCIA, no CORRECCIÓN: una afirmación puede citar una línea
real y aun así concluir mal a partir de ella (riesgo 2 del spec).

LIGADURA AFIRMACIÓN ↔ CITA (decisión de Fernando, 2026-09-16, con GO)
-------------------------------------------------------------------
La primera versión nunca leía `texto`: «el servidor está en Marte» citando una
línea real de `free -h` salía respaldada (V1 medido = 0 de 11). Ahora cada
afirmación dice qué `dato` literal afirma y se exige, ANTES de mirar capturas:

1. `dato` no vacío tras normalizar (tercer bypass por vacío, después de la
   cita y de la máquina).
2. `dato` literal dentro de la `linea` citada   → si no: `dato_fuera_de_linea`.
3. `dato` literal dentro de `texto`             → si no: `dato_fuera_de_texto`.
4. CADA número de `texto` está, entero, entre los números de la `linea` o
   sus tramos partidos por `:` `-` `/`
                                                → si no: `numero_sin_respaldo`.

La regla 4 es la que importa: sin ella el modelo verifica `dato="89Gi"` y
escribe «hay 89Gi de RAM, o sea 512 TB», y la persona lee el texto.

Qué es un número: ver `numeros`; qué lo respalda: `numeros_que_respaldan`.

LÍMITES CONOCIDOS (medidos, no arreglados; tests `LIMITE_*` en xfail estricto):
- Las reglas 2 y 3 las cumple cualquier subcadena común, hasta una letra. Si
  el texto no tiene números, nada ata su prosa a la línea: el verificador ata
  NÚMEROS, no conclusiones (#8, #9, #11 de U3 y las etiquetas inventadas).
- Un número escrito con palabras («quinientos») no es un número para la regla 4.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

RESPALDADA = "respaldada"
SIN_RESPALDO = "sin_respaldo"
FUENTE_TRUNCADA = "fuente_truncada"
FUENTE_INEXISTENTE = "fuente_inexistente"
DATO_FUERA_DE_LINEA = "dato_fuera_de_linea"
DATO_FUERA_DE_TEXTO = "dato_fuera_de_texto"
NUMERO_SIN_RESPALDO = "numero_sin_respaldo"


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
    texto: str
    comando: str
    linea: str
    dato: str


@dataclass(frozen=True)
class Veredicto:
    estado: str
    motivo: str


def normalizar(linea: str) -> str:
    """Recorta los laterales y colapsa espacios internos. NADA MÁS.

    No toca mayúsculas, ni puntuación, ni números: `131.074` no puede
    coincidir con `131.072` (invención real de U3, tarea 3) y `Docker` no
    puede coincidir con `docker`.
    """
    return " ".join(linea.split())


# Dígitos, y separadores `.` `,` `:` `-` `/` sólo ENTRE dos dígitos.
_NUMERO = re.compile(r"\d+(?:[.,:/-]\d+)*")


def numeros(texto: str) -> tuple[str, ...]:
    """Los números de un texto, en orden, como tokens maximales.

    Número = secuencia de dígitos que puede llevar separadores internos
    `.` `,` `:` `-` `/`, cada uno ENTRE dos dígitos. Por qué cada cosa:

    - `.` y `,`: miles, decimales, versiones e IPs. `131.074`, `131,074` y
      `131072` son tres literales distintos: se compara la cadena, no el valor,
      porque el literal es lo que la persona lee.
    - `:`: ata IP y puerto (`172.16.20.11:3001`) y horas (`7:02`). Sin él, una
      IP de una columna y un puerto de otra respaldarían un par que no existe
      (la clase de error de la invención #8 de U3).
    - `-` y `/`: atan fechas (`2026-09-16`), versiones (`6.8.0-139`) y pares
      (`12/24`). Sin ellos, el `16` de una hora respaldaría el día de la fecha.
    - Un separador al final o seguido de espacio NO es interno: `3001.` da
      `3001`, `1, 2 y 3` da tres números.
    - Las letras cortan pero no esconden: `89Gi` da `89`, `512TB` da `512`,
      `sda1` da `1`. Un número pegado a una unidad sigue siendo un número.
    - Del lado seguro: los dígitos Unicode también cuentan; un formato que junta de
      más (`1,2,3` sin espacios) da un solo token que tiene que estar tal cual.
    """
    return tuple(_NUMERO.findall(texto))


# Separadores que unen CAMPOS independientes en la salida de un comando.
_ENTRE_CAMPOS = re.compile(r"[:/-]")


def numeros_que_respaldan(linea: str) -> frozenset[str]:
    """Qué números del texto quedan respaldados por esta línea.

    Cada número de la línea (ver `numeros`) entero, y además cada tramo
    contiguo que resulta de partirlo por `:`, `-` o `/`:

    - `0.0.0.0:8188` respalda `0.0.0.0:8188`, `0.0.0.0` y `8188`. `ss` imprime
      así los puertos; sin esto, «el puerto 8188» no tendría respaldo en la
      línea que lo prueba (medido en U3, tarea 5).
    - `6.8.0-139` respalda `6.8.0` y `139`; `16:00:05` respalda `16:00`.
    - `.` y `,` NO parten: `131.072` no respalda `131`, `24.04.5` no respalda
      `24.04`. Son una sola cantidad, no campos.
    - La unión en el texto se sigue exigiendo entera: `172.16.20.11:3001` no la
      respalda una línea con `172.16.20.11:8080` y `0.0.0.0:3001`.
    """
    respaldo: set[str] = set()
    for n in numeros(linea):
        partes = _ENTRE_CAMPOS.split(n)
        seps = _ENTRE_CAMPOS.findall(n)
        for i in range(len(partes)):
            tramo = partes[i]
            respaldo.add(tramo)
            for j in range(i + 1, len(partes)):
                tramo += seps[j - 1] + partes[j]
                respaldo.add(tramo)
    return frozenset(respaldo)


def verificar(afirmacion: Afirmacion, capturas) -> Veredicto:
    """¿La afirmación está ligada a su cita, y la cita está literal en la salida
    de ese comando, en esa máquina?

    Primero la ligadura (docstring del módulo, reglas 1-4); después la fuente.

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
    if dato not in aguja:
        return Veredicto(DATO_FUERA_DE_LINEA,
                         f"el dato {dato!r} no está en la línea citada")
    if dato not in normalizar(afirmacion.texto):
        return Veredicto(DATO_FUERA_DE_TEXTO,
                         f"el dato {dato!r} no está en el texto de la afirmación")
    en_linea = numeros_que_respaldan(aguja)
    sueltos = [n for n in dict.fromkeys(numeros(afirmacion.texto)) if n not in en_linea]
    if sueltos:
        return Veredicto(NUMERO_SIN_RESPALDO,
                         "el texto afirma números que no están en la línea citada: "
                         + ", ".join(sueltos))
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
                    return Veredicto(RESPALDADA, "")
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
