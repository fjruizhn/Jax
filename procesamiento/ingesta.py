"""Ingesta: el original entra a `fuente/` y su extracto queda en `procesado/`.

Dos invariantes:
  1. `fuente/` es inmutable. El original entra y no se toca mas. Dos
     archivos DISTINTOS con el mismo nombre no pueden pisarse -- ni en
     secuencia ni bajo concurrencia (C-3) -- y un symlink plantado ahi
     jamas se sigue, ni para leer ni para escribir (C-1).
  2. `procesado/` es DESECHABLE: se borra entero y se reconstruye. Nada que
     no sea reconstruible vive ahi. Un cache del que no te podes fiar para
     borrarlo no es un cache: es una segunda base de datos.

El cache se indexa por `sha256` del original, pero la huella SOLA no
alcanza para decidir si un acierto de cache es valido -- ver
`_ficha_de_cache_valida` (I-1 a I-5).

=====================================================================
Ronda de arreglo 1 (2026-09-21, task-8-hallazgos.md -- 3 Criticos, 8
Importantes, veredicto spec NO, sin ratificar nada):
=====================================================================

- **C-1 (incumplimiento de spec §4):** el draft resolvia el destino con
  `candidato.exists()`, que SIGUE symlinks -- un enlace roto en `fuente/`
  hacia que la copia escribiera *a traves* del enlace, fuera del
  workspace, con contenido de un cliente. El spec manda reusar
  `tool_authority.resolve_jailed_path` (ya probado con casos
  adversariales) en vez de escribir codigo de seguridad nuevo. Acá se
  reusa para DOS cosas: (a) `trabajo` se valida contra el jail antes de
  tocar el filesystem (cierra I-7 de paso), y (b) cada nombre candidato en
  `fuente/` se valida igual. Ademas, defensa en profundidad: la apertura
  real usaba `O_NOFOLLOW` (nunca sigue un symlink al escribir, exista o no
  su destino) -- por si el jail se equivocara. *(Superado en ronda 2: la
  escritura ya no usa `os.open` -- ver más abajo.)*

  `resolve_jailed_path` exige un path RELATIVO (es el contrato de los
  tool_calls, que nunca traen rutas absolutas). La ingesta SI trabaja con
  rutas absolutas (`trabajo` es una carpeta real del filesystem), asi que
  `_resolver_bajo_jail` hace la traduccion: resuelve la forma canonica,
  la expresa relativa a `WORKSPACE_ROOT`, y RECIEN AHI delega el chequeo
  de fondo -- no se reescribe la logica de "sigue symlinks y compara
  contra el jail", esa sigue viviendo en un solo lugar.

- **C-2 (el defecto que este PR vino a cerrar, reabierto por
  concurrencia):** el temporal `<huella>.parcial` era determinista y
  COMPARTIDO -- dos ingestas del mismo archivo entrelazadas dejaban un
  `procesado/` con parte de las salidas de una y parte de la otra, con
  una ficha que afirmaba que estaban todas. Y como el segundo pase daba
  acierto de cache, nunca se curaba. Ahora el temporal es
  `<huella>.<pid>-<uuid4>.parcial` -- unico por LLAMADA, nunca
  compartido -- y se borra en un `finally` (cierra I-8 de paso: no quedan
  temporales huerfanos si algo revienta a mitad de camino).

- **C-3 (mismo defecto de nombres del draft original, reabierto por
  concurrencia):** entre "decidir el nombre" y "copiar" habia una ventana
  -- dos ingestas concurrentes de archivos DISTINTOS con el mismo nombre
  podian terminar con un solo archivo en `fuente/` y una ficha que
  MENTIA sobre cual era. La creacion se hizo ATOMICA con
  `O_CREAT|O_EXCL|O_NOFOLLOW` *(ronda 2: reemplazado por `os.link()`, ver
  más abajo)*. La comprobacion "no existe" y la creacion son la MISMA
  operacion del kernel -- no hay ventana que otro escritor pueda
  colarse. Si el nombre ya esta ocupado (por otro contenido, o por un
  symlink que jamas se sigue), se prueba el siguiente nombre candidato;
  el que ya estaba ahi no se toca nunca.

- **I-1 (el mas revelador -- la prueba de que era un accidente, no una
  decision, es que la mutacion "no cachear cuando no hubo salidas" pasaba
  los 8 tests en verde):** antes, CUALQUIER estado se cacheaba para
  siempre -- un `error` por "tesseract no instalado" quedaba condenado
  aunque se instalara tesseract un minuto despues. Regla nueva: `ok` y
  `parcial` se cachean, `error` y `sin_extractor` NO. Un error no es un
  extracto reconstruible, es la AUSENCIA de uno, y cachear una ausencia
  como si fuera trabajo hecho es lo que la volvia permanente. La ficha
  se escribe igual (queda el registro de que se intento y por que
  fallo); lo que cambia es que leerla cuenta como fallo de cache y se
  reintenta.

- **I-2 (invalidacion por version del extractor):** cambiar el
  extractor (o su version) no invalidaba una ficha ya escrita -- los
  Criticos de las rondas anteriores de este mismo proyecto NO alcanzaban
  a un archivo ya ingerido. Al leer, se compara `extractor_version`
  contra lo que el extractor VIGENTE reporta hoy (consulta barata: leer
  un `__version__` ya importado, no releer el archivo) -- si no
  coincide, fallo de cache.

- **I-3 (la clave ignoraba la extension, que es lo que la compuerta usa
  para rutear):** dos archivos con los mismos bytes y distinta extension
  compartian entrada de cache. La ficha ahora guarda la extension con la
  que se ruteo (`detalle["_extension_ingesta"]`) y el acierto de cache la
  valida contra `destino.suffix`.

- **I-4 (ficha corrupta = excepcion para siempre):** una `ficha.json`
  truncada a 0 bytes (corte de luz tipico) hacia `ValueError` en CADA
  pase. Ahora una ficha ilegible o invalida es, simplemente, fallo de
  cache -- nunca una excepcion hacia el llamador.

- **I-5 (la completitud del cache era inverificable POR DISEÑO):** sin
  esto, C-2/I-4 son reparables SOLO por casualidad. La ficha ahora lista
  sus salidas (`detalle["_salidas_ingesta"]`) y el acierto de cache
  verifica que TODAS esten presentes en disco -- borrar un `.csv` de
  `procesado/` ahora se nota y se regenera.

- **I-6 (nombre largo + colision -> OSError crudo):** el tronco del
  nombre se recorta ANTES de agregarle el sufijo de huella, para que el
  resultado siempre quepa dentro del limite tipico de un filesystem
  (255).

=====================================================================
Ronda de arreglo 2 (2026-09-21, task-8-hallazgos-r2.md -- lo que la ronda
1 trajo de nuevo, y lo que destapo):
=====================================================================

- **A-1/A-2 (mismo modelo de amenaza que C-1, ahora por el camino de leer
  en vez de escribir):** `O_NOFOLLOW` cerro el symlink, pero el camino de
  comparacion de huellas volvia a "abrir lo que haya" -- un FIFO plantado
  en `fuente/` colgaba `sha256_de` PARA SIEMPRE (sin timeout); un
  directorio con el mismo nombre reventaba con `IsADirectoryError`, en
  cada pase. Arreglo unico para los dos: solo se le calcula la huella a
  un archivo REGULAR (`Path.is_file()`) -- cierra tambien socket y
  dispositivo sin enumerarlos.

- **A-3 (el reintento acotado en 10 no aguantaba lo que decia aguantar):**
  medido con hilos reales, SIN jitter, a 16 hilos fallaba 21 de 25 veces
  con un `OSError(39)` crudo -- reintentos INMEDIATOS se resincronizan
  entre si. Arreglo: (1) jitter con backoff entre reintentos; (2) un error
  de dominio (`ReemplazoProcesadoAgotado`) en vez del `OSError` crudo si
  se agotan; (3) medido de nuevo despues del arreglo -- el numero esta en
  task-8-report.md, no es una suposicion.

  **Hallazgo PROPIO durante esa misma medicion, mas alla de lo que pedia
  A-3** (no esta en task-8-hallazgos-r2.md): a mas hilos escribiendo el
  MISMO archivo, un lector que pierde la carrera de `O_CREAT|O_EXCL`
  podia ver el archivo GANADOR a mitad de escribir -- `sha256_de` lee lo
  que hay EN ESE INSTANTE, calcula una huella que no coincide (esta
  incompleto), y lo trata como "contenido distinto" cuando es el MISMO
  archivo en vuelo. `_asegurar_en_fuente` se reescribio: el contenido se
  escribe COMPLETO a un temporal propio de la llamada primero, y solo se
  PUBLICA con `os.link()` (hardlink) -- atomico igual que
  `O_CREAT|O_EXCL` (verificado empiricamente: falla con `FileExistsError`
  ante CUALQUIER cosa que ya exista en ese nombre -- symlink valido, roto,
  o archivo regular -- sin seguirlo jamas), pero sin la ventana de lectura
  a medio escribir, porque el nombre final nunca es el que se escribe.
  Este cambio estructural tambien vuelve moot dos de los seis hallazgos
  menores de A-5 (ver abajo): ya no hay ningun `os.open` con `O_NOFOLLOW`
  al que quitarle la bandera, ni una escritura directa al nombre final
  cuyo fallo a mitad de camino haya que limpiar con un `unlink` -- las
  dos preocupaciones quedaron cerradas por construccion, no por un test.

- **A-4 (el agujero estaba en el test, no en el codigo):** `test_I3_...`
  solo cubria el orden `.xlsx -> .docx`. Al reves (`.docx` primero)
  reaparecia el mismo defecto. Parametrizado en los DOS ordenes.

- **A-5 (seis mutaciones menores):** tres se atacaron con test propio
  (temporal huerfano sin limpiar en `finally`; `ficha.sha256 != huella`
  neutralizado; `.lower()` de la extension quitado). Las otras tres --
  `O_NOFOLLOW`, el `unlink` de limpieza, y el jail sobre
  `destino_literal` -- se investigaron a fondo, con evidencia empirica
  pegada en task-8-report.md: las dos primeras dejaron de existir como
  codigo ejecutable con la reescritura de arriba: la tercera
  (`_resolver_bajo_jail(destino_literal)`) se mantiene como defensa en
  profundidad DELIBERADA aunque se demostro que, con la construccion
  ACTUAL de `_nombre_candidato` (un solo componente de ruta, y
  `fuente_abs` siempre un nivel por debajo de `trabajo_abs`), un ".."
  como unico candidato posible de escape SOLO puede llegar hasta
  `trabajo_abs` -- que ya esta, por definicion, dentro del jail. No se
  fabrico un test que simulara una vulnerabilidad que hoy no existe; se
  deja documentado el porque, para que si `_nombre_candidato` cambia
  algun dia (mas de un componente de ruta, u otro nivel de anidamiento),
  quede claro que ESTA es la linea que hay que revisar primero.
"""
from __future__ import annotations

import errno
import os
import random
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from motor_registry import tool_authority

from procesamiento import compuerta
from procesamiento.extractores import excel, ocr, pdf, word
from procesamiento.ficha import Ficha, sha256_de


class IngestaError(Exception):
    """Error de dominio de la ingesta. Nunca se deja escapar un `OSError`
    crudo del filesystem hacia el llamador sin contexto -- ver A-3,
    task-8-hallazgos-r2.md."""


class ReemplazoProcesadoAgotado(IngestaError):
    """Se agotaron los reintentos para reemplazar `procesado/<huella>/`:
    otro escritor de la MISMA huella siguió ganando la carrera entre el
    `rmtree` y el `rename` (ver el comentario en `ingerir`). No es
    corrupción -- el invariante de C-2 no se reabre -- es contención
    sostenida bajo concurrencia real más alta de la medida."""

# 255: limite tipico de NAME_MAX en filesystems Linux (ext4 y la mayoria).
# Se trata como caracteres, no bytes, por simplicidad -- los casos de uso
# reales (nombres subidos por un cliente) son mayormente ASCII; un nombre
# con caracteres multibyte que se pase del limite en BYTES pero no en
# caracteres queda fuera de este recorte (I-6 pide "que quepa", no un
# calculo exacto de bytes de filesystem).
_LIMITE_NOMBRE = 255

# Cuántas veces se reintenta el reemplazo final de `procesado/<huella>/`
# cuando otro escritor de la MISMA huella gana la carrera entre nuestro
# `rmtree` y nuestro `rename` (ver el comentario en `ingerir`).
#
# A-3 (task-8-hallazgos-r2.md): medido con hilos reales, SIN jitter, el
# reintento en 10 no aguantaba lo que decía aguantar -- a 16 hilos fallaba
# 21 de 25 veces con un OSError(39) crudo. La causa: reintentos
# INMEDIATOS se resincronizan (todos los perdedores de una ronda vuelven
# a chocar en la siguiente, a la vez). Con jitter, el número medido
# después del arreglo está en task-8-report.md -- "debería aguantar" es
# una suposición, así que se mide y se escribe.
_MAX_INTENTOS_REEMPLAZO = 10

# Tope del sleep aleatorio entre reintentos, que CRECE con el número de
# intento (backoff simple: intento 1 sortea en [0, _JITTER_MAXIMO_SEGUNDOS],
# intento 2 en [0, 2*_JITTER_MAXIMO_SEGUNDOS], etc.) -- desincroniza a los
# escritores que arrancaron casi juntos sin convertir esto en una espera
# larga: en el caso normal (sin contención) el jitter nunca se ejecuta,
# porque el primer intento siempre corre sin sleep.
_JITTER_MAXIMO_SEGUNDOS = 0.02

# nombre del extractor (Resultado.extractor / Ficha.extractor) -> modulo
# que sabe reportar SU version vigente -- consulta barata (un __version__
# ya importado, o un `--version` de proceso, nunca una re-extraccion) para
# I-2: invalidar el cache cuando el extractor cambio de version.
_MODULOS_POR_EXTRACTOR = {
    excel.EXTRACTOR: excel,
    ocr.EXTRACTOR: ocr,
    pdf.EXTRACTOR: pdf,
    word.EXTRACTOR: word,
}


def _version_vigente(nombre_extractor: str) -> str | None:
    modulo = _MODULOS_POR_EXTRACTOR.get(nombre_extractor)
    if modulo is None:
        return None
    try:
        return modulo._version()
    except Exception:
        return None


def _ahora() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _resolver_bajo_jail(ruta: Path) -> Path:
    """Valida `ruta` contra el MISMO jail que `tool_authority` usa para
    tool_calls (`WORKSPACE_ROOT` + `resolve_jailed_path`) -- no se escribe
    un jail nuevo (spec §4, C-1). `resolve_jailed_path` exige un path
    RELATIVO; acá se hace la traducción antes de delegarle el chequeo de
    fondo (sigue symlinks vía `Path.resolve()`, compara contra
    `WORKSPACE_ROOT`) -- esa lógica sigue viviendo en un solo lugar."""
    resuelto = Path(ruta).resolve(strict=False)
    try:
        relativo = resuelto.relative_to(tool_authority.WORKSPACE_ROOT)
    except ValueError:
        raise ValueError(
            f"ingesta: la ruta escapa del workspace (directo o vía symlink): "
            f"resuelve a '{resuelto}', fuera de '{tool_authority.WORKSPACE_ROOT}'"
        ) from None

    if str(relativo) == ".":
        return tool_authority.WORKSPACE_ROOT

    validado, razon = tool_authority.resolve_jailed_path(str(relativo), [])
    if validado is None:
        raise ValueError(f"ingesta: {razon}")
    return validado


def ruta_procesado(trabajo: Path, huella: str) -> Path:
    trabajo_abs = _resolver_bajo_jail(Path(trabajo))
    return trabajo_abs / "procesado" / huella


def _nombre_candidato(origen: Path, sufijo_extra: str = "") -> str:
    """I-6: el TRONCO se recorta antes de agregar el sufijo de huella, así
    el resultado siempre cabe en el límite del filesystem -- nunca un
    `OSError` crudo por un nombre demasiado largo."""
    extension = origen.suffix
    tronco = origen.stem
    nombre = f"{tronco}{sufijo_extra}{extension}"
    if len(nombre) <= _LIMITE_NOMBRE:
        return nombre
    recorte = max(1, _LIMITE_NOMBRE - len(sufijo_extra) - len(extension))
    return f"{tronco[:recorte]}{sufijo_extra}{extension}"


def _candidatos_de_nombre(origen: Path, huella: str) -> list[str]:
    return [
        _nombre_candidato(origen),
        _nombre_candidato(origen, f"-{huella[:8]}"),
        _nombre_candidato(origen, f"-{huella}"),
    ]


def _asegurar_en_fuente(origen: Path, fuente_abs: Path, huella: str) -> Path:
    """Copia `origen` a `fuente_abs` bajo un nombre LIBRE. Nunca pisa un
    archivo existente de contenido distinto, y nunca sigue un symlink --
    ni para leerlo (para decidir si "ya está") ni para escribir a través
    de él (C-1).

    Hallazgo PROPIO (no está en task-8-hallazgos-r2.md; lo destapó medir
    A-3 con más hilos, tal como pedía el ruling): con `O_CREAT|O_EXCL`
    escribiendo DIRECTO al nombre final, un lector concurrente que pierde
    la carrera puede ver el archivo a MITAD de escribir -- `sha256_de` lo
    lee TAL COMO ESTÁ en ese instante, calcula una huella que no coincide
    (porque está incompleto), y lo trata como "contenido distinto" cuando
    en realidad es EL MISMO archivo, todavía en vuelo. Con suficientes
    hilos esto cascadea por los tres nombres candidatos y agota los
    reintentos con un `ValueError` que no debería existir (medido: 1-2 de
    30 corridas a partir de 6 hilos, antes de este arreglo).

    Arreglo: el contenido se escribe COMPLETO a un temporal PROPIO de esta
    llamada primero, y sólo se PUBLICA con `os.link()` (hardlink) --
    atómico igual que `O_CREAT|O_EXCL` (falla con `FileExistsError` si el
    nombre ya existe, EN CUALQUIER FORMA -- symlink válido, roto, o
    archivo regular -- sin seguirlo jamás, verificado empíricamente antes
    de este cambio), pero con la ventaja de que cuando `link()` tiene
    éxito el contenido YA está completo: ningún lector puede ver un
    archivo a medio escribir bajo el nombre final, porque el nombre final
    nunca es el que se escribe."""
    temporal = fuente_abs / f".tmp-{os.getpid()}-{uuid.uuid4().hex}"
    try:
        with open(temporal, "wb") as escritura, open(origen, "rb") as lectura:
            shutil.copyfileobj(lectura, escritura)

        candidatos = _candidatos_de_nombre(origen, huella)
        for nombre_candidato in candidatos:
            destino_literal = fuente_abs / nombre_candidato

            if destino_literal.is_symlink():
                # Nunca se sigue, nunca se pisa, nunca se lee a través de
                # él: se trata como una colisión de nombre más y se
                # prueba el siguiente candidato. El symlink queda intacto.
                continue

            # Defensa en profundidad (spec §4): el mismo jail que valida
            # `trabajo`, ahora sobre el destino final -- por si
            # `origen.name` compusiera algo que terminara escapando.
            _resolver_bajo_jail(destino_literal)

            try:
                os.link(temporal, destino_literal)
            except FileExistsError:
                # Colisión: el nombre ya está ocupado. Si mientras tanto
                # se volvió un symlink (TOCTOU), tampoco se sigue.
                if destino_literal.is_symlink():
                    continue
                # A-1/A-2 (task-8-hallazgos-r2.md): el MISMO modelo de
                # amenaza que C-1 -- algo plantado en `fuente/` -- pero
                # por el camino de comparar huellas en vez de escribir.
                # Un FIFO cuelga `sha256_de` PARA SIEMPRE (abre y
                # bloquea, sin timeout); un directorio revienta con
                # `IsADirectoryError`. Sólo se le calcula la huella a un
                # archivo REGULAR -- esto cierra los dos Y cualquier otro
                # tipo raro (socket, dispositivo) sin tener que
                # enumerarlos uno por uno.
                if not destino_literal.is_file():
                    continue
                if sha256_de(destino_literal) == huella:
                    return destino_literal  # mismo contenido -- ya está
                continue  # contenido distinto -- siguiente candidato
            except OSError as exc:
                if exc.errno == errno.ELOOP:
                    continue  # se volvió symlink justo antes del link() (TOCTOU)
                raise
            return destino_literal

        raise ValueError(
            f"ingesta: no se pudo asegurar un nombre libre en 'fuente/' para "
            f"'{origen.name}' tras {len(candidatos)} intentos (colisiones de "
            "contenido o symlinks en cada nombre candidato)"
        )
    finally:
        # El temporal es descartable en cuanto se publicó (o se decidió
        # no usarlo): `link()` exitoso deja el contenido vivo bajo el
        # nombre final -- son dos nombres para el mismo inodo, borrar uno
        # no afecta al otro.
        temporal.unlink(missing_ok=True)


def _ficha_de_cache_valida(carpeta: Path, huella: str, extension_actual: str) -> Ficha | None:
    """Lee la ficha cacheada en `carpeta` y decide si es un acierto de
    caché VÁLIDO. `None` si hay que reextraer -- ficha ilegible (I-4),
    estado no reconstruible (I-1), extractor/versión desactualizados
    (I-2), extensión distinta a la que rutea hoy (I-3), o falta alguna
    salida listada (I-5). Nunca lanza -- un fallo de lectura es, siempre,
    un fallo de caché, no una excepción hacia el llamador."""
    ficha_json = carpeta / "ficha.json"
    if not ficha_json.is_file():
        return None
    try:
        ficha = Ficha.desde_json(ficha_json.read_text(encoding="utf8"))
    except (OSError, ValueError):
        return None  # I-4

    if ficha.sha256 != huella:
        return None  # defensivo: no debería pasar, la carpeta ya está indexada por huella

    if ficha.estado not in {"ok", "parcial"}:
        return None  # I-1: un error/sin_extractor no es un extracto reconstruible

    version_vigente = _version_vigente(ficha.extractor)
    if version_vigente is None or ficha.extractor_version != version_vigente:
        return None  # I-2

    if ficha.detalle.get("_extension_ingesta") != extension_actual:
        return None  # I-3

    salidas_listadas = ficha.detalle.get("_salidas_ingesta")
    if not isinstance(salidas_listadas, list):
        return None  # ficha sin inventario de salidas -- no hay forma de verificar completitud
    for nombre_salida in salidas_listadas:
        if not (carpeta / nombre_salida).is_file():
            return None  # I-5

    return ficha


def ingerir(origen: Path, trabajo: Path) -> Ficha:
    origen = Path(origen)
    trabajo_abs = _resolver_bajo_jail(Path(trabajo))  # C-1/I-7
    fuente_abs = trabajo_abs / "fuente"
    fuente_abs.mkdir(parents=True, exist_ok=True)

    huella = sha256_de(origen)
    extension_actual = origen.suffix.lower()
    destino = _asegurar_en_fuente(origen, fuente_abs, huella)  # C-1/C-3

    carpeta = ruta_procesado(trabajo_abs, huella)
    ficha_cacheada = _ficha_de_cache_valida(carpeta, huella, extension_actual)
    if ficha_cacheada is not None:
        # Cache vivo y VERIFICADO -- cero trabajo.
        return ficha_cacheada

    resultado = compuerta.extraer(destino)

    # Temporal único POR LLAMADA (C-2): dos ingestas del mismo archivo,
    # incluso entrelazadas, nunca comparten directorio de escritura -- no
    # hay ventana en la que una pise las salidas de la otra. Se limpia en
    # `finally`: si algo revienta a mitad de camino, no queda un huérfano
    # (cierra I-8).
    temporal = carpeta.with_name(f"{carpeta.name}.{os.getpid()}-{uuid.uuid4().hex}.parcial")
    try:
        temporal.mkdir(parents=True)

        for nombre, contenido in resultado.salidas.items():
            (temporal / nombre).write_text(contenido, encoding="utf8")

        detalle = {
            **dict(resultado.detalle),
            # I-3/I-5: lo que hace falta para verificar un acierto de
            # caché sin tener que confiar ciegamente en él.
            "_extension_ingesta": extension_actual,
            "_salidas_ingesta": sorted(resultado.salidas.keys()),
        }

        ficha = Ficha(
            sha256=huella,
            origen=str(destino.relative_to(trabajo_abs)),
            extractor=resultado.extractor,
            extractor_version=resultado.version,
            fecha=_ahora(),
            estado=resultado.estado,
            detalle=detalle,
        )
        (temporal / "ficha.json").write_text(ficha.a_json(), encoding="utf8")

        # Escritura atómica por carpeta: primero temporal (único, ver
        # arriba), después rename. Si algo se cae a la mitad, no queda un
        # `procesado/` a medias que parezca completo -- `carpeta` (el
        # nombre final) sólo aparece de un solo golpe.
        #
        # Hallazgo propio durante la verificación de C-2 (no estaba en
        # task-8-hallazgos.md, lo destapó un test de concurrencia REAL
        # fallando ~50% de las veces): el `rmtree` y el `rename` son DOS
        # operaciones separadas -- entre una y otra, otro escritor de la
        # MISMA huella puede colarse y volver a crear `carpeta` con su
        # propio rename exitoso. El nuestro entonces revienta con
        # `OSError(ENOTEMPTY)` porque `rename()` no pisa un directorio NO
        # vacío en POSIX. Como la extracción es determinista (misma
        # huella -> mismo contenido, sólo cambia `fecha`), es seguro
        # reintentar: eventualmente hay una ventana en la que nadie más
        # está escribiendo, y CUALQUIERA de los dos resultados que gane
        # es válido -- lo que no puede pasar es una excepción sin
        # sentido hacia el llamador por una carrera entre dos escritores
        # legítimos.
        for intento in range(_MAX_INTENTOS_REEMPLAZO):
            if intento > 0:
                # A-3: jitter con backoff. Reintentos INMEDIATOS se
                # resincronizan -- todos los perdedores de una ronda
                # vuelven a chocar juntos en la siguiente. El primer
                # intento SIEMPRE corre sin sleep (caso normal, sin
                # contención, cero costo).
                time.sleep(random.uniform(0, _JITTER_MAXIMO_SEGUNDOS * intento))
            if carpeta.exists():
                shutil.rmtree(carpeta, ignore_errors=True)
            try:
                temporal.rename(carpeta)
                break
            except OSError as exc:
                if exc.errno != errno.ENOTEMPTY:
                    raise
                if intento == _MAX_INTENTOS_REEMPLAZO - 1:
                    # A-3: nunca un OSError(39) crudo hacia el llamador --
                    # un error de dominio con contexto (qué huella, cuántos
                    # intentos) es lo mínimo que alguien necesita para
                    # decidir si reintentar más arriba tiene sentido.
                    raise ReemplazoProcesadoAgotado(
                        f"ingesta: no se pudo reemplazar '{carpeta}' tras "
                        f"{_MAX_INTENTOS_REEMPLAZO} intentos -- contención "
                        f"sostenida con otro escritor de la misma huella "
                        f"({huella})"
                    ) from exc
    finally:
        if temporal.exists():
            shutil.rmtree(temporal, ignore_errors=True)

    return ficha
