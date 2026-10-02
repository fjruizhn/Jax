"""El aviso inmediato del Faro (plan 0.3b, P-3): cada denegacion y cada rechazo de conexion llega a Telegram.

Mismo patron que `ci-aislada-vigia` (`/etc/ci-aislada/vigia.sh`): las credenciales viven en un archivo tipo
`/etc/restic/telegram.env` (`TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, con o sin `export` y con o sin comillas),
cuya RUTA sale de `JAX_FARO_AVISO_CREDS`; el token no esta en el codigo, ni en el repo, ni en un log.

COMO SE ENGANCHA. `Avisador` es un OBSERVADOR de la bitacora (`Bitacora(emisores=[...], observadores=[av])`):
corre antes que los emisores y aunque alguno falle, de modo que una denegacion se avisa incluso si la bitacora
durable esta caida. Lo que se avisa lo decide `es_avisable`: los eventos de `EVENTOS_AVISABLES` y toda
anotacion con `decision == "denegado"`.

UN AVISO QUE FALLA NO CAMBIA NADA. `__call__` es sincrono, no espera la red y NO lanza nunca: decide, aplica la
tasa y encola. Un trabajador `asyncio` saca de la cola y entrega por HTTP en un hilo PROPIO (un ejecutor
dedicado, no el compartido de `asyncio.to_thread`, que el freno usa en cada pedido) con plazo. Si el envio
falla, vence o la cola esta llena, se cuenta (`fallidos`, `descartados`) y se sigue: la denegacion ya se aplico.

TASA. Un cupo por CLASE (`evento|motivo`): una rafaga de `rafaga` avisos y despues uno por `intervalo_s`. Una
tormenta de rechazos de conexion no oculta una denegacion por freno (otra clase). Lo suprimido no se pierde:
se cuenta y sale en UN resumen con la cuenta exacta cuando el cupo vuelve. La cuenta total queda en
`suprimidos` (medicion). Las clases distintas se acotan a `max_clases`; el resto comparte la clase `otros`.

FALLA CERRADO EN LA CONFIGURACION: sin `JAX_FARO_AVISO_CREDS` (o con credenciales incompletas, ausentes, con
escritura ajena o que son un enlace) el servicio no arranca. Fallar al ENVIAR, en cambio, nunca se propaga.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import ipaddress
import socket
import stat
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from cli_sandbox import _campo_log

from .config import ConfigFaroInvalida

logger = logging.getLogger(__name__)

API_URL_POR_DEFECTO = "https://api.telegram.org"
RAFAGA_POR_DEFECTO = 5
INTERVALO_S_POR_DEFECTO = 10.0
TIMEOUT_S_POR_DEFECTO = 10.0
COLA_POR_DEFECTO = 100
MAX_CLASES_POR_DEFECTO = 64
_MAX_CREDS_BYTES = 64 * 1024
_MAX_TEXTO = 3500                   # Telegram admite 4096; queda margen
_FIN = object()
_HILOS_ENVIO = 4                    # un envio colgado no frena a los demas; el numero de hilos esta acotado

# Eventos que se avisan siempre (ademas de toda anotacion con decision == "denegado").
EVENTOS_AVISABLES = frozenset({
    "conexion_rechazada",           # el Puerto rechazo una conexion (uid o token)
    "control_creado", "control_rechazado",      # canal de control (0.3c)
    "tope_superado", "tope_no_verificable", "tope_resultado_desconocido",     # topes (0.3b): al llegar falla cerrado
    "tope_sin_regla",                           # consumo SIN regla de tope: se mide y se avisa, no niega
})

_TITULOS = {
    "conexion_rechazada": "CONEXION RECHAZADA",
    "control_creado": "EJECUCION CREADA",
    "control_rechazado": "PEDIDO DE CONTROL RECHAZADO",
    "tope_superado": "TOPE ALCANZADO (denegado)",
    "tope_no_verificable": "TOPE NO VERIFICABLE (denegado)",
    "tope_resultado_desconocido": "RESULTADO DESCONOCIDO DE UN CONTEO DE TOPE",
    "tope_sin_regla": "CONSUMO SIN REGLA DE TOPE (medido, no niega)",
}
_CAMPOS = ("motivo", "run_id", "usuario", "tenant", "faceta", "motor", "pipeline", "entry_point", "metodo", "objetivo",
           "peer_uid", "uid_jaula", "recurso", "cantidad", "usado", "tope", "periodo")


class AvisoNoEntregado(Exception):
    """El envio fallo. El mensaje NUNCA lleva la URL ni el token: solo el tipo de fallo."""


# --------------------------------------------------------------------------- #
# configuracion y credenciales                                                #
# --------------------------------------------------------------------------- #

def _es_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


@dataclass(frozen=True)
class ConfigAviso:
    """`JAX_FARO_AVISO_CREDS` es obligatoria; el resto tiene un valor por defecto razonable."""
    creds: Path
    api_url: str = API_URL_POR_DEFECTO
    rafaga: int = RAFAGA_POR_DEFECTO
    intervalo_s: float = INTERVALO_S_POR_DEFECTO
    timeout_s: float = TIMEOUT_S_POR_DEFECTO
    cola: int = COLA_POR_DEFECTO
    max_clases: int = MAX_CLASES_POR_DEFECTO

    def __post_init__(self) -> None:
        if not Path(self.creds).is_absolute():
            raise ConfigFaroInvalida(f"JAX_FARO_AVISO_CREDS tiene que ser una ruta absoluta, no {str(self.creds)!r}")
        partes = urllib.parse.urlsplit(self.api_url)
        if partes.scheme not in ("http", "https") or not partes.netloc or not partes.hostname:
            raise ConfigFaroInvalida("JAX_FARO_AVISO_API_URL tiene que ser una URL http(s)")
        if partes.scheme == "http" and not _es_loopback(partes.hostname):
            raise ConfigFaroInvalida(
                "JAX_FARO_AVISO_API_URL: http:// solo hacia loopback (el token viaja en la URL); para cualquier otro destino, https://")
        if not isinstance(self.rafaga, int) or self.rafaga < 1:
            raise ConfigFaroInvalida("JAX_FARO_AVISO_RAFAGA tiene que ser un entero >= 1")
        if not self.intervalo_s > 0:
            raise ConfigFaroInvalida("JAX_FARO_AVISO_INTERVALO_S tiene que ser positivo")
        if not self.timeout_s > 0:
            raise ConfigFaroInvalida("JAX_FARO_AVISO_TIMEOUT_S tiene que ser positivo")
        if not isinstance(self.cola, int) or self.cola < 1:
            raise ConfigFaroInvalida("JAX_FARO_AVISO_COLA tiene que ser un entero >= 1")
        if not isinstance(self.max_clases, int) or self.max_clases < 1:
            raise ConfigFaroInvalida("max_clases tiene que ser un entero >= 1")

    @classmethod
    def desde_entorno(cls, env: Mapping[str, str]) -> "ConfigAviso":
        crudo = (env.get("JAX_FARO_AVISO_CREDS") or "").strip()
        if not crudo:
            raise ConfigFaroInvalida("JAX_FARO_AVISO_CREDS no esta definida: sin aviso de las denegaciones el Faro no arranca")

        def numero(nombre: str, tipo, defecto):
            valor = (env.get(nombre) or "").strip()
            try:
                return tipo(valor) if valor else defecto
            except ValueError as exc:
                raise ConfigFaroInvalida(f"{nombre} no es un numero valido") from exc

        return cls(
            creds=Path(crudo),
            api_url=(env.get("JAX_FARO_AVISO_API_URL") or "").strip() or API_URL_POR_DEFECTO,
            rafaga=numero("JAX_FARO_AVISO_RAFAGA", int, RAFAGA_POR_DEFECTO),
            intervalo_s=numero("JAX_FARO_AVISO_INTERVALO_S", float, INTERVALO_S_POR_DEFECTO),
            timeout_s=numero("JAX_FARO_AVISO_TIMEOUT_S", float, TIMEOUT_S_POR_DEFECTO),
            cola=numero("JAX_FARO_AVISO_COLA", int, COLA_POR_DEFECTO),
        )


@dataclass(frozen=True)
class Credenciales:
    token: str = field(repr=False)      # nunca se imprime
    chat_id: str = ""

    def __str__(self) -> str:
        return "Credenciales(<oculto>)"


def _leer_clave(texto: str, clave: str) -> str:
    # Mismo formato que `lib-avisar.sh`: con o sin `export`, con o sin comillas dobles, espacios delante.
    m = re.search(rf'^[ \t]*(?:export[ \t]+)?{clave}="?([^"\s]+)', texto, re.MULTILINE)
    return m.group(1) if m else ""


def leer_credenciales(ruta: Path) -> Credenciales:
    """Lee el archivo de credenciales o FALLA CERRADO: ausente, enlace, no regular, con escritura de grupo u
    otros (quien lo escriba redirige los avisos), de otro dueño que no sea root ni el servicio, o incompleto."""
    try:
        fd = os.open(ruta, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except OSError as exc:
        raise ConfigFaroInvalida(f"no se pueden leer las credenciales del aviso ({type(exc).__name__}): {ruta}") from exc
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise ConfigFaroInvalida(f"las credenciales del aviso no son un archivo regular: {ruta}")
        if st.st_mode & 0o022:
            raise ConfigFaroInvalida(f"las credenciales del aviso tienen escritura de grupo u otros: {ruta}")
        if st.st_uid not in (0, os.geteuid()):
            raise ConfigFaroInvalida(f"las credenciales del aviso no son de root ni del usuario del servicio: {ruta}")
        texto = os.read(fd, _MAX_CREDS_BYTES).decode("utf-8", errors="replace")
    finally:
        os.close(fd)
    token, chat = _leer_clave(texto, "TELEGRAM_BOT_TOKEN"), _leer_clave(texto, "TELEGRAM_CHAT_ID")
    if not token or not chat:
        raise ConfigFaroInvalida(f"las credenciales del aviso no traen TELEGRAM_BOT_TOKEN y TELEGRAM_CHAT_ID: {ruta}")
    return Credenciales(token=token, chat_id=chat)


# --------------------------------------------------------------------------- #
# que se avisa, como se dice, con que tasa                                    #
# --------------------------------------------------------------------------- #

def es_avisable(registro: object) -> bool:
    if not isinstance(registro, Mapping):
        return False
    return registro.get("evento") in EVENTOS_AVISABLES or registro.get("decision") == "denegado"


def redactar(registro: Mapping, host: str) -> str:
    """El texto del aviso: que paso, de quien y donde. Cada valor pasa por `_campo_log` (una linea, escapado,
    acotado): un dato hostil no puede fabricar una linea ni un campo. NUNCA lleva los argumentos de la
    llamada ni el resultado."""
    evento = registro.get("evento")
    if evento in _TITULOS:
        titulo = _TITULOS[evento]
    elif registro.get("decision") == "denegado":
        titulo = "DENEGADO"
    else:
        titulo = "EVENTO"
    if evento == "llamada" and registro.get("decision") == "denegado":
        titulo = "DENEGADO"
    lineas = [f"FARO · {titulo} · {_campo_log(evento, 48)} · {_campo_log(host, 64)}"]
    campos = [f"{k}={_campo_log(registro[k], 120)}" for k in _CAMPOS if k in registro and registro[k] is not None]
    if campos:
        lineas.append(" ".join(campos))
    return "\n".join(lineas)[:_MAX_TEXTO]


def _clase(registro: Mapping) -> str:
    return f"{_campo_log(registro.get('evento'), 48)}|{_campo_log(registro.get('motivo', ''), 48)}"


class LimiteTasa:
    """Un cupo por clase: `rafaga` y luego uno por `intervalo_s` (nunca acumula mas que la rafaga)."""

    def __init__(self, rafaga: int, intervalo_s: float, reloj: Callable[[], float] = time.monotonic,
                 max_clases: int = MAX_CLASES_POR_DEFECTO):
        self._rafaga, self._intervalo, self._reloj, self._max = rafaga, intervalo_s, reloj, max_clases
        self._estado: dict[str, tuple[float, float]] = {}

    @property
    def clases_registradas(self) -> int:
        return len(self._estado)

    def admitir(self, clase: str) -> bool:
        if clase not in self._estado and len(self._estado) >= self._max:
            clase = "otros"
        ahora = self._reloj()
        fichas, ultimo = self._estado.get(clase, (float(self._rafaga), ahora))
        fichas = min(float(self._rafaga), fichas + (ahora - ultimo) / self._intervalo)
        if fichas >= 1.0:
            self._estado[clase] = (fichas - 1.0, ahora)
            return True
        self._estado[clase] = (fichas, ahora)
        return False


# --------------------------------------------------------------------------- #
# el envio por HTTP                                                           #
# --------------------------------------------------------------------------- #

class _SinRedirecciones(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):    # un redireccionamiento no se sigue: el token no viaja a otro destino
        return None


def enviar_telegram(cfg: ConfigAviso, cred: Credenciales, texto: str) -> None:
    """`POST {api}/bot<token>/sendMessage` (sincrono: va en un hilo). Lanza `AvisoNoEntregado` SIN la URL."""
    url = f"{cfg.api_url.rstrip('/')}/bot{cred.token}/sendMessage"
    datos = urllib.parse.urlencode({"chat_id": cred.chat_id, "text": texto[:_MAX_TEXTO],
                                    "disable_web_page_preview": "true"}).encode()
    peticion = urllib.request.Request(url, data=datos, method="POST")
    abridor = urllib.request.build_opener(_SinRedirecciones)
    try:
        with abridor.open(peticion, timeout=cfg.timeout_s) as r:    # noqa: S310 (el esquema se valida en ConfigAviso)
            cuerpo = json.loads(r.read(65536) or b"{}")
            if r.status != 200 or not cuerpo.get("ok", False):
                raise AvisoNoEntregado(f"respuesta {r.status}")
    except AvisoNoEntregado:
        raise
    except urllib.error.HTTPError as exc:
        raise AvisoNoEntregado(f"http {exc.code}") from None
    except (urllib.error.URLError, OSError, ValueError, socket.timeout) as exc:
        raise AvisoNoEntregado(type(exc).__name__) from None


# --------------------------------------------------------------------------- #
# el avisador                                                                 #
# --------------------------------------------------------------------------- #

class Avisador:
    """Observador de la bitacora. Contadores de medicion: `enviados`, `fallidos`, `suprimidos`, `descartados`."""

    def __init__(self, cfg: ConfigAviso, cred: Credenciales, *, enviar: Callable[[str], None] | None = None,
                 reloj: Callable[[], float] = time.monotonic, host: str | None = None):
        self._cfg = cfg
        self._enviar = enviar if enviar is not None else (lambda texto: enviar_telegram(cfg, cred, texto))
        self._host = host if host is not None else socket.gethostname()
        self._limite = LimiteTasa(cfg.rafaga, cfg.intervalo_s, reloj, cfg.max_clases)
        self._pendientes: dict[str, int] = {}          # clase -> suprimidos sin resumir
        self._cola: asyncio.Queue = asyncio.Queue(maxsize=cfg.cola)
        self._ejecutor: ThreadPoolExecutor | None = None
        self._tarea: asyncio.Task | None = None
        self._cerrando = False
        self.enviados = self.fallidos = self.suprimidos = self.descartados = 0
        self.fallidos_por_clase: dict[str, int] = {}     # medicion: que clase de aviso no llega
        self.ultimo_fallo: dict | None = None            # {"clase", "run_id", "error"} del ultimo aviso que fallo

    # -- observador: sincrono, no espera la red, no lanza nunca ---------------------------------
    def __call__(self, registro: object) -> None:
        try:
            if not es_avisable(registro):
                return
            clase = _clase(registro)
            if self._limite.admitir(clase):
                self._encolar(redactar(registro, self._host), clase, _campo_log(registro.get("run_id", "-"), 64))
            else:
                self.suprimidos += 1
                self._pendientes[clase] = self._pendientes.get(clase, 0) + 1
        except Exception as exc:  # fail-soft: un aviso no puede cambiar la decision que lo origino; solo se anota el tipo
            logger.warning("el avisador no pudo procesar un registro (%s)", type(exc).__name__)

    def _encolar(self, texto: str, clase: str, run_id: str = "-") -> None:
        try:
            self._cola.put_nowait((texto, clase, run_id))
        except asyncio.QueueFull:
            self.descartados += 1

    def _resumenes(self) -> None:
        for clase in list(self._pendientes):
            if self._limite.admitir(clase):
                n = self._pendientes.pop(clase)
                self._encolar(f"FARO · {n} avisos suprimidos por tasa · {_campo_log(self._host, 64)}\nclase={clase}", "resumen")

    # -- trabajador ----------------------------------------------------------------------------
    async def _entregar(self, texto: str, clase: str, run_id: str) -> None:
        try:
            futuro = asyncio.get_running_loop().run_in_executor(self._ejecutor, self._enviar, texto)
            await asyncio.wait_for(futuro, self._cfg.timeout_s)
            self.enviados += 1
        except Exception as exc:  # fail-soft: el aviso no llego; se cuenta y se sigue (la denegacion ya se aplico)
            self.fallidos += 1
            self.fallidos_por_clase[clase] = self.fallidos_por_clase.get(clase, 0) + 1
            self.ultimo_fallo = {"clase": clase, "run_id": run_id, "error": type(exc).__name__}
            logger.warning("aviso no entregado (%s) clase=%s run_id=%s fallidos_de_la_clase=%s", type(exc).__name__, clase, run_id,
                           self.fallidos_por_clase[clase])

    async def _trabajar(self) -> None:
        while True:
            try:
                item = await asyncio.wait_for(self._cola.get(), self._cfg.intervalo_s)
            except TimeoutError:
                self._resumenes()
                continue
            if item is _FIN:
                return
            await self._entregar(*item)
            self._resumenes()
            if self._cerrando and self._cola.empty():
                return

    async def iniciar(self) -> None:
        if self._tarea is None:
            self._cerrando = False
            self._ejecutor = ThreadPoolExecutor(max_workers=_HILOS_ENVIO, thread_name_prefix="faro-aviso")
            self._tarea = asyncio.create_task(self._trabajar(), name="faro-aviso")

    async def cerrar(self) -> None:
        """Entrega lo pendiente (con plazo) y para. Nunca lanza."""
        tarea, self._tarea = self._tarea, None
        if tarea is None:
            return
        self._cerrando = True
        try:
            self._cola.put_nowait(_FIN)
        except asyncio.QueueFull:  # fail-soft: el trabajador revisa `_cerrando` tras cada envio y termina al vaciar la cola
            pass
        hechas, _ = await asyncio.wait({tarea}, timeout=self._cfg.timeout_s + 1.0)
        if not hechas:
            tarea.cancel()
            await asyncio.gather(tarea, return_exceptions=True)
        if self._ejecutor is not None:
            self._ejecutor.shutdown(wait=False, cancel_futures=True)
            self._ejecutor = None

    async def __aenter__(self) -> "Avisador":
        await self.iniciar()
        return self

    async def __aexit__(self, *_exc) -> None:
        await self.cerrar()
