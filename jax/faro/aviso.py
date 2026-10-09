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
import errno
import fcntl
import json
import logging
import os
import re
import secrets
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
from datetime import datetime, timezone
from pathlib import Path

from cli_sandbox import _campo_log

from policy.rule_authority.models import RuleDecision, RuleDecisionStatus

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

# --------------------------------------------------------------------------- #
# F1.1 paso 8 r2: el aviso de una decision del kernel de reglas               #
# --------------------------------------------------------------------------- #
#
# §11 del diseno del kernel (2026-10-05): "los avisos inmediatos son fail-soft
# y se emiten DESPUES de la decision durable. Incluyen identificador de regla,
# reason code y hashes necesarios; excluyen contenido completo y argumentos
# sensibles." La constitucion (LA AUTONOMIA) fija los dos plazos: "al instante
# si obliga o si se nego por falta de regla; en un resumen diario lo demas".
#
# CONTRATO: el aviso consume el `RuleDecision` REAL de #371
# (policy/rule_authority/models.py) y usa SUS nombres: status, required_rule_id,
# reason_code, request_hash, decided_at_utc. Esa decision NO trae la clase del
# acto ni argumentos: quien orquesta, que si conoce la capacidad, pasa `obliga`.
#
# DIRECCION DEL IMPORT: aviso -> modelos de rule_authority esta permitido; la
# contraria (rule_authority -> aviso) esta prohibida y la cierran las pruebas
# por AST y en un interprete limpio: el aviso se consume DESPUES de la decision
# durable, nunca dentro de ella.
#
# FALLO CERRADO ante lo no reconocible (tipo que no es RuleDecision, status o
# campo ilegible): aviso INMEDIATO "DECISION NO RECONOCIBLE" que nombra la clase
# del objeto y nada mas (su repr puede traer secretos). Nunca None, nunca raise.

# Lo que decide el canal de un PERMIT es `obliga` (True = obliga, False = no
# obliga, None = el llamador no lo sabe). "Si hay duda, obliga": solo un False
# explicito difiere el aviso al resumen diario.
_TITULOS_REGLA = {
    RuleDecisionStatus.DENY: "REGLA DENEGADA",
    RuleDecisionStatus.MISSING_RULE: "SIN REGLA QUE CUBRA EL ACTO",
}
_TITULO_PERMIT_OBLIGA = "PERMISO (OBLIGA O NO SE PUDO DESCARTAR QUE OBLIGUE)"
_TITULO_PERMIT_NO_OBLIGA = "PERMISO (NO OBLIGA)"
_RE_HASH = re.compile(r"sha256:[0-9a-f]{64}")
_MAX_CUERPO = 1500                  # un cuerpo mas largo se recorta CON marca, nunca en silencio


@dataclass(frozen=True)
class AvisoRegla:
    """Un aviso listo para mandar. `clase` es la llave del LimiteTasa (incluye
    la REGLA: una tormenta de una regla no tapa la negativa de otra); `texto`
    ya viene redactado, aplanado y acotado; `inmediato` decide el canal."""

    texto: str
    clase: str
    inmediato: bool
    creado_utc: str


def _ahora_utc() -> datetime:
    return datetime.now(timezone.utc)


def _iso_utc(momento: datetime) -> str:
    if momento.tzinfo is None:      # una hora ingenua es hora local disfrazada: no se acepta nunca
        raise ValueError("hora sin zona")
    return momento.astimezone(timezone.utc).isoformat(timespec="seconds")


def _hash_12(valor) -> str | None:
    """`sha256:` + 12 hex del digest. None si no es el hash del contrato."""
    if not isinstance(valor, str) or _RE_HASH.fullmatch(valor) is None:
        return None
    return f"sha256:{valor[7:19]}"


def _no_reconocible(decision, host: str) -> AvisoRegla:
    """El aviso de fallo cerrado: clase del objeto, NADA mas. El repr de un
    objeto cualquiera puede traer arguments/subject con secretos: no viaja."""
    creado = _iso_utc(_ahora_utc())
    nombre = _campo_log(type(decision).__qualname__, 120)
    texto = (f"FARO · DECISION NO RECONOCIBLE · {_campo_log(host, 64)}\n"
             f"clase={nombre} a={creado}")
    return AvisoRegla(texto=texto[:_MAX_TEXTO], clase=f"no_reconocible|{nombre}",
                      inmediato=True, creado_utc=creado)


def aviso_de_decision(decision, *, host: str, obliga: bool | None = None) -> AvisoRegla:
    """Arma el aviso de UNA decision del kernel. Pura: sin I/O ni efectos.

    Siempre devuelve un AvisoRegla. DENY y MISSING_RULE salen al instante. Un
    PERMIT sale al instante salvo que el llamador diga `obliga=False`: la
    decision de #371 no trae la clase del acto, asi que sin esa palabra no se
    puede excluir que obligue (LA AUTONOMIA: "si hay duda, obliga"). Cualquier
    otra cosa (no es un RuleDecision, status o campos ilegibles) es el aviso
    INMEDIATO de decision no reconocible.

    El texto NUNCA lleva argumentos, montos, subject, contenido ni nombres de
    archivo: solo regla (required_rule_id), razon, el request_hash recortado a
    `sha256:` + 12 hex y la hora UTC de la DECISION. Todo pasa por `_campo_log`."""
    try:
        if not isinstance(decision, RuleDecision) or not isinstance(decision.status, RuleDecisionStatus):
            return _no_reconocible(decision, host)
        status, regla, razon = decision.status, decision.required_rule_id, decision.reason_code
        req = _hash_12(decision.request_hash)
        if (req is None or not isinstance(regla, str) or not isinstance(decision.decided_at_utc, datetime)
                or (razon is not None and not isinstance(razon, str))):
            return _no_reconocible(decision, host)
        creado = _iso_utc(decision.decided_at_utc)
        if status is RuleDecisionStatus.PERMIT:
            inmediato = obliga is not False
            titulo = _TITULO_PERMIT_OBLIGA if inmediato else _TITULO_PERMIT_NO_OBLIGA
        else:
            inmediato, titulo = True, _TITULOS_REGLA[status]
        clase = f"{status.value}|{_campo_log(razon or '-', 48)}|{_campo_log(regla, 128)}"
        texto = (f"FARO · {titulo} · {_campo_log(host, 64)}\n"
                 f"regla={_campo_log(regla, 128)} razon={_campo_log(razon or '-', 48)} "
                 f"req={req} a={creado}")
    except Exception:  # fail-soft: tipos raros, la decision se avisa igual, como no reconocible
        return _no_reconocible(decision, host)
    return AvisoRegla(texto=texto[:_MAX_TEXTO], clase=clase, inmediato=inmediato, creado_utc=creado)


# --- la cola del resumen diario: sin perdidas y 0600 ------------------------- #
#
# Escritor y resumen comparten un `flock` sobre `<cola>.candado`. El escritor
# appendea bajo el candado (abre la cola DENTRO de el, asi reabre tras una
# rotacion); el resumen ROTA antes de leer (`os.replace` de la cola a
# `<cola>.<ts>.procesando`, tambien bajo el candado): lo que llega despues va a
# la cola nueva. Un rotado que sobreviva (caida a mitad) lo recoge el resumen
# siguiente. Nunca se escribe un vacio sobre la cola. Todo archivo que este
# modulo crea (cola, candado) nace 0600 por `os.open`, sin depender del umask.

class _Candado:
    """flock EXclusivo sobre `<cola>.candado` (0600). Context manager."""

    def __init__(self, ruta_cola):
        self._ruta = str(ruta_cola) + ".candado"
        self._fd = -1

    def __enter__(self):
        self._fd = os.open(self._ruta, os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
        try:
            fcntl.flock(self._fd, fcntl.LOCK_EX)
        except BaseException:
            os.close(self._fd)
            raise
        return self

    def __exit__(self, *exc):
        try:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        finally:
            os.close(self._fd)
        return False


def acumular_para_resumen(aviso: AvisoRegla, ruta_cola: "os.PathLike[str] | str",
                          *, suprimido_por_tasa: bool = False, envio_fallido: bool = False) -> bool:
    """Deja el aviso en la cola del resumen diario: UNA linea JSON por aviso,
    bajo el candado, con `O_APPEND` y UN solo `write`. `suprimido_por_tasa`
    marca lo que el limite de tasa postergo y `envio_fallido` lo que no pudo enviarse: no se descarta, se resume.
    Fail-soft: si el disco falla, False y el log (la decision ya se aplico)."""
    try:
        linea = json.dumps(
            {"creado_utc": aviso.creado_utc, "clase": aviso.clase, "texto": aviso.texto,
             "suprimido_por_tasa": suprimido_por_tasa, "envio_fallido": envio_fallido},
            ensure_ascii=False, separators=(",", ":"))
        datos = (linea + "\n").encode("utf-8")
        with _Candado(ruta_cola):
            flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW | os.O_CLOEXEC
            fd = os.open(ruta_cola, flags, 0o600)
            try:
                st = os.fstat(fd)
                if (not stat.S_ISREG(st.st_mode) or st.st_uid != os.geteuid()
                        or st.st_mode & 0o777 != 0o600):
                    raise PermissionError("cola insegura para append")
                if len(datos) > _MAX_LEIDO:
                    raise OSError("aviso individual excede el limite de la cola")
                if st.st_size + len(datos) > _MAX_LEIDO:
                    os.close(fd)
                    fd = -1
                    os.replace(ruta_cola, f"{ruta_cola}.{time.time_ns():020d}.procesando")
                    fd = os.open(ruta_cola, flags, 0o600)
                    st = os.fstat(fd)
                    if not stat.S_ISREG(st.st_mode) or st.st_size + len(datos) > _MAX_LEIDO:
                        raise OSError("no se pudo crear un segmento acotado de la cola")
                escrito = os.write(fd, datos)
                if escrito != len(datos):
                    raise OSError("append parcial en la cola de avisos")
            finally:
                if fd >= 0:
                    os.close(fd)
        return True
    except OSError as exc:  # fail-soft: sin disco no hay resumen, pero tampoco excepcion hacia la decision
        logger.warning("no se pudo acumular el aviso diario (%s)", type(exc).__name__)
        return False


def _recortar(cuerpo: str) -> str:
    if len(cuerpo) <= _MAX_CUERPO:
        return cuerpo
    return f"{cuerpo[:_MAX_CUERPO]} …[+{len(cuerpo) - _MAX_CUERPO} car. recortados]"


def _mensajes_del_resumen(bloques: list[str], cabecera: str) -> list[str]:
    """Mensajes numerados (i/n) que caben en `_MAX_TEXTO` SIN cortar un bloque a
    la mitad: si no caben en uno, son MAS mensajes, no menos avisos."""
    reserva = len("\n· 9999/9999")
    mensajes: list[str] = []
    actual: str | None = None
    for bloque in bloques:
        if actual is not None and len(actual) + 1 + len(bloque) + reserva <= _MAX_TEXTO:
            actual += "\n" + bloque
            continue
        if actual is not None:
            mensajes.append(actual)
        actual = f"{cabecera}\n{bloque}"
    if actual is not None:
        mensajes.append(actual)
    total = len(mensajes)
    return [f"{m}\n· {i}/{total}" for i, m in enumerate(mensajes, 1)]


@dataclass(frozen=True)
class ResumenDiario:
    """El resumen compuesto y el TOKEN para confirmarlo: los `.procesando` que lo
    respaldan siguen en disco hasta `confirmar_resumen(token)`."""

    mensajes: tuple[str, ...]
    rotados: tuple[str, ...]
    ruta_cola: str


_MAX_LEIDO = 16 * 1024 * 1024


class _ColaDemasiadoGrande(PermissionError):
    """Segmento legado que supera el presupuesto: preservar en cuarentena."""


def _leer_seguro(ruta: str) -> str:
    """Lee un archivo de la cola SIN seguir enlaces: O_NOFOLLOW, y regular, del
    usuario actual y 0600. Un enlace plantado o un archivo ajeno levanta
    OSError (no se lee el destino)."""
    try:
        fd = os.open(ruta, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK)   # NONBLOCK: un FIFO no cuelga
    except OSError as exc:
        if exc.errno == errno.ELOOP:      # un enlace: inseguro, no un fallo de disco
            raise PermissionError(f"es un enlace: {ruta}") from None
        raise
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise PermissionError(f"no es un archivo regular: {ruta}")
        if st.st_uid != os.geteuid():
            raise PermissionError(f"dueno ajeno: {ruta}")
        if st.st_mode & 0o777 != 0o600:
            raise PermissionError(f"modo distinto de 0600: {ruta}")
        partes, total = [], 0
        while True:
            trozo = os.read(fd, 1 << 20)
            if not trozo:
                break
            total += len(trozo)
            if total > _MAX_LEIDO:
                raise _ColaDemasiadoGrande(f"cola demasiado grande: {ruta}")
            partes.append(trozo)
    finally:
        os.close(fd)
    return b"".join(partes).decode("utf-8", errors="replace")


LEASE_S_POR_DEFECTO = 900.0


def _patron_rotado(nombre_cola: str) -> "re.Pattern[str]":
    """`<cola>.<20 digitos>.procesando` y, si esta reclamado,
    `.<pid>-<16 hex>`. EXACTO: `cola.jsonl.otra.<ts>.procesando` no es de `cola.jsonl`."""
    return re.compile(re.escape(nombre_cola) + r"\.(\d{20}\.procesando)(?:\.(\d+)-([0-9a-f]{16}))?")


def resumen_diario(ruta_cola: "os.PathLike[str] | str", *, host: str,
                   lease_s: float = LEASE_S_POR_DEFECTO) -> "ResumenDiario | None":
    """Compone el resumen de la cola diaria en DOS FASES. Devuelve un
    `ResumenDiario` (mensajes numerados 1/n + token) o None si no hay nada o no
    se pudo trabajar (y entonces no se borra NADA).

    Bajo el candado rota la cola a `<cola>.<ts>.procesando` ANTES de leer y
    RECLAMA los rotados (los renombra a `….procesando.<pid>-<token>`): dos
    resumidores simultaneos no se pisan, el segundo no ve lo ya reclamado. Un
    reclamo con mas de `lease_s` segundos sin confirmar (proceso caido) se puede
    reclamar de nuevo. Los archivos reclamados NO se borran aqui: el llamador
    envia los mensajes y, solo si salieron, llama a `confirmar_resumen(token)`.
    Sin confirmar, pasado el lease se re-entregan (al-menos-una-vez).

    No sigue enlaces ni lee archivos ajenos: cada archivo se abre con
    O_NOFOLLOW y debe ser regular, del usuario actual y 0600. Uno que no lo
    cumple (PermissionError) NO aborta el resumen: pasa a cuarentena (`….cuarentena`, fuera del
    patron, no se vuelve a leer) y se cuenta como `en_cuarentena=N`. Solo
    cuentan las lineas JSON con `texto` str. Sincrono: los llamadores async lo
    envuelven en `asyncio.to_thread`."""
    try:
        with _Candado(ruta_cola):
            cola = Path(ruta_cola)
            if cola.exists() or cola.is_symlink():
                os.replace(cola, f"{cola}.{time.time_ns():020d}.procesando")   # rotar ANTES de leer
            patron = _patron_rotado(cola.name)
            ahora = time.time()
            reclamados: list[str] = []
            for p in sorted(cola.parent.iterdir(), key=lambda q: q.name):
                m = patron.fullmatch(p.name)
                if m is None:
                    continue
                if m.group(2) is not None and ahora - os.lstat(p).st_ctime < lease_s:
                    continue                    # reclamado por otro resumidor, con lease vigente
                base = p.name[:m.start(2)].rstrip(".") if m.group(2) is not None else p.name
                nuevo = cola.parent / f"{base}.{os.getpid()}-{secrets.token_hex(8)}"
                os.rename(p, nuevo)
                reclamados.append(str(nuevo))
            if not reclamados:
                return None
            lineas: list[str] = []
            buenos: list[str] = []
            cuarentena = 0
            for ruta in reclamados:
                try:
                    texto = _leer_seguro(ruta)
                except PermissionError as exc:   # enlace plantado, dueno o modo ajeno: se aparta, el resto sigue
                    os.replace(ruta, ruta + ".cuarentena")
                    cuarentena += 1
                    logger.warning("rotado a cuarentena (%s)", type(exc).__name__)
                    continue
                buenos.append(ruta)
                lineas.extend(l for l in texto.splitlines() if l.strip())
            por_clase: dict[str, int] = {}
            cuerpos: list[str] = []
            suprimidos = ilegibles = fallidos = 0
            for linea in lineas:
                try:
                    dato = json.loads(linea)
                    if not isinstance(dato, dict) or not isinstance(dato["texto"], str):
                        raise TypeError("linea sin texto str")
                    cuerpo = _recortar(dato["texto"].replace("\n", " · "))
                    clase = str(dato.get("clase", "-"))
                    suprimido = bool(dato.get("suprimido_por_tasa"))
                    fallido = bool(dato.get("envio_fallido"))
                except (ValueError, KeyError, TypeError):
                    ilegibles += 1   # se cuenta y se dice: una linea corrupta no esconde a las demas
                    continue
                cuerpos.append(cuerpo)
                suprimidos += suprimido
                fallidos += fallido
                por_clase[clase] = por_clase.get(clase, 0) + 1
            if not cuerpos and not ilegibles and not cuarentena:
                for ruta in buenos:
                    os.unlink(ruta)            # solo ruido sin avisos: se recoge y se sigue
                return None
            conteo = f"{len(cuerpos)} avisos"
            if suprimidos:
                conteo += f" · suprimidos_por_tasa={suprimidos}"
            if fallidos:
                conteo += f" · envio_fallido={fallidos}"
            if ilegibles:
                conteo += f" · ilegibles={ilegibles}"
            if cuarentena:
                conteo += f" · en_cuarentena={cuarentena}"
            cabecera = (f"FARO · RESUMEN DIARIO DE REGLAS · {_campo_log(host, 64)} · "
                        f"{_iso_utc(_ahora_utc())}\n{conteo}")
            bloques = [f"clase={_campo_log(c, 160)} n={n}" for c, n in sorted(por_clase.items())] + cuerpos
            return ResumenDiario(tuple(_mensajes_del_resumen(bloques or ["(sin avisos legibles)"], cabecera)),
                                 tuple(buenos), str(cola))
    except Exception as exc:  # fail-soft: sin lectura segura no hay resumen y no se borra nada
        logger.warning("no se pudo componer el resumen diario (%s)", type(exc).__name__)
        return None


def confirmar_resumen(token: ResumenDiario) -> bool:
    """Fase 2: el llamador ya ENVIO los mensajes; se borran SOLO los archivos que
    este token reclamo (`<cola>.<ts>.procesando.<pid>-<token>` de esa cola: un
    token fabricado no borra otra cosa). True si no queda ninguno."""
    try:
        cola = Path(token.ruta_cola)
        patron = _patron_rotado(cola.name)
        with _Candado(cola):
            for ruta in token.rotados:
                p = Path(ruta)
                m = patron.fullmatch(p.name)
                if p.parent != cola.parent or m is None or m.group(2) is None:
                    logger.warning("confirmar_resumen: ruta fuera de la cola, no se borra")
                    return False
                try:
                    os.unlink(p)
                except FileNotFoundError:  # fail-soft: ya no existe (otro reclamo o confirmacion previa), objetivo cumplido
                    pass
        return True
    except Exception as exc:  # fail-soft: si no se borra, tras el lease se recoge de nuevo
        logger.warning("no se pudo confirmar el resumen diario (%s)", type(exc).__name__)
        return False


def emitir_aviso_inmediato(
    aviso: AvisoRegla, cfg: ConfigAviso, cred: Credenciales, *,
    enviar: Callable[[ConfigAviso, Credenciales, str], None] = enviar_telegram,
    limite: LimiteTasa, ruta_cola: "os.PathLike[str] | str",
) -> bool:
    """Manda UN aviso inmediato por el canal que trae `cfg`. Fail-soft TOTAL:
    cualquier fallo del envio queda en el log y devuelve False; jamas levanta y
    jamas altera la decision (que ya se aplico y persistio, §11). True solo si
    el envio salio.

    ENVIO SINCRONO: bloquea lo que tarde la red; los llamadores async la
    envuelven en `asyncio.to_thread`.

    `limite` y `ruta_cola` son OBLIGATORIOS a proposito: una tasa que no se
    pasa no limita nada, y una cola que no se pasa pierde lo suprimido. Si la
    tasa corta el aviso, NO se descarta: va a la cola del resumen diario con la
    marca `suprimido_por_tasa` y devuelve False. Si el ENVIO falla (excepcion o
    False) tambien va a la cola, con la marca `envio_fallido`: una negativa no
    puede quedar solo en un log."""
    try:
        if not limite.admitir(aviso.clase):
            logger.info("aviso de regla suprimido por tasa clase=%s", aviso.clase)
            acumular_para_resumen(aviso, ruta_cola, suprimido_por_tasa=True)
            return False
        if enviar(cfg, cred, aviso.texto) is False:
            raise AvisoNoEntregado("el envio devolvio False")
        return True
    except Exception as exc:  # fail-soft: la decision ya esta hecha y persistida; ningun fallo de Telegram puede tumbar al que la invoco
        logger.warning("aviso de regla no entregado (%s) clase=%s creado=%s",
                       type(exc).__name__, aviso.clase, aviso.creado_utc)
        # que no llego NO se pierde: a la cola del resumen con la marca (y el resumen la cuenta)
        acumular_para_resumen(aviso, ruta_cola, envio_fallido=True)
        return False
