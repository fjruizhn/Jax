"""Proxy con carril para el Ejecutor. Spec: §3.4 bis.

El Ejecutor es Claude Code (Node) corriendo como `axioma`: un `flock` en Python
no frena su tráfico HTTP. Por eso `ANTHROPIC_BASE_URL` apunta ACÁ, y el proxy,
por CADA petición:

1. toma `carril_ejecutor_async` (la Mesa pasa primero);
2. reenvía método, ruta, cabeceras y cuerpo al upstream (Ollama);
3. devuelve la respuesta en STREAMING, trozo a trozo (Claude Code usa SSE);
4. suelta el carril cuando termina el stream, o cuando el cliente corta, o
   cuando el upstream falla.

Tope vencido → HTTP 503 con un cuerpo de máquina y `x-should-retry: false`: la
misión falla, no se cuela, y la petición NUNCA llega al upstream.

Qué se usa y por qué: servidor con `asyncio.start_server` + `h11`, cliente con
`httpx`. Sin dependencias nuevas: `httpx` está en requirements.txt y `h11`
también (lo trae httpx vía httpcore). `fastapi` está, pero `uvicorn` NO está
instalado en el venv de jax ni en el job de CI del Ejecutor, y Starlette sin
servidor ASGI no escucha un puerto.

Una petición por conexión (`connection: close`). Así, cualquier cosa que
llegue por el socket del cliente después del pedido sólo puede ser su cierre,
y el proxy se entera del corte AUNQUE el upstream esté callado (esperando en la
cola de Ollama) — que es justo cuando retener el carril sale más caro. Contra
localhost, un handshake por petición no se nota al lado de una inferencia.

NO se loguean cabeceras ni cuerpos: llevan la llave y datos de clientes. Los
logs dicen método, ruta SIN query, estado y códigos.

Configuración (sin defaults para lo que decide a dónde va el tráfico):
  JAX_PROXY_CARRIL_UPSTREAM   URL base del upstream (obligatoria)
  JAX_PROXY_CARRIL_RAIZ       directorio de los locks (obligatoria)
  JAX_PROXY_CARRIL_TOPE_S     tope de espera del carril, en segundos (obligatoria)
  JAX_PROXY_CARRIL_PUERTO     puerto de escucha (obligatoria)
  JAX_PROXY_CARRIL_HOST       dirección de escucha (opcional; por omisión sólo
                              loopback, por seguridad: el proxy no autentica)
  JAX_EJECUTOR_REGISTRO       registro de C3 (obligatoria, ruta absoluta): sin
                              registro no hay proxy, sin proxy no hay cerebro
  JAX_EJECUTOR_PAUSA          pausa propia del Ejecutor que pone C5 (obligatoria,
                              ruta absoluta)
  JAX_EJECUTOR_VIGIA_LATIDO   latido del vigía de C5 (obligatoria, ruta absoluta)
  JAX_EJECUTOR_VIGIA_LATIDO_MAX_S  edad máxima del latido, en segundos (obligatoria, > 0)
  JAX_PROXY_CARRIL_MODELO     el ÚNICO `model` que la jaula puede pedir (obligatoria).
                              Otro → 403 sin tocar Ollama: cargaría otro modelo y
                              desalojaría el de la Mesa (SP3, 2026-09-17)
  JAX_PROXY_CARRIL_MAX_SALIDA_TOKENS  tope de `max_tokens` por petición (obligatoria,
                              entero > 0). Más → 403. La cuenta: §6.3 del spec de Fase 2

C3 (registro intocable, decisión D-SP1-2 del índice de SP1): cada `tool_use` que
el cerebro pide se anota en el registro ANTES de reenviar el trozo que lo completa,
y cada `tool_result` nuevo de una petición, ANTES de subirla al upstream. Si no se
puede anotar, o lo que pasa no se puede leer, se corta: la herramienta nunca llega.

C5 (auditor en vivo, plan 4 de SP1): con la pausa del Ejecutor puesta, o sin un
vigía que lata, el proxy responde 423 sin tocar el upstream, y el trozo que completa un
`tool_use` no sale (se corta el stream DESPUÉS de anotarlo). Sin auditor no hay cerebro.

C4 (freno en vuelo, plan 3 de SP1): el interruptor global de JAX (`JAX_KILL_SWITCH_PATH`,
jax/core/interruptor.py, obligatorio: sin saber dónde está no arranca) frena igual que la
pausa del Ejecutor. Con cualquiera de las dos puestas: 423 sin tocar el upstream (los
resultados que ya corrieron se anotan igual), y lo que está en vuelo —esperando el carril,
esperando al upstream o en pleno stream— se corta en menos de un segundo: 423 legible si
todavía no salieron cabeceras, conexión abortada (stream truncado) si ya salieron.

Corre con:  python -m jax.ejecutor.proxy_carril
"""
from __future__ import annotations

import asyncio
import collections
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import h11
import httpx

from jax.core import interruptor
from jax.core.cliente_http_compartido import crear_cliente_http
from jax.ejecutor.cita import Motivo
from jax.ejecutor.contratos import lectura
from jax.ejecutor.contratos import pausa as pausa_c5
from jax.ejecutor.contratos.registro import Registro
from jax.ejecutor.prioridad import ESPERA_AGOTADA, EsperaAgotada, carril_ejecutor_async

log = logging.getLogger(__name__)

UPSTREAM_INALCANZABLE = "upstream_inalcanzable"
PETICION_INVALIDA = "peticion_invalida"
CONFIG_FALTA = "config_falta"
CONFIG_INVALIDA = "config_invalida"
REGISTRO_FALLO = "registro_fallo"
REGISTRO_ILEGIBLE = "registro_ilegible"
RUTA_NO_PERMITIDA = "ruta_no_permitida"
EJECUTOR_PAUSADO = "ejecutor_pausado"
VIGIA_SIN_LATIDO = "vigia_sin_latido"
MODELO_NO_PERMITIDO = "modelo_no_permitido"
SALIDA_NO_PERMITIDA = "salida_no_permitida"
#: Lo que el arnés manda de verdad (medido 2026-09-17, arnés 2.1.273 por este proxy, 43
#: peticiones en g1_20260917/proxy_v3.jsonl: `HEAD /api/hello` y `POST /v1/messages`). El
#: upstream es el Ollama de producción: nada más llega. SP3 (2026-09-17) cerró también las
#: lecturas: `/api/ps`, `/api/tags` o `/api/show` le dicen a la jaula qué hay cargado y con qué.
_RUTAS_PERMITIDAS = frozenset({("HEAD", "/api/hello"), ("POST", "/v1/messages")})
KILL_SWITCH_ACTIVO = "kill_switch_activo"
_RUTAS_DE_MENSAJES = frozenset({"/v1/messages"})
_RESULTADOS_RECORDADOS = 10000

_HOST_POR_OMISION = "127.0.0.1"
_LEER_BYTES = 65536

#: Cabeceras de salto (RFC 9110 §7.6.1) y las que el proxy recalcula. `accept-encoding`
#: lo pone SIEMPRE el proxy en `identity`: un cuerpo comprimido no se puede leer al pasar.
_NO_REENVIAR = frozenset({
    b"connection", b"keep-alive", b"proxy-connection", b"proxy-authenticate",
    b"proxy-authorization", b"te", b"trailer", b"transfer-encoding", b"upgrade",
    b"host", b"content-length", b"accept-encoding",
})
_NO_DEVOLVER = _NO_REENVIAR - {b"content-length"}

_TIMEOUT_UPSTREAM = httpx.Timeout(connect=10.0, read=None, write=None, pool=None)


class ConfigInvalida(ValueError):
    """Falta o no vale una variable de configuración. El argumento es un
    `Motivo` (sin prosa)."""


@dataclass(frozen=True)
class Config:
    upstream: str
    raiz: Path
    tope_s: float
    host: str
    puerto: int
    registro: Path
    pausa: Path
    latido: Path
    latido_max_s: float
    modelo: str
    max_salida_tokens: int


def config_desde_entorno(env=None) -> Config:
    env = os.environ if env is None else env

    def obligatoria(nombre):
        valor = env.get(nombre, "").strip()
        if not valor:
            raise ConfigInvalida(Motivo(CONFIG_FALTA, (("variable", nombre),)))
        return valor

    def numero(nombre, tipo):
        crudo = obligatoria(nombre)
        try:
            valor = tipo(crudo)
        except ValueError as exc:
            raise ConfigInvalida(Motivo(CONFIG_INVALIDA, (("variable", nombre),))) from exc
        if valor < 0:
            raise ConfigInvalida(Motivo(CONFIG_INVALIDA, (("variable", nombre),)))
        return valor

    def positivo(nombre):
        valor = numero(nombre, int)
        if valor <= 0:
            raise ConfigInvalida(Motivo(CONFIG_INVALIDA, (("variable", nombre),)))
        return valor

    upstream = obligatoria("JAX_PROXY_CARRIL_UPSTREAM")
    if not upstream.startswith(("http://", "https://")):
        raise ConfigInvalida(Motivo(CONFIG_INVALIDA, (("variable", "JAX_PROXY_CARRIL_UPSTREAM"),)))
    def absoluta(nombre):
        ruta = Path(obligatoria(nombre))
        if not ruta.is_absolute():
            raise ConfigInvalida(Motivo(CONFIG_INVALIDA, (("variable", nombre),)))
        return ruta

    registro = absoluta("JAX_EJECUTOR_REGISTRO")
    pausa = absoluta(pausa_c5.VARIABLE_RUTA)
    latido = absoluta(pausa_c5.VARIABLE_LATIDO)
    latido_max_s = numero(pausa_c5.VARIABLE_LATIDO_MAX_S, float)
    if not latido_max_s > 0 or latido_max_s == float("inf"):
        raise ConfigInvalida(Motivo(CONFIG_INVALIDA, (("variable", pausa_c5.VARIABLE_LATIDO_MAX_S),)))
    return Config(
        upstream=upstream.rstrip("/"),
        raiz=Path(obligatoria("JAX_PROXY_CARRIL_RAIZ")),
        tope_s=numero("JAX_PROXY_CARRIL_TOPE_S", float),
        host=env.get("JAX_PROXY_CARRIL_HOST", "").strip() or _HOST_POR_OMISION,
        puerto=numero("JAX_PROXY_CARRIL_PUERTO", int),
        registro=registro,
        pausa=pausa,
        latido=latido,
        latido_max_s=latido_max_s,
        modelo=obligatoria("JAX_PROXY_CARRIL_MODELO"),
        max_salida_tokens=positivo("JAX_PROXY_CARRIL_MAX_SALIDA_TOKENS"),
    )


def _cuerpo_error(codigo: str, datos: tuple = ()) -> bytes:
    """Formato de error de la API de Anthropic (`type: error`), con el código
    en `error.type` y sus datos al lado. Sin frases: la política es cero
    strings de usuario en el código."""
    error = {"type": codigo, **dict(datos)}
    return json.dumps({"type": "error", "error": error}, separators=(",", ":")).encode()


async def _enviar(conn: h11.Connection, writer: asyncio.StreamWriter, evento) -> None:
    writer.write(conn.send(evento))
    await writer.drain()


async def _responder_error(conn, writer, estado: int, motivo: Motivo, extra=(), metodo: str = "") -> None:
    cuerpo = _cuerpo_error(motivo.codigo, motivo.datos)
    cabeceras = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(cuerpo)).encode()),
        (b"connection", b"close"),
        *extra,
    ]
    await _enviar(conn, writer, h11.Response(status_code=estado, headers=cabeceras))
    if metodo != "HEAD":
        # A un HEAD se le contesta sin cuerpo (RFC 9110 §9.3.2); h11 lanza si se lo manda.
        await _enviar(conn, writer, h11.Data(data=cuerpo))
    await _enviar(conn, writer, h11.EndOfMessage())


async def _leer_peticion(conn: h11.Connection, reader: asyncio.StreamReader):
    peticion, trozos = None, []
    while True:
        evento = conn.next_event()
        if evento is h11.NEED_DATA:
            conn.receive_data(await reader.read(_LEER_BYTES))
            continue
        if isinstance(evento, h11.Request):
            peticion = evento
        elif isinstance(evento, h11.Data):
            trozos.append(evento.data)
        elif isinstance(evento, h11.EndOfMessage):
            return peticion, b"".join(trozos)
        elif isinstance(evento, h11.ConnectionClosed):
            return None, b""


def _fuera_de_limites(cuerpo: bytes, cfg: Config) -> str | None:
    """¿La petición de mensajes pide otro modelo o más salida que el tope? Devuelve el
    código, o None si está dentro. Lo que no se puede leer, no está dentro (fail-closed)."""
    try:
        pedido = json.loads(cuerpo)
    except (ValueError, UnicodeDecodeError):  # fail-soft: no se reenvía; un cuerpo ilegible se trata como modelo no permitido (403)
        return MODELO_NO_PERMITIDO
    if not isinstance(pedido, dict) or pedido.get("model") != cfg.modelo:
        return MODELO_NO_PERMITIDO
    salida = pedido.get("max_tokens")
    if type(salida) is not int or not 0 < salida <= cfg.max_salida_tokens:
        return SALIDA_NO_PERMITIDA
    return None


def _ruta_sin_query(destino: bytes) -> str:
    return destino.split(b"?", 1)[0].decode("latin-1")


class _Proxy:
    def __init__(self, cfg: Config, registro: Registro) -> None:
        self.cfg = cfg
        self.registro = registro
        # tool_use_id ya anotados como resultado: la historia se repite entera en cada
        # petición (medido con el arnés 2.1.273), cada resultado se anota una vez.
        self._resultados_anotados: collections.OrderedDict = collections.OrderedDict()
        # Un cliente propio del proxy (pool de conexiones), no uno por petición:
        # vive lo que vive el servidor y se cierra en `cerrar()`. Se construye con
        # crear_cliente_http() (E-24, el único constructor del árbol de servicio)
        # y no con obtener_cliente_http(): cerrar el del proceso al apagar el
        # proxy cerraría el de cualquier otro usuario del mismo loop.
        self.cliente = crear_cliente_http()
    async def cerrar(self) -> None:
        await self.cliente.aclose()

    async def atender(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        conn = h11.Connection(h11.SERVER)
        try:
            try:
                peticion, cuerpo = await _leer_peticion(conn, reader)
            except h11.RemoteProtocolError:
                await _responder_error(conn, writer, 400, Motivo(PETICION_INVALIDA))
                return
            if peticion is None:
                return
            en_vuelo = asyncio.Event()
            trabajo = asyncio.create_task(self._reenviar(conn, writer, peticion, cuerpo, en_vuelo))
            vigia = asyncio.create_task(_esperar_cierre(reader))
            # C4: el freno vigila sólo lo que ya está en vuelo. Antes, `_reenviar` mira el
            # freno él mismo DESPUÉS de anotar los resultados: lo que ya corrió no se pierde.
            freno = asyncio.create_task(self._esperar_freno(en_vuelo))
            await asyncio.wait({trabajo, vigia, freno}, return_when=asyncio.FIRST_COMPLETED)
            if not trabajo.done():
                # Cortó el cliente o se puso el freno: cancelar suelta el carril
                # (espera o stream) y cierra la respuesta del upstream.
                frenado = freno.result() if freno.done() else None
                log.info("proxy_carril %s metodo=%s ruta=%s", frenado or "corte_cliente",
                         peticion.method.decode("latin-1"), _ruta_sin_query(peticion.target))
                trabajo.cancel()
                await asyncio.wait({trabajo})
                if frenado is not None:
                    await _cortar_por_freno(conn, writer, frenado, peticion.method.decode("latin-1"))
            for t in (vigia, freno):
                t.cancel()
            await asyncio.wait({vigia, freno})
            if not trabajo.cancelled() and trabajo.exception() is not None:
                exc = trabajo.exception()
                if not isinstance(exc, ConnectionError):
                    raise exc
                log.info("proxy_carril corte_cliente tipo=%s", type(exc).__name__)
        finally:
            await _cerrar(writer)

    def _freno_puesto_ahora(self) -> str | None:
        """C4: el interruptor de JAX o la pausa del Ejecutor (las dos frenan en vuelo)."""
        try:
            if interruptor.interruptor_activo():
                return KILL_SWITCH_ACTIVO
        except interruptor.InterruptorSinConfigurar:
            return KILL_SWITCH_ACTIVO  # fail-closed: sin saber dónde está el interruptor se frena (arrancar ya lo exige; esto cubre que la variable desaparezca)
        if pausa_c5.pausa_puesta(self.cfg.pausa):
            return EJECUTOR_PAUSADO
        return None

    async def _esperar_freno(self, en_vuelo: asyncio.Event) -> str:
        await en_vuelo.wait()
        while (frenado := await asyncio.to_thread(self._freno_puesto_ahora)) is None:
            await asyncio.sleep(interruptor.INTERVALO_DE_SONDEO)
        return frenado

    def _frenado_ahora(self) -> str | None:
        if (puesto := self._freno_puesto_ahora()) is not None:
            return puesto
        if not pausa_c5.latido_fresco(self.cfg.latido, self.cfg.latido_max_s):
            return VIGIA_SIN_LATIDO
        return None

    async def _frenado(self) -> str | None:
        """C4/C5: ¿interruptor de JAX, pausa del Ejecutor, o vigía sin latido? (stat: fuera del loop)."""
        return await asyncio.to_thread(self._frenado_ahora)

    async def _anotar(self, evento: dict) -> None:
        # write + fsync bloquean: fuera del event loop.
        await asyncio.to_thread(self.registro.anotar, evento)

    async def _anotar_resultados(self, ruta: str, cuerpo: bytes) -> None:
        resultados = lectura.resultados_de_peticion(cuerpo)
        if resultados is None:
            await self._anotar({"evento": "peticion_ilegible", "ruta": ruta})
            return
        for r in resultados:
            if r.tool_use_id in self._resultados_anotados:
                continue
            await self._anotar(lectura.evento_de_resultado(r, ruta))
            self._resultados_anotados[r.tool_use_id] = None
            while len(self._resultados_anotados) > _RESULTADOS_RECORDADOS:
                self._resultados_anotados.popitem(last=False)

    async def _reenviar(self, conn, writer, peticion: h11.Request, cuerpo: bytes,
                        en_vuelo: asyncio.Event | None = None) -> None:
        metodo = peticion.method.decode("latin-1")
        ruta = _ruta_sin_query(peticion.target)
        de_mensajes = metodo == "POST" and ruta in _RUTAS_DE_MENSAJES
        # Orden (C5 + SP3 + C4): con un freno puesto (interruptor de JAX, pausa del Ejecutor
        # o vigía sin latido) TODA petición recibe 423 sin tocar el carril ni el upstream. La
        # única que se lee antes de responder es un POST de mensajes que pasa la política de
        # SP3 (modelo y salida): sus resultados de herramientas ya corrieron y tienen que
        # quedar en el registro de C3 (C4: el freno no borra el rastro). Lo demás, bajo freno,
        # ni se anota: una petición fuera de política no se anotaría tampoco sin freno.
        frenado = await self._frenado()
        fuera = _fuera_de_limites(cuerpo, self.cfg) if de_mensajes else None
        if frenado is not None and (not de_mensajes or fuera is not None):
            log.warning("proxy_carril %s metodo=%s ruta=%s", frenado, metodo, ruta)
            await _responder_error(conn, writer, 423, Motivo(frenado),
                                   extra=((b"x-should-retry", b"false"),), metodo=metodo)
            return
        if (metodo, ruta) not in _RUTAS_PERMITIDAS:
            log.warning("proxy_carril %s metodo=%s", RUTA_NO_PERMITIDA, metodo)
            await _responder_error(conn, writer, 403, Motivo(RUTA_NO_PERMITIDA), metodo=metodo)
            return
        if fuera is not None:
            # Antes del carril, del registro y del upstream: otro modelo desalojaría el de la
            # Mesa, y una salida sin tope rompe la cuenta de su espera (§6.3 del spec de Fase 2).
            log.warning("proxy_carril %s metodo=%s ruta=%s", fuera, metodo, ruta)
            await _responder_error(conn, writer, 403, Motivo(fuera), metodo=metodo)
            return
        if de_mensajes:
            try:
                await self._anotar_resultados(ruta, cuerpo)
            except OSError as exc:
                # Un resultado sin anotar no sube al cerebro.
                log.error("proxy_carril %s metodo=%s ruta=%s tipo=%s", REGISTRO_FALLO, metodo, ruta, type(exc).__name__)
                await _responder_error(conn, writer, 502, Motivo(REGISTRO_FALLO), metodo=metodo)
                return
        # Después de anotar lo que ya corrió (C3) y antes de pedirle nada al cerebro.
        frenado = await self._frenado()
        if frenado is not None:
            log.warning("proxy_carril %s metodo=%s ruta=%s", frenado, metodo, ruta)
            await _responder_error(conn, writer, 423, Motivo(frenado),
                                   extra=((b"x-should-retry", b"false"),), metodo=metodo)
            return
        if en_vuelo is not None:
            en_vuelo.set()
        try:
            async with carril_ejecutor_async(self.cfg.raiz, self.cfg.tope_s):
                cabeceras = [(k, v) for k, v in peticion.headers if k.lower() not in _NO_REENVIAR]
                cabeceras.append((b"accept-encoding", b"identity"))
                solicitud = self.cliente.build_request(
                    metodo, self.cfg.upstream + peticion.target.decode("latin-1"),
                    headers=cabeceras, content=cuerpo,
                    # Sin tope de lectura: el primer byte puede tardar lo que
                    # tarde la cola de Ollama; el corte lo decide el cliente, y
                    # el proxy lo ve. Por petición: el default del cliente es 5 s.
                    timeout=_TIMEOUT_UPSTREAM)
                try:
                    respuesta = await self.cliente.send(solicitud, stream=True)
                except httpx.HTTPError as exc:
                    log.warning("proxy_carril %s metodo=%s ruta=%s tipo=%s",
                                UPSTREAM_INALCANZABLE, metodo, ruta, type(exc).__name__)
                    await _responder_error(conn, writer, 502, Motivo(UPSTREAM_INALCANZABLE), metodo=metodo)
                    return
                try:
                    await self._devolver(conn, writer, respuesta, metodo, ruta, de_mensajes)
                finally:
                    await respuesta.aclose()
        except EsperaAgotada as exc:
            (motivo,) = exc.args
            log.info("proxy_carril %s metodo=%s ruta=%s", ESPERA_AGOTADA, metodo, ruta)
            # `x-should-retry: false`: el SDK de Anthropic respeta esta cabecera.
            # Reintentar un 503 del carril sería colarse en cuotas.
            await _responder_error(conn, writer, 503, motivo,
                                   extra=((b"x-should-retry", b"false"),), metodo=metodo)

    async def _devolver(self, conn, writer, respuesta, metodo: str, ruta: str, de_mensajes: bool) -> None:
        codificacion = respuesta.headers.get("content-encoding", "identity").strip().lower()
        if codificacion not in ("", "identity"):
            log.error("proxy_carril %s metodo=%s ruta=%s codificacion=%s", REGISTRO_ILEGIBLE, metodo, ruta, codificacion)
            await _responder_error(conn, writer, 502, Motivo(REGISTRO_ILEGIBLE), metodo=metodo)
            return
        devolver = [(k, v) for k, v in respuesta.headers.raw if k.lower() not in _NO_DEVOLVER]
        devolver.append((b"connection", b"close"))
        inicio = h11.Response(status_code=respuesta.status_code, headers=devolver,
                              reason=respuesta.reason_phrase.encode("latin-1"))
        es_sse = respuesta.headers.get("content-type", "").startswith("text/event-stream")
        if not es_sse:
            # Un cuerpo entero se lee ANTES de entregarlo: un tool_use dentro no sale sin anotar.
            try:
                crudo = b"".join([trozo async for trozo in respuesta.aiter_raw()])
            except httpx.HTTPError as exc:
                log.warning("proxy_carril upstream_cortado metodo=%s ruta=%s tipo=%s", metodo, ruta, type(exc).__name__)
                await _responder_error(conn, writer, 502, Motivo(UPSTREAM_INALCANZABLE), metodo=metodo)
                return
            if de_mensajes:
                pedidas = lectura.herramientas_de_mensaje(crudo)
                if pedidas is None and 200 <= respuesta.status_code < 300:
                    # El arnés decide si es stream por lo que pidió, no por el content-type:
                    # lo que el proxy no puede leer, no se entrega.
                    log.error("proxy_carril %s metodo=%s ruta=%s", REGISTRO_ILEGIBLE, metodo, ruta)
                    await _responder_error(conn, writer, 502, Motivo(REGISTRO_ILEGIBLE), metodo=metodo)
                    return
                try:
                    for pedida in pedidas or []:
                        await self._anotar(lectura.evento_de_pedida(pedida, ruta))
                except OSError as exc:
                    log.error("proxy_carril %s metodo=%s ruta=%s tipo=%s", REGISTRO_FALLO, metodo, ruta, type(exc).__name__)
                    await _responder_error(conn, writer, 502, Motivo(REGISTRO_FALLO), metodo=metodo)
                    return
                if pedidas and (frenado := await self._frenado()) is not None:
                    # Anotado y NO entregado: con C5 frenando, la herramienta no llega al arnés.
                    log.warning("proxy_carril %s metodo=%s ruta=%s", frenado, metodo, ruta)
                    await _responder_error(conn, writer, 423, Motivo(frenado),
                                           extra=((b"x-should-retry", b"false"),), metodo=metodo)
                    return
            await _enviar(conn, writer, inicio)
            log.info("proxy_carril peticion metodo=%s ruta=%s estado=%d", metodo, ruta, respuesta.status_code)
            await _enviar(conn, writer, h11.Data(data=crudo))
            await _enviar(conn, writer, h11.EndOfMessage())
            return
        await _enviar(conn, writer, inicio)
        log.info("proxy_carril peticion metodo=%s ruta=%s estado=%d", metodo, ruta, respuesta.status_code)
        lector = lectura.LectorSSE()
        try:
            async for trozo in respuesta.aiter_raw():
                try:
                    pedidas = lector.alimentar(trozo)
                    for pedida in pedidas:
                        await self._anotar(lectura.evento_de_pedida(pedida, ruta))
                except OSError as exc:
                    # El trozo que completa el tool_use NO sale: el arnés ve un stream cortado.
                    log.error("proxy_carril %s metodo=%s ruta=%s tipo=%s", REGISTRO_FALLO, metodo, ruta, type(exc).__name__)
                    writer.transport.abort()
                    return
                if pedidas and (frenado := await self._frenado()) is not None:
                    # C5 frenó: el tool_use queda anotado y su trozo final NO sale.
                    log.warning("proxy_carril %s metodo=%s ruta=%s", frenado, metodo, ruta)
                    writer.transport.abort()
                    return
                await _enviar(conn, writer, h11.Data(data=trozo))
        except httpx.HTTPError as exc:
            # Las cabeceras ya salieron: no hay estado que cambiar. Se corta la conexión
            # SIN cerrar el mensaje, para que el cliente vea un stream truncado.
            log.warning("proxy_carril upstream_cortado metodo=%s ruta=%s tipo=%s", metodo, ruta, type(exc).__name__)
            writer.transport.abort()
            return
        await _enviar(conn, writer, h11.EndOfMessage())


async def _cerrar(writer: asyncio.StreamWriter) -> None:
    writer.close()
    try:
        await writer.wait_closed()
    except ConnectionError:  # fail-soft: cerrar un socket que el cliente ya cerró; no queda nada que avisarle
        log.debug("proxy_carril cierre_con_conexion_perdida")


async def _cortar_por_freno(conn: h11.Connection, writer: asyncio.StreamWriter, frenado: str,
                            metodo: str = "") -> None:
    """C4: sin cabeceras enviadas, 423 legible con `x-should-retry: false` (el arnés no
    reintenta); con la respuesta ya empezada, abortar sin EndOfMessage: el arnés ve un
    stream truncado, nunca uno completo."""
    if conn.our_state is h11.SEND_RESPONSE:
        try:
            await _responder_error(conn, writer, 423, Motivo(frenado),
                                   extra=((b"x-should-retry", b"false"),), metodo=metodo)
            return
        except (ConnectionError, h11.LocalProtocolError):  # fail-soft: el cliente ya no está o h11 no admite respuesta; se aborta igual abajo
            log.info("proxy_carril corte_sin_respuesta codigo=%s", frenado)
    writer.transport.abort()


async def _esperar_cierre(reader: asyncio.StreamReader) -> None:
    """Con una petición por conexión, lo único que puede llegar después del
    pedido es el cierre. Los bytes de más se descartan."""
    while await reader.read(_LEER_BYTES):
        continue


class Servidor:
    """El servidor escuchando y el cliente HTTP compartido, que vive lo que
    vive el servidor."""

    def __init__(self, servidor: asyncio.Server, proxy: _Proxy) -> None:
        self._servidor = servidor
        self._proxy = proxy

    @property
    def sockets(self):
        return self._servidor.sockets

    def close(self) -> None:
        self._servidor.close()

    async def wait_closed(self) -> None:
        await self._servidor.wait_closed()
        await self._proxy.cerrar()
        self._proxy.registro.cerrar()

    async def serve_forever(self) -> None:
        try:
            await self._servidor.serve_forever()
        finally:
            self.close()
            await self.wait_closed()


async def arrancar(cfg: Config) -> Servidor:
    interruptor.ruta_del_interruptor()  # InterruptorSinConfigurar: sin saber dónde está el freno, no hay cerebro
    # Un registro que no cuadra NO se abre (RegistroCorrupto): sin registro no hay cerebro.
    registro = await asyncio.to_thread(Registro, cfg.registro)
    try:
        await asyncio.to_thread(registro.anotar, {"evento": "registro_abierto", "pid": os.getpid()})
        proxy = _Proxy(cfg, registro)
        servidor = await asyncio.start_server(proxy.atender, cfg.host, cfg.puerto)
    except BaseException:
        registro.cerrar()
        raise
    return Servidor(servidor, proxy)


async def _principal() -> None:
    cfg = config_desde_entorno()
    servidor = await arrancar(cfg)
    log.info("proxy_carril escuchando host=%s puerto=%d", cfg.host, cfg.puerto)
    await servidor.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(_principal())
