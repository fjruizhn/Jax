"""El Puerto: el servidor MCP del ecosistema, en MODO LECTURA (spec §3 «Catalogo del Puerto»,
fila de lectura; plan 0.2).

Expone lo que el paquete fijado contiene y nada mas:

  resource  ecosistema://constitucion        la constitucion (nucleo comun sellado con el SHA)
  resource  ecosistema://agentes             el catalogo de agentes (JSON, sin sus cuerpos)
  resource  skill://{nombre}                 el SKILL.md de una skill
  tool      skills.buscar(consulta, limite)  busca skills por nombre, descripcion y cuerpo
  tool      skills.leer(nombre, archivo)     lee una skill (o un archivo de su carpeta)
  prompt    skills.leer(nombre)              la skill como prompt
  tool      agentes.listar()                 SOLO el catalogo. Lanzar agentes es otro paso (0.5)

Se construye UN servidor por conexion, con la identidad de ESA conexion ya cerrada sobre la
guardia: nada de la identidad se lee de un argumento ni de `_meta`.

LA GUARDIA (`Guardia`, un `ServerMiddleware` del SDK) envuelve CADA pedido, antes de que se
valide o se despache:
  1. mira el freno global (`jax.faro.freno`, que lee `jax.core.interruptor`). Puesto, o sin
     poder mirarlo: responde error MCP con codigo 423 y NO ejecuta;
  2. ejecuta y registra en la bitacora: id de llamada, identidad del socket, SHA del paquete,
     metodo y objetivo, hash de los argumentos (sin `_meta`) y del resultado, decision.

Fuente de la API (version instalada, mcp 2.2.0, leida del paquete): `MCPServer`,
`@mcp.tool/resource/prompt`, `ServerMiddleware` en `mcp.server.context`, `MCPError` en
`mcp.shared.exceptions`, `ToolError`/`ResourceError` en `mcp.server.mcpserver.exceptions`.
https://py.sdk.modelcontextprotocol.io/ (pagina de la API) -- NO VERIFICADO contra la web, solo
contra el codigo instalado.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
import logging
import uuid
from collections.abc import Callable
from typing import Any

from mcp.server import MCPServer
from mcp.server.context import CallNext, HandlerResult, ServerRequestContext
from mcp.server.mcpserver.exceptions import ResourceError, ToolError
from mcp.shared.exceptions import MCPError

from .bitacora import Bitacora, _campo_log
from .identidad import Identidad
from .paquete import NoExiste, PaqueteCargado

logger = logging.getLogger(__name__)
CODIGO_FRENO = 423          # el HTTP 423 Locked como codigo del error MCP
CODIGO_BITACORA = -32603    # error interno JSON-RPC: la bitacora no esta disponible
LIMITE_POR_DEFECTO = 20
LIMITE_MAXIMO = 100
LARGO_MAX_TEXTO = 200


def _json_canonico(valor: Any) -> str:
    return json.dumps(valor, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def _sha(texto: str) -> str:
    return hashlib.sha256(texto.encode()).hexdigest()


def _hash_resultado(resultado: Any) -> str:
    if hasattr(resultado, "model_dump_json"):
        return _sha(resultado.model_dump_json(by_alias=True, exclude_unset=True))
    return _sha(_json_canonico(resultado))


class Guardia:
    """Freno + bitacora por pedido. Una por conexion."""

    def __init__(self, *, identidad: Identidad, bitacora: Bitacora, sha_paquete: str, freno: Callable[[], bool]):
        self._identidad = identidad
        self._bitacora = bitacora
        self._sha_paquete = sha_paquete
        self._freno = freno

    async def _registrar(self, ctx: ServerRequestContext, id_llamada: str, hash_args: str, argumentos: str, t0: float, **extra) -> None:
        params = ctx.params or {}
        objetivo = params.get("name") if ctx.method in ("tools/call", "prompts/get") else params.get("uri")
        await self._bitacora.registrar(
            "llamada", id_llamada=id_llamada, **self._identidad.campos(), sha_paquete=self._sha_paquete,
            metodo=ctx.method, objetivo=str(objetivo)[:512] if objetivo is not None else "-",
            hash_args=hash_args, argumentos=argumentos, duracion_ms=round((time.monotonic() - t0) * 1000, 3), **extra)

    async def _anotar(self, ctx, id_llamada, hash_args, argumentos, t0, **extra) -> None:
        """Registra y, si NO se puede registrar, la llamada falla: no se entrega un resultado sin bitacora."""
        try:
            await self._registrar(ctx, id_llamada, hash_args, argumentos, t0, **extra)
        except Exception as exc:
            logger.exception("la bitacora no pudo registrar la llamada %s", id_llamada)
            raise MCPError(CODIGO_BITACORA, "BITACORA_NO_DISPONIBLE: no se entrega ningun resultado sin registro") from exc

    async def __call__(self, ctx: ServerRequestContext[Any, Any], call_next: CallNext) -> HandlerResult:
        if ctx.request_id is None:      # una notificacion no ejecuta nada ni responde
            return await call_next(ctx)
        t0 = time.monotonic()
        id_llamada = uuid.uuid4().hex
        cuerpo = {k: v for k, v in (ctx.params or {}).items() if k != "_meta"}   # `_meta` es ruido de transporte
        hash_args = _sha(_json_canonico(cuerpo))
        argumentos = _campo_log(_json_canonico(cuerpo), 200)    # saneado y acotado: para leer, no para decidir
        try:
            frenado = await asyncio.to_thread(self._freno)
        except Exception:  # fail-closed: sin poder mirar el freno, se da por puesto
            frenado = True
        if frenado:
            try:
                await self._registrar(ctx, id_llamada, hash_args, argumentos, t0, decision="denegado", motivo="freno",
                                      resultado="no_ejecutado", hash_resultado="")
            except Exception:  # fail-closed: se deniega igual; solo la anotacion fallo y queda en el log
                logger.exception("la bitacora no pudo registrar la denegacion %s", id_llamada)
            raise MCPError(CODIGO_FRENO, "FRENO_PUESTO: el interruptor global esta puesto; no se ejecuta nada",
                           data={"http": CODIGO_FRENO})
        try:
            resultado = await call_next(ctx)
        except MCPError as exc:
            await self._anotar(ctx, id_llamada, hash_args, argumentos, t0, decision="permitido", motivo="", resultado="error",
                               hash_resultado=_sha(_json_canonico({"codigo": exc.code, "mensaje": exc.message})))
            raise
        except Exception as exc:
            await self._anotar(ctx, id_llamada, hash_args, argumentos, t0, decision="permitido", motivo="", resultado="error",
                               hash_resultado=_sha(_json_canonico({"excepcion": type(exc).__name__})))
            raise
        con_error = bool(getattr(resultado, "is_error", False))
        await self._anotar(ctx, id_llamada, hash_args, argumentos, t0, decision="permitido", motivo="",
                           resultado="error" if con_error else "ok", hash_resultado=_hash_resultado(resultado))
        return resultado


def _texto_acotado(valor: str, campo: str) -> str:
    if len(valor) > LARGO_MAX_TEXTO:
        raise ToolError(f"{campo} excede {LARGO_MAX_TEXTO} caracteres")
    return valor


def construir_servidor(paquete: PaqueteCargado, *, identidad: Identidad, bitacora: Bitacora,
                       freno: Callable[[], bool]) -> MCPServer:
    """Un servidor MCP para UNA conexion. Todo se sirve desde `paquete` (memoria verificada)."""
    guardia = Guardia(identidad=identidad, bitacora=bitacora, sha_paquete=paquete.sha, freno=freno)
    mcp = MCPServer("faro", version="0", instructions="El Faro: el ecosistema (constitucion, skills, agentes) en solo lectura.",
                    middleware=[guardia])

    @mcp.resource("ecosistema://constitucion", name="constitucion", mime_type="text/markdown",
                  description="La constitucion del ecosistema (nucleo comun), sellada con el SHA de claude-skills.")
    def constitucion() -> str:
        return paquete.constitucion

    @mcp.resource("ecosistema://agentes", name="agentes", mime_type="application/json",
                  description="Catalogo de agentes del ecosistema (sin sus cuerpos).")
    def agentes_recurso() -> str:
        return json.dumps(paquete.catalogo_agentes(), ensure_ascii=False)

    @mcp.resource("skill://{nombre}", name="skill", mime_type="text/markdown",
                  description="El SKILL.md de una skill del paquete.")
    def skill_recurso(nombre: str) -> str:
        try:
            return paquete.leer(nombre)
        except NoExiste as exc:
            raise ResourceError(str(exc)) from None

    @mcp.tool(name="skills.buscar", description="Busca skills del ecosistema por nombre, descripcion o contenido. "
              "Consulta vacia: lista todas.")
    def skills_buscar(consulta: str = "", limite: int = LIMITE_POR_DEFECTO) -> list[dict[str, str]]:
        _texto_acotado(consulta, "consulta")
        if not 1 <= limite <= LIMITE_MAXIMO:
            raise ToolError(f"limite fuera de rango (1..{LIMITE_MAXIMO})")
        return paquete.buscar(consulta, limite)

    @mcp.tool(name="skills.leer", description="Lee una skill del ecosistema: su SKILL.md, o un archivo de su carpeta.")
    def skills_leer(nombre: str, archivo: str = "SKILL.md") -> str:
        _texto_acotado(nombre, "nombre")
        _texto_acotado(archivo, "archivo")
        try:
            return paquete.leer(nombre, archivo)
        except NoExiste as exc:
            raise ToolError(str(exc)) from None

    @mcp.tool(name="agentes.listar", description="Catalogo de agentes del ecosistema (nombre, descripcion, modelo). "
              "Solo el catalogo: no lanza agentes.")
    def agentes_listar() -> list[dict[str, str]]:
        return paquete.catalogo_agentes()

    @mcp.prompt(name="skills.leer", description="La skill pedida, como prompt.")
    def skills_prompt(nombre: str) -> str:
        try:
            return paquete.leer(_texto_acotado(nombre, "nombre"))
        except (NoExiste, ToolError) as exc:
            raise ValueError(str(exc)) from None

    return mcp


def servidor_de_bajo_nivel(mcp: MCPServer):
    """El `Server` de bajo nivel que el SDK mantiene dentro de `MCPServer`.

    `MCPServer` solo trae ejecucion por stdio del PROCESO, SSE y streamable-HTTP; para servir
    por un socket Unix (el Puerto) hay que correr el bucle del bajo nivel sobre nuestras
    propias corrientes, que es lo que hace internamente `MCPServer.run_stdio_async`
    (mcp/server/mcpserver/server.py, `_lowlevel_server.run(...)`). Es un atributo PRIVADO del
    SDK: NO VERIFICADO como estable entre versiones. Por eso el SDK va fijado con hash
    (`requirements-faro.txt`) y el extremo a extremo con cliente real
    (`tests/test_faro_puerto_e2e.py`) rompe si una version nueva lo mueve."""
    return mcp._lowlevel_server
