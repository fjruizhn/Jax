"""
JAX 2.0 — Contrato de musculos.

Un "musculo" es un proveedor de inteligencia (API, CLI o local) que JAX
orquesta. El nucleo solo conoce el contrato:

    Muscle.invoke(prompt, model=None, history=None) -> str

MEMORIA DE CONVERSACION (compartida):
  history es una lista de turnos previos, en formato neutro:
      [{"role": "user", "content": "..."},
       {"role": "assistant", "content": "..."}, ...]
  El historial NO incluye el mensaje actual (ese va en `prompt`). Cada
  implementacion lo inserta en su formato nativo de mensajes, sin duplicar.
  Es compartido: una sola conversacion para todas las facetas (un solo JAX).

En memoria de Jairo Urbina.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod

import json

from jax.core.credential_resolver import resolve_credential_instrumented, CredentialUnavailableError
from jax.core.model_catalog import record_resolved_version_safe
from jax.core.grounding_sources import build_sources, render_sources_block, resolve_redirects
from jax.core.contrato_dispatch import ModelDispatchConfigError, limite_de_salida
from jax.core.redaccion import recortar_redactado
from jax.core.cliente_http_compartido import obtener_cliente_http

# provider (nombre interno de config.toml) -> provider_id (tabla `credential`).
# "kimi"/"zai" son alias historicos que no coinciden con el provider_id real.
_PROVIDER_ID_MAP = {
    "deepseek": "deepseek",
    "gemini": "gemini",
    "openai": "openai",
    "kimi": "moonshot",
    "zhipu": "zhipu",
    "zai": "zhipu",
}


def _sin_autoetiqueta(texto: str, etiqueta: str) -> str:
    """Quita las líneas en que el MODELO imita el origen de autoridad que el
    SISTEMA agrega después (_append_authority, Decisión 3). La cabecera sale de
    la etiqueta configurada (config.toml, `authority_origin`): lo que está
    antes del primer ':' -- p. ej., para kimi, el prefijo con el emoji de
    engranaje seguido de "Origen de autoridad". Antes era un literal de kimi
    en dos copias y las demás facetas no se limpiaban (E-15)."""
    cabecera = etiqueta.split(":", 1)[0].strip() if etiqueta else ""
    if not cabecera:
        return texto.strip()
    lineas = [l for l in texto.splitlines() if not l.strip().startswith(cabecera)]
    return "\n".join(lineas).strip()


# --- Politica de grounding (Decision 1: por TAREA, no por faceta) ------------
#   off                → no buscar. Tarea local o creativa.
#   auto               → puede buscar; si NO buscó, el sistema declara "no verificado".
#   required_web       → DEBE buscar; sin groundingChunks -> MuscleInvocationError.
#   local_context_only → solo el texto provisto; declara que se basa en el input.
GROUNDING_POLICIES = ("off", "auto", "required_web", "local_context_only")


def verificacion_label(estado: str, n_fuentes: int = 0) -> str:
    """Sello de verificacion (Decision 4: NUNCA una respuesta sin etiqueta).
    Posicion constante (al final), peso visual variable: sutil en exito,
    visible en fallo. El SISTEMA lo impone, no el modelo (Decision 3)."""
    if estado == "web":
        return f"🔍 *Verificación: búsqueda web completada ({n_fuentes} fuentes)*"
    if estado == "internal":
        return "🧠 *Verificación: conocimiento interno del modelo (no verificado en web)*"
    if estado == "local":
        return "📜 *Verificación: basado únicamente en el texto proporcionado*"
    if estado == "failed":
        return "✗ *Verificación: búsqueda web requerida pero fallida — tarea abortada*"
    return ""


class MuscleError(Exception):
    """Base de errores de musculos."""


class ModelNotAllowedError(MuscleError, ValueError):
    """Se pidio un modelo fuera de models_allowed. Fallo duro, sin fallback."""


class MuscleTimeoutError(MuscleError):
    """El musculo no respondio dentro del timeout."""


class MuscleInvocationError(MuscleError):
    """El musculo respondio error (HTTP != 2xx, salida no parseable, etc.)."""


class DispatchConfigMuscleError(MuscleInvocationError, ModelDispatchConfigError):
    """El catalogo no declara el limite de salida del modelo a despachar
    (PR-K): no sale ningun request. Es las dos cosas a proposito -- el REPL
    lo atrapa como MuscleError, y humanizar_error lo reconoce como
    ModelDispatchConfigError para mostrar el UPDATE entero."""


class Muscle(ABC):
    # PR-K ronda 2 (I1): motivo por el que esta faceta NO puede despachar
    # (su binding no coincide con el camino que arma config.toml). Lo pone
    # build_muscles desde registro_facetas.aplicar_registro. Vacío = despacha.
    dispatch_bloqueado: str = ""

    def __init__(
        self,
        name: str,
        model_default: str,
        models_allowed: list[str],
        system_prompt: str,
        timeout: float,
        authority_origin: str = "",
    ) -> None:
        self.name = name
        self.model_default = model_default
        self.models_allowed = models_allowed
        self.system_prompt = system_prompt
        self.timeout = timeout
        # Origen de autoridad (Decision 7): sello que el SISTEMA agrega al final
        # de la respuesta para que ninguna faceta aparente autoridad sin rendir
        # cuentas. Vacio = sin sello (p.ej. jax_local, el ser conversacional).
        self.authority_origin = authority_origin

    def _append_authority(self, text: str) -> str:
        """Agrega el origen de autoridad al final. Lo impone el sistema, no el
        modelo (Decision 3 y 4). Subclases que ya etiquetan (gemini) lo omiten."""
        if self.authority_origin:
            return f"{text}\n\n{self.authority_origin}"
        return text

    def _resolve_model(self, model: str | None) -> str:
        chosen = model or self.model_default
        if chosen not in self.models_allowed:
            raise ModelNotAllowedError(
                f"[{self.name}] modelo '{chosen}' no permitido. "
                f"Permitidos: {self.models_allowed}"
            )
        return chosen

    async def invoke(
        self,
        prompt: str,
        model: str | None = None,
        history: list[dict] | None = None,
        decorate: bool = True,
    ) -> str:
        """decorate=True: respuesta para Fernando -> lleva su etiqueta de origen.
        decorate=False: uso interno (p.ej. el clasificador del router) -> salida
        cruda, sin sello, para no contaminar el parseo."""
        if self.dispatch_bloqueado:
            raise DispatchConfigMuscleError(f"[{self.name}] dispatch abortado: {self.dispatch_bloqueado}")
        chosen = self._resolve_model(model)
        try:
            resultado = await asyncio.wait_for(
                self._call(prompt, chosen, history), timeout=self.timeout
            )
        except asyncio.TimeoutError as exc:
            raise MuscleTimeoutError(
                f"[{self.name}] sin respuesta en {self.timeout}s"
            ) from exc
        # Ninguna respuesta sale sin su etiqueta de origen (Decision 4 y 7),
        # salvo los usos internos que piden la salida cruda (decorate=False).
        if not decorate:
            return resultado
        return self._append_authority(resultado)

    @abstractmethod
    async def _call(
        self, prompt: str, model: str, history: list[dict] | None = None
    ) -> str:
        raise NotImplementedError


class HttpMuscle(Muscle):
    def __init__(
        self,
        name: str,
        provider: str,
        model_default: str,
        models_allowed: list[str],
        system_prompt: str,
        timeout: float,
        grounding_policy: str = "off",
        authority_origin: str = "",
        api_url: str = "",
    ) -> None:
        super().__init__(
            name, model_default, models_allowed, system_prompt, timeout,
            authority_origin=authority_origin,
        )
        self.provider = provider
        self.api_url = api_url  # override de URL para proveedores OpenAI-compatibles
        if grounding_policy not in GROUNDING_POLICIES:
            raise MuscleInvocationError(
                f"[{name}] grounding_policy '{grounding_policy}' invalido. "
                f"Validos: {GROUNDING_POLICIES}"
            )
        self.grounding_policy = grounding_policy

        if provider not in _PROVIDER_ID_MAP:
            raise MuscleInvocationError(f"[{name}] proveedor desconocido: {provider}")
        # La api_key NO se resuelve aca: __init__ corre una sola vez al
        # construir el muscle (vida del proceso). Resolverla aca seria
        # exactamente el cache de vida de proceso prohibido en el diseño
        # de Fase 1 (B1.2a) — se resuelve por-request en _resolve_api_key().

    async def _resolve_api_key(self) -> str:
        provider_id = _PROVIDER_ID_MAP[self.provider]
        try:
            return await resolve_credential_instrumented(provider_id)
        except CredentialUnavailableError as e:
            raise MuscleInvocationError(
                f"[{self.name}] sin credencial válida configurada para {provider_id}"
            ) from e

    async def _limite_de_salida(self, model: str) -> dict[str, int]:
        """{nombre: tope} del limite de salida, de la fila de `model` del
        modelo QUE SE DESPACHA (puede no ser el asignado: modo pesado). Antes
        era "max_tokens": 131072 fijo -- el literal que tumbo a thot en la
        Mesa web (2026-08-24). Ver jax/core/contrato_dispatch.py."""
        try:
            return await limite_de_salida("http_openai_compat", _PROVIDER_ID_MAP[self.provider], model)
        except ModelDispatchConfigError as e:
            raise DispatchConfigMuscleError(f"[{self.name}] dispatch abortado: {e}") from e

    def _url_del_catalogo(self) -> str:
        """E-21 (2026-09-16): la URL del proveedor sale SOLO del catálogo
        (provider.base_url, puesta en api_url por registro_facetas al arrancar).
        Antes había URLs de OpenAI/DeepSeek/Gemini como default: con la DB
        caída se despachaba a una URL que nadie eligió."""
        if not self.api_url:
            raise MuscleInvocationError(
                f"[{self.name}] sin URL del proveedor: sale del catálogo (provider.base_url) "
                f"al arrancar y el catálogo no la dio; no se despacha a una URL fija."
            )
        return self.api_url

    def _append_authority(self, text: str) -> str:
        # Gemini ya inserta su etiqueta de verificacion (dinamica, segun la
        # politica de grounding) dentro de _call_gemini. No la duplicamos.
        # DeepSeek/OpenAI no buscan en web: usan su origen de autoridad estatico.
        if self.provider == "gemini":
            return text
        return super()._append_authority(text)

    async def _call(
        self, prompt: str, model: str, history: list[dict] | None = None
    ) -> str:
        if self.provider == "deepseek":
            return await self._call_deepseek(prompt, model, history)
        if self.provider in ("openai", "kimi", "zhipu", "zai"):
            return await self._call_openai(prompt, model, history)
        return await self._call_gemini(prompt, model, history)

    async def _call_deepseek(
        self, prompt: str, model: str, history: list[dict] | None = None
    ) -> str:
        url = self._url_del_catalogo()
        api_key = await self._resolve_api_key()
        headers = {"Authorization": f"Bearer {api_key}"}

        # messages = system + historial previo + mensaje actual.
        # El historial ya viene en formato {"role": "user"|"assistant", ...},
        # que es exactamente lo que DeepSeek espera. Sin duplicar el actual.
        messages = [{"role": "system", "content": self.system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            **await self._limite_de_salida(model),
        }
        resp = await obtener_cliente_http().post(url, headers=headers, json=payload, timeout=self.timeout)
        if resp.status_code != 200:
            raise MuscleInvocationError(
                f"[{self.name}] DeepSeek HTTP {resp.status_code}: {recortar_redactado(resp.text, 200, [api_key])}"
            )
        data = resp.json()
        msg = data["choices"][0]["message"]
        texto = msg.get("content") or ""

        # D1.2 — best-effort, fuera del try/response: nunca debe poder
        # romper la respuesta al usuario (record_resolved_version_safe ya
        # atrapa sus propias excepciones).
        await record_resolved_version_safe(self.name, data.get("model"))
        # Kimi K2.7 trae reasoning_content aparte: no se usa. La autoetiqueta
        # que el modelo imite se quita con la cabecera configurada (E-15).
        return _sin_autoetiqueta(texto, self.authority_origin)


    async def _call_openai(
        self, prompt: str, model: str, history: list[dict] | None = None
    ) -> str:
        url = self._url_del_catalogo()
        api_key = await self._resolve_api_key()
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        messages = [{"role": "system", "content": self.system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            **await self._limite_de_salida(model),
        }
        texto = ""
        resolved_version = None
        async with obtener_cliente_http().stream("POST", url, headers=headers, json=payload, timeout=self.timeout) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                cuerpo = recortar_redactado(body.decode("utf-8", errors="replace"), 200, [api_key])
                raise MuscleInvocationError(
                    f"[{self.name}] OpenAI HTTP {resp.status_code}: {cuerpo}"
                )
            partes = []
            async for linea in resp.aiter_lines():
                if not linea or not linea.startswith("data:"):
                    continue
                payload_str = linea[5:].strip()      # quita "data:"
                if payload_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload_str)
                except json.JSONDecodeError:
                    continue
                # D1.2 — cada chunk trae 'model' (el resuelto, no el
                # alias pedido); alcanza con el primero, es constante
                # durante todo el stream.
                if resolved_version is None:
                    resolved_version = chunk.get("model")
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                pieza = delta.get("content")
                if pieza:
                    partes.append(pieza)
            texto = "".join(partes)

        await record_resolved_version_safe(self.name, resolved_version)

        # Kimi K2.7 trae reasoning_content aparte: no se usa. La autoetiqueta
        # que el modelo imite se quita con la cabecera configurada (E-15).
        return _sin_autoetiqueta(texto, self.authority_origin)

    @staticmethod
    def _extract_gemini(data: dict) -> tuple[str, list, list, list]:
        """Devuelve (texto, chunks, supports, queries) de una respuesta Gemini.
        Decision 5: validar chunks Y supports (presencia != garantia total)."""
        candidate = data["candidates"][0]
        parts = candidate.get("content", {}).get("parts", []) or []
        texto = "".join(p.get("text", "") for p in parts)
        meta = candidate.get("groundingMetadata", {}) or {}
        chunks = meta.get("groundingChunks") or []
        supports = meta.get("groundingSupports") or []
        queries = meta.get("webSearchQueries") or []
        return texto, chunks, supports, queries

    @staticmethod
    async def _format_sources(chunks: list, supports: list, queries: list) -> tuple[str, int]:
        """Bloque de fuentes verificables + conteo de fuentes únicas.
        Las fuentes son la prueba del research. Desde 2026-09-12 cada una lleva
        la URL FINAL (se sigue la redirección opaca de Google grounding) y los
        fragmentos de la respuesta que respalda -- el mismo módulo que Jacobs
        (jax/core/grounding_sources.py). Antes eran redirecciones con una
        etiqueta de dominio: nadie podía contrastarlas."""
        sources = build_sources(chunks, supports)
        await resolve_redirects(sources)
        bloque = ""
        if sources:
            bloque += "\n\n— Fuentes consultadas —\n" + render_sources_block(sources)
        if queries:
            bloque += "\n\n(Búsquedas: " + "; ".join(queries) + ")"
        return bloque, len(sources)

    async def _call_gemini(
        self, prompt: str, model: str, history: list[dict] | None = None
    ) -> str:
        api_key = await self._resolve_api_key()
        # PR-K ronda 2 (I1): la URL base sale del proveedor del modelo en el
        # catálogo (registro_facetas.aplicar_registro la pone en api_url).
        base = self._url_del_catalogo()
        # Ruling T6-6 (2026-09-15): la key va en la cabecera x-goog-api-key,
        # NO en `?key=` (httpx loguea la URL entera en INFO y la mete en
        # str(HTTPStatusError)).
        url = f"{base.rstrip('/')}/models/{model}:generateContent"
        headers = {"x-goog-api-key": api_key}

        # Gemini usa "contents" con role "user"/"model" (no "assistant") y
        # cada texto envuelto en parts. Convertimos el historial neutro.
        contents: list[dict] = []
        if history:
            for m in history:
                g_role = "model" if m["role"] == "assistant" else "user"
                contents.append({"role": g_role, "parts": [{"text": m["content"]}]})
        contents.append({"role": "user", "parts": [{"text": prompt}]})

        policy = self.grounding_policy
        # off / local_context_only NO envian tools. auto / required_web si.
        usar_tools = policy in ("auto", "required_web")

        async def _request(instruccion_extra: str = "") -> dict:
            sys_text = self.system_prompt
            if instruccion_extra:
                sys_text = self.system_prompt + "\n\n" + instruccion_extra
            payload: dict = {
                "system_instruction": {"parts": [{"text": sys_text}]},
                "contents": contents,
            }
            if usar_tools:
                # google_search es una capacidad, no una funcion declarada:
                # tool_config con mode ANY NO aplica aqui (es para
                # functionDeclarations). El retry estricto (Decision 6) es el
                # mecanismo real para forzar la busqueda.
                payload["tools"] = [{"google_search": {}}]
            resp = await obtener_cliente_http().post(url, headers=headers, json=payload, timeout=self.timeout)
            if resp.status_code != 200:
                # Google devuelve la key rechazada DENTRO del cuerpo del
                # error: redactar antes de recortar.
                cuerpo = recortar_redactado(resp.text, 200, [api_key])
                raise MuscleInvocationError(
                    f"[{self.name}] Gemini HTTP {resp.status_code}: {cuerpo}"
                )
            return resp.json()

        # Intento 1.
        data = await _request()
        texto, chunks, supports, queries = self._extract_gemini(data)
        resolved_version = data.get("modelVersion")

        # Decision 6: required_web -> UN retry estricto antes de fallar cerrado.
        if policy == "required_web" and not chunks:
            data = await _request(
                "Debes usar búsqueda web (google_search) para responder esta "
                "consulta. Si por cualquier razón no puedes buscar, responde "
                "EXACTAMENTE la palabra: NO_VERIFICADO"
            )
            texto, chunks, supports, queries = self._extract_gemini(data)
            resolved_version = data.get("modelVersion") or resolved_version
            if not chunks or texto.strip() == "NO_VERIFICADO":
                # Fallo cerrado: jamas entregar datos sin verificar disfrazados
                # de verificados. Un 'no se' honesto es mejor que inventar.
                raise MuscleInvocationError(
                    f"[{self.name}] required_web: Gemini no realizó búsqueda web "
                    f"(sin groundingChunks tras retry estricto). Respuesta abortada "
                    f"para no entregar datos sin verificar. "
                    f"{verificacion_label('failed')}"
                )

        # D1.2 — best-effort; 'modelVersion' es el campo real de Gemini
        # (distinto de 'model' que usan las APIs OpenAI-compatible — ver
        # nota de incertidumbre en CONTEXT.md: heredado de jax-platform,
        # nunca verificado contra una respuesta real de Gemini con curl).
        await record_resolved_version_safe(self.name, resolved_version)

        # Decision 3 y 4: el SISTEMA decide la etiqueta y SIEMPRE la pone.
        if policy == "local_context_only":
            return texto + "\n\n" + verificacion_label("local")

        if chunks:  # hubo busqueda real (auto o required_web)
            bloque, n = await self._format_sources(chunks, supports, queries)
            etiqueta = verificacion_label("web", n_fuentes=n)
            if not supports:
                etiqueta += "\n⚠ *Advertencia: sin groundingSupports — citas sin anclaje posicional.*"
            return texto + bloque + "\n\n" + etiqueta

        # off, o auto sin busqueda: conocimiento interno, declarado como tal.
        return texto + "\n\n" + verificacion_label("internal")
