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
import os
from abc import ABC, abstractmethod

import httpx
import json

from jax.core.crypto_secrets import decrypt_secret


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


class Muscle(ABC):
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

        if provider == "deepseek":
            self.api_key = decrypt_secret(os.environ["DEEPSEEK_API_KEY"])
        elif provider == "gemini":
            self.api_key = decrypt_secret(os.environ["GEMINI_API_KEY"])
        elif provider == "openai":
            self.api_key = decrypt_secret(os.environ["OPENAI_API_KEY"])
        elif provider == "kimi":
            self.api_key = decrypt_secret(os.environ["KIMI_API_KEY"])
        elif provider == "zhipu":
            self.api_key = decrypt_secret(os.environ.get("ZHIPU_API_KEY", ""))
        elif provider == "zai":
            self.api_key = decrypt_secret(os.environ.get("ZAI_API_KEY", ""))
        else:
            raise MuscleInvocationError(f"[{name}] proveedor desconocido: {provider}")

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
        url = "https://api.deepseek.com/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}

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
            "max_tokens": 131072,
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url, headers=headers, json=payload)
            if resp.status_code != 200:
                raise MuscleInvocationError(
                    f"[{self.name}] DeepSeek HTTP {resp.status_code}: {resp.text[:200]}"
                )
            data = resp.json()
            msg = data["choices"][0]["message"]
            texto = msg.get("content") or ""
            # Kimi K2.7 incluye reasoning_content separado — ignorarlo.
            # Limpiar auto-etiquetas que el modelo genere dentro del content.
            lineas = [l for l in texto.splitlines()
                      if not l.strip().startswith("⚙️ *Origen")]
            return "\n".join(lineas).strip()


    async def _call_openai(
        self, prompt: str, model: str, history: list[dict] | None = None
    ) -> str:
        url = self.api_url if self.api_url else "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
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
            "max_tokens": 131072,
        }
        texto = ""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream("POST", url, headers=headers, json=payload) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    raise MuscleInvocationError(
                        f"[{self.name}] OpenAI HTTP {resp.status_code}: {body[:200]!r}"
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
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    pieza = delta.get("content")
                    if pieza:
                        partes.append(pieza)
                texto = "".join(partes)

        # Kimi K2.7 incluye reasoning_content separado — ignorarlo (no llega en delta).
        # Limpiar auto-etiquetas que el modelo genere dentro del content.
        lineas = [l for l in texto.splitlines()
                  if not l.strip().startswith("⚙️ *Origen")]
        return "\n".join(lineas).strip()

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
    def _format_sources(chunks: list, queries: list) -> tuple[str, int]:
        """Bloque de fuentes (sin duplicar) + conteo de fuentes unicas.
        Sin esto, Hipatia 'busca' pero las fuentes se pierden y parece que
        respondio de memoria. Las fuentes son la prueba del research."""
        vistos: set = set()
        unicas: list = []
        for ch in chunks:
            web = ch.get("web") or {}
            uri = web.get("uri")
            if uri and uri not in vistos:
                vistos.add(uri)
                unicas.append((web.get("title", uri), uri))
        bloque = ""
        if unicas:
            bloque += "\n\n— Fuentes consultadas —"
            for i, (t, u) in enumerate(unicas, 1):
                bloque += f"\n  [{i}] {t}: {u}"
        if queries:
            bloque += "\n\n(Búsquedas: " + "; ".join(queries) + ")"
        return bloque, len(unicas)

    async def _call_gemini(
        self, prompt: str, model: str, history: list[dict] | None = None
    ) -> str:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/"
            f"models/{model}:generateContent?key={self.api_key}"
        )

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
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(url, json=payload)
                if resp.status_code != 200:
                    raise MuscleInvocationError(
                        f"[{self.name}] Gemini HTTP {resp.status_code}: {resp.text[:200]}"
                    )
                return resp.json()

        # Intento 1.
        texto, chunks, supports, queries = self._extract_gemini(await _request())

        # Decision 6: required_web -> UN retry estricto antes de fallar cerrado.
        if policy == "required_web" and not chunks:
            texto, chunks, supports, queries = self._extract_gemini(
                await _request(
                    "Debes usar búsqueda web (google_search) para responder esta "
                    "consulta. Si por cualquier razón no puedes buscar, responde "
                    "EXACTAMENTE la palabra: NO_VERIFICADO"
                )
            )
            if not chunks or texto.strip() == "NO_VERIFICADO":
                # Fallo cerrado: jamas entregar datos sin verificar disfrazados
                # de verificados. Un 'no se' honesto es mejor que inventar.
                raise MuscleInvocationError(
                    f"[{self.name}] required_web: Gemini no realizó búsqueda web "
                    f"(sin groundingChunks tras retry estricto). Respuesta abortada "
                    f"para no entregar datos sin verificar. "
                    f"{verificacion_label('failed')}"
                )

        # Decision 3 y 4: el SISTEMA decide la etiqueta y SIEMPRE la pone.
        if policy == "local_context_only":
            return texto + "\n\n" + verificacion_label("local")

        if chunks:  # hubo busqueda real (auto o required_web)
            bloque, n = self._format_sources(chunks, queries)
            etiqueta = verificacion_label("web", n_fuentes=n)
            if not supports:
                etiqueta += "\n⚠ *Advertencia: sin groundingSupports — citas sin anclaje posicional.*"
            return texto + bloque + "\n\n" + etiqueta

        # off, o auto sin busqueda: conocimiento interno, declarado como tal.
        return texto + "\n\n" + verificacion_label("internal")
