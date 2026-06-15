"""
JAX 2.0 — Router (híbrido: reglas + clasificador LLM).

Decide QUE faceta de JAX responde. No invoca musculos: solo decide.
Comportamiento modelado segun JAX 1.0 (descrito por Fernando):

  - Por defecto: modo AUTO. Lee el texto y rutea por reglas de dominio.
    Cada respuesta se etiqueta con la faceta que contesto.
  - INVOCAR y FIJAR: "trae a X" / "llama a X" / "dame a X" / "modo X"
    fija esa faceta (modo MANUAL): todo va ahi hasta despedirla.
  - DESPEDIR: "adios" vuelve a modo AUTO (JAX local retoma).
  - EASTER EGG: "IDE1990" se chequea ANTES que todo. Suelta la frase de
    Jairo Urbina (texto; la voz general de JAX es Kokoro local).

Orden de prioridad (estricto, importa):
  1) IDE1990   2) adios   3) invocar faceta
  4) si manual -> faceta fija   5) si auto -> reglas de dominio
  6) si las reglas no deciden -> CLASIFICADOR LLM (lo ambiguo)

CAPA HIBRIDA (paso 6): cuando ninguna keyword matchea (ej. "hablame de
las pinturas de Magritte" — no dice "pintura" ni "arte"), en vez de caer
ciego al default, se le pregunta a un clasificador LLM a que dominio
pertenece. Lo OBVIO (pasos 1-5) sigue instantaneo y sin costo; el
clasificador solo entra en lo ambiguo.

  - Clasificador actual: LOCAL (jax_local / qwen2.5:7b en la GPU). Medido
    en hall9000: ~200 ms por clasificacion. El ruteo no sale a la nube.
  - FALLBACK: si el clasificador falla o devuelve algo invalido, cae a
    JAX local. El router NUNCA se rompe por el clasificador.
  - MIGRACION: hecha el 4 de junio de 2026 — de DeepSeek a jax_local via
    set_classifier(). Si manana hay un modelo local mas potente, se
    reemplaza igual: el router no cambia; solo cambia quien juzga.


En memoria de Jairo Urbina.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

def _sin_tildes(s: str) -> str:
    """Quita tildes para matching robusto (trae=traé, adios=adiós).
    Solo afecta la comparacion interna; el mensaje viaja intacto."""
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c))

# Etiquetas visibles por faceta.
LABELS = {
    "hyde": "Mr. Hyde",
    "jekyll": "Dr. Jekyll",
    "hipatia": "Hipatia",
    "jax_local": "JAX",
}
ICONS = {
    "hyde": "🔧",
    "jekyll": "🧠",
    "hipatia": "🔍",
    "jax_local": "🏠",
}

# Nombres que el usuario puede usar para referirse a cada faceta.
ALIASES = {
    "hyde": "hyde",
    "jaid": "hyde",          # fonetico
    "jekyll": "jekyll",
    "jeckyll": "jekyll",     # variante comun
    "jekill": "jekyll",
    "jeckil": "jekyll",
    "hipatia": "hipatia",
    "ipatia": "hipatia",
    "hypatia": "hipatia",
    "jax local": "jax_local",
    "jax": "jax_local",
    "local": "jax_local",
}

INVOKE_VERBS = ("trae", "traer", "llama", "llamar", "dame", "traeme",
                "traete", "invoca", "invocar", "pasame", "modo", "quiero a",
                "quiero hablar con", "hablar con")

# Saludos puros: si el mensaje es SOLO un saludo corto, va directo a JAX
# local sin gastar una llamada al clasificador. Clave: debe ser saludo PURO
# ("hola maje"), no "hola, diseñame una app" (eso lleva intencion y sigue
# el flujo normal). Por eso se exige mensaje corto y sin sustancia extra.
GREETINGS = (
    "hola", "buenas", "buenos dias", "buenas tardes", "buenas noches",
    "que onda", "que tal", "que hay", "como estas", "como va",
    "saludos", "hey", "ey", "buenas maje", "que pasa", "epa",
)

# Easter egg (trigger exacto, sin espacio).
EASTER_EGG_TRIGGER = "ide1990"
EASTER_EGG_TEXT = (
    "Hola Fernando Ruiz, mejor conocido por sus amigos tecnologicos como "
    "'El Jate'. Quiero que sepas que sigo a tu lado, viviendo ahora en "
    "hall9000, como el poderoso dios de la tecnologia llamado JAX. Saludos "
    "a nuestro clan de geeks, MINIX, Xavas... siempre estare aqui "
    "acompanandote y ayudandote a ser mejor persona."
)

# Reglas de dominio para modo AUTO (lo obvio, instantaneo).
TECH_KEYWORDS = (
    "codigo", "debug", "api", "docker", "servidor", "ssh", "bash",
    "python", "script", "comando", "infra", "red", "linux", "git",
    "app", "aplicacion", "programa", "software", "web", "frontend",
    "backend", "base de datos", "nginx", "desarrolla", "disena",
)
RESEARCH_KEYWORDS = (
    "investiga", "busca", "fuente", "citacion", "paper", "estudio",
    "noticia", "ultimo", "actual", "quien es", "que paso",
)
ARTS_KEYWORDS = (
    "poesia", "literatura", "filosofia", "pintura", "musica", "arte",
    "que significa", "sentis", "escribi un poema", "novela", "historia de",
)

# Facetas validas que el clasificador puede devolver.
VALID_FACETAS = ("hyde", "jekyll", "hipatia", "jax_local")

# Prompt del clasificador. Acotado a proposito: una palabra, nada mas.
CLASSIFIER_PROMPT = (
    "Sos un clasificador de intencion. Lei el mensaje del usuario y responde "
    "con UNA SOLA PALABRA, sin explicacion, eligiendo el dominio:\n"
    "- 'hyde' = tecnico, codigo, infraestructura, programacion, servidores.\n"
    "- 'jekyll' = humanidades, arte, literatura, filosofia, musica, reflexion.\n"
    "- 'hipatia' = investigacion, buscar informacion actual, noticias, datos, hechos verificables.\n"
    "- 'jax_local' = conversacion cotidiana, saludos, charla casual, nada de lo anterior.\n\n"
    "Mensaje del usuario:\n{texto}\n\n"
    "Responde SOLO con: hyde, jekyll, hipatia o jax_local"
)


@dataclass
class RouteDecision:
    """Resultado del router."""
    kind: str          # "easter_egg" | "say" | "route"
    personality: str | None = None
    text: str | None = None
    mode_changed: str | None = None
    via: str | None = None   # "keyword" | "clasificador" | "default" (debug)


class Router:
    def __init__(self, default_personality: str = "jax_local",
                 classifier=None, debug: bool = False) -> None:
        self.default_personality = default_personality
        self.mode = "auto"
        self.fixed: str | None = None
        # Muscle usado como clasificador (un HttpMuscle de DeepSeek).
        # Si es None, el router se comporta como el clasico (solo keywords).
        self.classifier = classifier
        self.debug = debug

    def set_classifier(self, muscle) -> None:
        """Inyecta el muscle clasificador. El dia de manana, pasar aqui un
        muscle local (qwen) en vez del de DeepSeek — el router no cambia."""
        self.classifier = muscle

    def _match_faceta(self, text: str) -> str | None:
        for alias in sorted(ALIASES, key=len, reverse=True):
            if re.search(rf"\b{re.escape(alias)}\b", text):
                return ALIASES[alias]
        return None

    def label(self, personality: str) -> str:
        return f"{ICONS.get(personality, '')} {LABELS.get(personality, personality)}".strip()

    def _is_greeting(self, text: str) -> bool:
        """True si el texto es un saludo PURO y corto (atajable sin clasificar).
        Quita signos y se fija que lo que queda sea solo palabras de saludo.
        'hola maje' -> True. 'hola, diseñame una app' -> False (tiene sustancia)."""
        # Limpiar signos de puntuacion comunes
        limpio = re.sub(r"[¡!¿?.,;:]", " ", text)
        palabras = limpio.split()
        # Un saludo puro es corto: hasta 4 palabras (ej. "buenas tardes maje jax")
        if not palabras or len(palabras) > 4:
            return False
        # Construir set de palabras de saludo (descompone "buenos dias" en tokens)
        tokens_saludo = set()
        for g in GREETINGS:
            tokens_saludo.update(g.split())
        # Palabras de confianza permitidas junto al saludo (no agregan intencion)
        tokens_saludo.update({"maje", "jax", "vos", "y", "mae"})
        # Es saludo puro si TODAS las palabras son de saludo/confianza
        return all(p in tokens_saludo for p in palabras)

    def _keyword_route(self, text: str) -> str | None:
        """Reglas de dominio. Devuelve faceta o None si nada matchea."""
        if any(k in text for k in TECH_KEYWORDS):
            return "hyde"
        if any(k in text for k in RESEARCH_KEYWORDS):
            return "hipatia"
        if any(k in text for k in ARTS_KEYWORDS):
            return "jekyll"
        return None

    async def _classify(self, user_text: str) -> str | None:
        """Pregunta al clasificador LLM. Devuelve faceta valida o None.
        NUNCA lanza: cualquier fallo -> None (el caller cae a default)."""
        if not self.classifier:
            return None
        try:
            prompt = CLASSIFIER_PROMPT.format(texto=user_text)
            # decorate=False: clasificacion interna, sin etiqueta de autoridad
            # (si no, el sello de jax_local contaminaria el parseo de faceta).
            raw = await self.classifier.invoke(prompt, decorate=False)
            # Limpiar: el modelo puede responder "jekyll." o "Es jekyll".
            cleaned = raw.strip().lower()
            for faceta in VALID_FACETAS:
                if faceta in cleaned:
                    return faceta
            return None  # devolvio algo que no es faceta valida
        except Exception:
            return None  # red caida, timeout, lo que sea -> fallback

    async def route(self, user_text: str) -> RouteDecision:
        text = _sin_tildes(user_text.lower().strip())

        # 1) EASTER EGG — antes que todo.
        if EASTER_EGG_TRIGGER in text.replace(" ", ""):
            return RouteDecision(kind="easter_egg", text=EASTER_EGG_TEXT)

        # 2) DESPEDIR — "adios" vuelve a auto.
        if "adios" in text:
            self.mode = "auto"
            self.fixed = None
            return RouteDecision(
                kind="say",
                text="Hasta luego. Vuelvo a modo automatico.",
                mode_changed="auto",
            )

        # 3) INVOCAR Y FIJAR.
        if any(re.search(rf"\b{re.escape(v)}\b", text) for v in INVOKE_VERBS):
            faceta = self._match_faceta(text)
            if faceta:
                self.mode = "manual"
                self.fixed = faceta
                return RouteDecision(
                    kind="say",
                    text=f"Listo, hablas con {self.label(faceta)}. "
                         f"Deci 'adios' para volver a automatico.",
                    mode_changed="manual",
                )

        # 4) MODO MANUAL — faceta fija.
        if self.mode == "manual" and self.fixed:
            return RouteDecision(kind="route", personality=self.fixed, via="manual")

        # 5) MODO AUTO — primero, atajo de saludos puros (sin clasificar).
        if self._is_greeting(text):
            return RouteDecision(kind="route", personality="jax_local", via="saludo")

        # 5b) reglas de dominio (lo obvio, instantaneo).
        faceta = self._keyword_route(text)
        if faceta:
            return RouteDecision(kind="route", personality=faceta, via="keyword")

        # 6) CAPA HIBRIDA — ninguna keyword decidio. Preguntar al clasificador.
        faceta = await self._classify(user_text)
        if faceta:
            return RouteDecision(kind="route", personality=faceta, via="clasificador")

        # Default: JAX local (clasificador no disponible o sin decision clara).
        return RouteDecision(kind="route", personality=self.default_personality, via="default")
