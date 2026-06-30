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
    "thot": "Thot",
    "kimi": "Kimi",
    "ada": "Ada",
}
ICONS = {
    "hyde": "🔧",
    "jekyll": "🧠",
    "hipatia": "🔍",
    "jax_local": "🏠",
    "thot": "📜",
    "kimi": "⚙️",
    "ada": "⚛️",
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
    "thot": "thot",
    "kimi": "kimi",
    "ada": "ada",
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

# Reglas de dominio para modo AUTO — scoring multi-faceta.
# Hyde NO es destino del auto-routing: es ejecutor, no conversador.

KIMI_KW = frozenset((
    "codigo", "programar", "programa", "script", "funcion", "clase", "metodo",
    "modulo", "libreria", "api", "endpoint", "backend", "frontend",
    "implementar", "implementa", "construir", "refactor", "refactorizar",
    "refactoriza", "debug", "depurar", "bug", "traceback", "excepcion",
    "compilar", "test", "tests", "pytest", "variable", "bucle", "array",
    "regex", "fastapi", "react", "typescript", "javascript", "python", "sql",
    "docker", "nginx", "commit", "branch", "merge",
))
KIMI_STRONG = frozenset((
    "refactor", "refactoriza", "implementar", "debug", "depurar", "pytest",
    "fastapi", "docker", "nginx", "endpoint",
))

HIPATIA_KW = frozenset((
    "busca", "buscar", "investiga", "investigar", "verifica", "verificar",
    "fuentes", "fuente", "citas", "referencias", "noticias", "noticia",
    "actualidad", "reciente", "ultima", "ultimo", "vigente", "precio",
    "precios", "cotizacion", "mercado", "ley", "regulacion", "normativa",
    "paper", "papers", "estudio", "informe", "estadistica", "lanzamiento",
    "version actual", "quien es",
))
HIPATIA_STRONG = frozenset((
    "busca", "buscar", "investiga", "investigar", "noticias", "fuentes",
    "version actual",
))

JEKYLL_KW = frozenset((
    "poesia", "poema", "cuento", "novela", "literatura", "ensayo", "arte",
    "pintura", "musica", "filosofia", "etica", "estetica", "humanidades",
    "barroco", "renacimiento", "romanticismo", "mito", "mitologia", "simbolo",
    "simbolismo", "metafora", "narrativa", "personaje", "estilo",
    "interpretacion", "sentido", "significado", "reflexion", "reflexiona",
    "contempla", "humanista", "cultura", "historia del arte", "historia cultural",
))
JEKYLL_STRONG = frozenset((
    "poema", "poesia", "filosofia", "literatura", "mitologia",
    "historia del arte", "barroco",
))

THOT_KW = frozenset((
    "audita", "auditar", "auditoria", "critica", "criticar", "criticamente",
    "cuestiona", "cuestionar", "adversarial", "abogado del diablo", "riesgo",
    "riesgos", "falla", "fallas", "debilidad", "debilidades", "vulnerabilidad",
    "vulnerabilidades", "amenaza", "amenazas", "threat model",
    "modelo de amenazas", "ataque", "donde se rompe", "punto ciego",
    "supuesto", "supuestos", "contraargumento", "refuta", "refutar",
    "no-go", "revisa criticamente",
))
THOT_STRONG = frozenset((
    "audita", "auditar", "auditoria", "vulnerabilidad", "vulnerabilidades",
    "threat model", "adversarial", "refuta",
))

ADA_KW = frozenset((
    "formaliza", "formalizar", "formalizacion", "modelo formal", "pseudocodigo",
    "logica", "demuestra", "demostrar", "demostracion", "prueba formal",
    "teorema", "lema", "corolario", "axioma", "proposicion", "invariante",
    "invariantes", "precondicion", "postcondicion", "maquina de estados",
    "automata", "complejidad", "big o", "o(n)", "estructura de datos",
    "grafo", "arbol", "matriz", "vector", "ecuacion", "optimizacion",
    "funcion objetivo", "matematica", "calculo", "algebra", "probabilidad",
    "determinista", "induccion", "algoritmo",
))
ADA_STRONG = frozenset((
    "formaliza", "formalizar", "demuestra", "demostrar", "teorema",
    "invariante", "invariantes", "precondicion", "postcondicion",
    "complejidad", "maquina de estados",
))

# Facetas validas para invocacion explicita (incluye hyde).
VALID_FACETAS = ("hyde", "jekyll", "hipatia", "jax_local", "thot", "kimi", "ada")

# Facetas del auto-routing (hyde excluido: es ejecutor, no conversador).
AUTO_FACETAS = ("jax_local", "kimi", "hipatia", "jekyll", "thot", "ada")

# Prioridad de desempate en scoring (izquierda gana sobre derecha).
_TIEBREAK = ("hipatia", "thot", "ada", "kimi", "jekyll")

# Mapa faceta → (conjunto_completo, conjunto_strong)
_KW_SETS = {
    "kimi":    (KIMI_KW,    KIMI_STRONG),
    "hipatia": (HIPATIA_KW, HIPATIA_STRONG),
    "jekyll":  (JEKYLL_KW,  JEKYLL_STRONG),
    "thot":    (THOT_KW,    THOT_STRONG),
    "ada":     (ADA_KW,     ADA_STRONG),
}

# Prompt del clasificador — 6 facetas, hyde nunca elegible.
CLASSIFIER_PROMPT = (
    "Sos un clasificador de intencion. Responde con UNA SOLA PALABRA eligiendo la faceta:\n"
    "- jax_local = charla casual, saludos, conversacion cotidiana, nada de lo de abajo.\n"
    "- kimi = codigo, programacion, implementacion, debugging, infraestructura tecnica.\n"
    "- hipatia = investigacion, buscar info actual, noticias, fuentes, hechos verificables.\n"
    "- jekyll = humanidades, arte, literatura, filosofia, musica, interpretacion, reflexion.\n"
    "- thot = auditoria critica, riesgos, fallas, vulnerabilidades, revision adversarial.\n"
    "- ada = formalizacion, logica, algoritmos, demostraciones, matematica, invariantes, complejidad.\n"
    "Si dudas: interpretacion->jekyll; critica/riesgo->thot; formalizacion/demostracion->ada;\n"
    "codigo/implementacion->kimi; actualidad/fuentes->hipatia.\n"
    "NUNCA elijas hyde (es ejecutor, no conversador).\n"
    "Mensaje:\n{texto}\n"
    "Responde SOLO con: jax_local, kimi, hipatia, jekyll, thot o ada"
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
        """Scoring multi-keyword con umbral. Hyde nunca es destino.

        Regla:
        - score[f] = n° de keywords de f que matchean en text.
        - top = faceta con mayor score (desempate: _TIEBREAK).
        - score >= 2 → enrutar a top.
        - score == 1 y keyword STRONG → enrutar a top.
        - else → None (el caller cae al clasificador LLM).
        """
        scores: dict[str, int] = {}
        hit_strong: dict[str, bool] = {}

        for faceta, (kws, strong) in _KW_SETS.items():
            score = 0
            is_strong = False
            for kw in kws:
                if " " in kw:
                    hit = kw in text
                else:
                    hit = bool(re.search(rf"\b{re.escape(kw)}\b", text))
                if hit:
                    score += 1
                    if kw in strong:
                        is_strong = True
            scores[faceta] = score
            hit_strong[faceta] = is_strong

        max_score = max(scores.values())
        if max_score == 0:
            return None

        top: str | None = None
        for faceta in _TIEBREAK:
            if scores[faceta] == max_score:
                top = faceta
                break

        if top is None:
            return None

        if max_score >= 2:
            return top
        if max_score == 1 and hit_strong[top]:
            return top
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
            for faceta in AUTO_FACETAS:
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
