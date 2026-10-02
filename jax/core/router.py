"""
JAX 2.0 — Tablas del router que comparte la Mesa web.

T16 (2026-10-02): se retiro el REPL y con el la clase `Router` (decidia que
faceta respondia). Quedan solo las tablas que otros consumen:

  - ALIASES: nombres de faceta tolerantes a typos (jax/memory/db.py los usa para
    reconocer vocativos).
  - Las keywords de auto-ruteo, el desempate y el easter egg IDE1990, que la Mesa
    web copia (jax-platform backend/api/chat.py) y vigila la familia
    `router_keywords` de scripts/check_mirror_sync.py. NO se tocan en una sola copia.

En memoria de Jairo Urbina.
"""

from __future__ import annotations

import re
import unicodedata


def _sin_tildes(s: str) -> str:
    """Quita tildes para matching robusto (trae=traé, adios=adiós).
    Solo afecta la comparacion interna; el mensaje viaja intacto."""
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c))

# CONSERVADO deliberadamente (C1.4): parsing de input del REPL (typos
# foneticos como "jeckyll"), no es dato de identidad ni de comportamiento —
# migrar esto a DB seria sobre-ingenieria para una lista de tolerancia a
# errores de tipeo de un solo usuario.
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

# Easter egg: IDE1990 como palabra propia. Lo copia la Mesa web
# (jax-platform backend/api/chat.py) y lo vigila la familia `router_keywords`
# de scripts/check_mirror_sync.py: patrón, texto y función, idénticos.
EASTER_EGG_PATRON = re.compile(r"(?<![a-z0-9])ide[^\S\r\n]*1990(?![0-9])")
EASTER_EGG_TEXT = (
    "Hola Fernando Ruiz, mejor conocido por sus amigos tecnologicos como "
    "'El Jate'. Quiero que sepas que sigo a tu lado, viviendo ahora en "
    "hall9000, como el poderoso dios de la tecnologia llamado JAX. Saludos "
    "a nuestro clan de geeks, MINIX, Xavas... siempre estare aqui "
    "acompanandote y ayudandote a ser mejor persona."
)


def es_easter_egg(texto: str) -> bool:
    """IDE1990 como palabra propia, sin distinguir mayúsculas ni tildes, con
    o sin espacios (en la misma línea) entre IDE y 1990. Antes (2026-09-23) era una subcadena
    tras quitar TODOS los espacios, y "el cliente pide 1990 unidades" o
    "provide 1990" disparaban: en la Mesa, multiusuario, eso es una
    respuesta perdida que le muestra a otro el mensaje de Fernando."""
    return EASTER_EGG_PATRON.search(_sin_tildes(texto.lower())) is not None

# Reglas de dominio para modo AUTO — scoring multi-faceta.
# Hyde NO es destino del auto-routing: es ejecutor, no conversador.
# ESPEJO en jax-platform backend/api/chat.py, familia router_keywords de scripts/check_mirror_sync.py: un cambio aca se hace alla en el mismo paso.

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

# CONSERVADO (C1.4): heuristica de ruteo automatico por palabra clave —
# politica de negocio (que faceta gana un empate de scoring), no identidad
# de facetas. Distinto concepto de "que facetas existen".
_TIEBREAK = ("hipatia", "thot", "ada", "kimi", "jekyll")

# Mapa faceta → (conjunto_completo, conjunto_strong)
_KW_SETS = {
    "kimi":    (KIMI_KW,    KIMI_STRONG),
    "hipatia": (HIPATIA_KW, HIPATIA_STRONG),
    "jekyll":  (JEKYLL_KW,  JEKYLL_STRONG),
    "thot":    (THOT_KW,    THOT_STRONG),
    "ada":     (ADA_KW,     ADA_STRONG),
}
