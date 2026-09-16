"""Redaccion de secretos en textos de error -- Ruling T6-6 (2026-09-15).

La key de Gemini viajaba en la query (`?key=`) en los dos caminos de hipatia
(jacobs/executor.py::_invoke_http_gemini y jax/muscles/base.py::_call_gemini),
y httpx mete la URL COMPLETA en str(HTTPStatusError) y en su log INFO de cada
pedido. Hoy la key va en la cabecera `x-goog-api-key`; esto cubre lo que queda:
un proveedor que devuelve la key en el cuerpo del error, y cualquier texto de
error que se guarde (jacobs_steps.error, jacobs_events), se loguee o se muestre.

PARIDAD (fix wave final, ronda 3, portado 2026-09-15): las reglas son las de
jax-platform/backend/redaccion.py VERBATIM (su fix wave final hasta la ronda
3 -- comillas escapadas y comilla sin cerrar), y tests/test_redaccion.py
porta sus tablas de casos para que las dos copias queden clavadas. Ver
_AUTH_CONTEXTO y _PARAM_SECRETO.

Puro: sin I/O, sin estado. Vive en jax/core; las_manos/redaccion.py es un
symlink (jacobs importa plano, como grounding_sources).
"""
from __future__ import annotations

import re
from collections.abc import Iterable

MARCA = "***"

# Nombre de secreto seguido de `=` o `:` (con o sin espacios) y su valor, en
# query strings, cabeceras, JSON o texto libre: `key=`, `api_key="…"`,
# `'token': '…'`, `"api_key": "…"`, `token = x`, `x-goog-api-key: …`,
# `password=`, `secret=`, `credential=`.
#   - El nombre empieza en un borde (lookbehind: ni letra, ni digito, ni `_`
#     ni `-`), asi `monkey=5` y `turkey=` NO se tocan.
#   - Puede llevar prefijos separados por `_`/`-` (`api_key`, `access_token`,
#     `x-goog-api-key`, `client_secret`). Perdida aceptada: `sort_key=` y
#     `cache_key=` tambien se tapan -- mejor de mas que de menos.
#   - Sufijo `_id` en TODOS los nombres: `private_key_id`, `token_id`,
#     `secret_id`, `credential_id`, `client_secret_id`.
#   - `for key 'PRIMARY'` no tiene `=`/`:` -> intacto. El nombre `key` A SECAS
#     con `:` y sin comillas tampoco se tapa: `for key: PRIMARY` es el texto
#     de un error de MariaDB (`key=` y `"key": "…"` si se tapan).
#   - El valor entre comillas conserva las comillas; sin comillas termina en
#     `&`, espacio, comilla, `,`, `;`, `<`, `>`, `}` o `]`.
#   - Fix wave final, ronda 3 (paridad con jax-platform/backend/redaccion.py):
#     la forma entre comillas acepta escapes con barra (`\"`, `\'` dentro del
#     valor no lo cierran antes de tiempo) y, si la comilla no cierra, tapa
#     hasta el final del texto (perdida aceptada, como con `sort_key`).
#     `(?:\\.|[^"\\])*` parte el texto de una sola manera -- lineal, sin
#     backtracking catastrofico (medido con 100 KB adversariales). Flag `s`
#     para que `\\.` tome tambien un salto de linea escapado.
_PARAM_SECRETO = re.compile(
    r"""(?isx)
    (?<![a-z0-9_\-])
    (["']?)
    ((?:[a-z0-9]+[_\-])*(?:api_?key|key|token|password|passwd|secret|credentials?)(?:[_\-]id)?)
    \1
    (\s*[=:]\s*)
    (?:"((?:\\.|[^"\\])*\\?)(?:"|\Z)|'((?:\\.|[^'\\])*\\?)(?:'|\Z)|([^&\s'",;<>}\]]+))
    """)

# Esquemas de autenticacion (Bearer/Basic/Token/Digest):
#   1. En contexto Authorization -- cabecera `Authorization: <esquema> <valor>`,
#      parametro `authorization=<valor>` o JSON `"authorization": "…"` -- el
#      valor se tapa SIEMPRE, con o sin esquema; el esquema queda visible.
#      Clase de valor amplia (`abc!def`, `Bearer%20abc123` enteros).
#      Fix round 1 jax (hueco compartido con la plataforma): el valor puede ir
#      ENTRE COMILLAS despues del esquema (`Bearer 'x'`, `Bearer "x y"`); antes
#      la clase sin comillas no lo tomaba, el esquema pasaba a ser el "valor"
#      y el secreto entre comillas quedaba entero en claro.
#   2. Fuera de ese contexto, un esquema suelto solo se trata como credencial
#      si lo que sigue TIENE FORMA de credencial: 16 caracteres de token o mas
#      y al menos un digito. Asi "basic idea of it" y "the bearer of bad news"
#      quedan intactos, y "Bearer eyJhbGciOiJIUzI1NiJ9…" no.
#   3. Fix wave final, ronda 2 (paridad con jax-platform/backend/redaccion.py):
#      el valor ENTERO puede ir entre comillas, con o sin esquema adentro y con
#      espacios (`authorization: "secret value"`, `"authorization": "Bearer abc
#      def"`). Antes un grupo de comilla opcional se tragaba la de apertura y
#      el valor sin comillas cortaba en el primer espacio: `"*** value"`. Ahora
#      cada forma entre comillas es su propia alternativa y se tapa hasta la
#      comilla de cierre.
#   4. Ronda 3 (2026-09-15): toda forma entre comillas aca (igual que en
#      _PARAM_SECRETO) acepta escapes con barra -- `\"` o `\'` dentro del valor
#      ya no lo cierra antes de tiempo -- y una comilla SIN CERRAR tapa hasta
#      el final del texto (antes no entraba en ninguna alternativa y el valor
#      salia entero en claro). Lineal, sin backtracking catastrofico.
_AUTH_CONTEXTO = re.compile(
    r"""(?isx)
    (?<![a-z0-9_\-])
    (["']?)(authorization)\1
    (\s*[=:]\s*)
    (?:
        "(?:(bearer|basic|token|digest)\s+)?((?:\\.|[^"\\])*\\?)(?:"|\Z)
      | '(?:(bearer|basic|token|digest)\s+)?((?:\\.|[^'\\])*\\?)(?:'|\Z)
      | (?:(bearer|basic|token|digest)\s+)?
        (?:"((?:\\.|[^"\\])*\\?)(?:"|\Z)
         | '((?:\\.|[^'\\])*\\?)(?:'|\Z)
         | ([^\s"'&,;<>}\]]+))
    )
    """)
_ESQUEMA_SUELTO = re.compile(
    r"(?i)\b(bearer|basic|token|digest)\s+(?=[A-Za-z0-9._~+/=\-]*\d)[A-Za-z0-9._~+/=\-]{16,}")

# Forma de las API keys de Google (Gemini): "AIza" + 35 caracteres. Se exige
# un minimo de 10 para no comerse palabras cortas que empiecen igual.
_KEY_GOOGLE = re.compile(r"AIza[0-9A-Za-z_\-]{10,}")


def _tapar_auth(m: re.Match) -> str:
    comilla, nombre, separador = m.group(1, 2, 3)
    if m.group(5) is not None:              # "<esquema> valor" entero entre dobles
        esquema = m.group(4)
        prefijo = f"{esquema} " if esquema else ""
        return f'{comilla}{nombre}{comilla}{separador}"{prefijo}{MARCA}"'
    if m.group(7) is not None:              # '<esquema> valor' entero entre simples
        esquema = m.group(6)
        prefijo = f"{esquema} " if esquema else ""
        return f"{comilla}{nombre}{comilla}{separador}'{prefijo}{MARCA}'"
    esquema = m.group(8)                    # esquema sin comillas; valor con o sin
    prefijo = f"{esquema} " if esquema else ""
    if m.group(9) is not None:
        valor = f'"{MARCA}"'
    elif m.group(10) is not None:
        valor = f"'{MARCA}'"
    else:
        valor = MARCA
    return f"{comilla}{nombre}{comilla}{separador}{prefijo}{valor}"


def _tapar_param(m: re.Match) -> str:
    comilla, nombre, separador = m.group(1), m.group(2), m.group(3)
    if nombre.lower() == "key" and not comilla and ":" in separador:
        return m.group(0)          # `for key: PRIMARY` -- texto de MariaDB
    if m.group(4) is not None:
        valor = f'"{MARCA}"'
    elif m.group(5) is not None:
        valor = f"'{MARCA}'"
    else:
        valor = MARCA
    return f"{comilla}{nombre}{comilla}{separador}{valor}"


def redactar_secretos(texto: str | None, secretos: Iterable[str | None] = ()) -> str | None:
    """Reemplaza por `***` los secretos conocidos (`secretos`), los valores
    de `key=`/`api_key=`/`token=`/`password=`/`secret=`/`credential=` (y sus
    `_id`), el valor de un contexto Authorization, un esquema suelto con forma
    de credencial y las keys con forma `AIza...`.

    None y "" pasan tal cual. Un texto sin secretos no cambia."""
    if not texto:
        return texto
    # Los mas largos primero: si un secreto contiene a otro, el corto no
    # deja un resto del largo sin tapar.
    for s in sorted((s for s in secretos if s), key=len, reverse=True):
        texto = texto.replace(s, MARCA)
    # Authorization primero: si _PARAM_SECRETO viera antes `Token abc`, no lo
    # reconoceria, y en `Authorization: Bearer x` taparia el esquema y dejaria x.
    texto = _AUTH_CONTEXTO.sub(_tapar_auth, texto)
    texto = _PARAM_SECRETO.sub(_tapar_param, texto)
    texto = _ESQUEMA_SUELTO.sub(lambda m: f"{m.group(1)} {MARCA}", texto)
    return _KEY_GOOGLE.sub(MARCA, texto)


def recortar_redactado(texto: str | None, limite: int,
                       secretos: Iterable[str | None] = ()) -> str | None:
    """Redacta PRIMERO y recorta despues. Al reves, un secreto que cruza el
    corte queda partido: el pedazo ya no tiene la forma que reconocen las
    reglas (ni el secreto conocido entero) y se filtra en claro. Es la unica
    forma de recortar un texto de error de proveedor en jax."""
    limpio = redactar_secretos(texto, secretos)
    return limpio if limpio is None else limpio[:limite]


def texto_de_error(e: BaseException, secretos: Iterable[str | None] = ()) -> str:
    """`"<Tipo>: <mensaje>"` de una excepcion, ya redactado."""
    return redactar_secretos(f"{type(e).__name__}: {e}", secretos)
