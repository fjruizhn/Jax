"""Redaccion de secretos en textos de error -- Ruling T6-6 (2026-09-15).

La key de Gemini viajaba en la query (`?key=`) en los dos caminos de hipatia
(jacobs/executor.py::_invoke_http_gemini y jax/muscles/base.py::_call_gemini),
y httpx mete la URL COMPLETA en str(HTTPStatusError) y en su log INFO de cada
pedido. Hoy la key va en la cabecera `x-goog-api-key`; esto cubre lo que queda:
un proveedor que devuelve la key en el cuerpo del error, y cualquier texto de
error que se guarde (jacobs_steps.error, jacobs_events), se loguee o se muestre.

PARIDAD (fix round 1, review de 05c028b): las reglas son las de
jax-platform/backend/redaccion.py VERBATIM (su fix round 2), y
tests/test_redaccion.py porta sus tablas de casos para que las dos copias
queden clavadas. Unica diferencia, pendiente de portarse a la plataforma: un
valor ENTRE COMILLAS despues del esquema (`Authorization: Bearer 'x'`) -- ver
_AUTH_CONTEXTO.

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
_PARAM_SECRETO = re.compile(
    r"""(?ix)
    (?<![a-z0-9_\-])
    (["']?)
    ((?:[a-z0-9]+[_\-])*(?:api_?key|key|token|password|passwd|secret|credentials?)(?:[_\-]id)?)
    \1
    (\s*[=:]\s*)
    (?:"([^"]*)"|'([^']*)'|([^&\s'",;<>}\]]+))
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
_AUTH_CONTEXTO = re.compile(
    r"""(?ix)
    (?<![a-z0-9_\-])
    (["']?)(authorization)\1
    (\s*[=:]\s*)
    (["']?)
    (?:(bearer|basic|token|digest)\s+)?
    (?:"([^"]*)"|'([^']*)'|([^\s"'&,;<>}\]]+))
    """)
_ESQUEMA_SUELTO = re.compile(
    r"(?i)\b(bearer|basic|token|digest)\s+(?=[A-Za-z0-9._~+/=\-]*\d)[A-Za-z0-9._~+/=\-]{16,}")

# Forma de las API keys de Google (Gemini): "AIza" + 35 caracteres. Se exige
# un minimo de 10 para no comerse palabras cortas que empiecen igual.
_KEY_GOOGLE = re.compile(r"AIza[0-9A-Za-z_\-]{10,}")


def _tapar_auth(m: re.Match) -> str:
    comilla, nombre, separador, comilla_valor, esquema = m.group(1, 2, 3, 4, 5)
    prefijo = f"{esquema} " if esquema else ""
    if m.group(6) is not None:
        valor = f'"{MARCA}"'
    elif m.group(7) is not None:
        valor = f"'{MARCA}'"
    else:
        valor = MARCA
    return f"{comilla}{nombre}{comilla}{separador}{comilla_valor}{prefijo}{valor}"


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
