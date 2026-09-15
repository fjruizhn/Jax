"""Redaccion de secretos en textos de error -- Ruling T6-6 (2026-09-15).

La key de Gemini viajaba en la query (`?key=`) en los dos caminos de hipatia
(jacobs/executor.py::_invoke_http_gemini y jax/muscles/base.py::_call_gemini),
y httpx mete la URL COMPLETA en str(HTTPStatusError) y en su log INFO de cada
pedido. Hoy la key va en la cabecera `x-goog-api-key`; esto cubre lo que queda:
un proveedor que devuelve la key en el cuerpo del error, y cualquier texto de
error que se guarde (jacobs_steps.error, jacobs_events), se loguee o se muestre.

Mismas reglas que jax-platform/backend/redaccion.py (rama
feat/pendientes-2026-09-22), con la re-revision de la plataforma del mismo dia:
los esquemas Basic/Bearer/Token SOLO se tapan tras una cabecera Authorization o
en un contexto `authorization=` -- "the basic idea" es prosa, no un secreto.

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
# `password=`, `secret=`, `credential=`, `private_key_id: …`.
#   - El nombre empieza en un borde (lookbehind: ni letra, ni digito, ni `_`
#     ni `-`), asi `monkey=5` y `turkey=` NO se tocan.
#   - Puede llevar prefijos separados por `_`/`-` (`api_key`, `access_token`,
#     `x-goog-api-key`, `client_secret`, `private_key_id`). Perdida aceptada:
#     `sort_key=` y `cache_key=` tambien se tapan -- mejor de mas que de menos.
#   - `for key 'PRIMARY'` no tiene `=`/`:` -> intacto.
#   - `authorization` NO esta aca: tiene su propia regla (conserva el esquema).
#   - El valor entre comillas conserva las comillas; sin comillas termina en
#     `&`, espacio, comilla, `,`, `;`, `<`, `>`, `}` o `]`.
_PARAM_SECRETO = re.compile(
    r"""(?ix)
    (?<![a-z0-9_\-])
    (["']?)
    ((?:[a-z0-9]+[_\-])*(?:api_?key|key_id|key|token|password|passwd|secret|credentials?))
    \1
    (\s*[=:]\s*)
    (?:"([^"]*)"|'([^']*)'|([^&\s'",;<>}\]]+))
    """)

# `Authorization: Bearer x`, `Authorization: Token x`, `authorization=x`,
# `"Authorization": "Basic x"`: el nombre y el esquema quedan, el valor no.
# Un esquema FUERA de este contexto no se toca (prosa).
_AUTORIZACION = re.compile(
    r"""(?ix)
    (?<![a-z0-9_\-])
    (["']?)(authorization)\1
    (\s*[=:]\s*)
    (["']?)
    (?:(bearer|basic|token)\s+)?
    [A-Za-z0-9._~+/=\-]+
    """)

# Forma de las API keys de Google (Gemini): "AIza" + 35 caracteres. Se exige
# un minimo de 10 para no comerse palabras cortas que empiecen igual.
_KEY_GOOGLE = re.compile(r"AIza[0-9A-Za-z_\-]{10,}")


def _tapar_param(m: re.Match) -> str:
    comilla, nombre, separador = m.group(1), m.group(2), m.group(3)
    if m.group(4) is not None:
        valor = f'"{MARCA}"'
    elif m.group(5) is not None:
        valor = f"'{MARCA}'"
    else:
        valor = MARCA
    return f"{comilla}{nombre}{comilla}{separador}{valor}"


def _tapar_autorizacion(m: re.Match) -> str:
    comilla, nombre, separador, comilla_valor, esquema = m.groups()
    prefijo = f"{esquema} " if esquema else ""
    return f"{comilla}{nombre}{comilla}{separador}{comilla_valor}{prefijo}{MARCA}"


def redactar_secretos(texto: str | None, secretos: Iterable[str | None] = ()) -> str | None:
    """Reemplaza por `***` los secretos conocidos (`secretos`), los valores
    de `key=`/`api_key=`/`token=`/`password=`/`secret=`/`credential=`, el
    valor de una cabecera Authorization y las keys con forma `AIza...`.

    None y "" pasan tal cual. Un texto sin secretos no cambia."""
    if not texto:
        return texto
    # Los mas largos primero: si un secreto contiene a otro, el corto no
    # deja un resto del largo sin tapar.
    for s in sorted((s for s in secretos if s), key=len, reverse=True):
        texto = texto.replace(s, MARCA)
    texto = _AUTORIZACION.sub(_tapar_autorizacion, texto)
    texto = _PARAM_SECRETO.sub(_tapar_param, texto)
    return _KEY_GOOGLE.sub(MARCA, texto)


def recortar_redactado(texto: str | None, limite: int,
                       secretos: Iterable[str | None] = ()) -> str | None:
    """Redacta PRIMERO y recorta despues. Al reves, un secreto que cruza el
    corte queda partido: el pedazo ya no tiene la forma que reconocen las
    reglas (ni el secreto conocido entero) y se filtra en claro."""
    limpio = redactar_secretos(texto, secretos)
    return limpio if limpio is None else limpio[:limite]


def texto_de_error(e: BaseException, secretos: Iterable[str | None] = ()) -> str:
    """`"<Tipo>: <mensaje>"` de una excepcion, ya redactado."""
    return redactar_secretos(f"{type(e).__name__}: {e}", secretos)
