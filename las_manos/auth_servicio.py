"""
LAS MANOS — Autenticación de servicio (2026-09-17).

Hasta este cambio LAS MANOS no autenticaba nada: la identidad salía del cuerpo
del pedido. Cualquier proceso local que alcanzara 127.0.0.1:7777 —— Hyde desde
su jaula (bwrap `--share-net`) o cualquier proceso de `fruiz` —— mandaba
`{"invoked_by": "plataforma"}` a `POST /jacobs/pipeline/{id}/approve-step` y
aprobaba los pasos de Hyde bloqueados en el gate.

Contrato:

- **Deny by default.** Toda ruta exige credencial salvo `PUBLICAS` (hoy sólo
  `GET /health`, que miran monitores y k6). Una ruta nueva queda cubierta sin
  que nadie se acuerde de protegerla.
- **La identidad sale de la credencial**, no del cuerpo. Cada identidad tiene
  su secreto en /etc/jax/.env (`VARIABLES`), un conjunto de rutas permitidas y
  los valores de `invoked_by` / `caller` que puede declarar. Un cuerpo que
  declara otra identidad se rechaza ANTES de llegar a la ruta.
- **Sólo `plataforma` aprueba o reanuda.** La credencial `jacobs` (la usa el
  propio proceso de LAS MANOS para despachar motores y para los sub-pipelines
  de Ada) no alcanza `/approve-step` ni `/resume`: aunque un paso de Hyde corra
  dentro de un proceso legítimo, no se aprueba solo.
- **Comparación en tiempo constante** contra TODAS las credenciales, sin salir
  en la primera coincidencia.
- **Fail-closed**: sin las variables, con valores cortos o repetidos, LAS MANOS
  no arranca (`EntornoInvalido`). Un cuerpo ilegible o con claves duplicadas se
  rechaza: si este módulo y Pydantic pudieran leer distinto el mismo cuerpo, el
  chequeo de identidad se podría esquivar.

Por qué credencial y no SO_PEERCRED: Hyde, jax-platform y LAS MANOS corren
todos como `fruiz`; el uid del par no distingue a Hyde de la plataforma. Lo que
sí los distingue es lo que cada uno puede LEER: la jaula de Hyde no monta
/etc/jax, corre con `--clearenv` y en su propio espacio de PIDs (no ve el
/proc/<pid>/environ de los servicios).

Sin caché nueva: las credenciales se leen una vez al construir el middleware.
Cambiarlas exige reiniciar LAS MANOS y jax-platform (van en el mismo .env).

Las respuestas de rechazo llevan un `code` estable, no prosa: quien lo muestre
al usuario lo traduce.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import hmac
import json
import os
import re
from dataclasses import dataclass

from config_entorno import EntornoInvalido

#: Cabecera con la que un llamador presenta su credencial.
ENCABEZADO = "X-Jax-Credencial-Servicio"

IDENTIDAD_PLATAFORMA = "plataforma"
IDENTIDAD_JACOBS = "jacobs"

#: identidad -> variable de /etc/jax/.env con su secreto.
VARIABLES = {
    IDENTIDAD_PLATAFORMA: "JAX_LAS_MANOS_CREDENCIAL_PLATAFORMA",
    IDENTIDAD_JACOBS: "JAX_LAS_MANOS_CREDENCIAL_JACOBS",
}

#: `secrets.token_urlsafe(32)` da 43 caracteres. Menos que eso no es un secreto
#: de servicio, es una contraseña.
LARGO_MINIMO = 43
_ALFABETO = re.compile(r"^[A-Za-z0-9_\-]+$")

#: (método, path exacto) que no exigen credencial.
PUBLICAS = frozenset({("GET", "/health"), ("HEAD", "/health")})

#: Campos del cuerpo que declaran identidad.
CAMPOS_DE_IDENTIDAD = ("invoked_by", "caller")

CODIGO_SIN_CREDENCIAL = "credencial_de_servicio_invalida"
CODIGO_RUTA_NO_PERMITIDA = "ruta_no_permitida_para_la_identidad"
CODIGO_IDENTIDAD_DECLARADA = "identidad_declarada_no_coincide"
CODIGO_CUERPO_ILEGIBLE = "cuerpo_ilegible"


@dataclass(frozen=True)
class Permiso:
    rutas: tuple[tuple[str, re.Pattern], ...]
    declarables: dict[str, frozenset[str]]

    def admite_ruta(self, metodo: str, path: str) -> bool:
        return any(m == metodo and patron.fullmatch(path) for m, patron in self.rutas)

    def admite_declaracion(self, campo: str, valor: object) -> bool:
        return isinstance(valor, str) and valor in self.declarables.get(campo, frozenset())


_SEGMENTO = r"[^/]+"

PERMISOS: dict[str, Permiso] = {
    # jax-platform: pipelines de la Mesa (crear, leer, reanudar, aprobar,
    # cancelar, y lo que se agregue bajo /jacobs/) + el gate de facetas HTTP
    # del chat.
    IDENTIDAD_PLATAFORMA: Permiso(
        rutas=(
            ("GET", re.compile(r"/jacobs/.+")),
            ("POST", re.compile(r"/jacobs/.+")),
            ("POST", re.compile(r"/motor/authorize-facet")),
            # Endpoint de Procesamiento de Archivos (2026-09-20): jax-platform
            # ya habla HTTP con LAS MANOS para todo lo demás por esta misma
            # identidad -- mismo mecanismo, ninguna autenticación nueva.
            ("POST", re.compile(r"/procesamiento/trabajos")),
            ("GET", re.compile(rf"/procesamiento/trabajos/{_SEGMENTO}")),
        ),
        declarables={
            "invoked_by": frozenset({"plataforma"}),
            "caller": frozenset({"jax_platform_chat"}),
        },
    ),
    # El propio proceso de LAS MANOS: Jacobs despacha pasos a motores y Ada
    # crea sub-pipelines (con su token de un solo uso). Nunca aprueba ni reanuda.
    IDENTIDAD_JACOBS: Permiso(
        rutas=(
            ("POST", re.compile(r"/motor/dispatch")),
            ("GET", re.compile(rf"/motor/job/{_SEGMENTO}")),
            ("POST", re.compile(rf"/motor/job/{_SEGMENTO}/cancel")),
            ("POST", re.compile(r"/jacobs/pipeline")),
        ),
        declarables={
            "invoked_by": frozenset({"ada"}),
            "caller": frozenset({"jacobs"}),
        },
    ),
}


def cargar_credenciales(entorno: dict[str, str] | None = None) -> dict[str, bytes]:
    """identidad -> secreto, validado. Falta, corto, fuera del alfabeto o
    repetido entre identidades -> EntornoInvalido (el servicio no arranca)."""
    entorno = os.environ if entorno is None else entorno
    credenciales: dict[str, bytes] = {}
    for identidad, variable in VARIABLES.items():
        valor = (entorno.get(variable) or "").strip()
        if not valor:
            raise EntornoInvalido(
                f"{variable} no está seteada: agregala a /etc/jax/.env (sin default silencioso)."
            )
        if len(valor) < LARGO_MINIMO or not _ALFABETO.fullmatch(valor):
            raise EntornoInvalido(
                f"{variable} tiene que tener al menos {LARGO_MINIMO} caracteres "
                "[A-Za-z0-9_-] (secrets.token_urlsafe(32))."
            )
        credenciales[identidad] = valor.encode("ascii")
    if len(set(credenciales.values())) != len(credenciales):
        raise EntornoInvalido(
            f"{', '.join(VARIABLES.values())} tienen que ser distintas entre sí."
        )
    return credenciales


def identidad_de(presentada: bytes | None, credenciales: dict[str, bytes]) -> str | None:
    """Compara contra TODAS en tiempo constante; sin cortar en la primera."""
    if not presentada:
        return None
    hallada = None
    for identidad, secreto in credenciales.items():
        if hmac.compare_digest(presentada, secreto):
            hallada = identidad
    return hallada


def encabezado_propio(identidad: str) -> dict[str, str]:
    """Cabecera para un llamador que corre con el .env cargado (Jacobs dentro de
    LAS MANOS). Sin la variable levanta EntornoInvalido: no se manda un pedido
    sin credencial que igual iba a ser rechazado."""
    return {ENCABEZADO: cargar_credenciales()[identidad].decode("ascii")}


class _DuplicadaError(ValueError):
    pass


def _sin_duplicadas(pares: list[tuple[str, object]]) -> dict:
    vistas: dict = {}
    for clave, valor in pares:
        if clave in vistas:
            raise _DuplicadaError(clave)
        vistas[clave] = valor
    return vistas


class CredencialDeServicio:
    """Middleware ASGI. Se instala con `proteger(app)`."""

    def __init__(self, app, credenciales: dict[str, bytes]):
        self.app = app
        self._credenciales = credenciales

    async def __call__(self, scope, receive, send):
        tipo = scope["type"]
        if tipo == "lifespan":
            return await self.app(scope, receive, send)
        if tipo != "http":
            # Ni websockets ni nada más: LAS MANOS no los usa.
            return await _responder_cerrar(send, scope)

        metodo = scope["method"]
        path = scope["path"]
        if (metodo, path) in PUBLICAS:
            return await self.app(scope, receive, send)

        presentada = None
        for nombre, valor in scope.get("headers", []):
            if nombre.decode("latin-1").lower() == ENCABEZADO.lower():
                presentada = valor
                break
        identidad = identidad_de(presentada, self._credenciales)
        if identidad is None:
            return await _responder(send, 401, CODIGO_SIN_CREDENCIAL)

        permiso = PERMISOS[identidad]
        if not permiso.admite_ruta(metodo, path):
            return await _responder(send, 403, CODIGO_RUTA_NO_PERMITIDA)

        cuerpo = await _leer_cuerpo(receive)
        if cuerpo.strip():
            try:
                datos = json.loads(cuerpo, object_pairs_hook=_sin_duplicadas)
            except (ValueError, UnicodeDecodeError):  # fail-closed: cuerpo ilegible o con claves duplicadas -> 400; la ruta nunca lo ve
                return await _responder(send, 400, CODIGO_CUERPO_ILEGIBLE)
            if isinstance(datos, dict):
                for campo in CAMPOS_DE_IDENTIDAD:
                    if campo in datos and not permiso.admite_declaracion(campo, datos[campo]):
                        return await _responder(send, 403, CODIGO_IDENTIDAD_DECLARADA)

        scope.setdefault("state", {})["identidad_servicio"] = identidad
        return await self.app(scope, _reproducir(cuerpo, receive), send)


async def _leer_cuerpo(receive) -> bytes:
    partes = []
    while True:
        mensaje = await receive()
        if mensaje["type"] == "http.disconnect":
            break
        partes.append(mensaje.get("body", b""))
        if not mensaje.get("more_body", False):
            break
    return b"".join(partes)


def _reproducir(cuerpo: bytes, receive):
    entregado = False

    async def _receive():
        nonlocal entregado
        if not entregado:
            entregado = True
            return {"type": "http.request", "body": cuerpo, "more_body": False}
        return await receive()

    return _receive


async def _responder(send, status: int, codigo: str) -> None:
    cuerpo = json.dumps({"detail": {"code": codigo}}).encode("utf-8")
    await send({
        "type": "http.response.start",
        "status": status,
        "headers": [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(cuerpo)).encode("ascii")),
        ],
    })
    await send({"type": "http.response.body", "body": cuerpo})


async def _responder_cerrar(send, scope) -> None:
    if scope["type"] == "websocket":
        await send({"type": "websocket.close", "code": 1008})


def proteger(app, credenciales: dict[str, bytes] | None = None) -> None:
    """Instala el middleware en `app`. Sin `credenciales`, las lee del entorno
    y falla cerrado si faltan."""
    app.add_middleware(
        CredencialDeServicio,
        credenciales=cargar_credenciales() if credenciales is None else credenciales,
    )
