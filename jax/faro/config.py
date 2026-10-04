"""Configuracion del Faro: todo sale del entorno y FALLA CERRADO si falta algo.

Nada de rutas, SHA ni sockets escritos en el codigo (Principio IV). En produccion las
variables las pone `/etc/jax/.env` (root); en pruebas se construye `ConfigFaro` a mano
con directorios temporales. Misma disciplina que `cli_sandbox` con `JAX_CLI_LOCK_DIR`:
una variable ausente o vacia no tiene valor por defecto, es un error.

  JAX_FARO_REPO            checkout de `claude-skills` del que se LEEN objetos git
                           (nunca su arbol de trabajo)
  JAX_FARO_SHA             SHA completo (40 hex) de `origin/main` que se fija
  JAX_FARO_ECOSISTEMA_DIR  destino de los paquetes (en produccion
                           `/srv/jax-prod/ecosistema`; el paquete queda en `<dir>/<SHA>/`)
  JAX_FARO_DUENIO_UID      opcional, uid del dueño esperado del paquete (default 0 = root)
  JAX_FARO_REF_FRESCURA    opcional, la ref contra la que se mide la frescura
                           (default `origin/main`; es un nombre de ref, no un dato de entorno)
  JAX_FARO_SOCKET_DIR      directorio de los sockets del Puerto (`<dir>/<run_id>.sock`; en
                           produccion `/run/faro`). Absoluto, del usuario del servicio y sin
                           escritura de grupo/otros
  JAX_FARO_MAX_MENSAJE     opcional, tope en bytes de UN mensaje MCP (default 1 MiB)
  JAX_FARO_PRESUPUESTO_BYTES, JAX_FARO_HANDSHAKE_S, JAX_FARO_MENSAJE_TIMEOUT_S, JAX_FARO_COSTO_CONEXION_BYTES   opcionales; ver ConfigPuerto
"""
from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

REF_FRESCURA_POR_DEFECTO = "origin/main"
_RE_SHA = re.compile(r"^[0-9a-f]{40}$")


class ConfigFaroInvalida(ValueError):
    """La configuracion falta o no es valida: el Faro no arranca."""


@dataclass(frozen=True)
class ConfigMemoria:
    """Compuerta explícita de memoria. Solo se conecta al perfil local de prueba.

    B9 no está reactivada en producción y su contrato de integración para el
    Faro sigue pendiente. La allowlist evita que este adaptador provisional
    pueda apuntar a `jax_memory` o a un servidor distinto.
    """
    habilitada: bool = False
    host: str = "127.0.0.1"
    port: int = 3308
    usuario: str = "jax_test"
    clave: str = field(default="", repr=False)
    base: str = "jax_memory_test"

    @classmethod
    def desde_entorno(cls, env: Mapping[str, str]) -> "ConfigMemoria":
        raw = (env.get("JAX_FARO_MEMORIA_HABILITADA") or "false").strip().lower()
        if raw in {"", "false", "0", "no"}:
            return cls()
        if raw not in {"true", "1", "si", "sí"}:
            raise ConfigFaroInvalida("JAX_FARO_MEMORIA_HABILITADA debe ser true o false")
        host = (env.get("JAX_FARO_MEMORIA_TEST_DB_HOST") or "").strip()
        port_raw = (env.get("JAX_FARO_MEMORIA_TEST_DB_PORT") or "").strip()
        user = (env.get("JAX_FARO_MEMORIA_TEST_DB_USER") or "").strip()
        database = (env.get("JAX_FARO_MEMORIA_TEST_DB_NAME") or "").strip()
        password = env.get("JAX_FARO_MEMORIA_TEST_DB_PASSWORD") or ""
        try:
            port = int(port_raw)
        except ValueError as exc:
            raise ConfigFaroInvalida("JAX_FARO_MEMORIA_TEST_DB_PORT debe ser 3308") from exc
        if (host, port, user, database) != ("127.0.0.1", 3308, "jax_test", "jax_memory_test"):
            raise ConfigFaroInvalida(
                "memoria no disponible: este adaptador solo permite la base de prueba "
                "jax_memory_test en 127.0.0.1:3308 con jax_test")
        if not password:
            raise ConfigFaroInvalida("memoria no disponible: falta la credencial de la base de prueba")
        return cls(True, host, port, user, password, database)


def sha_valido(sha: object) -> bool:
    return isinstance(sha, str) and _RE_SHA.fullmatch(sha) is not None


@dataclass(frozen=True)
class ConfigFaro:
    repo: Path
    sha: str
    destino: Path
    ref_frescura: str = REF_FRESCURA_POR_DEFECTO
    uid_duenio: int = 0   # dueño esperado del paquete: root en produccion; las pruebas lo inyectan

    def __post_init__(self) -> None:
        if not sha_valido(self.sha):
            raise ConfigFaroInvalida(f"JAX_FARO_SHA tiene que ser un SHA completo de 40 hex, no {self.sha!r}")

    @property
    def raiz_paquete(self) -> Path:
        return self.destino / self.sha

    @classmethod
    def desde_entorno(cls, env: Mapping[str, str], *, solo_pruebas_duenio_igual_servicio: bool = False) -> "ConfigFaro":
        def pedir(nombre: str) -> str:
            valor = (env.get(nombre) or "").strip()
            if not valor:
                raise ConfigFaroInvalida(f"{nombre} no esta definida: sin ella el Faro no arranca")
            return valor

        def ruta_absoluta(nombre: str) -> Path:
            ruta = Path(pedir(nombre))
            if not ruta.is_absolute():
                raise ConfigFaroInvalida(f"{nombre} tiene que ser una ruta absoluta, no {str(ruta)!r}")
            return ruta

        if (env.get("JAX_FARO_PLUGINS") or "").strip() or "JAX_FARO_PLUGINS" in env:
            raise ConfigFaroInvalida(
                "JAX_FARO_PLUGINS no se acepta todavia: los plugins se leian del arbol de trabajo (mutable tras fijar el "
                "SHA); solo entraran cuando se lean por objetos git de su SHA")
        crudo_uid = (env.get("JAX_FARO_DUENIO_UID") or "").strip()
        try:
            uid_duenio = int(crudo_uid) if crudo_uid else 0
        except ValueError as exc:
            raise ConfigFaroInvalida("JAX_FARO_DUENIO_UID no es un entero") from exc
        if uid_duenio == os.geteuid() and os.geteuid() != 0 and not solo_pruebas_duenio_igual_servicio:
            raise ConfigFaroInvalida(
                f"JAX_FARO_DUENIO_UID={uid_duenio} es el usuario del servicio: el dueño del paquete (root) no puede ser quien lo carga; "
                "si el servicio fuera dueño del paquete podria reescribirlo")
        return cls(
            uid_duenio=uid_duenio,
            repo=ruta_absoluta("JAX_FARO_REPO"),
            sha=pedir("JAX_FARO_SHA"),
            destino=ruta_absoluta("JAX_FARO_ECOSISTEMA_DIR"),
            ref_frescura=(env.get("JAX_FARO_REF_FRESCURA") or "").strip() or REF_FRESCURA_POR_DEFECTO,
        )


MAX_MENSAJE_POR_DEFECTO = 1024 * 1024
PRESUPUESTO_POR_DEFECTO = 256 * 1024 * 1024
COSTO_CONEXION_POR_DEFECTO = 256 * 1024
HANDSHAKE_S_POR_DEFECTO = 5.0
MENSAJE_TIMEOUT_S_POR_DEFECTO = 30.0


@dataclass(frozen=True)
class ConfigPuerto:
    """Configuracion del transporte del Puerto.

    - `max_mensaje`: tope de UN mensaje MCP (default 1 MiB). Uno mas largo cierra la conexion.
    - `presupuesto_bytes`: presupuesto GLOBAL de bytes en vuelo (mensajes a medio leer y respuestas
      a medio escribir, de todas las conexiones juntas). No hay tope de CONEXIONES (D-4: el unico
      limite son los recursos de la maquina); el presupuesto acota lo que el servicio retiene, no
      cuantos hablan. Para la unidad de systemd: `MemoryMax` >= presupuesto + el paquete cargado
      + ~200 MiB de base del interprete y del SDK.
    - `costo_conexion_bytes`: costo FIJO que cada conexion cobra al presupuesto desde que se acepta hasta
      que se cierra (default 256 KiB: lo que cuesta tenerla viva, un servidor MCP con sus tareas y
      buffers). La que no entra ESPERA, no se rechaza (D-4): el numero de conexiones simultaneas queda
      acotado por `presupuesto / costo` y por nada mas.
    - `handshake_s`: plazo para que llegue la linea del token. `mensaje_timeout_s`: plazo maximo
      de un mensaje a medias (o de una escritura que el par no lee) antes de cortar y devolver
      lo que retenia."""
    socket_dir: Path
    max_mensaje: int = MAX_MENSAJE_POR_DEFECTO
    presupuesto_bytes: int = PRESUPUESTO_POR_DEFECTO
    handshake_s: float = HANDSHAKE_S_POR_DEFECTO
    mensaje_timeout_s: float = MENSAJE_TIMEOUT_S_POR_DEFECTO
    costo_conexion_bytes: int = COSTO_CONEXION_POR_DEFECTO

    def __post_init__(self) -> None:
        if not Path(self.socket_dir).is_absolute():
            raise ConfigFaroInvalida(f"JAX_FARO_SOCKET_DIR tiene que ser una ruta absoluta, no {str(self.socket_dir)!r}")
        if not isinstance(self.max_mensaje, int) or self.max_mensaje < 1024:
            raise ConfigFaroInvalida("JAX_FARO_MAX_MENSAJE tiene que ser un entero de al menos 1024 bytes")
        if not isinstance(self.presupuesto_bytes, int) or self.presupuesto_bytes < 4 * self.max_mensaje:
            raise ConfigFaroInvalida("JAX_FARO_PRESUPUESTO_BYTES tiene que ser un entero de al menos 4 veces max_mensaje "
                                     "(si no, no cabria ni un mensaje maximo con sus copias)")
        if not isinstance(self.costo_conexion_bytes, int) or not 0 <= self.costo_conexion_bytes <= self.presupuesto_bytes:
            raise ConfigFaroInvalida("JAX_FARO_COSTO_CONEXION_BYTES tiene que ser un entero entre 0 y el presupuesto")
        if not self.handshake_s > 0 or not self.mensaje_timeout_s > 0:
            raise ConfigFaroInvalida("los plazos del Puerto tienen que ser positivos")

    @classmethod
    def desde_entorno(cls, env: Mapping[str, str]) -> "ConfigPuerto":
        crudo = (env.get("JAX_FARO_SOCKET_DIR") or "").strip()
        if not crudo:
            raise ConfigFaroInvalida("JAX_FARO_SOCKET_DIR no esta definida: sin ella el Puerto no arranca")

        def numero(nombre: str, tipo, defecto):
            valor = (env.get(nombre) or "").strip()
            try:
                return tipo(valor) if valor else defecto
            except ValueError as exc:
                raise ConfigFaroInvalida(f"{nombre} no es un numero valido") from exc

        return cls(
            socket_dir=Path(crudo),
            max_mensaje=numero("JAX_FARO_MAX_MENSAJE", int, MAX_MENSAJE_POR_DEFECTO),
            presupuesto_bytes=numero("JAX_FARO_PRESUPUESTO_BYTES", int, PRESUPUESTO_POR_DEFECTO),
            handshake_s=numero("JAX_FARO_HANDSHAKE_S", float, HANDSHAKE_S_POR_DEFECTO),
            mensaje_timeout_s=numero("JAX_FARO_MENSAJE_TIMEOUT_S", float, MENSAJE_TIMEOUT_S_POR_DEFECTO),
            costo_conexion_bytes=numero("JAX_FARO_COSTO_CONEXION_BYTES", int, COSTO_CONEXION_POR_DEFECTO),
        )
