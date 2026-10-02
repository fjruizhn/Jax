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
  JAX_FARO_REF_FRESCURA    opcional, la ref contra la que se mide la frescura
                           (default `origin/main`; es un nombre de ref, no un dato de entorno)
  JAX_FARO_SOCKET_DIR      directorio de los sockets del Puerto (`<dir>/<run_id>.sock`; en
                           produccion `/run/faro`). Absoluto, del usuario del servicio y sin
                           escritura de grupo/otros
  JAX_FARO_MAX_MENSAJE     opcional, tope en bytes de UN mensaje MCP (default 16 MiB)
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

REF_FRESCURA_POR_DEFECTO = "origin/main"
_RE_SHA = re.compile(r"^[0-9a-f]{40}$")


class ConfigFaroInvalida(ValueError):
    """La configuracion falta o no es valida: el Faro no arranca."""


def sha_valido(sha: object) -> bool:
    return isinstance(sha, str) and _RE_SHA.fullmatch(sha) is not None


@dataclass(frozen=True)
class ConfigFaro:
    repo: Path
    sha: str
    destino: Path
    ref_frescura: str = REF_FRESCURA_POR_DEFECTO

    def __post_init__(self) -> None:
        if not sha_valido(self.sha):
            raise ConfigFaroInvalida(f"JAX_FARO_SHA tiene que ser un SHA completo de 40 hex, no {self.sha!r}")

    @property
    def raiz_paquete(self) -> Path:
        return self.destino / self.sha

    @classmethod
    def desde_entorno(cls, env: Mapping[str, str]) -> "ConfigFaro":
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

        return cls(
            repo=ruta_absoluta("JAX_FARO_REPO"),
            sha=pedir("JAX_FARO_SHA"),
            destino=ruta_absoluta("JAX_FARO_ECOSISTEMA_DIR"),
            ref_frescura=(env.get("JAX_FARO_REF_FRESCURA") or "").strip() or REF_FRESCURA_POR_DEFECTO,
        )


MAX_MENSAJE_POR_DEFECTO = 16 * 1024 * 1024


@dataclass(frozen=True)
class ConfigPuerto:
    socket_dir: Path
    max_mensaje: int = MAX_MENSAJE_POR_DEFECTO

    def __post_init__(self) -> None:
        if not Path(self.socket_dir).is_absolute():
            raise ConfigFaroInvalida(f"JAX_FARO_SOCKET_DIR tiene que ser una ruta absoluta, no {str(self.socket_dir)!r}")
        if not isinstance(self.max_mensaje, int) or self.max_mensaje < 1024:
            raise ConfigFaroInvalida("JAX_FARO_MAX_MENSAJE tiene que ser un entero de al menos 1024 bytes")

    @classmethod
    def desde_entorno(cls, env: Mapping[str, str]) -> "ConfigPuerto":
        crudo = (env.get("JAX_FARO_SOCKET_DIR") or "").strip()
        if not crudo:
            raise ConfigFaroInvalida("JAX_FARO_SOCKET_DIR no esta definida: sin ella el Puerto no arranca")
        tope = (env.get("JAX_FARO_MAX_MENSAJE") or "").strip()
        try:
            max_mensaje = int(tope) if tope else MAX_MENSAJE_POR_DEFECTO
        except ValueError as exc:
            raise ConfigFaroInvalida("JAX_FARO_MAX_MENSAJE no es un entero") from exc
        return cls(socket_dir=Path(crudo), max_mensaje=max_mensaje)
