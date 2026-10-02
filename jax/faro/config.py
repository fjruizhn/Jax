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
  JAX_FARO_PLUGINS         opcional, JSON: [{"nombre","ruta","sha_declarado"}]
  JAX_FARO_REF_FRESCURA    opcional, la ref contra la que se mide la frescura
                           (default `origin/main`; es un nombre de ref, no un dato de entorno)
"""
from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

REF_FRESCURA_POR_DEFECTO = "origin/main"
_RE_SHA = re.compile(r"^[0-9a-f]{40}$")
_RE_NOMBRE_PLUGIN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


class ConfigFaroInvalida(ValueError):
    """La configuracion falta o no es valida: el Faro no arranca."""


def sha_valido(sha: object) -> bool:
    return isinstance(sha, str) and _RE_SHA.fullmatch(sha) is not None


@dataclass(frozen=True)
class PluginFuente:
    """Un plugin instalado cuyas skills y agentes entran al paquete. Los plugins NO viven
    en `claude-skills`, asi que el SHA del paquete no los cubre: el operador declara su
    `sha_declarado` (el `gitCommitSha` de `installed_plugins.json`) y queda en el manifiesto."""
    nombre: str
    ruta: Path
    sha_declarado: str


@dataclass(frozen=True)
class ConfigFaro:
    repo: Path
    sha: str
    destino: Path
    plugins: tuple[PluginFuente, ...] = ()
    ref_frescura: str = REF_FRESCURA_POR_DEFECTO

    def __post_init__(self) -> None:
        if not sha_valido(self.sha):
            raise ConfigFaroInvalida(f"JAX_FARO_SHA tiene que ser un SHA completo de 40 hex, no {self.sha!r}")
        for p in self.plugins:
            if not _RE_NOMBRE_PLUGIN.fullmatch(p.nombre):
                raise ConfigFaroInvalida(f"nombre de plugin invalido: {p.nombre!r}")
            if not sha_valido(p.sha_declarado):
                raise ConfigFaroInvalida(f"el plugin {p.nombre!r} necesita un sha_declarado de 40 hex")
        if len({p.nombre for p in self.plugins}) != len(self.plugins):
            raise ConfigFaroInvalida("hay plugins con el mismo nombre")

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

        plugins: tuple[PluginFuente, ...] = ()
        crudo = (env.get("JAX_FARO_PLUGINS") or "").strip()
        if crudo:
            try:
                lista = json.loads(crudo)
                plugins = tuple(PluginFuente(str(d["nombre"]), Path(d["ruta"]), str(d["sha_declarado"])) for d in lista)
            except (ValueError, TypeError, KeyError) as exc:
                raise ConfigFaroInvalida(f"JAX_FARO_PLUGINS no es una lista JSON valida: {type(exc).__name__}") from exc
        return cls(
            repo=ruta_absoluta("JAX_FARO_REPO"),
            sha=pedir("JAX_FARO_SHA"),
            destino=ruta_absoluta("JAX_FARO_ECOSISTEMA_DIR"),
            plugins=plugins,
            ref_frescura=(env.get("JAX_FARO_REF_FRESCURA") or "").strip() or REF_FRESCURA_POR_DEFECTO,
        )
