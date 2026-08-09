"""
Descifra en memoria las API keys de proveedor que llegan cifradas (Fernet)
desde /etc/jax/.env via EnvironmentFile de systemd.

Espejo minimo de jax-platform/backend/crypto_secrets.py — ese modulo cifra
las keys (sync bidireccional BD->.env) pero vive en otro repo/venv, asi que
los procesos de JAX (worker, las_manos) necesitan su propia copia liviana
para poder leerlas. Tolerante con keys legacy en texto plano: si el valor
no es un token Fernet valido, se devuelve tal cual.

En memoria de Jairo Urbina.
"""
import os
from abc import ABC, abstractmethod

from cryptography.fernet import Fernet, InvalidToken


class KeyProvider(ABC):
    """Interfaz para obtener la llave maestra de cifrado. El dia que se
    mueva a un KMS/Vault, se cambia la implementacion instanciada en
    _key_provider — ningun otro modulo vuelve a leer FERNET_KEY directo.
    Ver jax-platform/docs/fase1-credenciales-diseno.md (B1.3)."""

    @abstractmethod
    def get_master_key(self) -> bytes: ...


class EnvKeyProvider(KeyProvider):
    """Unica implementacion de esta fase. NO resuelve R2 (FERNET_KEY
    co-ubicada con lo que cifra en /etc/jax/.env) — deuda abierta."""

    def get_master_key(self) -> bytes:
        return os.environ.get("FERNET_KEY", "").encode()


_key_provider: KeyProvider = EnvKeyProvider()

PROVIDER_ENV_KEYS = [
    "OPENAI_API_KEY",
    "DEEPSEEK_API_KEY",
    "GEMINI_API_KEY",
    "KIMI_API_KEY",
    "ZAI_API_KEY",
    "ZHIPU_API_KEY",
]


def _get_fernet() -> "Fernet | None":
    key = _key_provider.get_master_key()
    if not key:
        return None
    return Fernet(key)


def decrypt_secret(value: str) -> str:
    """Descifra un valor Fernet. Si no es un token valido (key legacy en
    texto plano) o falta FERNET_KEY, devuelve el valor tal cual."""
    fernet = _get_fernet()
    if not fernet or not value:
        return value
    try:
        return fernet.decrypt(value.encode()).decode()
    except InvalidToken:
        return value


def decrypt_provider_keys_in_env() -> None:
    """Descifra en memoria (os.environ) las API keys de proveedor que
    systemd cargo desde /etc/jax/.env via EnvironmentFile. Debe llamarse
    antes de que cualquier modulo lea os.getenv()/os.environ para estas
    variables. No modifica el archivo en disco."""
    for k in PROVIDER_ENV_KEYS:
        v = os.environ.get(k)
        if v:
            os.environ[k] = decrypt_secret(v)
