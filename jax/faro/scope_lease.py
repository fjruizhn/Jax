"""Leases persistentes por UID para el ciclo de vida R-2 del Faro.

El directorio de leases es parte del limite de confianza: ya debe existir, ser
privado del usuario del servicio y no puede ser un enlace simbolico.  Cada
operacion abre el directorio sin seguir enlaces y trabaja relativa a su
descriptor; asi, ni el nombre de un UID ni una sustitucion de ruta pueden sacar
al servicio de ese directorio.

Un lease es una prueba durable de que un intento posee el nombre de unidad de un
UID.  Cualquier archivo inesperado, metadato inseguro o JSON que no cumpla el
contrato vuelve la operacion inconclusa: no se libera ni se reutiliza el UID.
"""
from __future__ import annotations

import contextlib
import enum
import fcntl
import hashlib
import json
import os
import re
import stat
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


_VERSION = 3
_MODO_DIRECTORIO = 0o700
_MODO_ARCHIVO = 0o600
_MAX_ARCHIVO_BYTES = 16 * 1024
_UID_MAX = 2 ** 32 - 2
_RE_UID_NOMBRE = re.compile(r"^[1-9][0-9]*\.json$")


class LeaseStoreError(RuntimeError):
    """El almacenamiento durable de leases no permite concluir con seguridad."""


class LeaseStoreConfigurationError(LeaseStoreError):
    """El directorio de leases no cumple el contrato de seguridad."""


class LeaseStoreConflict(LeaseStoreError):
    """Ya hay un lease durable para el UID solicitado."""


class LeaseStoreNotFound(LeaseStoreError):
    """No existe un lease durable para el UID solicitado."""


class LeaseStoreCorrupt(LeaseStoreError):
    """Un lease o una entrada del directorio no cumple el esquema cerrado."""


class LeaseStage(str, enum.Enum):
    """Punto durable del protocolo externo de systemd-run."""

    PREPARED = "PREPARED"
    REQUESTED = "REQUESTED"
    ACCEPTED = "ACCEPTED"
    CLIENT_CLOSED = "CLIENT_CLOSED"


@dataclass(frozen=True)
class ScopeLease:
    """El único contenido aceptado en el archivo durable de un UID."""

    uid: int
    attempt_id: str
    unit: str
    stage: LeaseStage = LeaseStage.PREPARED
    cgroup_path: str | None = None
    cgroup_dev: int | None = None
    cgroup_ino: int | None = None

    def __post_init__(self) -> None:
        _validar_uid(self.uid)
        _validar_texto("attempt_id", self.attempt_id)
        _validar_texto("unit", self.unit)
        if not isinstance(self.stage, LeaseStage):
            raise ValueError("stage no pertenece al conjunto cerrado de etapas")
        if self.unit != _unidad_esperada(self.uid, self.attempt_id):
            raise ValueError("unit no corresponde al uid y attempt_id del lease")
        evidencia = (self.cgroup_path, self.cgroup_dev, self.cgroup_ino)
        if self.stage in (LeaseStage.ACCEPTED, LeaseStage.CLIENT_CLOSED):
            if (not isinstance(self.cgroup_path, str) or not self.cgroup_path.startswith("/")
                    or "\x00" in self.cgroup_path or any(p in ("", ".", "..") for p in self.cgroup_path.split("/")[1:])
                    or isinstance(self.cgroup_dev, bool) or not isinstance(self.cgroup_dev, int) or self.cgroup_dev < 0
                    or isinstance(self.cgroup_ino, bool) or not isinstance(self.cgroup_ino, int) or self.cgroup_ino <= 0):
                raise ValueError("evidencia de cgroup inválida o ausente")
        elif any(value is not None for value in evidencia):
            raise ValueError("PREPARED/REQUESTED no pueden declarar evidencia kernel")

    def documento(self) -> dict[str, object]:
        """Proyección canónica y versionada que se persiste como JSON."""
        return {
            "attempt_id": self.attempt_id,
            "schema_version": _VERSION,
            "uid": self.uid,
            "unit": self.unit,
            "stage": self.stage.value,
            "cgroup_path": self.cgroup_path,
            "cgroup_dev": self.cgroup_dev,
            "cgroup_ino": self.cgroup_ino,
        }


def _validar_uid(uid: object) -> int:
    if isinstance(uid, bool) or not isinstance(uid, int) or not 1 <= uid <= _UID_MAX:
        raise ValueError(f"uid tiene que ser un entero entre 1 y {_UID_MAX}")
    return uid


def _validar_texto(nombre: str, valor: object) -> str:
    if (not isinstance(valor, str) or not valor or len(valor) > 255
            or "\x00" in valor or "\n" in valor or "\r" in valor):
        raise ValueError(f"{nombre} tiene que ser texto no vacio, de hasta 255 caracteres y sin NUL ni saltos")
    return valor


def _unidad_esperada(uid: int, attempt_id: str) -> str:
    sufijo = attempt_id if re.fullmatch(r"[0-9a-f]{32}", attempt_id) else hashlib.sha256(attempt_id.encode()).hexdigest()[:32]
    return f"faro-jaula-u{uid}-{sufijo}.service"


def _nombre(uid: int) -> str:
    return f"{_validar_uid(uid)}.json"


def _cargar_objeto(texto: str) -> object:
    def sin_duplicados(pares: list[tuple[str, object]]) -> dict[str, object]:
        resultado: dict[str, object] = {}
        for clave, valor in pares:
            if clave in resultado:
                raise LeaseStoreCorrupt(f"JSON de lease contiene clave duplicada: {clave!r}")
            resultado[clave] = valor
        return resultado

    try:
        return json.loads(texto, object_pairs_hook=sin_duplicados)
    except LeaseStoreCorrupt:
        raise
    except (TypeError, ValueError, UnicodeDecodeError) as exc:
        raise LeaseStoreCorrupt("JSON de lease ilegible") from exc


def _lease_desde_documento(documento: object, *, nombre: str) -> ScopeLease:
    if not isinstance(documento, dict) or set(documento) != {
            "schema_version", "uid", "attempt_id", "unit", "stage", "cgroup_path", "cgroup_dev", "cgroup_ino"}:
        raise LeaseStoreCorrupt(f"esquema invalido en {nombre!r}")
    if (isinstance(documento["schema_version"], bool) or not isinstance(documento["schema_version"], int)
            or documento["schema_version"] != _VERSION):
        raise LeaseStoreCorrupt(f"version de schema invalida en {nombre!r}")
    try:
        lease = ScopeLease(uid=documento["uid"], attempt_id=documento["attempt_id"], unit=documento["unit"],
                           stage=LeaseStage(documento["stage"]), cgroup_path=documento["cgroup_path"],
                           cgroup_dev=documento["cgroup_dev"], cgroup_ino=documento["cgroup_ino"])
    except ValueError as exc:
        raise LeaseStoreCorrupt(f"contenido invalido en {nombre!r}") from exc
    if nombre != _nombre(lease.uid):
        raise LeaseStoreCorrupt(f"nombre {nombre!r} no corresponde al uid del lease")
    return lease


class LeaseStore:
    """Almacenamiento local, durable y fail-closed de :class:`ScopeLease`.

    El directorio se valida al construir el store y antes de cada operacion. Los
    bloqueos ``flock`` sobre el descriptor del directorio serializan instancias
    cooperantes y hacen que ``release`` no pueda borrar un lease nuevo tras leer
    uno antiguo del mismo UID.
    """

    def __init__(self, directory: str | os.PathLike[str]) -> None:
        self._directory = Path(directory)
        if not self._directory.is_absolute():
            raise LeaseStoreConfigurationError("el directorio de leases tiene que ser una ruta absoluta")
        with self._directory_abierto(exclusivo=False):
            pass

    @property
    def directory(self) -> Path:
        """Ruta configurada; las operaciones usan siempre un descriptor seguro."""
        return self._directory

    @contextlib.contextmanager
    def _directory_abierto(self, *, exclusivo: bool) -> Iterator[int]:
        banderas = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
        try:
            descriptor = os.open(self._directory, banderas)
        except OSError as exc:
            raise LeaseStoreConfigurationError(f"no se puede abrir el directorio de leases: {self._directory}") from exc
        try:
            self._validar_directorio(descriptor)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX if exclusivo else fcntl.LOCK_SH)
            except OSError as exc:
                raise LeaseStoreError("no se pudo bloquear el directorio de leases") from exc
            try:
                self._validar_directorio(descriptor)
                yield descriptor
            finally:
                with contextlib.suppress(OSError):
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    @staticmethod
    def _validar_directorio(descriptor: int) -> None:
        datos = os.fstat(descriptor)
        if not stat.S_ISDIR(datos.st_mode):
            raise LeaseStoreConfigurationError("la ruta de leases no es un directorio real")
        if datos.st_uid != os.geteuid():
            raise LeaseStoreConfigurationError("el directorio de leases no pertenece al usuario del servicio")
        if stat.S_IMODE(datos.st_mode) != _MODO_DIRECTORIO:
            raise LeaseStoreConfigurationError("el directorio de leases tiene que tener modo 0700 exacto")

    @staticmethod
    def _validar_archivo(descriptor: int, *, nombre: str) -> os.stat_result:
        datos = os.fstat(descriptor)
        if not stat.S_ISREG(datos.st_mode):
            raise LeaseStoreCorrupt(f"{nombre!r} no es un archivo regular")
        if datos.st_uid != os.geteuid():
            raise LeaseStoreCorrupt(f"{nombre!r} no pertenece al usuario del servicio")
        if stat.S_IMODE(datos.st_mode) != _MODO_ARCHIVO:
            raise LeaseStoreCorrupt(f"{nombre!r} tiene que tener modo 0600 exacto")
        if datos.st_nlink != 1:
            raise LeaseStoreCorrupt(f"{nombre!r} tiene enlaces adicionales")
        if not 0 < datos.st_size <= _MAX_ARCHIVO_BYTES:
            raise LeaseStoreCorrupt(f"{nombre!r} tiene un tamano invalido")
        return datos

    @classmethod
    def _leer_archivo(cls, directorio_fd: int, nombre: str) -> tuple[ScopeLease, os.stat_result]:
        try:
            descriptor = os.open(nombre, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=directorio_fd)
        except FileNotFoundError as exc:
            raise LeaseStoreNotFound(f"no existe lease para {nombre!r}") from exc
        except OSError as exc:
            raise LeaseStoreCorrupt(f"no se puede abrir {nombre!r} sin seguir enlaces") from exc
        try:
            datos = cls._validar_archivo(descriptor, nombre=nombre)
            fragmentos: list[bytes] = []
            faltan = datos.st_size + 1
            while faltan:
                parte = os.read(descriptor, faltan)
                if not parte:
                    break
                fragmentos.append(parte)
                faltan -= len(parte)
            contenido = b"".join(fragmentos)
            if len(contenido) != datos.st_size:
                raise LeaseStoreCorrupt(f"{nombre!r} cambio mientras se leia")
            try:
                texto = contenido.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise LeaseStoreCorrupt(f"{nombre!r} no es UTF-8") from exc
            return _lease_desde_documento(_cargar_objeto(texto), nombre=nombre), datos
        finally:
            os.close(descriptor)

    @staticmethod
    def _fsync_directorio(descriptor: int) -> None:
        try:
            os.fsync(descriptor)
        except OSError as exc:
            raise LeaseStoreError("no se pudo confirmar el directorio de leases en disco") from exc

    @staticmethod
    def _escribir_todo(descriptor: int, contenido: bytes) -> None:
        vista = memoryview(contenido)
        while vista:
            escritos = os.write(descriptor, vista)
            if escritos <= 0:
                raise LeaseStoreError("escritura incompleta del lease")
            vista = vista[escritos:]

    def acquire(self, uid: int, attempt_id: str, unit: str) -> ScopeLease:
        """Crea de forma exclusiva y durable el lease de ``uid``.

        Si ya existe algo bajo ese UID, incluso un archivo corrupto, no se toma
        el UID: el llamador debe reconciliar antes de reintentarlo.
        """
        lease = ScopeLease(uid=uid, attempt_id=attempt_id, unit=unit)
        nombre = _nombre(lease.uid)
        contenido = (json.dumps(lease.documento(), ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode()
        with self._directory_abierto(exclusivo=True) as directorio_fd:
            try:
                descriptor = os.open(
                    nombre, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                    _MODO_ARCHIVO, dir_fd=directorio_fd)
            except FileExistsError as exc:
                raise LeaseStoreConflict(f"ya existe un lease para uid {lease.uid}") from exc
            except OSError as exc:
                raise LeaseStoreError(f"no se pudo crear lease para uid {lease.uid}") from exc
            try:
                self._escribir_todo(descriptor, contenido)
                os.fsync(descriptor)
            except OSError as exc:
                raise LeaseStoreError(f"no se pudo confirmar lease para uid {lease.uid} en disco") from exc
            finally:
                os.close(descriptor)
            self._fsync_directorio(directorio_fd)
        return lease

    def acquire_instance(self) -> int:
        """Toma el lock exclusivo de instancia y lo conserva hasta cerrar el descriptor."""
        with self._directory_abierto(exclusivo=True) as directorio_fd:
            fd: int | None = None
            transferido = False
            try:
                fd = os.open(".instance.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC,
                             _MODO_ARCHIVO, dir_fd=directorio_fd)
                datos = os.fstat(fd)
                if (not stat.S_ISREG(datos.st_mode) or datos.st_uid != os.geteuid()
                        or stat.S_IMODE(datos.st_mode) != _MODO_ARCHIVO or datos.st_nlink != 1):
                    raise LeaseStoreConfigurationError("lock de instancia inseguro")
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                transferido = True
                return fd
            except BlockingIOError as exc:
                raise LeaseStoreConflict("ya existe otra instancia activa del Faro") from exc
            except OSError as exc:
                raise LeaseStoreError("no se pudo adquirir lock exclusivo de instancia") from exc
            finally:
                if fd is not None and not transferido:
                    os.close(fd)

    def read(self, uid: int) -> ScopeLease:
        """Lee y valida el lease de ``uid`` sin seguir enlaces."""
        nombre = _nombre(uid)
        with self._directory_abierto(exclusivo=False) as directorio_fd:
            lease, _ = self._leer_archivo(directorio_fd, nombre)
        return lease

    def transition(self, uid: int, attempt_id: str, expected: LeaseStage, target: LeaseStage, *,
                   cgroup_path: str | None = None, cgroup_dev: int | None = None,
                   cgroup_ino: int | None = None) -> ScopeLease:
        """CAS durable del mismo attempt/etapa antes de cambiar el protocolo externo."""
        uid = _validar_uid(uid)
        _validar_texto("attempt_id", attempt_id)
        if not isinstance(expected, LeaseStage) or not isinstance(target, LeaseStage):
            raise ValueError("expected y target deben ser LeaseStage")
        permitidas = {
            LeaseStage.PREPARED: {LeaseStage.REQUESTED},
            LeaseStage.REQUESTED: {LeaseStage.ACCEPTED},
            LeaseStage.ACCEPTED: {LeaseStage.CLIENT_CLOSED},
            LeaseStage.CLIENT_CLOSED: set(),
        }
        if target not in permitidas[expected]:
            raise LeaseStoreConflict("transicion de lease no permitida")
        nombre = _nombre(uid)
        with self._directory_abierto(exclusivo=True) as directorio_fd:
            lease, antes = self._leer_archivo(directorio_fd, nombre)
            if lease.attempt_id != attempt_id or lease.stage is not expected:
                raise LeaseStoreConflict(f"el lease de uid {uid} ya no coincide con el attempt/etapa esperados")
            if target is LeaseStage.ACCEPTED:
                nuevo = ScopeLease(uid=uid, attempt_id=attempt_id, unit=lease.unit, stage=target,
                                   cgroup_path=cgroup_path, cgroup_dev=cgroup_dev, cgroup_ino=cgroup_ino)
            else:
                nuevo = ScopeLease(uid=uid, attempt_id=attempt_id, unit=lease.unit, stage=target,
                                   cgroup_path=lease.cgroup_path, cgroup_dev=lease.cgroup_dev,
                                   cgroup_ino=lease.cgroup_ino)
            contenido = (json.dumps(nuevo.documento(), ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode()
            temporal = f".{uid}.{attempt_id}.{os.getpid()}.{os.urandom(8).hex()}.tmp"
            try:
                fd = os.open(temporal, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                             _MODO_ARCHIVO, dir_fd=directorio_fd)
                try:
                    self._escribir_todo(fd, contenido)
                    os.fsync(fd)
                finally:
                    os.close(fd)
                actual = os.stat(nombre, dir_fd=directorio_fd, follow_symlinks=False)
                if (actual.st_dev, actual.st_ino, actual.st_nlink) != (antes.st_dev, antes.st_ino, antes.st_nlink):
                    raise LeaseStoreCorrupt(f"{nombre!r} cambio durante CAS")
                os.replace(temporal, nombre, src_dir_fd=directorio_fd, dst_dir_fd=directorio_fd)
                self._fsync_directorio(directorio_fd)
            except LeaseStoreError:
                with contextlib.suppress(FileNotFoundError):
                    os.unlink(temporal, dir_fd=directorio_fd)
                raise
            except OSError as exc:
                with contextlib.suppress(FileNotFoundError):
                    os.unlink(temporal, dir_fd=directorio_fd)
                raise LeaseStoreError(f"no se pudo persistir transicion del uid {uid}") from exc
        return nuevo

    def _abort_unsubmitted(self, uid: int, attempt_id: str) -> None:
        """CAS privado del REQUESTED no enviado, invocable solo por el owner vivo del ScopeManager.

        El store no ofrece un argumento booleano de caller que pueda falsificarse: ScopeManager
        comprueba identidad de tarea y que su sender aún no se invocó antes de llegar aquí.
        """
        uid = _validar_uid(uid)
        _validar_texto("attempt_id", attempt_id)
        nombre = _nombre(uid)
        with self._directory_abierto(exclusivo=True) as directorio_fd:
            lease, antes = self._leer_archivo(directorio_fd, nombre)
            if lease.attempt_id != attempt_id or lease.stage is not LeaseStage.REQUESTED:
                raise LeaseStoreConflict("abort no enviado no coincide con REQUESTED del attempt esperado")
            try:
                ahora = os.stat(nombre, dir_fd=directorio_fd, follow_symlinks=False)
            except OSError as exc:
                raise LeaseStoreCorrupt(f"no se pudo comprobar {nombre!r} antes de abortar") from exc
            if (ahora.st_dev, ahora.st_ino, ahora.st_nlink) != (antes.st_dev, antes.st_ino, antes.st_nlink):
                raise LeaseStoreCorrupt(f"{nombre!r} cambio antes del abort no enviado")
            try:
                os.unlink(nombre, dir_fd=directorio_fd)
            except OSError as exc:
                raise LeaseStoreError(f"no se pudo abortar REQUESTED no enviado para uid {uid}") from exc
            self._fsync_directorio(directorio_fd)

    def all(self) -> tuple[ScopeLease, ...]:
        """Devuelve todos los leases validados; una entrada inesperada falla cerrado."""
        with self._directory_abierto(exclusivo=False) as directorio_fd:
            try:
                nombres = os.listdir(directorio_fd)
            except OSError as exc:
                raise LeaseStoreError("no se pudo listar el directorio de leases") from exc
            leases: list[ScopeLease] = []
            for nombre in nombres:
                if nombre == ".instance.lock":
                    continue
                if not _RE_UID_NOMBRE.fullmatch(nombre):
                    raise LeaseStoreCorrupt(f"entrada inesperada en directorio de leases: {nombre!r}")
                texto_uid = nombre.removesuffix(".json")
                if str(int(texto_uid)) != texto_uid or not 1 <= int(texto_uid) <= _UID_MAX:
                    raise LeaseStoreCorrupt(f"nombre de lease invalido: {nombre!r}")
                lease, _ = self._leer_archivo(directorio_fd, nombre)
                leases.append(lease)
        return tuple(sorted(leases, key=lambda lease: lease.uid))

    def release(self, uid: int, attempt_id: str, *, cgroup_verified: bool = False) -> None:
        """Borra un lease solo si sigue perteneciendo al ``attempt_id`` exacto.

        La exclusión cooperante sobre el directorio evita que un ``release``
        atrasado borre un lease que otro intento acaba de adquirir.
        """
        uid = _validar_uid(uid)
        _validar_texto("attempt_id", attempt_id)
        nombre = _nombre(uid)
        with self._directory_abierto(exclusivo=True) as directorio_fd:
            lease, antes = self._leer_archivo(directorio_fd, nombre)
            if lease.attempt_id != attempt_id:
                raise LeaseStoreConflict(f"el lease de uid {uid} pertenece a otro intento")
            if lease.stage in (LeaseStage.REQUESTED, LeaseStage.ACCEPTED):
                raise LeaseStoreConflict(f"{lease.stage.value} no se libera sin cierre causal del cliente")
            if lease.stage is LeaseStage.CLIENT_CLOSED and cgroup_verified is not True:
                raise LeaseStoreConflict("CLIENT_CLOSED requiere prueba kernel verificada por ScopeManager")
            try:
                ahora = os.stat(nombre, dir_fd=directorio_fd, follow_symlinks=False)
            except FileNotFoundError as exc:
                raise LeaseStoreCorrupt(f"{nombre!r} desaparecio antes de liberar") from exc
            except OSError as exc:
                raise LeaseStoreCorrupt(f"no se pudo comprobar {nombre!r} antes de liberar") from exc
            if (ahora.st_dev, ahora.st_ino, ahora.st_nlink) != (antes.st_dev, antes.st_ino, antes.st_nlink):
                raise LeaseStoreCorrupt(f"{nombre!r} cambio antes de liberar")
            try:
                os.unlink(nombre, dir_fd=directorio_fd)
            except OSError as exc:
                raise LeaseStoreError(f"no se pudo liberar lease para uid {uid}") from exc
            self._fsync_directorio(directorio_fd)
