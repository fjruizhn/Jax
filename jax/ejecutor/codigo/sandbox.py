"""Sandbox de las instalaciones de dependencias de una misión de código (spec 2026-09-28 v1.2,
§3.1; ruling del controlador tras la re-revisión de escalón 3, BLOCK-1).

pip, npm y composer ejecutan código de terceros (un `.pth` de una rueda, un `setup.py` que una
línea `--no-binary` fuerza a construir, lo que venga) y el proceso que prepara la misión es
`jaxsvc`, que puede leer `/etc/jax/.env`. Filtrar el entorno no alcanza: el código de terceros
lee archivos. Por eso la instalación corre en bwrap por LISTA BLANCA:

- `--unshare-all --share-net`: todos los espacios de nombres nuevos salvo la red (hace falta
  para bajar paquetes). `--die-with-parent --new-session`.
- `--clearenv` y solo PATH, HOME=/tmp/h y LANG=C.UTF-8.
- De solo lectura, SOLO: /usr, /lib, /lib64, /bin, /sbin (enlaces simbólicos como enlaces,
  directorios reales montados), /etc/ssl, /etc/ca-certificates, /etc/resolv.conf, /etc/hosts,
  /etc/nsswitch.conf, /etc/passwd, /etc/group; el prefijo de node si se da (en producción el
  de JAX_EJECUTOR_NODE_BIN); y las COPIAS de los archivos de dependencias hechas antes, fuera.
- Escribible, SOLO `deps/` de la misión. /tmp es un tmpfs propio.

Nada de /etc/jax, /var/lib/jaxsvc, /srv, /home ni el directorio de misiones: ni se monta ni se
acepta como origen de un montaje (`ValueError`)."""
from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

PROHIBIDAS = (Path("/etc/jax"), Path("/var/lib/jaxsvc"), Path("/srv"), Path("/home"))
_SISTEMA_DIRS = (Path("/usr"), Path("/lib"), Path("/lib64"), Path("/bin"), Path("/sbin"))
_SISTEMA_ETC = (Path("/etc/ssl"), Path("/etc/ca-certificates"), Path("/etc/resolv.conf"), Path("/etc/hosts"),
                Path("/etc/nsswitch.conf"), Path("/etc/passwd"), Path("/etc/group"))
_PATH_BASE = "/usr/local/bin:/usr/bin:/bin"
HOME = "/tmp/h"


def _bajo(ruta: Path, raiz: Path) -> bool:
    return ruta == raiz or raiz in ruta.parents


def _permitida(ruta: Path) -> Path:
    """La ruta que SE MONTA (no otra): ni dentro de una prohibida ni conteniéndola (montar
    `/etc` expondría `/etc/jax`; montar `/` lo expondría todo). Se mira también el realpath."""
    ruta = Path(os.path.abspath(ruta))
    real = Path(os.path.realpath(ruta))
    for prohibida in PROHIBIDAS:
        for r in (ruta, real):
            if _bajo(r, prohibida) or _bajo(prohibida, r):
                raise ValueError(f"montaje_prohibido: {ruta}")
    return ruta


def _sistema() -> list[str]:
    argv: list[str] = []
    for d in _SISTEMA_DIRS:
        if d.is_symlink():
            argv += ["--symlink", os.readlink(d), str(d)]
        elif d.is_dir():
            argv += ["--ro-bind", str(d), str(d)]
    for e in _SISTEMA_ETC:
        if e.exists():
            argv += ["--ro-bind", str(e), str(e)]
    return argv


def argv_sandbox(comando: Sequence[str], *, deps: Path, cwd: Path,
                 solo_lectura: Sequence[tuple[Path, Path]] = (), node_bin: Path | None = None) -> list[str]:
    """Pura (salvo mirar qué de /usr, /lib… es enlace o directorio): el argv completo de bwrap.
    `solo_lectura` son pares (origen, destino); el destino puede estar dentro de `deps` (bwrap
    monta el archivo encima: la instalación no puede reescribir su propio lockfile)."""
    deps = Path(os.path.abspath(deps))
    cwd = Path(os.path.abspath(cwd))
    if not _bajo(cwd, deps):
        raise ValueError(f"cwd_fuera_de_deps: {cwd}")
    path = _PATH_BASE
    argv = ["bwrap", "--unshare-all", "--share-net", "--die-with-parent", "--new-session", "--clearenv"]
    argv += _sistema()
    argv += ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--dir", HOME]
    if node_bin is not None:
        # MINOR-B: se valida lo que efectivamente se monta, el prefijo (padre de node_bin).
        node_bin = Path(os.path.abspath(node_bin))
        prefijo = _permitida(node_bin.parent)
        argv += ["--ro-bind-try", str(prefijo), str(prefijo)]
        path = f"{node_bin}:{path}"
    argv += ["--bind", str(deps), str(deps)]
    for origen, destino in solo_lectura:
        argv += ["--ro-bind", str(_permitida(Path(origen))), str(Path(os.path.abspath(destino)))]
    argv += ["--setenv", "PATH", path, "--setenv", "HOME", HOME, "--setenv", "LANG", "C.UTF-8",
             "--chdir", str(cwd), "--", *comando]
    return argv
