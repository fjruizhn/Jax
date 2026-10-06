"""Candado entre herramientas de archivo y movimientos de carpetas a proyectos/."""
from __future__ import annotations

import fcntl
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


def _git_directory(workspace_root: Path) -> Path:
    marker = workspace_root / ".git"
    if marker.is_dir():
        return marker
    if marker.is_file():
        text = marker.read_text(encoding="utf-8").strip()
        if text.startswith("gitdir:"):
            target = Path(text.split(":", 1)[1].strip())
            return target if target.is_absolute() else (workspace_root / target).resolve()
    raise OSError(f"no se pudo resolver el directorio Git de {workspace_root}")


def abrir_project_tree_lock(workspace_root: Path) -> int:
    """Abre el archivo estable de flock común a todos los worktrees del repo."""
    ruta = _git_directory(workspace_root) / "project-tree.lock"
    return os.open(ruta, os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0), 0o660)


@contextmanager
def project_tree_lock(workspace_root: Path) -> Iterator[None]:
    """Exclusión mutua para leer/escribir rutas y mover carpetas a proyectos/."""
    fd = abrir_project_tree_lock(workspace_root)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)
