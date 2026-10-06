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
            git_dir = target if target.is_absolute() else (workspace_root / target).resolve()
            common_dir_file = git_dir / "commondir"
            if common_dir_file.is_file():
                common_dir = Path(common_dir_file.read_text(encoding="utf-8").strip())
                return common_dir if common_dir.is_absolute() else (git_dir / common_dir).resolve()
            return git_dir
    raise OSError(f"no se pudo resolver el directorio Git de {workspace_root}")


def abrir_project_tree_lock(workspace_root: Path) -> int:
    """Abre el git-dir común para flock, sin un inode de archivo con dueño fijo.

    El worker y E2A corren como usuarios distintos. El directorio .git ya
    es accesible por ambos; flock sobre su fd evita que quien cree primero un
    archivo determine si el otro actor puede abrirlo.
    """
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    return os.open(_git_directory(workspace_root), flags)


@contextmanager
def project_tree_lock(workspace_root: Path) -> Iterator[None]:
    """Exclusión mutua por git-dir para leer/escribir/mover el árbol de proyectos."""
    fd = abrir_project_tree_lock(workspace_root)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)
