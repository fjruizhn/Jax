"""UNA constante de exclusiones, compartida (Enmienda 1 del pre-registro, reglas 1 y 6).

Directorios de estado y dependencias que cambian solos (el daemon de Ruflo escribe
en `.claude-flow/` sin parar) y no son afirmaciones sobre el trabajo. La usan el
recorrido del árbol en disco y la huella de `git status`. Vive en su propio módulo,
sin subprocesos, para que el escáner de CLIs de policy/tests no lo confunda con un
lanzador.
"""
from __future__ import annotations

EXCLUSIONES = frozenset({".git", ".claude-flow", "node_modules", ".venv", "__pycache__"})
