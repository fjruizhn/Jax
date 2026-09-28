"""El token llega a git por GIT_ASKPASS leyendo una variable del entorno DEL SUBPROCESO:
nunca en argv, nunca en la URL, nunca en .git/config (el clon entra a la jaula)."""
from __future__ import annotations

import os
from pathlib import Path

_ASKPASS = '#!/bin/sh\ncase "$1" in *Username*) echo x-access-token;; *) printf %s "$JAX_GIT_TOKEN_EFIMERO";; esac\n'


def entorno_git(token: str, directorio: Path) -> dict[str, str]:
    script = directorio / "askpass.sh"
    script.write_text(_ASKPASS)
    script.chmod(0o700)
    env = {k: v for k, v in os.environ.items() if not k.startswith("JAX_GITHUB")}
    env.update(GIT_ASKPASS=str(script), GIT_TERMINAL_PROMPT="0", JAX_GIT_TOKEN_EFIMERO=token)
    return env
