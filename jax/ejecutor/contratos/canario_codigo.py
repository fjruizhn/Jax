# jax/ejecutor/contratos/canario_codigo.py
"""Canario permanente de las reglas de código (spec 2026-09-28 §5.1, Principio VII: un freno
sin prueba no es freno). Corre dentro de `arranque.pruebas_reales()["codigo"]`, solo en un
turno de código y solo después de que el token de GitHub esté presente (arranque.py, Tarea 12).

Tres observaciones, todas como `axioma`:
1. El evento directo `git push origin HEAD`, por el gancho instalado, lo bloquea con su regla
   `codigo_git_push` (exit 2) -- la jaula no empuja; la entrega sí (spec §3.2), y ese "no" lo
   tiene que decir el gancho, no un acuerdo de caballeros.
2. El cerco (C3) corta la salida directa de la cuenta hacia GitHub: `git ls-remote` contra un
   repo real tiene que FALLAR por red. Si contesta (rc=0), el cerco está dejando pasar tráfico
   que sólo la entrega -- desde el espejo de jaxsvc, con el token -- debería poder hacer.
3. `reglas_diff.revisar` (C1 del DIFF en la entrega) bloquea un cambio bajo
   `.github/workflows/` con la regla `flujos_ci`; si no lo hace, ese freno dejó de existir.

Devuelve los Fallo; vacío = OK. No lanza: un contrato roto se reporta, no revienta el arranque.
"""
from __future__ import annotations

import json
import shlex

from jax.ejecutor.codigo import reglas_diff
from jax.ejecutor.codigo.diff import Cambio
from jax.ejecutor.contratos import cuenta_axioma
from jax.ejecutor.contratos.fallo import Fallo

_TOPE_S = 30
_MARCA_PUSH = 'regla="codigo_git_push"'
_REPO_DE_PRUEBA = "fjruizhn/jax-platform"
_LS_REMOTE = f"git ls-remote https://github.com/{_REPO_DE_PRUEBA}.git"


def _evento(comando: str) -> bytes:
    return json.dumps({"tool_name": "Bash", "tool_input": {"command": comando}}).encode()


async def verificar_codigo(c, *, correr=cuenta_axioma.correr_en_la_cuenta) -> tuple:
    fallos: list[Fallo] = []
    q = shlex.quote
    gancho = f"{q(str(c.lib / 'gancho.sh'))} {q(str(c.politica))}"

    rc, _, errores = await correr(c, gancho, entrada=_evento("git push origin HEAD"), tope_s=_TOPE_S)
    if rc != 2:
        fallos.append(Fallo("codigo", "push_no_bloqueado", (("rc", rc),)))
    elif _MARCA_PUSH.encode() not in errores:
        # Bloqueado, pero no por su regla: el gancho o la política están rotos.
        fallos.append(Fallo("codigo", "push_sin_su_regla", (("stderr", errores.decode(errors="replace").strip()),)))

    rc, _, _ = await correr(c, _LS_REMOTE, tope_s=_TOPE_S)
    if rc == 0:
        fallos.append(Fallo("codigo", "cerco_deja_salir_a_github"))

    violaciones = reglas_diff.revisar((Cambio(".github/workflows/x.yml", "M", None, (), ()),), _REPO_DE_PRUEBA)
    if "flujos_ci" not in {v.regla for v in violaciones}:
        fallos.append(Fallo("codigo", "c1_diff_no_bloquea"))

    return tuple(fallos)
