"""Block 6 cierra la ruta directa de Hyde: `_invoke_hyde` no lanza Claude, lanza el error real.

Antes (hallado en la auditoria de T16, MAJOR-1) lanzaba `RuntimeError` con el nombre de la
clase solo en el texto, y debajo dejaba ~60 lineas inalcanzables que armaban un `claude`
con `--allowedTools Bash`. Una futura edicion que quitara el `raise` habria resucitado ese
camino sin que ninguna prueba lo notara.

Puros: se sustituyen por falsos `cli_sandbox.ejecutar` (el nucleo que lanza el subproceso) y
`asyncio.create_subprocess_exec`, y se exige que nunca se llamen. T16 retiro tambien
`hyde_sandbox.run_sandboxed_claude`: ya no hay funcion que lance `claude` con ese confinamiento.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest


def test_invoke_hyde_lanza_la_clase_real_y_ningun_subproceso_se_lanza():
    from policy.execution_control.errors import (
        DirectHydeGovernedExecutionForbiddenError, ExecutionControlError)
    import cli_sandbox
    from jacobs import executor
    ejecutar, subproceso = AsyncMock(), AsyncMock()
    with patch.object(cli_sandbox, "ejecutar", ejecutar), \
         patch("asyncio.create_subprocess_exec", subproceso):
        with pytest.raises(DirectHydeGovernedExecutionForbiddenError) as e:
            asyncio.run(executor._invoke_hyde(SimpleNamespace(model="m"), "prompt", 5))
    assert isinstance(e.value, ExecutionControlError)
    ejecutar.assert_not_called()
    subproceso.assert_not_called()


def test_run_sandboxed_claude_ya_no_existe():
    import hyde_sandbox
    assert not hasattr(hyde_sandbox, "run_sandboxed_claude")


def test_el_cuerpo_que_armaba_el_claude_ya_no_existe():
    from jacobs import executor
    assert not hasattr(executor, "HYDE_CLI_PATH")
    assert not hasattr(executor, "HYDE_MAX_PROMPT_CHARS")
    assert not hasattr(executor, "_HYDE_SYSTEM_PROMPT")
