"""Block 6 cierra la ruta directa de Hyde: `_invoke_hyde` no lanza Claude, lanza el error real.

Antes (hallado en la auditoria de T16, MAJOR-1) lanzaba `RuntimeError` con el nombre de la
clase solo en el texto, y debajo dejaba ~60 lineas inalcanzables que armaban un `claude`
con `--allowedTools Bash`. Una futura edicion que quitara el `raise` habria resucitado ese
camino sin que ninguna prueba lo notara.

Puros: se sustituye `run_sandboxed_claude` por un falso y se exige que nunca se llame.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest


def test_invoke_hyde_lanza_la_clase_real_y_el_sandbox_nunca_se_llama():
    from policy.execution_control.errors import (
        DirectHydeGovernedExecutionForbiddenError, ExecutionControlError)
    falso = AsyncMock()
    from jacobs import executor
    with patch.object(executor, "run_sandboxed_claude", falso, create=True):
        with pytest.raises(DirectHydeGovernedExecutionForbiddenError) as e:
            asyncio.run(executor._invoke_hyde(SimpleNamespace(model="m"), "prompt", 5))
    assert isinstance(e.value, ExecutionControlError)
    falso.assert_not_called()


def test_el_cuerpo_que_armaba_el_claude_ya_no_existe():
    from jacobs import executor
    assert not hasattr(executor, "HYDE_CLI_PATH")
    assert not hasattr(executor, "HYDE_MAX_PROMPT_CHARS")
    assert not hasattr(executor, "_HYDE_SYSTEM_PROMPT")
