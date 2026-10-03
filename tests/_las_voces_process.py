"""Local process helper for the sync fixture and CLI exit-code check."""

from __future__ import annotations

import subprocess
from collections.abc import Sequence


def run_local(argv: Sequence[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, **kwargs)
