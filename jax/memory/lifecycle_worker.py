"""B9 lifecycle/reconciliation job entry point.

The production implementation intentionally performs no implicit repair.  A
projection mismatch is reported for controlled reconciliation; automatic
repair would hide a history/projection divergence.
"""
from __future__ import annotations

import asyncio
import logging


async def main() -> int:
    logging.getLogger(__name__).info(
        "B9 lifecycle worker started: reconciliation requires the B9 persistent store composition"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
