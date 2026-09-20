"""Typed failures for the Block 4 authority ledger boundary."""
from __future__ import annotations


class AuthorityLedgerError(Exception):
    """Base class for authority-ledger failures."""


class LedgerIntegrityError(AuthorityLedgerError):
    pass


class LedgerRollbackError(LedgerIntegrityError):
    pass


class UnanchoredLedgerHeadError(LedgerIntegrityError):
    pass


class TrustedRootMismatchError(LedgerIntegrityError):
    pass


class AuthorityEventValidationError(AuthorityLedgerError):
    pass


class AuthorityAuthenticationError(AuthorityLedgerError):
    pass


class AuthorityStateError(AuthorityLedgerError):
    pass


class OverlayConflictError(AuthorityStateError):
    pass


class OverlayApplicabilityIndeterminateError(AuthorityStateError):
    pass
