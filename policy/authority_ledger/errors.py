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


class LedgerCheckpointError(LedgerIntegrityError):
    """El evento quedó escrito en el ledger pero su checkpoint externo no.

    El evento no se considera aceptado hasta que el checkpoint quedó escrito;
    este error nombra al evento huérfano y la reconciliación disponible
    (``reanchor_authority_checkpoint``): re-anclar el checkpoint al head
    existente tras verificarlo."""


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
