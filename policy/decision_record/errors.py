"""Typed failures for immutable Block 5 decision records."""


class DecisionRecordError(Exception):
    pass


class InvalidDecisionInputError(DecisionRecordError):
    pass


class InvalidDecisionFactError(InvalidDecisionInputError):
    pass


class DecisionContractError(DecisionRecordError):
    pass


class UnverifiedAuthorityEvaluationError(DecisionRecordError):
    pass


class DecisionRecordIntegrityError(DecisionRecordError):
    pass


class DecisionIdConflictError(DecisionRecordError):
    pass


class DecisionStorageError(DecisionRecordError):
    pass


class DecisionEvidenceUnavailableError(DecisionRecordError):
    pass


class DecisionReplayError(DecisionRecordError):
    pass


class HistoricalAuthorityUnavailableError(DecisionReplayError):
    pass


class HistoricalCheckpointMismatchError(DecisionReplayError):
    pass


class UnsupportedResolverError(DecisionReplayError):
    pass


class UnsupportedDecisionSchemaError(DecisionReplayError):
    pass


class DecisionInvariantError(DecisionRecordError):
    pass
