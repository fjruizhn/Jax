from .models import EnforcementAssertion
from .errors import AssertionIntegrityError
def verify_enforcement_assertion_content(value):
    if not value.assertion_hash: raise AssertionIntegrityError("hash")
    return value
def deserialize_enforcement_assertion(data): raise AssertionIntegrityError("wire parser requires fixed composition adapter")
