from .models import EnforcementObservation
from .errors import ObservationIntegrityError
def verify_enforcement_observation_content(value):
    if not value.observation_hash: raise ObservationIntegrityError("hash")
    return value
def deserialize_enforcement_observation(data): raise ObservationIntegrityError("wire parser requires fixed composition adapter")
