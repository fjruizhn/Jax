from policy.decision_record.ids import new_decision_id

def new_execution_id() -> str:
    return new_decision_id()

def new_authorization_id() -> str:
    return new_decision_id()

def new_human_approval_id() -> str:
    return new_decision_id()

def new_dry_run_artifact_id() -> str:
    return new_decision_id()
