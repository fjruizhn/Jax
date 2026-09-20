from policy.execution_control.dry_run import DryRunArtifact
def test_dry_run_artifact_exists(): assert DryRunArtifact.__dataclass_params__.frozen
