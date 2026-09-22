# Trusted composition map

- B4 root: `policy/authority_ledger/trusted_root.py`.
- B4 checkpoint: `policy/authority_ledger/trusted_checkpoint.py`.
- B6 approvers: `policy/execution_control/adapters/trusted_approver.py`.
- B6 execution: `policy/execution_control/service.py`.
- B7 composition: `las_manos/server.py::_configure_b7_trusted_runtime` owns EvidenceStore, ImplementationIdentity provider, RuntimeEvidenceRecorder, DB inspector, and worker-result ingester.

This map does not expose provenance internals.
