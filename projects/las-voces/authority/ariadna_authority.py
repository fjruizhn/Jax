"""Deterministic, fail-closed authority contract for proposed Ariadna.

This module has no runtime runner or reverse-sync operation. The only writer
uses a recoverable journal protocol; it does not pretend independent files are
one atomic filesystem transaction.
"""
from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
import subprocess
import tempfile
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Protocol

from jsonschema import Draft202012Validator, FormatChecker

PROJECT_ID = "las-voces"
ARIADNA_ID = "ariadna-project-manager"
FORBIDDEN = {
    "merge", "deploy", "production_mutation", "capability_grant", "runtime_execution",
    "authority_change", "authority_contract_edit", "verifier_replacement",
    "verifier_configuration_change", "policy_gate_disable", "self_activation",
    "self_approval", "bypass_human_required", "arbitrary_command_execution",
    "shell_execution",
}
TRANSITIONS = {"READY": {"IN_PROGRESS", "BLOCKED"}, "IN_PROGRESS": {"BLOCKED", "READY", "DONE"}, "BLOCKED": {"READY", "IN_PROGRESS"}}
TEST_MANIFEST_KIND = "JAX_TEST_EVIDENCE_MANIFEST"


class Verdict(str, Enum):
    ALLOW = "ALLOW"
    DENY = "DENY"
    HUMAN_REQUIRED = "HUMAN_REQUIRED"


@dataclass(frozen=True)
class Decision:
    verdict: Verdict
    reason: str


class AuthorityError(ValueError):
    """A deterministic contract rejection."""


class HumanRequired(AuthorityError):
    """A controlled execution/human authority record is needed externally."""


class CIExecutionVerifier(Protocol):
    """Adapter owned by the privileged composition boundary, never a request."""
    def execution_record(self, manifest: dict[str, Any]) -> bool | None: ...


class HumanAuthorityVerifier(Protocol):
    """Adapter owned by the privileged composition boundary, never a request."""
    def acceptance_record(self, record: dict[str, Any]) -> bool | None: ...


@dataclass(frozen=True)
class ValidatedEvidence:
    """Immutable snapshot of the exact evidence that produced authorization."""
    task_id: str
    implementation_commit_sha: str | None
    test_manifest_digest: str | None
    acceptance_record_digest: str | None
    pr_record_digest: str | None
    normalized_evidence_digest: str


@dataclass(frozen=True)
class EvidenceSnapshot:
    """One immutable local evidence object captured for one authorization."""
    ref: str
    value: dict[str, Any]
    sha256: str


class EvidenceSet:
    """The complete, de-duplicated local evidence closure for one request.

    A path is opened only by ``snapshot``.  Every later validation helper is
    deliberately passed a snapshot, never a repository path.
    """
    def __init__(self, root: Path):
        self.root = root
        self._by_ref: dict[str, EvidenceSnapshot] = {}

    def snapshot(self, ref: str) -> EvidenceSnapshot:
        if ref not in self._by_ref:
            self._by_ref[ref] = _evidence_document(self.root, ref)
        return self._by_ref[ref]

    def values(self) -> list[EvidenceSnapshot]:
        return list(self._by_ref.values())


def _load(path: Path) -> Any:
    try: return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc: raise AuthorityError(f"invalid canonical JSON: {path}") from exc


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha(value: bytes) -> str: return hashlib.sha256(value).hexdigest()
def _project_hash(path: Path) -> str: return _sha(path.read_bytes())


def _task(project: dict[str, Any], task_id: str) -> dict[str, Any]:
    if project.get("project", {}).get("id") != PROJECT_ID: raise AuthorityError("wrong canonical project_id")
    matches = [item for item in project.get("tasks", []) if item.get("id") == task_id]
    if len(matches) != 1: raise AuthorityError("unknown or ambiguous task")
    return matches[0]


def _safe_repo_file(root: Path, ref: str) -> Path:
    if not isinstance(ref, str) or not ref or ref.startswith(("/", "~")): raise AuthorityError("evidence reference is not repository-relative")
    candidate = (root / ref).resolve()
    try: candidate.relative_to(root.resolve())
    except ValueError as exc: raise AuthorityError("evidence reference escapes repository") from exc
    if not candidate.is_file() or candidate.is_symlink(): raise AuthorityError(f"unresolvable evidence reference: {ref}")
    return candidate


def _evidence_document(root: Path, ref: str) -> EvidenceSnapshot:
    """Read once and retain a content digest; no later authorization re-read."""
    path = _safe_repo_file(root, ref)
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AuthorityError(f"invalid canonical JSON: {path}") from exc
    if not isinstance(value, dict):
        raise AuthorityError("evidence document must be a JSON object")
    return EvidenceSnapshot(ref=ref, value=value, sha256=_sha(raw))


def _git_commit_in_context(root: Path, sha: str) -> None:
    if not isinstance(sha, str) or len(sha) != 40 or any(ch not in "0123456789abcdef" for ch in sha): raise AuthorityError("invalid implementation commit SHA")
    if subprocess.run(["git", "-C", str(root), "merge-base", "--is-ancestor", sha, "HEAD"], capture_output=True).returncode != 0:
        raise AuthorityError("implementation commit is not in current repository ancestry")


def _commit_ref(root: Path, ref: str) -> str:
    if not isinstance(ref, str) or not ref.startswith("commit:"): raise AuthorityError("invalid commit evidence reference")
    sha = ref.removeprefix("commit:"); _git_commit_in_context(root, sha); return sha


def _verify_test_manifest(root: Path, snapshot: EvidenceSnapshot, task_id: str, ci_verifier: CIExecutionVerifier | None) -> str:
    value = snapshot.value
    required = {"schema_version", "kind", "provider", "repository_id", "commit_sha", "workflow", "run_id", "job_id", "environment", "tests", "counts", "raw_output_blob_hash", "task_id"}
    if not required <= set(value) or value.get("kind") != TEST_MANIFEST_KIND or value.get("task_id") != task_id:
        raise AuthorityError("test evidence is not a task-bound controlled execution manifest")
    if value.get("provider") != "github-actions" or value.get("environment") != "CI" or not all(isinstance(value.get(k), str) and value[k] for k in ("workflow", "run_id", "job_id")):
        raise AuthorityError("test execution manifest has no supported controlled execution identity")
    sha = value.get("commit_sha"); _git_commit_in_context(root, sha)
    counts = value.get("counts")
    if not isinstance(counts, dict) or not isinstance(value.get("tests"), list) or not value["tests"] or counts.get("total", 0) <= 0 or counts.get("failed") != 0 or counts.get("error") != 0:
        raise AuthorityError("test execution manifest does not establish a passing non-empty run")
    raw_hash = value.get("raw_output_blob_hash")
    if not isinstance(raw_hash, str) or len(raw_hash.removeprefix("sha256:")) != 64: raise AuthorityError("test execution manifest omits raw artifact hash")
    if ci_verifier is None: raise HumanRequired("controlled CI execution record cannot be independently resolved offline")
    result = ci_verifier.execution_record(value)
    if result is None: raise HumanRequired("controlled CI execution record is unavailable")
    if result is not True: raise AuthorityError("controlled CI execution record rejected manifest")
    return sha


def _verify_acceptance(value: dict[str, Any], task_id: str, commit_sha: str, human_authority_verifier: HumanAuthorityVerifier | None) -> None:
    required = {"schema_version", "kind", "task_id", "acceptance_actor", "authority_source", "commit_sha", "criteria", "evidence_refs", "decision"}
    if not required <= set(value) or value.get("kind") != "acceptance" or value.get("task_id") != task_id or value.get("decision") != "ACCEPTED": raise AuthorityError("acceptance evidence does not establish accepted result for task")
    if value.get("commit_sha") != commit_sha or not isinstance(value.get("criteria"), list) or not value["criteria"] or not isinstance(value.get("evidence_refs"), list): raise AuthorityError("acceptance evidence is not bound to this commit and criteria")
    source = value.get("authority_source")
    if not isinstance(source, dict) or source.get("type") != "human_authority" or not isinstance(value.get("acceptance_actor"), str) or value["acceptance_actor"] == ARIADNA_ID: raise AuthorityError("acceptance has no eligible human authority source")
    if human_authority_verifier is None: raise HumanRequired("human acceptance authority cannot be independently resolved offline")
    result = human_authority_verifier.acceptance_record(value)
    if result is None: raise HumanRequired("human acceptance authority record is unavailable")
    if result is not True: raise AuthorityError("human acceptance authority rejected record")


def _verify_pull_request(root: Path, snapshot: EvidenceSnapshot, evidence: EvidenceSet, task_id: str, ci_verifier: CIExecutionVerifier | None) -> str:
    """Resolve a PR only through its controlled execution manifest.

    A number or PR-shaped JSON cannot establish its head offline. The manifest
    is the independently resolved CI record and must bind the exact head.
    """
    value = snapshot.value
    required = {"kind", "task_id", "provider", "repository_id", "number", "head_commit_sha", "execution_manifest"}
    if not required <= set(value) or value.get("kind") != "pull_request" or value.get("task_id") != task_id or value.get("provider") != "github-actions" or not isinstance(value.get("number"), int):
        raise AuthorityError("invalid PR evidence record")
    head = value.get("head_commit_sha"); _git_commit_in_context(root, head)
    manifest = evidence.snapshot(value["execution_manifest"])
    if _verify_test_manifest(root, manifest, task_id, ci_verifier) != head:
        raise AuthorityError("PR evidence and controlled execution bind different commits")
    return head


def _resolve_implementation(root: Path, refs: list[str], evidence: EvidenceSet, task_id: str, ci_verifier: CIExecutionVerifier | None) -> str:
    commits = [_commit_ref(root, ref) for ref in refs if ref.startswith("commit:")]
    manifests: list[str] = []
    for ref in refs:
        if ref.startswith("commit:"): continue
        snapshot = evidence.snapshot(ref)
        value = snapshot.value
        if value.get("kind") == TEST_MANIFEST_KIND: manifests.append(_verify_test_manifest(root, snapshot, task_id, ci_verifier))
        elif value.get("kind") == "pull_request": manifests.append(_verify_pull_request(root, snapshot, evidence, task_id, ci_verifier))
    all_shas = commits + manifests
    if not all_shas: raise AuthorityError("DONE requires commit or PR evidence bound to controlled execution")
    if len(set(all_shas)) != 1: raise AuthorityError("commit and controlled execution evidence bind different commits")
    return all_shas[0]


def _validate_evidence(root: Path, task_id: str, evidence_refs: list[str], target_status: str, *, ci_verifier: CIExecutionVerifier | None = None, human_authority_verifier: HumanAuthorityVerifier | None = None) -> ValidatedEvidence:
    """Validate a single immutable evidence snapshot and return its binding.

    The returned digest binds both identities and bytes of every evaluated
    local record.  The caller must carry this value into the journal rather
    than recomputing it from mutable paths later.
    """
    if not isinstance(evidence_refs, list) or not all(isinstance(item, str) for item in evidence_refs): raise AuthorityError("evidence_refs must be strings")
    evidence = EvidenceSet(root)
    # Capture direct evidence before structural validation or external checks.
    # PR records can expand this set below, still before verification.
    docs = [evidence.snapshot(ref) for ref in evidence_refs if not ref.startswith("commit:")]
    for snapshot in docs:
        if snapshot.value.get("kind") == "pull_request":
            nested = snapshot.value.get("execution_manifest")
            if isinstance(nested, str): evidence.snapshot(nested)
    binding_items: list[dict[str, str]] = [{"ref": snapshot.ref, "sha256": snapshot.sha256} for snapshot in evidence.values()]
    for ref in evidence_refs:
        if ref.startswith("commit:"):
            binding_items.append({"ref": ref, "sha256": _sha(ref.encode("utf-8"))})
    def result(commit: str | None = None, test: str | None = None, acceptance: str | None = None, pull: str | None = None) -> ValidatedEvidence:
        return ValidatedEvidence(task_id, commit, test, acceptance, pull, _sha(_canonical({"task_id": task_id, "evidence": sorted(binding_items, key=lambda x: x["ref"])})))
    if target_status == "DONE":
        # Resolve all local objects before any external call: malformed and
        # nonexistent evidence is DENY, never obscured as an offline condition.
        implementation = _resolve_implementation(root, evidence_refs, evidence, task_id, ci_verifier)
        acceptance = [snapshot for snapshot in docs if snapshot.value.get("kind") == "acceptance"]
        if len(acceptance) != 1: raise AuthorityError("DONE requires exactly one acceptance evidence record")
        _verify_acceptance(acceptance[0].value, task_id, implementation, human_authority_verifier)
        tests = [snapshot.sha256 for snapshot in docs if snapshot.value.get("kind") == TEST_MANIFEST_KIND]
        if len(tests) != 1: raise AuthorityError("DONE requires exactly one controlled test execution evidence")
        pulls = [snapshot.sha256 for snapshot in docs if snapshot.value.get("kind") == "pull_request"]
        if len(pulls) > 1: raise AuthorityError("DONE permits at most one PR evidence record")
        unknown = [snapshot.ref for snapshot in docs if snapshot.value.get("kind") not in {TEST_MANIFEST_KIND, "pull_request", "acceptance"}]
        if unknown: raise AuthorityError("unaccepted completion evidence: " + ", ".join(unknown))
        return result(implementation, tests[0], acceptance[0].sha256, pulls[0] if pulls else None)
    elif target_status == "BLOCKED":
        for snapshot in docs:
            value = snapshot.value
            if value.get("kind") == "blocker" and value.get("task_id") == task_id and value.get("state") == "OPEN" and value.get("blocker"): return result()
        raise AuthorityError("BLOCKED requires a resolvable open blocker evidence record")
    return result()


class AuthorityEngine:
    """Privileged composition boundary for independently controlled trust.

    Only the hosting integration constructs this engine and supplies adapters
    to CI and human-authority systems.  Authorization requests contain no
    verifier and cannot select, replace, or construct these dependencies.
    An engine without both adapters deliberately leaves DONE HUMAN_REQUIRED.
    """
    def __init__(self, root: Path, *, ci_verifier: CIExecutionVerifier | None = None, human_authority_verifier: HumanAuthorityVerifier | None = None):
        self.root = root
        self._ci_verifier = ci_verifier
        self._human_authority_verifier = human_authority_verifier

    def _evaluate(self, *, sender_agent: str, task_id: str, action: str, target_status: str | None = None, evidence_refs: list[str] | None = None, handoff: dict[str, Any] | None = None) -> tuple[Decision, ValidatedEvidence | None]:
      try:
        project = _load(self.root / "projects/las-voces/project.json"); agent = _load(self.root / "projects/las-voces/agents/ariadna.json")
        if sender_agent != ARIADNA_ID or agent.get("id") != ARIADNA_ID: return Decision(Verdict.DENY, "unknown sender_agent"), None
        if action in FORBIDDEN: return Decision(Verdict.DENY, f"Ariadna forbidden action: {action}"), None
        if action == "coordinate_verified_work":
            refs = evidence_refs or []
            if not refs: return Decision(Verdict.DENY, "coordination requires resolvable evidence"), None
            for ref in refs: _safe_repo_file(self.root, ref)
            _task(project, task_id); return Decision(Verdict.ALLOW, "evidence-backed coordination"), None
        if action == "emit_handoff":
            if not isinstance(handoff, dict): return Decision(Verdict.DENY, "handoff payload is required"), None
            validate_handoff(self.root, handoff); return Decision(Verdict.ALLOW, "validated structured handoff"), None
        if action != "transition_status": return Decision(Verdict.HUMAN_REQUIRED, "action is outside Ariadna deterministic contract"), None
        task = _task(project, task_id)
        if target_status not in TRANSITIONS.get(task.get("status"), set()): return Decision(Verdict.DENY, "invalid lifecycle transition"), None
        evidence = _validate_evidence(self.root, task_id, evidence_refs or [], target_status or "", ci_verifier=self._ci_verifier, human_authority_verifier=self._human_authority_verifier)
        return Decision(Verdict.ALLOW, "evidence-backed canonical transition"), evidence
      except HumanRequired as exc: return Decision(Verdict.HUMAN_REQUIRED, str(exc)), None
      except AuthorityError as exc: return Decision(Verdict.DENY, str(exc)), None

    def evaluate(self, **request: Any) -> Decision:
        return self._evaluate(**request)[0]

    def interrupted_transitions(self) -> list[dict[str, Any]]:
        return interrupted_transitions(self.root / "projects/las-voces/activity.ndjson")

    def transition(self, *, sender_agent: str, task_id: str, target_status: str, evidence_refs: list[str], expected_project_hash: str | None = None, failure_hook: Callable[[str], None] | None = None) -> Decision:
        return _transition(self, sender_agent=sender_agent, task_id=task_id, target_status=target_status, evidence_refs=evidence_refs, expected_project_hash=expected_project_hash, failure_hook=failure_hook)


def _append_fsync(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(_canonical(value).decode("utf-8") + "\n"); handle.flush(); os.fsync(handle.fileno())


def _atomic_json(path: Path, value: Any) -> None:
    fd, temporary = tempfile.mkstemp(prefix=".ariadna-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(_canonical(value).decode("utf-8") + "\n"); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_DIRECTORY)
        try: os.fsync(directory_fd)
        finally: os.close(directory_fd)
    except BaseException:
        try: os.unlink(temporary)
        except FileNotFoundError:  # fail-soft: el temporal ya puede haber desaparecido durante cleanup; la excepción original se vuelve a propagar
            pass
        raise


def interrupted_transitions(history_path: Path) -> list[dict[str, Any]]:
    """Return durable intents without a matching commit; never reconcile them."""
    if not history_path.exists() or history_path.is_symlink(): raise AuthorityError("activity history unavailable")
    intents: dict[str, dict[str, Any]] = {}
    for number, line in enumerate(history_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip(): continue
        try: event = json.loads(line)
        except json.JSONDecodeError as exc: raise AuthorityError(f"invalid activity history at line {number}") from exc
        if event.get("event_type") == "TRANSITION_INTENT": intents[event.get("transition_id")] = event
        elif event.get("event_type") == "TRANSITION_COMMITTED":
            transition_id, intent = event.get("transition_id"), intents.get(event.get("transition_id"))
            if intent is None or any(intent.get(k) != event.get(k) for k in ("task_id", "from_status", "to_status", "project_hash_before", "evidence_binding_hash", "implementation_commit_sha", "test_manifest_digest", "acceptance_record_digest", "pr_record_digest", "actor", "decision")): raise AuthorityError("invalid transition commit audit binding")
            del intents[transition_id]
    return list(intents.values())


@contextmanager
def _transition_lock(root: Path):
    path = root / "projects/las-voces/.ariadna-transition.lock"; fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try: fcntl.flock(fd, fcntl.LOCK_EX); yield
    finally: fcntl.flock(fd, fcntl.LOCK_UN); os.close(fd)


def _transition(engine: AuthorityEngine, *, sender_agent: str, task_id: str, target_status: str, evidence_refs: list[str], expected_project_hash: str | None = None, failure_hook: Callable[[str], None] | None = None) -> Decision:
    """Journal intent → CAS state → committed record under exclusive lock."""
    root = engine.root
    project_path, history_path = root / "projects/las-voces/project.json", root / "projects/las-voces/activity.ndjson"
    if history_path.is_symlink(): return Decision(Verdict.DENY, "activity history unavailable")
    hook = failure_hook or (lambda _boundary: None)
    with _transition_lock(root):
        try:
            if interrupted_transitions(history_path): return Decision(Verdict.DENY, "interrupted transition requires human reconciliation")
            before_bytes, before = project_path.read_bytes(), _load(project_path); task = _task(before, task_id)
            if expected_project_hash is not None and _sha(before_bytes) != expected_project_hash:
                return Decision(Verdict.DENY, "stale project observation")
            decision, evidence = engine._evaluate(sender_agent=sender_agent, task_id=task_id, action="transition_status", target_status=target_status, evidence_refs=evidence_refs)
            if decision.verdict is not Verdict.ALLOW: return decision
            if evidence is None: raise AuthorityError("authorized transition lacks validated evidence binding")
            # Evidence evaluation may be slow or invoke controlled verifiers.
            # A change during that interval must not leave a stale intent.
            if project_path.read_bytes() != before_bytes:
                return Decision(Verdict.DENY, "stale project state before transition intent")
            base = {"transition_id": str(uuid.uuid4()), "project_id": PROJECT_ID, "task_id": task_id, "from_status": task["status"], "to_status": target_status, "project_hash_before": _sha(before_bytes), "evidence_binding_hash": evidence.normalized_evidence_digest, "implementation_commit_sha": evidence.implementation_commit_sha, "test_manifest_digest": evidence.test_manifest_digest, "acceptance_record_digest": evidence.acceptance_record_digest, "pr_record_digest": evidence.pr_record_digest, "actor": sender_agent, "decision": "ALLOW"}
            intent = {**base, "event_id": f"{base['transition_id']}:intent", "event_type": "TRANSITION_INTENT", "state": "INTENT"}
            hook("before_intent_append"); _append_fsync(history_path, intent)
            hook("before_project_cas")
            if project_path.read_bytes() != before_bytes: raise AuthorityError("stale project state; transition compare-and-swap failed")
            updated = copy.deepcopy(before); _task(updated, task_id)["status"] = target_status
            hook("before_project_write"); _atomic_json(project_path, updated)
            committed = {**base, "event_id": f"{base['transition_id']}:committed", "event_type": "TRANSITION_COMMITTED", "state": "COMMITTED", "project_hash_after": _project_hash(project_path)}
            hook("before_commit_append"); _append_fsync(history_path, committed)
            return decision
        except OSError as exc: raise AuthorityError("transition persistence failed; durable intent (if any) requires human reconciliation") from exc


def evaluate(root: Path, **request: Any) -> Decision:
    """Unconfigured convenience entrypoint; DONE cannot establish trust here."""
    return AuthorityEngine(root).evaluate(**request)


def transition(root: Path, *, sender_agent: str, task_id: str, target_status: str, evidence_refs: list[str], expected_project_hash: str | None = None, failure_hook: Callable[[str], None] | None = None) -> Decision:
    """Unconfigured convenience entrypoint; no request-controlled verifier."""
    return AuthorityEngine(root).transition(sender_agent=sender_agent, task_id=task_id, target_status=target_status, evidence_refs=evidence_refs, expected_project_hash=expected_project_hash, failure_hook=failure_hook)


def validate_handoff(root: Path, envelope: dict[str, Any]) -> None:
    schema = _load(root / "projects/las-voces/sync/message-envelope.schema.json")
    errors = sorted(Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(envelope), key=lambda item: item.path)
    if errors: raise AuthorityError("malformed MessageEnvelope: " + errors[0].message)
    project = _load(root / "projects/las-voces/project.json"); _task(project, envelope["task_id"])
    if envelope["sender_agent"] != ARIADNA_ID: raise AuthorityError("unknown sender_agent")
    context = envelope["authority_context"]; handoff = context.get("handoff") if isinstance(context, dict) else None
    required = {"owner", "branch_worktree", "scope", "acceptance_criteria", "commit_pr", "test_evidence", "blockers", "next_action"}
    if not isinstance(handoff, dict) or set(handoff) != required or not all(isinstance(value, str) for value in handoff.values()) or not handoff["owner"] or not handoff["next_action"]: raise AuthorityError("malformed Ariadna handoff payload")
