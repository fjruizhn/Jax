from pathlib import Path

import pytest
import os
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from policy.authority_ledger.bootstrap import (
    BootstrapInspectionStatus, _write_root_exclusively, inspect_authority_bootstrap,
    stage_authority_genesis,
)
from policy.authority_ledger.signatures import encode_public_key


class _Cursor:
    def __init__(self, rows, *, schema_exists=True):
        self.rows = iter(rows)
        self.closed = False
        self.schema_query = False
        self.schema_exists = schema_exists

    def execute(self, statement, *_args):
        self.schema_query = "information_schema.schemata" in statement
        return None

    def fetchall(self):
        return next(self.rows)

    def fetchone(self):
        if self.schema_query:
            self.schema_query = False
            return (1 if self.schema_exists else 0,)
        return next(self.rows)

    def close(self):
        self.closed = True


class _Connection:
    def __init__(self, rows):
        self.cursor_value = _Cursor(rows)

    def cursor(self):
        return self.cursor_value


def test_stage_is_public_deterministic_and_not_authority():
    public = encode_public_key(Ed25519PrivateKey.generate().public_key())
    staged = stage_authority_genesis(public, "fernando-1")
    assert staged.classification == "NOT_AUTHORITY"
    assert staged.genesis.constitutional_actor_id == "human:fernando"
    assert staged.trusted_root.genesis_hash == staged.genesis_hash
    assert staged.canonical_genesis


def test_stage_rejects_non_ed25519_public_material():
    with pytest.raises(Exception):
        stage_authority_genesis("not-a-public-key", "fernando-1")


def test_root_publication_is_exclusive_and_never_replaces(tmp_path):
    staged = stage_authority_genesis(
        encode_public_key(Ed25519PrivateKey.generate().public_key()), "fernando-1",
    )
    root = tmp_path / "trusted-root.json"
    _write_root_exclusively(root, staged.trusted_root, owner_uid=os.getuid())
    original = root.read_bytes()
    with pytest.raises(FileExistsError):
        _write_root_exclusively(root, staged.trusted_root, owner_uid=os.getuid())
    assert root.read_bytes() == original


def test_root_publication_rejects_symlinked_parent(tmp_path):
    staged = stage_authority_genesis(
        encode_public_key(Ed25519PrivateKey.generate().public_key()), "fernando-1",
    )
    target = tmp_path / "target"
    target.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(target, target_is_directory=True)
    with pytest.raises(OSError):
        _write_root_exclusively(linked / "trusted-root.json", staged.trusted_root, owner_uid=os.getuid())


def test_inspection_rejects_symlinked_external_artifact(tmp_path):
    outside = tmp_path / "outside"
    outside.write_text("not authority")
    root = tmp_path / "root"
    root.symlink_to(outside)
    connection = _Connection([
        [("authority_ledger_genesis",), ("authority_events",), ("authority_ledger_head",)],
        None, (0, None, None), (0,),
    ])
    result = inspect_authority_bootstrap(
        connection, trusted_root_path=root, checkpoint_path=tmp_path / "check",
        receipt_path=tmp_path / "receipt",
    )
    assert result.status is BootstrapInspectionStatus.RECOVERY


def test_inspection_marks_permission_error_unavailable_not_empty(tmp_path):
    class Denied:
        def cursor(self):
            raise PermissionError("1142")
    result = inspect_authority_bootstrap(
        Denied(), trusted_root_path=tmp_path / "root", checkpoint_path=tmp_path / "check",
        receipt_path=tmp_path / "receipt",
    )
    assert result.status is BootstrapInspectionStatus.INSPECTION_UNAVAILABLE


def test_inspection_accepts_only_exact_admin_absence_after_metadata_visibility(tmp_path):
    class MissingTable(Exception):
        args = (1146, "table missing")

    class Cursor(_Cursor):
        def execute(self, statement, *_args):
            if "information_schema.schemata" in statement:
                self.schema_query = True
                return
            if self.closed:
                raise MissingTable()
            self.closed = True

    class Connection:
        def cursor(self):
            return Cursor([[]], schema_exists=False)

    result = inspect_authority_bootstrap(
        Connection(), trusted_root_path=tmp_path / "root", checkpoint_path=tmp_path / "check",
        receipt_path=tmp_path / "receipt",
    )
    assert result.status is BootstrapInspectionStatus.CANDIDATE


def test_inspection_never_treats_limited_metadata_as_absence(tmp_path):
    class Denied(Exception):
        args = (1142, "command denied")

    class Cursor(_Cursor):
        def execute(self, *_args):
            raise Denied()

    class Connection:
        def cursor(self):
            return Cursor([[]])

    result = inspect_authority_bootstrap(
        Connection(), trusted_root_path=tmp_path / "root", checkpoint_path=tmp_path / "check",
        receipt_path=tmp_path / "receipt",
    )
    assert result.status is BootstrapInspectionStatus.INSPECTION_UNAVAILABLE


def test_inspection_marks_empty_complete_schema_candidate(tmp_path):
    connection = _Connection([
        [("authority_ledger_genesis",), ("authority_events",), ("authority_ledger_head",)],
        None, (0, None, None), (0,),
    ])
    result = inspect_authority_bootstrap(
        connection, trusted_root_path=tmp_path / "root", checkpoint_path=tmp_path / "check",
        receipt_path=tmp_path / "receipt",
    )
    assert result.status is BootstrapInspectionStatus.CANDIDATE
    assert connection.cursor_value.closed


def test_inspection_does_not_call_events_empty_when_head_is_inconsistent(tmp_path):
    connection = _Connection([
        [("authority_ledger_genesis",), ("authority_events",), ("authority_ledger_head",)],
        None, (1, "event", "sha256:" + "a" * 64), (0,),
    ])
    result = inspect_authority_bootstrap(
        connection, trusted_root_path=tmp_path / "root", checkpoint_path=tmp_path / "check",
        receipt_path=tmp_path / "receipt",
    )
    assert result.status is BootstrapInspectionStatus.INCONSISTENT
