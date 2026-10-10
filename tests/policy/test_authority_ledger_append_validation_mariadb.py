"""Falla 2 del auditor r3: append valida el evento contra el estado ANTES de escribir.

Con triggers append-only en `jax_authority.authority_events`, un evento
semánticamente inválido (p.ej. ACTIVATION_GRANTED hacia un id que no es una
ratificación vigente) que llegue a firmarse y guardarse vuelve el ledger
inverificable para siempre: `verify_authority_ledger` lo rechaza en cada
replay posterior y no hay UPDATE/DELETE para retirarlo. Esta prueba, contra
una MariaDB 12.3.3 efímera y aislada (`--network none`), deja escrito que el
camino de rechazo no escribe NI UNA fila: el replay completo con el evento
nuevo corre antes de `store.append`.
"""
from datetime import datetime, timedelta, timezone

import pymysql
import pytest

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from policy.authority_ledger.canonical import canonical_bytes
from policy.authority_ledger.errors import AuthorityStateError
from policy.authority_ledger.models import (
    AuthorityEventIntent, AuthorityEventType, AuthorityLedgerGenesis,
    OverlayPayload, OverlayScope, OverlayType,
)
from policy.authority_ledger.replay import genesis_hash
from tests.policy.test_authority_ledger_events import verify_authority_ledger
from tests.policy.test_authority_ledger_events import append_authority_event
from policy.authority_ledger.signatures import encode_public_key, public_key_bytes, public_key_fingerprint
from policy.authority_ledger.storage import MariaDBAuthorityLedgerStore
from policy.authority_ledger.trusted_root import TrustedAuthorityRoot
from tests.policy.test_authority_ledger_events import append_ratification
from tests.policy.test_authority_ledger_storage_mariadb import (
    MIGRATION, _apply_migration, _ephemeral_mariadb,
)


def test_append_rejects_state_invalid_event_before_any_write():
    with _ephemeral_mariadb() as connect:
        with connect() as admin:
            _apply_migration(admin, MIGRATION)

        key = Ed25519PrivateKey.generate()
        public_key = key.public_key()
        genesis = AuthorityLedgerGenesis(
            "1.0", "JAX_AUTHORITY_LEDGER_GENESIS", "JAX-AUTHORITY-LEDGER/1",
            "human:fernando", "f1.1-test-key", encode_public_key(public_key),
        )
        root = TrustedAuthorityRoot(
            "1.0", "JAX_TRUSTED_AUTHORITY_ROOT", genesis.ledger_identity,
            genesis_hash(genesis), genesis.constitutional_key_id,
            public_key_fingerprint(public_key_bytes(public_key)),
        )
        with connect() as db:
            with db.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO jax_authority.authority_ledger_genesis "
                    "(singleton,canonical_genesis,genesis_hash) VALUES (1,%s,%s)",
                    (canonical_bytes(genesis.projection()), genesis_hash(genesis)),
                )
            db.commit()

        store = MariaDBAuthorityLedgerStore(lambda: connect(database="jax_authority"))
        now = datetime(2026, 10, 7, tzinfo=timezone.utc)
        ratification = append_ratification(
            store, root, key,
            event_id="018cc251-f400-7000-8000-000000000001", recorded_at_utc=now,
        )
        with connect() as db:
            with db.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM jax_authority.authority_events")
                assert cursor.fetchone()[0] == 1

        # El veneno: activación hacia un id que no es ratificación de corpus
        # vigente. En master esto se firmaba y se guardaba; con el replay
        # previo, se rechaza cerrado antes de tocar la base.
        with pytest.raises(AuthorityStateError, match="activación requiere ratificación"):
            append_authority_event(
                store, root, key,
                AuthorityEventIntent(
                    AuthorityEventType.ACTIVATION_GRANTED, "human:fernando",
                    ratification_event_id="018cc251-f400-7000-8000-0000000000dd",
                ),
                event_id="018cc251-f400-7000-8000-000000000002", recorded_at_utc=now,
            )

        with connect() as db:
            with db.cursor() as cursor:
                cursor.execute(
                    "SELECT COUNT(*) FROM jax_authority.authority_events"
                )
                assert cursor.fetchone()[0] == 1, "el evento rechazado no debe dejar filas"
                cursor.execute(
                    "SELECT sequence,head_event_id FROM jax_authority.authority_ledger_head WHERE singleton=1"
                )
                assert cursor.fetchone() == (1, ratification.event_id)

        state = verify_authority_ledger(store.get_genesis(), store.events(), root)
        assert state.checkpoint.sequence == 1
        assert state.active_policy_corpus_hash is None


def test_overlay_a_corpus_no_ratificado_no_escribe_nada():
    """Falla 2 del auditor de #377 en storage: el overlay sin ratificación
    vigente se rechaza ANTES de escribir (replay previo) — cero filas nuevas."""
    with _ephemeral_mariadb() as connect:
        with connect() as admin:
            _apply_migration(admin, MIGRATION)

        key = Ed25519PrivateKey.generate()
        public_key = key.public_key()
        genesis = AuthorityLedgerGenesis(
            "1.0", "JAX_AUTHORITY_LEDGER_GENESIS", "JAX-AUTHORITY-LEDGER/1",
            "human:fernando", "f1.1-test-key", encode_public_key(public_key),
        )
        root = TrustedAuthorityRoot(
            "1.0", "JAX_TRUSTED_AUTHORITY_ROOT", genesis.ledger_identity,
            genesis_hash(genesis), genesis.constitutional_key_id,
            public_key_fingerprint(public_key_bytes(public_key)),
        )
        with connect() as db:
            with db.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO jax_authority.authority_ledger_genesis "
                    "(singleton,canonical_genesis,genesis_hash) VALUES (1,%s,%s)",
                    (canonical_bytes(genesis.projection()), genesis_hash(genesis)),
                )
            db.commit()

        store = MariaDBAuthorityLedgerStore(lambda: connect(database="jax_authority"))
        now = datetime(2026, 10, 7, tzinfo=timezone.utc)
        ratification = append_ratification(
            store, root, key,
            event_id="018cc251-f400-7000-8000-000000000001", recorded_at_utc=now,
        )
        huerfano = OverlayPayload(
            "huerfano-mariadb", OverlayType.EXCEPTION, "sha256:" + "b" * 64,
            OverlayScope(("ALICE",), ("READ",)), now, now + timedelta(hours=1),
            target_rule_ids=("send-receipt",), exception_code="HUERFANO",
        )
        with pytest.raises(AuthorityStateError, match="overlay exige ratificación"):
            append_authority_event(
                store, root, key,
                AuthorityEventIntent(AuthorityEventType.OVERLAY_ISSUED, "human:fernando", overlay=huerfano),
                event_id="018cc251-f400-7000-8000-000000000002", recorded_at_utc=now,
            )

        with connect() as db:
            with db.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM jax_authority.authority_events")
                assert cursor.fetchone()[0] == 1, "el overlay rechazado no debe dejar filas"
                cursor.execute(
                    "SELECT sequence,head_event_id FROM jax_authority.authority_ledger_head WHERE singleton=1"
                )
                assert cursor.fetchone() == (1, ratification.event_id)

        state = verify_authority_ledger(store.get_genesis(), store.events(), root)
        assert state.checkpoint.sequence == 1
