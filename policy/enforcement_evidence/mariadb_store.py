"""Synchronous authoritative MariaDB blob store.

It deliberately exposes only bytes by content identity.  Higher level objects
are parsed as untrusted values and must be verified by their fixed lifecycle.
"""
from __future__ import annotations
from .evidence_store import EvidenceBlob, MAX_BLOB_BYTES
from .ids import sha256_bytes, require_hash
from .errors import EvidenceBlobMissingError, EvidenceBlobHashMismatchError, EvidenceBlobTooLargeError
from .canonical import canonical_bytes
from .models import MAX_REFERENCED_BYTES_PER_ARTIFACT, MAX_ARTIFACT_ENVELOPE_BYTES
from .errors import EvidenceArtifactIntegrityError, ObservationIntegrityError, AssertionIntegrityError
class MariaDBEvidenceStore:
    def __init__(self, connection_factory): self._connection_factory=connection_factory
    def put_evidence_blob(self, data: bytes) -> EvidenceBlob:
        if not isinstance(data,bytes): raise TypeError("bytes requeridos")
        if len(data)>MAX_BLOB_BYTES: raise EvidenceBlobTooLargeError("blob > 1 MiB")
        h=sha256_bytes(data); con=self._connection_factory()
        try:
            cur=con.cursor()
            cur.execute("SELECT size_bytes,blob_bytes FROM jax_evidence.evidence_blobs WHERE evidence_hash=%s FOR UPDATE",(h,))
            row=cur.fetchone()
            if row is None:
                cur.execute("INSERT INTO jax_evidence.evidence_blobs(evidence_hash,size_bytes,blob_bytes) VALUES (%s,%s,%s)",(h,len(data),data))
            elif int(row[0])!=len(data) or bytes(row[1])!=data:
                raise EvidenceBlobHashMismatchError("same hash with different bytes")
            con.commit(); return EvidenceBlob(h,len(data),data)
        except Exception: con.rollback(); raise
        finally: con.close()
    def get_evidence_blob(self, evidence_hash: str) -> bytes:
        require_hash(evidence_hash); con=self._connection_factory()
        try:
            cur=con.cursor(); cur.execute("SELECT size_bytes,blob_bytes FROM jax_evidence.evidence_blobs WHERE evidence_hash=%s",(evidence_hash,)); row=cur.fetchone()
            if row is None: raise EvidenceBlobMissingError(evidence_hash)
            data=bytes(row[1])
            if int(row[0])!=len(data) or sha256_bytes(data)!=evidence_hash: raise EvidenceBlobHashMismatchError(evidence_hash)
            return data
        finally: con.close()
    def __record_identity(self, identity):
        """Persist an identity only from a fixed composition boundary."""
        from .evidence_store import _require_fixed_composition_write
        _require_fixed_composition_write()
        # A manifest reference is meaningful only if its bytes are present.
        self.get_evidence_blob(identity.build_manifest_blob_hash)
        h=identity.implementation_identity_hash
        payload=canonical_bytes(identity.projection()).decode("utf-8")
        con=self._connection_factory()
        try:
            cur=con.cursor(); cur.execute("SELECT canonical_identity FROM jax_evidence.implementation_identities WHERE identity_hash=%s FOR UPDATE",(h,)); row=cur.fetchone()
            if row is None:
                cur.execute("INSERT INTO jax_evidence.implementation_identities(identity_hash,canonical_identity) VALUES (%s,%s)",(h,payload))
            elif (bytes(row[0]).decode() if isinstance(row[0],bytes) else row[0]) != payload:
                raise EvidenceArtifactIntegrityError("identity collision")
            con.commit(); return identity
        except Exception: con.rollback(); raise
        finally: con.close()
    def load_implementation_identity(self, identity_hash):
        require_hash(identity_hash); con=self._connection_factory()
        try:
            cur=con.cursor(); cur.execute("SELECT canonical_identity FROM jax_evidence.implementation_identities WHERE identity_hash=%s",(identity_hash,)); row=cur.fetchone()
            if row is None: raise EvidenceBlobMissingError(identity_hash)
            import json
            from .implementation_identity import implementation_identity_from_projection
            value=implementation_identity_from_projection(json.loads(bytes(row[0]).decode() if isinstance(row[0],bytes) else row[0]))
            if value.implementation_identity_hash != identity_hash: raise EvidenceArtifactIntegrityError("identity row/canonical mismatch")
            self.get_evidence_blob(value.build_manifest_blob_hash)
            from .evidence_store import _loaded, _identities
            return _loaded(_identities,value)
        finally: con.close()
    def __persist_control_definition(self, definition):
        from .control_registry import require_trusted_definition
        require_trusted_definition(definition)
        h=definition.control_definition_hash; payload=canonical_bytes(definition.projection()).decode("utf-8")
        con=self._connection_factory()
        try:
            cur=con.cursor(); cur.execute("SELECT canonical_definition FROM jax_evidence.control_definitions WHERE control_definition_hash=%s FOR UPDATE",(h,)); row=cur.fetchone()
            if row is None:
                cur.execute("INSERT INTO jax_evidence.control_definitions(control_definition_hash,control_id,control_version,canonical_definition) VALUES (%s,%s,%s,%s)",(h,definition.control_id,definition.control_version,payload))
            elif (bytes(row[0]).decode() if isinstance(row[0],bytes) else row[0]) != payload: raise EvidenceArtifactIntegrityError("definition collision")
            con.commit(); return definition
        except Exception: con.rollback(); raise
        finally: con.close()
    def load_control_definition(self, control_id, control_version=1):
        """A DB snapshot is accepted only when it exactly equals packaged V1."""
        from .control_registry import load_control_definition
        expected=load_control_definition(control_id,control_version); h=expected.control_definition_hash
        con=self._connection_factory()
        try:
            cur=con.cursor(); cur.execute("SELECT control_id,control_version,canonical_definition FROM jax_evidence.control_definitions WHERE control_definition_hash=%s",(h,)); row=cur.fetchone()
            if row is None: raise EvidenceBlobMissingError(h)
            payload=bytes(row[2]).decode() if isinstance(row[2],bytes) else row[2]
            if row[0]!=control_id or int(row[1])!=control_version or payload!=canonical_bytes(expected.projection()).decode("utf-8"):
                raise EvidenceArtifactIntegrityError("definition row/canonical mismatch")
            return expected
        finally: con.close()
    def __record_artifact(self, artifact):
        """Persist only after every referenced blob is authoritatively readable."""
        from .evidence_store import _require_fixed_composition_write
        _require_fixed_composition_write()
        if len(canonical_bytes(artifact.projection())) > MAX_ARTIFACT_ENVELOPE_BYTES:
            raise EvidenceArtifactIntegrityError("artifact envelope too large")
        total=0
        for ref in artifact.blob_refs:
            total += len(self.get_evidence_blob(ref.evidence_hash))
        if total > MAX_REFERENCED_BYTES_PER_ARTIFACT:
            raise EvidenceArtifactIntegrityError("artifact referenced bytes too large")
        h=artifact.artifact_hash; con=self._connection_factory()
        try:
            cur=con.cursor(); payload=canonical_bytes(artifact.projection()).decode("utf-8")
            cur.execute("SELECT canonical_artifact FROM jax_evidence.evidence_artifacts WHERE artifact_hash=%s FOR UPDATE",(h,)); row=cur.fetchone()
            if row is None:
                cur.execute("INSERT INTO jax_evidence.evidence_artifacts(artifact_hash,canonical_artifact) VALUES (%s,%s)",(h,payload))
                for ref in artifact.blob_refs: cur.execute("INSERT INTO jax_evidence.evidence_artifact_blobs(artifact_hash,evidence_hash) VALUES (%s,%s)",(h,ref.evidence_hash))
            elif (bytes(row[0]).decode() if isinstance(row[0],bytes) else row[0]) != payload: raise EvidenceArtifactIntegrityError("artifact collision")
            con.commit(); return artifact
        except Exception: con.rollback(); raise
        finally: con.close()
    def load_evidence_artifact(self, artifact_hash):
        """Only this store-load path may establish artifact provenance."""
        require_hash(artifact_hash); con=self._connection_factory()
        try:
            cur=con.cursor(); cur.execute("SELECT canonical_artifact FROM jax_evidence.evidence_artifacts WHERE artifact_hash=%s",(artifact_hash,)); row=cur.fetchone()
            if row is None: raise EvidenceBlobMissingError(artifact_hash)
            from .artifacts import deserialize_evidence_artifact, verify_evidence_artifact_content
            value=deserialize_evidence_artifact(row[0]);
            if value.artifact_hash != artifact_hash: raise EvidenceArtifactIntegrityError("row/canonical hash mismatch")
            cur.execute("SELECT evidence_hash FROM jax_evidence.evidence_artifact_blobs WHERE artifact_hash=%s ORDER BY evidence_hash",(artifact_hash,)); refs=tuple(x[0] for x in cur.fetchall())
            if refs != tuple(sorted(x.evidence_hash for x in value.blob_refs)): raise EvidenceArtifactIntegrityError("artifact refs mismatch")
            verify_evidence_artifact_content(value,self.get_evidence_blob)
            from .control_registry import load_control_definition
            definition=load_control_definition(value.control_id,value.control_version)
            if definition.control_definition_hash != value.control_definition_hash: raise EvidenceArtifactIntegrityError("artifact control binding mismatch")
            self.load_implementation_identity(value.implementation_identity_hash)
            from .evidence_store import _loaded, _artifacts
            return _loaded(_artifacts,value)
        finally: con.close()
    def __record_observation(self, observation):
        from .evidence_store import _require_fixed_composition_write
        _require_fixed_composition_write()
        for h in observation.evidence_artifact_hashes:
            # FK validates persistence; select makes the failure deterministic before write.
            con0=self._connection_factory()
            try:
                c0=con0.cursor(); c0.execute("SELECT artifact_hash FROM jax_evidence.evidence_artifacts WHERE artifact_hash=%s",(h,))
                if c0.fetchone() is None: raise EvidenceBlobMissingError(h)
            finally: con0.close()
        con=self._connection_factory()
        try:
            cur=con.cursor(); payload=canonical_bytes(observation.projection()).decode("utf-8")
            cur.execute("SELECT observation_hash FROM jax_evidence.enforcement_observations WHERE observation_id=%s FOR UPDATE",(observation.observation_id,)); row=cur.fetchone()
            if row is None:
                cur.execute("INSERT INTO jax_evidence.enforcement_observations(observation_id,observation_hash,canonical_observation) VALUES (%s,%s,%s)",(observation.observation_id,observation.observation_hash,payload))
                for h in observation.evidence_artifact_hashes: cur.execute("INSERT INTO jax_evidence.observation_artifacts(observation_id,artifact_hash) VALUES (%s,%s)",(observation.observation_id,h))
            elif row[0]!=observation.observation_hash: raise ObservationIntegrityError("observation collision")
            con.commit(); return observation
        except Exception: con.rollback(); raise
        finally: con.close()
    def __write_observation_in_transaction(self, cursor, observation):
        """Internal B6/B7 composition writer; cursor is the B6 transaction."""
        from .evidence_store import _require_fixed_composition_write
        _require_fixed_composition_write()
        from .canonical import canonical_bytes
        payload=canonical_bytes(observation.projection()).decode("utf-8")
        cursor.execute("INSERT INTO jax_evidence.enforcement_observations(observation_id,observation_hash,canonical_observation) VALUES (%s,%s,%s)", (observation.observation_id,observation.observation_hash,payload))
        for h in observation.evidence_artifact_hashes:
            cursor.execute("INSERT INTO jax_evidence.observation_artifacts(observation_id,artifact_hash) VALUES (%s,%s)", (observation.observation_id,h))
    def __record_assertion(self, assertion):
        from .evidence_store import _require_fixed_composition_write
        _require_fixed_composition_write()
        con=self._connection_factory()
        try:
            cur=con.cursor(); payload=canonical_bytes(assertion.projection()).decode("utf-8")
            cur.execute("SELECT canonical_assertion FROM jax_evidence.enforcement_assertions WHERE assertion_hash=%s FOR UPDATE",(assertion.assertion_hash,)); row=cur.fetchone()
            if row is None:
                cur.execute("INSERT INTO jax_evidence.enforcement_assertions(assertion_hash,canonical_assertion) VALUES (%s,%s)",(assertion.assertion_hash,payload))
                for h in assertion.evidence_artifact_hashes:
                    cur.execute("INSERT INTO jax_evidence.assertion_artifacts(assertion_hash,artifact_hash) VALUES (%s,%s)",(assertion.assertion_hash,h))
                for oid in assertion.observation_ids:
                    cur.execute("INSERT INTO jax_evidence.assertion_observations(assertion_hash,observation_id) VALUES (%s,%s)",(assertion.assertion_hash,oid))
            elif (bytes(row[0]).decode() if isinstance(row[0],bytes) else row[0]) != payload: raise AssertionIntegrityError("assertion collision")
            con.commit(); return assertion
        except Exception: con.rollback(); raise
        finally: con.close()
    def load_observation(self, observation_id):
        con=self._connection_factory()
        try:
            cur=con.cursor(); cur.execute("SELECT observation_hash,canonical_observation FROM jax_evidence.enforcement_observations WHERE observation_id=%s",(observation_id,)); row=cur.fetchone()
            if row is None: raise ObservationIntegrityError("observation missing")
            from .observations import deserialize_enforcement_observation
            value=deserialize_enforcement_observation(row[1])
            if value.observation_id != observation_id or value.observation_hash != row[0]: raise ObservationIntegrityError("row/canonical mismatch")
            cur.execute("SELECT artifact_hash FROM jax_evidence.observation_artifacts WHERE observation_id=%s ORDER BY artifact_hash",(observation_id,)); refs=tuple(x[0] for x in cur.fetchall())
            if refs != tuple(sorted(value.evidence_artifact_hashes)): raise ObservationIntegrityError("observation refs mismatch")
            for h in refs: self.load_evidence_artifact(h)
            from .control_registry import load_control_definition
            definition=load_control_definition(value.control_id,value.control_version)
            if definition.control_definition_hash != value.control_definition_hash: raise ObservationIntegrityError("observation control binding mismatch")
            self.load_implementation_identity(value.implementation_identity_hash)
            from .evidence_store import _loaded, _observations
            return _loaded(_observations,value)
        finally: con.close()
    load_enforcement_observation = load_observation
    def load_assertion(self, assertion_hash):
        require_hash(assertion_hash); con=self._connection_factory()
        try:
            cur=con.cursor(); cur.execute("SELECT canonical_assertion FROM jax_evidence.enforcement_assertions WHERE assertion_hash=%s",(assertion_hash,)); row=cur.fetchone()
            if row is None: raise AssertionIntegrityError("assertion missing")
            from .assertions import deserialize_enforcement_assertion
            value=deserialize_enforcement_assertion(row[0])
            if value.assertion_hash != assertion_hash: raise AssertionIntegrityError("row/canonical mismatch")
            cur.execute("SELECT artifact_hash FROM jax_evidence.assertion_artifacts WHERE assertion_hash=%s ORDER BY artifact_hash",(assertion_hash,)); refs=tuple(x[0] for x in cur.fetchall())
            if refs != tuple(sorted(value.evidence_artifact_hashes)): raise AssertionIntegrityError("assertion artifact refs mismatch")
            cur.execute("SELECT observation_id FROM jax_evidence.assertion_observations WHERE assertion_hash=%s ORDER BY observation_id",(assertion_hash,)); oids=tuple(x[0] for x in cur.fetchall())
            if oids != tuple(sorted(value.observation_ids)): raise AssertionIntegrityError("assertion observation refs mismatch")
            for h in value.evidence_artifact_hashes: self.load_evidence_artifact(h)
            for oid in value.observation_ids: self.load_observation(oid)
            from .control_registry import load_control_definition
            definition=load_control_definition(value.control_id,value.control_version)
            if definition.control_definition_hash != value.control_definition_hash: raise AssertionIntegrityError("assertion control binding mismatch")
            self.load_implementation_identity(value.implementation_identity_hash)
            from .evidence_store import _loaded, _assertions
            return _loaded(_assertions,value)
        finally: con.close()
    load_enforcement_assertion = load_assertion
    def observations(self):
        """Complete authoritative observation set; never caller-supplied."""
        con=self._connection_factory()
        try:
            cur=con.cursor(); cur.execute("SELECT observation_id FROM jax_evidence.enforcement_observations ORDER BY observation_id")
            return tuple(self.load_observation(row[0]) for row in cur.fetchall())
        finally: con.close()
    def __ingest_test_manifest(self, manifest):
        """Persist the already context-validated CI projection immutably."""
        from .evidence_store import _require_fixed_composition_write
        _require_fixed_composition_write()
        from .ids import sha256_bytes
        payload=canonical_bytes(manifest).decode("utf-8"); digest=sha256_bytes(payload.encode("utf-8"))
        con=self._connection_factory()
        try:
            cur=con.cursor(); cur.execute("SELECT canonical_manifest FROM jax_evidence.test_evidence_manifests WHERE manifest_hash=%s FOR UPDATE",(digest,)); row=cur.fetchone()
            if row is None:
                cur.execute("INSERT INTO jax_evidence.test_evidence_manifests(manifest_hash,implementation_identity_hash,repository_id,commit_sha,job_id,canonical_manifest) VALUES (%s,%s,%s,%s,%s,%s)",(digest,manifest["implementation_identity_hash"],manifest["repository_id"],manifest["commit_sha"],manifest["job_id"],payload))
            elif (bytes(row[0]).decode() if isinstance(row[0],bytes) else row[0]) != payload: raise AssertionIntegrityError("test manifest collision")
            con.commit(); return manifest
        except Exception: con.rollback(); raise
        finally: con.close()
    def _test_manifests_for(self, identity_hash):
        con=self._connection_factory()
        try:
            cur=con.cursor(); cur.execute("SELECT implementation_identity_hash,canonical_manifest FROM jax_evidence.test_evidence_manifests WHERE implementation_identity_hash=%s ORDER BY manifest_hash",(identity_hash,))
            import json
            values=[]
            for row in cur.fetchall():
                value=json.loads(bytes(row[1]).decode() if isinstance(row[1],bytes) else row[1])
                if value.get("implementation_identity_hash") != row[0] or row[0] != identity_hash: raise AssertionIntegrityError("test manifest row binding mismatch")
                values.append(value)
            return tuple(values)
        finally: con.close()
    def derive_in_repeatable_read(self, derive):
        """Run deterministic derivation over one MariaDB repeatable-read snapshot."""
        con=self._connection_factory()
        try:
            cur=con.cursor(); cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ"); cur.execute("START TRANSACTION WITH CONSISTENT SNAPSHOT")
            value=derive(cur)
            con.commit(); return value
        except Exception:
            con.rollback(); raise
        finally: con.close()
