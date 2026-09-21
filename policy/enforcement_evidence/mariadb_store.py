"""Synchronous authoritative MariaDB blob store.

It deliberately exposes only bytes by content identity.  Higher level objects
are parsed as untrusted values and must be verified by their fixed lifecycle.
"""
from __future__ import annotations
from .evidence_store import EvidenceBlob, MAX_BLOB_BYTES
from .ids import sha256_bytes, require_hash
from .errors import EvidenceBlobMissingError, EvidenceBlobHashMismatchError, EvidenceBlobTooLargeError
from .canonical import canonical_bytes
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
    def record_artifact(self, artifact):
        """Persist only after every referenced blob is authoritatively readable."""
        for ref in artifact.blob_refs: self.get_evidence_blob(ref.evidence_hash)
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
            from .evidence_store import _seal, _artifacts
            return _seal(_artifacts,value)
        finally: con.close()
    def record_observation(self, observation):
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
    def record_assertion(self, assertion):
        con=self._connection_factory()
        try:
            cur=con.cursor(); payload=canonical_bytes(assertion.projection()).decode("utf-8")
            cur.execute("SELECT canonical_assertion FROM jax_evidence.enforcement_assertions WHERE assertion_hash=%s FOR UPDATE",(assertion.assertion_hash,)); row=cur.fetchone()
            if row is None: cur.execute("INSERT INTO jax_evidence.enforcement_assertions(assertion_hash,canonical_assertion) VALUES (%s,%s)",(assertion.assertion_hash,payload))
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
            from .evidence_store import _seal, _observations
            return _seal(_observations,value)
        finally: con.close()
