"""Synchronous authoritative MariaDB blob store.

It deliberately exposes only bytes by content identity.  Higher level objects
are parsed as untrusted values and must be verified by their fixed lifecycle.
"""
from __future__ import annotations
from .evidence_store import EvidenceBlob, MAX_BLOB_BYTES
from .ids import sha256_bytes, require_hash
from .errors import EvidenceBlobMissingError, EvidenceBlobHashMismatchError, EvidenceBlobTooLargeError
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
