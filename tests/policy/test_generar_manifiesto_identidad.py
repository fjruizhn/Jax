"""jax#260: sin este generador, LAS MANOS no arranca desde master.

**El problema, medido.** `las_manos/server.py::_configure_b7_trusted_runtime`
construye `TrustedImplementationIdentityProvider`, cuyo `.load()` lee
`/etc/jax/build/implementation-identity.json` y lo verifica con
`verify_build_manifest(..., repository_root="/srv/jax")`. Ese archivo no
existía en ningún lado del árbol -- `docs/operations/trusted-files.md` lo
listaba fail-closed sin decir quién lo genera. Arranque muerto con
`FileNotFoundError`, producción parada en `e09c3b3`.

**Qué prueba este archivo**, contra `scripts/generar_manifiesto_identidad.py`:

1. el manifiesto que el generador produce pasa `implementation_identity_from_projection`
   y `verify_build_manifest` -- las MISMAS funciones que usa el runtime real
   (`TrustedImplementationIdentityProvider.load()`), no una reimplementación
   propia del control -- contra una copia temporal del repo (nunca `/srv/jax`);
2. si una fuente requerida cambia DESPUÉS de generado el manifiesto, la
   verificación FALLA -- ese es el propósito del manifiesto: hace fallar el
   drift, no lo tolera;
3. un árbol con cambios sin commitear se rechaza por defecto (un manifiesto
   de producción no puede describir bytes que no están en un commit), y
   sólo con `--allow-dirty` explícito se genera igual, marcado `DIRTY`;
4. las claves del JSON de identidad son EXACTAMENTE las que
   `implementation_identity_from_projection` espera, ni de más ni de menos;
5. el archivo de salida (identidad y blob del manifiesto) queda en modo 600.

La "copia temporal" es literal: se copian los bytes REALES de cada ruta de
`_V1_REQUIRED_SOURCE_PATHS` de este mismo checkout a un `tmp_path`, se hace
`git init` + commit ahí, y el generador corre contra ESA copia -- nunca
contra este worktree ni contra `/srv/jax`. `EvidenceStore` (el store en
memoria de test) sustituye a MariaDB: `verify_build_manifest` sólo necesita
`store.get_evidence_blob`, que el store en memoria cumple igual que
`MariaDBEvidenceStore` (mismo contrato, `policy/enforcement_evidence/evidence_store.py`).

Corre con:
  PYTHONPATH=. python -m pytest tests/policy/test_generar_manifiesto_identidad.py -v

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import json
import stat
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "generar_manifiesto_identidad.py"

from policy.enforcement_evidence.evidence_store import EvidenceStore
from policy.enforcement_evidence.implementation_identity import (
    _V1_REQUIRED_SOURCE_PATHS,
    implementation_identity_from_projection,
    verify_build_manifest,
)
from policy.enforcement_evidence.errors import UntrustedImplementationIdentityError

_EXPECTED_IDENTITY_KEYS = {
    "schema_version", "kind", "repository_id", "source_revision_kind",
    "git_commit_sha", "git_tree_id", "source_state",
    "build_manifest_blob_hash", "schema_versions", "build_id",
}


def _run(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def _copia_temporal_del_repo(tmp_path: Path) -> Path:
    """Copia SOLO las fuentes requeridas (bytes reales de este checkout) a
    un repo git nuevo en tmp_path -- nunca toca /srv/jax ni este worktree."""
    destino = tmp_path / "repo"
    for rel in sorted(_V1_REQUIRED_SOURCE_PATHS):
        origen = _REPO_ROOT / rel
        objetivo = destino / rel
        objetivo.parent.mkdir(parents=True, exist_ok=True)
        objetivo.write_bytes(origen.read_bytes())
    for cmd in (
        ["git", "init", "-q"],
        ["git", "config", "user.email", "test@example.invalid"],
        ["git", "config", "user.name", "Test"],
        ["git", "remote", "add", "origin", "git@github.com:fjruizhn/Jax.git"],
        ["git", "add", "-A"],
        ["git", "commit", "-q", "-m", "snapshot de prueba"],
    ):
        result = _run(cmd, cwd=destino)
        assert result.returncode == 0, result.stderr
    return destino


def _generar(repo_root: Path, output: Path, manifest_output: Path, *extra: str) -> subprocess.CompletedProcess:
    return _run([
        sys.executable, str(_SCRIPT),
        "--repo-root", str(repo_root),
        "--output", str(output),
        "--manifest-output", str(manifest_output),
        *extra,
    ])


def test_manifiesto_generado_pasa_la_verificacion_real(tmp_path):
    repo = _copia_temporal_del_repo(tmp_path)
    output = tmp_path / "out" / "implementation-identity.json"
    manifest_output = tmp_path / "out" / "implementation-identity.manifest.json"

    result = _generar(repo, output, manifest_output)
    assert result.returncode == 0, result.stderr

    data = json.loads(output.read_text())
    identity = implementation_identity_from_projection(data)
    assert identity.source_state.value == "CLEAN"
    assert identity.git_commit_sha == _run(["git", "rev-parse", "HEAD"], cwd=repo).stdout.strip()
    assert identity.git_tree_id == _run(["git", "rev-parse", "HEAD^{tree}"], cwd=repo).stdout.strip()
    assert identity.repository_id == "fjruizhn/Jax"

    store = EvidenceStore()
    store.put_evidence_blob(manifest_output.read_bytes())
    verified = verify_build_manifest(store, identity, repository_root=str(repo))
    assert set(verified["files"]) == _V1_REQUIRED_SOURCE_PATHS


def test_manifiesto_falla_verificacion_si_una_fuente_cambia_despues(tmp_path):
    repo = _copia_temporal_del_repo(tmp_path)
    output = tmp_path / "out" / "implementation-identity.json"
    manifest_output = tmp_path / "out" / "implementation-identity.manifest.json"
    result = _generar(repo, output, manifest_output)
    assert result.returncode == 0, result.stderr

    identity = implementation_identity_from_projection(json.loads(output.read_text()))
    store = EvidenceStore()
    store.put_evidence_blob(manifest_output.read_bytes())

    alguna_fuente = repo / sorted(_V1_REQUIRED_SOURCE_PATHS)[0]
    alguna_fuente.write_bytes(alguna_fuente.read_bytes() + b"\n# drift post-generacion\n")

    with pytest.raises(UntrustedImplementationIdentityError):
        verify_build_manifest(store, identity, repository_root=str(repo))


def test_arbol_sucio_se_rechaza_por_defecto(tmp_path):
    repo = _copia_temporal_del_repo(tmp_path)
    alguna_fuente = repo / sorted(_V1_REQUIRED_SOURCE_PATHS)[0]
    alguna_fuente.write_bytes(alguna_fuente.read_bytes() + b"\n# sin commitear\n")
    output = tmp_path / "out" / "implementation-identity.json"
    manifest_output = tmp_path / "out" / "implementation-identity.manifest.json"

    result = _generar(repo, output, manifest_output)

    assert result.returncode != 0
    assert "sin commitear" in result.stderr or "allow-dirty" in result.stderr
    assert not output.exists()
    assert not manifest_output.exists()


def test_arbol_sucio_con_allow_dirty_queda_marcado_dirty(tmp_path):
    repo = _copia_temporal_del_repo(tmp_path)
    alguna_fuente = repo / sorted(_V1_REQUIRED_SOURCE_PATHS)[0]
    alguna_fuente.write_bytes(alguna_fuente.read_bytes() + b"\n# sin commitear\n")
    output = tmp_path / "out" / "implementation-identity.json"
    manifest_output = tmp_path / "out" / "implementation-identity.manifest.json"

    result = _generar(repo, output, manifest_output, "--allow-dirty")

    assert result.returncode == 0, result.stderr
    data = json.loads(output.read_text())
    assert data["source_state"] == "DIRTY"


def test_claves_exactas_sin_de_mas_ni_de_menos(tmp_path):
    repo = _copia_temporal_del_repo(tmp_path)
    output = tmp_path / "out" / "implementation-identity.json"
    manifest_output = tmp_path / "out" / "implementation-identity.manifest.json"
    result = _generar(repo, output, manifest_output)
    assert result.returncode == 0, result.stderr

    data = json.loads(output.read_text())
    assert set(data) == _EXPECTED_IDENTITY_KEYS


def test_archivos_de_salida_quedan_en_modo_600(tmp_path):
    repo = _copia_temporal_del_repo(tmp_path)
    output = tmp_path / "out" / "sub" / "implementation-identity.json"
    manifest_output = tmp_path / "out" / "sub" / "implementation-identity.manifest.json"
    result = _generar(repo, output, manifest_output)
    assert result.returncode == 0, result.stderr

    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert stat.S_IMODE(manifest_output.stat().st_mode) == 0o600


def test_no_imprime_nada_que_parezca_un_secreto(tmp_path):
    repo = _copia_temporal_del_repo(tmp_path)
    output = tmp_path / "out" / "implementation-identity.json"
    manifest_output = tmp_path / "out" / "implementation-identity.manifest.json"
    result = _generar(repo, output, manifest_output)
    assert result.returncode == 0, result.stderr
    salida = (result.stdout + result.stderr).lower()
    for palabra in ("password", "secret", "token", "private_key", "jax_db_password"):
        assert palabra not in salida
