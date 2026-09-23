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

import importlib.util
import json
import os
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


# --- Ronda de revisión PR#264 -------------------------------------------
#
# 6. repository_id nunca lleva credenciales de un remoto HTTPS.
# 7. --output es obligatorio salvo --production (nunca un default apuntando
#    silenciosamente a /etc/jax/build/).
# 8. --production y --output juntos es un error explícito, no "el último gana".
# 9. arbol_sucio() no puede leer CLEAN cuando git status advirtió por stderr
#    con exit 0 (el caso real: 'dubious ownership' o Permission denied sobre
#    el checkout de producción, jaxsvc:jaxsvc corrido por otro usuario).


def _cargar_modulo_generador():
    """Carga scripts/generar_manifiesto_identidad.py como módulo Python para
    probar su lógica pura (resolver_salidas) sin pasar por subprocess ni
    arriesgar una escritura real en /etc/jax/."""
    spec = importlib.util.spec_from_file_location("generar_manifiesto_identidad", _SCRIPT)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def test_repository_id_sin_credenciales_en_remoto_https(tmp_path):
    # Deliberadamente NO github.com: la v1 de este script (PR#264 ronda 1)
    # partía el remoto con `url.split("github.com/", 1)[-1]` -- un remoto
    # https://user:token@github.com/... "funcionaba" por COINCIDENCIA (el
    # split cae justo después de las credenciales), pero cualquier remoto
    # que no tenga ese substring literal (un espejo interno, GitLab, un
    # proxy corporativo) caía al `else: tail = url` y filtraba la URL
    # COMPLETA, credenciales incluidas, a una tabla con triggers
    # no-update/no-delete. Este host prueba el caso real.
    repo = _copia_temporal_del_repo(tmp_path)
    token = "ghp_SuperSecretoDeMentira1234567890"
    remoto = f"https://x-access-token:{token}@git.axioma-ia.internal/fjruizhn/Jax.git"
    result = _run(["git", "remote", "set-url", "origin", remoto], cwd=repo)
    assert result.returncode == 0, result.stderr

    output = tmp_path / "out" / "implementation-identity.json"
    manifest_output = tmp_path / "out" / "implementation-identity.manifest.json"
    result = _generar(repo, output, manifest_output)
    assert result.returncode == 0, result.stderr

    data = json.loads(output.read_text())
    assert data["repository_id"] == "fjruizhn/Jax"
    assert token not in output.read_text()
    assert token not in manifest_output.read_text()
    assert token not in (result.stdout + result.stderr)


def test_output_es_obligatorio_sin_production(tmp_path):
    repo = _copia_temporal_del_repo(tmp_path)
    result = _run([sys.executable, str(_SCRIPT), "--repo-root", str(repo)])
    assert result.returncode != 0
    assert "--output" in result.stderr
    assert "--production" in result.stderr


def test_output_y_production_juntos_es_error(tmp_path):
    repo = _copia_temporal_del_repo(tmp_path)
    result = _run([
        sys.executable, str(_SCRIPT), "--repo-root", str(repo),
        "--production", "--output", str(tmp_path / "otra-ruta.json"),
    ])
    assert result.returncode != 0
    assert "--production" in result.stderr


def test_resolver_salidas_production_usa_la_ruta_fija_de_despliegue():
    modulo = _cargar_modulo_generador()
    args = modulo._construir_parser().parse_args(["--repo-root", "/inexistente", "--production"])
    output, manifest_output = modulo.resolver_salidas(args)
    assert output == modulo._DEFAULT_OUTPUT
    assert manifest_output == modulo._DEFAULT_OUTPUT.with_name(
        modulo._DEFAULT_OUTPUT.name + ".manifest.json"
    )


def test_resolver_salidas_sin_output_ni_production_falla():
    modulo = _cargar_modulo_generador()
    args = modulo._construir_parser().parse_args(["--repo-root", "/inexistente"])
    with pytest.raises(SystemExit):
        modulo.resolver_salidas(args)


def test_resolver_salidas_output_y_production_juntos_falla():
    modulo = _cargar_modulo_generador()
    args = modulo._construir_parser().parse_args([
        "--repo-root", "/inexistente", "--production", "--output", "/tmp/x.json",
    ])
    with pytest.raises(SystemExit):
        modulo.resolver_salidas(args)


_FAKE_GIT_QUE_ADVIERTE_EN_STATUS = """#!/bin/sh
# Simula el caso real de produccion: 'git status --porcelain' advierte por
# stderr (dubious ownership, Permission denied sobre un archivo puntual...)
# pero igual sale con codigo 0. $3 es el subcomando porque el script llama
# siempre "git -C <root> <subcomando> ...".
if [ "$3" = "status" ]; then
  echo "warning: simulacion de advertencia de git para test (stderr, exit 0)" >&2
  exit 0
fi
exec /usr/bin/git "$@"
"""


def test_arbol_sucio_se_niega_si_git_status_advierte_por_stderr_aunque_exit_sea_0(tmp_path):
    repo = _copia_temporal_del_repo(tmp_path)
    fake_bin = tmp_path / "fakebin"
    fake_bin.mkdir()
    fake_git = fake_bin / "git"
    fake_git.write_text(_FAKE_GIT_QUE_ADVIERTE_EN_STATUS)
    fake_git.chmod(0o755)

    env = dict(os.environ)
    env["PATH"] = f"{fake_bin}{os.pathsep}{env.get('PATH', '')}"

    output = tmp_path / "out" / "implementation-identity.json"
    manifest_output = tmp_path / "out" / "implementation-identity.manifest.json"
    result = subprocess.run(
        [sys.executable, str(_SCRIPT), "--repo-root", str(repo),
         "--output", str(output), "--manifest-output", str(manifest_output)],
        capture_output=True, text=True, env=env,
    )

    assert result.returncode != 0
    assert "no se puede confiar" in result.stderr
    assert not output.exists()
    assert not manifest_output.exists()


def test_permission_error_al_escribir_no_sale_como_traceback_crudo(tmp_path):
    """BLOQUE-1 de la ronda 2 de revisión (PR#264): las dos llamadas a
    escribir_atomico() vivían FUERA del try/except de main(). El caso real:
    /etc/jax/build/ no existe todavía y jaxsvc no puede crearlo bajo
    /etc/jax (root:root 755) -- PermissionError sin capturar, traceback
    crudo en vez de un `ERROR: ...` con exit 1."""
    repo = _copia_temporal_del_repo(tmp_path)
    sin_permiso = tmp_path / "sin-permiso"
    sin_permiso.mkdir()
    sin_permiso.chmod(0o000)
    try:
        output = sin_permiso / "sub" / "implementation-identity.json"
        manifest_output = sin_permiso / "sub" / "implementation-identity.manifest.json"
        result = _generar(repo, output, manifest_output)

        assert result.returncode != 0
        assert result.stderr.strip().startswith("ERROR:")
        assert "Traceback" not in result.stderr
    finally:
        sin_permiso.chmod(0o755)  # para que tmp_path se pueda limpiar solo
