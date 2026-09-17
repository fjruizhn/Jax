# tests/test_ejecutor_revocar_sh.py
"""ejecutor-revocar (C6): quita las llaves de acceso de la cuenta, deja la del freno,
guarda lo quitado y mata las sesiones vivas. pkill es FALSO: anota y sale."""
import subprocess
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
GUION = RAIZ / "ops" / "ejecutor" / "ejecutor-revocar"
LLAVES = (
    "ssh-ed25519 AAAAejecutor ejecutor-axioma\n"
    'command="/usr/local/sbin/ejecutor-freno-remoto",restrict ssh-ed25519 AAAAfreno ejecutor-freno\n'
    "ssh-ed25519 AAAAcontrolador ejecutor-controlador\n"
)


def _entorno(tmp_path, rc_pkill=1):
    bin_ = tmp_path / "bin"
    bin_.mkdir(exist_ok=True)
    (bin_ / "pkill").write_text(f"#!/bin/sh\necho \"$@\" >> {tmp_path / 'pkill.log'}\nexit {rc_pkill}\n")
    (bin_ / "pkill").chmod(0o755)
    return {"PATH": f"{bin_}:/usr/bin:/bin"}


def _correr(tmp_path, *args, rc_pkill=1):
    return subprocess.run([str(GUION), *args], env=_entorno(tmp_path, rc_pkill), capture_output=True, timeout=20)


def test_quita_acceso_deja_freno_guarda_y_mata(tmp_path):
    archivo = tmp_path / "axioma"
    archivo.write_text(LLAVES)
    r = _correr(tmp_path, str(archivo), "axioma")
    assert (r.returncode, r.stdout) == (0, b"revocar=ok quitadas=2 quedan=0\n")
    assert archivo.read_text() == 'command="/usr/local/sbin/ejecutor-freno-remoto",restrict ssh-ed25519 AAAAfreno ejecutor-freno\n'
    (guardado,) = tmp_path.glob("axioma.revocadas-*")
    assert guardado.read_text() == "ssh-ed25519 AAAAejecutor ejecutor-axioma\nssh-ed25519 AAAAcontrolador ejecutor-controlador\n"
    assert (tmp_path / "pkill.log").read_text() == "-KILL -u axioma\n"


def test_sin_procesos_vivos_tambien_es_ok(tmp_path):
    archivo = tmp_path / "axioma"
    archivo.write_text(LLAVES)
    assert _correr(tmp_path, str(archivo), "axioma", rc_pkill=0).returncode == 0


def test_pkill_que_falla_de_verdad_es_error(tmp_path):
    archivo = tmp_path / "axioma"
    archivo.write_text(LLAVES)
    r = _correr(tmp_path, str(archivo), "axioma", rc_pkill=3)
    assert r.returncode == 2 and r.stdout == b"revocar=error codigo=pkill\n"


def test_revocar_dos_veces_es_idempotente(tmp_path):
    archivo = tmp_path / "axioma"
    archivo.write_text(LLAVES)
    _correr(tmp_path, str(archivo), "axioma")
    r = _correr(tmp_path, str(archivo), "axioma")
    assert r.stdout == b"revocar=ok quitadas=0 quedan=0\n"


def test_argumentos_y_archivo(tmp_path):
    assert _correr(tmp_path).stdout == b"revocar=error codigo=argumentos\n"
    assert _correr(tmp_path, str(tmp_path / "no-existe"), "axioma").stdout == b"revocar=error codigo=sin_archivo\n"


def test_archivo_ilegible_no_se_vacia(tmp_path):
    archivo = tmp_path / "axioma"
    archivo.write_text(LLAVES)
    archivo.chmod(0o000)
    try:
        r = _correr(tmp_path, str(archivo), "axioma")
    finally:
        archivo.chmod(0o644)
    assert (r.returncode, r.stdout) == (2, b"revocar=error codigo=lectura\n")
    assert archivo.read_text() == LLAVES
    assert not (tmp_path / "pkill.log").exists()
    assert list(tmp_path.glob("axioma.*")) == []
