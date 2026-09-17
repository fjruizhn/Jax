# tests/test_ejecutor_freno_remoto.py
"""Comando forzado del freno (C4): mata lo de la cuenta salvo su sesión y el sshd que la
sostiene, y dice cuántos quedan. Se prueba con un `ps` FALSO que sólo lista procesos
creados por el test: correrlo de verdad como el usuario del test mataría su sesión entera."""
import os
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
GUION = RAIZ / "ops" / "ejecutor" / "ejecutor-freno-remoto"

_PS_FALSO = """#!{python}
import sys
if "-p" in sys.argv:
    print(" 1")
    sys.exit(0)
for linea in open({lista!r}):
    pid, sid = linea.split()
    try:
        estado = open(f"/proc/{{pid}}/stat").read().rsplit(")", 1)[1].split()[0]
    except FileNotFoundError:
        continue
    if estado != "Z":
        print(f" {{pid}} {{sid}}")
"""


def _entorno(tmp_path, victimas, de_su_sesion=()):
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    lista = tmp_path / "pids"
    lista.write_text("".join(f"{p.pid} 999\n" for p in victimas) + "".join(f"{p.pid} 1\n" for p in de_su_sesion))
    (bin_ / "ps").write_text(_PS_FALSO.format(python=sys.executable, lista=str(lista)))
    (bin_ / "id").write_text("#!/bin/sh\necho cuenta-de-prueba\n")
    for f in ("ps", "id"):
        (bin_ / f).chmod(0o755)
    return {"PATH": f"{bin_}:/usr/bin:/bin"}, lista


def _vivo(pid):
    try:
        estado = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except FileNotFoundError:
        return False
    return estado != "Z"


def test_mata_lo_de_otras_sesiones_y_dice_cero(tmp_path):
    victimas = [subprocess.Popen(["sleep", "60"], start_new_session=True) for _ in range(2)]
    try:
        env, _ = _entorno(tmp_path, victimas)
        r = subprocess.run([str(GUION)], env=env, capture_output=True, timeout=20)
        assert r.stdout == b"freno_remoto=ok quedan=0\n", r
        assert [_vivo(p.pid) for p in victimas] == [False, False]
    finally:
        for p in victimas:
            p.kill()
            p.wait(5)


def test_no_mata_su_propia_sesion(tmp_path):
    propia = subprocess.Popen(["sleep", "60"], start_new_session=True)
    try:
        env, _ = _entorno(tmp_path, [], de_su_sesion=[propia])
        r = subprocess.run([str(GUION)], env=env, capture_output=True, timeout=20)
        assert r.stdout == b"freno_remoto=ok quedan=0\n"
        assert _vivo(propia.pid) is True
    finally:
        propia.kill()
        propia.wait(5)


def test_no_mata_al_sshd_que_sostiene_la_sesion(tmp_path):
    """Con OpenSSH el padre del comando forzado (`sshd-session: cuenta@notty`) corre como la
    cuenta y en OTRA sesión: matarlo cortaría la conexión antes de la respuesta."""
    env, lista = _entorno(tmp_path, [])
    padre = (f"import os, subprocess, sys\n"
             f"open({str(lista)!r}, 'a').write(f'{{os.getpid()}} 999\\n')\n"
             f"r = subprocess.run([{str(GUION)!r}], capture_output=True, timeout=20)\n"
             f"sys.stdout.buffer.write(r.stdout)\n")
    r = subprocess.run([sys.executable, "-c", padre], env=env, capture_output=True, timeout=30)
    assert r.returncode == 0, r
    assert r.stdout == b"freno_remoto=ok quedan=0\n"


def test_el_guion_es_ejecutable_y_sh():
    assert os.access(GUION, os.X_OK)
    assert GUION.read_text().startswith("#!/bin/sh\n")
