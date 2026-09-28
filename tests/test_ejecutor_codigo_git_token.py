"""Todo git que corre `jaxsvc` pasa por `git_token` (spec 2026-09-28 v1.2, §3.3 y §4.2):
entorno en lista blanca, sin configuración de sistema ni global, sin ganchos, sin ayudantes
de credenciales, y el token solo lo entrega un GIT_ASKPASS que responde a github.com."""
from __future__ import annotations

import asyncio
import os
import subprocess

import pytest

from jax.ejecutor.codigo import git_token as G

TOKEN = "github_pat_FALSO_0123456789abcdefghij"


def _preguntar(script, prompt, env):
    return subprocess.run([str(script), prompt], capture_output=True, text=True, env=env)


def test_askpass_responde_solo_a_github(tmp_path):
    env = G.entorno_red(tmp_path, TOKEN)
    script = env["GIT_ASKPASS"]
    usuario = _preguntar(script, "Username for 'https://github.com': ", env)
    assert usuario.returncode == 0 and usuario.stdout.strip() == "x-access-token"
    clave = _preguntar(script, "Password for 'https://x-access-token@github.com': ", env)
    assert clave.returncode == 0 and clave.stdout.strip() == TOKEN
    for ajeno in ("Password for 'https://x-access-token@github.com.evil.io': ",
                  "Password for 'https://x-access-token@evil.io': ",
                  "Username for 'https://evil.io/github.com': ",
                  "Enter passphrase for key '/home/x/.ssh/id': "):
        r = _preguntar(script, ajeno, env)
        assert r.returncode != 0 and r.stdout == "", ajeno


def test_el_askpass_no_contiene_el_token_y_es_0700(tmp_path):
    env = G.entorno_red(tmp_path, TOKEN)
    script = tmp_path / "askpass.sh"
    assert env["GIT_ASKPASS"] == str(script)
    assert TOKEN not in script.read_text()
    assert oct(script.stat().st_mode & 0o777) == "0o700"


def test_entorno_base_en_lista_blanca(tmp_path, monkeypatch):
    monkeypatch.setenv("JAX_GITHUB_TOKEN", "no-debe-pasar")
    monkeypatch.setenv("GIT_DIR", "/otro")
    monkeypatch.setenv("https_proxy", "http://127.0.0.1:9")
    env = G.entorno_base(tmp_path)
    assert env["HOME"] == str(tmp_path)
    assert env["GIT_CONFIG_NOSYSTEM"] == "1" and env["GIT_CONFIG_GLOBAL"] == os.devnull
    assert set(env) <= {"PATH", "HOME", "LANG", "GIT_CONFIG_NOSYSTEM", "GIT_CONFIG_GLOBAL", "GIT_TERMINAL_PROMPT"}
    assert "GIT_ASKPASS" not in env and "JAX_GIT_TOKEN_EFIMERO" not in env
    red = G.entorno_red(tmp_path, TOKEN)
    assert set(red) - set(env) == {"GIT_ASKPASS", "JAX_GIT_TOKEN_EFIMERO"}
    assert not any(k.startswith("JAX_GITHUB") for k in red)


def test_correr_git_blinda_cada_llamada(tmp_path, monkeypatch):
    vistos = []
    original = asyncio.create_subprocess_exec

    async def espia(*argv, **kw):
        vistos.append((argv, kw.get("env")))
        return await original(*argv, **kw)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", espia)
    asyncio.run(G.correr_git(["init", "--bare", "--", str(tmp_path / "r.git")], env=G.entorno_base(tmp_path),
                             error="init_fallo"))
    argv, env = vistos[0]
    assert argv[:5] == ("git", "-c", "core.hooksPath=/dev/null", "-c", "credential.helper=")
    assert env["GIT_CONFIG_GLOBAL"] == os.devnull


def test_error_saneado_sin_token_y_con_tope(tmp_path):
    repo = tmp_path / "r"
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    with pytest.raises(G.GitFallo) as e:
        # `git show <x>` repite el argumento en su error: el token viaja dentro del mensaje.
        asyncio.run(G.correr_git(["-C", str(repo), "show", "x" * 400 + TOKEN + "y" * 10],
                                 env=G.entorno_base(tmp_path), error="show_fallo", sanear_con=TOKEN))
    texto = str(e.value)
    assert texto.startswith("show_fallo: ") and TOKEN not in texto and "***" in texto
    assert len(texto) <= len("show_fallo: ") + 300


def test_error_sin_stderr_cuando_la_salida_no_es_confiable(tmp_path):
    with pytest.raises(G.GitFallo) as e:
        asyncio.run(G.correr_git(["-C", str(tmp_path), "rev-parse", "--verify", "DATO-DEL-CLON"],
                                 env=G.entorno_base(tmp_path), error="traer_del_clon_fallo", mostrar_error=False))
    assert str(e.value) == "traer_del_clon_fallo" and "DATO-DEL-CLON" not in str(e.value)


def test_correr_git_con_entrada_la_pasa_por_stdin_y_sin_entrada_stdin_es_devnull(tmp_path):
    """`cat-file --batch-check` lee los oids por stdin (ruling 4b, Tarea 9)."""
    subprocess.run(["git", "init", "-q", "--bare", str(tmp_path / "r.git")], check=True)
    oid = subprocess.run(["git", "-C", str(tmp_path / "r.git"), "hash-object", "-w", "--stdin"], input=b"hola\n",
                         capture_output=True, check=True).stdout.decode().strip()
    env = G.entorno_base(tmp_path)
    salida = asyncio.run(G.correr_git(["-C", str(tmp_path / "r.git"), "cat-file", "--batch-check"], env=env,
                                      error="x", entrada=f"{oid}\n".encode()))
    assert salida.decode().split() == [oid, "blob", "5"]
    vacia = asyncio.run(G.correr_git(["-C", str(tmp_path / "r.git"), "cat-file", "--batch-check"], env=env,
                                     error="x"))
    assert vacia == b""


# --- tope de tiempo (ronda final de la auditoría, Tarea 9) ---------------------------------------

import time  # noqa: E402


def test_correr_git_con_tope_vencido_mata_el_proceso_y_falla_cerrado(tmp_path):
    """Un git que no termina (red colgada, upload-pack trabado) no cuelga el turno: al vencer el tope se
    mata el grupo de procesos entero y se lanza GitFallo."""
    marca = tmp_path / "siguio-vivo"
    env = G.entorno_base(tmp_path)
    inicio = time.monotonic()
    with pytest.raises(G.GitFallo, match="lento: tope_vencido"):
        asyncio.run(G.correr_git(["-c", f"alias.dormir=!sleep 3; touch {marca}", "dormir"], env=env,
                                 error="lento", tope_s=0.5))
    assert time.monotonic() - inicio < 2.5
    time.sleep(3.2)
    assert not marca.exists(), "el sleep hijo siguió vivo: no se mató el grupo"


def test_tope_por_omision_300_y_configurable(monkeypatch):
    monkeypatch.delenv("JAX_EJECUTOR_GIT_TOPE_S", raising=False)
    assert G.tope_por_omision() == 300.0
    monkeypatch.setenv("JAX_EJECUTOR_GIT_TOPE_S", "42")
    assert G.tope_por_omision() == 42.0
    for malo in ("0", "-1", "x", "inf", "nan"):
        monkeypatch.setenv("JAX_EJECUTOR_GIT_TOPE_S", malo)
        with pytest.raises(ValueError):
            G.tope_por_omision()
