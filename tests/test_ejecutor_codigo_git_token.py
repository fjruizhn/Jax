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
