"""Entrega de la misión de código (spec 2026-09-28 §3.3): la única pieza con el
token, y solo empuja `axioma/<misión>` -- nunca la rama por omisión, ni ninguna
otra. El token nunca viaja en argv, en la URL ni en `.git/config`: entra al
subproceso de `git` por `GIT_ASKPASS`, leyendo una variable del entorno DEL
SUBPROCESO (`git_token.entorno_git`)."""
from __future__ import annotations

import asyncio
import subprocess

import httpx
import pytest

from jax.ejecutor.codigo import entrega as E

MID = "3f1c9a2e-7b4d-4e8a-9c1f-2a3b4c5d6e7f"


def test_referencia_permitida():
    assert E.referencia_permitida(f"axioma/{MID}", "main", MID)
    assert not E.referencia_permitida("main", "main", MID)
    assert not E.referencia_permitida(f"axioma/{MID}", f"axioma/{MID}", MID)  # por omisión aunque coincida
    assert not E.referencia_permitida("axioma/otra", "main", MID)
    with pytest.raises(ValueError):
        E.rama_de_la_mision("no-es-uuid")


def _git(*a, cwd):
    subprocess.run(["git", *a], cwd=cwd, check=True, capture_output=True)


def test_empujar_solo_la_rama_de_la_mision(tmp_path):
    remoto = tmp_path / "remoto.git"
    _git("init", "--bare", "-b", "main", str(remoto), cwd=tmp_path)
    clon = tmp_path / "clon"
    _git("clone", str(remoto), str(clon), cwd=tmp_path)
    for k, v in (("user.name", "t"), ("user.email", "t@t")):
        _git("config", k, v, cwd=clon)
    (clon / "a").write_text("1")
    _git("add", "a", cwd=clon)
    _git("commit", "-m", "base", cwd=clon)
    _git("push", "origin", "main", cwd=clon)
    _git("checkout", "-b", f"axioma/{MID}", cwd=clon)
    (clon / "a").write_text("2")
    _git("commit", "-am", "cambio", cwd=clon)

    asyncio.run(E.empujar(clon, mision_id=MID, rama_por_omision="main", remoto_url=str(remoto), token="falso"))

    ramas = subprocess.run(["git", "branch", "--list"], cwd=remoto, capture_output=True, text=True).stdout
    assert f"axioma/{MID}" in ramas
    main = subprocess.run(["git", "log", "--oneline", "main"], cwd=remoto, capture_output=True, text=True).stdout
    assert "cambio" not in main


def test_empujar_niega_si_head_no_es_la_rama(tmp_path):
    """(Fallo del controlador 2026-09-28): un repo recién inicializado sin ningún commit
    no tiene HEAD resoluble -- `git rev-parse --abbrev-ref HEAD` fallaría con
    `git_fallo`, no con el rechazo de rama que este test quiere ejercitar. Se hace un
    commit en `main` primero para que HEAD sea de verdad "main" (una rama no permitida),
    y `empujar` llegue al chequeo de `referencia_permitida` y lo rechace ahí."""
    clon = tmp_path / "c"
    _git("init", "-b", "main", str(clon), cwd=tmp_path)
    for k, v in (("user.name", "t"), ("user.email", "t@t")):
        _git("config", k, v, cwd=clon)
    (clon / "a").write_text("1")
    _git("add", "a", cwd=clon)
    _git("commit", "-m", "base", cwd=clon)

    with pytest.raises(E.EntregaRechazada, match="rama_no_permitida"):
        asyncio.run(E.empujar(clon, mision_id=MID, rama_por_omision="main", remoto_url="x", token="t"))


def test_pr_nuevo_y_existente():
    pedidos = []
    estado = {"abiertos": []}

    def manejar(req: httpx.Request):
        pedidos.append((req.method, req.url.path))
        if req.method == "GET" and req.url.path.endswith("/pulls"):
            return httpx.Response(200, json=estado["abiertos"])
        if req.method == "POST" and req.url.path.endswith("/pulls"):
            return httpx.Response(201, json={"number": 7, "html_url": "https://gh/pr/7"})
        if req.method == "PATCH":
            return httpx.Response(200, json={"number": 7, "html_url": "https://gh/pr/7"})
        return httpx.Response(200, json=[])

    cli = httpx.AsyncClient(transport=httpx.MockTransport(manejar), base_url="https://api.github.com")
    url = asyncio.run(E.abrir_o_actualizar_pr(cli, repo="o/r", rama=f"axioma/{MID}", base="main", titulo="t",
                                              cuerpo="c"))
    assert url == "https://gh/pr/7" and ("POST", "/repos/o/r/pulls") in pedidos
    assert ("POST", "/repos/o/r/issues/7/labels") in pedidos

    estado["abiertos"] = [{"number": 7, "html_url": "https://gh/pr/7"}]
    pedidos.clear()
    asyncio.run(E.abrir_o_actualizar_pr(cli, repo="o/r", rama=f"axioma/{MID}", base="main", titulo="t", cuerpo="c2"))
    assert ("PATCH", "/repos/o/r/pulls/7") in pedidos and ("POST", "/repos/o/r/pulls") not in pedidos


# --- El token nunca en argv ni en .git/config (ruling del controlador 2026-09-28) ----

def test_empujar_no_pone_el_token_en_argv_ni_en_git_config(tmp_path, monkeypatch):
    remoto = tmp_path / "remoto.git"
    _git("init", "--bare", "-b", "main", str(remoto), cwd=tmp_path)
    clon = tmp_path / "clon"
    _git("clone", str(remoto), str(clon), cwd=tmp_path)
    for k, v in (("user.name", "t"), ("user.email", "t@t")):
        _git("config", k, v, cwd=clon)
    (clon / "a").write_text("1")
    _git("add", "a", cwd=clon)
    _git("commit", "-m", "base", cwd=clon)
    _git("push", "origin", "main", cwd=clon)
    _git("checkout", "-b", f"axioma/{MID}", cwd=clon)
    (clon / "a").write_text("2")
    _git("commit", "-am", "cambio", cwd=clon)

    token = "SECRETO-QUE-NUNCA-DEBE-APARECER-EN-ARGV"
    argv_vistos = []
    original = asyncio.create_subprocess_exec

    async def espia(*args, **kwargs):
        argv_vistos.append(args)
        return await original(*args, **kwargs)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", espia)

    asyncio.run(E.empujar(clon, mision_id=MID, rama_por_omision="main", remoto_url=str(remoto), token=token))

    for args in argv_vistos:
        assert token not in args, f"el token apareció en argv: {args}"
    cfg = (clon / ".git" / "config").read_text()
    assert token not in cfg
