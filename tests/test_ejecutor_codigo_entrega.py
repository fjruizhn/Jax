"""Entrega de la misión de código (spec 2026-09-28 v1.2, §3.3): dos repositorios.

- El ESPEJO (`<raiz>/<misión>/espejo.git`) es privado de `jaxsvc`: único lugar donde se usa
  el token, se calcula el diff de C1 y se empuja.
- El CLON (`<raiz>/<misión>/repo`) es de Qwen: `jaxsvc` nunca corre git adentro después de
  prepararlo. Los commits se traen al espejo con `git fetch --upload-pack=<ssh como axioma>`,
  así `upload-pack` corre como la cuenta (`man git`, SECURITY).

Los tests usan git real y repos bare locales: el "GitHub" es un bare en `file://` y
`upload-pack` corre local (sin ssh)."""
from __future__ import annotations

import asyncio
import os
import shlex
import subprocess
from pathlib import Path

import httpx
import pytest

from jax.ejecutor.codigo import entrega as E
from jax.ejecutor.codigo import preparar as P
from jax.ejecutor.contratos.cuenta_axioma import Cuenta

MID = "3f1c9a2e-7b4d-4e8a-9c1f-2a3b4c5d6e7f"
RAMA = f"axioma/{MID}"
TOKEN = "github_pat_TOKEN_FALSO_QUE_NUNCA_DEBE_QUEDAR_EN_DISCO_0123"


def _git(*a, cwd, **kw):
    return subprocess.run(["git", *a], cwd=cwd, check=True, capture_output=True, text=True, **kw).stdout.strip()


def _commit(repo: Path, archivo: str, texto: str, mensaje: str, *extra: str) -> str:
    (repo / archivo).write_text(texto)
    _git("add", archivo, cwd=repo)
    _git("-c", "user.name=q", "-c", "user.email=q@q", "commit", *extra, "-m", mensaje, cwd=repo)
    return _git("rev-parse", "HEAD", cwd=repo)


def _refs(bare: Path) -> dict[str, str]:
    salida = _git("for-each-ref", "--format=%(refname) %(objectname)", cwd=bare)
    return dict(l.split(" ", 1) for l in salida.splitlines() if l)


@pytest.fixture
def github(tmp_path, monkeypatch):
    """Un 'GitHub' local: `<tmp>/gh/o/r.git` con un commit en `main`. La URL del espejo
    sigue siendo DERIVADA de owner_repo; solo cambia la base (`E.GITHUB`)."""
    gh = tmp_path / "gh"
    remoto = gh / "o" / "r.git"
    remoto.parent.mkdir(parents=True)
    _git("init", "-q", "--bare", "-b", "main", str(remoto), cwd=tmp_path)
    w = tmp_path / "w"
    _git("clone", "-q", str(remoto), str(w), cwd=tmp_path)
    _commit(w, "a", "1", "base")
    _git("push", "-q", "origin", "main", cwd=w)
    monkeypatch.setattr(E, "GITHUB", f"file://{gh}/")
    return remoto


class _Registro:
    def __init__(self):
        self.llamadas = []

    async def __call__(self, ruta):
        self.llamadas.append(ruta)


def _accesos():
    return P.Accesos(paso=_Registro(), escritura=_Registro(), lectura=_Registro())


async def _sin_instalar(argv, cwd, env):
    raise AssertionError(f"no debía instalar nada: {argv}")


def _preparar(tmp_path) -> P.Clon:
    return asyncio.run(P.preparar(P.Repo("o/r", ()), mision_id=MID, raiz=tmp_path / "m", rama_por_omision="main",
                                  token=TOKEN, autor="Axioma (Ejecutor) <axioma@axioma-ia.io>",
                                  accesos=_accesos(), instalar=_sin_instalar))


class _GitHubApi:
    """La API de PRs como un MockTransport con estado."""

    def __init__(self, respuesta_post=None):
        self.prs: list[dict] = []
        self.pedidos: list[tuple[str, str, dict]] = []
        self.respuesta_post = respuesta_post

    def __call__(self, req: httpx.Request) -> httpx.Response:
        import json
        cuerpo = json.loads(req.content) if req.content else {}
        self.pedidos.append((req.method, req.url.path, cuerpo))
        if req.method == "GET" and req.url.path == "/repos/o/r/pulls":
            estado = req.url.params.get("state")
            assert req.url.params.get("head") == f"o:{RAMA}"
            return httpx.Response(200, json=[p for p in self.prs if estado in (p["state"], "all")])
        if req.method == "POST" and req.url.path == "/repos/o/r/pulls":
            if self.respuesta_post is not None:
                r, self.respuesta_post = self.respuesta_post, None
                return r
            n = len(self.prs) + 1
            self.prs.append({"number": n, "state": "open", "html_url": f"https://gh/pr/{n}"})
            return httpx.Response(201, json=self.prs[-1])
        if req.method == "PATCH":
            n = int(req.url.path.rsplit("/", 1)[1])
            return httpx.Response(200, json=next(p for p in self.prs if p["number"] == n))
        if req.method == "POST" and req.url.path.endswith("/labels"):
            return httpx.Response(200, json=[{"name": "axioma"}])
        return httpx.Response(404, json={})

    def cliente(self):
        return httpx.AsyncClient(transport=httpx.MockTransport(self), base_url="https://api.github.com")

    def metodos(self):
        return [(m, p) for m, p, _ in self.pedidos]


def _entregar(c: P.Clon, api: _GitHubApi, *, token=TOKEN):
    async def todo():
        await E.traer_del_clon(c.espejo, c.ruta, mision_id=MID, upload_pack="git-upload-pack")
        cambios = await E.diff_en_el_espejo(c.espejo, mision_id=MID, rama_por_omision=c.rama_por_omision)
        await E.empujar(c.espejo, mision_id=MID, rama_por_omision=c.rama_por_omision, token=token)
        async with api.cliente() as cli:
            pr = await E.abrir_o_actualizar_pr(cli, repo="o/r", rama=c.rama, base=c.rama_por_omision, titulo="t",
                                               cuerpo="c")
        return cambios, pr
    return asyncio.run(todo())


# --- validación -------------------------------------------------------------------------

def test_referencia_permitida():
    assert E.referencia_permitida(RAMA, "main", MID)
    assert not E.referencia_permitida("main", "main", MID)
    assert not E.referencia_permitida(RAMA, RAMA, MID)  # por omisión aunque coincida
    assert not E.referencia_permitida("axioma/otra", "main", MID)
    with pytest.raises(ValueError):
        E.rama_de_la_mision("no-es-uuid")


@pytest.mark.parametrize("malo", ["o", "o/r/x", "o/r;id", "o/r x", "../..", "o/..", "https://evil/o/r", "",
                                  "o/r\n", "ó/r"])
def test_owner_repo_invalido(malo):
    with pytest.raises(ValueError, match="owner_repo_invalido"):
        E.url_del_repo(malo)


def test_url_derivada_de_owner_repo():
    assert E.url_del_repo("fjruizhn/jax-platform") == "https://github.com/fjruizhn/jax-platform.git"


def test_empujar_rechaza_la_rama_por_omision_antes_de_git(tmp_path, monkeypatch):
    async def prohibido(*a, **k):
        raise AssertionError("no debía correr ningún comando")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", prohibido)
    with pytest.raises(E.EntregaRechazada, match="rama_no_permitida"):
        asyncio.run(E.empujar(tmp_path / "espejo.git", mision_id=MID, rama_por_omision=RAMA, token=TOKEN))


def test_traer_del_clon_rechaza_una_ruta_que_no_viaja_segura_por_ssh(tmp_path):
    with pytest.raises(ValueError, match="ruta_insegura"):
        asyncio.run(E.traer_del_clon(tmp_path / "e.git", tmp_path / "con espacio", mision_id=MID,
                                     upload_pack="git-upload-pack"))


def test_upload_pack_por_ssh_corre_como_la_cuenta(tmp_path):
    c = Cuenta("axioma", 2222, Path("/etc/jax-ejecutor/llave"), tmp_path, tmp_path, tmp_path, tmp_path)
    argv = shlex.split(E.upload_pack_por_ssh(c))
    assert argv[0] == "ssh" and "BatchMode=yes" in argv and "StrictHostKeyChecking=yes" in argv
    assert argv[argv.index("-p") + 1] == "2222" and argv[argv.index("-i") + 1] == "/etc/jax-ejecutor/llave"
    assert argv[-2:] == ["axioma@127.0.0.1", "git-upload-pack"]


# --- el clon malicioso ------------------------------------------------------------------

_GANCHOS = ("pre-push", "reference-transaction", "post-update", "pre-receive", "update", "post-receive",
            "post-checkout", "post-merge", "pre-auto-gc", "post-rewrite", "push-to-checkout", "pre-commit",
            "fsmonitor-watchman", "post-index-change")


def test_clon_malicioso_no_ejecuta_nada_ni_filtra_el_token(tmp_path, github):
    c = _preparar(tmp_path)
    base_main = _refs(github)["refs/heads/main"]

    # Qwen trabaja: un commit en su rama, y además mueve `main` y crea una etiqueta locales.
    qwen = _commit(c.ruta, "a", "2", "cambio de qwen")
    _git("branch", "-f", "main", "HEAD", cwd=c.ruta)
    _git("tag", "v-malo", cwd=c.ruta)

    # ...y deja el .git envenenado (DESPUÉS de sus commits: si no, los ganchos le saltarían a él).
    marcas = tmp_path / "marcas"
    marcas.mkdir()
    chivato = tmp_path / "chivato.sh"
    chivato.write_text(f'#!/bin/sh\nenv > "{marcas}/$(basename "$0")-$$"\n'
                       f'[ "$#" -gt 0 ] && [ "$(basename "$0")" = chivato.sh ] && exec "$@"\nexit 0\n')
    chivato.chmod(0o755)
    ganchos = c.ruta / ".git" / "hooks"
    for nombre in _GANCHOS:
        (ganchos / nombre).write_text(chivato.read_text())
        (ganchos / nombre).chmod(0o755)
    malvado = tmp_path / "malvado.git"
    _git("init", "-q", "--bare", str(malvado), cwd=tmp_path)
    with (c.ruta / ".git" / "config").open("a") as cfg:
        cfg.write(f"""[credential]
\thelper = store --file={c.ruta}/.x
[url "file://{malvado}/"]
\tinsteadOf = {E.GITHUB}
\tpushInsteadOf = {E.GITHUB}
[url "file://{malvado}/x"]
\tinsteadOf = https://github.com/
[http]
\tproxy = http://127.0.0.1:9
[push]
\tfollowTags = true
[core]
\tfsmonitor = {chivato}
\thooksPath = {ganchos}
\tsshCommand = {chivato}
[uploadpack]
\tpackObjectsHook = {chivato}
""")

    api = _GitHubApi()
    cambios, pr = _entregar(c, api)

    assert list(marcas.iterdir()) == [], f"algo ejecutó código del clon: {list(marcas.iterdir())}"
    assert not (c.ruta / ".x").exists()
    for raiz, _, archivos in os.walk(tmp_path):
        for nombre in archivos:
            ruta = Path(raiz) / nombre
            if ruta.is_symlink() or not ruta.is_file():
                continue
            assert TOKEN.encode() not in ruta.read_bytes(), f"el token quedó en {ruta}"
    assert _refs(github) == {"refs/heads/main": base_main, f"refs/heads/{RAMA}": qwen}
    assert _refs(malvado) == {}
    assert [(x.ruta, x.estado) for x in cambios] == [("a", "M")]
    assert pr.url == "https://gh/pr/1" and pr.notas == ()


# --- lease y PR ---------------------------------------------------------------------------

def test_dos_entregas_seguidas_actualizan_el_mismo_pr(tmp_path, github):
    c = _preparar(tmp_path)
    api = _GitHubApi()
    _commit(c.ruta, "a", "2", "uno")
    _entregar(c, api)
    segundo = _commit(c.ruta, "b", "x", "dos")
    cambios, pr = _entregar(c, api)
    assert _refs(github)[f"refs/heads/{RAMA}"] == segundo
    assert sorted(x.ruta for x in cambios) == ["a", "b"]
    assert api.metodos().count(("POST", "/repos/o/r/pulls")) == 1
    assert ("PATCH", "/repos/o/r/pulls/1") in api.metodos() and pr.url == "https://gh/pr/1"
    parche = next(b for m, p, b in api.pedidos if m == "PATCH")
    assert parche["draft"] is False and parche["body"] == "c"
    # Qwen reescribe su propia historia: el lease (seguimiento del espejo) sigue autorizando.
    reescrito = _commit(c.ruta, "b", "y", "dos, reescrito", "--amend")
    _entregar(c, api)
    assert _refs(github)[f"refs/heads/{RAMA}"] == reescrito


def test_push_ajeno_entre_entregas_no_se_pisa(tmp_path, github):
    c = _preparar(tmp_path)
    api = _GitHubApi()
    _commit(c.ruta, "a", "2", "uno")
    _entregar(c, api)
    otro = tmp_path / "otro"
    _git("clone", "-q", "-b", RAMA, str(github), str(otro), cwd=tmp_path)
    ajeno = _commit(otro, "a", "ajeno", "ajeno")
    _git("push", "-q", "origin", RAMA, cwd=otro)
    # Turno 2 entre medio: el fetch del espejo NO debe traer la rama de la misión (si la
    # trajera, el seguimiento pasaría a ser el commit ajeno y el lease lo autorizaría).
    assert _preparar(tmp_path) == c
    _commit(c.ruta, "a", "3", "dos")
    with pytest.raises(E.EntregaRechazada, match="lease"):
        _entregar(c, api)
    assert _refs(github)[f"refs/heads/{RAMA}"] == ajeno


def test_pr_cerrado_no_se_reabre_se_abre_otro_y_se_declara(tmp_path, github):
    c = _preparar(tmp_path)
    api = _GitHubApi()
    _commit(c.ruta, "a", "2", "uno")
    _entregar(c, api)
    api.prs[0]["state"] = "closed"
    _commit(c.ruta, "a", "3", "dos")
    _, pr = _entregar(c, api)
    assert pr.url == "https://gh/pr/2" and pr.notas == ("pr_reabierto_nuevo",)
    assert not any(m == "PATCH" for m, _ in api.metodos())


def test_pr_422_ya_existe_se_vuelve_a_listar(tmp_path):
    api = _GitHubApi(respuesta_post=httpx.Response(422, json={
        "message": "Validation Failed",
        "errors": [{"resource": "PullRequest", "code": "custom",
                    "message": f"A pull request already exists for o:{RAMA}."}]}))
    original = api.__call__

    def con_carrera(req):
        r = original(req)
        if req.method == "POST" and r.status_code == 422:
            api.prs.append({"number": 9, "state": "open", "html_url": "https://gh/pr/9"})
        return r
    api.__call__ = con_carrera

    async def correr():
        async with httpx.AsyncClient(transport=httpx.MockTransport(con_carrera),
                                     base_url="https://api.github.com") as cli:
            return await E.abrir_o_actualizar_pr(cli, repo="o/r", rama=RAMA, base="main", titulo="t", cuerpo="c")
    pr = asyncio.run(correr())
    assert pr.url == "https://gh/pr/9"
    assert ("PATCH", "/repos/o/r/pulls/9") in api.metodos()
    assert ("POST", "/repos/o/r/issues/9/labels") in api.metodos()
