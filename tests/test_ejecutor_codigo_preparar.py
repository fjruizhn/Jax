"""Preparar el clon de la misión de código (spec 2026-09-28 §3.1): clon, rama de la
misión, dependencias de lockfile, identidad de autor -- sin la cuenta axioma, que entra
después por `dar_acceso` (ACL, inyectable, ver `cuenta_axioma.dar_acceso_recursivo`)."""
from __future__ import annotations

import asyncio
import subprocess

import httpx

from jax.ejecutor.codigo import preparar as P

MID = "3f1c9a2e-7b4d-4e8a-9c1f-2a3b4c5d6e7f"


def _git(*a, cwd):
    subprocess.run(["git", *a], cwd=cwd, check=True, capture_output=True)


def _remoto(tmp_path, rama="master"):
    r = tmp_path / "r.git"
    _git("init", "--bare", "-b", rama, str(r), cwd=tmp_path)
    w = tmp_path / "w"
    _git("clone", str(r), str(w), cwd=tmp_path)
    for k, v in (("user.name", "t"), ("user.email", "t@t")):
        _git("config", k, v, cwd=w)
    (w / "a").write_text("1")
    _git("add", "a", cwd=w)
    _git("commit", "-m", "x", cwd=w)
    _git("push", "origin", rama, cwd=w)
    return r


async def _acceso(ruta):
    pass


def test_clona_rama_de_mision_desde_master_y_autor(tmp_path):
    r = _remoto(tmp_path)
    c = asyncio.run(P.preparar(P.Repo("o/r", str(r), ()), mision_id=MID, raiz=tmp_path / "m",
                               rama_por_omision="master", token="t", autor="Axioma (Ejecutor) <axioma@axioma-ia.io>",
                               dar_acceso=_acceso))
    head = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=c.ruta, capture_output=True,
                          text=True).stdout.strip()
    assert head == f"axioma/{MID}" and c.rama_por_omision == "master"
    cfg = (c.ruta / ".git" / "config").read_text()
    assert "Axioma (Ejecutor)" in cfg and "t@" not in cfg and "token" not in cfg.lower()
    assert c.dependencias == ("sin_lockfile",)


def test_turno_dos_reusa_el_clon(tmp_path):
    r = _remoto(tmp_path)
    args = dict(mision_id=MID, raiz=tmp_path / "m", rama_por_omision="master", token="t", autor="A <a@a>",
               dar_acceso=_acceso)
    c1 = asyncio.run(P.preparar(P.Repo("o/r", str(r), ()), **args))
    (c1.ruta / "nuevo").write_text("x")
    _git("add", "nuevo", cwd=c1.ruta)
    _git("commit", "-m", "qwen", cwd=c1.ruta)
    c2 = asyncio.run(P.preparar(P.Repo("o/r", str(r), ()), **args))
    assert (c2.ruta / "nuevo").exists()


def test_rama_por_omision_de_la_api():
    cli = httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(200,
                                                                                     json={"default_branch": "master"})),
                            base_url="https://api.github.com")
    assert asyncio.run(P.rama_por_omision(cli, "o/r")) == "master"


# --- ruling del controlador 2026-09-28: `dar_acceso(base)` se llama con `base` ya
# creado (`base.mkdir(parents=True, exist_ok=True)` ANTES) -- en producción `dar_acceso`
# es `cuenta_axioma.dar_acceso_recursivo`, que exige que la ruta YA EXISTA (os.lstat) y
# nunca hace mkdir. Un stub que no hace nada (como `_acceso` de arriba) no lo detectaría
# si `preparar()` llamara `dar_acceso` antes de crear `base`; este test sí lo detecta,
# fallando si `base` no existe en el momento del primer llamado.

def test_dar_acceso_se_llama_con_la_base_ya_creada(tmp_path):
    vistos = []

    async def acceso_exigente(ruta):
        vistos.append((ruta, ruta.is_dir()))

    r = _remoto(tmp_path)
    asyncio.run(P.preparar(P.Repo("o/r", str(r), ()), mision_id=MID, raiz=tmp_path / "m",
                           rama_por_omision="master", token="t", autor="A <a@a>",
                           dar_acceso=acceso_exigente))
    assert vistos, "dar_acceso nunca se llamó"
    for ruta, existia in vistos:
        assert existia, f"dar_acceso se llamó con {ruta} sin existir todavía"
