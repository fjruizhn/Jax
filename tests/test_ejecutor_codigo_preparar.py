"""Preparar la misión de código (spec 2026-09-28 v1.2, §3.1): espejo privado de `jaxsvc`,
clon de trabajo de Qwen clonado DESDE el espejo, dependencias fuera del clon.

Turno 1: espejo (origin = URL derivada de owner_repo) → dependencias desde la rama por
omisión en `<misión>/deps` → clon `--no-hardlinks` en la rama `axioma/<misión>` → ACL.
Turno ≥ 2: solo `fetch` del espejo desde GitHub; ni reinstala, ni re-aplica ACL, ni corre
git dentro del clon."""
from __future__ import annotations

import asyncio
import getpass
import json
import os
import shutil
import subprocess
from pathlib import Path

import httpx
import pytest

from jax.ejecutor.codigo import entrega as E
from jax.ejecutor.codigo import preparar as P
from jax.ejecutor.contratos import cuenta_axioma as CA

MID = "3f1c9a2e-7b4d-4e8a-9c1f-2a3b4c5d6e7f"
RAMA = f"axioma/{MID}"
TOKEN = "github_pat_TOKEN_FALSO_0123456789"


def _git(*a, cwd):
    return subprocess.run(["git", *a], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def _remoto(tmp_path, monkeypatch, rama="master", archivos=None):
    gh = tmp_path / "gh"
    r = gh / "o" / "r.git"
    r.parent.mkdir(parents=True)
    _git("init", "-q", "--bare", "-b", rama, str(r), cwd=tmp_path)
    w = tmp_path / "w"
    _git("clone", "-q", str(r), str(w), cwd=tmp_path)
    for ruta, texto in (archivos or {"a": "1"}).items():
        (w / ruta).parent.mkdir(parents=True, exist_ok=True)
        (w / ruta).write_text(texto)
    _git("add", "-A", cwd=w)
    _git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "x", cwd=w)
    _git("push", "-q", "origin", rama, cwd=w)
    monkeypatch.setattr(E, "GITHUB", f"file://{gh}/")
    return r


class _Registro:
    def __init__(self):
        self.llamadas = []

    async def __call__(self, ruta):
        assert ruta.is_dir(), f"acceso pedido sobre {ruta} antes de que exista"
        self.llamadas.append(ruta)


def _accesos():
    return P.Accesos(paso=_Registro(), escritura=_Registro(), lectura=_Registro())


class _Instalador:
    """Doble del ejecutor de instalaciones: registra argv/cwd/env y responde."""

    def __init__(self, fallos=None):
        self.llamadas = []
        self.fallos = fallos or {}
        self.copias = {}

    async def __call__(self, argv, cwd, env):
        # Lo que el sandbox monta de solo lectura se lee AHORA: la copia se borra al terminar.
        copias = {argv[i + 2]: Path(argv[i + 1]).read_text() for i, a in enumerate(argv)
                  if a == "--ro-bind" and argv[i + 1].startswith(self.copias_bajo) and Path(argv[i + 1]).is_file()}
        self.llamadas.append((tuple(argv), Path(cwd), dict(env)))
        self.copias.update(copias)
        for clave, (rc, err) in self.fallos.items():
            if clave in " ".join(argv):
                return rc, err
        return 0, b""

    copias_bajo = "/tmp"

    def internos(self):
        """Los comandos DENTRO del sandbox (lo que va después de `--`)."""
        salida = []
        for argv, _, _ in self.llamadas:
            assert argv[0] == "bwrap" and "--clearenv" in argv, f"instalación fuera del sandbox: {argv}"
            salida.append(argv[argv.index("--") + 1:])
        return salida

    def entornos(self):
        return [{argv[i + 1]: argv[i + 2] for i, a in enumerate(argv[:argv.index("--")]) if a == "--setenv"}
                for argv, _, _ in self.llamadas]


def _preparar(tmp_path, *, accesos=None, instalar=None, rama="master", owner_repo="o/r", autor="A <a@a.io>",
              node_bin=None):
    return asyncio.run(P.preparar(P.Repo(owner_repo, ()), mision_id=MID, raiz=tmp_path / "m",
                                  rama_por_omision=rama, token=TOKEN, autor=autor,
                                  accesos=accesos or _accesos(), instalar=instalar or _Instalador(),
                                  node_bin=node_bin))


def test_espejo_privado_y_clon_desde_el_espejo(tmp_path, monkeypatch):
    _remoto(tmp_path, monkeypatch)
    accesos = _accesos()
    c = _preparar(tmp_path, accesos=accesos, autor="Axioma (Ejecutor) <axioma@axioma-ia.io>")
    base = tmp_path / "m" / MID
    assert c.ruta == base / "repo" and c.espejo == base / "espejo.git"
    assert c.rama == RAMA and c.rama_por_omision == "master"
    assert oct(c.espejo.stat().st_mode & 0o777) == "0o700"
    assert _git("config", "remote.origin.url", cwd=c.espejo) == f"{E.GITHUB}o/r.git"
    assert _git("rev-parse", "--abbrev-ref", "HEAD", cwd=c.ruta) == RAMA
    assert _git("config", "remote.origin.url", cwd=c.ruta) == str(c.espejo)
    cfg = (c.ruta / ".git" / "config").read_text()
    assert "Axioma (Ejecutor)" in cfg and "axioma@axioma-ia.io" in cfg and TOKEN not in cfg
    assert "github" not in cfg
    # --no-hardlinks: ningún objeto del clon comparte inodo con el espejo.
    inodos_espejo = {p.stat().st_ino for p in (c.espejo / "objects").rglob("*") if p.is_file()}
    assert not any(p.stat().st_ino in inodos_espejo for p in (c.ruta / ".git" / "objects").rglob("*")
                   if p.is_file())
    assert accesos.paso.llamadas == [base]
    assert accesos.escritura.llamadas == [c.ruta]
    assert accesos.lectura.llamadas == [base / "deps"]
    assert c.dependencias == ("sin_lockfile",)


def test_turno_dos_no_reinstala_ni_toca_acl_ni_el_clon(tmp_path, monkeypatch):
    _remoto(tmp_path, monkeypatch, archivos={"requirements.txt": "x==1\n", "package.json": "{}",
                                             "package-lock.json": "{}"})
    c1 = _preparar(tmp_path)
    (c1.ruta / "nuevo").write_text("de qwen")
    accesos, instalar = _accesos(), _Instalador()
    vistos = []
    original = asyncio.create_subprocess_exec

    async def espia(*argv, **kw):
        vistos.append(argv)
        return await original(*argv, **kw)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", espia)
    c2 = _preparar(tmp_path, accesos=accesos, instalar=instalar)
    assert c2 == c1 and (c2.ruta / "nuevo").exists()
    assert instalar.llamadas == []
    assert accesos.paso.llamadas == accesos.escritura.llamadas == accesos.lectura.llamadas == []
    assert not any(a[0] == "setfacl" for a in vistos)
    # Solo git, y solo en el espejo: nunca dentro del clon.
    assert vistos and all(a[0] == "git" and str(c1.espejo) in a for a in vistos), vistos
    assert any("fetch" in a for a in vistos)


def test_turno_dos_trae_lo_nuevo_de_la_rama_por_omision_al_espejo(tmp_path, monkeypatch):
    r = _remoto(tmp_path, monkeypatch)
    c = _preparar(tmp_path)
    w = tmp_path / "w"
    (w / "z").write_text("nuevo")
    _git("add", "z", cwd=w)
    _git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "z", cwd=w)
    _git("push", "-q", "origin", "master", cwd=w)
    _preparar(tmp_path)
    assert _git("rev-parse", "refs/remotes/origin/master", cwd=c.espejo) == _git("rev-parse", "master", cwd=r)
    # La rama de la misión en GitHub NO se trae: el lease depende de eso.
    assert _git("for-each-ref", "refs/remotes/origin/axioma", cwd=c.espejo) == ""


def test_dependencias_con_banderas_y_sin_entorno_de_jax(tmp_path, monkeypatch):
    monkeypatch.setenv("JAX_GITHUB_TOKEN", "no-debe-pasar")
    monkeypatch.setenv("JAX_DB_PASSWORD", "no-debe-pasar")
    monkeypatch.setenv("NPM_TOKEN", "no-debe-pasar")
    _remoto(tmp_path, monkeypatch, archivos={
        "requirements.txt": "x==1\n", "requirements-dev.txt": "-r requirements.txt\n",
        "backend/requirements.txt": "y==2\n",
        "package.json": "{}", "package-lock.json": "{}",
        "frontend/package.json": "{}", "frontend/package-lock.json": "{}",
        "tools/package.json": "{}",
        "composer.json": "{}", "composer.lock": "{}",
    })
    instalar = _Instalador()
    c = _preparar(tmp_path, instalar=instalar, node_bin=Path("/opt/ejecutor/node-v24/bin"))
    deps = tmp_path / "m" / MID / "deps"
    internos = instalar.internos()  # todas pasan por bwrap con --clearenv
    npm = [a for a in internos if a[0] == "npm"]
    assert npm == [("npm", "ci", "--ignore-scripts", "--no-audit", "--no-fund")] * 2
    assert [a for a in internos if a[0] == "composer"] == [
        ("composer", "install", "--no-interaction", "--no-scripts", "--no-plugins")]
    pips = [a for a in internos if a[0].endswith("/pip")]
    # UNA invocación por venv, con todos sus requirements juntos.
    assert len(pips) == 2, pips
    raiz = next(a for a in pips if "/_raiz/" in a[0])
    assert raiz[1:3] == ("install", "--only-binary=:all:")
    assert [Path(x).name for x in raiz[3:][1::2]] == ["requirements-dev.txt", "requirements.txt"]
    assert list(raiz[3:][0::2]) == ["-r", "-r"]
    for argv, cwd, env in instalar.llamadas:
        assert not any(k.startswith("JAX_") for k in env) and "NPM_TOKEN" not in env
        assert cwd.is_relative_to(deps), f"{argv} corrió fuera de deps: {cwd}"
    for env in instalar.entornos():
        assert set(env) == {"PATH", "HOME", "LANG"} and env["HOME"] == "/tmp/h"
    npm_argv = next(a for a, _, _ in instalar.llamadas if "npm" in a[a.index("--"):])
    assert "/opt/ejecutor/node-v24" in npm_argv
    assert "sin_lockfile:tools/package.json" in c.dependencias
    for enlace in (".venv", "backend/.venv", "node_modules", "frontend/node_modules", "vendor"):
        ruta = c.ruta / enlace
        assert ruta.is_symlink() and Path(os.readlink(ruta)).is_relative_to(deps), enlace
    excluidos = (c.ruta / ".git" / "info" / "exclude").read_text().splitlines()
    for enlace in ("/.venv", "/backend/.venv", "/node_modules", "/frontend/node_modules", "/vendor"):
        assert enlace in excluidos
    # Los archivos de bloqueo salen de la rama por omisión del ESPEJO (no del clon) y entran
    # al sandbox como COPIA de solo lectura, montada sobre la ruta que npm espera.
    assert instalar.copias[str(deps / "node" / "frontend" / "package-lock.json")] == "{}"
    assert instalar.copias[str(deps / "php" / "_raiz" / "composer.lock")] == "{}"


@pytest.mark.parametrize("linea", ["--no-binary x", "--no-binary=:all:", "--index-url https://evil/simple",
                                   "-i https://evil/simple", "--extra-index-url https://evil/simple",
                                   "-f https://evil/links", "--find-links=/tmp/x", "-e .", "--editable git+https://x",
                                   "--trusted-host evil", "-c constraints.txt", "-r ../fuera.txt",
                                   "paquete @ https://evil/paquete-1.0.tar.gz", "https://evil/p-1.0.tar.gz",
                                   "./local", "git+https://evil/r.git#egg=r"])
def test_requirements_con_opcion_no_se_instala_y_se_declara(tmp_path, monkeypatch, linea):
    _remoto(tmp_path, monkeypatch, archivos={"requirements.txt": f"x==1\n{linea}\n"})
    instalar = _Instalador()
    c = _preparar(tmp_path, instalar=instalar)
    assert instalar.llamadas == [], "no debía ejecutarse nada: ni venv ni pip"
    assert "pip_opcion_rechazada:requirements.txt" in c.dependencias


def test_la_opcion_rechazada_arrastra_a_quien_la_incluye_y_no_al_resto(tmp_path, monkeypatch):
    _remoto(tmp_path, monkeypatch, archivos={"requirements.txt": "x==1\n",
                                             "requirements-mala.txt": "--no-binary x\n",
                                             "requirements-dev.txt": "-r requirements-mala.txt\ny==1  # nota\n"})
    instalar = _Instalador()
    c = _preparar(tmp_path, instalar=instalar)
    pips = [a for a in instalar.internos() if a[0].endswith("/pip")]
    assert len(pips) == 1 and [Path(x).name for x in pips[0][3:][1::2]] == ["requirements.txt"]
    assert "pip_opcion_rechazada:requirements-mala.txt" in c.dependencias
    assert "pip_opcion_rechazada:requirements-dev.txt" in c.dependencias
    assert "requirements.txt" in c.dependencias


def test_pip_sin_wheel_se_declara_y_sigue(tmp_path, monkeypatch):
    _remoto(tmp_path, monkeypatch, archivos={"requirements.txt": "solo-sdist==1\n"})
    instalar = _Instalador(fallos={"--only-binary": (1, b"ERROR: No matching distribution found for solo-sdist==1")})
    c = _preparar(tmp_path, instalar=instalar)
    assert "pip_sin_wheel:requirements.txt" in c.dependencias


def test_fallo_de_instalacion_no_filtra_su_salida(tmp_path, monkeypatch):
    _remoto(tmp_path, monkeypatch, archivos={"package.json": "{}", "package-lock.json": "{}"})
    instalar = _Instalador(fallos={"npm": (1, b"SALIDA-CONTROLADA-POR-EL-REPO")})
    with pytest.raises(RuntimeError, match="preparar_fallo") as e:
        _preparar(tmp_path, instalar=instalar)
    assert "SALIDA-CONTROLADA-POR-EL-REPO" not in str(e.value)


@pytest.mark.parametrize("campo,valor,mensaje", [
    ("owner_repo", "o/r;id", "owner_repo_invalido"),
    ("owner_repo", "../x", "owner_repo_invalido"),
    ("autor", "Sin correo", "autor_invalido"),
    ("autor", "A <a@a> extra", "autor_invalido"),
    ("autor", "A\n <a@a>", "autor_invalido"),
    ("rama", "-x", "rama_por_omision_invalida"),
    ("rama", "a..b", "rama_por_omision_invalida"),
])
def test_validacion_antes_de_cualquier_comando(tmp_path, monkeypatch, campo, valor, mensaje):
    async def prohibido(*a, **k):
        raise AssertionError("no debía correr ningún comando")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", prohibido)
    accesos, instalar = _accesos(), _Instalador()
    kw = {campo: valor}
    with pytest.raises(ValueError, match=mensaje):
        _preparar(tmp_path, accesos=accesos, instalar=instalar, **kw)
    assert not (tmp_path / "m").exists()
    assert instalar.llamadas == [] and accesos.paso.llamadas == []


def test_parsear_autor():
    assert P.parsear_autor("Axioma (Ejecutor) <axioma@axioma-ia.io>") == ("Axioma (Ejecutor)", "axioma@axioma-ia.io")


def test_error_de_git_no_filtra_el_token(tmp_path, monkeypatch):
    monkeypatch.setattr(E, "GITHUB", f"file://{tmp_path}/no-existe/")
    with pytest.raises(RuntimeError) as e:
        _preparar(tmp_path)
    assert TOKEN not in str(e.value)


def test_preparacion_a_medias_se_rehace(tmp_path, monkeypatch):
    _remoto(tmp_path, monkeypatch)
    base = tmp_path / "m" / MID
    (base / "repo").mkdir(parents=True)
    (base / "repo" / "resto").write_text("de un turno 1 que se cayó")
    c = _preparar(tmp_path)
    assert not (c.ruta / "resto").exists() and (c.ruta / "a").exists()


@pytest.mark.parametrize("repo", ["o/r/../../x", "o", "o/r?x=1"])
def test_rama_por_omision_valida_el_repo_antes_de_pedir(repo):
    def prohibido(req):
        raise AssertionError("no debía llamar a la API")
    cli = httpx.AsyncClient(transport=httpx.MockTransport(prohibido), base_url="https://api.github.com")
    with pytest.raises(ValueError, match="owner_repo_invalido"):
        asyncio.run(P.rama_por_omision(cli, repo))


def test_rama_por_omision_de_la_api_mal_formada_se_rechaza():
    cli = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, json={"default_branch": "--upload-pack=x"})), base_url="https://api.github.com")
    with pytest.raises(ValueError, match="rama_por_omision_invalida"):
        asyncio.run(P.rama_por_omision(cli, "o/r"))


def test_rama_por_omision_de_la_api():
    cli = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, json={"default_branch": "master"})), base_url="https://api.github.com")
    assert asyncio.run(P.rama_por_omision(cli, "o/r")) == "master"


requiere_acl = pytest.mark.skipif(shutil.which("setfacl") is None or shutil.which("getfacl") is None,
                                  reason="setfacl/getfacl no están instalados en este runner")


def _acl(ruta):
    salida = subprocess.run(["getfacl", "-p", str(ruta)], capture_output=True, text=True, check=True).stdout
    return {l.split("\t", 1)[0].strip() for l in salida.splitlines() if l.strip() and not l.startswith("#")}


@requiere_acl
def test_acl_real_base_solo_paso_espejo_sin_acl(tmp_path, monkeypatch):
    """ACL reales (la cuenta es el propio usuario del test: en el runner no existe `axioma`).
    El padre trae una ACL por omisión ajena, como `/var/lib/jax-ejecutor-misiones`."""
    _remoto(tmp_path, monkeypatch)
    yo = getpass.getuser()
    (tmp_path / "m").mkdir()
    subprocess.run(["setfacl", "-d", "-m", "u:99999:rwx", str(tmp_path / "m")], check=True)
    c = CA.Cuenta(yo, 22, tmp_path / "k", tmp_path / "n", tmp_path / "l", tmp_path / "p", tmp_path / "h")
    clon = _preparar(tmp_path, accesos=P.accesos_de_la_cuenta(c))
    base = tmp_path / "m" / MID
    acl_base = _acl(base)
    assert f"user:{yo}:--x" in acl_base and not any(l.startswith("default:") for l in acl_base)
    assert not any("99999" in l for l in acl_base)
    acl_espejo = _acl(clon.espejo)
    assert not any(l.startswith(("user:" + yo, "user:99999", "default:")) for l in acl_espejo), acl_espejo
    assert oct(clon.espejo.stat().st_mode & 0o777) == "0o700"
    assert f"user:{yo}:rwx" in _acl(clon.ruta)
    acl_deps = _acl(base / "deps")
    assert f"user:{yo}:r-x" in acl_deps and f"default:user:{yo}:r-x" in acl_deps
    assert json.loads((base / "preparado.json").read_text())["dependencias"] == ["sin_lockfile"]
