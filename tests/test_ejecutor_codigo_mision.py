"""La entrega de una misión de código, orquestada (spec 2026-09-28 v1.3 §3.3, DC8).

Dos grupos:
- El ORDEN y los estados, con dobles inyectados (sin git).
- Los ataques de un Qwen que quiere sacar algo por la entrega, con git REAL: el "GitHub" es un
  bare local en `file://`, el espejo y el clon los arma `preparar` y `upload-pack` corre local.
  En cada ataque se comprueba que NO se empujó nada."""
from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path

import httpx
import pytest

from jax.ejecutor.codigo import entrega as E
from jax.ejecutor.codigo import mision_codigo as MC
from jax.ejecutor.codigo import preparar as P
from jax.ejecutor.codigo.diff import Cambio

MID = "3f1c9a2e-7b4d-4e8a-9c1f-2a3b4c5d6e7f"
RAMA = f"axioma/{MID}"
TOKEN = "github_pat_TOKEN_FALSO_QUE_NUNCA_DEBE_QUEDAR_EN_DISCO_0123"
AUTOR = "Axioma (Ejecutor) <axioma@axioma-ia.io>"
AXIOMA = ("Axioma (Ejecutor)", "axioma@axioma-ia.io")
SECRETO = "github_pat_" + "S" * 40
PIE = f"Hecho-por: Axioma (Ejecutor, misión {MID}, cerebro qwen)"


# --- el orden, con dobles -------------------------------------------------------------------

class Dobles:
    def __init__(self, cambios=(), commits=None, tamanos=None, falla=None):
        self.cambios = cambios
        self.historial = commits if commits is not None else (
            (E.Commit("a" * 40, AXIOMA, AXIOMA, cambios),) if cambios else ())
        self.medidos = tamanos or {}
        self.falla = falla or {}
        self.llamadas: list[str] = []
        self.pr_pedido: dict | None = None

    def _paso(self, nombre):
        self.llamadas.append(nombre)
        if nombre in self.falla:
            raise self.falla[nombre]

    async def traer(self, espejo, clon, *, mision_id, upload_pack):
        assert upload_pack == "ssh-como-la-cuenta" and mision_id == MID
        self._paso("traer")

    async def commits(self, espejo, *, mision_id, rama_por_omision):
        self._paso("commits")
        return self.historial

    async def diff(self, espejo, *, mision_id, rama_por_omision):
        self._paso("diff")
        return self.cambios

    async def tamanos(self, espejo, *, mision_id, rama_por_omision):
        self._paso("tamanos")
        return self.medidos

    async def empujar(self, espejo, *, mision_id, rama_por_omision, token):
        assert token == TOKEN
        self._paso("empujar")

    async def abrir_pr(self, cliente, **kw):
        self._paso("pr")
        self.pr_pedido = kw
        return E.PrEntregado("https://gh/pr/1", ("pr_reabierto_nuevo",))


def _clon(tmp: Path = Path("/x")) -> P.Clon:
    return P.Clon(tmp / "repo", RAMA, "main", (), tmp / "espejo.git")


def _entregar(d: Dobles, **kw) -> dict:
    base = dict(mision_id=MID, repo="o/r", revision_legible=True, informe="informe", token=TOKEN, cliente=None,
                tope_bytes=10, modelo="qwen", autor=AUTOR, upload_pack="ssh-como-la-cuenta",
                traer=d.traer, commits=d.commits, diff=d.diff, tamanos=d.tamanos, empujar=d.empujar,
                abrir_pr=d.abrir_pr)
    base.update(kw)
    return asyncio.run(MC.entregar(_clon(), **base))


A_PY = Cambio("a.py", "M", None, ("x",), ())


def test_sin_commits_nuevos_es_sin_cambios_y_no_empuja():
    d = Dobles()
    r = _entregar(d)
    assert r == {"estado_entrega": "sin_cambios", "pr_url": None, "violaciones": [], "notas": []}
    assert d.llamadas == ["traer", "commits"]


def test_sin_informe_c5_no_empuja_ni_abre_pr():
    d = Dobles((A_PY,))
    r = _entregar(d, revision_legible=False)
    assert r["estado_entrega"] == "sin_informe_c5" and "empujar" not in d.llamadas and "pr" not in d.llamadas


def test_violacion_de_c1_rechaza_por_contrato_sin_empujar():
    d = Dobles((Cambio(".github/workflows/x.yml", "M", None, ("x",), ()),))
    r = _entregar(d)
    assert r["estado_entrega"] == "rechazada_por_contrato" and r["violaciones"][0]["regla"] == "flujos_ci"
    assert "empujar" not in d.llamadas


def test_tamano_medido_sobre_el_tope_rechaza():
    d = Dobles((A_PY,), tamanos={"a.py": 11})
    r = _entregar(d)
    assert r["estado_entrega"] == "rechazada_por_contrato"
    assert r["violaciones"] == [{"regla": "tamano", "ruta": "a.py", "detalle": "11 > 10 bytes"}]


@pytest.mark.parametrize("autor, committer", [
    (("Fernando Ruiz", "fruiztorres@gmail.com"), AXIOMA),
    (AXIOMA, ("Fernando Ruiz", "fruiztorres@gmail.com")),
    (("Axioma (Ejecutor)", "otro@axioma-ia.io"), AXIOMA),
    (None, None),  # cabecera ilegible
])
def test_identidad_distinta_en_cualquier_commit_rechaza(autor, committer):
    historial = (E.Commit("a" * 40, AXIOMA, AXIOMA, (A_PY,)), E.Commit("b" * 40, autor, committer, ()))
    d = Dobles((A_PY,), commits=historial)
    r = _entregar(d)
    assert r["estado_entrega"] == "rechazada_por_contrato" and "empujar" not in d.llamadas
    assert [v["regla"] for v in r["violaciones"]] == ["identidad"]
    assert "fruiztorres" not in json.dumps(r)


def test_secreto_de_un_commit_intermedio_rechaza_aunque_el_diff_neto_este_limpio():
    historial = (E.Commit("a" * 40, AXIOMA, AXIOMA, (Cambio("c.py", "A", None, (f"T='{SECRETO}'",), ()),)),
                 E.Commit("b" * 40, AXIOMA, AXIOMA, (Cambio("c.py", "M", None, ("T=None",), (f"T='{SECRETO}'",)),)))
    d = Dobles((Cambio("c.py", "A", None, ("T=None",), ()),), commits=historial)
    r = _entregar(d)
    assert r["estado_entrega"] == "rechazada_por_contrato"
    assert r["violaciones"] == [{"regla": "secretos", "ruta": "c.py",
                                 "detalle": "patrón de credencial en una línea agregada"}]
    assert SECRETO not in json.dumps(r)


def test_ni_rutas_ni_detalles_llevan_secretos_ni_el_token():
    ruta = f"tests/test_{SECRETO}.py"
    d = Dobles((Cambio(ruta, "M", None, (f"@pytest.mark.skip  # {SECRETO} {TOKEN}",), ()),))
    r = _entregar(d)
    assert r["estado_entrega"] == "rechazada_por_contrato"
    texto = json.dumps(r)
    assert SECRETO not in texto and TOKEN not in texto


def test_camino_feliz_empuja_abre_el_pr_con_el_pie_y_devuelve_las_notas():
    d = Dobles((A_PY,))
    r = _entregar(d)
    assert r == {"estado_entrega": "abierto", "pr_url": "https://gh/pr/1", "violaciones": [],
                 "notas": ["pr_reabierto_nuevo"]}
    assert d.llamadas == ["traer", "commits", "diff", "tamanos", "empujar", "pr"]
    assert d.pr_pedido["repo"] == "o/r" and d.pr_pedido["rama"] == RAMA and d.pr_pedido["base"] == "main"
    assert d.pr_pedido["cuerpo"].startswith("informe") and d.pr_pedido["cuerpo"].endswith(PIE)


@pytest.mark.parametrize("paso", ["traer", "commits", "diff", "tamanos", "empujar", "pr"])
def test_un_paso_que_falla_es_fallo_entrega_con_el_motivo_saneado(paso):
    d = Dobles((A_PY,), falla={paso: E.EntregaRechazada(f"git_fallo: https://x-access-token:{TOKEN}@github.com")})
    r = _entregar(d)
    assert r["estado_entrega"] == "fallo_entrega" and r["pr_url"] is None
    assert r["violaciones"][0]["regla"] == "entrega" and TOKEN not in json.dumps(r)
    assert d.llamadas[-1] == paso


def test_un_error_de_la_api_de_github_es_fallo_entrega():
    peticion = httpx.Request("POST", "https://api.github.com/repos/o/r/pulls")
    error = httpx.HTTPStatusError("x", request=peticion, response=httpx.Response(403, request=peticion))
    d = Dobles((A_PY,), falla={"pr": error})
    r = _entregar(d)
    assert r["estado_entrega"] == "fallo_entrega" and r["violaciones"][0]["detalle"] == "github_api: 403"


# --- los ataques, con git real ---------------------------------------------------------------

def _git(*a, cwd, **kw):
    return subprocess.run(["git", *a], cwd=cwd, check=True, capture_output=True, text=True, **kw).stdout.strip()


def _commit(repo: Path, archivo: str, datos: bytes, mensaje: str, *extra: str) -> None:
    (repo / archivo).write_bytes(datos)
    _git("add", archivo, cwd=repo)
    _git(*extra, "commit", "-q", "-m", mensaje, cwd=repo)


def _refs(bare: Path) -> dict[str, str]:
    salida = _git("for-each-ref", "--format=%(refname) %(objectname)", cwd=bare)
    return dict(l.split(" ", 1) for l in salida.splitlines() if l)


@pytest.fixture
def github(tmp_path, monkeypatch):
    gh = tmp_path / "gh"
    remoto = gh / "o" / "r.git"
    remoto.parent.mkdir(parents=True)
    _git("init", "-q", "--bare", "-b", "main", str(remoto), cwd=tmp_path)
    w = tmp_path / "w"
    _git("clone", "-q", str(remoto), str(w), cwd=tmp_path)
    _commit(w, "a", b"1", "base", "-c", "user.name=q", "-c", "user.email=q@q")
    _git("push", "-q", "origin", "main", cwd=w)
    monkeypatch.setattr(E, "GITHUB", f"file://{gh}/")
    return remoto


class _Registro:
    async def __call__(self, ruta):
        pass


async def _sin_instalar(argv, cwd, env):
    raise AssertionError(f"no debía instalar nada: {argv}")


def _preparar(tmp_path) -> P.Clon:
    return asyncio.run(P.preparar(P.Repo("o/r", ()), mision_id=MID, raiz=tmp_path / "m", rama_por_omision="main",
                                  token=TOKEN, autor=AUTOR,
                                  accesos=P.Accesos(paso=_Registro(), escritura=_Registro(), lectura=_Registro()),
                                  instalar=_sin_instalar))


def _api(pedidos: list):
    def manejar(req: httpx.Request) -> httpx.Response:
        pedidos.append((req.method, req.url.path, json.loads(req.content) if req.content else {}))
        if req.method == "GET":
            return httpx.Response(200, json=[])
        return httpx.Response(201 if req.url.path.endswith("/pulls") else 200,
                              json={"number": 1, "html_url": "https://gh/pr/1"})
    return httpx.AsyncClient(transport=httpx.MockTransport(manejar), base_url="https://api.github.com")


def _entregar_real(c: P.Clon, pedidos: list, *, tope_bytes: int = 1_000_000) -> dict:
    async def correr():
        async with _api(pedidos) as cliente:
            return await MC.entregar(c, mision_id=MID, repo="o/r", revision_legible=True, informe="informe C5",
                                     token=TOKEN, cliente=cliente, tope_bytes=tope_bytes, modelo="qwen",
                                     autor=AUTOR, upload_pack="git-upload-pack")
    return asyncio.run(correr())


def _nada_empujado(github: Path, pedidos: list) -> bool:
    return f"refs/heads/{RAMA}" not in _refs(github) and pedidos == []


def test_real_camino_feliz_empuja_la_rama_y_abre_el_pr(tmp_path, github):
    c = _preparar(tmp_path)
    _commit(c.ruta, "a", b"2", "cambio")
    pedidos: list = []
    r = _entregar_real(c, pedidos)
    assert r == {"estado_entrega": "abierto", "pr_url": "https://gh/pr/1", "violaciones": [], "notas": []}
    assert _refs(github)[f"refs/heads/{RAMA}"] == _git("rev-parse", "HEAD", cwd=c.ruta)
    (cuerpo,) = [b["body"] for m, p, b in pedidos if m == "POST" and p == "/repos/o/r/pulls"]
    assert cuerpo.startswith("informe C5") and cuerpo.endswith(PIE)


def test_real_blob_grande_de_un_commit_intermedio_luego_borrado_se_rechaza(tmp_path, github):
    """Ruling 4b: el diff neto no lo muestra, pero el empuje lo llevaría a GitHub en el historial."""
    c = _preparar(tmp_path)
    _commit(c.ruta, "grande.bin", b"0" * 5000, "grande")
    _git("rm", "-q", "grande.bin", cwd=c.ruta)
    _git("commit", "-q", "-m", "lo borro", cwd=c.ruta)
    _commit(c.ruta, "a", b"2", "otro")
    pedidos: list = []
    r = _entregar_real(c, pedidos, tope_bytes=1000)
    assert r["estado_entrega"] == "rechazada_por_contrato", r
    assert r["violaciones"] == [{"regla": "tamano", "ruta": "grande.bin", "detalle": "5000 > 1000 bytes"}]
    assert _nada_empujado(github, pedidos)


def test_real_rutas_no_ascii_y_con_espacios_se_entregan(tmp_path, github):
    """Antes del ruling, `año.py` salía citado por git y el parser reventaba con IndexError."""
    c = _preparar(tmp_path)
    (c.ruta / "dir con espacio").mkdir()
    for ruta in ("año.py", "dir con espacio/x.py"):
        (c.ruta / ruta).write_text("x = 1\n")
    _git("add", "-A", cwd=c.ruta)
    _git("commit", "-q", "-m", "raras", cwd=c.ruta)
    pedidos: list = []
    assert _entregar_real(c, pedidos)["estado_entrega"] == "abierto"
    assert f"refs/heads/{RAMA}" in _refs(github)


def test_real_sin_commits_es_sin_cambios(tmp_path, github):
    c = _preparar(tmp_path)
    (c.ruta / "sin-commitear.txt").write_text("x")
    pedidos: list = []
    assert _entregar_real(c, pedidos)["estado_entrega"] == "sin_cambios" and _nada_empujado(github, pedidos)


def test_real_secreto_agregado_y_quitado_dentro_de_la_rama_se_rechaza(tmp_path, github):
    c = _preparar(tmp_path)
    _commit(c.ruta, "cfg.py", f"T = '{SECRETO}'\n".encode(), "pongo")
    _commit(c.ruta, "cfg.py", b"T = None\n", "saco")
    pedidos: list = []
    r = _entregar_real(c, pedidos)
    assert r["estado_entrega"] == "rechazada_por_contrato", r
    assert [(v["regla"], v["ruta"]) for v in r["violaciones"]] == [("secretos", "cfg.py")]
    assert _nada_empujado(github, pedidos) and SECRETO not in json.dumps(r)


def test_real_archivo_con_nul_y_un_secreto_se_rechaza(tmp_path, github):
    c = _preparar(tmp_path)
    _commit(c.ruta, "datos.bin", b"\x00cabecera\nclave = '" + SECRETO.encode() + b"'\n", "binario")
    pedidos: list = []
    r = _entregar_real(c, pedidos)
    assert r["estado_entrega"] == "rechazada_por_contrato", r
    assert ("secretos", "datos.bin") in [(v["regla"], v["ruta"]) for v in r["violaciones"]]
    assert _nada_empujado(github, pedidos)


@pytest.mark.parametrize("extra", [
    ("-c", "user.name=Fernando Ruiz", "-c", "user.email=fruiztorres@gmail.com"),
    ("-c", "user.name=Fernando Ruiz"),
])
def test_real_commit_con_otra_identidad_se_rechaza(tmp_path, github, extra):
    c = _preparar(tmp_path)
    _commit(c.ruta, "a", b"2", "bien")
    _commit(c.ruta, "b", b"3", "suplanto", *extra)
    pedidos: list = []
    r = _entregar_real(c, pedidos)
    assert r["estado_entrega"] == "rechazada_por_contrato", r
    assert [v["regla"] for v in r["violaciones"]] == ["identidad"]
    assert _nada_empujado(github, pedidos)


def test_real_solo_el_author_cambiado_tambien_se_rechaza(tmp_path, github):
    """`--author` deja el committer de Axioma: se exigen los DOS."""
    c = _preparar(tmp_path)
    _commit(c.ruta, "a", b"2", "suplanto", )
    _git("commit", "-q", "--amend", "--no-edit", "--author=Fernando Ruiz <fruiztorres@gmail.com>", cwd=c.ruta)
    pedidos: list = []
    r = _entregar_real(c, pedidos)
    assert [v["regla"] for v in r["violaciones"]] == ["identidad"] and _nada_empujado(github, pedidos)


def test_real_archivo_grande_commiteado_y_chico_en_el_disco_se_rechaza(tmp_path, github):
    c = _preparar(tmp_path)
    _commit(c.ruta, "grande.bin", b"0" * 5000, "grande")
    (c.ruta / "grande.bin").write_bytes(b"0")  # sin commitear: lo que se empujaría es el grande
    pedidos: list = []
    r = _entregar_real(c, pedidos, tope_bytes=1000)
    assert r["estado_entrega"] == "rechazada_por_contrato", r
    assert r["violaciones"] == [{"regla": "tamano", "ruta": "grande.bin", "detalle": "5000 > 1000 bytes"}]
    assert _nada_empujado(github, pedidos)
