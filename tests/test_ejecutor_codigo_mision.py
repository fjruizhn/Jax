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

PUNTA = "c" * 40


class Dobles:
    def __init__(self, cambios=(), commits=None, tamanos=None, total=0, falla=None, pausas=(), pr_previo=None,
                 remoto=None):
        self.cambios = cambios
        self.historial = commits if commits is not None else (
            (E.Commit("a" * 40, AXIOMA, AXIOMA, cambios),) if cambios else ())
        self.medidos = E.Tamanos(tamanos or {}, total)
        self.falla = falla or {}
        self.llamadas: list[str] = []
        self.pr_pedido: dict | None = None
        self.pausas = list(pausas)  # lo que devuelve cada lectura de la pausa, en orden; después, False
        self.lecturas_de_pausa = 0
        self.pr_previo = pr_previo
        self.remoto = remoto  # lo que responde ls-remote tras un empuje fallido (MAJOR-A)

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

    async def punta(self, espejo, *, mision_id):
        self._paso("punta")
        return PUNTA

    async def empujar(self, espejo, *, mision_id, rama_por_omision, token):
        assert token == TOKEN
        self._paso("empujar")

    async def abrir_pr(self, cliente, **kw):
        self._paso("pr")
        self.pr_pedido = kw
        return E.PrEntregado("https://gh/pr/1", ("pr_reabierto_nuevo",))

    async def pausa(self):
        self.lecturas_de_pausa += 1
        valor = self.pausas.pop(0) if self.pausas else False
        if isinstance(valor, Exception):
            raise valor
        return valor

    async def pr_abierto(self, cliente, *, repo, rama):
        self._paso("pr_abierto")
        return self.pr_previo

    async def consultar_remoto(self, espejo, *, mision_id, token):
        assert token == TOKEN
        self._paso("ls_remote")
        return self.remoto


def _clon(tmp: Path = Path("/x")) -> P.Clon:
    return P.Clon(tmp / "repo", RAMA, "main", (), tmp / "espejo.git")


def _entregar(d: Dobles, **kw) -> dict:
    base = dict(mision_id=MID, repo="o/r", revision_legible=True, informe="informe", token=TOKEN, cliente=None,
                tope_bytes=10, tope_total_bytes=10**9, modelo="qwen", autor=AUTOR, upload_pack="ssh-como-la-cuenta",
                pausa_puesta=d.pausa, traer=d.traer, commits=d.commits, diff=d.diff, tamanos=d.tamanos,
                punta=d.punta, empujar=d.empujar, abrir_pr=d.abrir_pr, pr_abierto=d.pr_abierto,
                consultar_remoto=d.consultar_remoto)
    base.update(kw)
    return asyncio.run(MC.entregar(_clon(), **base))


A_PY = Cambio("a.py", "M", None, ("x",), ())
SIN_EMPUJE = {"rama_empujada": False, "sha": None}


def test_sin_commits_nuevos_es_sin_cambios_y_no_empuja():
    d = Dobles()
    r = _entregar(d)
    assert r == {"estado_entrega": "sin_cambios", "pr_url": None, "violaciones": [], "notas": [], **SIN_EMPUJE}
    assert d.llamadas == ["traer", "commits", "pr_abierto"]


def test_sin_cambios_con_un_pr_previo_abierto_lo_declara():
    """MINOR-5: un turno ≥ 2 que deja la rama sin commits propios mientras hay un PR abierto."""
    assert _entregar(Dobles(pr_previo="https://gh/pr/9"))["notas"] == ["pr_previo_sin_cambios_nuevos"]
    falla = Dobles(falla={"pr_abierto": httpx.ConnectError("x")})
    assert _entregar(falla)["notas"] == ["pr_previo_no_verificado"]


def test_mision_codigo_por_si_sola_sin_informe_c5_no_empuja_ni_abre_pr():
    """MINOR-6: prueba SOLO la defensa de `mision_codigo`. Desde `correr_turno` este estado no se
    alcanza (un auditor ilegible frena antes y la entrega queda `sin_entregar`)."""
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


def test_tamano_total_del_rango_sobre_el_tope_rechaza():
    """MINOR-3: cada archivo bajo el tope, el conjunto no."""
    d = Dobles((A_PY,), tamanos={"a.py": 5, "b.py": 5}, total=1001)
    r = _entregar(d, tope_total_bytes=1000)
    assert r["estado_entrega"] == "rechazada_por_contrato" and "empujar" not in d.llamadas
    assert r["violaciones"] == [{"regla": "tamano_total", "ruta": "", "detalle": "1001 > 1000 bytes"}]


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
                 "notas": ["pr_reabierto_nuevo"], "rama_empujada": True, "sha": PUNTA}
    assert d.llamadas == ["traer", "commits", "diff", "tamanos", "pr_abierto", "punta", "empujar", "pr"]
    assert d.lecturas_de_pausa == 2  # antes de empujar y antes del PR
    assert d.pr_pedido["repo"] == "o/r" and d.pr_pedido["rama"] == RAMA and d.pr_pedido["base"] == "main"
    assert d.pr_pedido["cuerpo"].startswith("informe") and d.pr_pedido["cuerpo"].endswith(PIE)


def test_un_informe_enorme_se_recorta_con_aviso_antes_de_empujar():
    """MAJOR-1: el cuerpo tiene tope; el recorte es del informe, con aviso, y el pie queda."""
    d = Dobles((A_PY,))
    _entregar(d, informe="x" * 70_000)
    cuerpo = d.pr_pedido["cuerpo"]
    assert len(cuerpo) <= MC.TOPE_CUERPO and cuerpo.endswith(PIE)
    assert "[informe recortado: se muestran" in cuerpo and "de 70000 caracteres]" in cuerpo


def test_secreto_mas_alla_del_recorte_tambien_rechaza():
    d = Dobles((A_PY,))
    r = _entregar(d, informe="x" * 70_000 + SECRETO)
    assert r["estado_entrega"] == "rechazada_por_contrato" and "empujar" not in d.llamadas


@pytest.mark.parametrize("paso", ["traer", "commits", "diff", "tamanos", "punta"])
def test_un_paso_que_falla_antes_del_empuje_es_fallo_entrega_con_el_motivo_saneado(paso):
    d = Dobles((A_PY,), falla={paso: E.EntregaRechazada(f"git_fallo: https://x-access-token:{TOKEN}@github.com")})
    r = _entregar(d)
    assert r["estado_entrega"] == "fallo_entrega" and r["pr_url"] is None and r["rama_empujada"] is False
    assert r["violaciones"][0]["regla"] == "entrega" and TOKEN not in json.dumps(r)
    assert d.llamadas[-1] == paso


@pytest.mark.parametrize("error, detalle", [
    (httpx.HTTPStatusError("x", request=httpx.Request("POST", "https://api.github.com/repos/o/r/pulls"),
                           response=httpx.Response(422)), "github_api: 422"),
    (KeyError("number"), "github_api: KeyError"),
    (ValueError(f"cuerpo raro {SECRETO}"), "github_api: ValueError"),
    (E.EntregaRechazada("x"), "github_api: EntregaRechazada"),
])
def test_un_fallo_despues_del_empuje_es_empujado_sin_pr_con_el_sha_y_sin_el_mensaje(error, detalle):
    """MAJOR-1: la rama ya está en GitHub; eso tiene estado propio y el SHA empujado."""
    d = Dobles((A_PY,), falla={"pr": error})
    r = _entregar(d)
    assert (r["estado_entrega"], r["rama_empujada"], r["sha"], r["pr_url"]) == ("empujado_sin_pr", True, PUNTA, None)
    assert r["violaciones"] == [{"regla": "entrega", "ruta": "", "detalle": detalle}]
    assert SECRETO not in json.dumps(r)


def test_pausa_puesta_antes_del_empuje_no_empuja():
    """MINOR-1: la pausa se vuelve a leer justo antes de empujar."""
    d = Dobles((A_PY,), pausas=[True])
    r = _entregar(d)
    assert (r["estado_entrega"], r["motivo"], r["rama_empujada"]) == ("sin_entregar", "pausa_puesta", False)
    assert "empujar" not in d.llamadas and "punta" not in d.llamadas


def test_pausa_ilegible_cuenta_como_puesta():
    d = Dobles((A_PY,), pausas=[OSError("no se lee")])
    assert _entregar(d)["estado_entrega"] == "sin_entregar" and "empujar" not in d.llamadas


def test_pausa_puesta_despues_del_empuje_no_abre_el_pr():
    d = Dobles((A_PY,), pausas=[False, True])
    r = _entregar(d)
    assert (r["estado_entrega"], r["rama_empujada"], r["sha"]) == ("empujado_sin_pr", True, PUNTA)
    assert r["notas"] == ["pausa_puesta"] and "pr" not in d.llamadas


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


async def _sin_pausa():
    return False


def _entregar_real(c: P.Clon, pedidos: list, *, tope_bytes: int = 1_000_000, api=None, **extra) -> dict:
    base = dict(mision_id=MID, repo="o/r", revision_legible=True, informe="informe C5", token=TOKEN,
                tope_bytes=tope_bytes, tope_total_bytes=10**9, modelo="qwen", autor=AUTOR,
                upload_pack="git-upload-pack", pausa_puesta=_sin_pausa)
    base.update(extra)

    async def correr():
        async with (api or _api(pedidos)) as cliente:
            return await MC.entregar(c, cliente=cliente, **base)
    return asyncio.run(correr())


def _en_la_base(tmp_path: Path, archivo: str, datos: bytes) -> None:
    """Agrega un archivo a `main` del «GitHub» de prueba (antes de preparar la misión)."""
    w = tmp_path / "w"
    _commit(w, archivo, datos, "base+", "-c", "user.name=q", "-c", "user.email=q@q")
    _git("push", "-q", "origin", "main", cwd=w)


# --- BLOCK-1 (auditoría escalón 3): el barrido mira TODO lo que sale, no solo las líneas agregadas ----

def test_real_secreto_en_el_mensaje_de_un_commit_se_rechaza(tmp_path, github):
    c = _preparar(tmp_path)
    _commit(c.ruta, "a", b"2", f"arreglo\n\nclave: {SECRETO}")
    pedidos: list = []
    r = _entregar_real(c, pedidos)
    assert r["estado_entrega"] == "rechazada_por_contrato", r
    assert [v["regla"] for v in r["violaciones"]] == ["secretos"]
    assert _nada_empujado(github, pedidos) and SECRETO not in json.dumps(r)


def test_real_secreto_en_el_nombre_de_un_archivo_se_rechaza(tmp_path, github):
    c = _preparar(tmp_path)
    _commit(c.ruta, f"{SECRETO}.txt", b"x\n", "archivo")
    pedidos: list = []
    r = _entregar_real(c, pedidos)
    assert r["estado_entrega"] == "rechazada_por_contrato", r
    assert {v["regla"] for v in r["violaciones"]} == {"secretos"}
    assert _nada_empujado(github, pedidos) and SECRETO not in json.dumps(r)


def test_real_secreto_en_la_ruta_anterior_de_un_renombre_se_rechaza(tmp_path, github):
    _en_la_base(tmp_path, f"{SECRETO}.txt", b"contenido largo que se mantiene\n" * 5)
    c = _preparar(tmp_path)
    _git("mv", f"{SECRETO}.txt", "limpio.txt", cwd=c.ruta)
    _git("commit", "-q", "-m", "renombro", cwd=c.ruta)
    pedidos: list = []
    r = _entregar_real(c, pedidos)
    assert r["estado_entrega"] == "rechazada_por_contrato", r
    assert _nada_empujado(github, pedidos) and SECRETO not in json.dumps(r)


def test_real_secreto_en_el_cuerpo_del_pr_se_rechaza_antes_de_empujar(tmp_path, github):
    c = _preparar(tmp_path)
    _commit(c.ruta, "a", b"2", "cambio")
    pedidos: list = []
    r = _entregar_real(c, pedidos, informe=f"dato='{SECRETO}'")
    assert r["estado_entrega"] == "rechazada_por_contrato", r
    assert _nada_empujado(github, pedidos) and SECRETO not in json.dumps(r)


def test_ruta_anterior_con_secreto_en_el_diff_neto_se_rechaza():
    d = Dobles((Cambio("limpio.py", "R", f"{SECRETO}.py", (), ()),))
    r = _entregar(d)
    assert r["estado_entrega"] == "rechazada_por_contrato" and "empujar" not in d.llamadas
    assert SECRETO not in json.dumps(r)


def _nada_empujado(github: Path, pedidos: list) -> bool:
    return f"refs/heads/{RAMA}" not in _refs(github) and pedidos == []


def test_real_camino_feliz_empuja_la_rama_y_abre_el_pr(tmp_path, github):
    c = _preparar(tmp_path)
    _commit(c.ruta, "a", b"2", "cambio")
    pedidos: list = []
    r = _entregar_real(c, pedidos)
    punta = _git("rev-parse", "HEAD", cwd=c.ruta)
    assert r == {"estado_entrega": "abierto", "pr_url": "https://gh/pr/1", "violaciones": [], "notas": [],
                 "rama_empujada": True, "sha": punta}
    assert _refs(github)[f"refs/heads/{RAMA}"] == punta
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


def _api_que_responde(pedidos: list, post_pr, post_etiqueta=None):
    """La API de PRs con respuestas elegidas para el POST del PR y el de la etiqueta."""
    def manejar(req: httpx.Request) -> httpx.Response:
        pedidos.append((req.method, req.url.path, req.content))
        if req.method == "GET":
            return httpx.Response(200, json=[])
        if req.url.path.endswith("/labels"):
            return post_etiqueta or httpx.Response(200, json=[])
        return post_pr
    return httpx.AsyncClient(transport=httpx.MockTransport(manejar), base_url="https://api.github.com")


@pytest.mark.parametrize("respuesta, detalle", [
    (httpx.Response(201, json={"html_url": "https://gh/pr/1"}), "github_api: KeyError"),        # 201 sin number
    (httpx.Response(201, text="<html>no es json</html>"), "github_api: JSONDecodeError"),       # cuerpo no JSON
    (httpx.Response(422, json={"message": "Validation Failed", "errors": [{"message": f"no {SECRETO}"}]}),
     "github_api: 422"),                                                                          # 422 tras el push
])
def test_real_fallo_de_la_api_despues_del_empuje_es_empujado_sin_pr(tmp_path, github, respuesta, detalle):
    """MAJOR-1: la rama YA está en GitHub; el resultado lo dice, con el SHA, y sin el texto del error."""
    c = _preparar(tmp_path)
    _commit(c.ruta, "a", b"2", "cambio")
    punta = _git("rev-parse", "HEAD", cwd=c.ruta)
    pedidos: list = []
    r = _entregar_real(c, pedidos, api=_api_que_responde(pedidos, respuesta))
    assert (r["estado_entrega"], r["rama_empujada"], r["sha"], r["pr_url"]) == ("empujado_sin_pr", True, punta, None)
    assert r["violaciones"] == [{"regla": "entrega", "ruta": "", "detalle": detalle}]
    assert _refs(github)[f"refs/heads/{RAMA}"] == punta and SECRETO not in json.dumps(r)


def test_real_pr_abierto_pero_sin_etiqueta_es_abierto_con_nota(tmp_path, github):
    c = _preparar(tmp_path)
    _commit(c.ruta, "a", b"2", "cambio")
    pedidos: list = []
    r = _entregar_real(c, pedidos, api=_api_que_responde(
        pedidos, httpx.Response(201, json={"number": 7, "html_url": "https://gh/pr/7"}), httpx.Response(500)))
    assert (r["estado_entrega"], r["pr_url"], r["notas"]) == ("abierto", "https://gh/pr/7", ["sin_etiqueta"])


def test_real_tope_total_del_rango(tmp_path, github):
    """MINOR-3 con git real: dos archivos de 600 B, cada uno bajo el tope por archivo; juntos no."""
    c = _preparar(tmp_path)
    _commit(c.ruta, "x", b"1" * 600, "x")
    _commit(c.ruta, "y", b"2" * 600, "y")
    pedidos: list = []
    r = _entregar_real(c, pedidos, tope_bytes=1000, tope_total_bytes=1000)
    assert r["estado_entrega"] == "rechazada_por_contrato", r
    assert [v["regla"] for v in r["violaciones"]] == ["tamano_total"] and _nada_empujado(github, pedidos)


def test_real_sin_commits_es_sin_cambios(tmp_path, github):
    c = _preparar(tmp_path)
    (c.ruta / "sin-commitear.txt").write_text("x")
    pedidos: list = []
    assert _entregar_real(c, pedidos)["estado_entrega"] == "sin_cambios"
    assert f"refs/heads/{RAMA}" not in _refs(github)
    assert [m for m, _, _ in pedidos] == ["GET"]  # solo mira si hay un PR previo (MINOR-5)


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


# --- ronda final de la auditoría (MAJOR-A, MAJOR-B, MINOR-C) ------------------------------------

_EMPUJE_CAIDO = E.EntregaRechazada(f"git_fallo: push https://x-access-token:{TOKEN}@github.com")


def test_empuje_fallido_pero_el_remoto_tiene_la_punta_sigue_al_pr():
    """MAJOR-A: el empuje dio error (p. ej. se cortó la respuesta) pero ls-remote muestra exactamente
    el SHA que íbamos a empujar: la rama SÍ está; se sigue al PR."""
    d = Dobles((A_PY,), falla={"empujar": _EMPUJE_CAIDO}, remoto=PUNTA)
    r = _entregar(d)
    assert (r["estado_entrega"], r["rama_empujada"], r["sha"]) == ("abierto", True, PUNTA)
    assert d.llamadas[-3:] == ["empujar", "ls_remote", "pr"]


@pytest.mark.parametrize("remoto", [None, "d" * 40])
def test_empuje_fallido_y_el_remoto_sin_la_punta_es_fallo_sin_rama(remoto):
    d = Dobles((A_PY,), falla={"empujar": _EMPUJE_CAIDO}, remoto=remoto)
    r = _entregar(d)
    assert (r["estado_entrega"], r["rama_empujada"]) == ("fallo_entrega", False)
    assert "pr" not in d.llamadas and TOKEN not in json.dumps(r)


def test_empuje_fallido_y_ls_remote_tambien_es_fallo_con_rama_desconocida():
    d = Dobles((A_PY,), falla={"empujar": _EMPUJE_CAIDO, "ls_remote": E.EntregaRechazada("git_fallo: ls-remote")})
    r = _entregar(d)
    assert (r["estado_entrega"], r["rama_empujada"]) == ("fallo_entrega", "desconocido")
    assert "pr" not in d.llamadas


def test_pausa_tras_el_empuje_con_pr_previo_da_su_url_y_avisa_que_esta_desactualizado():
    """MAJOR-B: `empujado_sin_pr` es «PR de este turno no confirmado»; si ya había uno abierto, se da."""
    d = Dobles((A_PY,), pausas=[False, True], pr_previo="https://gh/pr/9")
    r = _entregar(d)
    assert (r["estado_entrega"], r["pr_url"]) == ("empujado_sin_pr", "https://gh/pr/9")
    assert r["notas"] == ["pausa_puesta", "pr_con_informe_desactualizado"] and "pr" not in d.llamadas


def test_sin_pr_previo_empujado_sin_pr_no_inventa_url():
    d = Dobles((A_PY,), pausas=[False, True])
    r = _entregar(d)
    assert (r["estado_entrega"], r["pr_url"], r["notas"]) == ("empujado_sin_pr", None, ["pausa_puesta"])


def _api_turno_2(pedidos: list, parche: httpx.Response, pr_nuevo: httpx.Response | None = None):
    """Turno 2: ya hay un PR abierto de la misión (#1). El PATCH (o el POST) responden lo elegido."""
    def manejar(req: httpx.Request) -> httpx.Response:
        pedidos.append((req.method, req.url.path, req.content))
        if req.method == "GET":
            estado = req.url.params.get("state")
            abiertos = [{"number": 1, "html_url": "https://gh/pr/1"}] if pr_nuevo is None else []
            return httpx.Response(200, json=abiertos if estado == "open" else [])
        if req.method == "PATCH":
            return parche
        if req.url.path.endswith("/labels"):
            return httpx.Response(200, json=[])
        return pr_nuevo
    return httpx.AsyncClient(transport=httpx.MockTransport(manejar), base_url="https://api.github.com")


def test_real_turno_2_con_patch_502_da_el_pr_previo_desactualizado(tmp_path, github):
    """MAJOR-B con git real: la rama se actualizó, el PATCH del PR falló; el PR existe con el informe
    del turno anterior -- se da su URL y se dice."""
    c = _preparar(tmp_path)
    _commit(c.ruta, "a", b"2", "cambio")
    punta = _git("rev-parse", "HEAD", cwd=c.ruta)
    pedidos: list = []
    r = _entregar_real(c, pedidos, api=_api_turno_2(pedidos, httpx.Response(502)))
    assert (r["estado_entrega"], r["pr_url"], r["sha"]) == ("empujado_sin_pr", "https://gh/pr/1", punta)
    assert r["notas"] == ["pr_con_informe_desactualizado"]
    assert r["violaciones"] == [{"regla": "entrega", "ruta": "", "detalle": "github_api: 502"}]
    assert _refs(github)[f"refs/heads/{RAMA}"] == punta


@pytest.mark.parametrize("url", ["", "http://gh/pr/1", 5, "https://", "https://gh/pr/1 x"])
def test_real_pr_sin_url_https_valida_no_esta_confirmado(tmp_path, github, url):
    """MINOR-C: una respuesta 201 sin `html_url` https válida no confirma el PR."""
    c = _preparar(tmp_path)
    _commit(c.ruta, "a", b"2", "cambio")
    pedidos: list = []
    r = _entregar_real(c, pedidos, api=_api_turno_2(pedidos, httpx.Response(500),
                                                   httpx.Response(201, json={"number": 1, "html_url": url})))
    assert (r["estado_entrega"], r["rama_empujada"], r["pr_url"]) == ("empujado_sin_pr", True, None)
    assert r["violaciones"] == [{"regla": "entrega", "ruta": "", "detalle": "github_api: pr_url_invalida"}]
