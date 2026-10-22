"""Whitelist anti-fabricación (capa 2): derivada por job, nunca escrita a mano.

Sin DB ni red: las fuentes se inyectan (`Fuentes`). Lo que se ejercita es el
contrato: qué fuentes son obligatorias (caída -> DetectorNoDisponible, JAMÁS una
lista vacía), cuáles son opcionales (caída -> declarada, no excepción), y el
memo de 4 entradas con clave (huella de esquema, HEAD, hash de status).
"""
from __future__ import annotations

import asyncio
import subprocess
from collections import Counter

import pytest

from jax.anti_fabricacion import whitelist as wl_mod
from jax.anti_fabricacion.whitelist import (
    DetectorNoDisponible, EsquemaDB, EstadoGit, Fuentes, derivar,
)


def _esquema():
    return EsquemaDB(
        columnas=(("jacobs_steps", "step_id"), ("jacobs_steps", "status"), ("axioma_config", "config_key")),
        funciones=frozenset({"JSON_EXTRACT", "COALESCE"}),
        palabras=frozenset({"SELECT", "WHERE"}),
        motores=frozenset({"InnoDB"}),
        charsets=frozenset({"utf8mb4"}),
    )


def _git(head="h1", status="s1"):
    return EstadoGit(head=head, status_hash=status, entradas=(
        ("b1", "jax/core/x.py"), ("b2", "las_manos/motor_registry/worker.py"), ("b3", "README.md"),
    ))


class Mundo:
    """Fuentes falsas con contador de llamadas y estado mutable."""

    def __init__(self):
        self.llamadas = Counter()
        self.huella = "H1"
        self.git = _git()
        self.cae: set[str] = set()

    def fuentes(self, **extra) -> Fuentes:
        async def huella():
            self.llamadas["huella"] += 1
            if "huella" in self.cae:
                raise ConnectionError("huella caida")
            return self.huella

        async def esquema():
            self.llamadas["esquema"] += 1
            if "esquema" in self.cae:
                raise ConnectionError("db caida")
            return _esquema()

        async def git():
            self.llamadas["git"] += 1
            if "git" in self.cae:
                raise OSError("git caido")
            return self.git

        async def simbolos(_estado):
            self.llamadas["simbolos"] += 1
            return frozenset({"MAX_WRITE_BYTES", "authorize_and_execute_tool_call"})

        return Fuentes(huella=huella, esquema=esquema, git=git, simbolos=simbolos, **extra)


@pytest.fixture(autouse=True)
def _memo_limpio():
    wl_mod._MEMO.clear()
    yield
    wl_mod._MEMO.clear()


def _correr(coro):
    return asyncio.run(coro)


def test_deriva_tablas_columnas_pares_archivos_y_sql():
    m = Mundo()
    wl = _correr(derivar(m.fuentes()))
    assert wl.tablas == {"jacobs_steps", "axioma_config"}
    assert "status" in wl.columnas
    assert "jacobs_steps.status" in wl.pares and "jacobs_steps.nada" not in wl.pares
    assert "las_manos/motor_registry/worker.py" in wl.archivos
    assert "las_manos/motor_registry" in wl.directorios and "las_manos" in wl.directorios
    assert "JSON_EXTRACT" in wl.sql and "InnoDB" in wl.sql and "utf8mb4" in wl.sql
    assert "MAX_WRITE_BYTES" in wl.simbolos
    assert wl.clave_cache == ("H1", "h1", "s1")


def test_misma_clave_reusa_y_cambio_de_huella_reconstruye():
    m = Mundo()
    _correr(derivar(m.fuentes()))
    _correr(derivar(m.fuentes()))
    assert m.llamadas["esquema"] == 1, "la misma clave no debe reconstruir"
    assert m.llamadas["huella"] == 2, "la clave se compara en CADA uso (invalida entre procesos)"
    m.huella = "H2"
    wl = _correr(derivar(m.fuentes()))
    assert m.llamadas["esquema"] == 2, "un cambio de huella de esquema debe reconstruir"
    assert wl.clave_cache[0] == "H2"


@pytest.mark.parametrize("campo, nuevo", [("head", "h2"), ("status_hash", "s2")])
def test_cambio_de_head_o_de_status_reconstruye(campo, nuevo):
    m = Mundo()
    _correr(derivar(m.fuentes()))
    m.git = EstadoGit(**{**m.git.__dict__, campo: nuevo})
    _correr(derivar(m.fuentes()))
    assert m.llamadas["esquema"] == 2


def test_memo_de_cuatro_entradas_expulsa_la_mas_vieja():
    m = Mundo()
    for i in range(5):
        m.huella = f"H{i}"
        _correr(derivar(m.fuentes()))
    assert m.llamadas["esquema"] == 5
    assert len(wl_mod._MEMO) == 4
    m.huella = "H4"
    _correr(derivar(m.fuentes()))
    assert m.llamadas["esquema"] == 5, "la mas reciente sigue en memo"
    m.huella = "H0"
    _correr(derivar(m.fuentes()))
    assert m.llamadas["esquema"] == 6, "la mas vieja fue expulsada y se reconstruye"


@pytest.mark.parametrize("fuente", ["esquema", "git", "huella"])
def test_fuente_obligatoria_caida_levanta_y_nunca_devuelve_lista_vacia(fuente):
    m = Mundo()
    m.cae.add(fuente)
    with pytest.raises(DetectorNoDisponible) as exc:
        _correr(derivar(m.fuentes()))
    assert fuente in str(exc.value) or "caid" in str(exc.value).lower()


def test_obligatoria_caida_no_se_enmascara_con_el_memo():
    """Con memo caliente, si la huella ya no se puede leer no se sirve el viejo."""
    m = Mundo()
    _correr(derivar(m.fuentes()))
    m.cae.add("huella")
    with pytest.raises(DetectorNoDisponible):
        _correr(derivar(m.fuentes()))


def test_fuente_opcional_caida_queda_declarada_sin_excepcion():
    m = Mundo()

    async def rutas_caidas():
        raise RuntimeError("app no cargada")

    async def modelos():
        return frozenset({"qwen3.6", "kimi-k2.7-code"})

    wl = _correr(derivar(m.fuentes(rutas_app=rutas_caidas, modelos=modelos)))
    assert "rutas_app" in wl.no_disponibles
    assert "modelos" not in wl.no_disponibles and "qwen3.6" in wl.modelos
    assert wl.rutas_app == frozenset()


def test_opcional_ausente_tambien_se_declara():
    wl = _correr(derivar(Mundo().fuentes()))
    assert {"rutas_app", "claves", "modelos"} <= wl.no_disponibles


def test_las_raices_del_sistema_de_archivos_se_derivan_no_se_escriben():
    wl = _correr(derivar(Mundo().fuentes()))
    assert "etc" in wl.raices_fs and "usr" in wl.raices_fs


# --- ast: constantes y funciones del codigo, cache por SHA de blob -------------------

def test_simbolos_por_ast_con_cache_por_sha_de_blob(tmp_path, monkeypatch):
    (tmp_path / "a.py").write_text("MAX_WRITE_BYTES = 5\n\ndef authorize():\n    pass\n\nclass Cosa:\n    pass\n")
    (tmp_path / "roto.py").write_text("def (:\n")
    wl_mod._CACHE_AST.clear()
    s = wl_mod.simbolos_de_archivos(tmp_path, (("sha-a", "a.py"), ("sha-r", "roto.py"), ("sha-m", "README.md")))
    assert {"MAX_WRITE_BYTES", "authorize", "Cosa"} <= s
    # mismo sha, contenido distinto en disco: se sirve del cache (la clave es el blob, no la ruta)
    (tmp_path / "a.py").write_text("OTRA = 1\n")
    s2 = wl_mod.simbolos_de_archivos(tmp_path, (("sha-a", "a.py"),))
    assert "MAX_WRITE_BYTES" in s2 and "OTRA" not in s2
    # sha nuevo: se vuelve a parsear
    s3 = wl_mod.simbolos_de_archivos(tmp_path, (("sha-a2", "a.py"),))
    assert "OTRA" in s3 and "MAX_WRITE_BYTES" not in s3


# --- git real: ls-files -s, HEAD y status --------------------------------------------

def _repo(tmp_path):
    def g(*a):
        subprocess.run(["git", "-C", str(tmp_path), *a], check=True, capture_output=True,
                       env={"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
                            "GIT_COMMITTER_EMAIL": "t@t", "PATH": "/usr/bin:/bin"})
    g("init", "-q")
    (tmp_path / "m.py").write_text("X = 1\n")
    g("add", "m.py")
    g("commit", "-q", "-m", "uno")
    return g


def test_lector_de_git_real_cambia_con_el_status(tmp_path):
    _repo(tmp_path)
    e1 = wl_mod.leer_git(tmp_path)
    assert e1.entradas and e1.entradas[0][1] == "m.py" and len(e1.head) == 40
    (tmp_path / "sucio.txt").write_text("x")
    e2 = wl_mod.leer_git(tmp_path)
    assert e2.head == e1.head and e2.status_hash != e1.status_hash


def test_lector_de_git_en_un_directorio_que_no_es_repo_levanta(tmp_path):
    with pytest.raises(DetectorNoDisponible):
        wl_mod.leer_git(tmp_path)
