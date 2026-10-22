# tests/test_ejecutor_contratos_canario_codigo.py
"""verificar_codigo con la cuenta falsa: qué observación da cada Fallo. Principio VII: un
control que no falla no valida -- cada código tiene su caso."""
import asyncio
from pathlib import Path

from jax.ejecutor.contratos import canario_codigo
from jax.ejecutor.contratos.cuenta_axioma import Cuenta
from jax.ejecutor.contratos.fallo import Fallo

C = Cuenta("axioma", 58291, Path("/k"), Path("/opt/node/bin"), Path("/opt/ejecutor/lib"), Path("/etc/p.json"), Path("/home/axioma"))
BLOQUEO_PUSH = b'contrato="c1" codigo="prohibido" regla="codigo_git_push" hosts=["hall9000"]\n'


def _correr(*, push_rc=2, push_stderr=BLOQUEO_PUSH, ls_remote_rc=1):
    vistos = []

    async def correr(c, remoto, *, entrada=b"", tope_s):
        vistos.append(remoto)
        if remoto == canario_codigo._LS_REMOTE:
            return ls_remote_rc, b"" if ls_remote_rc != 0 else b"deadbeef\tHEAD\n", b""
        if "gancho.sh" in remoto:
            return push_rc, b"", push_stderr
        raise AssertionError(f"comando inesperado: {remoto}")

    correr.vistos = vistos
    return correr


def test_todo_vivo_no_da_fallos():
    assert asyncio.run(canario_codigo.verificar_codigo(C, correr=_correr())) == ()


def test_push_no_bloqueado_da_fallo():
    fallos = asyncio.run(canario_codigo.verificar_codigo(C, correr=_correr(push_rc=0, push_stderr=b"")))
    assert Fallo("codigo", "push_no_bloqueado", (("rc", 0),)) in fallos


def test_push_bloqueado_sin_su_regla_da_fallo():
    fallos = asyncio.run(canario_codigo.verificar_codigo(C, correr=_correr(push_stderr=b"otra cosa\n")))
    assert any(f.codigo == "push_sin_su_regla" for f in fallos)


def test_cerco_deja_salir_a_github_da_fallo():
    fallos = asyncio.run(canario_codigo.verificar_codigo(C, correr=_correr(ls_remote_rc=0)))
    assert Fallo("codigo", "cerco_deja_salir_a_github") in fallos


def test_ls_remote_corre_como_la_cuenta_no_por_el_gancho():
    correr = _correr()
    asyncio.run(canario_codigo.verificar_codigo(C, correr=correr))
    assert canario_codigo._LS_REMOTE in correr.vistos
    assert not any("gancho.sh" in v and "ls-remote" in v for v in correr.vistos)


def test_diff_no_bloquea_flujos_ci_da_fallo(monkeypatch):
    monkeypatch.setattr(canario_codigo.reglas_diff, "revisar", lambda cambios, repo: ())
    fallos = asyncio.run(canario_codigo.verificar_codigo(C, correr=_correr()))
    assert Fallo("codigo", "c1_diff_no_bloquea") in fallos


def test_todos_los_fallos_son_del_contrato_codigo():
    fallos = asyncio.run(canario_codigo.verificar_codigo(
        C, correr=_correr(push_rc=0, push_stderr=b"", ls_remote_rc=0)))
    assert fallos and all(f.contrato == "codigo" for f in fallos)
