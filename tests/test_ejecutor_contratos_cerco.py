# tests/test_ejecutor_contratos_cerco.py
"""Cerco de red de la cuenta del Ejecutor (C3): sólo proxy, canario y SSH del inventario."""
import pytest

from jax.ejecutor.contratos import cerco
from tests.test_ejecutor_contratos_politica import OTRO_UID, doc_base, escribir

ESPERADO = """# Generado por jax/ejecutor/contratos/cerco.py: no se edita a mano.
table inet ejecutor_cerco
delete table inet ejecutor_cerco
table inet ejecutor_cerco {
\tset destinos_ssh {
\t\ttype ipv4_addr . inet_service
\t\telements = { 127.0.0.1 . 58291, 192.0.2.20 . 58291 }
\t}
\tchain salida {
\t\ttype filter hook output priority filter; policy accept;
\t\tmeta skuid 1001 jump cuenta
\t}
\tchain cuenta {
\t\tct state established,related accept
\t\toifname "lo" ip daddr 127.0.0.1 tcp dport { 18435, 18436 } accept
\t\tip daddr . tcp dport @destinos_ssh accept
\t\tmeta l4proto tcp counter reject with tcp reset
\t\tcounter reject with icmpx admin-prohibited
\t}
}
"""


def test_render_exacto():
    assert cerco.renderizar(1001, (18436, 18435), (("192.0.2.20", 58291), ("127.0.0.1", 58291))) == ESPERADO


@pytest.mark.parametrize("uid, puertos, destinos", [
    (True, (1,), (("127.0.0.1", 22),)),
    (0, (1,), (("127.0.0.1", 22),)),
    (1001, (), (("127.0.0.1", 22),)),
    (1001, (70000,), (("127.0.0.1", 22),)),
    (1001, (1,), ()),
    (1001, (1,), (("::1", 22),)),
    (1001, (1,), (("no-es-ip", 22),)),
])
def test_rechaza_lo_que_no_puede_ser(uid, puertos, destinos):
    with pytest.raises(ValueError):
        cerco.renderizar(uid, puertos, destinos)


def test_desde_la_politica(tmp_path):
    from jax.ejecutor.contratos import politica
    p = politica.cargar(escribir(tmp_path, doc_base()), uid_de_la_cuenta=OTRO_UID)
    texto = cerco.desde_politica(p, 1001, (18435,), frozenset({"atemai"}))
    assert "elements = { 192.0.2.5 . 58291, 192.0.2.11 . 58291 }" in texto


def test_sin_remotas_habilitadas_solo_la_local(tmp_path):
    """Una máquina sin contratos remotos (sin llave del freno) no entra al cerco: la cuenta no la alcanza."""
    from jax.ejecutor.contratos import politica
    p = politica.cargar(escribir(tmp_path, doc_base()), uid_de_la_cuenta=OTRO_UID)
    assert "elements = { 192.0.2.5 . 58291 }" in cerco.desde_politica(p, 1001, (18435,), frozenset())


def test_habilitada_fuera_de_la_politica_es_un_error(tmp_path):
    from jax.ejecutor.contratos import politica
    p = politica.cargar(escribir(tmp_path, doc_base()), uid_de_la_cuenta=OTRO_UID)
    with pytest.raises(ValueError):
        cerco.desde_politica(p, 1001, (18435,), frozenset({"otra"}))


def test_principal_lee_las_remotas_del_freno(tmp_path, monkeypatch):
    import pwd
    ruta = escribir(tmp_path, doc_base())
    monkeypatch.setattr(pwd, "getpwnam", lambda n: type("P", (), {"pw_uid": 1001})())
    env = {"JAX_EJECUTOR_CUENTA": "axioma", "JAX_PROXY_CARRIL_PUERTO": "18435", "JAX_EJECUTOR_CANARIO_PUERTO": "18436",
           "JAX_EJECUTOR_FRENO_REMOTOS": " bridge ,"}
    assert cerco.principal([str(ruta), str(tmp_path / "c.nft")], env) == 0
    assert "elements = { 192.0.2.5 . 58291, 192.0.2.20 . 58291 }" in (tmp_path / "c.nft").read_text()
    del env["JAX_EJECUTOR_FRENO_REMOTOS"]
    with pytest.raises(KeyError):
        cerco.principal([str(ruta), str(tmp_path / "c.nft")], env)
