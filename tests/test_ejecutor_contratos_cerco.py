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
    texto = cerco.desde_politica(p, 1001, (18435,))
    assert "elements = { 192.0.2.5 . 58291, 192.0.2.11 . 58291, 192.0.2.20 . 58291 }" in texto
