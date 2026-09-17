# tests/test_ejecutor_cuenta_de_servicio.py
"""Los servicios de JAX no corren como el operador (decisión de Fernando, 2026-09-17): corren
como la cuenta de servicio `jaxsvc`, y por eso `/etc/jax/.env` puede dejar de ser legible por
cualquier proceso de `fruiz`. Este control ata las unidades y los instaladores a esa cuenta:
un `User=fruiz` o un `install -o fruiz` que vuelva deja al servicio sin poder escribir lo suyo
(o, peor, devuelve el secreto al operador) y nadie se entera hasta el despliegue.

La cuenta de la JAULA (`JAX_EJECUTOR_CUENTA`, axioma) es otra cosa y no se toca acá.
"""
from pathlib import Path

import pytest

OPS = Path(__file__).resolve().parents[1] / "ops" / "ejecutor"
CUENTA_DE_SERVICIO = "jaxsvc"
UNIDADES = ("jax-ejecutor-proxy.service", "ejecutor-vigia@.service")
INSTALADORES = ("instalar_contratos.sh", "instalar_vigia.sh", "instalar_registro_y_cerco.sh")


@pytest.mark.parametrize("nombre", UNIDADES)
def test_las_unidades_corren_como_la_cuenta_de_servicio(nombre):
    lineas = [l.strip() for l in (OPS / nombre).read_text().splitlines()]
    assert f"User={CUENTA_DE_SERVICIO}" in lineas, f"{nombre} no corre como {CUENTA_DE_SERVICIO}"
    assert not any(l.startswith("User=fruiz") for l in lineas)


@pytest.mark.parametrize("nombre", UNIDADES)
def test_las_unidades_le_dan_un_hogar_propio_a_la_cuenta(nombre):
    """Sin HOME propio, ssh busca known_hosts en el del operador y el turno muere con 255."""
    texto = (OPS / nombre).read_text()
    assert "Environment=HOME=/var/lib/jaxsvc" in texto


@pytest.mark.parametrize("nombre", INSTALADORES)
def test_los_instaladores_crean_los_directorios_para_la_cuenta_de_servicio(nombre):
    texto = (OPS / nombre).read_text()
    assert "-o fruiz" not in texto, f"{nombre} sigue creando directorios del operador"
    assert f"-o {CUENTA_DE_SERVICIO}" in texto
