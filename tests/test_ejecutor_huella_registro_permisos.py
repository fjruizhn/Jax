# tests/test_ejecutor_huella_registro_permisos.py
"""M-1 (ronda 8): verifica CONTRA PERMISOS REALES -- replicados en el scratch, no
supuestos -- que la CLI `aceptar` no puede escribir el registro de C3 corriendo como
`fruiz`, y sí puede escribir corriendo como el dueño real del archivo (`jaxsvc` en
producción). Los números vienen de producción, verificados el 2026-09-22 con
`sudo getfacl` en hall9000:

    /var/log/jax-ejecutor/registro.jsonl  -rw-r-----+ jaxsvc:jaxsvc, user:fruiz:r--
    /var/log/jax-ejecutor/               drwxr-x---+ jaxsvc:jaxsvc, user:fruiz:r-x

"Su equivalente comprobable" (mismo patrón que
`test_la_cuenta_puede_atravesar_pero_no_listar_ronda4_b2`): no se puede cambiar de UID
sin privilegios dentro de un test, así que se reducen los permisos DE BASE del propio
usuario del test a lo que la ACL real le da a `fruiz` (`r--`, sin escritura) y se
comprueba el mismo comportamiento que tendría la CLI corriendo sin `sudo -u jaxsvc`.
"""
from __future__ import annotations

import os
import shutil

import pytest

from jax.ejecutor.contratos.registro import Registro

requiere_setfacl = pytest.mark.skipif(shutil.which("setfacl") is None,
                                      reason="setfacl no está instalado")


@requiere_setfacl
def test_escribir_el_registro_con_solo_r_falla_como_fruiz_sin_sudo(tmp_path):
    """Replica la ACL real (`fruiz:r--`) sobre un archivo propio: sin `sudo -u jaxsvc`,
    la CLI NO puede anotar en el registro -- confirma que la vieja afirmación de
    `huella.py` ("fruiz ya puede escribir ahí: es el mismo dueño") era falsa."""
    registro_ruta = tmp_path / "registro.jsonl"
    registro_ruta.write_text("")
    registro_ruta.chmod(0o400)  # r--------: ni siquiera el propio dueño escribe

    try:
        with pytest.raises(PermissionError):
            Registro(registro_ruta)  # el open() con O_WRONLY ya falla al construirlo
    finally:
        registro_ruta.chmod(0o600)  # se restaura para que tmp_path se limpie sin lío


@requiere_setfacl
def test_escribir_el_registro_funciona_siendo_el_dueno(tmp_path):
    """El mismo código, con permisos de ESCRITURA (lo que tiene `jaxsvc` sobre su
    propio archivo) -- SÍ ancla una línea. Simetría con el test anterior: no es que
    `Registro.anotar` esté roto, es una cuestión de identidad del proceso (M-1)."""
    registro_ruta = tmp_path / "registro.jsonl"
    registro_ruta.write_text("")
    registro_ruta.chmod(0o600)  # rw-------: como el dueño real

    reg = Registro(registro_ruta)
    try:
        n = reg.anotar({"evento": "prueba"})
        assert n == 1
        assert "prueba" in registro_ruta.read_text()
    finally:
        reg.cerrar()


def test_docstring_de_registrar_aceptacion_no_afirma_que_fruiz_sea_el_dueno():
    """La frase falsa señalada en ronda 8 (M-1, `huella.py:257`) no puede volver: el
    registro es de `jaxsvc`, no de `fruiz`."""
    import jax.ejecutor.contratos.huella as H

    doc = H._registrar_aceptacion.__doc__ or ""
    assert "es el mismo dueño" not in doc
    assert "sudo -u jaxsvc" in doc
