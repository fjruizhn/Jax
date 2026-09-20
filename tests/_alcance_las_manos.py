"""¿Una identidad de test ALCANZA una ruta de LAS MANOS, o esta cerrada?

**Por que existe (2026-09-20).** Tres archivos de test pegan contra `/plan` y
`/audit/tail` de `server.app`. Desde `las_manos/auth_servicio.py` (commit
cbddb38, 2026-09-17, "LAS MANOS autentica a sus llamadores") esas rutas quedaron
**sin ninguna identidad que las alcance, a proposito**: es una decision de
seguridad escrita de Fernando, no un defecto. El propio mensaje de ese commit
lo dice: "/execute, /plan y /audit/tail sin identidad que los alcance (sin
llamadores reales)".

Consecuencia: esos ocho tests **no pueden pasar**, y fallaban para cualquiera
que corriera la suite. Estaban declarados como excepcion en
`policy/tests/test_archivos_de_test_wireados_en_ci.py` -- o sea que CI no los
corre y nadie los ve -- pero el motivo vivia ahi y no donde duele. Ocho rojos
permanentes entrenan a mirar el rojo sin leerlo, que es como se cuela el noveno
que si importa.

**No se saltan a mano.** Se SONDEA el candado: si la ruta contesta con el codigo
del candado, se salta con el motivo escrito; si contesta cualquier otra cosa, el
test corre. Asi, **el dia que una identidad real necesite esas rutas y
`auth_servicio` le abra permiso, los ocho vuelven solos a la vida** -- sin que
nadie tenga que acordarse de borrar un `skip`.

El candado en si NO se queda sin vigilancia: lo defiende
`tests/test_las_manos_auth_servicio.py`, que SI corre en CI e incluye
`test_execute_y_audit_no_los_alcanza_ninguna_identidad` y
`test_toda_ruta_no_publica_exige_credencial`.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import sys
from pathlib import Path

LAS_MANOS = Path(__file__).resolve().parent.parent / "las_manos"
if str(LAS_MANOS) not in sys.path:
    sys.path.insert(0, str(LAS_MANOS))

#: El codigo con el que `auth_servicio` rechaza a un llamador sin identidad.
CODIGO_DEL_CANDADO = "credencial_de_servicio_invalida"

MOTIVO = (
    "auth_servicio.py (cbddb38, 2026-09-17) deja esta ruta sin ninguna "
    "identidad que la alcance, a proposito: decision de seguridad, no un "
    "defecto del test. Vuelve a correr solo el dia que se abra un permiso. "
    "El candado lo vigila tests/test_las_manos_auth_servicio.py."
)


def _codigo_del_cuerpo(respuesta) -> str | None:
    try:
        detalle = respuesta.json().get("detail")
    except Exception:  # noqa: BLE001  # fail-closed: un cuerpo que no es JSON no es el candado; devuelve None y el test NO se salta
        return None
    if isinstance(detalle, dict):
        return detalle.get("code")
    return None


def cerrada(metodo: str, ruta: str, **kwargs) -> bool:
    """True si `auth_servicio` no deja alcanzar la ruta con un cliente pelado.

    Sondea de verdad, no supone. Si el sondeo no se puede hacer (la app no
    levanta en este entorno) devuelve False: mejor que el test falle y se vea,
    a saltarlo por una razon que no se comprobo.
    """
    try:
        from fastapi.testclient import TestClient

        import server  # noqa: PLC0415 -- import tardio: necesita el sys.path de arriba

        # SIN `with`: el context manager corre el lifespan de la app y eso
        # levanta el reaper, que manda alertas REALES a Telegram. Medido el
        # 2026-09-20: dos alertas salieron de verdad por hacerlo asi. Los tres
        # archivos que usan este ayudante construyen el TestClient igual de
        # pelado, por el mismo motivo.
        c = TestClient(server.app, raise_server_exceptions=False)
        r = c.request(metodo, ruta, **kwargs)
    except Exception:  # noqa: BLE001  # fail-closed: si el sondeo no se pudo hacer NO se salta nada -- un skip por una razon no comprobada es peor que un rojo
        return False
    return r.status_code in (401, 403) and _codigo_del_cuerpo(r) == CODIGO_DEL_CANDADO
