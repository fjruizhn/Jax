"""Ningún módulo/paquete de primer nivel de `las_manos/` puede llamarse
igual que algo de primer nivel en la raíz del repo -- el próximo `policy.py`.

**MAJOR-2 (ronda 1 de revisión de PR#262).** El fix de `jax#260`
(`tests/test_arranque_las_manos_no_shadowea_policy.py`) arregla la
colisión CONOCIDA (`las_manos/policy.py` vs `policy/` de la raíz), pero
nada detectaba que hubiera una en primer lugar. LAS MANOS arranca con
`WorkingDirectory=las_manos/` (`uvicorn server:app`), y con eso `sys.path[0]`
termina siendo el cwd, por delante de CUALQUIER `PYTHONPATH` (verificado a
mano contra el venv real de producción -- ver el docstring de
`test_arranque_las_manos_no_shadowea_policy.py`). Cualquier `.py` o paquete
que se cree mañana en `las_manos/` con el mismo nombre que algo de la raíz
(`las_manos/ops.py`, `las_manos/tools.py`, `las_manos/jaxctl.py`, ...)
rompe el arranque real exactamente igual, y sin este test nadie se entera
hasta el próximo despliegue roto.

**La excepción real, y por qué se verifica por INODO y no por nombre.** Hoy
la intersección es `{hyde_sandbox, jacobs}` -- pero NO son shadows: son
symlinks al MISMO archivo físico
(`las_manos/hyde_sandbox.py -> ../hyde_sandbox.py`,
`las_manos/jacobs -> ../jacobs`). Permitirlos por NOMBRE sería una
allowlist ciega: el día que alguien reemplace el symlink por un archivo de
contenido distinto (mismo nombre, ya no el mismo archivo), la excepción
seguiría verde sin razón para seguir viva. Por eso el chequeo resuelve
symlinks (`Path.resolve()`) y compara la ruta REAL, no el nombre.

**Por qué "rastreado por git" y no una lista de exclusiones a mano.** Un
checkout limpio de producción nunca tiene `las_manos/__pycache__/` ni
`las_manos/logs/` (los dos en `.gitignore`) -- mirar el filesystem sin
filtrar los trataría como shadows fantasma que no existen en ningún
despliegue real. `git check-ignore` responde exactamente "¿esto termina en
un checkout limpio?", así que no hay que mantener la lista de exclusiones
a mano ni que se desincronice de `.gitignore`.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
LAS_MANOS = RAIZ / "las_manos"

#: Nombres que, si coinciden entre la raíz y las_manos/, son el MISMO
#: archivo físico (symlink) -- no un shadow. Se verifica por inodo/ruta
#: real en el test, esto sólo dice CUÁLES nombres tienen permiso de
#: intentarlo.
PERMITIDOS_POR_SYMLINK = frozenset({"hyde_sandbox", "jacobs"})


def _rastreado_en_un_checkout_limpio(ruta: Path) -> bool:
    """True si `ruta` NO está en `.gitignore` -- o sea, si aparece en un
    checkout limpio de este repo (clonado, no en el árbol de trabajo actual
    con basura de correr tests o el intérprete)."""
    resultado = subprocess.run(
        ["git", "check-ignore", "-q", str(ruta)],
        cwd=RAIZ, capture_output=True,
    )
    # check-ignore sale 0 si ESTÁ ignorado -- acá se quiere lo contrario.
    return resultado.returncode != 0


def _nombres_de_primer_nivel(directorio: Path) -> dict[str, Path]:
    """Nombre de módulo/paquete (sin `.py`) -> ruta, para cada entrada de
    primer nivel que un `import <nombre>` desde ese directorio podría
    resolver: un archivo `.py` o un directorio-paquete. Excluye lo oculto
    (`.git`, `.github`, ...), lo privado por convención (`_algo.py`, y de
    paso `__pycache__`/`__init__.py`) y lo que no llega a un checkout
    limpio."""
    nombres: dict[str, Path] = {}
    for entrada in sorted(directorio.iterdir()):
        if entrada.name.startswith((".", "_")):
            continue
        if not _rastreado_en_un_checkout_limpio(entrada):
            continue
        if entrada.is_dir():
            nombres[entrada.name] = entrada
        elif entrada.suffix == ".py":
            nombres[entrada.stem] = entrada
    return nombres


def test_las_manos_no_shadowea_ningun_paquete_de_la_raiz() -> None:
    de_la_raiz = _nombres_de_primer_nivel(RAIZ)
    de_la_raiz.pop("las_manos", None)  # el contenedor, no un shadow de sí mismo
    de_las_manos = _nombres_de_primer_nivel(LAS_MANOS)

    # Sanity del propio test: si esto viene vacío, `git check-ignore` no
    # está funcionando como se espera (por ejemplo, corriendo fuera de un
    # árbol de trabajo git) y el test daría un falso verde por no comparar
    # nada -- mejor fallar acá, alto y claro, que en silencio.
    assert len(de_la_raiz) >= 10, (
        f"se esperaban al menos 10 nombres de primer nivel en la raíz del "
        f"repo y se encontraron {len(de_la_raiz)} ({sorted(de_la_raiz)}) -- "
        f"revisar si `git check-ignore` corrió bien contra {RAIZ}"
    )
    assert len(de_las_manos) >= 10, (
        f"se esperaban al menos 10 nombres de primer nivel en las_manos/ y "
        f"se encontraron {len(de_las_manos)} ({sorted(de_las_manos)}) -- "
        f"revisar si `git check-ignore` corrió bien contra {LAS_MANOS}"
    )

    colisiones = sorted(set(de_la_raiz) & set(de_las_manos))
    shadows_reales = []
    for nombre in colisiones:
        ruta_raiz = de_la_raiz[nombre].resolve()
        ruta_las_manos = de_las_manos[nombre].resolve()
        es_el_mismo_archivo = ruta_raiz == ruta_las_manos
        if nombre in PERMITIDOS_POR_SYMLINK and es_el_mismo_archivo:
            continue
        shadows_reales.append((nombre, ruta_raiz, ruta_las_manos, es_el_mismo_archivo))

    assert not shadows_reales, (
        "las_manos/ tiene un módulo/paquete de primer nivel con el MISMO "
        "nombre que algo de la raíz del repo -- exactamente el bug que "
        "tumbó el arranque real de LAS MANOS en producción (jax#260, "
        "las_manos/policy.py vs policy/ de la raíz, rollback a e09c3b3): "
        "uvicorn arranca con cwd=las_manos/, y eso deja el cwd en "
        "sys.path[0] por delante de CUALQUIER PYTHONPATH -- cualquier "
        "import de primer nivel que coincida con este nombre va a "
        "resolver el archivo de las_manos/, nunca el de la raíz. "
        "Arreglo: renombrar el archivo/paquete de las_manos/ a algo sin "
        "ambigüedad (ver las_manos/motor_de_politica.py para el "
        "precedente) -- o, si de verdad tiene que ser el MISMO archivo, "
        "symlinkearlo de verdad y agregar el nombre a "
        "PERMITIDOS_POR_SYMLINK acá arriba. Colisiones encontradas "
        "(nombre, ruta en la raíz, ruta en las_manos/, ¿mismo archivo?): "
        f"{[(n, str(r), str(l), m) for n, r, l, m in shadows_reales]}"
    )
