#!/usr/bin/env python3
"""Todo archivo de test del repo corre en algún job de CI, o está declarado
como excepción con motivo escrito -- nunca por olvido.

**El problema, con datos** (2026-09-18, historial-y-arreglos-de-pipeline).
`.github/workflows/policy.yml` lista sus archivos de test UNO POR UNO en
cada job: un archivo nuevo no corre solo, hay que acordarse de agregarlo.
Nadie se acuerda. Medido el mismo día: de 6 archivos de test creados en esta
ronda, 5 no corrían en ningún job. Peor: `las_manos/_tool_authority_test.py`
-- 26 casos que ejercitan el jail de rutas del Ejecutor, el freno que impide
que un modelo lea `/etc/passwd` -- llevaba MESES sin correr en CI (arreglado
el mismo día por otra tarea de esta ronda, commit 950f8c8, Task 5). Ver
también el job `tests-puros`, más abajo en policy.yml: un audit manual de
2026-09-11 encontró "18 de 26 sin job" -- el mismo hallazgo, hecho a mano,
que se vuelve a romper con el próximo archivo nuevo.

Esta prueba reemplaza el audit manual por un DETECTOR mecánico: enumera cada
archivo de test del árbol (no sólo tests/, jacobs/_*_test.py y
las_manos/_*_test.py -- también policy/tests/, scripts/, la raíz del repo y
cualquier subcarpeta como las_manos/motor_registry/, con el mismo criterio
de nombre) y falla si no aparece invocado por pytest/unittest en ningún
`run:` de ningún job de policy.yml, salvo que esté declarado en EXCEPCIONES
con un motivo escrito.

Corriendo este detector contra el árbol real HOY (antes de este commit)
encontró 21 archivos reales sin cubrir -- 18 quedaron wireados en el mismo
commit que agrega este test, cada uno en el job que le corresponde según si
necesita DB, y 3 quedaron en EXCEPCIONES con motivo escrito:
  - 9 puros (sin I/O, sin DB) -> tests-puros: las_manos/motor_registry/_policy_test.py,
    jacobs/_arbitro_test.py, jacobs/_aviso_executor_test.py, jacobs/_aviso_test.py,
    jacobs/_encadenado_por_defecto_test.py, jacobs/_truncado_falla_test.py,
    jacobs/_usage_reconciliation_test.py, las_manos/_motor_job_model_test.py,
    tests/test_ejecutor_reproduccion_u3.py
  - 2 que usan jacobs.store real (init_tables() propio, sin gobernanza de
    jax-platform) -> subpipeline-contrato-db: jacobs/_pipeline_identity_test.py,
    jacobs/_step_motor_test.py
  - 7 que necesitan la DB de gobernanza (facet/capability/model de
    jax-platform) o `server.py` real -> jacobs-gobernanza-db:
    las_manos/motor_registry/_facet_policy_test.py, jacobs/_http_facet_admission_test.py,
    las_manos/motor_registry/_authorize_facet_endpoint_test.py, jacobs/_usage_writer_test.py,
    las_manos/_catalog_from_db_test.py, jacobs/_modelo_real_test.py,
    las_manos/_output_validator_db_drift_test.py
  - este mismo archivo -> el job nuevo `archivos-de-test-en-ci`, más abajo
    en policy.yml.
Los DB-dependientes, y el paso nuevo que instala requirements.txt de ESTE
repo en jacobs-gobernanza-db (server.py, que
_authorize_facet_endpoint_test.py importa, necesita más que las dependencias
de jax-platform), se verificaron corriendo de verdad contra una MariaDB 11.8
efímera en un contenedor Docker aislado (no la de producción de hall9000)
con las migraciones reales de jax-platform clonadas en frío -- la misma
receta que usa cada job, reproducida a mano. El propio docstring de
`_authorize_facet_endpoint_test.py` cuenta la historia: "Nunca se vio el
fallo porque el archivo no está en ningún job de CI" -- y sin embargo su
ruta SÍ aparecía citada en comentarios de otros jobs, explicando por qué.
Ese es justo el caso que este detector tiene que distinguir: una mención en
un comentario de shell no es una ejecución.

Dos de los siete archivos de jacobs-gobernanza-db (jacobs/_usage_writer_test.py
completo, y 2 de los 4 casos de las_manos/_catalog_from_db_test.py) están
ROJOS hoy contra jax-platform@master -- no por un problema de wireo, sino
por la MISMA dependencia de orden de despliegue que ese job ya declaraba
ANTES de este commit (ver su "Piso exacto de tests CORRIDOS": Task 7b de
esta misma ronda, en jax-platform, todavía no mergeada). Wireados igual,
con el motivo escrito en el piso del job, siguiendo el mismo criterio que
ese job ya usa para sus propios tests pendientes de una migración ajena.

Los otros tres (tests/test_audit_traffic_class.py, tests/test_envelope_brutal.py,
tests/test_thot_connection.py) NO son un problema de wireo: pegan contra
rutas de server.app (/plan, /audit/tail) que el middleware deny-by-default
de las_manos/auth_servicio.py (commit cbddb38, 2026-09-17) deja sin ningún
caller real a propósito. Ver EXCEPCIONES más abajo para el detalle de cada
uno.

Enforcement mecánico, no una lista: se lee policy.yml con PyYAML, se juntan
todos los `run:` de todos los jobs, y las líneas que son comentarios de
shell (empiezan con `#` tras quitar espacios) se descartan ANTES de buscar
-- si no, cualquier comentario que mencione un archivo lo marcaría como
"cubierto" sin que ningún pytest lo ejecute nunca. Ver
test_un_comentario_de_shell_que_cita_un_archivo_no_cuenta_como_wireado.

Se niega a dar verde si no encuentra nada que mirar (Principio I -- el que
supone se equivoca): si el escaneo del repo ve cero archivos de test bajo
tests/, jacobs/ o las_manos/, o si la lectura de policy.yml no encuentra
ningún job con un `run:` que mencione pytest/unittest, el test FALLA -- no
pasa por vacío. Ver test_el_escaneo_ve_archivos_conocidos y
test_la_lectura_de_jobs_ve_comandos_de_pytest.

Auto-verificado (Principio VII -- un freno sin prueba no es freno): hay un
test que arma un archivo de test sintético SIN referenciar en un policy.yml
sintético, en un directorio temporal, y comprueba que el detector lo marca
ROJO; lo retira y comprueba que vuelve a VERDE. Sin ese test no se sabe si
el detector detecta o si sólo dice que sí. Ver
test_un_archivo_sin_referenciar_se_detecta.

Corre con:
  python3 -m pytest policy/tests/test_archivos_de_test_wireados_en_ci.py -v

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import yaml

_THIS_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOW_PATH = _THIS_REPO_ROOT / ".github" / "workflows" / "policy.yml"

# Mismo criterio que policy/tests/test_no_fail_open_except.py, más las
# carpetas propias de este repo que no son código de producción ni de test.
EXCLUDE_DIR_NAMES = {
    ".venv", "venv", "node_modules", ".git", ".worktrees", "worktrees",
    "__pycache__", "dist", "build", ".superpowers", ".claude", ".claude-flow",
    ".pytest_cache",
}

# Lo que define "archivo de test" en este repo: tests/ usa test_*.py;
# jacobs/ y las_manos/ (con motor_registry/ debajo) usan _*_test.py;
# policy/tests/ y scripts/ mezclan los dos patrones. Un archivo que no
# matchea ninguno de los dos NO es un test para este control aunque tenga
# "test" en el nombre en otro lugar -- scripts/load_test.py (arnés de carga
# de LAS CUATRO DEL RENDIMIENTO) y base_de_test.py (el módulo fixture de la
# base de sesión) son ejemplos reales del árbol que quedan afuera a
# propósito: ninguno matchea `test_*.py` ni `_*_test.py`.
_PATTERNS = ("test_*.py", "_*_test.py")

# Piso: medido 2026-09-18, 209 archivos de test en el árbol. El piso va por
# debajo a propósito (mismo criterio que test_no_fail_open_except.py):
# crecer no rompe nada, perder un árbol entero sí.
PISO_ARCHIVOS_TEST = 200

# Piso: medido 2026-09-18, 17 de 17 jobs de policy.yml corren pytest o
# unittest (el job nuevo que agrega este archivo será el 18vo). Por debajo
# a propósito, mismo criterio.
PISO_JOBS_CON_PYTEST = 15

# Excepciones EXPLÍCITAS, con motivo escrito -- nunca por olvido. Un archivo
# que aparece acá Y TAMBIÉN corre en un job no es un error (sería
# redundante, nada más); lo que este control prohíbe es que falte de LOS
# DOS lados a la vez. Hay un test que comprueba que cada excepción sigue
# existiendo y tiene motivo (test_las_excepciones_siguen_vigentes).
EXCEPCIONES: dict[str, str] = {
    "policy/tests/test_no_deploy_from_unmerged_branch.py": (
        "requiere axioma-ia.io EN VIVO, un checkout local de jax-platform "
        "en el host (/home/fruiz/jax-platform) y un npm install/build "
        "real; el propio docstring del archivo lo dice: 'no corre en un "
        "entorno sin ese checkout'. Se corre a mano desde hall9000 después "
        "de cada deploy, no en un runner efímero de GitHub Actions."
    ),
    # ACTUALIZADO 2026-09-20: los tres de abajo ya NO fallan cuando alguien
    # corre la suite a mano. Se SALTAN solos, sondeando el candado con
    # `tests/_alcance_las_manos.py`, y el motivo sale impreso en el propio
    # skip en vez de vivir solo aca. Siguen fuera de CI (necesitan levantar
    # `server.app`, que es mas de lo que estos jobs traen), pero el dia que
    # `auth_servicio` le abra permiso a una identidad real vuelven a correr
    # SOLOS donde se los corra, sin que nadie tenga que acordarse de nada.
    #
    # Los siguientes tres (tests/test_audit_traffic_class.py,
    # tests/test_envelope_brutal.py, tests/test_thot_connection.py) hablan
    # directo con server.app por /plan y /audit/tail. Verificado corriéndolos
    # de verdad (2026-09-18): las_manos/auth_servicio.py (commit cbddb38,
    # 2026-09-17, "LAS MANOS autentica a sus llamadores") es un middleware
    # ASGI deny-by-default, y su propio mensaje de commit dice, textual:
    # "/execute, /plan y /audit/tail sin identidad que los alcance (sin
    # llamadores reales)". No es un bug de estos tres archivos ni un problema
    # de wireo -- es una decisión de seguridad explícita y escrita de
    # Fernando: esas rutas quedaron sin ningún caller real que las autorice
    # todavía. Un test que las alcanza HOY pega 401/403 contra ese candado a
    # propósito. Correrán de nuevo el día que alguna identidad real necesite
    # esas rutas (y auth_servicio.py le abra un permiso).
    "tests/test_audit_traffic_class.py": (
        "pega /audit/tail vía FacetClient.for_thot(); auth_servicio.py "
        "(commit cbddb38, 2026-09-17) deja esa ruta sin ninguna identidad "
        "que la alcance a propósito ('sin llamadores reales', mensaje del "
        "commit). 401 hoy no es una falla del test."
    ),
    "tests/test_envelope_brutal.py": (
        "pega POST /plan con TestClient(server.app); auth_servicio.py "
        "(commit cbddb38, 2026-09-17) deja /plan sin ninguna identidad que "
        "la alcance a propósito ('sin llamadores reales', mensaje del "
        "commit). 401 hoy no es una falla del test."
    ),
    "tests/test_thot_connection.py": (
        "dos motivos reales, cualquiera de los dos ya lo excluye: (1) exige "
        "JAX_ENV_STAGING_HOSTS sourceada de /etc/jax/.env -- un host de "
        "staging real, no algo que se inventa en un runner efímero; (2) "
        "aunque se le diera ese host, pega /audit/tail y /plan contra "
        "server.app, las mismas rutas sin identidad que las alcance por "
        "auth_servicio.py (commit cbddb38, 2026-09-17) que excluyen a "
        "tests/test_audit_traffic_class.py y tests/test_envelope_brutal.py."
    ),
}


def _iter_test_files(repo_root: Path) -> list[Path]:
    vistos: set[Path] = set()
    out: list[Path] = []
    for patron in _PATTERNS:
        for path in repo_root.rglob(patron):
            if not path.is_file():
                continue
            rel_parts = path.relative_to(repo_root).parts
            if any(part in EXCLUDE_DIR_NAMES for part in rel_parts):
                continue
            if path in vistos:
                continue
            vistos.add(path)
            out.append(path)
    return sorted(out)


def _run_text_de_todos_los_jobs(workflow_path: Path) -> str:
    """Todo el texto de todos los `run:` de todos los jobs, con las líneas
    que son comentarios de shell descartadas -- policy.yml usa `run: |` con
    comentarios en prosa para explicar cambios de piso, y esas líneas citan
    nombres de archivo sin ejecutarlos."""
    with workflow_path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    jobs = (data or {}).get("jobs") or {}
    piezas: list[str] = []
    for job in jobs.values():
        for step in job.get("steps") or []:
            run = step.get("run")
            if not run:
                continue
            for line in run.splitlines():
                if line.strip().startswith("#"):
                    continue
                piezas.append(line)
    return "\n".join(piezas)


def _run_text_por_job(workflow_path: Path) -> dict[str, str]:
    with workflow_path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    jobs = (data or {}).get("jobs") or {}
    out = {}
    for name, job in jobs.items():
        out[name] = "\n".join(
            step.get("run", "") for step in (job.get("steps") or []) if step.get("run")
        )
    return out


def _referenciado(rel_posix: str, run_text: str) -> bool:
    if rel_posix in run_text:
        return True
    # Forma de módulo con puntos: `python -m unittest paquete.submodulo`.
    # Ningún job de hoy invoca así, pero un job futuro podría -- sin esto,
    # ese caso legítimo se marcaría como sin cubrir.
    if rel_posix.endswith(".py"):
        dotted = rel_posix[:-3].replace("/", ".")
        if dotted and dotted in run_text:
            return True
    return False


def archivos_sin_referenciar(
    workflow_path: Path = _WORKFLOW_PATH,
    repo_root: Path = _THIS_REPO_ROOT,
    excepciones: dict[str, str] | None = None,
) -> list[str]:
    excepciones = EXCEPCIONES if excepciones is None else excepciones
    texto_total = _run_text_de_todos_los_jobs(workflow_path)
    faltantes = []
    for path in _iter_test_files(repo_root):
        rel = path.relative_to(repo_root).as_posix()
        if rel in excepciones:
            continue
        if not _referenciado(rel, texto_total):
            faltantes.append(rel)
    return sorted(faltantes)


# --- guardas del propio control (Principio I: no dar verde por vacío) -----

def test_el_escaneo_ve_archivos_conocidos():
    vistos = {p.relative_to(_THIS_REPO_ROOT).as_posix() for p in _iter_test_files(_THIS_REPO_ROOT)}
    assert vistos, "el escaneo de archivos de test no vio NINGUN archivo"
    for prefijo in ("tests/", "jacobs/", "las_manos/"):
        assert any(v.startswith(prefijo) for v in vistos), (
            f"el escaneo no ve ningun archivo de test bajo {prefijo!r}: "
            f"¿cambio la convencion de nombres o la estructura de carpetas?")
    assert len(vistos) >= PISO_ARCHIVOS_TEST, (
        f"el escaneo ve {len(vistos)} archivos de test (< {PISO_ARCHIVOS_TEST}): "
        f"¿se excluyo un arbol por error?")


def test_la_lectura_de_jobs_ve_comandos_de_pytest():
    """Si policy.yml cambia de forma (jobs sin `run:`, YAML no parseable) y
    la lectura deja de ver comandos, el control tiene que gritar -- no
    aprobar porque no encontro nada que objetar."""
    runs_por_job = _run_text_por_job(_WORKFLOW_PATH)
    assert runs_por_job, "no se encontro NINGUN job en policy.yml"
    con_pytest = [n for n, txt in runs_por_job.items() if "pytest" in txt or "unittest" in txt]
    assert con_pytest, (
        "ningun job de policy.yml tiene un `run:` que mencione pytest/unittest -- "
        "¿cambio la forma del workflow?")
    assert len(con_pytest) >= PISO_JOBS_CON_PYTEST, (
        f"solo {len(con_pytest)} jobs corren pytest/unittest (< {PISO_JOBS_CON_PYTEST})")


def test_las_excepciones_siguen_vigentes():
    """Una excepcion para un archivo que ya no existe es ruido: no protege
    nada y esconde que el motivo escrito perdio vigencia. Una excepcion sin
    motivo es la puerta de atras que este control existe para cerrar."""
    for rel, motivo in EXCEPCIONES.items():
        assert motivo and motivo.strip(), f"excepcion sin motivo escrito: {rel}"
        ruta = _THIS_REPO_ROOT / rel
        assert ruta.is_file(), f"la excepcion {rel!r} ya no existe en el repo: sobra, hay que quitarla"


# --- auto-verificacion (Principio VII: un freno sin prueba no es freno) ---

def test_un_archivo_sin_referenciar_se_detecta():
    """Un archivo de test de mentira, sin job, tiene que APARECER en
    archivos_sin_referenciar(); uno referenciado no. Sin este test no se
    sabe si el detector detecta o si solo dice que si."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        (tmp_root / "tests").mkdir()
        sin_cubrir = tmp_root / "tests" / "test_archivo_de_mentira_sin_ci.py"
        sin_cubrir.write_text("def test_algo():\n    assert True\n")
        cubierto = tmp_root / "tests" / "test_archivo_de_mentira_cubierto.py"
        cubierto.write_text("def test_algo():\n    assert True\n")

        workflow = tmp_root / "policy.yml"
        workflow.write_text(
            "jobs:\n"
            "  ejemplo:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: pip install pytest\n"
            "      - run: python -m pytest tests/test_archivo_de_mentira_cubierto.py -v\n"
        )

        # ROJO: el archivo sin job aparece.
        faltantes = archivos_sin_referenciar(workflow_path=workflow, repo_root=tmp_root, excepciones={})
        assert "tests/test_archivo_de_mentira_sin_ci.py" in faltantes, (
            "el detector NO marco un archivo de test real, sin job, como faltante -- "
            "esta roto: dice que si sin haber mirado.")
        assert "tests/test_archivo_de_mentira_cubierto.py" not in faltantes

        # VERDE: se retira el archivo sin job y el hallazgo desaparece.
        sin_cubrir.unlink()
        faltantes_tras_retirar = archivos_sin_referenciar(
            workflow_path=workflow, repo_root=tmp_root, excepciones={})
        assert "tests/test_archivo_de_mentira_sin_ci.py" not in faltantes_tras_retirar, (
            "el detector sigue marcando un archivo que ya no existe.")


def test_un_comentario_de_shell_que_cita_un_archivo_no_cuenta_como_wireado():
    """Hallazgo real de esta tarea: los `run: |` de policy.yml tienen
    comentarios de shell que citan archivos al explicar cambios de piso.
    Si el detector buscara sobre el `run:` crudo sin quitar comentarios,
    esa mencion en prosa bastaria para marcar el archivo como cubierto
    aunque ningun pytest lo ejecute -- exactamente lo que paso de verdad
    con las_manos/motor_registry/_authorize_facet_endpoint_test.py."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        (tmp_root / "tests").mkdir()
        (tmp_root / "tests" / "test_solo_en_comentario.py").write_text(
            "def test_algo():\n    assert True\n"
        )
        workflow = tmp_root / "policy.yml"
        workflow.write_text(
            "jobs:\n"
            "  ejemplo:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: pip install pytest\n"
            "      - run: |\n"
            "          # tests/test_solo_en_comentario.py se corre en otro lado\n"
            "          python -m pytest tests/test_no_es_este.py -v\n"
        )
        faltantes = archivos_sin_referenciar(workflow_path=workflow, repo_root=tmp_root, excepciones={})
        assert "tests/test_solo_en_comentario.py" in faltantes, (
            "un archivo citado SOLO en un comentario de shell se marco como cubierto: "
            "el detector esta leyendo prosa, no ejecuciones.")


def test_una_excepcion_declarada_no_cuenta_como_faltante():
    faltantes_con_excepcion = archivos_sin_referenciar()
    for rel in EXCEPCIONES:
        assert rel not in faltantes_con_excepcion, (
            f"{rel} tiene excepcion declarada pero igual aparece como faltante")


# --- el control real ------------------------------------------------------

def test_todo_archivo_de_test_corre_en_algun_job_o_tiene_excepcion():
    faltantes = archivos_sin_referenciar()
    assert not faltantes, (
        "archivos de test sin job de CI y sin excepcion declarada:\n  "
        + "\n  ".join(faltantes)
        + "\n\nAgregalos a un job de .github/workflows/policy.yml (eligiendo uno "
        "con DB o sin DB segun lo que el archivo necesite), o declara una "
        "excepcion en EXCEPCIONES (variable de este mismo archivo) con motivo "
        "escrito -- nunca por olvido."
    )


if __name__ == "__main__":
    import sys

    import pytest as _pytest

    sys.exit(_pytest.main([__file__, "-v"]))
