"""Aritmética de la medición de la semilla de capability.min_output_tokens
(spec 2026-09-17 §4.4). La lectura de producción no se testea acá: la hace el
script con GO, en solo lectura. La medición YA SE HIZO UNA VEZ (plan P,
jax-platform, Task 2, commit d79b0f9): este archivo no la repite (Ruling R2
del ledger de este plan).

Fix round 1 (revisión, 2026-09-17): P's committed method
(`git -C jax-platform-prevuelo show d79b0f9`, brief y salida de Task 2) es la
referencia. Tres hallazgos: (1) el JOIN por ventana de tiempo sin la
exclusión de filas ambiguas de P atribuye tokens de un paso a la capability
de OTRO paso solapado; y su ventana llevaba holgura de más (resta del lado
izquierdo que P no tiene); (2) la prueba de sólo lectura era "fail-open":
sólo miraba `.execute(...)` y sólo marcaba lo que SÍ podía resolver
estáticamente, así que un `f"UPDATE ..."` o un `.executemany("DELETE ...",
...)` inyectados quedaban sin marcar, y el script no abría una transacción de
sólo lectura de verdad; (3) el parseo de motor_jobs.jsonl contaba TODAS las
líneas `completed`, no sólo el último registro por job_id (un job cuyo
estado FINAL no es `completed` no debe contar aunque haya pasado por
`completed` antes).

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import ast
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from medir_min_output_tokens import combinar, maximos_de_jobs, redondear  # noqa: E402

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "medir_min_output_tokens.py"


def test_redondea_hacia_arriba_al_multiplo_de_1024():
    assert [redondear(n) for n in (1, 1024, 1025, 7997)] == [1024, 1024, 2048, 8192]


def test_cero_o_negativo_queda_en_cero():
    assert redondear(0) == 0 and redondear(-5) == 0


def test_maximos_de_jobs_solo_completed_con_completion_tokens():
    lineas = [
        json.dumps({"status": "completed", "capability": "generate", "_usage": {"completion_tokens": 3000}}),
        json.dumps({"status": "completed", "capability": "generate", "_usage": {"completion_tokens": 5000}}),
        json.dumps({"status": "failed", "capability": "generate", "_usage": {"completion_tokens": 8000}}),
        json.dumps({"status": "completed", "capability": "design", "_usage": {}}),
        "",
    ]
    assert maximos_de_jobs(lineas) == ({"generate": 5000}, 0)


def test_una_linea_rota_se_cuenta_no_se_esconde():
    lineas = ["{roto", json.dumps({"status": "completed", "capability": "reason", "_usage": {"completion_tokens": 10}})]
    assert maximos_de_jobs(lineas) == ({"reason": 10}, 1)


def test_combinar_toma_el_maximo_por_capability():
    assert combinar({"a": 10, "b": 5}, {"a": 7, "c": 1}) == {"a": 10, "b": 5, "c": 1}


def test_el_join_http_directo_lleva_la_collate_del_esquema_real():
    """Plan P (jax-platform Task 2, commit d79b0f9) midió contra producción y
    tuvo que agregar `COLLATE utf8mb4_uca1400_ai_ci` al JOIN por facet: error
    1267 "Illegal mix of collations" entre axioma_usage.facet
    (utf8mb4_uca1400_ai_ci) y jacobs_steps.facet (utf8mb4_unicode_ci). Sin
    esto, este script fallaría contra el esquema real la primera vez que se
    corriera con GO."""
    from medir_min_output_tokens import _SQL_HTTP_DIRECTO

    assert "COLLATE utf8mb4_uca1400_ai_ci" in _SQL_HTTP_DIRECTO
    assert "u.facet = s.facet COLLATE utf8mb4_uca1400_ai_ci" in _SQL_HTTP_DIRECTO.replace("\n", " ")


def test_la_ventana_del_join_es_la_de_p_sin_holgura_del_lado_izquierdo():
    """P (`SQL_PASOS_HTTP`, task-2-brief.md) usa
    `UNIX_TIMESTAMP(u.created_at) BETWEEN FLOOR(s.started_at) AND
    CEIL(s.finished_at) + 5` -- FLOOR/CEIL para no perder una fila por el
    truncamiento de UNIX_TIMESTAMP (entero) contra started_at/finished_at
    (DOUBLE), y holgura SOLO del lado derecho. La versión anterior de este
    script (commit a215959) restaba holgura también del lado izquierdo
    (`s.started_at - %s`), una ventana MÁS ancha que la de P -- no es una
    adaptación de esquema, es una divergencia de comportamiento que agranda
    la probabilidad de fila ambigua."""
    from medir_min_output_tokens import _SQL_HTTP_DIRECTO

    plano = " ".join(_SQL_HTTP_DIRECTO.split())
    assert "FLOOR(s.started_at)" in plano
    assert "CEIL(s.finished_at) + %s" in plano
    assert "s.started_at - " not in plano, "la ventana no debe restar holgura del lado izquierdo (diverge de P)"


def test_fila_ambigua_no_se_atribuye_a_ninguna_capability():
    """Ejemplo de la revisión: un paso `reconcile` de la faceta `thot`
    termina en t=100 con una fila de axioma_usage de 20664 tokens; un paso
    `file_write` de la MISMA faceta arranca en t=103. La ventana de los dos
    pasos puede calzar con la MISMA fila (mismo `u.id`). Sin excluirla,
    MAX()+GROUP BY se la atribuye a las dos -- acá, a file_write, que no la
    generó (P mide file_write en 1301/2048, no en 20664/21504). Con la
    exclusión de P, la fila ambigua queda fuera y no infla a ninguna."""
    from medir_min_output_tokens import maximos_http

    filas = [(1, "reconcile", 20664), (1, "file_write", 20664)]
    maximos, ambiguas = maximos_http(filas)
    assert maximos == {}
    assert ambiguas == [1]


def test_maximos_http_toma_el_mayor_sin_ambiguas_y_descarta_cero():
    from medir_min_output_tokens import maximos_http

    filas = [
        (1, "generate", 3000),
        (2, "generate", 5000),
        (3, "generate", 0),  # nada que medir: se descarta, como en P (`if tokens_por_fila[fila_id]`)
        (4, "design", 1200),
    ]
    maximos, ambiguas = maximos_http(filas)
    assert maximos == {"generate": 5000, "design": 1200}
    assert ambiguas == []


def test_maximos_http_sin_filas_no_revienta():
    from medir_min_output_tokens import maximos_http

    assert maximos_http([]) == ({}, [])


def test_conserva_solo_el_ultimo_registro_por_job_id():
    """Mismo criterio que P (`corridas_de_motor`, task-2-brief.md) y
    jacobs/reaper.py:228 (`_load_terminal_motor_jobs`): una línea `completed`
    de un job cuyo estado FINAL es otro no cuenta. La versión anterior
    (commit a215959) contaba CADA línea `completed` sin mirar si el job
    terminó en otro estado."""
    lineas = [
        json.dumps({"job_id": "j1", "status": "completed", "capability": "generate", "_usage": {"completion_tokens": 9000}}),
        json.dumps({"job_id": "j1", "status": "failed", "capability": "generate", "_usage": {"completion_tokens": 9000}}),
    ]
    assert maximos_de_jobs(lineas) == ({}, 0)


def test_el_ultimo_registro_completed_gana_aunque_sea_menor():
    """Distingue "deduplicar por job_id" de "tomar el máximo de todas las
    líneas": el último registro de un mismo job_id gana aunque su
    completion_tokens sea MENOR que uno anterior del mismo job."""
    lineas = [
        json.dumps({"job_id": "j2", "status": "completed", "capability": "reason", "_usage": {"completion_tokens": 9000}}),
        json.dumps({"job_id": "j2", "status": "completed", "capability": "reason", "_usage": {"completion_tokens": 500}}),
    ]
    assert maximos_de_jobs(lineas) == ({"reason": 500}, 0)


def test_lineas_sin_job_id_se_cuentan_por_si_mismas():
    """Sin job_id no hay nada contra qué deduplicar: cada línea cuenta por sí
    misma (control -- así se comportaba ya el script y así lo siguen
    ejerciendo los tests del brief, que no tienen job_id)."""
    lineas = [
        json.dumps({"status": "completed", "capability": "generate", "_usage": {"completion_tokens": 100}}),
        json.dumps({"status": "completed", "capability": "generate", "_usage": {"completion_tokens": 200}}),
    ]
    assert maximos_de_jobs(lineas) == ({"generate": 200}, 0)


# --- Prueba de sólo lectura, fail-closed --------------------------------

_PERMITIDAS_EXACTAS = {
    "SET SESSION TRANSACTION READ ONLY",
    "START TRANSACTION READ ONLY",
    "ROLLBACK",
}


def _constantes_de_modulo(arbol: ast.AST) -> dict[str, str]:
    return {
        nodo.targets[0].id: nodo.value.value
        for nodo in ast.walk(arbol)
        if isinstance(nodo, ast.Assign)
        and len(nodo.targets) == 1
        and isinstance(nodo.targets[0], ast.Name)
        and isinstance(nodo.value, ast.Constant)
        and isinstance(nodo.value.value, str)
    }


def _resolver_str(nodo: ast.AST, constantes: dict[str, str]) -> str | None:
    """Devuelve el string que arma `nodo` SI Y SÓLO SI se puede resolver
    ESTÁTICAMENTE (literal, o Name que apunta a una constante de módulo).
    Cualquier otra forma (f-string, concatenación con variable, llamada,
    atributo) devuelve None -- y quien llama a esto trata None como
    HALLAZGO, no como permiso (fail-closed)."""
    if isinstance(nodo, ast.Constant) and isinstance(nodo.value, str):
        return nodo.value
    if isinstance(nodo, ast.Name) and nodo.id in constantes:
        return constantes[nodo.id]
    return None


def _llamadas_db(arbol: ast.AST) -> list[ast.Call]:
    """Toda llamada `<algo>.execute(...)`, `.executemany(...)`,
    `.callproc(...)` o `.commit()` -- las formas DB-API de mandar SQL o de
    confirmar una transacción."""
    return [
        nodo
        for nodo in ast.walk(arbol)
        if isinstance(nodo, ast.Call)
        and isinstance(nodo.func, ast.Attribute)
        and nodo.func.attr in {"execute", "executemany", "callproc", "commit"}
    ]


def hallazgos_de_solo_lectura(ruta: Path) -> list[str]:
    """FAIL-CLOSED (ronda de arreglo 1, revisión): la versión anterior de
    este detector sólo miraba `.execute(...)` y sólo marcaba lo que SÍ podía
    resolver -- verificado que un `f"UPDATE capability SET ..."` pasado a
    `.execute(...)` y un `.executemany("DELETE FROM jacobs_steps", filas)`
    inyectados en una copia de prueba quedaban SIN marcar (task-13-report.md,
    Fix round 1). Ahora: cada llamada real tiene que resolver a un SELECT o a
    una de las tres sentencias de transacción de sólo lectura permitidas;
    cualquier argumento no resoluble ESTÁTICAMENTE es un hallazgo (no un
    permiso); y `.commit()` SIEMPRE es un hallazgo -- una sesión de sólo
    lectura no tiene nada que confirmar."""
    arbol = ast.parse(ruta.read_text(encoding="utf-8"))
    constantes = _constantes_de_modulo(arbol)
    hallazgos: list[str] = []
    for nodo in _llamadas_db(arbol):
        if nodo.func.attr == "commit":
            hallazgos.append(f"línea {nodo.lineno}: .commit() -- una sesión de sólo lectura no confirma nada")
            continue
        if not nodo.args:
            hallazgos.append(f"línea {nodo.lineno}: {nodo.func.attr}() sin argumentos")
            continue
        resuelto = _resolver_str(nodo.args[0], constantes)
        if resuelto is None:
            hallazgos.append(f"línea {nodo.lineno}: {nodo.func.attr}(...) con SQL NO resoluble estáticamente")
            continue
        cuerpo = resuelto.strip()
        if cuerpo.upper() in _PERMITIDAS_EXACTAS or cuerpo.upper().startswith("SELECT"):
            continue
        hallazgos.append(
            f"línea {nodo.lineno}: {nodo.func.attr}(...) no es SELECT ni transacción de sólo lectura permitida: {cuerpo[:60]!r}"
        )
    return hallazgos


def test_el_script_es_de_solo_lectura_fail_closed():
    hallazgos = hallazgos_de_solo_lectura(_SCRIPT)
    assert hallazgos == [], "\n".join(hallazgos)
    arbol = ast.parse(_SCRIPT.read_text(encoding="utf-8"))
    assert _llamadas_db(arbol), "no se encontró ninguna llamada db-api -- el detector no vigila nada"


def test_los_update_que_imprime_para_la_migracion_no_se_ejecutan():
    fuente = _SCRIPT.read_text(encoding="utf-8")
    assert "UPDATE capability SET min_output_tokens" in fuente, (
        "el script tiene que imprimir los UPDATE que consume la migración del plan P"
    )
    assert hallazgos_de_solo_lectura(_SCRIPT) == []


# --- Prueba de que el CÓDIGO abre la sesión de sólo lectura -------------


class _CursorFalso:
    def __init__(self) -> None:
        self.ejecutadas: list[object] = []

    async def execute(self, sql, params=None):
        self.ejecutadas.append(sql)

    async def fetchall(self):
        return []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _ConexionFalsa:
    def __init__(self) -> None:
        self.cursor_falso = _CursorFalso()
        self.cerrada = False

    def cursor(self):
        return self.cursor_falso

    def close(self):
        self.cerrada = True


def test_la_conexion_abre_sesion_de_solo_lectura_y_hace_rollback_no_commit(monkeypatch):
    """Mockea jacobs.store.get_conn (sin tocar producción ni exigir DB): el
    código de `_maximos_http_directo` tiene que abrir `SET SESSION
    TRANSACTION READ ONLY` + `START TRANSACTION READ ONLY` ANTES de la
    consulta, cerrar con `ROLLBACK` (nunca commit) y cerrar la conexión --
    igual que P. Que el SERVIDOR real rechace una escritura dentro de esa
    sesión lo garantiza MariaDB, no este test; este test verifica que el
    CÓDIGO la abre."""
    import jacobs.store as store
    from medir_min_output_tokens import _maximos_http_directo

    conexion = _ConexionFalsa()

    async def _get_conn_falso(*a, **kw):
        return conexion

    monkeypatch.setattr(store, "get_conn", _get_conn_falso)

    asyncio.run(_maximos_http_directo())

    ejecutadas = conexion.cursor_falso.ejecutadas
    normalizadas = [s.strip().upper() if isinstance(s, str) else s for s in ejecutadas]
    assert normalizadas[0] == "SET SESSION TRANSACTION READ ONLY"
    assert normalizadas[1] == "START TRANSACTION READ ONLY"
    assert normalizadas[-1] == "ROLLBACK"
    assert not any(
        isinstance(s, str) and s.strip().upper().startswith(("INSERT", "UPDATE", "DELETE", "COMMIT"))
        for s in normalizadas
    )
    assert conexion.cerrada
