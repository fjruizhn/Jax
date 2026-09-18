"""Ningún test borra filas AJENAS de una tabla compartida de jax_memory_test.

Origen (R38 fix round 3, incidente declarado en r38-report.md): una limpieza
de filas de prueba filtró por columnas "que parecían mías" (modelo + tokens +
job_id NULL) y borró `axioma_usage.id=2792989`, una fila de otra sesión. La
base de tests es COMPARTIDA entre sesiones: un DELETE sólo es seguro si su
WHERE filtra por un marcador que el test creó (uuid, marca, pid propio).

LA REGLA (estática, por AST, sobre los archivos de test):
- Se mira cada llamada cuyo PRIMER argumento es SQL literal (str o f-string)
  con un DELETE, en sus tres formas de MariaDB (m3 de la re-review):
  `DELETE FROM t ...`, `DELETE LOW_PRIORITY/QUICK/IGNORE FROM t ...` y la
  multi-tabla con alias `DELETE u FROM t u JOIN ...` / `DELETE a, b FROM ...`.
  Se miran TODAS las tablas del FROM (comas y JOIN), no sólo la primera. Si
  alguna es de TABLAS_COMPARTIDAS, o no se puede saber (`DELETE FROM {nombre}`),
  el DELETE:
  1. tiene que tener WHERE, y
  2. los parámetros de la llamada (segundo argumento posicional, o `args=` /
     `params=`) tienen que nombrar un marcador propio: un nombre o atributo que
     calce con NOMBRE_DE_MARCADOR (pid, ids, marca, claves, job_id, model_id,
     id_previo, ...), o la llamada tiene que llevar el comentario
     `# marcador-propio: <por qué>` en alguna de sus líneas.
- Excepción explícita: PERMITIDOS, (archivo, función) de los DELETE globales
  de los tres tests de Ruling R50 (sólo corren con base exclusiva). Si una
  entrada de PERMITIDOS deja de existir, el test falla: la lista no envejece.

LÍMITES (lo que esta regla NO ve), declarados a propósito:
- SQL armado en una variable y ejecutado después (`sql = "DELETE ..."`),
  por concatenación entre sentencias o fuera de la llamada: sólo se mira el
  literal que es primer argumento.
- Un WHERE que filtra por una SUBCONSULTA (`WHERE id IN (SELECT ...)`): se
  exige el marcador en los parámetros, no se analiza la subconsulta.
- Borrados que no son un DELETE de SQL: TRUNCATE, DROP, `REPLACE INTO`,
  `INSERT ... ON DUPLICATE KEY UPDATE` que pisa una fila ajena (el caso de
  R46 con los precios), o un borrado por ORM/ayudante sin SQL literal.
- Confía en los NOMBRES: no prueba que `pid` contenga un uuid ni que la marca
  sea única. Un `pid = "jekyll"` pasaría.
- No mira UPDATE.
- Sólo archivos de test del repo (tests/, jacobs/_*_test.py,
  las_manos/**/_*_test.py, las_manos/**/test_*.py, scripts/_*_test.py): los
  scripts sueltos de una sesión (scratchpad) no pasan por acá -- el incidente
  ocurrió justamente en uno de esos.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]

TABLAS_COMPARTIDAS = frozenset({
    "axioma_usage", "facet_health_event", "facet_health_alert", "jacobs_pipelines",
    "jacobs_steps", "jacobs_events", "credential", "capability",
})
NOMBRE_DE_MARCADOR = re.compile(
    r"^(pid|pids|ids|pipeline_ids?|marca|marcas|marcador|marker|clave|claves|"
    r"job_id|model_id|id_previo|propi[oa]s?|uuid\w*)$"
)
PRAGMA = "# marcador-propio:"
PERMITIDOS = {
    # Ruling R50: los tres tests globales de salud de facets, sólo con
    # JAX_TEST_FACET_HEALTH_TABLA_EXCLUSIVA=1 (base exclusiva, job facet-health-io).
    ("jacobs/_facet_health_io_test.py", "_tabla_limpia"),
}
_MODIFICADORES = r"(?:LOW_PRIORITY|QUICK|IGNORE)"
# `DELETE [modificadores] [alias, ...] FROM <tablas> [WHERE ...]`. Lo que va
# entre DELETE y FROM son modificadores y/o alias de la forma multi-tabla.
_DELETE = re.compile(
    rf"\bDELETE\b(?P<entre>(?:\s+{_MODIFICADORES}|\s+[`\w.,{{}}]+)*)\s+FROM\s+(?P<resto>.*)",
    re.IGNORECASE | re.DOTALL,
)
_SEPARADOR_DE_TABLAS = re.compile(r",|\bJOIN\b|\bUSING\b", re.IGNORECASE)


def _tablas_del_from(resto: str) -> list[str]:
    """Nombres de tabla del FROM (multi-tabla con comas o JOIN), hasta el
    WHERE. Un `{placeholder}` entra como desconocido."""
    cuerpo = re.split(r"\bWHERE\b", resto, maxsplit=1, flags=re.IGNORECASE)[0]
    nombres = []
    for parte in _SEPARADOR_DE_TABLAS.split(cuerpo):
        parte = parte.strip()
        if not parte:
            continue
        primero = parte.split()[0].strip("`;")
        if primero:
            nombres.append(primero)
    return nombres


def archivos_de_test(raiz: Path = RAIZ) -> list[Path]:
    patrones = ("tests/*.py", "jacobs/_*_test.py", "las_manos/**/_*_test.py",
                "las_manos/**/test_*.py", "scripts/_*_test.py")
    vistos = {p for patron in patrones for p in raiz.glob(patron) if p.is_file()}
    # Este archivo se excluye: sus mensajes y controles contienen "DELETE FROM".
    propio = Path(__file__).resolve()
    return sorted(p for p in vistos if ".venv" not in p.parts and p.resolve() != propio)


def _sql_literal(nodo) -> str | None:
    if isinstance(nodo, ast.Constant) and isinstance(nodo.value, str):
        return nodo.value
    if isinstance(nodo, ast.JoinedStr):
        partes = []
        for v in nodo.values:
            if isinstance(v, ast.Constant):
                partes.append(str(v.value))
            else:
                partes.append("{" + ast.unparse(v.value) + "}")
        return "".join(partes)
    return None


def _nombres(nodo) -> set[str]:
    if nodo is None:
        return set()
    salida = set()
    for n in ast.walk(nodo):
        if isinstance(n, ast.Name):
            salida.add(n.id)
        elif isinstance(n, ast.Attribute):
            salida.add(n.attr)
    return salida


def violaciones(archivos: list[Path], raiz: Path = RAIZ) -> tuple[list[str], set[tuple[str, str]]]:
    """(violaciones "archivo:línea: motivo", permitidos que sí aparecieron)."""
    encontradas, permitidos_vistos = [], set()
    for ruta in archivos:
        texto = ruta.read_text(encoding="utf-8")
        lineas = texto.splitlines()
        try:
            arbol = ast.parse(texto)
        except SyntaxError:
            continue
        relativa = ruta.relative_to(raiz).as_posix() if ruta.is_relative_to(raiz) else ruta.name
        funcion_de = {}
        for fn in ast.walk(arbol):
            if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for hijo in ast.walk(fn):
                    funcion_de.setdefault(id(hijo), fn.name)
        for llamada in ast.walk(arbol):
            if not isinstance(llamada, ast.Call) or not llamada.args:
                continue
            sql = _sql_literal(llamada.args[0])
            coincide = _DELETE.search(sql) if sql else None
            if not coincide:
                continue
            resto = coincide.group("resto")
            tablas = _tablas_del_from(resto)
            comprometidas = [t for t in tablas if t.startswith("{") or t.lower() in TABLAS_COMPARTIDAS]
            if not comprometidas:
                continue
            tabla = ", ".join(comprometidas)
            funcion = funcion_de.get(id(llamada), "<módulo>")
            if (relativa, funcion) in PERMITIDOS:
                permitidos_vistos.add((relativa, funcion))
                continue
            sitio = f"{relativa}:{llamada.lineno}"
            if not re.search(r"\bWHERE\b", resto, re.IGNORECASE):
                encontradas.append(f"{sitio}: DELETE FROM {tabla} sin WHERE")
                continue
            params = llamada.args[1] if len(llamada.args) > 1 else next(
                (k.value for k in llamada.keywords if k.arg in ("args", "params")), None)
            con_marcador = any(NOMBRE_DE_MARCADOR.match(n) for n in _nombres(params))
            fin = getattr(llamada, "end_lineno", llamada.lineno)
            con_pragma = any(PRAGMA in l for l in lineas[llamada.lineno - 1:fin])
            if not (con_marcador or con_pragma):
                encontradas.append(
                    f"{sitio}: DELETE FROM {tabla} sin marcador propio en los parámetros "
                    f"({sorted(_nombres(params)) or 'ninguno'})"
                )
    return encontradas, permitidos_vistos


def test_ningun_test_borra_filas_de_tablas_compartidas_sin_marcador_propio():
    encontradas, _ = violaciones(archivos_de_test())
    assert encontradas == []


def test_los_permitidos_existen_todavia():
    """Una excepción que ya no existe en el código se saca de PERMITIDOS."""
    _, vistos = violaciones(archivos_de_test())
    assert vistos == PERMITIDOS


def _escribir(tmp_path, codigo):
    ruta = tmp_path / "test_ejemplo.py"
    ruta.write_text(codigo, encoding="utf-8")
    return violaciones([ruta], raiz=tmp_path)[0]


def test_control_detecta_where_por_columna_compartida(tmp_path):
    """El incidente: un WHERE que filtra por un valor que también usan otros."""
    assert len(_escribir(tmp_path, 'async def f(cur):\n'
                                   '    await cur.execute("DELETE FROM axioma_usage WHERE facet=%s", ("jekyll",))\n')) == 1


def test_control_detecta_las_formas_multi_tabla_y_con_modificadores(tmp_path):
    """m3 de la re-review: `DELETE u FROM t u ...` y
    `DELETE LOW_PRIORITY QUICK IGNORE FROM t ...` también borran."""
    codigo = ('async def f(cur, facet):\n'
              '    await cur.execute("DELETE u FROM axioma_usage u WHERE u.facet=%s", ("jekyll",))\n'
              '    await cur.execute("DELETE LOW_PRIORITY QUICK IGNORE FROM facet_health_event WHERE facet=%s", (facet,))\n'
              '    await cur.execute("DELETE e, a FROM facet_health_event e JOIN facet_health_alert a ON a.facet=e.facet")\n')
    assert len(_escribir(tmp_path, codigo)) == 3


def test_control_la_forma_multi_tabla_con_marcador_propio_no_se_marca(tmp_path):
    codigo = ('async def f(cur, pid):\n'
              '    await cur.execute("DELETE s FROM jacobs_steps s WHERE s.pipeline_id=%s", (pid,))\n')
    assert _escribir(tmp_path, codigo) == []


def test_control_detecta_delete_sin_where_y_tabla_desconocida(tmp_path):
    codigo = ('async def f(cur, nombre, facet):\n'
              '    await cur.execute("DELETE FROM jacobs_steps")\n'
              '    await cur.execute(f"DELETE FROM {nombre} WHERE facet=%s", (facet,))\n')
    assert len(_escribir(tmp_path, codigo)) == 2


def test_control_acepta_marcador_propio_pragma_y_tablas_no_compartidas(tmp_path):
    codigo = ('async def f(cur, pid, s, self):\n'
              '    await cur.execute("DELETE FROM jacobs_pipelines WHERE pipeline_id=%s", (pid,))\n'
              '    await cur.execute("DELETE FROM axioma_usage WHERE id > %s AND model = %s", (self.id_previo, self.model_id))\n'
              '    await cur.execute("DELETE FROM credential WHERE provider_id=%s", (s.proveedor,))  # marcador-propio: s = _Semilla (uuid)\n'
              '    await cur.execute("DELETE FROM provider WHERE id=%s", (s.proveedor,))\n')
    assert _escribir(tmp_path, codigo) == []
