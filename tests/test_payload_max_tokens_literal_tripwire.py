#!/usr/bin/env python3
"""TRIPWIRE -- ningún payload de jax lleva el límite de salida como NÚMERO
LITERAL (PR-K, 2026-09-14).

`"max_tokens": 131072` fijo en jax/muscles/base.py (x2) y jacobs/plan.py fue
la misma clase de defecto que tumbó a thot en la Mesa web (2026-08-24): el
nombre y el tope son propiedades POR MODELO y viven en la fila de `model`
(jax/core/contrato_dispatch.py). Este test es de CLASE, como
test_aiomysql_connect_timeout_tripwire.py: recorre con ast todos los .py de
código (no tests: sus fixtures de config del Motor Registry llevan
`"max_tokens": 8000` y no son payloads) y falla ante cualquiera de:

    {"max_tokens": 131072}           payload["max_tokens"] = 131072
    dict(max_tokens=131072)          payload.update(max_tokens=131072)
    payload.setdefault("max_tokens", 131072)
    TOPE = 131072 ... {"max_tokens": TOPE}   (constante de módulo, ronda 2)
    (y lo mismo con max_completion_tokens, maxOutputTokens y num_predict)

Ronda 2 de PR-K (M1): se agregaron `.update(...)`, `.setdefault(...)`, el
literal detrás de una constante de módulo y `num_predict` (Ollama). Un valor
leído de configuración con el patrón de la casa (`int(os.getenv("X", "N"))`,
p. ej. JACOBS_PLAN_NUM_PREDICT) no es un literal: es configuración, y el
valor que se manda igual pasa por el tope del catálogo.

Un valor que NO es literal (una variable, un campo de config, lo que devuelve
el helper) no se reporta: el tripwire vigila el literal, que es el síntoma
que se repitió; que el valor venga del catálogo lo prueban los tests de
tests/test_contrato_dispatch_repl_ada.py.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import ast
import tempfile
import unittest
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
EXCLUIDOS = {".git", "__pycache__", ".pytest_cache", "node_modules", "tests"}
CLAVES = {"max_tokens", "max_completion_tokens", "maxOutputTokens", "num_predict"}

# Mismo criterio que el tripwire de connect_timeout: un archivo que no parsea
# se declara con motivo o hace fallar la corrida.
# SONDAS DE MEDICION (2026-09-16). En un instrumento de medida, el limite de
# salida literal NO es el defecto que este tripwire persigue: ES EL
# INSTRUMENTO. `num_predict: 8` en la sonda de cola existe para medir la ESPERA
# y no el tiempo de generacion; si ese numero saliera del catalogo, la sonda
# mediria otra cosa cada vez que alguien cambiara la fila del modelo, y las
# corridas dejarian de ser comparables entre si.
#
# Es el mismo motivo por el que este tripwire ya excluye `tests/`: sus fixtures
# tampoco son payloads. La diferencia con una allowlist es que la exclusion se
# gana: `test_las_sondas_declaradas_no_son_codigo_de_servicio` comprueba que
# ningun modulo de jax/, jacobs/ o las_manos/ las importe. El dia que una de
# estas sondas entre al camino de produccion, este test se pone rojo.
_SONDAS_DE_MEDICION = {
    "scripts/ejecutor_fase0/contexto.py":
        "mide tok/s por contexto: num_predict fijo para que las tres corridas sean comparables",
    "scripts/ejecutor_fase0/endpoints.py":
        "prueba de contrato de /v1/messages: max_tokens minimo, no genera texto util",
    "scripts/ejecutor_fase0/sonda_cola.py":
        "mide la ESPERA en cola: num_predict=8 para que el tiempo de generacion no la tape",
    "scripts/ejecutor_fase0/auditor_costo.py":
        "mide el costo del auditor: el tope acota el gasto de la medicion, no una respuesta",
}

# Vacía desde el 2026-09-16 (E-01): el único no-módulo real del árbol,
# _director_patch/routes_block.py, se retiró. El mecanismo se conserva y se
# ejercita con un .py roto declarado a propósito dentro del test.
_NO_PARSEA: dict[str, str] = {}


def _es_test(path: Path) -> bool:
    nombre = path.name
    return nombre.startswith("test_") or nombre.endswith("_test.py")


def _excluido(path: Path) -> bool:
    for parte in path.relative_to(RAIZ).parts:
        if parte in EXCLUIDOS:
            return True
        minuscula = parte.lower()
        if "venv" in minuscula or "scratch" in minuscula:
            return True
    return _es_test(path)


def _es_numero(nodo: ast.AST) -> bool:
    if isinstance(nodo, ast.UnaryOp) and isinstance(nodo.op, (ast.USub, ast.UAdd)):
        nodo = nodo.operand
    return (isinstance(nodo, ast.Constant) and isinstance(nodo.value, (int, float))
            and not isinstance(nodo.value, bool))


def _constantes_de_modulo(arbol: ast.Module) -> set[str]:
    """Nombres asignados a un número literal a nivel de módulo (`TOPE = 131072`)."""
    nombres = set()
    for nodo in arbol.body:
        if isinstance(nodo, ast.Assign) and _es_numero(nodo.value):
            nombres.update(t.id for t in nodo.targets if isinstance(t, ast.Name))
        elif isinstance(nodo, ast.AnnAssign) and nodo.value is not None and _es_numero(nodo.value) \
                and isinstance(nodo.target, ast.Name):
            nombres.add(nodo.target.id)
    return nombres


_CONSTANTES: set[str] = set()


def _literal_numerico(nodo: ast.AST) -> bool:
    return _es_numero(nodo) or (isinstance(nodo, ast.Name) and nodo.id in _CONSTANTES)


def _clave(nodo: ast.AST) -> str | None:
    if isinstance(nodo, ast.Constant) and nodo.value in CLAVES:
        return nodo.value
    return None


def _hallazgos_en(path: Path) -> list[str]:
    rel = path.relative_to(RAIZ)
    if rel.as_posix() in _SONDAS_DE_MEDICION:
        return []
    try:
        arbol = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        if str(rel) in _NO_PARSEA:
            return []
        return [f"{rel}: no parsea (SyntaxError) -- declaralo en _NO_PARSEA o arreglalo."]
    global _CONSTANTES
    _CONSTANTES = _constantes_de_modulo(arbol)
    encontrados = []
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Dict):
            for k, v in zip(nodo.keys, nodo.values):
                clave = _clave(k) if k is not None else None
                if clave and _literal_numerico(v):
                    encontrados.append(f"{rel}:{v.lineno}: {{{clave!r}: literal}}")
        elif isinstance(nodo, ast.Assign):
            for t in nodo.targets:
                if isinstance(t, ast.Subscript) and _clave(t.slice) and _literal_numerico(nodo.value):
                    encontrados.append(f"{rel}:{nodo.lineno}: [{_clave(t.slice)!r}] = literal")
        elif isinstance(nodo, ast.Call):
            es_dict = isinstance(nodo.func, ast.Name) and nodo.func.id == "dict"
            metodo = nodo.func.attr if isinstance(nodo.func, ast.Attribute) else None
            if es_dict or metodo == "update":
                for kw in nodo.keywords:
                    if kw.arg in CLAVES and _literal_numerico(kw.value):
                        forma = "dict" if es_dict else ".update"
                        encontrados.append(f"{rel}:{kw.value.lineno}: {forma}({kw.arg}=literal)")
            if metodo == "setdefault" and len(nodo.args) >= 2 and _clave(nodo.args[0]) \
                    and _literal_numerico(nodo.args[1]):
                encontrados.append(f"{rel}:{nodo.lineno}: .setdefault({_clave(nodo.args[0])!r}, literal)")
    return encontrados


def _con_codigo(codigo: str) -> list[str]:
    with tempfile.NamedTemporaryFile("w", suffix=".py", dir=RAIZ, delete=True) as fh:
        fh.write(codigo)
        fh.flush()
        return _hallazgos_en(Path(fh.name))


class PayloadMaxTokensLiteralTripwireTest(unittest.TestCase):
    def test_ningun_payload_con_limite_de_salida_literal(self):
        hallazgos = []
        for p in RAIZ.rglob("*.py"):
            if not _excluido(p):
                hallazgos.extend(_hallazgos_en(p))
        self.assertEqual(
            hallazgos, [],
            "límite de salida como número literal -- sale de la fila de `model` "
            "(jax/core/contrato_dispatch.py::limite_de_salida):\n  " + "\n  ".join(hallazgos),
        )

    def test_el_detector_ve_las_tres_formas(self):
        """El freno se ejercita: un detector que no atrapa el caso obvio da un
        verde que no significa nada."""
        hallazgos = _con_codigo(
            "a = {'model': 'm', 'max_tokens': 131072}\n"
            "a['max_completion_tokens'] = 128000\n"
            "b = dict(maxOutputTokens=8192)\n"
        )
        self.assertEqual(len(hallazgos), 3, hallazgos)

    def test_el_detector_no_reporta_valores_que_no_son_literales(self):
        """Control negativo: el valor del catálogo, una variable o un campo
        de config no son el síntoma."""
        hallazgos = _con_codigo(
            "def f(tope, cfg, limite):\n"
            "    p = {'max_tokens': tope, **limite}\n"
            "    p['max_tokens'] = cfg['max_tokens']\n"
            "    return dict(max_completion_tokens=tope), p\n"
            "# {'max_tokens': 131072} en un comentario no es código\n"
        )
        self.assertEqual(hallazgos, [], hallazgos)

    def test_el_detector_ve_update_setdefault_y_constante_de_modulo(self):
        """Ronda 2 (M1): los puntos ciegos de la revisión, cada uno por separado."""
        casos = {
            "update": "p = {}\np.update(max_tokens=131072)\n",
            "setdefault": "p = {}\np.setdefault('max_completion_tokens', 128000)\n",
            "constante": "TOPE = 131072\ndef f():\n    return {'max_tokens': TOPE}\n",
            "constante_anotada": "TOPE: int = 8192\nb = dict(maxOutputTokens=TOPE)\n",
            "num_predict": "p = {'options': {'num_predict': 3000}}\n",
            "num_predict_constante": "N = 3000\np = {}\np['num_predict'] = N\n",
        }
        for nombre, codigo in casos.items():
            self.assertEqual(len(_con_codigo(codigo)), 1, nombre)

    def test_configuracion_leida_no_es_un_literal(self):
        """Control: el patrón de la casa para configuración no se reporta, y
        una constante local (no de módulo) tampoco se confunde."""
        hallazgos = _con_codigo(
            "import os\n"
            "N = int(os.getenv('JACOBS_PLAN_NUM_PREDICT', '3000'))\n"
            "def f(tope):\n"
            "    p = {'options': {'num_predict': min(N, tope)}}\n"
            "    p.update(max_tokens=tope)\n"
            "    p.setdefault('max_tokens', tope)\n"
            "    return p\n"
        )
        self.assertEqual(hallazgos, [], hallazgos)

    def test_los_tests_quedan_fuera_y_el_codigo_no(self):
        self.assertTrue(_excluido(RAIZ / "tests" / "test_x.py"))
        self.assertTrue(_excluido(RAIZ / "las_manos" / "_worker_max_tokens_test.py"))
        self.assertFalse(_excluido(RAIZ / "jax" / "muscles" / "base.py"))
        self.assertFalse(_excluido(RAIZ / "jacobs" / "plan.py"))


if __name__ == "__main__":
    unittest.main()


class SondasDeMedicionTest(unittest.TestCase):
    """La exclusion de las sondas se GANA, no se declara y ya."""

    def test_las_sondas_declaradas_existen(self):
        for rel in _SONDAS_DE_MEDICION:
            self.assertTrue((RAIZ / rel).exists(),
                            f"{rel} ya no existe: retirar de _SONDAS_DE_MEDICION")

    def test_las_sondas_declaradas_no_son_codigo_de_servicio(self):
        """Ningun modulo de servicio puede importarlas. El dia que una sonda
        entre al camino de produccion, su limite literal SI es el defecto que
        este tripwire persigue, y esto se pone rojo."""
        modulos = {Path(rel).stem for rel in _SONDAS_DE_MEDICION}
        ofensores = []
        for arbol_dir in ("jax", "jacobs", "las_manos"):
            base = RAIZ / arbol_dir
            if not base.is_dir():
                continue
            for py in base.rglob("*.py"):
                if "__pycache__" in py.parts:
                    continue
                try:
                    texto = py.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                for m in modulos:
                    if f"import {m}" in texto or f"from {m} " in texto or "ejecutor_fase0" in texto:
                        ofensores.append(f"{py.relative_to(RAIZ)} importa {m}")
        self.assertEqual(ofensores, [], "una sonda de medicion entro al codigo de servicio:\n"
                                        + "\n".join(ofensores))


class NoParseaTest(unittest.TestCase):
    """E-01 (2026-09-16): con la tabla vacía, las dos ramas del SyntaxError
    quedaban sin ejercitar. Se prueban con archivos de mentira."""

    def test_un_archivo_roto_no_declarado_se_reporta(self):
        hallazgos = _con_codigo("def f(:\n    pass\n")
        self.assertEqual(len(hallazgos), 1, hallazgos)
        self.assertIn("no parsea", hallazgos[0])

    def test_un_archivo_roto_declarado_no_se_reporta(self):
        from unittest.mock import patch
        with tempfile.NamedTemporaryFile("w", suffix=".py", dir=RAIZ, delete=True) as fh:
            fh.write("def f(:\n    pass\n")
            fh.flush()
            ruta = Path(fh.name)
            with patch.dict(_NO_PARSEA, {str(ruta.relative_to(RAIZ)): "roto a propósito"}):
                self.assertEqual(_hallazgos_en(ruta), [])
