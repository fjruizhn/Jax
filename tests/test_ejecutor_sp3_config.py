# tests/test_ejecutor_sp3_config.py
"""SP3 del Ejecutor (2026-09-17): los números de ops/ejecutor/sp3_entorno.env y los freno de los
dos scripts de despliegue. Spec de Fase 2 §6.3.

La cuenta: una petición del Ejecutor en curso NO se interrumpe, así que la Mesa espera, en el peor
caso, lo que tarda esa petición entera: leer su entrada (sin caché: la Mesa acaba de pisar el KV
del único slot) y generar su salida. V3 exige espera de la Mesa ≤ 60 s; con margen, ≤ 48 s.
"""
import os
import subprocess
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
ENTORNO = RAIZ / "ops" / "ejecutor" / "sp3_entorno.env"
ESPERA_MAXIMA_V3_S = 60.0
MARGEN = 0.8

#: MEDIDO 2026-09-17 03:06–03:13 en hall9000 (qwen3.6:35b-a3b-q4_K_M a 131072, bge-m3 al lado,
#: ~/ejecutor-producto/sp3-mediciones/largos_131072.jsonl y peor_caso_131072.jsonl): tokens de
#: entrada → segundos hasta el primer token (Ollama, sin carga del modelo), el PEOR de cada tamaño.
LECTURA_MEDIDA = ((3818, 1.92), (15167, 7.95), (26361, 15.59), (59886, 48.01), (93915, 95.40))
#: La generación MÁS LENTA medida a 131072 (93303 tokens de contexto): 62,57 tok/s.
GENERACION_MIN_TOK_S = 62.57


def _entorno():
    valores = {}
    for linea in ENTORNO.read_text(encoding="utf-8").splitlines():
        if linea.strip() and not linea.startswith("#"):
            clave, valor = linea.split("=", 1)
            valores[clave] = valor
    return valores


def _lectura_s(tokens: int) -> float:
    """Cota SUPERIOR del tiempo de lectura: la cuerda entre los dos puntos medidos que lo rodean.
    Vale porque la lectura es convexa (la velocidad por token CAE con el tamaño: lo comprueba el
    test de abajo), y la cuerda de una convexa queda por encima de la curva."""
    for (p0, t0), (p1, t1) in zip(LECTURA_MEDIDA, LECTURA_MEDIDA[1:]):
        if p0 <= tokens <= p1:
            return t0 + (tokens - p0) * (t1 - t0) / (p1 - p0)
    raise AssertionError(f"{tokens} tokens fuera de lo medido: no se extrapola")


def test_la_lectura_medida_es_convexa_y_la_cuerda_es_cota_superior():
    velocidades = [p / t for p, t in LECTURA_MEDIDA]
    assert velocidades == sorted(velocidades, reverse=True), velocidades
    pendientes = [(t1 - t0) / (p1 - p0) for (p0, t0), (p1, t1) in zip(LECTURA_MEDIDA, LECTURA_MEDIDA[1:])]
    assert pendientes == sorted(pendientes), pendientes


def test_el_techo_de_entrada_es_la_ventana_mas_un_salto_de_herramienta():
    e = _entorno()
    salto = max(int(e["JAX_EJECUTOR_BASH_MAX_OUTPUT_LENGTH"]) // int(e["JAX_EJECUTOR_CARACTERES_POR_TOKEN_MIN"]),
                int(e["JAX_EJECUTOR_FILE_READ_MAX_OUTPUT_TOKENS"]))
    assert int(e["JAX_EJECUTOR_AUTO_COMPACT_WINDOW"]) + salto <= int(e["JAX_EJECUTOR_MAX_ENTRADA_TOKENS"])


def test_la_peor_peticion_del_ejecutor_deja_a_la_mesa_esperando_menos_del_umbral_con_margen():
    e = _entorno()
    lectura = _lectura_s(int(e["JAX_EJECUTOR_MAX_ENTRADA_TOKENS"]))
    generacion = int(e["JAX_PROXY_CARRIL_MAX_SALIDA_TOKENS"]) / GENERACION_MIN_TOK_S
    espera = lectura + generacion
    assert espera <= ESPERA_MAXIMA_V3_S * MARGEN, (lectura, generacion, espera)


def test_el_proxy_fija_el_mismo_modelo_que_la_mesa_a_131072():
    e = _entorno()
    assert e["JAX_PROXY_CARRIL_MODELO"] == e["JAX_MESA_MODELO_DERIVADO"]
    assert e["JAX_MESA_NUM_CTX"] == "131072"


def _camino_espia(tmp_path, nombres):
    """Un PATH donde sudo, mariadb, curl, ollama y docker sólo anotan que los llamaron."""
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    llamadas = tmp_path / "llamadas"
    for n in nombres:
        (bin_ / n).write_text(f'#!/bin/sh\necho {n} "$@" >> "{llamadas}"\nexit 0\n')
        (bin_ / n).chmod(0o755)
    return f"{bin_}:/usr/bin:/bin", llamadas


def test_el_carril_comun_se_niega_a_dar_acceso_a_la_cuenta_del_ejecutor_antes_de_tocar_nada(tmp_path):
    camino, llamadas = _camino_espia(tmp_path, ["sudo", "groupadd", "usermod", "install", "chgrp", "chmod"])
    r = subprocess.run(["bash", str(RAIZ / "ops" / "ejecutor" / "instalar_carril_comun.sh")], capture_output=True,
                       text=True, env={"PATH": camino, "JAX_PROXY_CARRIL_RAIZ": str(tmp_path / "carril"),
                                       "JAX_CARRIL_GRUPO": "jax-carril", "JAX_CARRIL_CUENTAS": "fruiz axioma",
                                       "JAX_EJECUTOR_CUENTA": "axioma"})
    assert r.returncode == 2
    assert "cuenta_del_ejecutor_en_el_grupo_del_carril=axioma" in r.stderr
    assert not llamadas.exists(), llamadas.read_text()


def test_el_entorno_versionado_no_pone_al_ejecutor_en_el_grupo_del_carril():
    assert "axioma" not in _entorno()["JAX_CARRIL_CUENTAS"].split()


def test_unificar_no_aplica_si_el_proxy_fijaria_otro_modelo_y_no_toca_nada(tmp_path):
    camino, llamadas = _camino_espia(tmp_path, ["sudo", "mariadb", "mariadb-dump", "curl", "ollama", "docker"])
    env = {**{k: v for k, v in _entorno().items()}, "PATH": camino, "HOME": str(tmp_path),
           "JAX_DB_USER": "u", "JAX_DB_PASSWORD": "p", "JAX_DB_NAME": "n", "JAX_DB_HOST": "127.0.0.1",
           "JAX_DB_PORT": "1", "JAX_OLLAMA_URL": "http://127.0.0.1:9", "JAX_PROXY_CARRIL_MODELO": "otro-modelo"}
    r = subprocess.run(["bash", str(RAIZ / "ops" / "ejecutor" / "unificar_contexto_mesa.sh"), "--aplicar"],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 1 and "proxy_y_mesa_con_modelos_distintos" in r.stderr
    assert not llamadas.exists(), llamadas.read_text()


def test_unificar_cambia_el_binding_solo_por_el_endpoint_aprobado():
    fuente = (RAIZ / "ops" / "ejecutor" / "unificar_contexto_mesa.sh").read_text(encoding="utf-8").upper()
    for prohibido in ("UPDATE FACET_BINDING", "INSERT INTO FACET_BINDING", "DELETE FROM FACET_BINDING",
                      "UPDATE MODEL", "INSERT INTO MODEL"):
        assert prohibido not in fuente, prohibido
    assert "/API/ADMIN/FACET-BINDINGS/" in fuente
    assert subprocess.run(["bash", "-n", str(RAIZ / "ops" / "ejecutor" / "unificar_contexto_mesa.sh")]).returncode == 0
