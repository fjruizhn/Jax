# tests/test_ejecutor_sp3_config.py
"""SP3 del Ejecutor (2026-09-17): los números de ops/ejecutor/sp3_entorno.conf y los freno de los
dos scripts de despliegue. Spec de Fase 2 §6.3.

La cuenta: una petición del Ejecutor en curso NO se interrumpe, así que la Mesa espera, en el peor
caso, lo que tarda esa petición entera: leer su entrada (sin caché: la Mesa acaba de pisar el KV
del único slot) y generar su salida. V3 exige espera de la Mesa ≤ 60 s; con margen, ≤ 48 s.
"""
import json
import os
import subprocess
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
ENTORNO = RAIZ / "ops" / "ejecutor" / "sp3_entorno.conf"
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


# --- --aplicar de punta a punta contra falsos (2026-09-17: dos defectos vistos en la corrida real) ---

_FALSO = r'''#!/usr/bin/env python3
import json, os, sys
nombre, args = os.path.basename(sys.argv[0]), sys.argv[1:]
estado = os.environ["FALSO_ESTADO"]
with open(os.path.join(estado, "llamadas"), "a") as f:
    f.write(json.dumps([nombre] + args) + "\n")
esc = json.load(open(os.path.join(estado, "escenario.json")))
CHECKSUM = "facet_binding\t11\nmodel\t22"
todo = " ".join(args)
if nombre == "mariadb":
    q = args[args.index("-e") + 1] if "-e" in args else ""
    if "CHECKSUM TABLE" in q:
        print("jax_memory." + CHECKSUM.replace("\nmodel", "\njax_memory.model"))
    elif "SELECT VERSION()" in q:
        print("12.3.3-MariaDB")
    elif "FROM axioma_usage" in q:
        print(0)
    elif "SELECT b.model_ref" in q:
        print(esc["binding_base"])
    elif "SELECT id FROM model" in q:
        # El sync del catálogo guarda el derivado CON el tag, como lo lista Ollama.
        for model_id, ref in esc["catalogo"].items():
            if "'" + model_id + "'" in q:
                print(ref)
    elif "SELECT outcome FROM facet_health_event" in q:
        print("ok")
    elif "SELECT model_ref FROM facet_binding" in q:
        print(open(os.path.join(estado, "binding")).read().strip())
elif nombre == "mariadb-dump":
    print("-- dump")
elif nombre == "sudo":
    if "CHECKSUM TABLE" in todo:
        print("prueba." + CHECKSUM.replace("\nmodel", "\nprueba.model"))
elif nombre == "ollama":
    if args[:1] == ["show"]:
        print("num_ctx                        " + esc["ctx"])
elif nombre == "curl":
    url = args[-1]
    if url.endswith("/api/ps"):
        print(json.dumps({"models": [{"name": esc["derivado"] + ":latest", "context_length": int(esc["ctx"])}]}))
        sys.exit(0)
    salida = args[args.index("-o") + 1]
    metodo = args[args.index("-X") + 1]
    datos = args[args.index("--data") + 1] if "--data" in args else ""
    codigo, cuerpo = 200, "{}"
    if metodo == "PUT" and "/contrato-dispatch" in url:
        c = json.loads(datos)
        if c["max_tokens_param"] is None and c["max_output_tokens"] is None:
            codigo, cuerpo = 422, '{"detail":"contrato_vacio"}'  # lo que respondió producción
    elif metodo == "PUT" and "/facet-bindings/" in url:
        open(os.path.join(estado, "binding"), "w").write(str(json.loads(datos)["model_ref"]))
    open(salida, "w").write(cuerpo)
    sys.stdout.write(str(codigo))
'''


def _aplicar_contra_falsos(tmp_path, binding_base, catalogo):
    e = _entorno()
    estado = tmp_path / "estado"
    estado.mkdir()
    (estado / "binding").write_text(binding_base.split("\t")[0])
    (estado / "escenario.json").write_text(json.dumps({
        "binding_base": binding_base, "catalogo": catalogo,
        "derivado": e["JAX_MESA_MODELO_DERIVADO"], "ctx": e["JAX_MESA_NUM_CTX"]}))
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    for n in ("mariadb", "mariadb-dump", "sudo", "ollama", "curl"):
        (bin_ / n).write_text(_FALSO)
        (bin_ / n).chmod(0o755)
    token = tmp_path / "token"
    token.write_text("t")
    token.chmod(0o600)
    env = {**e, "PATH": f"{bin_}:/usr/bin:/bin", "HOME": str(tmp_path), "FALSO_ESTADO": str(estado),
           "JAX_SP3_RESPALDOS": str(tmp_path / "respaldos"), "JAX_ADMIN_TOKEN_ARCHIVO": str(token),
           "JAX_PLATFORM_URL": "http://127.0.0.1:9", "JAX_OLLAMA_URL": "http://127.0.0.1:9",
           "JAX_DB_USER": "u", "JAX_DB_PASSWORD": "p", "JAX_DB_NAME": "n", "JAX_DB_HOST": "127.0.0.1",
           "JAX_DB_PORT": "1"}
    r = subprocess.run(["bash", str(RAIZ / "ops" / "ejecutor" / "unificar_contexto_mesa.sh"), "--aplicar"],
                       capture_output=True, text=True, env=env, timeout=60)
    llamadas = [json.loads(l) for l in (estado / "llamadas").read_text().splitlines()]
    contratos = [c for c in llamadas if c[0] == "curl" and any("/contrato-dispatch" in a for a in c)]
    return r, (estado / "binding").read_text(), contratos


def test_unificar_encuentra_el_derivado_con_el_tag_que_guarda_el_sync(tmp_path):
    derivado = _entorno()["JAX_MESA_MODELO_DERIVADO"]
    r, binding, _ = _aplicar_contra_falsos(tmp_path, "2798\tollama\tqwen3.6:35b\tNULL\tNULL",
                                           {derivado + ":latest": "2799"})
    assert r.returncode == 0, r.stdout + r.stderr
    assert "unificado=true" in r.stdout and binding == "2799"


def test_unificar_no_copia_un_contrato_de_dispatch_vacio_del_base(tmp_path):
    """Transporte `ollama`: el base no declara contrato (NULL, NULL) y el PUT lo rechaza con 422."""
    derivado = _entorno()["JAX_MESA_MODELO_DERIVADO"]
    r, binding, contratos = _aplicar_contra_falsos(tmp_path, "2798\tollama\tqwen3.6:35b\tNULL\tNULL",
                                                   {derivado + ":latest": "2799"})
    assert r.returncode == 0, r.stdout + r.stderr
    assert contratos == [] and binding == "2799"


def test_unificar_si_copia_un_contrato_declarado_en_el_base(tmp_path):
    derivado = _entorno()["JAX_MESA_MODELO_DERIVADO"]
    r, binding, contratos = _aplicar_contra_falsos(tmp_path, "2798\tollama\tqwen3.6:35b\tmax_tokens\t8192",
                                                   {derivado + ":latest": "2799"})
    assert r.returncode == 0, r.stdout + r.stderr
    assert len(contratos) == 1 and binding == "2799"
    cuerpo = json.loads(contratos[0][contratos[0].index("--data") + 1])
    assert cuerpo == {"max_tokens_param": "max_tokens", "max_output_tokens": 8192}


def test_unificar_no_aplica_si_el_derivado_esta_dos_veces_en_el_catalogo(tmp_path):
    derivado = _entorno()["JAX_MESA_MODELO_DERIVADO"]
    r, binding, _ = _aplicar_contra_falsos(tmp_path, "2798\tollama\tqwen3.6:35b\tNULL\tNULL",
                                           {derivado: "2800", derivado + ":latest": "2799"})
    assert r.returncode == 1 and "derivado_no_esta_en_el_catalogo" in r.stderr
    assert binding == "2798"
