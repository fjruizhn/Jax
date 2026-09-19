# tests/test_ejecutor_ollama_cpu_gpu_off.py
"""La garantía de "cero GPU" del auditor local vive en el repo, no sólo en
`/etc/jax/ollama-cpu.env` (hallazgo ALTO de la revisión, 2026-09-18): sin plantilla en
control de versiones, sin instalador y sin canario, alguien podía borrar
`GGML_VK_VISIBLE_DEVICES` del archivo en producción, el proceso arrancaría viendo la GPU
por Vulkan otra vez (el mismo hallazgo del journal, ver ops/ejecutor/ollama-cpu.service) y
nadie se enteraría hasta que la Mesa perdiera su modelo.

Estos tests son ESTRUCTURALES (sin systemd, sin Ollama real, sin red -- eso lo hace
`ops/ejecutor/canario_ollama_cpu.sh`, verificado en vivo a mano el 2026-09-18: pasa con la
configuración correcta, size_vram=0; falla con `GGML_VK_VISIBLE_DEVICES` quitada,
size_vram>0 -- ver CONTEXT.md/la Biblioteca de esta ronda). Lo que SÍ se puede probar en CI
sin infraestructura real es que el CONTRATO quede escrito: las cuatro variables presentes
y vacías en la plantilla que gobierna producción, el instalador exige lo que hace falta y
corre el canario, y el canario de verdad mira `size_vram` (no un `grep` sobre el journal,
cuyo formato es de Ollama y puede cambiar sin aviso)."""
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_PLANTILLA = _REPO / "ops" / "ejecutor" / "ollama-cpu.env.plantilla"
_UNIDAD = _REPO / "ops" / "ejecutor" / "ollama-cpu.service"
_INSTALADOR = _REPO / "ops" / "ejecutor" / "instalar_ollama_cpu.sh"
_CANARIO = _REPO / "ops" / "ejecutor" / "canario_ollama_cpu.sh"

# Las cuatro variables medidas en vivo (spec 2026-09-18-auditor-local-opcion.md, hallazgo
# de Vulkan): con sólo las dos primeras, `ollama serve` seguía viendo la GPU discreta por
# el backend Vulkan. Vacías (no ausentes, no con un valor cualquiera): un valor no vacío
# podría, según la versión de Ollama, significar "sólo este dispositivo" en vez de
# "ninguno".
_VARIABLES_GPU_OFF = ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES",
                      "GGML_VK_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")


def _variables_vacias(texto: str) -> set[str]:
    """`{VAR}` para cada línea `VAR=` (valor vacío) que no sea un comentario."""
    vacias = set()
    for linea in texto.splitlines():
        linea = linea.strip()
        if not linea or linea.startswith("#") or "=" not in linea:
            continue
        clave, _, valor = linea.partition("=")
        if valor.strip() == "":
            vacias.add(clave.strip())
    return vacias


def test_los_archivos_existen():
    for ruta in (_PLANTILLA, _UNIDAD, _INSTALADOR, _CANARIO):
        assert ruta.is_file(), f"falta {ruta.relative_to(_REPO)}"


def test_el_instalador_y_el_canario_son_ejecutables():
    import os
    for ruta in (_INSTALADOR, _CANARIO):
        assert os.access(ruta, os.X_OK), f"{ruta.relative_to(_REPO)} no tiene el bit +x"


def test_las_cuatro_variables_de_gpu_off_estan_vacias_en_la_plantilla():
    presentes = _variables_vacias(_PLANTILLA.read_text(encoding="utf-8"))
    faltan = set(_VARIABLES_GPU_OFF) - presentes
    assert not faltan, (
        f"la plantilla no declara VACÍAS estas variables: {sorted(faltan)} -- sin ellas, "
        "ollama-cpu.service puede arrancar viendo la GPU (ver el hallazgo de Vulkan en "
        "ops/ejecutor/ollama-cpu.service)")


def test_la_deteccion_de_variables_vacias_distingue_ausente_de_con_valor():
    """Auto-verificación (Principio VII): el helper de arriba tiene que fallar ante una
    plantilla mutada -- si no, este control da verde sin haber mirado nada."""
    sintetica = "HIP_VISIBLE_DEVICES=\nROCR_VISIBLE_DEVICES=0\n# CUDA_VISIBLE_DEVICES=\n"
    assert _variables_vacias(sintetica) == {"HIP_VISIBLE_DEVICES"}
    faltan = set(_VARIABLES_GPU_OFF) - _variables_vacias(sintetica)
    assert faltan == {"ROCR_VISIBLE_DEVICES", "GGML_VK_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"}, faltan


def test_la_unidad_apunta_al_archivo_que_renderiza_el_instalador():
    unidad = _UNIDAD.read_text(encoding="utf-8")
    instalador = _INSTALADOR.read_text(encoding="utf-8")
    assert "EnvironmentFile=/etc/jax/ollama-cpu.env" in unidad
    assert "/etc/jax/ollama-cpu.env" in instalador
    assert "ollama-cpu.env.plantilla" in instalador


def test_el_instalador_exige_puerto_modelos_y_modelo_sin_default():
    texto = _INSTALADOR.read_text(encoding="utf-8")
    for variable in ("JAX_OLLAMA_CPU_PUERTO", "JAX_OLLAMA_CPU_MODELOS", "JAX_OLLAMA_CPU_MODELO"):
        assert f'"${{{variable}:?}}"' in texto, (
            f"{variable} no está exigida sin default en instalar_ollama_cpu.sh -- un puerto "
            "hardcodeado sobreviviría a cualquier cambio de infraestructura sin que nadie "
            "se entere")


def test_el_instalador_corre_el_canario_y_no_declara_exito_sin_el():
    texto = _INSTALADOR.read_text(encoding="utf-8")
    assert "canario_ollama_cpu.sh" in texto
    # El canario tiene que correr ANTES de la línea final de éxito, no después (si no, un
    # canario que falla igual imprimiría "ollama_cpu_instalado=true").
    pos_canario = texto.index("canario_ollama_cpu.sh")
    pos_exito = texto.index("ollama_cpu_instalado=true")
    assert pos_canario < pos_exito, "el canario corre DESPUÉS de declarar éxito -- no protege nada"


def test_el_canario_verifica_size_vram_no_el_journal():
    """El journal es texto de Ollama, no nuestro, y puede cambiar de formato entre
    versiones sin aviso -- `size_vram` de /api/ps es la API estable. Ver la verificación
    en vivo (2026-09-18): con la configuración rota (sin GGML_VK_VISIBLE_DEVICES) el
    canario detectó size_vram=5347003596 y salió con código != 0."""
    texto = _CANARIO.read_text(encoding="utf-8")
    assert "size_vram" in texto and "/api/ps" in texto and "/api/generate" in texto
    assert "journalctl" not in texto, "el canario no debe depender del formato del journal"
    assert 'exit 1' in texto
