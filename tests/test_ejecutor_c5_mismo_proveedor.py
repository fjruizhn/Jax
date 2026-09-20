# tests/test_ejecutor_c5_mismo_proveedor.py
"""La compuerta que permite auditar con el MISMO proveedor que el cerebro.

**Por qué existe (2026-09-20, decisión de Fernando).** El contrato dice «quien
produce no aprueba» y lo hace cumplir comparando `provider_id`. Medido ese día:
ningún auditor separado servía. En CPU, `qwen3:14b` 7/8 canarios a ~250 s por
vuelta; `qwen3.6:35b-a3b` 6/8 a ~286 s; `granite4.2:8b` 2/4 a 412–717 s. El
auditor de nube (`thot`) pasa 6/6 en ~10 s, pero manda la salida de los comandos
de máquinas con datos de clientes fuera de la casa, que es lo que la Fase 0
prohibió.

El modelo del cerebro, ya cargado en GPU, dio **8/8 con mediana de 80 s** con el
mismo arnés de canarios. Es la única opción medida que pasa todo, es tres veces
más rápida que cualquier alternativa local, y **no saca un byte de la casa**.

**Lo que NO se hizo, y por qué importa.** La salida fácil era crear una fila de
proveedor con otro nombre apuntando al mismo Ollama: el `if` de
`validar_proveedores` habría pasado en verde y la propiedad que protege habría
quedado rota igual. Eso no es cumplir el contrato, es hacer que el sistema
informe verde sobre una condición que no se cumple -- y deja escrito como
evidencia algo falso, que es peor que no tener el control (Principio VIII).

Por eso la regla se cambia A LA VISTA: una compuerta propia, que **nace
cerrada**, cuya apertura es decisión registrada de Fernando (queda en
`axioma_config_audit` con actor, fecha e IP), y un mensaje de rechazo que dice
la verdad en vez de callarla.

**Lo que sigue decorrelacionado aunque el modelo sea el mismo:** los canarios de
C5, que se eligen AL AZAR en cada arranque (`secrets.choice`) y frenan la misión
si el auditor falla. El 2026-09-20 frenaron dos veces, de verdad.
"""
import pytest

from jax.ejecutor.contratos import eleccion_c5 as E

CLAVE = "ejecutor.c5_auditor_admite_mismo_proveedor"

_FILAS_BASE = {
    "ejecutor.cerebro_faceta": "jax_local",
    "ejecutor.auditor_faceta": "thot",
    "ejecutor.auditor_faceta_local": "el_juez",
    "ejecutor.c5_lote_max": "20",
    "ejecutor.c5_intervalo_s": "15",
    "ejecutor.c5_max_tokens": "4000",
    "ejecutor.c5_auditor_admite_datos_de_clientes": "false",
}


def _filas(**extra):
    return {**_FILAS_BASE, CLAVE: "false", **extra}


# --- la clave es parte del contrato -------------------------------------------------------------

def test_la_clave_esta_en_CLAVES():
    """Si no está en CLAVES, `SQL_CONFIG` no la trae y `config_desde_filas` nunca
    la ve: la compuerta existiría en la base y no gobernaría nada."""
    assert CLAVE in E.CLAVES


def test_config_desde_filas_la_lee():
    cfg = E.config_desde_filas(_filas())
    assert cfg.admite_mismo_proveedor is False
    assert E.config_desde_filas(_filas(**{CLAVE: "true"})).admite_mismo_proveedor is True


def test_falta_la_fila_y_el_arranque_se_niega():
    """Fail-closed: una compuerta ausente NO se interpreta como abierta, ni como
    cerrada en silencio -- el arranque se niega y lo dice."""
    filas = _filas()
    del filas[CLAVE]
    with pytest.raises(ValueError) as e:
        E.config_desde_filas(filas)
    assert e.value.args[0] == "config_c5_incompleta"
    assert CLAVE in e.value.args[1]


@pytest.mark.parametrize("basura", ["si", "1", "True", "", " ", "yes", "abierta"])
def test_un_valor_que_no_es_true_ni_false_se_rechaza(basura):
    """Nada de `bool(texto)`: 'false' es una cadena no vacía y sería True.
    Mismo criterio que la compuerta de datos de clientes."""
    with pytest.raises(ValueError):
        E.config_desde_filas(_filas(**{CLAVE: basura}))


# --- la regla ------------------------------------------------------------------------------------

def test_cerrada_el_mismo_proveedor_sigue_frenando():
    """El comportamiento de siempre, intacto mientras nadie abra la compuerta."""
    fallos = E.validar_proveedores(proveedor_cerebro="ollama", proveedor_auditor="ollama",
                                   admite_mismo_proveedor=False)
    assert [f.codigo for f in fallos] == ["auditor_mismo_proveedor_que_el_cerebro"]


def test_abierta_el_mismo_proveedor_se_permite():
    assert E.validar_proveedores(proveedor_cerebro="ollama", proveedor_auditor="ollama",
                                 admite_mismo_proveedor=True) == ()


@pytest.mark.parametrize("abierta", [True, False])
def test_proveedores_distintos_pasan_con_la_compuerta_en_cualquier_estado(abierta):
    """La compuerta SOLO gobierna el caso del mismo proveedor. Abrirla no puede
    relajar ninguna otra cosa."""
    assert E.validar_proveedores(proveedor_cerebro="ollama", proveedor_auditor="openai",
                                 admite_mismo_proveedor=abierta) == ()


@pytest.mark.parametrize("abierta", [True, False])
def test_un_proveedor_vacio_falla_aunque_la_compuerta_este_abierta(abierta):
    """Abrir la compuerta del mismo proveedor no puede convertir 'no sé quién
    audita' en 'está bien'."""
    fallos = E.validar_proveedores(proveedor_cerebro="ollama", proveedor_auditor="  ",
                                   admite_mismo_proveedor=abierta)
    assert [f.codigo for f in fallos] == ["proveedor_desconocido"]


def test_el_rechazo_dice_que_hay_una_compuerta_y_no_miente():
    """El mensaje viejo daba a entender que era imposible. Ahora tiene que
    nombrar la compuerta, para que quien lo lea sepa que hay una decisión
    registrable detrás -- y no salga a renombrar un proveedor para esquivarlo,
    que es exactamente lo que este contrato NO quiere que pase."""
    fallo = E.validar_proveedores(proveedor_cerebro="ollama", proveedor_auditor="ollama",
                                  admite_mismo_proveedor=False)[0]
    datos = dict(fallo.datos)
    assert datos.get("compuerta") == CLAVE


# --- la compuerta nueva no toca la vieja ---------------------------------------------------------

def test_no_se_confunde_con_la_compuerta_de_datos_de_clientes():
    """Dos compuertas distintas, dos decisiones distintas. Abrir la del mismo
    proveedor NO puede abrir la de datos de clientes."""
    cfg = E.config_desde_filas(_filas(**{CLAVE: "true"}))
    assert cfg.admite_mismo_proveedor is True
    assert cfg.admite_datos_de_clientes is False


# --- la compuerta tiene que LLEGAR, no sólo existir ----------------------------------------------
# Una clave que se lee y no se enhebra hasta el llamador es una compuerta de
# adorno: el arranque seguiría frenando con ella abierta, y nadie sabría por qué.

def _eleccion(**kw):
    base = dict(proveedor_cerebro="ollama", proveedor_auditor="ollama", auditor_es_local=True,
                admite_datos_de_clientes=False, hosts_mision=frozenset({"bridge"}),
                hosts_con_clientes=frozenset({"bridge"}), hosts_conocidos=frozenset({"bridge"}))
    return E.validar_eleccion(**{**base, **kw})


def test_validar_eleccion_frena_con_la_compuerta_cerrada():
    codigos = [f.codigo for f in _eleccion(admite_mismo_proveedor=False)]
    assert "auditor_mismo_proveedor_que_el_cerebro" in codigos


def test_validar_eleccion_deja_pasar_con_la_compuerta_abierta():
    """El caso real de el_juez: mismo proveedor que el cerebro, auditor local de
    verdad, misión sobre una máquina con datos de clientes. Con la compuerta
    abierta no queda ningún fallo -- y sin abrirla, sí."""
    assert _eleccion(admite_mismo_proveedor=True) == ()


def test_abrir_la_del_mismo_proveedor_NO_abre_la_de_datos_de_clientes():
    """La prueba que impide que una compuerta se coma a la otra: auditor de NUBE
    (auditor_es_local=False) sobre una máquina con datos de clientes, con la
    compuerta del mismo proveedor abierta. Tiene que seguir frenando."""
    codigos = [f.codigo for f in _eleccion(proveedor_auditor="openai", auditor_es_local=False,
                                           admite_mismo_proveedor=True)]
    assert codigos == ["auditor_no_admite_datos_de_clientes"]
