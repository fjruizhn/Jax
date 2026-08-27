"""Maquina de estados de salud de facets. Funciones PURAS -- sin I/O, sin
red, sin DB. Ningun test de este archivo puede disparar una llamada paga."""
from jacobs import facet_health as fh

NOW = 1_000_000.0
W = fh.HEALTH_WINDOW_SECONDS


def test_cero_eventos_es_unknown_NUNCA_ok():
    """Ausencia de datos NO es salud. El chequeo es
    `if total_eventos == 0: unknown`, nunca `if fallos == 0: ok` -- esa
    segunda forma es el bug escrito como codigo."""
    got = fh.evaluate_states({}, ["thot", "ada"], NOW)
    assert got == {"thot": "unknown", "ada": "unknown"}
    assert "ok" not in got.values()


def test_evento_viejo_fuera_de_ventana_es_unknown():
    got = fh.evaluate_states({"thot": (NOW - W - 1, "ok")}, ["thot"], NOW)
    assert got["thot"] == "unknown"


def test_ultimo_evento_ok_es_ok():
    got = fh.evaluate_states({"thot": (NOW - 10, "ok")}, ["thot"], NOW)
    assert got["thot"] == "ok"


def test_ultimo_evento_de_falla_es_down():
    for bad in ("provider_error", "gate_denied", "gate_unreachable",
                "unbound", "unsupported_transport", "probe_error"):
        got = fh.evaluate_states({"thot": (NOW - 10, bad)}, ["thot"], NOW)
        assert got["thot"] == "down", bad


def test_ningun_facet_con_eventos_produce_una_sola_alerta_de_sistema():
    got = fh.evaluate_states({}, ["thot", "ada", "jekyll", "kimi"], NOW)
    notify = fh.transitions_to_notify(got, ledger={}, now=NOW)
    keys = [k for k, _ in notify]
    assert keys == [fh.SYSTEM_KEY]      # una sola, no cuatro


def test_TABLA_VACIA_alerta_system_y_NO_devuelve_lista_vacia():
    """EL test que impide que el agujero vuelva en un refactor.

    known_facets sale de facet_health_event (ruling 2), asi que con la
    tabla vacia `current` queda {} -- y {} es falsy. Un guard escrito como
    `if current and all(...)` SALTA el bloque de __system__, el bucle no
    itera nada, y la funcion devuelve []: SILENCIO TOTAL.

    Cuando pasa: la sonda nunca se cableo, murio al arrancar, el escritor
    esta roto y no escribe nunca, o jax-platform lleva caido mas que la
    retencion. O sea, los casos en que el detector esta MUERTO son
    exactamente los que no alertarian. Es el bug del punto A dentro del
    codigo que implementa el punto A.

    Por eso transitions_to_notify() arranca con
    `if not current: return _maybe(SYSTEM_KEY, "unknown", ledger, now)`."""
    assert fh.transitions_to_notify({}, ledger={}, now=NOW) == [(fh.SYSTEM_KEY, "unknown")]


def test_tabla_vacia_tambien_respeta_la_supresion_de_6h():
    ledger = {fh.SYSTEM_KEY: ("unknown", NOW - 60)}
    assert fh.transitions_to_notify({}, ledger, NOW) == []


def test_la_alerta_de_sistema_respeta_la_repeticion_de_6h():
    """Si la sonda muere un viernes, no queremos 288 mensajes el sabado."""
    states = fh.evaluate_states({}, ["thot", "ada"], NOW)
    ledger = {fh.SYSTEM_KEY: ("unknown", NOW - 60)}   # ya avisado hace 1 min
    assert fh.transitions_to_notify(states, ledger, NOW) == []

    ledger = {fh.SYSTEM_KEY: ("unknown", NOW - fh.ALERT_REPEAT_SECONDS - 1)}
    assert [k for k, _ in fh.transitions_to_notify(states, ledger, NOW)] == [fh.SYSTEM_KEY]


def test_facet_caido_no_re_alerta_en_cada_barrido():
    states = {"thot": "down"}
    ledger = {"thot": ("down", NOW - 60)}
    assert fh.transitions_to_notify(states, ledger, NOW) == []


def test_recuperacion_se_notifica():
    states = {"thot": "ok"}
    ledger = {"thot": ("down", NOW - 60)}
    assert fh.transitions_to_notify(states, ledger, NOW) == [("thot", "ok")]


def test_system_key_no_puede_colisionar_con_un_facet_real():
    """Los nombres de facet vienen de facet.key; ninguno empieza con
    guion bajo."""
    assert fh.SYSTEM_KEY.startswith("__")
    for real in ("thot", "ada", "hipatia", "jekyll", "kimi", "jax_local", "hyde"):
        assert real != fh.SYSTEM_KEY
        assert not real.startswith("_")
