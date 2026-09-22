# tests/test_ejecutor_contratos_pausa.py
"""La pausa PROPIA del Ejecutor (C5 la pone; el proxy la obedece). No es el interruptor
global de JAX: un falso positivo del auditor frena al Ejecutor, no a la Mesa.
Fail-closed: sin saber dónde está, o sin poder mirarla, se la da por PUESTA."""
import json
import os
from pathlib import Path

import pytest

from jax.ejecutor.contratos import pausa as P


def test_la_ruta_sale_del_entorno_y_tiene_que_ser_absoluta(tmp_path):
    assert P.ruta_de_la_pausa({P.VARIABLE_RUTA: str(tmp_path / "PAUSA")}) == tmp_path / "PAUSA"
    for env in ({}, {P.VARIABLE_RUTA: "  "}, {P.VARIABLE_RUTA: "relativa/PAUSA"}):
        with pytest.raises(P.PausaSinConfigurar):
            P.ruta_de_la_pausa(env)


def test_poner_una_vez_y_leer(tmp_path):
    ruta = tmp_path / "PAUSA"
    assert P.pausa_puesta(ruta) is False
    assert P.poner_pausa(ruta, {"origen": "c5", "motivo": "fuera_de_mision", "paso": 3}) is True
    assert P.pausa_puesta(ruta) is True
    doc = json.loads(ruta.read_text())
    assert (doc["origen"], doc["motivo"], doc["paso"]) == ("c5", "fuera_de_mision", 3) and doc["momento"]
    # La segunda no pisa la primera: el motivo que quedó es el del primer freno.
    assert P.poner_pausa(ruta, {"origen": "c5", "motivo": "auditor_caido", "paso": None}) is False
    assert json.loads(ruta.read_text())["motivo"] == "fuera_de_mision"
    assert [p.name for p in tmp_path.iterdir()] == ["PAUSA"], "no quedan temporales"


def test_sin_poder_mirarla_esta_puesta(tmp_path):
    archivo = tmp_path / "no-es-directorio"
    archivo.write_text("x")
    assert P.pausa_puesta(archivo / "PAUSA") is True  # ENOTDIR


@pytest.mark.skipif(os.geteuid() == 0, reason="root lee todo")
def test_directorio_ilegible_esta_puesta(tmp_path):
    cerrado = tmp_path / "cerrado"
    cerrado.mkdir()
    cerrado.chmod(0)
    try:
        assert P.pausa_puesta(cerrado / "PAUSA") is True
    finally:
        cerrado.chmod(0o700)


def test_latido_fresco_viejo_o_ausente(tmp_path):
    ruta = tmp_path / "latido"
    assert P.latido_fresco(ruta, 5, ahora=1000.0) is False
    P.latir(ruta)
    mtime = ruta.stat().st_mtime
    assert P.latido_fresco(ruta, 5, ahora=mtime + 4) is True
    assert P.latido_fresco(ruta, 5, ahora=mtime + 6) is False
    assert P.latido_fresco(tmp_path / "x" / "latido", 5) is False


# --- quitar_pausa_si (ronda 8, B-1): nunca levantar una pausa ajena --------------------

def test_quitar_pausa_si_coincide_la_borra(tmp_path):
    ruta = tmp_path / "PAUSA"
    P.poner_pausa(ruta, {"origen": "huella", "host": "atemai", "mision_id": "m1"})
    borro, datos = P.quitar_pausa_si(ruta, coincide=lambda d: d.get("origen") == "huella")
    assert borro is True
    assert datos["host"] == "atemai"
    assert not ruta.exists()


def test_quitar_pausa_si_no_coincide_la_deja(tmp_path):
    ruta = tmp_path / "PAUSA"
    P.poner_pausa(ruta, {"origen": "c4", "motivo": "freno"})
    borro, datos = P.quitar_pausa_si(ruta, coincide=lambda d: d.get("origen") == "huella")
    assert borro is False
    assert datos["origen"] == "c4"
    assert ruta.exists()
    assert P.pausa_puesta(ruta) is True


def test_quitar_pausa_si_sin_pausa_no_falla(tmp_path):
    ruta = tmp_path / "PAUSA"
    borro, datos = P.quitar_pausa_si(ruta, coincide=lambda d: True)
    assert borro is False and datos is None


def test_quitar_pausa_si_no_coincide_ruta_nunca_desaparece_ronda9(tmp_path, monkeypatch):
    """BLOCK-1/MAJOR-1 (ronda 9), invierte el test de la línea 87 de la ronda 8: ANTES,
    "no coincide" pasaba por robarse `ruta` con `os.rename` y reponerla -- había una
    VENTANA real donde `ruta` no existía. Ahora usa `os.link` (nunca quita el nombre
    original): se prueba instrumentando la lectura de `tmp` (el único punto donde el
    código real toca el disco antes de decidir) para observar, en ESE instante, que
    `ruta` sigue existiendo con su nombre propio -- no sólo antes y después de llamar."""
    ruta = tmp_path / "PAUSA"
    P.poner_pausa(ruta, {"origen": "c4", "motivo": "freno"})

    vistas = []
    real_read_text = Path.read_text

    def read_text_que_observa(self, *a, **kw):
        if self.name.startswith(f".{ruta.name}.quitar-tmp-"):
            vistas.append(ruta.exists())
        return real_read_text(self, *a, **kw)

    monkeypatch.setattr(Path, "read_text", read_text_que_observa)
    borro, datos = P.quitar_pausa_si(ruta, coincide=lambda d: d.get("origen") == "huella")
    assert borro is False
    assert datos["origen"] == "c4"
    assert vistas == [True]  # en el único punto de lectura, `ruta` YA existía con su nombre
    assert ruta.exists()
    assert json.loads(ruta.read_text())["origen"] == "c4"  # intacta, nunca se tocó


def test_quitar_pausa_si_coincide_pero_el_inode_cambio_en_el_medio_no_borra_nada(tmp_path):
    """BLOCK-1 (ronda 9): si -- entre que `quitar_pausa_si` la enlazó y el momento de
    borrar -- `ruta` pasó a ser OTRO archivo (otro inode), no se borra nada, aunque el
    contenido que se leyó al principio coincidiera. El `coincide()` en sí mismo es el
    punto de inyección más real: se dispara DESPUÉS de leer, así que un swap ahí adentro
    reproduce la carrera sin inventar mecanismo."""
    ruta = tmp_path / "PAUSA"
    P.poner_pausa(ruta, {"origen": "huella", "host": "atemai", "mision_id": "m1"})

    def coincide_con_carrera(d):
        if d.get("origen") == "huella":
            ruta.unlink()
            P.poner_pausa(ruta, {"origen": "c5", "motivo": "llegó_en_el_medio"})
            return True
        return False

    borro, datos = P.quitar_pausa_si(ruta, coincide=coincide_con_carrera)
    assert borro is False
    assert datos["origen"] == "huella"  # lo que se había leído
    # La pausa de C5 -- la que ahora ocupa `ruta` -- sigue intacta, no se tocó.
    assert json.loads(ruta.read_text())["origen"] == "c5"


def test_quitar_pausa_si_pausa_nueva_de_c5_sobrevive_si_llega_justo_tras_el_borrado(tmp_path, monkeypatch):
    """MAJOR-1 (ronda 9): "C5 pone su pausa mientras aceptar corre: la pausa de C5
    sobrevive." Una vez que `quitar_pausa_si` YA decidió borrar la suya (inode
    verificado, coincide), si C5 alcanza a poner una pausa nueva justo después del
    unlink -- antes de que termine la limpieza del temporal --, esa pausa nueva no se
    barre: el barrido sólo toca temporales `.PAUSA.quitar-tmp-*`, nunca `PAUSA`."""
    import os as _os
    ruta = tmp_path / "PAUSA"
    P.poner_pausa(ruta, {"origen": "huella", "host": "atemai", "mision_id": "m1"})

    real_unlink = _os.unlink

    def unlink_con_carrera(path, *a, **kw):
        real_unlink(path, *a, **kw)
        if str(path) == str(ruta):
            P.poner_pausa(ruta, {"origen": "c5", "motivo": "justo_despues"})

    monkeypatch.setattr(_os, "unlink", unlink_con_carrera)
    borro, datos = P.quitar_pausa_si(ruta, coincide=lambda d: d.get("origen") == "huella")
    assert borro is True
    assert datos["origen"] == "huella"
    # La pausa nueva de C5 sigue ahí -- el proceso no la tocó, ni la barrió.
    assert json.loads(ruta.read_text())["origen"] == "c5"


def test_quitar_pausa_si_kill_simulado_entre_cada_paso_deja_la_pausa_puesta(tmp_path, monkeypatch):
    """BLOCK-1 (ronda 9): un kill (excepción no atrapable) en cualquier punto entre el
    `os.link` inicial y el `os.unlink(ruta)` final deja la pausa puesta -- nunca a
    medio camino. Se simula reventando cada función que el código real llama, una por
    vez."""
    import os as _os

    class Bomba(Exception):
        pass

    def _con_pausa_fresca():
        ruta = tmp_path / f"PAUSA-{len(list(tmp_path.glob('PAUSA*')))}"
        P.poner_pausa(ruta, {"origen": "huella", "host": "atemai", "mision_id": "m1"})
        return ruta

    # Revienta read_text (lectura de tmp, tras el link).
    ruta1 = _con_pausa_fresca()
    real_read_text = Path.read_text

    def read_text_bomba(self, *a, **kw):
        if self.name.startswith(f".{ruta1.name}.quitar-tmp-"):
            raise Bomba("kill simulado tras el link")
        return real_read_text(self, *a, **kw)

    monkeypatch.setattr(Path, "read_text", read_text_bomba)
    with pytest.raises(Bomba):
        P.quitar_pausa_si(ruta1, coincide=lambda d: True)
    monkeypatch.undo()
    assert ruta1.exists()
    assert json.loads(ruta1.read_text())["origen"] == "huella"

    # Revienta el stat de `ruta` (verificación de inode, justo antes de decidir borrar).
    ruta2 = _con_pausa_fresca()
    real_stat = _os.stat

    def stat_bomba(path, *a, **kw):
        if str(path) == str(ruta2):
            raise Bomba("kill simulado antes de verificar el inode")
        return real_stat(path, *a, **kw)

    monkeypatch.setattr(_os, "stat", stat_bomba)
    with pytest.raises(Bomba):
        P.quitar_pausa_si(ruta2, coincide=lambda d: True)
    monkeypatch.undo()
    assert ruta2.exists()
    assert json.loads(ruta2.read_text())["origen"] == "huella"


def test_el_mutante_except_oserror_en_el_stat_muere(tmp_path, monkeypatch):
    """MINOR-1 (ronda 9): si `os.stat(ruta)` revienta con un `OSError` que NO sea
    `FileNotFoundError` (p. ej. `PermissionError`), la implementación correcta no lo
    traga para seguir de largo y borrar igual -- un mutante `except OSError: pass`
    (que asumiera "no importa, borro de todos modos") tiene que morir: la pausa sigue
    puesta y la excepción se nota (no hay un `True` silencioso)."""
    import os as _os
    ruta = tmp_path / "PAUSA"
    P.poner_pausa(ruta, {"origen": "huella", "host": "atemai", "mision_id": "m1"})

    real_stat = _os.stat

    def stat_permiso_denegado(path, *a, **kw):
        if str(path) == str(ruta):
            raise PermissionError(13, "Permission denied")
        return real_stat(path, *a, **kw)

    monkeypatch.setattr(_os, "stat", stat_permiso_denegado)
    with pytest.raises(PermissionError):
        P.quitar_pausa_si(ruta, coincide=lambda d: True)
    monkeypatch.undo()
    assert ruta.exists()
    assert json.loads(ruta.read_text())["origen"] == "huella"


def test_quitar_pausa_si_sigue_devolviendo_false_none_sin_pausa(tmp_path):
    """Con `os.link` en vez de `os.rename`, el caso "no hay pausa" tiene que seguir
    devolviendo lo mismo que antes -- `FileNotFoundError` en el link, no en el rename."""
    ruta = tmp_path / "PAUSA"
    borro, datos = P.quitar_pausa_si(ruta, coincide=lambda d: True)
    assert borro is False and datos is None
    assert not ruta.exists()


def test_barrer_temporales_huerfanos_solo_toca_temporales_nunca_la_pausa(tmp_path):
    """Barrido (ronda 9): al arrancar `aceptar` y el vigía, se limpian los
    `.PAUSA.quitar-tmp-*` huérfanos que un kill puede haber dejado atrás -- nunca la
    pausa misma."""
    ruta = tmp_path / "PAUSA"
    P.poner_pausa(ruta, {"origen": "huella", "host": "atemai", "mision_id": "m1"})
    huerfano1 = tmp_path / ".PAUSA.quitar-tmp-1234-aaaa"
    huerfano2 = tmp_path / ".PAUSA.quitar-tmp-5678-bbbb"
    huerfano1.write_text(ruta.read_text())
    huerfano2.write_text(ruta.read_text())
    otro_archivo = tmp_path / "otra-cosa.json"
    otro_archivo.write_text("{}")

    P.barrer_temporales_huerfanos(ruta)

    assert ruta.exists()
    assert json.loads(ruta.read_text())["origen"] == "huella"
    assert not huerfano1.exists()
    assert not huerfano2.exists()
    assert otro_archivo.exists()  # no toca nada que no matchee el patrón exacto


def test_barrer_temporales_huerfanos_sin_nada_que_barrer_no_falla(tmp_path):
    ruta = tmp_path / "PAUSA"
    P.barrer_temporales_huerfanos(ruta)  # ni la pausa ni el directorio existen todavía
    assert not ruta.exists()


# --- ronda 10, MINOR ------------------------------------------------------------------

def test_quitar_pausa_si_filenotfound_en_el_unlink_no_escapa_como_traceback(tmp_path, monkeypatch):
    """MINOR (ronda 10): si `ruta` desaparece justo entre el chequeo de inodo y el
    `unlink` de verdad (otro actor, ajeno a esta función, la borró por su cuenta),
    `quitar_pausa_si` NO puede dejar escapar un `FileNotFoundError` sin atrapar --
    devuelve `(False, datos)`, igual que cuando no coincide."""
    ruta = tmp_path / "PAUSA"
    P.poner_pausa(ruta, {"origen": "huella", "host": "atemai", "mision_id": "m1"})

    real_unlink = os.unlink

    def unlink_que_se_adelanta(path, *a, **kw):
        if str(path) == str(ruta):
            real_unlink(path)  # alguien más ya se adelantó y la borró
            raise FileNotFoundError(2, "No such file or directory")
        return real_unlink(path, *a, **kw)

    monkeypatch.setattr(os, "unlink", unlink_que_se_adelanta)
    borro, datos = P.quitar_pausa_si(ruta, coincide=lambda d: True)
    assert borro is False
    assert datos["origen"] == "huella"
    assert not ruta.exists()


def test_barrer_temporales_huerfanos_ignora_directorios_sin_reventar(tmp_path):
    """MINOR (ronda 10): si algo (no debería, pero) deja un DIRECTORIO con el nombre de
    un temporal huérfano, el barrido lo ignora -- no revienta con
    `IsADirectoryError`. Usa `os.lstat` + `S_ISREG`, no `os.unlink` a ciegas."""
    ruta = tmp_path / "PAUSA"
    P.poner_pausa(ruta, {"origen": "huella", "host": "atemai", "mision_id": "m1"})
    directorio_falso = tmp_path / ".PAUSA.quitar-tmp-raro-un-directorio"
    directorio_falso.mkdir()
    (directorio_falso / "adentro").write_text("x")
    huerfano_de_verdad = tmp_path / ".PAUSA.quitar-tmp-normal"
    huerfano_de_verdad.write_text(ruta.read_text())

    borrados = P.barrer_temporales_huerfanos(ruta)

    assert borrados == 1
    assert not huerfano_de_verdad.exists()
    assert directorio_falso.is_dir()  # intacto -- no se tocó
    assert ruta.exists()


def test_barrer_temporales_huerfanos_usa_glob_escape_sobre_el_nombre(tmp_path):
    """MINOR (ronda 10): un nombre de pausa con caracteres especiales de glob (`[`,
    `]`, `*`, `?`) no puede hacer que el patrón matchee de más -- `glob.escape()` sobre
    el NOMBRE (el sufijo `.quitar-tmp-*` sigue siendo un patrón real)."""
    ruta = tmp_path / "PAUSA[1]"
    P.poner_pausa(ruta, {"origen": "huella", "host": "atemai", "mision_id": "m1"})
    # Sin escapar, el patron `.PAUSA[1].quitar-tmp-*` interpretaria `[1]` como una
    # clase de caracteres (matchea "1") -- este archivo NO debe barrerse igual.
    trampa = tmp_path / ".PAUSA1.quitar-tmp-trampa"
    trampa.write_text("{}")
    huerfano_de_verdad = tmp_path / ".PAUSA[1].quitar-tmp-normal"
    huerfano_de_verdad.write_text(ruta.read_text())

    borrados = P.barrer_temporales_huerfanos(ruta)

    assert borrados == 1
    assert not huerfano_de_verdad.exists()
    assert trampa.exists()  # el patrón escapado NO debía tocarlo
    assert ruta.exists()


# --- ronda 10, MAJOR: candado (flock) para serializar quitar/barrer -------------------

def test_candado_se_puede_adquirir_y_soltar_en_secuencia(tmp_path):
    ruta = tmp_path / "PAUSA"
    with P.candado(ruta):
        pass
    with P.candado(ruta):
        pass  # si el primer `with` no soltó bien, este quedaría colgado -- no cuelga


def test_candado_crea_el_archivo_propio_junto_a_la_pausa(tmp_path):
    ruta = tmp_path / "PAUSA"
    with P.candado(ruta):
        assert (tmp_path / ".PAUSA.candado").exists()


def _tarea_sostener_candado(ruta_str, adquirido_evt, soltar_evt):
    """Proceso hijo: adquiere el candado, avisa que lo tiene, y lo sostiene hasta que
    el padre le diga que lo suelte."""
    from jax.ejecutor.contratos import pausa as _P
    with _P.candado(Path(ruta_str)):
        adquirido_evt.set()
        soltar_evt.wait(timeout=10)


def test_candado_bloquea_a_un_segundo_proceso_real_hasta_que_el_primero_libera(tmp_path):
    """MAJOR (ronda 10): mutex de VERDAD entre procesos -- no una simulación en el
    mismo proceso. Un hijo real sostiene el candado; el padre confirma que NO puede
    adquirirlo mientras tanto, y que SÍ puede en cuanto el hijo lo suelta."""
    import multiprocessing as mp

    ruta = tmp_path / "PAUSA"
    ctx = mp.get_context("fork")
    adquirido = ctx.Event()
    soltar = ctx.Event()
    hijo = ctx.Process(target=_tarea_sostener_candado, args=(str(ruta), adquirido, soltar))
    hijo.start()
    try:
        assert adquirido.wait(timeout=5), "el hijo no llegó a adquirir el candado"

        # Mientras el hijo lo sostiene, un intento NO bloqueante del padre tiene que
        # fallar -- confirma que es el MISMO candado de sistema operativo, no uno de
        # otro proceso ni una ilusión de threading.
        ruta_candado = tmp_path / ".PAUSA.candado"
        fd_prueba = os.open(ruta_candado, os.O_CREAT | os.O_RDWR, 0o660)
        try:
            import fcntl
            with pytest.raises(BlockingIOError):
                fcntl.flock(fd_prueba, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(fd_prueba)

        soltar.set()
        hijo.join(timeout=5)
        assert hijo.exitcode == 0

        # Ahora que el hijo terminó y liberó, el padre SÍ puede adquirirlo.
        with P.candado(ruta):
            pass
    finally:
        soltar.set()
        if hijo.is_alive():
            hijo.terminate()
            hijo.join(timeout=5)


def test_barrer_temporales_huerfanos_usa_el_candado_de_la_pausa(tmp_path, monkeypatch):
    """MAJOR (ronda 10): `barrer_temporales_huerfanos` toma el MISMO candado que
    `quitar_pausa_si` vía `aceptar()` -- se verifica que la llamada quede DENTRO de un
    `with P.candado(ruta)` real (instrumentando `fcntl.flock` para ver que se pidió
    ANTES de tocar los temporales)."""
    import fcntl as _fcntl
    ruta = tmp_path / "PAUSA"
    huerfano = tmp_path / ".PAUSA.quitar-tmp-x"
    huerfano.write_text("{}")

    eventos = []
    real_flock = _fcntl.flock

    def flock_que_registra(fd, operacion):
        if operacion == _fcntl.LOCK_EX:
            eventos.append("lock")
        elif operacion == _fcntl.LOCK_UN:
            eventos.append("unlock")
        return real_flock(fd, operacion)

    monkeypatch.setattr(_fcntl, "flock", flock_que_registra)
    P.barrer_temporales_huerfanos(ruta)
    assert eventos == ["lock", "unlock"]
    assert not huerfano.exists()
