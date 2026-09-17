# tests/test_ejecutor_probar_c4_conteo_remoto.py
"""C4 cuenta los procesos de la cuenta en la máquina remota para probar que el freno los mató.
Ese conteo tiene que ir como ROOT: en bridge (2026-09-17, al habilitar las máquinas de clientes)
`/proc` está montado con `hidepid=invisible` y el administrador ve CERO procesos de la cuenta
aunque haya doce vivos. Sin `sudo -n`, la prueba mide ceguera, no el freno: el escenario nunca
parecía arrancar (fail-closed que salvó el falso verde) y, de haber pasado, el conteo después
del freno habría dado 0 sin haber matado nada."""
import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ejecutor_contratos" / "probar_c4.py"


def _cargar():
    spec = importlib.util.spec_from_file_location("probar_c4_conteo", _SCRIPT)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


class _Remoto:
    ip, puerto, nombre, es_local = "192.0.2.20", 58291, "bridge", False


def test_el_conteo_remoto_corre_como_root():
    argv = _cargar().argv_conteo_remoto(_Remoto(), "axioma", "fruiz")
    orden = argv[-1]
    assert argv[0] == "ssh" and "fruiz@192.0.2.20" in argv
    assert orden.startswith("sudo -n ") and "ps -u axioma" in orden and "| wc -l" in orden


def test_el_conteo_remoto_cita_la_cuenta_y_el_administrador():
    argv = _cargar().argv_conteo_remoto(_Remoto(), "cuenta rara", "admin raro")
    assert "'cuenta rara'" in argv[-1] and "admin raro@192.0.2.20" in argv


def test_el_gestor_de_sesion_no_cuenta_como_trabajo_pero_si_al_final():
    """En bridge (hidepid, 2026-09-17) el freno remoto mata el trabajo y deja unos segundos el
    `systemd --user` que logind vuelve a levantar; muere solo (medido: a los ~60 s, user@ failed).
    Ese gestor no ejecuta el trabajo del Ejecutor, pero puede lanzar unidades: no se ignora, se
    exige que también se haya ido al cerrar la ronda."""
    m = _cargar()
    salida = ("12 /usr/lib/systemd/systemd --user\n"
              "13 (sd-pam)\n"
              "44 sleep 90\n")
    trabajo, total = m.clasificar_procesos(salida)
    assert (trabajo, total) == (["44"], 3)
    solo_gestor = "12 /usr/lib/systemd/systemd --user\n13 (sd-pam)\n"
    assert m.clasificar_procesos(solo_gestor) == ([], 2)
    assert m.clasificar_procesos("") == ([], 0)
