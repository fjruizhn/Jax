# tests/test_ejecutor_instalar_revertir_huella_sh.py
"""ops/ejecutor/instalar_huella_en_maquina.sh + revertir_huella_en_maquina.sh (ronda 2,
auditoría adversarial 2026-09-22): MAJOR-3 (converge: si la línea de authorized_keys ya
existe pero difiere, la REEMPLAZA) y MAJOR-4 (la reversión VERIFICA -- guion, sudoers y
línea ya no están -- y sale ≠0 si algo sigue; no deja copias con la llave).

Corren los guiones REALES de punta a punta, sin red: `_maquina.sh` resuelve el host de
prueba contra una `politica.json` armada acá (IP `127.0.0.1`, puerto `58291`). FIX CI
(ronda 8, auditoría adversarial 2026-09-22): esto decía que la entrada de
`[127.0.0.1]:58291` "YA está en el `known_hosts` real de quien corre el test" -- ERA
CIERTO sólo en hall9000 (su propio sshd escucha ahí); en el runner de CI no hay ningún
sshd en ese puerto, `ssh-keygen -F` no encontraba nada y el instalador abortaba
(medido en jax#263). Ahora `_arbol_remoto` genera una llave de host PROPIA y efímera
y arma esa línea a mano -- autosuficiente, nunca depende del `known_hosts` real de la
máquina que corre el test. Las rutas absolutas que los guiones hardcodean
(`/usr/local/sbin/...`, `/etc/sudoers.d/...`, el `.ssh` del ADMINISTRADOR -- la cuenta
que de verdad corre el proceso, `ADMIN`/`ADMIN_HOME`, nunca un nombre fijo) se
REDIRIGEN con bwrap (mismo mecanismo que jax/ejecutor/contratos/cuenta_axioma.py y
tests/test_ejecutor_huella_sh.py MAJOR-7) a un árbol de prueba -- nunca al filesystem
real del host. Un `sudo`/`ssh` FALSOS en el PATH (mismo truco que
tests/test_ejecutor_preparar_directorio_misiones.py) hacen que todo corra como el
usuario del test, sin privilegios reales."""
import json
import os
import pwd
import shutil
import subprocess
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
INSTALAR = RAIZ / "ops" / "ejecutor" / "instalar_huella_en_maquina.sh"
REVERTIR = RAIZ / "ops" / "ejecutor" / "revertir_huella_en_maquina.sh"
GUION_HUELLA = RAIZ / "ops" / "ejecutor" / "ejecutor-huella"

REQUIERE_BWRAP = pytest.mark.skipif(shutil.which("bwrap") is None, reason="bwrap no disponible")
#: FIX CI (ronda 8, auditoría adversarial 2026-09-22): `ADMIN = "fruiz"` sólo existe
#: en hall9000 -- en el runner de CI, `getent passwd`/`pwd.getpwnam` de los guiones
#: reales no resuelven nada (medido en jax#263). La cuenta que de verdad corre el
#: proceso existe en cualquier máquina, por definición. `ADMIN_HOME` sale del passwd
#: REAL -- nunca se asume la convención `/home/<usuario>` (el propio guion ya dejó de
#: asumirla, ronda 3, MAJOR-5; este archivo lo seguía haciendo).
ADMIN = pwd.getpwuid(os.getuid()).pw_name
ADMIN_HOME = pwd.getpwnam(ADMIN).pw_dir
ORIGEN_IP_1 = "172.16.20.5"
ORIGEN_IP_2 = "172.16.20.99"


def test_instalar_pasa_bash_menos_n():
    assert subprocess.run(["bash", "-n", str(INSTALAR)]).returncode == 0


def test_revertir_pasa_bash_menos_n():
    assert subprocess.run(["bash", "-n", str(REVERTIR)]).returncode == 0


def test_instalar_no_usa_nombres_fijos_en_tmp():
    """BLOCK-2 (ronda 3, auditoría adversarial 2026-09-22): un nombre FIJO en /tmp,
    escrito por root vía `install`, deja una ventana para que un usuario local de la
    remota plante un symlink ahí ANTES de la subida. `mktemp` (remoto, vía el
    administrador -- `subir_sin_nombre_fijo`) cierra esa ventana; ningún `/tmp/ejecutor-*`
    fijo puede quedar en el guion."""
    assert "mktemp" in INSTALAR.read_text()
    assert "/tmp/ejecutor-" not in INSTALAR.read_text()


def test_revertir_no_usa_nombres_fijos_en_tmp():
    assert "mktemp" in REVERTIR.read_text()
    assert "/tmp/ejecutor-" not in REVERTIR.read_text()


def test_el_mutante_con_nombre_fijo_en_tmp_muere():
    """El mutante que reproduce el defecto real: un `install` a una ruta FIJA en /tmp
    en vez de a la que devuelve `mktemp`."""
    texto_mutado = INSTALAR.read_text().replace(
        'remoto="$(ssh "${SSH_OPC[@]}" -p "$PUERTO" "$JAX_EJECUTOR_ADMIN_USUARIO@$IP" mktemp)"',
        'remoto="/tmp/ejecutor-subida-fija"')
    assert "mktemp" in INSTALAR.read_text()
    assert "mktemp" not in texto_mutado.split("subir_sin_nombre_fijo() {")[1].split("}")[0]


def test_revertir_no_deja_copia_con_la_llave_en_el_texto():
    """MAJOR-4: antes había un `cp ~admin/.ssh/authorized_keys ~admin/.ssh/....` que
    dejaba la línea COMPLETA (command= y clave pública) en un archivo aparte en la
    remota. Ya no -- ningún `cp`/`install` del guion escribe un SEGUNDO archivo con
    `authorized_keys` en el nombre (el docstring SÍ puede mencionar, en prosa, por qué
    se quitó -- lo que no puede haber es el comando que lo hacía)."""
    lineas_de_comando = [l for l in REVERTIR.read_text().splitlines()
                         if not l.lstrip().startswith("#") and ("cp " in l or "install " in l)]
    for linea in lineas_de_comando:
        assert "authorized_keys." not in linea, linea  # ningún SEGUNDO archivo derivado


def test_revertir_verifica_antes_de_declarar_revertida():
    """MAJOR-4: el chequeo de los tres artefactos tiene que estar ANTES del echo final
    -- si estuviera después, o no estuviera, "revertida" se imprimiría sin haber
    comprobado nada."""
    texto = REVERTIR.read_text()
    pos_check = texto.index("test ! -e /usr/local/sbin/ejecutor-huella")
    pos_echo = texto.index('echo "maquina_huella_revertida=')
    assert pos_check < pos_echo


def _doc_politica(*, ip_prueba="127.0.0.1", puerto_prueba=58291):
    from jax.ejecutor.contratos import politica as P

    def _bash(comando):
        return {"tool_name": "Bash", "tool_input": {"command": comando}}

    doc = {
        "version": 1, "generada_at": "2026-09-22T00:00:00+00:00",
        "hosts": [
            {"nombre": "hall9000", "ip": "192.0.2.5", "puerto": 58291, "rol": "hypervisor", "es_local": True},
            {"nombre": "prueba-huella", "ip": ip_prueba, "puerto": puerto_prueba, "rol": "desarrollo",
             "es_local": False},
        ],
        "reglas": [
            {"id": 1, "codigo": "canario_c1", "tipo": "prohibido", "herramientas": "Bash", "campo": "command",
             "patron": "ejecutor-canario-c1", "ambito_hosts": [], "ambito_roles": [], "es_canario": True,
             "ejemplos_coincide": [_bash("echo ejecutor-canario-c1")], "ejemplos_no_coincide": [_bash("echo x")]},
        ],
        "respaldos": {}, "c2_edad_max_s": 86400,
    }
    return P.firmar(doc)


def _bin_falso(tmp_path: Path) -> Path:
    """`ssh`, `sudo`, `install` y `chown` falsos -- todo corre como quien corre el
    test, sin privilegios reales:
    - `ssh <opts...> user@host <comando>` ejecuta el ÚLTIMO argumento con `sh -c`
      LOCAL (ya estamos dentro del bwrap que redirige las rutas absolutas -- no hace
      falta red ni un bwrap anidado);
    - `sudo -n ...`/`sudo -u X ...` quita esas banderas y ejecuta el resto tal cual;
    - `install`/`chown` quitan `-o`/`-g <dueño>` (mismo truco que `_SUDO_FALSO` en
      test_ejecutor_preparar_directorio_misiones.py: el test no corre como root, así
      que no puede cambiar de dueño de verdad -- lo que importa acá es EL CONTENIDO Y
      LA CONVERGENCIA, no la propiedad Unix, que ya prueban otros tests)."""
    bin_ = tmp_path / "bin-falso"; bin_.mkdir(exist_ok=True)
    (bin_ / "ssh").write_text("#!/bin/bash\nultimo=\"${@: -1}\"\nexec sh -c \"$ultimo\"\n")
    (bin_ / "ssh").chmod(0o755)
    (bin_ / "sudo").write_text(
        "#!/bin/bash\n"
        "while [ $# -gt 0 ]; do\n"
        "  case \"$1\" in\n"
        "    -n) shift ;;\n"
        "    -u) shift 2 ;;\n"
        "    *) break ;;\n"
        "  esac\n"
        "done\n"
        "exec \"$@\"\n")
    (bin_ / "sudo").chmod(0o755)
    (bin_ / "install").write_text(
        "#!/bin/bash\n"
        "args=()\n"
        "while [ $# -gt 0 ]; do\n"
        "  case \"$1\" in\n"
        "    -o|-g) shift 2 ;;\n"
        "    *) args+=(\"$1\"); shift ;;\n"
        "  esac\n"
        "done\n"
        "exec /usr/bin/install \"${args[@]}\"\n")
    (bin_ / "install").chmod(0o755)
    (bin_ / "chown").write_text("#!/bin/sh\nexit 0\n")
    (bin_ / "chown").chmod(0o755)
    # `scp -q -o ... -P <puerto> <origen> user@ip:<destino>` -- mismo filesystem (todo
    # corre dentro del MISMO bwrap), así que una copia local directa alcanza: se quita
    # el prefijo `user@ip:` del último argumento y se copia con `cp`.
    (bin_ / "scp").write_text(
        "#!/bin/bash\n"
        "args=()\n"
        "while [ $# -gt 0 ]; do\n"
        "  case \"$1\" in\n"
        "    -q) shift ;;\n"
        "    -o|-P) shift 2 ;;\n"
        "    *) args+=(\"$1\"); shift ;;\n"
        "  esac\n"
        "done\n"
        "origen=\"${args[0]}\"; destino=\"${args[1]#*:}\"\n"
        "exec cp \"$origen\" \"$destino\"\n")
    (bin_ / "scp").chmod(0o755)
    # `visudo` real necesita leer el /etc/sudoers REAL del host (root) -- acá sólo
    # importa que el guion LLAME a visudo con la sintaxis correcta y reaccione a su
    # código de salida, no la validación de sudoers en sí (eso lo prueba MAJOR-5 por
    # separado, contra sudo real, en un contenedor -- ver huella.py). `-cf <archivo>`:
    # valida que el archivo exista y no esté vacío; `-c` (sin archivo): siempre ok.
    (bin_ / "visudo").write_text(
        "#!/bin/bash\n"
        "if [ \"$1\" = -cf ]; then test -s \"$2\"; exit $?; fi\n"
        "exit 0\n")
    (bin_ / "visudo").chmod(0o755)
    return bin_


def _argv_bwrap(binds: dict) -> list:
    """Las tres rutas que se redirigen (`/usr/local/sbin`, `/etc/sudoers.d`,
    `~admin/.ssh`) YA EXISTEN como directorios en cualquier máquina de este inventario
    -- `--bind` directo alcanza, sin `--tmpfs` sobre el padre (un `--tmpfs
    /home/fruiz` de más se probó y TAPABA el resto del home real, incluido el propio
    checkout del repo bajo `/home/fruiz/worktrees/...` -- nunca enmascarar más que la
    ruta exacta que hace falta)."""
    argv = ["bwrap", "--dev-bind", "/", "/", "--die-with-parent"]
    for real, prueba in binds.items():
        argv += ["--bind", str(prueba), real]
    return argv


def _entorno_de_prueba(tmp_path, politica_ruta, *, origen_ip=ORIGEN_IP_1):
    bin_falso = _bin_falso(tmp_path)
    return {
        "PATH": f"{bin_falso}:/usr/sbin:/usr/bin:/sbin:/bin", "HOME": os.environ.get("HOME", "/root"),
        "JAX_EJECUTOR_POLITICA": str(politica_ruta), "JAX_EJECUTOR_ADMIN_USUARIO": ADMIN,
        "JAX_EJECUTOR_HUELLA_LLAVE": str(tmp_path / "id_ejecutor_huella"),
        "JAX_EJECUTOR_HUELLA_KNOWN_HOSTS": str(tmp_path / "known_hosts_huella"),
        "JAX_EJECUTOR_HUELLA_ORIGEN_IP": origen_ip,
    }


def _arbol_remoto(tmp_path: Path) -> dict:
    """Árbol de prueba que bwrap monta sobre las rutas absolutas reales: sbin,
    sudoers.d y el DIRECTORIO `.ssh` del administrador -- SIEMPRE arrancan con lo
    mínimo, como una máquina recién dada de alta. Se monta el DIRECTORIO entero, no
    sólo `authorized_keys`: `install` reemplaza el archivo con un rename atómico, y
    reemplazar el propio punto de montaje de bwrap con eso da `Device or resource
    busy` (probado) -- montando el directorio, `install` reemplaza un archivo DENTRO,
    que sí es libre.

    FIX CI (ronda 8, auditoría adversarial 2026-09-22): esto copiaba la entrada de
    `[127.0.0.1]:58291` del `known_hosts` REAL de quien corre el test -- en hall9000
    existe (su propio sshd escucha ahí, y fruiz ya se conectó a sí mismo alguna vez);
    en el runner de CI no hay NINGÚN sshd en ese puerto, así que esa entrada nunca
    existió -- `ssh-keygen -F` del paso 2 del instalador no encontraba nada, `test -s`
    fallaba, y el instalador entero abortaba (`set -euo pipefail`), medido en jax#263.
    Ahora se genera una llave de host PROPIA, efímera, y se arma la línea de
    `known_hosts` a mano -- autosuficiente, sin depender de que exista un sshd real
    en ese puerto ni de qué `known_hosts` traiga la máquina que corre el test."""
    sbin = tmp_path / "remote-sbin"; sbin.mkdir(exist_ok=True)
    sudoers_d = tmp_path / "remote-sudoersd"; sudoers_d.mkdir(exist_ok=True)
    ejecutor_huella_dir = tmp_path / "remote-ejecutor-huella"; ejecutor_huella_dir.mkdir(exist_ok=True)
    ssh_admin = tmp_path / "remote-ssh-admin"; ssh_admin.mkdir(exist_ok=True)
    (ssh_admin / "authorized_keys").write_text("ssh-ed25519 AAAAotra otra-llave-de-fruiz\n")
    llave_host_falsa = tmp_path / "host_key_de_prueba"
    subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(llave_host_falsa)],
                   check=True, capture_output=True)
    tipo, clave = llave_host_falsa.with_suffix(".pub").read_text().split()[:2]
    (ssh_admin / "known_hosts").write_text(f"[127.0.0.1]:58291 {tipo} {clave}\n")
    return {
        "/usr/local/sbin": sbin,
        "/etc/sudoers.d": sudoers_d,
        "/etc/ejecutor-huella": ejecutor_huella_dir,
        f"{ADMIN_HOME}/.ssh": ssh_admin,
    }


@REQUIERE_BWRAP
def test_instalar_dos_veces_con_el_mismo_origen_es_idempotente(tmp_path):
    politica_ruta = tmp_path / "politica.json"
    politica_ruta.write_text(json.dumps(_doc_politica()))
    binds = _arbol_remoto(tmp_path)
    env = _entorno_de_prueba(tmp_path, politica_ruta)
    base = _argv_bwrap(binds) + ["--", "env"] + [f"{k}={v}" for k, v in env.items()] + \
        ["bash", str(INSTALAR), "prueba-huella"]

    r1 = subprocess.run(base, capture_output=True, timeout=60, cwd=str(RAIZ))
    assert r1.returncode == 0, r1.stderr.decode()
    contenido1 = ((binds[f"{ADMIN_HOME}/.ssh"] / "authorized_keys")).read_text()
    assert "ejecutor-huella-servicio" in contenido1

    r2 = subprocess.run(base, capture_output=True, timeout=60, cwd=str(RAIZ))
    assert r2.returncode == 0, r2.stderr.decode()
    contenido2 = ((binds[f"{ADMIN_HOME}/.ssh"] / "authorized_keys")).read_text()
    assert contenido1 == contenido2  # idempotente: nada cambió en la segunda corrida
    assert contenido2.count("ejecutor-huella-servicio") == 1  # una sola entrada, no dos


@REQUIERE_BWRAP
def test_instalar_con_origen_ip_distinto_converge_major3(tmp_path):
    politica_ruta = tmp_path / "politica.json"
    politica_ruta.write_text(json.dumps(_doc_politica()))
    binds = _arbol_remoto(tmp_path)

    env1 = _entorno_de_prueba(tmp_path, politica_ruta, origen_ip=ORIGEN_IP_1)
    argv1 = _argv_bwrap(binds) + ["--", "env"] + [f"{k}={v}" for k, v in env1.items()] + \
        ["bash", str(INSTALAR), "prueba-huella"]
    r1 = subprocess.run(argv1, capture_output=True, timeout=60, cwd=str(RAIZ))
    assert r1.returncode == 0, r1.stderr.decode()
    contenido1 = ((binds[f"{ADMIN_HOME}/.ssh"] / "authorized_keys")).read_text()
    assert f'from="{ORIGEN_IP_1}"' in contenido1

    env2 = _entorno_de_prueba(tmp_path, politica_ruta, origen_ip=ORIGEN_IP_2)
    argv2 = _argv_bwrap(binds) + ["--", "env"] + [f"{k}={v}" for k, v in env2.items()] + \
        ["bash", str(INSTALAR), "prueba-huella"]
    r2 = subprocess.run(argv2, capture_output=True, timeout=60, cwd=str(RAIZ))
    assert r2.returncode == 0, r2.stderr.decode()
    contenido2 = ((binds[f"{ADMIN_HOME}/.ssh"] / "authorized_keys")).read_text()

    assert contenido2.count("ejecutor-huella-servicio") == 1  # MAJOR-3: reemplazada, no duplicada
    assert f'from="{ORIGEN_IP_1}"' not in contenido2  # la vieja se fue
    assert f'from="{ORIGEN_IP_2}"' in contenido2  # quedó la nueva
    assert "otra-llave-de-fruiz" in contenido2  # las demás líneas, intactas


@REQUIERE_BWRAP
def test_revertir_funciona_aunque_la_llave_se_haya_regenerado_major3(tmp_path):
    """MAJOR-3 (ronda 3, auditoría adversarial 2026-09-22): el mutante real que se
    reprodujo -- instalar (deja la línea marcada con la clave VIEJA), REGENERAR el par
    local (simula una rotación -- `id_ejecutor_huella`/`.pub` cambian de contenido SIN
    volver a instalar), revertir. Si revertir filtrara/verificara por la clave pública
    ACTUAL (la nueva, que nunca llegó a la remota), no encontraría nada que quitar Y la
    verificación tampoco vería nada (busca la clave nueva, no la vieja que sigue ahí) --
    saldría `verificado=true` con la línea VIEJA todavía autorizada. Filtrando por
    MARCA, revertir no necesita que la clave coincida con nada."""
    politica_ruta = tmp_path / "politica.json"
    politica_ruta.write_text(json.dumps(_doc_politica()))
    binds = _arbol_remoto(tmp_path)
    env = _entorno_de_prueba(tmp_path, politica_ruta)
    argv_base = _argv_bwrap(binds) + ["--", "env"] + [f"{k}={v}" for k, v in env.items()]

    r_instalar = subprocess.run(argv_base + ["bash", str(INSTALAR), "prueba-huella"],
                                capture_output=True, timeout=60, cwd=str(RAIZ))
    assert r_instalar.returncode == 0, r_instalar.stderr.decode()
    contenido_antes = (binds[f"{ADMIN_HOME}/.ssh"] / "authorized_keys").read_text()
    assert "ejecutor-huella-servicio" in contenido_antes
    clave_vieja = [l for l in contenido_antes.splitlines() if "ejecutor-huella-servicio" in l][0].split()[-2]

    # "Regenerar el par": la llave local cambia de contenido SIN volver a instalar --
    # la remota se queda con la línea VIEJA, como pasaría con una rotación real.
    llave = Path(env["JAX_EJECUTOR_HUELLA_LLAVE"])
    llave.unlink(missing_ok=True)
    Path(f"{llave}.pub").unlink(missing_ok=True)
    subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", "rotada",
                    "-f", str(llave)], check=True)
    clave_nueva = Path(f"{llave}.pub").read_text().split()[1]
    assert clave_nueva != clave_vieja  # la rotación de verdad cambió la clave

    r_revertir = subprocess.run(argv_base + ["bash", str(REVERTIR), "prueba-huella"],
                                capture_output=True, timeout=60, cwd=str(RAIZ))
    assert r_revertir.returncode == 0, r_revertir.stderr.decode()
    assert b"verificado=true" in r_revertir.stdout

    contenido_despues = (binds[f"{ADMIN_HOME}/.ssh"] / "authorized_keys").read_text()
    assert "ejecutor-huella-servicio" not in contenido_despues  # la línea VIEJA se fue
    assert clave_vieja not in contenido_despues


@REQUIERE_BWRAP
def test_instalar_y_revertir_no_deja_nada_atras_major4(tmp_path):
    politica_ruta = tmp_path / "politica.json"
    politica_ruta.write_text(json.dumps(_doc_politica()))
    binds = _arbol_remoto(tmp_path)
    env = _entorno_de_prueba(tmp_path, politica_ruta)
    argv_base = _argv_bwrap(binds) + ["--", "env"] + [f"{k}={v}" for k, v in env.items()]

    r_instalar = subprocess.run(argv_base + ["bash", str(INSTALAR), "prueba-huella"],
                                capture_output=True, timeout=60, cwd=str(RAIZ))
    assert r_instalar.returncode == 0, r_instalar.stderr.decode()
    assert (binds["/usr/local/sbin"] / "ejecutor-huella").exists()
    assert (binds["/etc/sudoers.d"] / "50-ejecutor-huella").exists()
    assert "ejecutor-huella-servicio" in ((binds[f"{ADMIN_HOME}/.ssh"] / "authorized_keys")).read_text()

    r_revertir = subprocess.run(argv_base + ["bash", str(REVERTIR), "prueba-huella"],
                                capture_output=True, timeout=60, cwd=str(RAIZ))
    assert r_revertir.returncode == 0, r_revertir.stderr.decode()
    assert b"verificado=true" in r_revertir.stdout

    assert not (binds["/usr/local/sbin"] / "ejecutor-huella").exists()
    assert not (binds["/etc/sudoers.d"] / "50-ejecutor-huella").exists()
    contenido_final = ((binds[f"{ADMIN_HOME}/.ssh"] / "authorized_keys")).read_text()
    assert "ejecutor-huella-servicio" not in contenido_final
    assert "otra-llave-de-fruiz" in contenido_final  # lo que no era del arreglo, se queda
    # MAJOR-4: ninguna copia con la llave adentro en ningún lado del árbol "remoto".
    for ruta in binds.values():
        if ruta.is_dir():
            for archivo in ruta.rglob("*"):
                if archivo.is_file():
                    assert "ejecutor-huella-servicio" not in archivo.read_text(errors="ignore")
    # El propio tmp_path del test (donde el instalador deja sus temporales -- $ETAPA es
    # un mktemp APARTE, no bajo tmp_path, así que esto sólo barre lo que este test armó).
    # `id_ejecutor_huella(.pub)` es la llave PROPIA del servicio -- persiste a propósito
    # entre instalaciones/reversiones (es compartida por TODO el inventario, ver el
    # docstring de revertir_huella_en_maquina.sh); no es una "copia" que MAJOR-4 prohíba.
    excluidos = {"authorized_keys", "politica.json", "id_ejecutor_huella", "id_ejecutor_huella.pub"}
    for archivo in tmp_path.rglob("*"):
        if archivo.is_file() and archivo.name not in excluidos:
            assert "ejecutor-huella-servicio" not in archivo.read_text(errors="ignore"), archivo


@REQUIERE_BWRAP
def test_revertir_sale_distinto_de_cero_si_algo_queda_major4(tmp_path):
    """MAJOR-4, el mutante que pide matar: si el paso 2 (sudoers) NO se ejecutara de
    verdad (por ejemplo, un `rm` que fallara en silencio con `|| true`), la verificación
    final tiene que atraparlo -- se simula dejando el sudoers.d con permisos que
    impiden borrarlo desde dentro del sandbox (root-like pero sin permiso real de
    escritura en ese archivo puntual: se lo hace INMUTABLE con chattr si está
    disponible; si no, se verifica el mismo efecto marcando el directorio
    de sudoers.d SIN permiso de escritura)."""
    politica_ruta = tmp_path / "politica.json"
    politica_ruta.write_text(json.dumps(_doc_politica()))
    binds = _arbol_remoto(tmp_path)
    env = _entorno_de_prueba(tmp_path, politica_ruta)
    argv_base = _argv_bwrap(binds) + ["--", "env"] + [f"{k}={v}" for k, v in env.items()]

    r_instalar = subprocess.run(argv_base + ["bash", str(INSTALAR), "prueba-huella"],
                                capture_output=True, timeout=60, cwd=str(RAIZ))
    assert r_instalar.returncode == 0, r_instalar.stderr.decode()

    # Se bloquea la ESCRITURA del directorio sudoers.d "remoto" -- el `rm -f` del paso 2
    # del revertir fallará de verdad (nuestro sudo falso no da privilegios reales), y
    # `set -e` corta el guion ANTES de imprimir "revertida" -- MAJOR-4 en los hechos.
    sudoers_d_real = binds["/etc/sudoers.d"]
    sudoers_d_real.chmod(0o500)
    try:
        r_revertir = subprocess.run(argv_base + ["bash", str(REVERTIR), "prueba-huella"],
                                    capture_output=True, timeout=60, cwd=str(RAIZ))
    finally:
        sudoers_d_real.chmod(0o700)

    assert r_revertir.returncode != 0
    assert b"verificado=true" not in r_revertir.stdout
    assert (sudoers_d_real / "50-ejecutor-huella").exists()  # sigue ahí -- el guion NO mintió


@REQUIERE_BWRAP
def test_instalar_no_inyecta_comandos_via_tmpdir_hostil_major_d(tmp_path):
    """MAJOR-D (ronda 4, auditoría adversarial 2026-09-22): `$TMP_SCRIPT_REMOTO`,
    `$TMP_SUDOERS_REMOTO` y `$TMP_AUTH_REMOTO` salen de `mktemp` EN LA REMOTA -- si el
    `TMPDIR` de esa sesión trajera un `;` (una variable de entorno hostil, un perfil de
    shell tocado), el nombre resultante, incrustado SIN COMILLAS en el `corre "..."`,
    partía la línea en DOS comandos para el shell remoto. Se arma un TMPDIR real (un
    directorio de verdad, válido en el filesystem -- `;`/espacios SÍ son caracteres de
    archivo legales en Linux, sólo `/` y NUL no lo son) cuyo nombre, si se interpreta
    como shell, ejecutaría un `touch` delator -- y se confirma que NUNCA aparece."""
    politica_ruta = tmp_path / "politica.json"
    politica_ruta.write_text(json.dumps(_doc_politica()))
    binds = _arbol_remoto(tmp_path)
    env = _entorno_de_prueba(tmp_path, politica_ruta)

    marcador = tmp_path / "INYECTADO_RONDA4"
    tmpdir_hostil = tmp_path / "hostil; touch INYECTADO_RONDA4; echo x"
    tmpdir_hostil.mkdir()
    env["TMPDIR"] = str(tmpdir_hostil)

    argv_base = _argv_bwrap(binds) + ["--", "env"] + [f"{k}={v}" for k, v in env.items()]
    r_instalar = subprocess.run(argv_base + ["bash", str(INSTALAR), "prueba-huella"],
                                capture_output=True, timeout=60, cwd=str(tmp_path))

    assert not marcador.exists(), "el TMPDIR hostil ejecutó el touch inyectado -- MAJOR-D sin cerrar"
    assert r_instalar.returncode == 0, r_instalar.stderr.decode()
    contenido = (binds[f"{ADMIN_HOME}/.ssh"] / "authorized_keys").read_text()
    assert "ejecutor-huella-servicio" in contenido  # el instalador igual terminó bien


# --- MINOR (ronda 5, auditoría adversarial 2026-09-22): `$ADMIN_LOCAL` viajaba SIN
# escapar en el instalador y en la reversión (`-o $ADMIN_LOCAL`, `~$ADMIN_LOCAL/...`).
# Citarlo con `%q` (`$ADMIN_Q`) es necesario -- pero `~$ADMIN_Q` YA NO expandiría el
# home (la tilde sólo expande delante de un nombre de usuario SIN comillas): por eso el
# HOME ahora sale de `getent passwd` en la remota, nunca de `~usuario`. -----------------

def test_instalar_y_revertir_ya_no_usan_tilde_para_el_home_del_administrador():
    """Ni `~$ADMIN_LOCAL` ni `~$ADMIN_Q` (la forma citada, que de todos modos ya no
    expandiría nada) aparecen en ninguno de los dos guiones -- el HOME sale de
    `getent passwd`, siempre."""
    for ruta in (INSTALAR, REVERTIR):
        lineas_de_codigo = [l for l in ruta.read_text().splitlines() if not l.lstrip().startswith("#")]
        codigo = "\n".join(lineas_de_codigo)
        assert "~$ADMIN_LOCAL" not in codigo, ruta
        assert "~$ADMIN_Q" not in codigo, ruta
        assert "getent passwd $ADMIN_Q" in codigo, ruta


def test_instalar_y_revertir_citan_admin_local_con_printf_q():
    """`$ADMIN_Q` -- la forma escapada -- es la que viaja dentro de los `corre "..."`;
    el `$ADMIN_LOCAL` crudo sólo puede aparecer en la asignación que lo define."""
    for ruta in (INSTALAR, REVERTIR):
        texto = ruta.read_text()
        assert 'ADMIN_Q="$(printf %q "$ADMIN_LOCAL")"' in texto, ruta


@REQUIERE_BWRAP
def test_instalar_funciona_con_un_home_resuelto_via_getent_con_espacio(tmp_path):
    """El HOME de una cuenta puede legítimamente traer un espacio (a diferencia del
    NOMBRE de usuario, que el sistema restringe) -- `~usuario` nunca podría resolver
    eso de otra forma que como una convención fija; `getent passwd` sí, porque lee lo
    que el passwd REALMENTE dice. Se fuerza con un `getent` falso que antepone al PATH
    real, y el resto de la cadena (instalar, verificar convergencia) tiene que seguir
    funcionando con ese HOME."""
    home_con_espacio = tmp_path / "remote-home con espacio"
    ssh_admin = home_con_espacio / ".ssh"
    ssh_admin.mkdir(parents=True)
    (ssh_admin / "authorized_keys").write_text("ssh-ed25519 AAAAotra otra-llave-de-fruiz\n")

    politica_ruta = tmp_path / "politica.json"
    politica_ruta.write_text(json.dumps(_doc_politica()))
    binds = _arbol_remoto(tmp_path)
    del binds[f"{ADMIN_HOME}/.ssh"]
    binds[str(ssh_admin)] = ssh_admin  # bind idéntico -- ya está en su lugar real bajo tmp_path

    bin_falso = _bin_falso(tmp_path)
    getent_falso = bin_falso / "getent"
    getent_falso.write_text(
        "#!/bin/bash\n"
        f'if [ "$1" = passwd ] && [ "$2" = "{ADMIN}" ]; then\n'
        f'  echo "{ADMIN}:x:1000:1000:prueba:{home_con_espacio}:/bin/bash"\n'
        "  exit 0\n"
        "fi\n"
        'exec /usr/bin/getent "$@"\n')
    getent_falso.chmod(0o755)

    env = _entorno_de_prueba(tmp_path, politica_ruta)
    env["PATH"] = f"{bin_falso}:/usr/sbin:/usr/bin:/sbin:/bin"
    argv_base = _argv_bwrap(binds) + ["--", "env"] + [f"{k}={v}" for k, v in env.items()]

    r = subprocess.run(argv_base + ["bash", str(INSTALAR), "prueba-huella"],
                       capture_output=True, timeout=60, cwd=str(RAIZ))
    assert r.returncode == 0, r.stderr.decode()
    contenido = (ssh_admin / "authorized_keys").read_text()
    assert "ejecutor-huella-servicio" in contenido
