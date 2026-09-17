# jax/ejecutor/contratos/cerco.py
"""Cerco de red de la cuenta del Ejecutor (C3). Decisión D-SP1-2.

Medido 2026-09-17 como `axioma`, sin cerco: alcanza LAS MANOS (7777, sin
autenticación, emite tokens de human gate), Ollama (11434), MariaDB (3308) y
jax-platform (8080). Con el cerco, la cuenta sólo abre conexiones a:
- 127.0.0.1 en los puertos locales dados (proxy con carril, upstream del canario);
- el ip:puerto SSH de la máquina local y de cada remota HABILITADA: las que tienen el freno
  remoto (JAX_EJECUTOR_FRENO_REMOTOS, la misma lista que carga ejecutor-freno.service).
  Una máquina del inventario sin contratos remotos (C4 remoto, C6) queda FUERA: la cuenta no
  la alcanza, y el arranque de cada misión lo prueba desde la cuenta (arranque.verificar_alcance).
Todo lo demás se rechaza (TCP con reset, el resto con icmp): falla rápido, no cuelga.

Por qué `meta skuid N jump cuenta` y no `meta skuid != N accept`: un paquete sin socket
completo (RST o ICMP que genera el kernel, SYN-ACK de un request sock) no tiene skuid, y
en nftables una expresión `meta` que no se puede leer NO coincide, ni con `==` ni con
`!=`. Con `!= N accept` esos paquetes de OTRAS cuentas caían en el reject del Ejecutor;
con el salto, sólo entra al cerco lo que seguro es de la cuenta.

`table …` + `delete table …` + definición = recarga idempotente y atómica con `nft -f`.
Sólo IPv4: una IP v6 en el inventario es un error, no un hueco silencioso.

Uso: python -m jax.ejecutor.contratos.cerco <politica.json> <salida.nft>
     (lee JAX_EJECUTOR_CUENTA, JAX_PROXY_CARRIL_PUERTO, JAX_EJECUTOR_CANARIO_PUERTO y
     JAX_EJECUTOR_FRENO_REMOTOS, que tiene que existir aunque esté vacía)
"""
from __future__ import annotations

import ipaddress
import json
import os
import pwd
import sys
from pathlib import Path

from jax.ejecutor.contratos import politica

TABLA = "ejecutor_cerco"


def _puerto(p) -> int:
    if isinstance(p, bool) or not isinstance(p, int) or not 0 < p < 65536:
        raise ValueError("puerto_invalido")
    return p


def renderizar(uid: int, puertos_locales, destinos_ssh) -> str:
    if isinstance(uid, bool) or not isinstance(uid, int) or uid <= 0:
        raise ValueError("uid_invalido")
    locales = sorted({_puerto(p) for p in puertos_locales})
    if not locales:
        raise ValueError("sin_puertos_locales")
    destinos = []
    for ip, puerto in destinos_ssh:
        if not isinstance(ipaddress.ip_address(ip), ipaddress.IPv4Address):
            raise ValueError("solo_ipv4")
        destinos.append((ipaddress.IPv4Address(ip), _puerto(puerto)))
    if not destinos:
        raise ValueError("sin_destinos")
    elementos = ", ".join(f"{ip} . {p}" for ip, p in sorted(set(destinos)))
    return (
        "# Generado por jax/ejecutor/contratos/cerco.py: no se edita a mano.\n"
        f"table inet {TABLA}\n"
        f"delete table inet {TABLA}\n"
        f"table inet {TABLA} {{\n"
        "\tset destinos_ssh {\n"
        "\t\ttype ipv4_addr . inet_service\n"
        f"\t\telements = {{ {elementos} }}\n"
        "\t}\n"
        "\tchain salida {\n"
        "\t\ttype filter hook output priority filter; policy accept;\n"
        f"\t\tmeta skuid {uid} jump cuenta\n"
        "\t}\n"
        "\tchain cuenta {\n"
        "\t\tct state established,related accept\n"
        f"\t\toifname \"lo\" ip daddr 127.0.0.1 tcp dport {{ {', '.join(map(str, locales))} }} accept\n"
        "\t\tip daddr . tcp dport @destinos_ssh accept\n"
        "\t\tmeta l4proto tcp counter reject with tcp reset\n"
        "\t\tcounter reject with icmpx admin-prohibited\n"
        "\t}\n"
        "}\n"
    )


def habilitadas_desde_texto(texto: str) -> frozenset:
    return frozenset(n.strip() for n in texto.split(",") if n.strip())


def desde_politica(p: politica.Politica, uid: int, puertos_locales, habilitadas) -> str:
    remotas = {h.nombre for h in p.hosts if not h.es_local}
    if not set(habilitadas) <= remotas:
        raise ValueError("habilitada_fuera_de_la_politica", tuple(sorted(set(habilitadas) - remotas)))
    return renderizar(uid, puertos_locales,
                      tuple((h.ip, h.puerto) for h in p.hosts if h.es_local or h.nombre in habilitadas))


def principal(argv, env=None) -> int:
    env = os.environ if env is None else env
    doc = json.loads(Path(argv[0]).read_bytes())
    p = politica.validar(doc)
    uid = pwd.getpwnam(env["JAX_EJECUTOR_CUENTA"]).pw_uid
    puertos = (int(env["JAX_PROXY_CARRIL_PUERTO"]), int(env["JAX_EJECUTOR_CANARIO_PUERTO"]))
    habilitadas = habilitadas_desde_texto(env["JAX_EJECUTOR_FRENO_REMOTOS"])
    Path(argv[1]).write_text(desde_politica(p, uid, puertos, habilitadas), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(principal(sys.argv[1:]))
