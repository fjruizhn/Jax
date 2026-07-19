# Resultado — reconocimiento bridge (Hestia) → lookingglass (aaPanel)

Todo lo que sigue fue obtenido por lectura pura vía SSH. No se copió, instaló
ni modificó nada en ninguno de los dos hosts.

## Paso 1 — Confirmación de acceso y ubicación real de lookingglass

`virsh list --all` en hall9000 solo mostraba `prod` y `dev` — no existe una VM
llamada literalmente `lookingglass`. Fernando confirmó: **`lookingglass` es la
VM `prod`, todavía sin renombrar.**

- IP: 172.16.20.10
- Hostname interno: `server-rich-hn`
- SSH: puerto 58291, usuario `fruiz` (misma clave que hall9000/GitHub,
  `hall9000-jax-github`)
- Confirmado con `systemctl` (no solo puertos abiertos):
  `btpanel.service`, `AdGuardHome.service`, `mysqld.service`,
  `BT-FirewallServices.service` — todos `active running`.

## Paso 2 — Inventario del bridge (172.16.20.20, Hestia)

Acceso inicial denegado (publickey) hasta que Fernando agregó la clave
pública al bridge. Confirmado después: hostname `server.rich-hn.com`,
Hestia CLI vive en `/usr/local/hestia/bin/` (no está en PATH), requiere
`sudo -n` (passwordless configurado para `fruiz` en este host) para leer
`hestia.conf` y correr los `v-list-*` sin permission-denied.

**6 usuarios reales**, no 5 como asumía el brief de la misión (más el
usuario `admin` del panel, que no cuenta como cliente):

| Usuario | Dominio(s) web | PHP | Mail (cuentas / disco) | DBs | Cron | SSL (vence) |
|---|---|---|---|---|---|---|
| richhn | rich-hn.com | 8.0 | 9 cuentas, 251MB | — | ninguno | 2026-09-09 |
| | server.rich-hn.com | (sin backend PHP propio) | | | | 2026-09-24 |
| | nextcloud.rich-hn.com | 8.2 | | `richhn_nextcloud` (53MB) | | 2026-09-07 |
| monhagro | monhagro.com | — | 4 cuentas, 31MB | — | ninguno | 2026-09-09 |
| melipaola | sol-lex.com | 8.0 | 2 cuentas, 8MB | — | ninguno | **VENCIDO 2025-05-03** |
| fynamicshn | fynamics-hn.com | 8.0 | 2 cuentas, 78MB | — | ninguno | 2026-08-26 |
| bdihn | bdihn.com | 8.0 | 12 cuentas, **5.66GB** | — | ninguno | **VENCIDO 2025-05-22** |
| gescorphn | gescorp-hn.com | 8.0 | 13 cuentas, **7.16GB** | `gescorph_website` (9MB), `gescorph_wp393` (22MB) | ninguno | 2026-08-26 |

**Hallazgo fuera de alcance de esta misión, pero real y urgente:** los
certificados SSL de `sol-lex.com` y `bdihn.com` están vencidos hace más de
un año (verificado con handshake TLS directo — `openssl s_client` contra
`172.16.20.20:443`, no asumido de un listado). No es parte de la migración,
pero Fernando debería saberlo independientemente de si/cuándo se hace el
cutover.

Ningún usuario tiene cron jobs configurados — ese ítem del inventario queda
vacío para los 6.

## Paso 3 — Estado actual de lookingglass (aaPanel, VM prod)

- PHP: 8.3 (cubre todo lo usado en origen, incluido Nextcloud en 8.2)
- Node: v24.16.0
- MariaDB: 11.4.4
- Panel aaPanel: puerto 888
- **Puertos de mail ya escuchando**: 25, 110, 143, 993, 995 — no se pudo
  confirmar qué proceso los tiene (`sudo -n` no es passwordless en este
  host, a diferencia del bridge; no se pidió contraseña ni se intentó
  bypass). **Pendiente de verificar antes de tocar mail en la migración** —
  podría ser el propio módulo de mail de aaPanel ya activo, o algo de
  AdGuard, o un residuo. No inventado, dato faltante declarado como tal.
- No se pudo listar vhosts existentes en
  `/www/server/panel/vhost/nginx/` (permission denied sin sudo interactivo).
  No se detectó ningún dominio de los 6 del bridge respondiendo ya en
  lookingglass (no se verificó exhaustivamente, solo por ausencia de vhosts
  visibles).

## Paso 4 — Mapeo conceptual Hestia → aaPanel

| Elemento Hestia | Equivalente aaPanel | Gap |
|---|---|---|
| Web domain (vhost + PHP backend) | Site (agregar sitio + elegir versión PHP) | Ninguno — PHP 8.3 destino cubre 8.0/8.2 origen |
| Database (MariaDB) | Database (MariaDB) | Ninguno — mismo motor, dump/restore directo |
| Mail domain + cuentas | Módulo de mail de aaPanel (o stack aparte) | **El gap más grande.** Hestia usa Exim+Dovecot con su propio esquema; aaPanel no trae un mail server tan integrado por defecto. Con ~13GB de buzones concentrados en 2 usuarios (bdihn, gescorphn), es el punto de mayor esfuerzo/riesgo de toda la migración. |
| Cron jobs | Crontab de aaPanel | N/A — no hay cron jobs que migrar en ningún usuario |
| SSL (Let's Encrypt) | SSL de aaPanel (también Let's Encrypt integrado) | Ninguno — más simple re-emitir en destino que copiar certs |
| Reverse proxies / .htaccess custom | — | No se detectó ninguno en el reconocimiento (no se buscó exhaustivamente config nginx custom por permisos) |

Hestia y aaPanel no tienen exportador/importador oficial entre sí — confirmado,
no hay atajo automatizado.

## Propuesta de orden de migración (piloto)

**`monhagro` primero.** Es el segundo más simple (después de `melipaola`),
pero `melipaola` tiene el certificado ya vencido — usarlo de piloto mezclaría
la migración con un problema de SSL preexistente, complicación innecesaria
para probar el flujo. `monhagro` prueba web + mail (4 cuentas chicas) sin
DB, sin los volúmenes grandes de Nextcloud (`richhn`) ni los buzones pesados
de `bdihn`/`gescorphn` (que deberían ir al final, una vez validado el
proceso de mail con algo chico primero).

## Explícitamente NO hecho (por diseño de la misión)

Ningún archivo copiado, ninguna DB volcada, nada instalado ni configurado en
lookingglass, el bridge no se tocó de ninguna forma que no sea lectura. No
se extrajeron contraseñas ni secretos en texto plano — solo conteos,
tamaños y metadata.

## Pendientes antes de diseñar el pipeline real de copia (misión futura)

1. Confirmar qué proceso tiene abiertos los puertos de mail en lookingglass.
2. Decidir la estrategia de mail server para aaPanel (su módulo nativo vs
   stack separado) — bloquea el diseño del pipeline para 5 de los 6
   usuarios (todos menos los que no tienen mail, que son ninguno acá).
3. Obtener acceso con más privilegio en lookingglass (o la contraseña de
   sudo) para listar vhosts existentes y descartar conflictos de dominio.
4. Resolver los 2 certificados vencidos (sol-lex.com, bdihn.com) —
   independiente de la migración, pero visible ahora.
