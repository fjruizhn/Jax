# MISIÓN: Migración completa bridge (.20) → VM prod (.10)
**PROTOCOLO: Six Impossible Things · Neverland · Junio 2026**

---

## LOS SEIS EN ACCIÓN — ROLES PARA ESTA MISIÓN

| Miembro | Rol en esta misión |
|---|---|
| **FERNANDO** | Autoridad máxima. Da el GO final. Nadie ejecuta en prod sin él. |
| **JAX LOCAL** | Orquestador. Coordina el flujo entre facetas. |
| **HIPATIA** | Fase de diagnóstico. Recolecta hechos, no supone. |
| **JEKYLL** | Recibe el diagnóstico. Valida lógica. Construye el script. |
| **THOT** | Recibe el script. Abogado del diablo. Señala riesgos. Propone mejoras. |
| **HYDE** | Recibe el script aprobado por Thot. Lo ejecuta en producción. |
| **CLAUDE** | Revisa resultado final. Valida objetivo cumplido. Actualiza docs. |

**FLUJO OBLIGATORIO — en este orden, sin saltarse pasos:**
```
HIPATIA diagnóstica → JEKYLL construye → THOT critica → [GO Fernando] → HYDE ejecuta
```

Las primeras tres fases se ejecutan SIN necesidad de GO — trabajad en paralelo y dejad el script listo. HYDE espera el GO explícito de Fernando antes de tocar producción.

---

## INFRAESTRUCTURA CONFIRMADA

**SSH sin password verificado:**
- hall9000 → prod (.10) ✅
- hall9000 → atemai (.11) ✅
- Hipatia (.11) → prod (.10) ✅
- Hipatia (.11) → bridge (.20) ✅

**ORIGEN — Mac mini bridge**
- IP: 172.16.20.20, puerto 58291, user fruiz
- OS: Ubuntu 24.04.4, Hestia 1.9.6
- Stack mail: Exim + Dovecot
- MariaDB 11.4 local

**Usuarios a migrar:**

| Usuario | Dominio(s) | Tamaño | Contenido |
|---|---|---|---|
| richhn | rich-hn.com, nextcloud.rich-hn.com | ~29GB | Nextcloud 21GB, mail |
| monhagro | monhagro.com | ~31MB | web + mail |
| melipaola | sol-lex.com | ~12MB | web + mail |
| fynamicshn | fynamics-hn.com | ~80MB | web + mail |
| bdihn | bdihn.com | ~5.5GB | 12 cuentas mail |
| gescorphn | gescorp-hn.com | ~7.4GB | WP + Joomla, 13 mail, 2 DBs |

**DESTINO — VM prod server-rich-hn**
- IP: 172.16.20.10, puerto 58291, user fruiz
- OS: Ubuntu 24.04, aaPanel PRO
- Stack: Nginx, PHP 8.3, MariaDB 11.4, Redis
- Mail: Postfix + Dovecot + rspamd
- postfixadmin.db SQLite en `/www/vmail/postfixadmin.db`
- Roundcube operativo en webmail.rich-hn.com
- Restic backup activo (3AM, R2 offsite)

---

## PROTOCOLO SIX IMPOSSIBLE THINGS — REGLAS ABSOLUTAS

1. **No suponer nunca.** El que supone se equivoca.
2. **Saber no cuesta nada** — pregúntale al que de verdad sabe.
3. **Mañana es el día que el fracasado tiene más que hacer.**
4. **Regla del carpintero:** medir dos veces, cortar una.
5. **Backup antes de tocar.** Siempre. Sin excepción.
6. **Pasos pequeños verificables.** `set -e` en todo script. Si falla, para y reporta.

---

## FASE 1 — HIPATIA: DIAGNÓSTICO

Ejecutar en el **BRIDGE (.20)** y reportar output completo:

```bash
# Usuarios y dominios Hestia
sudo /usr/local/hestia/bin/v-list-users list
for u in richhn monhagro melipaola fynamicshn bdihn gescorphn; do
  echo "=== $u ==="
  sudo /usr/local/hestia/bin/v-list-web-domains $u 2>/dev/null
  sudo /usr/local/hestia/bin/v-list-mail-domains $u 2>/dev/null
  sudo /usr/local/hestia/bin/v-list-databases $u 2>/dev/null
done

# Tamaños reales en disco
du -sh /home/*/
du -sh /home/*/mail/ 2>/dev/null
du -sh /home/*/web/ 2>/dev/null

# Bases de datos
sudo mariadb -e "SHOW DATABASES;"

# Hashes de contraseñas mail (formato para verificar compatibilidad bcrypt)
sudo head -1 /home/richhn/mail/rich-hn.com/*/passwd 2>/dev/null
sudo head -1 /home/bdihn/mail/bdihn.com/*/passwd 2>/dev/null

# Espacio disponible
df -h /home
```

Ejecutar en el **DESTINO (.10)** y reportar output completo:

```bash
# Stack activo
sudo systemctl status nginx postfix dovecot rspamd --no-pager | grep -E "Active|●"

# Cuentas mail existentes en postfixadmin.db
sudo sqlite3 /www/vmail/postfixadmin.db \
  "SELECT username, domain FROM mailbox ORDER BY domain, username;"

# Schema completo de postfixadmin.db (para que Jekyll sepa exactamente qué campos hay)
sudo sqlite3 /www/vmail/postfixadmin.db ".schema"

# Dominios ya registrados en Nginx
ls /www/server/panel/vhost/nginx/

# Bases de datos existentes
sudo mariadb -e "SHOW DATABASES;"

# Usuario vmail y permisos
id vmail 2>/dev/null || echo "usuario vmail no existe"
ls -la /www/vmail/

# Espacio disponible para recibir ~43GB
df -h
```

---

## FASE 2 — JEKYLL: CONSTRUIR EL SCRIPT

Jekyll recibe el output completo de Hipatia y construye:
`/usr/local/sbin/migrate-bridge.sh`

**El script se ejecuta DESDE el destino (.10), hace pull del bridge (.20) vía rsync+SSH.**

Estructura obligatoria con `set -e` y verificación entre cada fase:

```
FASE 2.1 — Snapshot Restic manual en .10 (verificar éxito antes de continuar)
FASE 2.2 — rsync de /home/* del bridge a /tmp/bridge-staging/ en destino
FASE 2.3 — Crear usuarios del sistema en destino (useradd, directorios)
FASE 2.4 — Registrar dominios web en aaPanel (generar vhosts Nginx)
FASE 2.5 — Importar bases de datos (mysqldump origen → crear DB+usuario+importar destino)
FASE 2.6 — Migrar cuentas mail a postfixadmin.db + rsync Maildirs a /www/vmail/
FASE 2.7 — Regenerar SSL Let's Encrypt para todos los dominios
FASE 2.8 — Verificación final: curl HTTP 200 + IMAP test por dominio
```

**Consideraciones críticas:**

**WEB:**
- Hestia webroot: `/home/[user]/web/[domain]/public_html/`
- aaPanel webroot: `/www/wwwroot/[domain]/`
- Nginx vhosts destino: `/www/server/panel/vhost/nginx/`

**MAIL — la parte más delicada:**
- Hashes Hestia son bcrypt `$2y$` — son compatibles con Dovecot, insertar directo en postfixadmin.db sin rehashear
- Maildir origen: `/home/[user]/mail/[domain]/[user]/Maildir/`
- Maildir destino: `/www/vmail/[domain]/[user]/`
- postfixadmin.db schema: `mailbox(username, password, name, maildir, quota, domain, active)`
- Respetar permisos del usuario vmail en `/www/vmail/`

**BASES DE DATOS:**
- gescorphn tiene 2 DBs (WordPress + Joomla) — exportar/importar ambas
- Patrón nombres Hestia: `[user]_[dbname]`
- Recrear usuarios MySQL con mismos permisos

**NEXTCLOUD (richhn — caso especial):**
- rsync completo del webroot de nextcloud.rich-hn.com
- Actualizar `/config/config.php`: `datadirectory` y `dbpassword` si cambian
- Exportar DB nextcloud → importar → actualizar config.php
- Verificar que Nextcloud responde antes de dar por bueno

---

## FASE 3 — THOT: ANÁLISIS CRÍTICO DEL SCRIPT

Thot recibe el script de Jekyll y responde estructuradamente:

1. **SUPUESTOS OCULTOS:** qué está asumiendo el script sin verificar explícitamente
2. **RIESGOS CRÍTICOS:** qué puede fallar en producción con impacto en clientes
3. **CASOS BORDE:** escenarios que Jekyll no contempló
4. **MEJORAS CONCRETAS:** cambios específicos con justificación técnica
5. **VEREDICTO:** ¿el script está listo para ejecutarse o necesita revisión?

Thot NO aprueba nada que tenga fallas críticas sin resolver.
Thot SÍ reconoce explícitamente cuando algo está bien hecho.
El script mejorado con las observaciones de Thot es lo que se entrega a Hyde.

---

## FASE 4 — HYDE: EJECUCIÓN (REQUIERE GO DE FERNANDO)

Hyde recibe el script final (post-Thot, con mejoras incorporadas).
Hyde ejecuta **SOLO** después del GO explícito de Fernando.
Hyde reporta el output de cada fase antes de continuar a la siguiente.
Si alguna fase falla → Hyde se detiene, reporta, y espera instrucciones.
**Nunca continúa ciego.**

---

## RESULTADO ESPERADO AL TERMINAR

- ✅ 6 usuarios operativos en aaPanel prod (.10)
- ✅ 7 dominios web respondiendo HTTP 200 con SSL válido
- ✅ 42+ cuentas mail accesibles vía IMAP/Roundcube
- ✅ 3+ bases de datos importadas correctamente
- ✅ Nextcloud operativo en nextcloud.rich-hn.com
- ✅ Mac mini bridge (.20) listo para apagar
- ✅ Claude valida resultado final y actualiza CONTEXT.md

---

## PRIMER PASO AHORA

**Hipatia ejecuta el diagnóstico en ambos servidores y reporta.**
Jekyll y Thot esperan su output.
Hyde espera el GO de Fernando.
Nadie supone nada. El que supone se equivoca.
