/using-superpowers
/ruflo

# Misión: reconocimiento — migración bridge (Hestia) → lookingglass (aaPanel)

## Contexto

Objetivo final (NO de esta misión, de una futura): migrar los 5 usuarios
del Mac mini "bridge" (172.16.20.20, Hestia 1.9.6) a la VM "lookingglass"
en hall9000 (aaPanel PRO ya instalado), reemplazando el rol que hoy cumple
Hestia. El bridge NUNCA se toca ni se apaga durante todo el proceso — sigue
sirviendo producción real hasta que Fernando decida el cutover manual,
después de verificar exhaustivamente que todo migró bien. Esta misión es
SOLO reconocimiento — no copiar, no instalar, no modificar nada todavía.
El pipeline real de copia se diseña en una misión posterior, con este
inventario como base.

## 1. Confirmar acceso y ubicación real de lookingglass

```bash
# Desde hall9000
virsh list --all   # o el comando KVM equivalente que ya se usa para las VMs
                    # conocidas (dev/atemai-net .11, prod/server-rich-hn .10)
```

Confirmar la IP real de la VM lookingglass, su usuario SSH, puerto, y que
aaPanel efectivamente está corriendo ahí (no asumir — verificar con
`systemctl status` o el puerto de aaPanel real).

## 2. Inventario completo del bridge (READ-ONLY, sin escribir nada)

Conectar por SSH a 172.16.20.20 (bridge, Hestia) y generar un inventario
completo, SOLO lectura:

```bash
ssh -p 58291 fruiz@172.16.20.20  # confirmar puerto/usuario real primero
```

Para cada uno de los 5 usuarios de Hestia, documentar:
- Dominios/subdominios alojados, con su document root
- Bases de datos (motor, tamaño, usuario/permisos — NO extraer contraseñas
  en texto plano al reporte, solo confirmar que existen y su tamaño)
- Cuentas de correo por dominio (cuántas, tamaño aproximado de buzones)
- Cron jobs configurados por usuario
- Certificados SSL activos (Let's Encrypt u otros, fecha de expiración)
- Versión de PHP/Node/lo que corresponda por sitio
- Cualquier configuración no estándar (redirects custom, .htaccess
  especiales, reverse proxies internos)

```bash
# Comandos de Hestia útiles para esto (ajustar a los reales disponibles):
v-list-users
v-list-web-domains <user>
v-list-mail-domains <user>
v-list-databases <user>
v-list-cron-jobs <user>
```

## 3. Inventario de lookingglass (aaPanel) — estado actual

Confirmar qué tiene hoy lookingglass ya configurado (si algo), qué versión
de aaPanel, PHP/Node/MariaDB disponibles, y si hay algo que ya esté en
conflicto con lo que habría que migrar (ej. un puerto ya ocupado, un
dominio ya apuntado a otra cosa ahí).

## 4. Identificar el gap Hestia → aaPanel

Hestia y aaPanel NO tienen un exportador/importador oficial entre sí.
Documentar, para cada elemento del inventario del paso 2, cómo se
correspondería en términos de aaPanel (ej. "vhost de Hestia X" → "site de
aaPanel Y", "usuario de mail de Hestia" → "cuenta de mail de aaPanel").
No implementar nada — solo mapear conceptualmente y señalar qué partes no
tienen equivalente directo (si las hay).

## 5. Explícitamente NO hacer en esta misión

- NO copiar ningún archivo, base de datos, ni configuración.
- NO instalar ni configurar nada en lookingglass.
- NO tocar el bridge de ninguna forma que no sea lectura pura.
- NO extraer contraseñas ni secretos en texto plano al reporte.

## 6. Reporte final

Inventario completo (los 5 usuarios, sus dominios/DBs/mail/cron/SSL),
confirmación del acceso y estado real de lookingglass, el mapeo conceptual
Hestia→aaPanel con los gaps identificados, y una propuesta de orden de
migración (cuál usuario primero como piloto, y por qué — probablemente el
más simple, no el más grande) para que Fernando la revise antes de diseñar
el pipeline real de copia.
