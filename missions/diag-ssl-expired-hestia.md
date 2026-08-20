/using-superpowers
/ruflo

# Misión: diagnosticar por qué sol-lex.com y bdihn.com tienen SSL vencido en Hestia

## Contexto

Del recon `recon-bridge-to-lookingglass` (2026-07-18,
`missions/recon-bridge-to-lookingglass_result.md`): en el bridge (Hestia,
<IP interna, ver /etc/jax/.env>) los certificados de `sol-lex.com` (usuario melipaola, vencido
desde 2025-05-03) y `bdihn.com` (usuario bdihn, vencido desde 2025-05-22)
llevan más de un año vencidos. No forma parte de la migración a
lookingglass — es un problema de producción real, independiente.

Esta misión es SOLO diagnóstico — no renovar, no reiniciar nada, no tocar
configuración. El fix (forzar renovación o lo que corresponda) se decide
después de entender la causa raíz.

## 1. Confirmar el estado real (no confiar en el handshake TLS solo)

```bash
V=/usr/local/hestia/bin
sudo -n $V/v-list-web-domain melipaola sol-lex.com
sudo -n $V/v-list-web-domain bdihn bdihn.com
ls -la /home/melipaola/conf/web/sol-lex.com/ssl/ 2>&1
ls -la /home/bdihn/conf/web/bdihn.com/ssl/ 2>&1
```

## 2. Logs del renovador de Let's Encrypt (Hestia usa v-update-user-domain-ssl / cron propio)

```bash
crontab -l -u admin 2>&1 | grep -i ssl
cat /var/log/hestia/nginx-error.log 2>&1 | tail -50
find /usr/local/hestia/log -iname "*ssl*" -o -iname "*le*" 2>/dev/null
# Hestia loguea intentos de renovación en algún lado — encontrar dónde y
# leer los últimos intentos para estos dos dominios específicamente
grep -ril "sol-lex.com\|bdihn.com" /var/log/ /usr/local/hestia/log/ 2>/dev/null | head -20
```

## 3. Verificar resolución DNS y accesibilidad HTTP (Let's Encrypt necesita
   HTTP-01 o DNS-01 funcionando)

```bash
# Desde el bridge y desde hall9000, comparar
dig +short sol-lex.com
dig +short bdihn.com
curl -sI http://sol-lex.com/.well-known/acme-challenge/test 2>&1
curl -sI http://bdihn.com/.well-known/acme-challenge/test 2>&1
```

Confirmar que el dominio resuelve a la IP correcta del bridge (<IP interna, ver /etc/jax/.env>)
y que el puerto 80 responde para el path de ACME challenge — si el DNS
apunta a otro lado o el firewall bloquea 80, ahí está la causa.

## 4. Config SSL actual en Hestia para ambos dominios

```bash
sudo -n $V/v-list-web-domain melipaola sol-lex.com | grep -i ssl
sudo -n $V/v-list-web-domain bdihn bdihn.com | grep -i ssl
# ¿SSL está en modo "letsencrypt" auto, o es un cert manual/custom que
# alguien subió una vez y nunca se pensó en renovar?
```

## 5. Comparar contra un dominio que SÍ renueva bien (ej. monhagro.com,
   válido hasta 2026-09-09) para ver la diferencia de configuración

```bash
sudo -n $V/v-list-web-domain monhagro monhagro.com | grep -i ssl
```

## Explícitamente NO hacer en esta misión

- NO correr ningún comando de renovación (`v-update-user-domain-ssl`, certbot,
  acme.sh, etc.)
- NO reiniciar nginx ni ningún servicio
- NO modificar configuración de ningún dominio

## Reporte final

Causa raíz identificada (o las hipótesis descartadas + la que queda en pie
si no se puede confirmar 100%), evidencia concreta (logs, output de
comandos, no suposición), y una recomendación de fix — pero NO ejecutarlo,
eso es una misión aparte con confirmación de Fernando.
