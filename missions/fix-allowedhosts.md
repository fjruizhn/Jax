/using-superpowers
/ruflo

# Misión: fix allowedHosts + verificación visual final de facet_models

## Contexto

Se agregó `allowedHosts` a `vite.config.js` para que Vite acepte requests con
Host header `axioma-ia.io` (proxeadas desde Nginx en atemai-net). El valor
quedó corrupto por un problema de copy-paste en terminal: la entrada de
`www.axioma-ia.io` se guardó literalmente como el string
`[www.axioma-ia.io](https://www.axioma-ia.io)` (con corchetes y paréntesis
de Markdown incluidos), en vez del dominio limpio.

## 1. Arreglar `vite.config.js`

Archivo: `/home/fruiz/jax-platform/frontend/vite.config.js`

La entrada `www.axioma-ia.io` es innecesaria — Nginx ya redirige
`www.axioma-ia.io` → `axioma-ia.io` con 301 antes de que la request llegue a
Vite (confirmado en `/www/server/panel/vhost/nginx/axioma-ia.io.conf` en
atemai-net, 172.16.20.11). Simplemente dejar:

```js
allowedHosts: ['axioma-ia.io'],
```

Backup antes de tocar, verificar el archivo completo después del cambio.

## 2. Reiniciar el servicio

```bash
sudo systemctl restart jax-platform-frontend.service
```
(Esto requiere sudo interactivo — usar el prefijo `!` y pedirle a Fernando
que lo confirme, igual que hicimos antes en esta misma sesión.)

## 3. Verificación en cadena — NO reportar éxito sin cada uno de estos pasos

```bash
systemctl is-active jax-platform-frontend.service
curl -s https://axioma-ia.io/ | head -5
```
Debe devolver HTML real (`<!doctype html>...`), NO "Blocked request".

## 4. Verificación visual del feature completo (el objetivo real de esta misión)

La misión anterior (`facet-models.md`) implementó un dropdown de selección
de modelo por faceta en `/admin/api-keys`, pero nunca se confirmó visualmente
porque Nginx estaba sirviendo un build estático viejo en vez de proxear en
vivo a Vite — ese es el bug que esta misión corrige en los pasos 1-3.

Con el proxy ya arreglado, usa las herramientas de browser disponibles (o si
no tenés acceso a navegador en esta sesión, un curl detallado + grep sobre
el HTML/JS servido) para confirmar en `https://axioma-ia.io/admin/api-keys`:

- El `<select>` de modelo aparece en cada fila de la tabla, poblado con las
  opciones correctas por faceta (ej. thot con gpt-5.5/gpt-5.6-sol/terra/luna).
- El panel expandible "···" lista los modelos alternativos con botón de borrado.
- El modal de confirmación aritmética aparece al intentar borrar un modelo.

Si tenés acceso real a browser (Chrome DevTools MCP o similar), navegá a la
URL, tomá screenshot, y confirmá visualmente — no solo por HTML crudo.

## 5. Reporte final

Confirmar explícito: ¿el dropdown se ve y funciona, sí o no? Si no, describir
exactamente qué se ve en su lugar (error de consola, elemento ausente, etc.)
en vez de asumir causa.
