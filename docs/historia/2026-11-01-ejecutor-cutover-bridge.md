# Ejecutor tras el cutover de bridge

**Fecha:** 2026-11-01. **Decidió:** Fernando. **Hizo:** Hyde, en hall9000, con ventana abierta.

## Qué se hizo

- `bridge` (.20) y `ejecutor-prueba` salieron del inventario. Todo el hosting quedó en `prod` (.10), con aaPanel en lugar de Hestia. Allí vive AdGuard principal, con réplica en Sésamo.
- jax-platform#256: migración `ejecutor_retirar_bridge_v1`. Los 21 puntos de restauración de bridge pasaron a `ejecutor_punto_restauracion_retirado`, y `apt_full_upgrade_bridge` pasó a `apt_full_upgrade_prod`.
- jax#413: inventario, contexto y examen. El inventario y examen de septiembre quedan congelados en `scripts/ejecutor_fase0/historico/` para reproducir el corpus de U3.
- Aplicado en vivo: base, `.env`, `runtime.env`, `migrate.env`, cerco, freno, proxy, plataforma, contratos y contexto. `/opt/ejecutor/lib` quedó igual que el par activo.
- Copias previas: `/srv/almacen/respaldos-sueltos/ejecutor-retirar-bridge-20261101`.

## Comparación de `probar_arranque`

| Corrida | Antes | Después |
|---|---|---|
| `/srv/jax-prod/jax`, jaxsvc | `false`: c6 de bridge y ejecutor-prueba | `false`: `instalado_distinto_del_repo`, esperado |
| Par activo, jaxsvc, sin máquinas | no medido | `true`, 9,2 s |
| Par activo, jaxsvc, hall9000 prod atemai | no medido | `true`, 107,8 s |
| Par activo, jaxweb | no medido | `false`: `PermissionError` en c1, c3 y c6 |

La política conserva 24 reglas con los mismos patrones. Cambian hosts, respaldos, el ámbito de la regla 3 y los ejemplos de seis reglas más.

## Lo que sigue abierto

- `backend/ejecutor/misiones.py` lanza el runner como `jaxweb`, que no alcanza política, llave ni interruptor. Una misión desde la plataforma falla cerrada, y ya era así antes de este cambio. Darle esos accesos es una decisión de autoridad.
- No se corrió ninguna misión real: sin `ejecutor-prueba` no hay máquina sin datos de clientes. Fernando decidió no recrearla.
- Activar un par no toca `/opt/ejecutor/lib`. Un par que cambie los `INSTALABLES` vuelve a cerrar la compuerta.
- C1 no ataja en `prod` comandos como `docker compose down -v`, `/etc/init.d/bt stop`, `nginx -s stop` o `mv /www/vmail`.

## Lecciones

- `probar_arranque` compara contra el árbol del código que corre: probar desde `/srv/jax-prod/jax` no dice nada del par que usa la plataforma.
- Retirar una máquina exige mover también sus reglas y respaldos: `politica.validar` falla cerrado si quedan.

## Alternativas descartadas

- Par nuevo desde master: traía además 14 merges, casi todos de Oficina, fuera del encargo.
- Recrear `ejecutor-prueba`: Fernando la dio por retirada.
