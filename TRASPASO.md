# Traspaso Codex · JAX Faro · 2026-10-08

## Objetivo

Continuar el cierre de la cadena Faro y del ledger solicitado por Fernando. #377 ya está
integrado. #379 debe quedar verde y auditado antes de #381; no desplegar.

## Estado comprobado

- `master` estaba en `709237f901e8ec8840f6b2b39a1a2ce88682674a` al auditar #379.
- #379 SHA `3371ecd0b926354e591b5a0bffe5fc96fac3f609`: Tier 3 APROBADO; comentarios
  de auditoría publicados. CI falló en `authority-ledger-mariadb/integration`: 5 de 8
  contenedores agotaron 60 s esperando `init process done`, aunque el log mostró la
  instancia temporal en puerto 0 y la final lista en 3306.
- Corrección local en `tests/policy/test_authority_ledger_storage_mariadb.py`: esperar el
  marcador de init o `ready for connections` asociado a `port: 3306`, y luego confirmar
  autenticación más `SELECT 1`. Prueba unitaria focal, py_compile y diff-check pasan.
  Piso actualizado 8 -> 9. Este cambio aún no se ha auditado ni publicado.
- #373: Fernando decidió alinear `FormaLimites` al schema: solo `NINGUNA`, `CANTIDAD`,
  `MONTO`; `OBLIGATING` exige exactamente cantidad o monto, con campos coherentes.
  El árbol local del worktree `jax-faro-f11-providers` mide 130 pruebas de providers y
  791 Identity Shadow (cero skipped); cambios aún no publicados.

## Pendiente

1. Publicar y volver a auditar el nuevo SHA de #379; esperar CI completa verde, ejecutar
   `verify-source`, `verify-integration` y `post-merge-guard`; entonces mergear.
2. En `/home/fruiz/wt/jax-faro-f11-providers`, revisar diff y pruebas, publicar la decisión
   de #373 con piso 791. Después de merge de #379, reapilar el branch sobre el nuevo master,
   volver a medir pisos/CI y pedir Tier 3 sobre SHA exacto.
3. Continuar #371/#376 y ramas apiladas #375/#378; #381 permanece apilado sobre #379.
4. No integrar `policy/**`; la excepción reserva esas rutas a Fernando. No desplegar.

## Próximo comando

```sh
cd /home/fruiz/wt/jax-loader-seal && git status --short --branch
```

Luego revisar también `git status` en `/home/fruiz/wt/jax-faro-f11-providers` y continuar
desde los cambios locales descritos arriba.
