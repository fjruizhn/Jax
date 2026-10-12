# Traspaso — Faro Fase 0.5, S-1

Fecha: 2026-10-12

## Punto actual

- Base de esta rama: `ecfe07f3c2760ed43f53e7d1e858dad8e864e7e1`.
- Implementado un primer contrato puro, todavía sin publicación ni activación:
  `perfiles_credenciales`, `proxy_credenciales`, `puente_http` y el montaje por
  inodo de los dos archivos del proxy en la jaula.
- El proxy escucha únicamente Unix, autentica primero `SO_PEERCRED` y luego el
  token de ejecución en comparación de tiempo constante. El perfil fija método,
  ruta, modelo, destino y cabecera de credencial; el pedido no puede cambiarlos.
- Codex no dispone aún de adaptador S-1 y falla explícitamente con `NO_S1`.

## Evidencia local

```
python3 -m pytest tests/test_faro_proxy_credenciales.py tests/test_faro_puente_http.py -q
6 passed
```

## Pendiente antes de auditoría/publicación

1. Añadir los negativos restantes del contrato S-1: uid distinto real, token de
   otra ejecución, CONNECT/Upgrade/redirect, cierre SSE y socket reemplazado.
2. Cablear configuración y ciclo de vida del servicio sin introducir secretos en
   argv, entorno de hijo o jaula.
3. Integrar el stop con el freno y el scope R-2 ya aprobado; no inventar una
   segunda autoridad de cancelación.
4. Medir el binario real dentro de bwrap y el aislamiento de red cuando exista el
   lanzador de motor de 0.7. No activar proveedores ni usar credenciales humanas.
5. Auditoría independiente Sol, CI exacta y coordinación de floors corresponden
   a la sesión principal.
