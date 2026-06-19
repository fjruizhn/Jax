faceta: hyde

# Fix: eliminar campo dedication de respuestas API de Jacobs

## Problema
POST /jacobs/plan devuelve un campo "dedication" en la respuesta JSON que no debe ser visible externamente. El campo existe en algún lugar del código de ~/jax/jacobs/ pero grep no lo encuentra claramente.

## Tarea
1. Encontrar EXACTAMENTE dónde se genera el campo "dedication" en la respuesta de POST /jacobs/plan
2. Eliminarlo de todas las respuestas de la API (plan, pipeline, get_pipeline, events)
3. Verificar que NO aparece en ninguna respuesta HTTP
4. Los comentarios en headers de archivos .py se quedan — son privados

## Verificación obligatoria
curl -s -X POST http://127.0.0.1:7777/jacobs/plan \
-H "Content-Type: application/json" \
-d '{"name":"test","objective":"test","invoked_by":"Fernando","mode":"dry_run"}' \
| python3 -m json.tool | grep -i "dedic"

Debe devolver: (vacío — ninguna línea)

Reiniciar servicio después del fix:
sudo systemctl restart jax-las-manos

Escribir resultado en ~/jax/missions/jacobs-fix-dedication_result.md
