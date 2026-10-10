# Plan — validar transición exacta de definición de piso

## Objetivo

Desbloquear el grant exacto de retiro de piso tras un endurecimiento legítimo del piso en #392, preservando autorización limitada y fallo cerrado.

## Reglas del contrato

Para el merge de introducción declarado, después de verificar su SHA y el orden de sus padres:

1. La definición de `floor_key` en el padre 1 debe ser distinta del objetivo (puede no existir o ser una definición previa).
2. La definición en el padre 2 debe ser exactamente igual a `floor_definition` del grant.
3. La definición del merge debe ser exactamente igual a `floor_definition` del grant.
4. El grant debe seguir verificando SHA canónico, rama/repositorio/evento, base vigente, PR dedicado, diff de una sola ruta para el grant, retiro único y huella exacta del diff del revert.

La transición no concede permiso si el piso no cambió, si el lado propuesto o el merge final no coinciden con el objetivo, ni si cambia cualquier otro binding.

## Ejecución

1. Añadir pruebas para transición válida desde ausente y desde una definición previa; añadir negativas para definición anterior ya igual al objetivo, propuesta incorrecta y merge final incorrecto.
2. Ejecutar la suite focal para observar rojo.
3. Implementar helper pequeño de transición exacta; usarlo en la validación del merge de introducción.
4. Ejecutar suite focal, análisis de diff y los gates del workflow; actualizar el piso exacto medido por CI.
5. Obtener auditoría Tier 3 independiente sobre SHA final; guardar hallazgos y resolverlos.
6. Abrir PR de soporte y seguir integración/guard oficiales.
7. Recalcular la autorización #390 sobre el merge real y candidato #387; no reutilizar hashes simulados sin recomputar.

## Riesgos / controles

- Riesgo: admitir que un grant antiguo retire otro estado del piso. Control: hashes exactos de merge, padres, objetivo y definición vigente más diff completo.
- Riesgo: history shallow o refs faltantes. Control: los checks existentes siguen fallando cerrado.
- Riesgo: modificar el retiro fuera del PR autorizado. Control: mantener el grant como merge de dos padres inmediatamente anterior al PR objetivo y exigir diff exacto.
- Riesgo de carrera de master: repetir el preflight justo antes de integrar y ejecutar post-merge guard inmediatamente después.

## Estado

Implementación local en curso; las 88 pruebas focales pasan. Falta auditoría, CI remota y los flujos de PR dependientes.
