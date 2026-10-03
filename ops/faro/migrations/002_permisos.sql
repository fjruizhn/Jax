-- Permisos del usuario del Puerto ${usuario} (debe existir; se crea en el alta del host): SOLO INSERT.
-- Sin SELECT, UPDATE, DELETE ni DROP: una bitacora que quien escribe puede editar no es evidencia.
-- La lectura y la verificacion las hace otro usuario (auditor), con SELECT.
GRANT INSERT ON `${base}`.faro_bitacora TO `${usuario}`@`${host_usuario}`
