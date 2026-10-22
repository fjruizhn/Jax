-- Permisos del usuario de los topes ${usuario_topes} (debe existir; se crea en el alta del host). Es OTRO usuario
-- que el de la bitacora (002): la bitacora sigue siendo de solo INSERT. Este lee, inserta la fila y actualiza el
-- contador; no borra, no altera, no toca la bitacora.
GRANT SELECT, INSERT, UPDATE ON `${base}`.faro_topes TO `${usuario_topes}`@`${host_usuario}`
