# Namespace de proyectos Office · OP343 / OP345

2026-10-20 · Decisión de producto: Fernando autorizó comenzar Virtual Office multiusuario, con administración/invitación, superusuario Fernando con acceso total al producto, administrador general, administrador de proyecto y colaborador. Contrato técnico de Sol, coordinado por Codex. Primera porción del desarrollo; no habilita una API ni activa clientes por sí sola.

## Contrato

JAX conserva proyectos y sus invariantes. JAX Platform conserva usuarios, roles y autorización de oficina. No se amplía la autoridad de usuarios sobre proyectos clásicos ni Faro.

`ProjectNamespace`: CLASSIC, OFFICE_CONFIDENTIAL. Columna persistida `jax_project_scope.authorization_namespace VARCHAR(32) NOT NULL DEFAULT 'CLASSIC'`, CHECK y índice `(authorization_namespace,tenant_id,status,project_id)`. Migración idempotente preserva filas existentes.

Resolvedores clásicos (proyecto, lectura, mutación, consultas de lista/detalle y administración existente) solo admiten CLASSIC. OFFICE_CONFIDENTIAL se deniega antes de heredar OWNER/roles administrativos. Todas las superficies que consumen el resolvedor quedan cubiertas; rastrear consumidores al implementar y verificar los caminos de memoria/RAG/herramientas que lean por proyecto. Contenido de oficina nunca entra en TENANT_SHARED/PROJECT_SHARED clásicos.

No se inventan membresías ni se suplanta el tenant del actor. Nueva creación bajo `ProjectAuthorityAdmin.create_namespaced_project(*,actor_user_id:int,target_tenant_id:int,namespace:ProjectNamespace,name:str,description:str|None,idempotency_key:str,authorizer:NamespaceMutationAuthorizer)->CreatedProject`.

`NamespaceMutationAuthorizer` es un Protocol con `async authorize_in_transaction(cursor,*,actor_user_id:int,target_tenant_id:int,namespace:ProjectNamespace,operation:str)->None`. La autorización ocurre en el mismo cursor/transacción/bloqueo que crea el proyecto, con operación CREATE_OFFICE_PROJECT. El protocolo es una integración de código confiable del servidor; ningún campo HTTP, booleano o token del navegador sustituye la consulta de autoridad. La creación rechaza CLASSIC; ese sigue usando la API existente.

Refactorizar solo la inserción privada de filas para reutilizar las invariantes existentes de `projects`/scope; no duplicar las reglas clásicas. Escrituras de proyecto y scope confidencial son atómicas: jamás se publica primero un proyecto CLASSIC. El método existente escribe CLASSIC explícitamente. Mantener retornos y excepciones vigentes donde corresponda.

Tabla `jax_namespace_project_creation_request`: namespace, idempotency_key, actor_user_id, target_tenant_id, request_digest, project_id, created_at. PK(namespace,idempotency_key), FKs e índice(target_tenant_id,actor_user_id,created_at). Digest incluye actor, tenant destino, namespace, nombre y descripción normalizados. Toda repetición vuelve a autorizar dentro de transacción; exacta devuelve mismo proyecto, cualquier diferencia de actor/tenant/payload da conflicto sin revelar proyecto ajeno. Concurrencia misma clave crea una fila/proyecto. Autorización fallida revierte todo.

La autorización concreta OFFICE_SUPERUSER vive en JXP y se implementa después: explícita, verificada en BD, sin límite singleton artificial, sin deducción por nombre/email o rol clásico. Solo cruza organizaciones del producto Office; no concede lectura de Axioma clásico. Esta porción no provisiona usuarios ni otorga permisos reales.

## Rendimiento y seguridad

- Índices y EXPLAIN sobre consulta real de lista por namespace/tenant; cero nuevas consultas de tabla creciente sin índice.
- Sin caché de permisos nueva; revalidación por transacción.
- aiomysql/pool asíncronos existentes, sin conexiones sincronas ni pool paralelo.
- Carga de la primitiva en MariaDB desechable: contención misma clave y claves distintas, registrar concurrencia/p95/errores. Prueba por dominio corresponde al conjunto final antes de activar.
- Solo base desechable en puerto33991; nunca producción3308. No tocar policy/Faro ni workflows.

## Aceptación

Migrar dos veces, filas clásicas preservadas; CLASSIC conserva conducta; OFFICE denegado por resolvedores/query/administración clásica; callback dentro de transacción; rechazo/autorización revierte filas; creación confidencial atómica; replay/CAS de actor/payload; concurrencia; acceso externo no se habilita hasta completar los contratos siguientes.
