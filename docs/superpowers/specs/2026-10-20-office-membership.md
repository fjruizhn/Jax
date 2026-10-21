# Office membership · contrato transaccional JAX

2026-10-20 · OP343/345 · Codex tras arquitecturaSol office_workflow_contract. Amplía creación namespace sin permitir administración clásica de Office. Implementar y auditar antes de aceptar invitaciones Office. No producción ni datos reales.

## Interfaz tipada

En jax.memory.project_authority:

- NamespacedMemberKind: INTERNAL, EXTERNAL_CLIENT.
- NamespaceMembershipOperation: GRANT_OFFICE_MEMBER, REVOKE_OFFICE_MEMBER.
- NamespaceMembershipMutationAuthorizer.authorize_membership_in_transaction(cursor, *, actor_user_id:int,target_tenant_id:int,project_id:int,target_user_id:int,namespace:ProjectNamespace,operation:NamespaceMembershipOperation,member_kind:NamespacedMemberKind|None)->None. Retorno normalautoriza, excepciónniega; objeto es adaptador confiable del servidor, nunca callback delrequest.
- ProjectAuthorityAdmin.grant_namespaced_member_in_transaction(cursor, *, actor_user_id,target_tenant_id,project_id,target_user_id,namespace,member_kind,idempotency_key,authorizer,request_id=None,trace_id=None)->NamespacedMembershipResult.
- revoke_namespaced_member_in_transaction mismaidentidadsinmember_kind. Operación fija por método, callbackrecibe member_kindNone enrevoke. No parámetroProjectRole.

Tipos/IDs estrictos, no bool/intcoercion; idempotency_key UUID canónico. Resultado inmutable conmembership_id,event_id,changed,project_id,target_user_id,project_role,status yreplayed (nuevamutaciónfalse,replaytrue). No empieza/commit/rollback, recibe cursor de transaccióncaller. Namespace único OFFICE_CONFIDENTIAL; CLASSIC/desconocidosrechazados antes deescribir. No ampliar NamespaceMutationAuthorizer existente de creación porque carece proyecto/target.

## Política de grant y revoke

Actor canónico activo, clase staffINTERNAL/OFFICE_INTERNAL, perfilOffice y autoridadactual losverificaadaptadorJXP en mismo cursor; corecompruebaidentidadescanónicas y alcance, sin importar módulosJXP/perfiles. Actoresnohumanos/clientesdesconocidosdeniegan. Actor!=target en este contrato de administración de invitaciones; salida voluntaria futura es otraoperación.

Grant exige tenantobjetivoactivo, scopeOfficeACTIVE y targetactivo deexactotenant. TargetclaseINTERNAL/OFFICE_INTERNAL + member_kindINTERNAL fijaCONTRIBUTOR; EXTERNAL_CLIENT +EXTERNAL_CLIENT fijaVIEWER. Ningunaotra combinación. No crea OWNER/REVIEWER/adminheredado ni altera rolglobalclásico. PROJECT_ADMIN vive solo engrantOffice, su membershipcanónica sigueCONTRIBUTOR. grant_originOFFICE y campospre_adminNULL; extenderCHECKsinreescribirfilasantiguas.

Revoke exige scope/tenant/target existentes y mismoalcanceOffice, pero permite targetinactivo/cambioclase y proyectoarchivado/deshabilitado si lifecycleconocido. No depende de grant/relationtarget vigentes ni visibilidad decontenido. Rolselee demembershipalmacenada, no loelige caller. Tenantobjetivoinactivo solo siadaptadorJXPverificaOfficeSUPERUSERactivo; general/projectadmin no. Se autoriza siempreantesdereplay.

AdaptadorJXP evalúa operación administrativa ADMINISTER_PROJECT_MEMBERS con mutación específica, no MANAGE_PROJECT que requiere scopeactivo para contenido. Rootactivo: grantcrossOffice yrevokelimpiezainactivostarget. Generaladmindelegado: propiotenantactivo, sincontenidoimplícito. Projectadmin: grant/membershippropiosactivos, proyectoexacto, puede limpiararchivado sin reabrircontenido. Colaborador/externo niegan. MFA/consentimiento deaceptación loscierra JXP antes dehabilitar; core no inventa tokens/perfiles.

## Idempotencia, locks y eventos

Nueva jax_namespace_membership_mutation_request: PK(namespace,idempotency_key), operación/actor/targettenant/project/targetuser/member_kindnullable/request_digest, membership_id/event_id/role/status/changed/created_at. FKs eíndices coherentes. Digest liga todosinputsrelevantes, no estado mutableposterior. Revoke no acepta tipo/rol delrequest.

Despuésdeautorizaractual: mismakey/digest retornaregistroexacto sin modificar membership aunque haya sido revocada/reconcedida después; cuerpo/actor/proyecto/target/tipo distinto conflicto sinresultado ajeno. Nuevakeygrant sobreactivo AlreadyMember; nuevakeyrevoke missing/revoked MemberNotFound. Reactivacióncongrant nuevo autorizado. Insertarrequestúltimo, fallorollbacktodocaller; no commitspropios.

Orden: targettenant FORUPDATE → invitaciónsiaplica (caller) → actoruser → targetuserexistente → scope → perfil/delegación/grant/membershipactor (callback) → requestidempotencia → membershiptarget → grant/relationtarget(caller) → audit/event → aceptación(caller). Calleraceptación toma tenantmutexprimero y locks compatibles antesdecrearusuario/calladapter; usuario totalmente nuevo no tiene rowcontendible previa. Reusarprimitivasprivadas dondecompatible, noresolverclásicoque niegaOffice.

Evento canónico jax_project_membership_event: operaciónGRANT_OFFICE_MEMBER/REVOKE_OFFICE_MEMBER, actor_principaluser:{id},tipo según ejecución descrita abajo, project/tenant/targetexactos, transiciónrol/status, request/trace. JXP registra fundamento de autorización ensuledger enmismatransacción; noinventar pseudousuario. Nuevo tipoorigenOffice no da autoridadclásica: consumidoresclásicos siguenfiltrandoCLASSIC.

## Versión y migración

OFFICE_NAMESPACE_CONTRACT_VERSION pasaoffice-namespace.2. Creaciónrootactual permanece sinowner/membership; postinitgrupoCLIENTidempotentefailclosed. JXP checkeaversiónexacta antesdeexponeraceptación. Migración idempotente, upgradeclassicpoblado conserva datos. Rollback rechaza si quedan scopesOffice orequestsOffice antesdecualquierDDLdestructivo; revertlimpioeliminanuevatabla/cambiaCHECKypermiteapply→revert→apply.

## Pruebas

Unit yMariaDB real33991 desechable: callbackmismocursor sincommit, grant/revokemapeo/locks/actor/target/tenant/scope/clase, revokeinactivoarchivado ydenyunknownstate, actualauthorizationbeforeeveryreplay, conflicto por actor/target/kind, replaynoresurrect/no revocar regrantposterior, AlreadyMember/MemberNotFound nuevoskeys, fallo trasmembership/event/request rollbacktodo, dos concurrentessamekey una mutación/event, clásico grant/revokecontinúanrechazandoOffice, creationnoowner, migrationupgrade/rollbacksafe. La columnaaccount_class perteneceJXP: fixturela preparaexplícita paraesteadapter; runtimeausentenoseasumeINTERNAL. EXPLAIN consultasrealesíndices y noI/O bloqueante/caché depermisos.

## Procedencia de aceptación por capacidad

RefinamientoCodex/Sol previo a habilitaraceptación: el invitador es fundamento de autoridad, no usuario interactivo presente24h después. Añadir NamespaceMembershipExecutionKind INTERACTIVE_USER|INVITATION_ACCEPTANCE y dataclassfrozen NamespaceMembershipExecution(kind,invitation_id=None), validada estrictamente; invitation_idUUIDcanónico obligatorio solo INVITATION_ACCEPTANCE. Métodosgrant/revoke aceptan execution opcional por defecto INTERACTIVE_USER; callback tipado recibeexecution. No campo libre actor_principal. JXP construye execution desde invitaciónbloqueadaypruebavalidada, nunca directamente delbody. Coreconservaactor_user_idcomoautoridadcanónica a revalidar.

Evento INTERACTIVE_USER: actor_principaluser:{actor_user_id},actor_typeUSER. Evento INVITATION_ACCEPTANCE: actor_principaloffice-invitation:{invitation_id},actor_typeCAPABILITY; authority_actor_user_id canónico se conserva en requestledger FK, ligado inequívocamente por event_id. Puedeañadirsecampo directo nullableeventoparafacilitarlectura, sinreescribirhistorialogenérico. Digestidempotencia liga ejecución+invitationID; misma key cambiandoejecuciónconflicto. Replayconservaprocedenciaoriginal. Ningúnpseudousuario ni mint de sesiónnueva, core no importa ni lee tablasOfficeinvitation; callbackJXP verifica esacapacidad eissueractual bajo mismatransacción.

Pruebas adicionales: evento de bearer identificaCAPABILITY y usuariofundamentoactual; interactivoidénticoapiusual; UUID/mode inválidosrechazados; ejecución/invitationID distinto con mismakeyconflicto ynoescrituras.
