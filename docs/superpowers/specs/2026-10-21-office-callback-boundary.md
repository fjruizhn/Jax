# Office work callback boundary

2026-10-21 · DECISIÓN técnica Codex, arquitectura Sol delivery_path. Complementa office-execution.md y office-runtime-policy.md; sin habilitación productiva. Corrige explícitamente una propuesta intermedia del arquitecto de canonicalizar JSON: fue descartada porque alteraba el texto exacto del proveedor. Una memoria corregida queda aquí con procedencia.

## DTO y contenido exacto

Todos los envelopes usan office-work.1, Pydantic extra=forbid/frozen y escalares JSON estrictos. Binding compartido: pipeline_id UUID canónico, run_epoch entero>=0, step_id y correlation_id determinísticos, request_sha256 lowercase64hex. IDs de job/stage provienen de petición/reserva Office existente, no de URL de callback elegible por cliente.

Materialize request añade phase PREFLIGHT|EXECUTION. Response MATERIALIZED con prompt:string exacta, materialization_sha256 (strict UTF8), binding_version entero>0 opaco emitido por DB JXP, contract_version; alternativa CANCELLED_REVOKED o INPUT_STALE en ejecución. No office_stage_version en materialize. JXP bloquea stage/job y reautoriza fuentes/actor/operación/tenant/proyecto/grupo, congela binding en PREFLIGHT, lo reconstruye y compara en EXECUTION. Jacobs compara digest UTF8 y binding_version contra parpersistido. Ningún texto de autoridad proviene del caller.

Complete request añade output_json:string ESTRICTA con el texto exacto del proveedor, output_sha256 y modelo_real token seguro existente. Hash=sha256(output_json.encode('utf-8', errors='strict')). Nada de parse, trim, normalización, cambios de saltos/Unicode o reserialización en Jacobs. Un texto que no sea JSON puede viajar: JXP verifica hash ANTES del parse de dominio/efecto. Un surrogate no representable falla con código estable sin contenido. Envelope JSON puede escapar, pero valor recibido es exacto. Versiones de whitespace/keyorder/newline/Unicode tienen digest distinto.

Receipt solo receipt_id UUID, effect_kind [A-Z][A-Z0-9_]{0,47}, effect_id ASCII [A-Za-z0-9][A-Za-z0-9._:-]* max128, output_sha256 exacto, office_stage_version entero>0. Response STORED|CANCELLED_REVOKED. Exact replay devuelve mismo recibo; output distinto409. No publicación implícita, permisos actuales postmodelo.

Fail request añade phase PREFLIGHT|GENERATION|DELIVERY, código de enumcerrado, model_invoked:bool estricto, output_sha256 opcional exacto. Response FAILED|CANCELLED_REVOKED|OUTCOME_UNKNOWN|ALREADY_COMPLETED; último trae receipt y repara pérdida de respuesta sin repetir generación. Registro de fallo verdadero exige servicio/binding, no autoridad humana todavía vigente. Errores de transporte ambiguos no se convierten en rechazo conocido.

## Transporte y gate

Solo servicio plataforma puede POST /jacobs/office-v1/runs. Callback Jacobs hacia JXP usa identidad existente jacobs y encabezado_propio(IDENTIDAD_JACOBS), secreto existente, clienteHTTP compartido con follow_redirects=False y scopeprivado. BaseURL absoluta configurada, sin userinfo/query/fragment, flag explícito, par exacto office-work.1, límites técnicos positivos explícitos y credencialusable antes de reservar. No defaults de activación, URLs en body ni secretfamily nueva. Respuesta limitada antes de parse, JSONenvelope sin duplicados/nonfinite y DTO estricto; jamás log de cuerpo/URL/prompt/output/cause.

## Estado y entrega

Mantener pipeline.plan=[]/context_refs={}, step determinístico index0 pending input_ref={} sin texto. Tras preflight input_ref solo office_materialization_sha256,binding_version. Output_ref solo receipt seguro; save_if_large recibe únicamente receipt, jamás output bruto. Misma cola/runner/cupo, sin ejecutor paralelo.

Store dedica operaciones transaccionales que bloquean pipeline→step: prepare_step idempotente y bindingexacto; claim_preflight RESERVED→PREFLIGHT_IN_FLIGHT; mark_ready con input_ref y eventoatómicos; claim_delivery READY/pending→DELIVERY_IN_FLIGHT/running; store_receipt termina step+pipeline RECEIPT_STORED en unaTX; finish_terminal deriva status desdeFAILED/CANCELLED_REVOKED/OUTCOME_UNKNOWN. Cada cambio requiereOffice+epoch+fase+IDs exactos y eventoopaco. No salidas desde terminal. Acknowledgement perdido exige rereadexacto.

Route materializa/preflight antes deREADY y programaciónrun_pipeline; exactreplayREADY puede programar otra vez, deliveryCAS permiteuna generación. RESERVED seguro con peticiónexacta; PREFLIGHT_IN_FLIGHT nunca repiteprobe por sísolo. run_pipeline OFFICE rama temprana antes de classic: EXECUTIONmaterialize/comparación→claimdelivery→_dispatch_step una vez sin herramientas→complete/fail→persistreceipt. Nunca _run_one_step, save_if_large(raw), repo, avisos, arbiter/devolucion ocontinueOffice.

CrashREADY puede reprogramarse por runnerexistente y CAS. In-flight ambiguo jamás regenera; reconciliación failALREADY_COMPLETED puede cerrar mediante recibo. Scanoffice puede reusar schedulingreaper pero no consulta classic y no ejecuta RESERVED/PREFLIGHT/DELIVERY. Lease/reconciliación final pertenecerá al coordinadorJXP5; pairgate falso hasta que exista.

## Aislamiento y pruebas

Lectoresclassic pipeline_get, steps_by_pipeline, events_by_pipeline, pipeline/step_status_snapshots deben filtrar CLASSIC; Office usa lectores nombrados. Toda rutaresultados/events/mutaciones/CLIrelaunch consideraOffice inexistente antes de leerStep/ref/evento. JXPSQLdirecto también filtra por separado.

Probar DTOestricto/hashtextual; DB2claims1winner/rollback/epoch/terminal; sentinelcontenido ausente enpipelines/steps/events/repo/artifacts/logs/excepciones; receipt-onlysave; revocaciónantes/después conservausage; completadorespuesta perdida→failALREADY_COMPLETED1efecto; failambiguo→UNKNOWN; READYreinicio1llamada vsDELIVERYreinicio0; classicrutas/CLIigual. Mock/inprocesscallbacks, nunca proveedor o producción. EXPLAINreal y carga antes deenable.

## Código INPUT_STALE y upgrade

Ruling Codex conSol delivery_path: conservar código nuevoOFFICE_INPUT_STALE porque cambios de fuentes no son runtimepolicydenial. Upgrade CHECK binding_v2 por expand/contract: inspecciónexactaold/new, ADD+verifynew antesDROPold, estadosfresh/old/both/new reanudables, desconocido failclosed. CosteDDLaditivo+verificación, sinampliarautoridad;4C1loimplementaantesdehabilitarcallbacks.

## Accepted POST response (4C2, Codex+Sol delivery_path)

2026-10-21 DECISIÓN técnica: HTTP201/200/202 usa OfficeRunResponse estricta frozen/extra=forbid con solo contract_version literaloffice-work.1,pipeline_id UUIDcanónico,run_epoch intestricto>=0,phase OfficePhase. 409 no expone ID canónico;422/503 devuelvencódigoestable sincontenido. Cierra el único detalle de wire no fijado antes del consumidor JXP5. CosteDTOpequeño+pruebadeconsumidor; no endpointdepolling ni protocolo extra. Lectores delrunner/recovery se fijan siempre a Office+running+READY, sinparámetro callerphase.
