from __future__ import annotations
import argparse, json
from datetime import datetime, timezone
from .formatters import json_output, human_output
from .runtime import UnavailableSource, control_status, health, authority, decision, execution, evidence
from policy.enforcement_evidence.models import ClaimEnvironment, ClaimLevel, ClaimScope, Coverage, EvidenceSubject, EvidenceSubjectType

def _json(value):
    try: return json.loads(value)
    except json.JSONDecodeError as exc: raise argparse.ArgumentTypeError("invalid JSON") from exc
def _scope(value):
    data=_json(value)
    return ClaimScope(ClaimEnvironment(data["environment"]),data.get("deployment_id"),data.get("database_scope_id"),Coverage(data.get("coverage", "OBSERVED_SUBJECTS_ONLY")))
def _subjects(value):
    return tuple(EvidenceSubject(EvidenceSubjectType(item["subject_type"]), item["identity"]) for item in _json(value))
def parser():
    p=argparse.ArgumentParser(prog="jaxctl",description="Read-only JAX query façade")
    sub=p.add_subparsers(dest="command",required=True)
    for name in ("authority","health","status"):
        q=sub.add_parser(name); q.add_argument("--json",action="store_true")
    for name,arg in (("decision","decision_id"),("execution","execution_id"),("evidence","identity")):
        q=sub.add_parser(name); q.add_argument(arg); q.add_argument("--json",action="store_true")
        if name=="decision": q.add_argument("--replay",action="store_true")
    q=sub.add_parser("control"); q.add_argument("control_id"); q.add_argument("--version",required=True,type=int); q.add_argument("--claim",required=True,choices=[x.value for x in ClaimLevel]); q.add_argument("--scope",required=True,type=_scope); q.add_argument("--subjects",required=True,type=_subjects); q.add_argument("--as-of",default=None); q.add_argument("--json",action="store_true")
    return p
def run(argv=None):
    args=parser().parse_args(argv)
    try:
        if args.command=="control":
            as_of=datetime.fromisoformat(args.as_of.replace("Z","+00:00")) if args.as_of else datetime.now(timezone.utc)
            result=control_status(control_id=args.control_id,control_version=args.version,claim_level=ClaimLevel(args.claim),scope=args.scope,subjects=args.subjects,as_of_utc=as_of)
        elif args.command=="health": result=health()
        elif args.command=="authority": result=authority()
        elif args.command=="decision": result=decision(args.decision_id,args.replay)
        elif args.command=="execution": result=execution(args.execution_id)
        elif args.command=="evidence": result=evidence(args.identity)
        elif args.command=="status": result={"classification":"OPERATIONAL_DIAGNOSTIC","source":"jaxctl live query façade","status":"AVAILABLE","health":health()}
    except UnavailableSource as exc:
        result={"classification":"UNAVAILABLE","status":"UNAVAILABLE","source":"jaxctl","detail":str(exc)}
        print(json_output(result) if args.json else human_output(result)); return 2
    print(json_output(result) if args.json else human_output(result)); return 0
