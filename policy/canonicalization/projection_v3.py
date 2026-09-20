"""Projection of C14N/3 leaf registry; only NORMATIVE survives."""
from __future__ import annotations
from copy import deepcopy
from .canonical_json import canonical_json_bytes
from .errors import SchemaValidationError

def _get(value, parts):
    if not parts: return value
    head,*tail=parts
    if head.endswith("[*]"):
        key=head[:-3]
        if not isinstance(value, dict): raise KeyError(key)
        return [_get(x,tail) for x in value.get(key,[])]
    return _get(value[head],tail)
def _put(out,parts,value):
    if not parts: return value
    head,*tail=parts
    if head.endswith("[*]"):
        key=head[:-3]
        built=[_build(tail,x) for x in value]
        if key in out:
            for old, new in zip(out[key], built):
                old.update(new)
        else:
            out[key]=built
        return out
    if tail:
        out[head]=_put(out.get(head,{}),tail,value)
    else: out[head]=deepcopy(value)
    return out
def _build(parts,value):
    if not parts: return deepcopy(value)
    o={}; _put(o,parts,value); return o
def _sort_sets(value, semantics, prefix=""):
    if isinstance(value,dict): return {k:_sort_sets(v,semantics,f"{prefix}.{k}" if prefix else k) for k,v in value.items()}
    if isinstance(value,list):
        mode=semantics.get(prefix)
        items=[_sort_sets(v,semantics,prefix+"[*]") for v in value]
        if mode in {"SET_SCALAR","SET_KEYED_ID"}:
            key=(lambda x:x["id"]) if mode=="SET_KEYED_ID" else canonical_json_bytes
            if len({canonical_json_bytes(x) for x in items})!=len(items): raise SchemaValidationError(f"duplicate set: {prefix}")
            return sorted(items,key=key)
        return items
    return value
def normative_projection_v3(value:dict, artifact:str, bootstrap)->dict:
    fields=bootstrap.field_classes["fields"]; out={}
    for path, cls in fields.items():
        if not path.startswith(artifact+".") or cls!="NORMATIVE": continue
        parts=path.split(".")[1:]
        try: val=_get(value,parts)
        except (KeyError,TypeError): continue
        _put(out,parts,val)
    return _sort_sets(out, bootstrap.field_classes["array_semantics"], artifact)
