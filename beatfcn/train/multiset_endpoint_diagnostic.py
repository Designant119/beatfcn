"""Diagnostic endpoint pairing: maximum head matches, then minimum tail error."""
import numpy as np

from .joint_native_set import injective_assignment


def endpoint_diagnostic(predicted,targets,tolerance_us=10000):
    p=[e for e in predicted if e.type=='hold'];t=[e for e in targets if e.type=='hold']
    if not p or not t:return dict(matches=0,pred=len(p),true=len(t),mae_ms=None,p95_ms=None,pairs=[])
    error=np.array([[abs(a.end_us-b.end_us) for a in p] for b in t],np.float64)
    allowed=np.array([[abs(a.onset_us-b.onset_us)<=tolerance_us for a in p] for b in t])
    # All real edge costs together are less than one dummy match. This makes
    # cardinality lexicographically prior to endpoint error, retaining copies.
    costs=np.ones((len(t)+len(p),len(p)))
    costs[:len(t)]=np.where(allowed,error/(max(float(error.max()),1.)*(len(p)+1)),2.)
    rows,cols=injective_assignment(costs)
    pairs=[(int(j),int(i)) for i,j in zip(rows,cols) if i<len(t) and allowed[i,j]]
    differences=[abs(p[i].end_us-t[j].end_us)/1000 for i,j in pairs]
    return dict(matches=len(pairs),pred=len(p),true=len(t),mae_ms=float(np.mean(differences)) if differences else None,
                p95_ms=float(np.percentile(differences,95)) if differences else None,pairs=pairs,
                scope='Head-threshold maximum-cardinality then minimum-tail-cost diagnostic;events unchanged')
