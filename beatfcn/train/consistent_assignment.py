"""Costs for the already specified balanced matching objective, not a new loss."""
import numpy as np


def balanced_assignment_cost(positive_nll, absent_nll, kinds):
    positive_nll=np.asarray(positive_nll,np.float64)
    absent_nll=np.asarray(absent_nll,np.float64)
    kinds=np.asarray(kinds,np.int64)
    q,n=positive_nll.shape
    if n>q or kinds.shape!=(n,) or absent_nll.shape!=(q,):raise ValueError('Invalid matching dimensions')
    if not np.isfinite(positive_nll).all() or not np.isfinite(absent_nll).all():raise ValueError('Nonfinite costs')
    if not n:return positive_nll.copy()
    values,counts=np.unique(kinds,return_counts=True)
    weights={int(k):.5/(len(values)*int(count)) for k,count in zip(values,counts)}
    w=np.array([weights[int(k)] for k in kinds])
    negative_weight=.5/(q-n) if q>n else 0.
    return positive_nll*w[None]-negative_weight*absent_nll[:,None]
