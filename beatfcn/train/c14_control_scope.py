"""Remove supplied values, preserving public applicability and scope clocks."""
import torch


def unspecified_controls(inputs):
    controls=inputs['controls']
    if controls.ndim!=3 or controls.shape[-1]!=6:raise ValueError('Expected lower/upper/specified/applicable/start/end')
    result=controls.clone();result[...,:3]=0
    return {**inputs,'controls':result}
