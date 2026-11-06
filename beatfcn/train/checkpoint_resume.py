"""Validate exact training-state continuation across execution-only repairs."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import torch

from beatfcn.model.full_event_model import FullEventConfig

SCIENTIFIC_FIELDS = (
    'data_contract_version', 'seed', 'config', 'tape', 'mert_features', 'features',
    'exposure', 'steps', 'accum_steps',
    'learning_rate', 'optimizer', 'learning_rate_schedule', 'loss',
    'mel_normalization', 'inputs_sha256', 'counterfactual_rates',
    'counterfactual_cell_us', 'counterfactual_relative',
    # Recipe v2: codec/vocabulary, EMA definition, counterfactual treatment
    # mix + condition dropout + curve grid, onset supervision, LR budget.
    'vocabulary', 'ema', 'counterfactual', 'counterfactual_curve_cell_us',
    'onset_supervision', 'optimizer_updates',
)
# Absent and the default value mean the same recipe for these fields.
_DEFAULT_EQUIVALENT = dict(vocabulary={}, onset_supervision=None, ema=None)


def _scientific(plan, field):
    value = plan.get(field)
    if field == 'vocabulary':
        from beatfcn.train.exposure_loader import make_vocabulary
        return asdict(make_vocabulary(value or None))
    if field in _DEFAULT_EQUIVALENT and not value:
        return _DEFAULT_EQUIVALENT[field]
    return value


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def validate_resume(plan, parent, saved, *, parent_sha, checkpoint_sha):
    source = plan['resume_checkpoint']
    if source['sha256'] != checkpoint_sha or source['parent_plan_sha256'] != parent_sha:
        raise ValueError('Resume checkpoint or parent plan identity mismatch')
    if saved['plan_sha256'] != parent_sha:
        raise ValueError('Checkpoint does not belong to the parent plan')
    for field in SCIENTIFIC_FIELDS:
        if _scientific(plan, field) != _scientific(parent, field):
            raise ValueError('Resume changes scientific field: '+field)
    if saved['config'] != asdict(FullEventConfig(**plan['config'])):
        raise ValueError('Checkpoint model configuration mismatch')
    if 'vocabulary' in saved and saved['vocabulary'] != _scientific(plan, 'vocabulary'):
        raise ValueError('Checkpoint vocabulary mismatch')
    if plan.get('ema') and not saved.get('ema'):
        raise ValueError('Plan enables EMA but the checkpoint has no EMA state')
    if plan.get('physical_microbatch_requests', 4) not in (1, 2, 4, 8, 16):
        raise ValueError('Unsupported physical microbatch')
    step = saved['updates']
    accum = int(plan.get('accum_steps', 1))
    if (not isinstance(step, int) or step < 0 or step % accum
            or step >= plan['steps'] or source.get('updates', step) != step
            or saved['progress']['updates'] != step
            or saved['progress']['optimizer_updates'] != step//accum):
        raise ValueError('Resume requires an exact complete optimizer boundary')
    if not saved.get('optimizer') or 'cpu_rng' not in saved:
        raise ValueError('Resume requires optimizer and RNG state, not weights only')
    if plan.get('device') == 'cuda' and saved.get('cuda_rng') is None:
        raise ValueError('Resume checkpoint lacks CUDA RNG state')
    return step


def load_resume(root, plan):
    source = plan['resume_checkpoint']
    parent_path = root/source['parent_plan']
    checkpoint_path = root/source['path']
    parent_sha = file_sha(parent_path)
    checkpoint_sha = file_sha(checkpoint_path)
    parent = json.loads(parent_path.read_text())
    saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    validate_resume(plan, parent, saved, parent_sha=parent_sha, checkpoint_sha=checkpoint_sha)
    return saved
