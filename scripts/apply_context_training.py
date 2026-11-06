"""Add history-context training and/or mode re-weighting to a frozen counterfactual plan.

Post-processes a plan written by scripts/prepare_native_counterfactual_training.py
(that script and the runner are not modified): the original exposure is first
REBUILT from its own recorded settings and must reproduce byte for byte; then
the exposure is rebuilt with ``mode_weights`` (if given), and the plan gets
``history_context`` / ``mode_weights`` keys, the new exposure identity and the
code SHA-256 of the files this feature uses. The runner reads history_context
through exposure_loader.plan_loader_settings. Metadata only: no model, no GPU,
no cloud call, no training launch.

  .venv/bin/python -m scripts.apply_context_training --plan PLAN.json \
      --history-context '{}' --mode-weights '{"phira": 1.6, "mania_4k": 1.0}' \
      --exposure inputs/<name>/exposure.json --plan-output reports/<name>/plan.json
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beatfcn.train.counterfactual_exposure import build_exposure, exposure_summary  # noqa: E402
from beatfcn.train.history_context import normalize_history_context  # noqa: E402

FEATURE_CODE = ('beatfcn/train/exposure_loader.py', 'beatfcn/train/counterfactual_exposure.py',
                'beatfcn/train/history_context.py', 'beatfcn/train/event_reference.py')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def key(path):
    path = Path(path)
    path = path if path.is_absolute() else ROOT/path
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def tape_records(root, tape):
    """Records in LocalRequestPanel/MertRequestPanel order (identity = pool hash)."""
    from scripts.freeze_benchmark_v2 import TapePanel
    return TapePanel(root, tape).records


def rebuild(original, records, *, mode_weights=None):
    train = [i for i, row in enumerate(records) if row['split'] == 'train']
    kwargs = dict(steps=len(original['steps']), seed=original['seed'], mix=original['treatment_mix'],
                  dropout=original['condition_dropout'], pool_identity=original['pool_identity'],
                  song_of={i: records[i]['audio_sha256'] for i in train}, song_chunk=original['song_chunk'],
                  unconditional_native=original.get('unconditional_native', False))
    if mode_weights is not None:
        kwargs.update(mode_weights=mode_weights, mode_of={i: records[i]['mode'] for i in train})
    return build_exposure(train, **kwargs)


def apply(plan, *, root=ROOT, mode_weights=None, history_context=None, records=None):
    """Return (new plan, new exposure or None when the exposure is unchanged)."""
    original = json.loads((Path(root)/plan['exposure']).read_text())
    records = tape_records(root, plan['tape']) if records is None else records
    pool = hashlib.sha256(json.dumps(records, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if pool != original['pool_identity']:
        raise ValueError('Tape records differ from the exposure pool identity')
    if rebuild(original, records) != original:
        raise ValueError('The original exposure does not reproduce from its recorded settings')
    plan = json.loads(json.dumps(plan))
    exposure = None
    if mode_weights is not None:
        exposure = rebuild(original, records, mode_weights=mode_weights)
        plan['mode_weights'] = exposure['mode_weights']
        counterfactual = plan.setdefault('counterfactual', {})
        counterfactual.update(epochs=exposure['epochs'], complete_epochs=exposure['complete_epochs'],
                              treatment_assignment=exposure['treatment_assignment'],
                              realized=exposure_summary(exposure))
    if history_context is not None:
        plan['history_context'] = normalize_history_context(history_context)
    return plan, exposure


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--plan-output', type=Path, required=True)
    p.add_argument('--exposure', type=Path, help='output path of the re-weighted exposure (with --mode-weights)')
    p.add_argument('--mode-weights', type=json.loads, default=None)
    p.add_argument('--history-context', type=json.loads, default=None,
                   help="JSON settings, '{}' for defaults (p .5, p_infill .25, 4/8/16/32 s)")
    p.add_argument('--root', type=Path, default=ROOT)
    a = p.parse_args()
    if a.mode_weights is None and a.history_context is None:
        p.error('nothing to apply: give --mode-weights and/or --history-context')
    if a.mode_weights is not None and a.exposure is None:
        p.error('--mode-weights needs --exposure')
    if a.plan_output.exists():
        raise FileExistsError(f'{a.plan_output} exists; plans are never overwritten')
    plan = json.loads(a.plan.read_text())
    plan, exposure = apply(plan, root=a.root, mode_weights=a.mode_weights, history_context=a.history_context)
    if exposure is not None:
        out = a.exposure if a.exposure.is_absolute() else a.root/a.exposure
        if out.exists():
            raise FileExistsError(f'{out} exists; exposures are never overwritten')
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(exposure, ensure_ascii=False, separators=(',', ':'))+'\n')
        plan['inputs_sha256'].pop(plan['exposure'], None)
        plan['exposure'] = key(out)
        plan['inputs_sha256'][plan['exposure']] = digest(out)
    for path in FEATURE_CODE:
        plan.setdefault('code_sha256', {})[path] = digest(a.root/path)
    plan['context_training_source_plan'] = dict(path=key(a.plan), sha256=digest(a.plan))
    a.plan_output.parent.mkdir(parents=True, exist_ok=True)
    a.plan_output.write_text(json.dumps(plan, ensure_ascii=False, indent=1)+'\n')
    print(json.dumps(dict(plan=str(a.plan_output), exposure=plan['exposure'],
        history_context=plan.get('history_context'), mode_weights=plan.get('mode_weights'),
        mode_mix=None if exposure is None else exposure['mode_mix']), indent=1))


if __name__ == '__main__':
    main()
