"""Freeze an exposure and plan for a built counterfactual density tape."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

from beatfcn.train.training_panel import build_training_panel

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT/'reports/data-contract-v2-final'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tape', required=True)
    parser.add_argument('--curves', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--name', default='counterfactual-density-v1')
    parser.add_argument('--scale-config', default=None)
    parser.add_argument('--learning-rate', type=float, default=None)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--loader', action='store_true')
    parser.add_argument('--exposure-dir', default=None)
    parser.add_argument('--plan-output', default=None)
    parser.add_argument('--mel-stats', default=None)
    parser.add_argument('--mert-features', default=None)
    parser.add_argument('--warmup', type=int, default=None)
    parser.add_argument('--min-lr', type=float, default=None)
    parser.add_argument('--weight-decay', type=float, default=None)
    parser.add_argument('--max-wall', type=int, default=None)
    parser.add_argument('--accum', type=int, default=None)
    args = parser.parse_args()
    base = copy.deepcopy(json.loads((REPORT/'probe-plan.json').read_text()))
    curves = json.loads((ROOT/args.curves).read_text())
    plan = copy.deepcopy(base)
    if args.mert_features:
        plan['mert_features'] = args.mert_features
    if args.mel_stats:
        stats = Path(args.mel_stats)
        plan['mel_normalization'] = dict(path=str(stats), sha256=digest(stats))
    if args.scale_config:
        scale = json.loads((ROOT/args.scale_config).read_text())
        plan['config'] = scale['config']
    plan.update(id=args.name, tape=args.tape, steps=args.steps, device=args.device,
        autocast='bf16' if args.device == 'cuda' else None,
        intensity=dict(cell_us=100_000, scale=16., hide_probability=0.,
            requested_curves=curves),
        loss=dict(type_weight=.1, z_weight=1e-4, first_token_weight=5.),
        output=f'reports/data-contract-v2-final/{args.name}-training.json',
        checkpoint_dir=f'inputs/{args.name}/checkpoints',
        checkpoint_every=args.steps, checkpoints=[], max_train_wall_s=1800,
        skip_initial_eval=True, skip_generation=True,
        note='Counterfactual density tape training')
    if args.learning_rate is not None:
        plan['learning_rate'] = args.learning_rate
    if args.warmup is not None or args.min_lr is not None:
        schedule = dict(plan.get('learning_rate_schedule') or {})
        if args.warmup is not None:
            schedule['warmup_updates'] = args.warmup
        if args.min_lr is not None:
            schedule['minimum_lr'] = args.min_lr
        plan['learning_rate_schedule'] = schedule
    if args.weight_decay is not None:
        optimizer = dict(plan.get('optimizer') or {})
        optimizer['weight_decay'] = args.weight_decay
        plan['optimizer'] = optimizer
    if args.max_wall is not None:
        plan['max_train_wall_s'] = args.max_wall
    if args.accum is not None:
        plan['accum_steps'] = args.accum
    if args.loader:
        plan.update(multiprocess_loader=True, loader_workers=12, loader_prefetch=4)
    panel = build_training_panel(ROOT, plan)
    train_rows = [(index, row) for index, row in enumerate(panel.records)
                  if row.get('split') == 'train']
    free = [index for index, row in train_rows if row.get('scope_role') == 'free']
    curve = [index for index, row in train_rows if row.get('scope_role') == 'counterfactual_curve']
    if not curve:
        raise ValueError('No curve samples in tape')
    steps = []
    for update in range(args.steps):
        chosen = []
        for slot in range(4):
            chosen.append(free[(update*4+slot) % len(free)] if free and slot == 0
                          else curve[(update*4+slot) % len(curve)])
        steps.append(dict(effective_update=update+1, indices=chosen,
            specified_condition_fields=[[]]*4))
    exposure = dict(pool_identity=hashlib.sha256(json.dumps(panel.records,
        sort_keys=True, separators=(',', ':')).encode()).hexdigest(), steps=steps)
    exposure_path = (Path(args.exposure_dir)/'exposure.json' if args.exposure_dir
                     else ROOT/f'inputs/{args.name}/exposure.json')
    exposure_path.parent.mkdir(parents=True, exist_ok=True)
    exposure_path.write_text(json.dumps(exposure, ensure_ascii=False, indent=1)+'\n')
    plan['exposure'] = (str(exposure_path) if exposure_path.is_absolute()
                        else str(exposure_path.relative_to(ROOT)))
    def key(path):
        path = Path(path)
        return str(path) if path.is_absolute() else str(path.relative_to(ROOT))
    inputs = [Path(args.tape), Path(args.curves), exposure_path,
              plan['mel_normalization']['path']]
    preflight = ROOT/'reports/data-contract-v2-final/preflight.json'
    if preflight.is_file():
        inputs.append(preflight)
    plan['inputs_sha256'] = {key(path): digest(path) for path in inputs}
    plan['code_sha256'] = {path: digest(path) for path in plan['code_sha256'] if Path(path).is_file()}
    target = (Path(args.plan_output) if args.plan_output
              else REPORT/f'{args.name}-plan.json')
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(plan, ensure_ascii=False, indent=1)+'\n')
    print(json.dumps(dict(plan=str(target), steps=args.steps,
        records=len(panel.records), free=len(free), curve=len(curve),
        pool=exposure['pool_identity'][:12]), ensure_ascii=False))


if __name__ == '__main__':
    main()
