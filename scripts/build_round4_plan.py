"""Round 4 (subdivision) plan + exposure from the round-3 v2 plan (CPU; no model, no training).

Changes vs round3v2-l-spread-r2, everything else (tape, features, config, loss, history_context, charter_timing,
treatment mix, condition dropout, song chunking, code pins) is copied:
  - init_weights: the round-3 v2 final export (owner-chosen final model, EMA weights);
  - exposure: build_exposure(mode_weights over subdivision tags) — tags from scripts/tag_subdivision_windows.py,
    keys '<mode>:triplet|fine|plain'; oversamples the beat feels the model gets wrong (Phira triplets);
  - alternative_context removed (the model never used the sibling: alignment_specific 0 at 4k/8k/9k);
  - optimizer_updates / steps / checkpoints / LR schedule / wall limits / output paths for a short fine-tune.

  python -m scripts.build_round4_plan --plan /data/round3v2-l/plan-r2.json --tags TAGS.json \
      --init /data/stage0/reports/goal-100usd-20261101/final/beatfcn-r3v2-l322m.pt --out-dir /data/round4-subdiv \
      --updates 3000 [--lr 3e-5]
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

WEIGHTS = {'phira:triplet': 6.0, 'phira:fine': 2.5, 'phira:plain': 1.5,
           'mania_4k:triplet': 1.5, 'mania_4k:fine': 1.0, 'mania_4k:plain': 0.7}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def round4_plan(plan, *, run_id, exposure_key, exposure_sha, init_key, init_sha, updates, lr, seed, weights, tags_key,
                tags_sha):
    p = json.loads(json.dumps(plan))
    accum = int(p['accum_steps'])
    p['id'] = run_id
    p['output'] = f'reports/data-contract-v2-final/{run_id}-training.json'
    p['checkpoint_dir'] = f'inputs/{run_id}/checkpoints'
    p['compile_trace_dir'] = f'reports/{run_id}-traces'
    p['generation_dir'] = f'artifacts/{run_id}-generation'
    p['generation_output'] = f'reports/{run_id}-generation.json'
    old_exposure = p['exposure']
    p['inputs_sha256'] = {k: v for k, v in p['inputs_sha256'].items() if k != old_exposure}
    p['exposure'] = exposure_key
    p['inputs_sha256'][exposure_key] = exposure_sha
    p['inputs_sha256'][init_key] = init_sha
    p['inputs_sha256'][tags_key] = tags_sha
    p['init_weights'] = init_key
    p.pop('alternative_context', None)
    p['mode_weights'] = dict(weights)
    p['seed'] = int(seed)
    p['optimizer_updates'] = int(updates)
    p['steps'] = int(updates) * accum
    every = int(p.get('checkpoint_every', 16000))
    p['checkpoints'] = list(range(every, p['steps'] + 1, every))
    if p['steps'] not in p['checkpoints']:
        p['checkpoints'].append(p['steps'])
    p['learning_rate'] = float(lr)
    p['learning_rate_schedule'] = dict(p['learning_rate_schedule'], warmup_updates=100, total_updates=int(updates))
    wall = int(updates / 0.6) + 1800
    p['max_train_wall_s'], p['max_run_wall_s'] = wall, wall + 2400
    bench = dict(p['benchmark_v2'], every_optimizer_updates=max(500, int(updates) // 2),
                 history_dir=f'reports/benchmark-v2/runs/{run_id}',
                 latest_output=f'reports/benchmark-v2/runs/{run_id}/latest.json')
    p['benchmark_v2'] = bench
    p['round4'] = dict(source_plan=plan['id'], reason='Phira beat feel: triplet beats rendered as straight 1/4 '
                       '(TEST recall 0.26); oversample subdivision-tagged windows; spread input removed',
                       subdivision_weights=dict(weights), tags=tags_key)
    p.pop('relaunch', None)
    return p


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--plan', type=Path, required=True)
    a.add_argument('--tags', type=Path, required=True)
    a.add_argument('--init', type=Path, required=True)
    a.add_argument('--out-dir', type=Path, required=True)
    a.add_argument('--updates', type=int, default=3000)
    a.add_argument('--lr', type=float, default=3e-5)
    a.add_argument('--seed', type=int, default=20261106)
    a.add_argument('--run-id', default='round4-subdiv')
    args = a.parse_args()
    from beatfcn.train.counterfactual_exposure import build_exposure
    from scripts.apply_context_training import tape_records
    plan = json.loads(args.plan.read_text())
    original = json.loads(Path(plan['exposure']).read_text())
    records = tape_records(Path('/'), plan['tape'])
    pool = hashlib.sha256(json.dumps(records, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if pool != original['pool_identity']:
        raise SystemExit('tape records differ from the round-3 exposure pool identity')
    tags = {int(k): v for k, v in json.loads(args.tags.read_text())['tags'].items()}
    train = [i for i, row in enumerate(records) if row['split'] == 'train']
    if set(train) != set(tags):
        raise SystemExit(f'tags cover {len(tags)} requests, train has {len(train)}')
    steps = args.updates * int(plan['accum_steps'])
    exposure = build_exposure(train, steps=steps, seed=args.seed, mix=original['treatment_mix'],
                              dropout=original['condition_dropout'], pool_identity=original['pool_identity'],
                              song_of={i: records[i]['audio_sha256'] for i in train}, song_chunk=original['song_chunk'],
                              unconditional_native=original.get('unconditional_native', False),
                              mode_weights=WEIGHTS, mode_of=tags)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    exposure_path = args.out_dir / 'exposure.json'
    exposure_path.write_text(json.dumps(exposure, ensure_ascii=False, separators=(',', ':')) + '\n')
    new = round4_plan(plan, run_id=args.run_id, exposure_key=str(exposure_path), exposure_sha=sha(exposure_path),
                      init_key=str(args.init), init_sha=sha(args.init), updates=args.updates, lr=args.lr,
                      seed=args.seed, weights=WEIGHTS, tags_key=str(args.tags), tags_sha=sha(args.tags))
    plan_path = args.out_dir / 'plan.json'
    plan_path.write_text(json.dumps(new, indent=1) + '\n')
    print(json.dumps(dict(plan=str(plan_path), plan_sha256=sha(plan_path), exposure=str(exposure_path),
                          steps=len(exposure['steps']), epochs=exposure['epochs'], mode_mix=exposure.get('mode_mix'),
                          train_requests=len(train)), indent=1))


if __name__ == '__main__':
    main()
