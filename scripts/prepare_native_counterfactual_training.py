"""Freeze a native-tape counterfactual-density plan; variants are built in the loader.

Exposure (counterfactual-exposure-v2, see beatfcn/train/counterfactual_exposure.py):
per-epoch seeded permutation of all train requests; each (request, epoch) draws
a treatment from --treatment-mix (default native 50%, x0.5/x1.5/x2.0 15% each,
x0 5%) as a stratified per-request deck, and frozen condition dropout. Every
sample is conditioned on its own actual target (2 s achieved-density curve +
recomputed window controls), with dropout so the unconditional mode is trained.

Budget: --optimizer-updates is the intended number of optimizer updates; the
exposure holds exactly updates*accum_steps steps, so the cosine learning-rate
schedule (computed by the runner over optimizer updates) actually decays
within the run. The old fixed 300000-step exposure (18750 updates at
accum 16) kept the LR near its peak for the ~2000 updates a run really lasts.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

from beatfcn.train.counterfactual_exposure import (DEFAULT_CONDITION_DROPOUT,
    DEFAULT_CURVE_CELL_US, DEFAULT_TREATMENT_MIX, build_exposure, exposure_summary,
    normalize_dropout, normalize_mix, select_validation)
from beatfcn.train.training_panel import build_training_panel

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT/'reports/data-contract-v2-final'
CODE = ('scripts/run_local_song_baseline.py', 'scripts/prepare_native_counterfactual_training.py',
        'beatfcn/train/exposure_loader.py', 'beatfcn/train/counterfactual_density.py',
        'beatfcn/train/counterfactual_exposure.py', 'beatfcn/train/ema.py',
        'beatfcn/train/local_panel.py', 'beatfcn/train/density_curve.py',
        'beatfcn/train/event_conditions.py', 'beatfcn/events/controls.py',
        'beatfcn/train/training_objective.py', 'beatfcn/model/full_event_codec.py',
        'beatfcn/model/full_event_model.py')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def key(path):
    """Repo-relative key when the file lives under ROOT, else the absolute path."""
    path = Path(path)
    path = path if path.is_absolute() else (ROOT/path)
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def parse_json(value, default):
    return copy.deepcopy(default) if value is None else json.loads(value)


def budget(args):
    """Exposure steps for the intended number of optimizer updates."""
    accum = int(args.accum_steps)
    if accum < 1:
        raise ValueError('accum_steps must be positive')
    if args.optimizer_updates is not None:
        if args.steps is not None:
            raise ValueError('Give --optimizer-updates or legacy --steps, not both')
        updates = int(args.optimizer_updates)
    elif args.steps is not None:
        if args.steps % accum:
            raise ValueError('Legacy --steps must be a multiple of accum_steps')
        updates = args.steps // accum
    else:
        raise ValueError('Declare the intended budget with --optimizer-updates')
    if updates < 2:
        raise ValueError('At least two optimizer updates are required')
    return updates, updates * accum


BENCHMARK_CODE = ('beatfcn/train/periodic_requirements.py', 'beatfcn/train/learning_evaluation.py',
                  'beatfcn/train/audio_dependence.py', 'beatfcn/train/requirement_diagnostics.py',
                  'beatfcn/model/full_event_generate.py')
BENCHMARK_READOUTS = [dict(name='sample_t1', temperature=1., top_p=None, seeds=[0, 1, 2, 3]),
                      dict(name='greedy')]


def benchmark_settings(args, tape, *, ema):
    """plan['benchmark_v2'] (periodic_requirements.PERIODIC_PLAN_KEYS) or None.

    Defaults: every 250 optimizer updates; teacher audio-dependence screen +
    4-seed T=1 per-row sampling + greedy; EMA weights when EMA is on. The
    frozen panels must belong to the same tape (tape_sha256), else no screen
    is planned (an explicit --benchmark-dir that does not match is an error).
    """
    if not args.benchmark_every_updates:
        return None, 'disabled (--benchmark-every-updates 0)'
    explicit = args.benchmark_dir is not None
    directory = Path(args.benchmark_dir) if explicit else Path('reports/benchmark-v2')/Path(tape).parent.name
    teacher, generation = directory/'teacher-panel.json', directory/'generation-panel.json'
    absolute = lambda p: p if p.is_absolute() else ROOT/p
    problem = None
    if not (absolute(teacher).is_file() and absolute(generation).is_file()):
        problem = f'no frozen benchmark panels in {directory} (scripts/freeze_benchmark_v2.py)'
    else:
        tape_sha = digest(absolute(Path(tape)))
        for path in (teacher, generation):
            if json.loads(absolute(path).read_text()).get('tape_sha256') != tape_sha:
                problem = f'{path} was frozen from a different tape'
    if problem:
        if explicit:
            raise ValueError(problem)
        return None, problem
    return dict(every_optimizer_updates=int(args.benchmark_every_updates),
        max_wall_s=float(args.benchmark_max_wall_s), minimum_wall_s=60., teacher_fraction=.35,
        teacher_protocol=key(teacher), teacher_protocol_sha256=digest(absolute(teacher)),
        generation_protocol=key(generation), generation_protocol_sha256=digest(absolute(generation)),
        history_dir=f'reports/benchmark-v2/runs/{args.name}',
        latest_output=f'reports/benchmark-v2/runs/{args.name}/latest.json',
        readouts=copy.deepcopy(BENCHMARK_READOUTS), token_budget_factor=2., max_batch_rows=32,
        teacher_batch_size=8, base_seed=20261101, weights='ema' if ema else 'raw'), None


def every(updates, accum):
    """Exposure-step cadence for a cadence in optimizer updates (0 disables)."""
    return int(updates) * accum if updates else 0


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--tape', required=True)
    parser.add_argument('--mert-features', required=True)
    parser.add_argument('--mel-stats', required=True)
    parser.add_argument('--scale-config', required=True)
    parser.add_argument('--name', default='600m-native-cf')
    parser.add_argument('--optimizer-updates', type=int, default=None,
                        help='intended optimizer updates for this run (sets steps and the LR schedule)')
    parser.add_argument('--steps', type=int, default=None,
                        help='legacy: exposure steps (must equal intended updates * accum)')
    parser.add_argument('--accum-steps', type=int, default=4)
    parser.add_argument('--physical-microbatch', type=int, default=None,
                        help='default: 8 under the h100-v1 efficiency profile, else 4')
    parser.add_argument('--song-chunk', type=int, default=None,
                        help='adjacent same-song windows per epoch (default: physical microbatch under h100-v1)')
    parser.add_argument('--unconditional-native', action='store_true',
                        help='samples with all conditions dropped always use the native human target')
    parser.add_argument('--init-weights', default=None, help="file with a 'model' state dict to start from (fresh optimizer)")
    parser.add_argument('--efficiency-profile', default='h100-v1', choices=('h100-v1', 'none'),
                        help='h100-v1 keeps the measured 600M-run throughput settings')
    parser.add_argument('--warmup-updates', type=int, default=None,
                        help='default 5%% of updates, capped at 500')
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--minimum-lr', type=float, default=1e-5)
    parser.add_argument('--first-token-weight', type=float, default=1.)
    parser.add_argument('--ema-decay', type=float, default=.999, help='0 disables EMA')
    parser.add_argument('--treatment-mix', default=None, help='JSON list of {name,multiplier,probability}')
    parser.add_argument('--condition-dropout', default=None,
                        help='JSON {all_probability,curve_probability,control_probability}')
    parser.add_argument('--curve-cell-us', type=int, default=DEFAULT_CURVE_CELL_US)
    parser.add_argument('--validation-count', type=int, default=200)
    parser.add_argument('--vocabulary', default=None, help='JSON EventVocabulary flags')
    parser.add_argument('--config-overrides', default=None, help='JSON merged into the scale config')
    parser.add_argument('--loss-overrides', default=None, help='JSON merged into the loss settings')
    parser.add_argument('--onset-scope', default='window', choices=('window', 'core'))
    parser.add_argument('--eval-every-updates', type=int, default=250)
    parser.add_argument('--checkpoint-every-updates', type=int, default=500)
    parser.add_argument('--light-checkpoint-every-updates', type=int, default=250)
    parser.add_argument('--max-train-wall-s', type=int, default=12000)
    parser.add_argument('--benchmark-every-updates', type=int, default=250,
                        help='periodic benchmark v2 cadence in optimizer updates (0 disables)')
    parser.add_argument('--benchmark-dir', default=None,
                        help='frozen benchmark v2 panels (default reports/benchmark-v2/<tape dir name>)')
    parser.add_argument('--benchmark-max-wall-s', type=float, default=600.)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--exposure', required=True)
    parser.add_argument('--plan-output', required=True)
    parser.add_argument('--base-plan', default=None)
    parser.add_argument('--health', default=None, help='health admission JSON with passed=true (default: base plan)')
    parser.add_argument('--features', default=None, help='feature admission JSON (default: base plan)')
    args = parser.parse_args(argv)
    updates, steps = budget(args)
    accum = int(args.accum_steps)
    warmup = (min(500, max(1, round(.05 * updates))) if args.warmup_updates is None
              else int(args.warmup_updates))
    if not 0 <= warmup < updates:
        raise ValueError('Warmup must be shorter than the run')
    mix = normalize_mix(parse_json(args.treatment_mix, DEFAULT_TREATMENT_MIX))
    dropout = normalize_dropout(parse_json(args.condition_dropout, DEFAULT_CONDITION_DROPOUT))
    vocabulary = parse_json(args.vocabulary, {})
    base_path = Path(args.base_plan) if args.base_plan else REPORT/'probe-plan.json'
    base = copy.deepcopy(json.loads(base_path.read_text()))
    scale = json.loads((ROOT/args.scale_config).read_text())
    config = dict(scale['config'], **parse_json(args.config_overrides, {}))
    loss = dict(type_weight=.1, z_weight=1e-4,
                # 5.0 made the first token dominate: 84% of updates were clipped.
                first_token_weight=float(args.first_token_weight))
    if config.get('onset_head'):
        loss['onset_weight'] = .1
    loss.update(parse_json(args.loss_overrides, {}))
    plan = copy.deepcopy(base)
    seed = int(plan.get('seed', 0) if args.seed is None else args.seed)
    plan.update(id=args.name, tape=args.tape, mert_features=args.mert_features,
        steps=steps, optimizer_updates=updates, seed=seed, device='cuda', autocast='bf16',
        config=config, vocabulary=vocabulary,
        physical_microbatch_requests=int(args.physical_microbatch if args.physical_microbatch is not None
                                         else (8 if args.efficiency_profile == 'h100-v1' else 4)),
        accum_steps=accum,
        learning_rate=float(args.learning_rate), optimizer=dict(weight_decay=.1, betas=[.9, .95]),
        learning_rate_schedule=dict(warmup_updates=warmup, minimum_lr=float(args.minimum_lr),
                                    total_updates=updates),
        loss=loss,
        counterfactual_rates=sorted({item['multiplier'] for item in mix if item['multiplier']}),
        counterfactual_cell_us=100_000, counterfactual_curve_cell_us=int(args.curve_cell_us),
        counterfactual_relative=True,
        multiprocess_loader=True, loader_workers=12, loader_prefetch=4,
        mel_normalization=dict(path=args.mel_stats, sha256=digest(ROOT/args.mel_stats)),
        skip_initial_eval=True, skip_generation=True,
        checkpoint_every=every(args.checkpoint_every_updates, accum), checkpoint_keep=1,
        light_checkpoint_every=every(args.light_checkpoint_every_updates, accum),
        light_checkpoint_keep=2,
        checkpoints=list(range(every(args.eval_every_updates, accum), steps+1,
                               every(args.eval_every_updates, accum))) if args.eval_every_updates else [],
        max_train_wall_s=int(args.max_train_wall_s),
        output=f'reports/data-contract-v2-final/{args.name}-training.json',
        checkpoint_dir=f'inputs/{args.name}/checkpoints',
        generation_output=f'reports/{args.name}-generation.json',
        generation_dir=f'artifacts/{args.name}-generation',
        note='native tape; counterfactual variants + own-target conditions built in loader')
    if args.init_weights:
        plan['init_weights'] = args.init_weights
    if args.health:
        plan['health'] = args.health
    if args.features:
        plan['features'] = args.features
    if args.efficiency_profile == 'h100-v1':
        # Measured on the 600M run (reports/600m-util-resume-20261101/plan-util70.json,
        # EXPERIMENT_LOG 2026-11-01): grouped microbatch 8 at effective 64, regional
        # compile (fwd/bwd 0.369 -> 0.187 s), bounded grouped tokens, default allocator.
        plan.update(compile_loss=dict(mode='regions', options={'compile_threads': 4,
                                                               'triton.cudagraphs': False}),
                    grouped_max_tokens=1280, cuda_allocator_config='', empty_cache_every=0,
                    profile_first_micro=False, prefetch_workers=1, prefetch_lookahead=2,
                    resource_guard=dict(minimum_util=70.0),
                    efficiency_profile='h100-v1', compile_trace_dir=f'reports/{args.name}-traces',
                    yield_control_file=f'/tmp/{args.name}-yield')
    else:
        for stale_key in ('compile_loss', 'grouped_max_tokens', 'efficiency_profile'):
            plan.pop(stale_key, None)
    if args.ema_decay:
        plan['ema'] = dict(decay=float(args.ema_decay), evaluate=True)
    else:
        plan.pop('ema', None)
    if config.get('onset_head'):
        plan['onset_supervision'] = dict(scope=args.onset_scope)
    for stale in ('intensity', 'input_shapes', 'sequence_buckets', 'cuda_graph'):
        plan.pop(stale, None)
    bench, bench_reason = benchmark_settings(args, args.tape, ema=bool(args.ema_decay))
    plan.pop('benchmark_v2', None); plan.pop('benchmark_v2_disabled_reason', None)
    if bench is not None:
        plan['benchmark_v2'] = bench
    else:
        plan['benchmark_v2_disabled_reason'] = bench_reason
    panel = build_training_panel(ROOT, plan)
    train = [index for index, row in enumerate(panel.records) if row['split'] == 'train']
    if not train:
        raise ValueError('No train records')
    validation, validation_summary = select_validation(panel.records,
        count=int(args.validation_count), seed=seed)
    plan['validation_indices'] = validation
    plan['validation_selection'] = validation_summary
    # The base plan's probe indices belong to another tape; probe this one.
    plan['tensor_probe_indices'] = [train[0], validation[0]]
    pool = hashlib.sha256(json.dumps(panel.records, sort_keys=True,
        separators=(',', ':')).encode()).hexdigest()
    song_chunk = int(args.song_chunk if args.song_chunk is not None
                     else (plan['physical_microbatch_requests'] if args.efficiency_profile == 'h100-v1' else 1))
    exposure = build_exposure(train, steps=steps, seed=seed, mix=mix, dropout=dropout,
                              pool_identity=pool, song_of={i: panel.records[i]['audio_sha256'] for i in train},
                              song_chunk=song_chunk, unconditional_native=args.unconditional_native)
    plan['exposure_song_chunk'] = song_chunk
    summary = exposure_summary(exposure)
    exposure_path = Path(args.exposure)
    exposure_path.parent.mkdir(parents=True, exist_ok=True)
    exposure_path.write_text(json.dumps(exposure, ensure_ascii=False, separators=(',', ':'))+'\n')
    plan['exposure'] = key(exposure_path)
    plan['counterfactual'] = dict(format=exposure['format'], treatment_mix=mix,
        treatment_assignment=exposure['treatment_assignment'], condition_dropout=dropout,
        condition_source='own actual (edited) target: 2 s achieved-density curve + recomputed 8x6 controls',
        curve_cell_us=int(args.curve_cell_us), render_cell_us=100_000,
        train_requests=len(train), epochs=exposure['epochs'], complete_epochs=exposure['complete_epochs'],
        realized=summary)
    plan['budget_note'] = (f'{updates} optimizer updates x accum {accum} = {steps} exposure steps; '
                           'cosine LR over optimizer updates; first_token_weight=1 (5 clipped 84% of updates)')
    inputs = [Path(args.tape), exposure_path, Path(args.mel_stats)]
    inputs = [path if path.is_absolute() else ROOT/path for path in inputs]
    preflight = REPORT/'preflight.json'
    if preflight.is_file():
        inputs.append(preflight)
    plan['inputs_sha256'] = {key(path): digest(path) for path in inputs}
    code = set(plan.get('code_sha256', {})) | set(CODE) | (set(BENCHMARK_CODE) if bench else set())
    plan['code_sha256'] = {path: digest(ROOT/path) for path in sorted(code) if (ROOT/path).is_file()}
    target = Path(args.plan_output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(plan, ensure_ascii=False, indent=1)+'\n')
    print(json.dumps(dict(plan=str(target), optimizer_updates=updates, steps=steps,
        records=len(panel.records), train=len(train), validation=len(validation),
        epochs=exposure['epochs'], pool=pool[:12], realized=summary,
        benchmark_v2=bool(bench), benchmark_v2_disabled_reason=bench_reason), ensure_ascii=False))
    return plan


if __name__ == '__main__':
    main()
