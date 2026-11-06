"""Continuation / infill benchmark (see beatfcn/train/continuation_benchmark.py, design/BENCHMARK_GUIDE_v2.md §5).

Readouts (never mixed):
  A. teacher_forced (primary, cheap): per-field CE of the human window with no
     reference / human prefix 8/16/32 s / prefix as model_history / infill
     (prefix + suffix) / FOREIGN prefix control; paired gaps with SE per mode.
  B. free_generation: whole-song continuation from X s (human [0, X) visible)
     and one 8-16 s infill gap per chart, each with an unconditioned twin.

Checkpoint mode (any device; never schedules cloud work, never trains):
  .venv/bin/python -m scripts.run_continuation_benchmark --plan PLAN.json --checkpoint step-X.pt \
      --weights ema --device cuda --out-dir reports/continuation-benchmark-v1/<run>

Local plumbing smoke (tiny RANDOM-INIT model on locally cached MERT features of the val songs):
  .venv/bin/python -m scripts.run_continuation_benchmark --local-smoke --device cpu --out-dir /tmp/cont-smoke
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beatfcn.train import continuation_benchmark as cb  # noqa: E402

TAPE = 'inputs/native-v2-d94-580/tape.json'


def sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def smoke_setup(args):
    import torch
    from beatfcn.model.full_event_model import FullEventConfig, FullEventModel
    from scripts.freeze_benchmark_v2 import TapePanel
    from scripts.run_benchmark_v2 import local_mert_store
    store = local_mert_store()
    tape = json.loads((ROOT/TAPE).read_text())
    audios = {c['audio_sha256'] for c in tape['charts'] if c['split'] == 'val' and store.available(c['audio_sha256'])}
    if not audios:
        raise SystemExit('No local MERT+mel features for any val song')
    panel = TapePanel(ROOT, TAPE, features=store, audios=audios)
    for chart in panel.charts:
        store.ensure(chart['audio_sha256'])
        if abs(store.records[chart['audio_sha256']]['duration_us']-int(chart['duration_us'])) > 1:
            raise ValueError('Tape and local feature clocks disagree')
        store.records[chart['audio_sha256']]['duration_us'] = int(chart['duration_us'])
    torch.manual_seed(0)
    config = FullEventConfig(d=64, heads=4, kv_heads=2, ffn=128, encoder_layers=1, decoder_layers=2,
        tap_dim=1024, taps=2, bands=1, band_slots=1, dropout=0., mel_bins=128, intensity_dim=3)
    model = FullEventModel(config).to(args.device).eval()
    identity = dict(path=None, sha256=None, note='RANDOM-INIT smoke model; plumbing only', tape=TAPE,
                    tape_sha256=sha_file(ROOT/TAPE), val_songs_with_local_features=len(audios))
    return model, panel, identity, 3, 100_000


def checkpoint_setup(args):
    import torch
    from beatfcn.model.full_event_codec import EventVocabulary
    from beatfcn.model.full_event_model import FullEventConfig, FullEventModel
    from beatfcn.train.training_panel import build_training_panel
    plan = json.loads(args.plan.read_text())
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    config = FullEventConfig(**saved['config'])
    model = FullEventModel(config, EventVocabulary(**saved.get('vocabulary', {}))).to(args.device)
    if args.weights == 'ema':
        if not saved.get('ema'):
            raise SystemExit('Checkpoint has no EMA state; use --weights raw')
        model.load_state_dict(saved['ema']['model'], strict=True)
    else:
        model.load_state_dict(saved['model'], strict=True)
    model.eval()
    updates = saved.get('updates')
    del saved
    panel = build_training_panel(args.root, plan)
    identity = dict(path=str(args.checkpoint), sha256=sha_file(args.checkpoint), weights=args.weights,
                    updates=updates, plan=str(args.plan), plan_sha256=sha_file(args.plan),
                    plan_history_context=plan.get('history_context'), tape=plan['tape'])
    return model, panel, identity, config.intensity_dim, int(plan.get('counterfactual_cell_us', 100_000))


def _fmt(value, se=None, digits=3, signed=False):
    if value is None:
        return '—'
    text = f'{value:+.{digits}f}' if signed else f'{value:.{digits}f}'
    if signed:
        text += f' ± {se:.{digits}f}' if se is not None else ' (n=1)'
    return text


def markdown(report):
    lines = ['# Continuation / infill benchmark', '',
             f"source: `{report['model']['note'] if report['model'].get('note') else report['model']['path']}`", '']
    teacher = report.get('teacher_forced')
    if teacher:
        s = teacher['summary']
        lines += ['## A. Teacher-forced (PRIMARY): CE gaps in nats, paired mean ± SE over windows', '',
                  'gap = CE(control) − CE(treatment); **> 0 = the reference helps**. '
                  '`song_specific_prefixL` = CE(foreign prefix) − CE(this song\'s prefix): > 0 = uses *this* song.', '',
                  f"windows {s['measured_windows']}/{s['expected_windows']} {s['windows_by_mode']}, "
                  f"max reference rows {s.get('reference_rows_max')}", '']
        for mode, block in s.get('by_mode', {}).items():
            lines += [f'### {mode}', '', '| comparison | ' + ' | '.join(cb.HEADLINE_FIELDS) + ' |',
                      '|---|' + '---|'*len(cb.HEADLINE_FIELDS)]
            for name, gaps in block['gaps'].items():
                cells = [_fmt(gaps[f]['paired_mean_nats'], gaps[f]['paired_se_nats'], signed=True) if f in gaps else '—'
                         for f in cb.HEADLINE_FIELDS]
                lines.append(f'| {name} | ' + ' | '.join(cells) + ' |')
            ce = block['field_ce']
            lines += ['', 'joint CE per condition: ' + ', '.join(
                f"{c} {_fmt(ce[c]['joint']['mean'])}" for c in ce), '']
    generation = report.get('free_generation')
    if generation:
        lines += ['## B. Free generation (whole-song continuation / infill), conditioned vs unconditioned twin', '',
                  f"decoding: {generation['decoding']}", '',
                  '| task | X s | mode | charts | arm | complete | CER ↓ | onset F1@50 ↑ | note F1@50 ↑ | '
                  'density ratio model / human (vs context) | |log dens err| ↓ | type JS ↓ | IOI JS ↓ |',
                  '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
        for g in generation['summary']:
            for arm in ('cond', 'uncond'):
                a = g['arms'][arm]
                lines.append(f"| {g['task']} | {g['prefix_s'] or '—'} | {g['mode']} | {g['charts']} | {arm} | "
                             f"{_fmt(g['complete_fraction'][arm], digits=2)} | {_fmt(a['cer'])} | "
                             f"{_fmt(a['onset_f1_50ms_pooled'])} | {_fmt(a['note_f1_50ms_pooled'])} | "
                             f"{_fmt(a['density_ratio_model'], digits=2)} / {_fmt(a['density_ratio_human'], digits=2)} | "
                             f"{_fmt(a['abs_log_density_error'])} | {_fmt(a['type_mix_js_vs_human'])} | "
                             f"{_fmt(a['ioi_js_vs_human'])} |")
        lines += ['', 'Paired cond − uncond (mean ± SE over charts; sign convention in `better`):', '',
                  '| task | X s | mode | CER | onset F1 | note F1 | |log dens err| | type JS | IOI JS |',
                  '|---|---|---|---|---|---|---|---|---|']
        for g in generation['summary']:
            d = g['paired_cond_minus_uncond']
            cell = lambda m: _fmt(d[m]['mean'], d[m]['se'], signed=True)  # noqa: E731
            lines.append(f"| {g['task']} | {g['prefix_s'] or '—'} | {g['mode']} | {cell('cer')} | "
                         f"{cell('onset_f1_50ms')} | {cell('note_f1_50ms')} | {cell('abs_log_density_error')} | "
                         f"{cell('type_mix_js_vs_human')} | {cell('ioi_js_vs_human')} |")
        lines += ['', 'CER/JS: lower is better (negative diff = history helps); F1: higher is better.']
    return '\n'.join(lines) + '\n'


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--local-smoke', action='store_true')
    p.add_argument('--plan', type=Path)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--weights', choices=('raw', 'ema'), default='raw')
    p.add_argument('--device', default='cpu')
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--protocol', type=Path, help='frozen continuation teacher panel (default: freeze from the tape)')
    p.add_argument('--windows', type=int, default=200)
    p.add_argument('--seed', type=int, default=0, help='sampling seed (per-row generators)')
    p.add_argument('--panel-seed', type=int, default=20261102)
    p.add_argument('--temperature', default='1', help="sampling temperature, or 'none' for greedy")
    p.add_argument('--top-p', type=float, default=None)
    p.add_argument('--token-cap', type=int, default=4096, help='per 8 s core')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--teacher-batch-size', type=int, default=8)
    p.add_argument('--max-wall-s', type=float, default=3600.)
    p.add_argument('--prefix-seconds', default='30,60')
    p.add_argument('--max-charts-per-mode', type=int, default=None)
    p.add_argument('--skip-teacher', action='store_true')
    p.add_argument('--skip-generation', action='store_true')
    a = p.parse_args()
    if not a.local_smoke:
        missing = [k for k in ('plan', 'checkpoint') if getattr(a, k) is None]
        if missing:
            p.error('checkpoint mode needs --'+', --'.join(missing))
    out = a.out_dir
    if (out/'continuation-benchmark.json').exists():
        raise FileExistsError(f'{out} already has results; never overwritten')
    out.mkdir(parents=True, exist_ok=True)
    model, panel, identity, intensity_dim, cell_us = smoke_setup(a) if a.local_smoke else checkpoint_setup(a)
    report = dict(schema='continuation-benchmark-v1', model=identity, device=a.device,
                  started_unix=time.time(), meaning=cb.MEANING)
    if not a.skip_teacher:
        if a.protocol:
            protocol = json.loads(a.protocol.read_text())
            protocol_path = a.protocol
        else:
            tape = TAPE if a.local_smoke else identity['tape']
            protocol = cb.freeze_continuation_panel(panel, windows=a.windows, seed=a.panel_seed,
                tape_sha256=sha_file(a.root/tape) if (a.root/tape).is_file() else None)
            if a.local_smoke:
                protocol['index_space'] = 'tape filtered to locally available audios (smoke only)'
            protocol_path = out/'continuation-teacher-panel.json'
            protocol_path.write_text(json.dumps(protocol, indent=1)+'\n')
        result = cb.run_continuation_teacher(model, panel, protocol, device=a.device, batch_size=a.teacher_batch_size)
        report['teacher_forced'] = dict(readout='teacher_forced (primary)', protocol=str(protocol_path),
                                        protocol_sha256=sha_file(protocol_path), **result)
        print(json.dumps(dict(phase='teacher_done', windows=result['summary']['measured_windows'],
                              elapsed_s=result['summary']['elapsed_s'])), flush=True)
    if not a.skip_generation:
        from scripts.run_workflow_benchmark import make_generator, parse_temperature
        temperature = parse_temperature(a.temperature)
        provider = None
        if intensity_dim:
            from beatfcn.train.density_curve import grid_density_curve
            provider = lambda audio, s, e: grid_density_curve([], s, e, cell_us=cell_us, scale=16.)  # noqa: E731
        max_charts = a.max_charts_per_mode if a.max_charts_per_mode is not None else (2 if a.local_smoke else None)
        prefixes = tuple(int(x) for x in a.prefix_seconds.split(',') if x)
        jobs, truths = cb.generation_jobs(panel, prefix_seconds=prefixes, seed=a.panel_seed,
                                          max_charts_per_mode=max_charts)
        states, wall = cb.rollout_jobs(model, panel.features, jobs, device=a.device,
            generator=make_generator(temperature, a.top_p, a.seed), batch_size=a.batch_size,
            token_cap=a.token_cap, max_wall_s=a.max_wall_s, intensity_provider=provider)
        scored = [cb.score_job(s, truths[cb.truth_key(s)], s['visible_events']) for s in states]
        report['free_generation'] = dict(
            readout='free_generation (whole-song continuation + infill; conditioned vs unconditioned twin)',
            decoding=dict(rule='greedy' if temperature is None else 'ancestral', temperature=temperature,
                          top_p=a.top_p, seed=a.seed, token_cap_per_core=a.token_cap,
                          rng='per-row generator from (seed, mode, duration, core start): twins share draws',
                          conditioning='unconditional controls + unspecified density curve; only the reference differs'),
            prefix_seconds=list(prefixes), history_s=cb.HISTORY_US/1e6, suffix_s=cb.SUFFIX_US/1e6,
            jobs=len(jobs), wall_s=wall, rows=scored, summary=cb.summarize_generation(scored),
            drafts={s['job_id']: [[e.onset_us, e.end_us, e.type, e.lane] for e in s['generated']] for s in states},
            cores={s['job_id']: s['cores'] for s in states})
        print(json.dumps(dict(phase='generation_done', jobs=len(jobs), wall_s=wall)), flush=True)
    report['finished_unix'] = time.time()
    (out/'continuation-benchmark.json').write_text(json.dumps(report, indent=1, default=str)+'\n')
    (out/'summary.md').write_text(markdown(report))
    print((out/'summary.md').read_text())


if __name__ == '__main__':
    main()
