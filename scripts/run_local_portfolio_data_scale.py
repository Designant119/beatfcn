"""Quality contrast for the two small arms that already passed scale-v1 health.

The conjunctive health gate stopped local-portfolio-scale-v1 before any quality
update. This run does not resume that process, does not train capacity64, and
does not score the sealed confirmation split.
"""
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import time

import torch

from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.sample import match_f1
from beatfcn.train.relative_canvas import generate_relative, relative_batch
from beatfcn.model.local_typed_canvas import canvas_loss
from scripts.run_frontend_canvas_comparison import setup
from scripts.run_local_portfolio_scale import fresh, row_factory, weight_digest
from scripts.run_local_typed_canvas import score
from scripts.review_frontend_canvas_comparison import metrics


ROOT = Path(__file__).resolve().parents[1]
PARENT_PLAN = ROOT / 'bench/rhythm-v2/local-portfolio-scale-v1.json'
PARENT_REPORT = ROOT / 'reports/local_portfolio_scale_v1.json'
PARENT_HEALTH = ROOT / 'artifacts/local-portfolio-v1/scale/health-predictions.json'
PARENT_CHECKPOINTS = ROOT / 'artifacts/local-portfolio-v1/scale'
PLAN = ROOT / 'bench/rhythm-v2/local-portfolio-data-scale-v1.json'
REPORT = ROOT / 'reports/local_portfolio_data_scale_v1.json'
ART = ROOT / 'artifacts/local-portfolio-v1/data-scale'
ARMS = ('small24', 'small64')
LIVE = {}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def save(report):
    write(REPORT, report)


def journal(message):
    with (ROOT / 'EXPERIMENT_LOG.md').open('a') as handle:
        handle.write('\n\nlocal-portfolio-data-scale-v1 ' + message + '\n')


def window_passed(item):
    prediction = item['prediction']
    events = prediction.get('events') or []
    target = item['target']
    scores = item['scores']
    if target:
        onset = (scores.get('onset_10ms') or {}).get('f1') or 0
        ratio = len(events) / len(target)
        return bool(prediction.get('complete') and scores.get('full_note_f1', 0) >= .75
            and onset >= .8 and .5 <= ratio <= 2)
    return bool(prediction.get('complete') and not events)


def arm_passed(values):
    if not values or not all(window_passed(item) for item in values):
        return False
    positive = [item['scores']['full_note_f1'] for item in values if item['target']]
    return bool(positive) and sum(positive) / len(positive) >= .9


def cite_saved_health(report, predictions, checkpoint_sha):
    if report.get('id') != 'local-portfolio-scale-v1' or report.get('status') != 'health_failed':
        raise ValueError('Need the stopped scale-v1 health result')
    if report.get('health_updates') != 100 or report.get('health_passed') is not False:
        raise ValueError('Scale-v1 health record is not the expected stop')
    if report.get('updates') != {'small24': 0, 'small64': 0, 'capacity64': 0}:
        raise ValueError('Scale-v1 already has quality updates')
    initial = report['initial_weight_sha256']
    if initial['small24'] != initial['small64']:
        raise ValueError('Saved small-arm initial weights differ')
    if checkpoint_sha['small24'] != checkpoint_sha['small64']:
        raise ValueError('Saved small-arm health checkpoints differ')
    if not arm_passed(predictions['small24']) or not arm_passed(predictions['small64']):
        raise ValueError('Saved small-arm health predictions do not pass the original gate')
    if arm_passed(predictions['capacity64']):
        raise ValueError('Capacity arm passed; do not replace the conjunctive run with this contrast')
    return dict(parent_plan_sha256=report['plan_sha256'], initial_weight_sha256=initial['small24'],
        health_checkpoint_sha256=checkpoint_sha['small24'], health_updates=100,
        capacity64_blocked_quality=True)


def load_parent():
    if PLAN.exists() or REPORT.exists():
        raise FileExistsError('Preserve the data-scale run')
    parent = json.loads(PARENT_PLAN.read_text())
    report = json.loads(PARENT_REPORT.read_text())
    predictions = json.loads(PARENT_HEALTH.read_text())
    if sha(PARENT_PLAN) != report['plan_sha256'] or parent['id'] != 'local-portfolio-scale-v1':
        raise ValueError('Scale-v1 plan identity drift')
    if parent['steps'] != 2000 or set(parent['sample_stream']) != {'train24', 'train64'}:
        raise ValueError('Unexpected scale-v1 stream')
    for track, stream in parent['sample_stream'].items():
        if len(stream) != parent['steps'] or any(len(step) != 2 for step in stream):
            raise ValueError(f'Incomplete {track} stream')
    checkpoint_sha = {arm: sha(PARENT_CHECKPOINTS / arm / 'health-0100.pt') for arm in ARMS}
    citation = cite_saved_health(report, predictions, checkpoint_sha)
    plan = dict(id='local-portfolio-data-scale-v1', seed=parent['seed'], steps=parent['steps'],
        effective_batch=parent['effective_batch'], models={arm: parent['models']['small24'] for arm in ARMS},
        density_reference=parent['density_reference'], indices=parent['indices'],
        learning_rate=parent['learning_rate'], minimum_learning_rate=parent['minimum_learning_rate'],
        warmup=parent['warmup'], weight_decay=0, clip=1,
        quality_checkpoints=parent['quality_checkpoints'],
        diagnostic_checkpoints=parent['diagnostic_checkpoints'],
        training_wall_s=3600, evaluation_wall_s=1200, cloud_compute_calls=0,
        shifts_us=parent['shifts_us'], views=parent['views'],
        predecessor='local-portfolio-scale-v1', health_citation=citation,
        causal_scope='small24 against small64 only: same architecture, seed, objective, and schedule; train24 versus train64 streams. Not a capacity result.',
        confirmation_policy='confirm indices stay sealed and are not scored.',
        excluded='capacity64 receives no further updates in this id.',
        source_sha256={name: sha(ROOT / name) for name in (
            'scripts/run_local_portfolio_data_scale.py', 'scripts/run_local_portfolio_scale.py',
            'bench/rhythm-v2/local-portfolio-scale-v1.json')})
    write(PLAN, plan)
    return parent, plan


def checkpoint(model, opt, plan, arm, stage, step):
    path = ART / arm / f'{stage}-{step:04d}.pt'
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict(), config=plan['models'][arm],
        plan_sha256=sha(PLAN), stage=stage, updates=step, cpu_rng=torch.get_rng_state(),
        mps_rng=torch.mps.get_rng_state()), path)


def evaluate(models, plan, row, tag, deadline, full=False):
    tasks = {'draft': [(i, 'draft', None, False) for i in plan['indices']['dev']],
        'empty': [(i, 'draft', None, False) for i in plan['indices']['empty_dev']]}
    if full:
        sample = plan['indices']['dev'][::2]
        tasks.update(completion=[(i, 'middle', None, False) for i in sample],
            reference_masked=[(i, 'middle', None, True) for i in sample],
            density_low=[(i, 'draft', (2., 3.), False) for i in sample[::4]],
            density_high=[(i, 'draft', (6., 8.), False) for i in sample[::4]])
    summary = {}
    for arm, model in models.items():
        summary[arm] = {}
        for task, specs in tasks.items():
            records = []
            for index, view, bands, masked in specs:
                if time.monotonic() > deadline:
                    raise TimeoutError('Frozen evaluation wall')
                example = row(index, view, 0, bands)
                inputs, _ = relative_batch([example], 'mps', mask_reference=masked)
                prediction = generate_relative(model, inputs, example['request'], deadline=deadline)
                predicted = [RhythmEvent(**event) for event in prediction['events']]
                scores = score(predicted, example['targets'])
                scores['extra_onset_tolerances'] = {str(ms): match_f1(
                    [event.onset_us for event in predicted], [event.onset_us for event in example['targets']], ms * 1000)[3]
                    for ms in (1, 5, 25)}
                start = example['request'].start_us
                width = example['request'].end_us - start
                shifts = [match_f1([start + (event.onset_us - start + delta) % width for event in predicted],
                    [event.onset_us for event in example['targets']], 10000)[3] for delta in plan['shifts_us']]
                records.append(dict(index=index, component=LIVE['panel'].records[index]['component'], view=view,
                    request=asdict(example['request']), prediction=prediction,
                    target=[asdict(event) for event in example['targets']], scores=scores, shifted_onset10=shifts))
                write(ART / arm / f'{tag}-{task}.json', records)
            summary[arm][task] = metrics(records)
    return summary


def train(parent, plan, report, row):
    models = {arm: fresh(plan, arm) for arm in ARMS}
    opts = {arm: torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=0) for arm, model in models.items()}
    LIVE.update(models=models, opts=opts, stage='quality')
    report['quality_initial_weight_sha256'] = {arm: weight_digest(model) for arm, model in models.items()}
    expected = plan['health_citation']['initial_weight_sha256']
    if any(digest != expected for digest in report['quality_initial_weight_sha256'].values()):
        raise ValueError('Fresh weights do not match the scale-v1 small-arm initialization')
    if report['quality_initial_weight_sha256']['small24'] != report['quality_initial_weight_sha256']['small64']:
        raise ValueError('Small controls diverged before the first update')
    started = time.monotonic()
    evaluation_s = 0.
    report['status'] = 'training'
    save(report)
    for step in range(1, plan['steps'] + 1):
        if time.monotonic() - started > plan['training_wall_s']:
            raise TimeoutError('Frozen training wall includes evaluations')
        if step <= plan['warmup']:
            learning_rate = plan['learning_rate'] * step / plan['warmup']
        else:
            progress = (step - plan['warmup']) / (plan['steps'] - plan['warmup'])
            learning_rate = plan['minimum_learning_rate'] + (plan['learning_rate'] - plan['minimum_learning_rate']) * (1 + math.cos(math.pi * progress)) / 2
        values = {}
        for arm, model in models.items():
            track = 'train24' if arm == 'small24' else 'train64'
            specs = parent['sample_stream'][track][step - 1]
            model.train()
            opt = opts[arm]
            opt.zero_grad(set_to_none=True)
            for group in opt.param_groups:
                group['lr'] = learning_rate
            total = 0.
            for spec in specs:
                example = row(spec['index'], spec['view'], spec['field_index'])
                inputs, target = relative_batch([example], 'mps')
                loss, _ = canvas_loss(model(inputs), target, inputs)
                (loss / 2).backward()
                total += float(loss.detach()) / 2
                exposure = report['exposure'][arm]
                exposure['windows'] += 1
                exposure['events'] += len(example['targets'])
                key = str(spec['index'])
                exposure['indices'][key] = exposure['indices'].get(key, 0) + 1
                for event in example['targets']:
                    exposure['types'][event.type] = exposure['types'].get(event.type, 0) + 1
            gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), plan['clip'], error_if_nonfinite=True))
            opt.step()
            report['updates'][arm] = step
            values[arm] = dict(loss=total, preclip_gradient_norm=gradient)
        if step == 1 or step % 25 == 0:
            point = dict(step=step, learning_rate=learning_rate, arms=values, wall_s=time.monotonic() - started)
            report['curves'].append(point)
            report['wall_s']['training'] = point['wall_s']
            save(report)
            print(json.dumps(dict(stage='quality', **point)), flush=True)
        if step in plan['quality_checkpoints'] + plan['diagnostic_checkpoints']:
            for arm in ARMS:
                checkpoint(models[arm], opts[arm], plan, arm, 'quality', step)
        if step in plan['quality_checkpoints']:
            moment = time.monotonic()
            deadline = min(started + plan['training_wall_s'], moment + plan['evaluation_wall_s'] - evaluation_s)
            report['evaluations'][str(step)] = evaluate(models, plan, row, f'quality-{step:04d}', deadline, full=step == plan['steps'])
            evaluation_s += time.monotonic() - moment
            save(report)
    report.update(status='completed', complete=True, confirmation_scored=False)
    report['wall_s'].update(training=time.monotonic() - started, evaluation=evaluation_s)
    report['final_weight_sha256'] = {arm: sha(ART / arm / 'quality-2000.pt') for arm in ARMS}
    save(report)
    journal(f'终态：updates={report["updates"]}，确认集未评分，云0；结果{REPORT.relative_to(ROOT)}。')


def main():
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '1':
        raise ValueError('No silent backend fallback')
    torch.set_num_threads(4)
    torch.mps.set_per_process_memory_fraction(.5)
    parent, plan = load_parent()
    comparison = json.loads((ROOT / 'bench/rhythm-v2/local-frontend-comparison-v2.json').read_text())
    panel, store = setup(comparison)
    LIVE['panel'] = panel
    row = row_factory(panel, store)
    report = dict(id=plan['id'], status='prepared', plan_sha256=sha(PLAN),
        predecessor_plan_sha256=plan['health_citation']['parent_plan_sha256'],
        parameters={arm: parent['parameters']['small24'] for arm in ARMS},
        updates={arm: 0 for arm in ARMS}, curves=[], evaluations={}, failures=[], wall_s={},
        cloud_compute_calls=0, confirmation_scored=False,
        exposure={arm: dict(windows=0, events=0, indices={}, types={}) for arm in ARMS})
    save(report)
    journal('冻结：沿用 scale-v1 的小模型种子和 24/64 流，只训练已过健康门的两臂。capacity64 不续训，确认集不评分，云0。')

    def stop(*_):
        raise KeyboardInterrupt('External stop')

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        train(parent, plan, report, row)
    except BaseException as exc:
        report.setdefault('failures', []).append(dict(stage=LIVE.get('stage'), error=f'{type(exc).__name__}: {exc}',
            updates=dict(report['updates'])))
        report['status'] = 'stopped'
        for arm, model in LIVE.get('models', {}).items():
            checkpoint(model, LIVE['opts'][arm], plan, arm, 'partial-quality', report['updates'][arm])
        save(report)
        journal(f'停止：{report["failures"][-1]}；不自动延长，不改 scale-v1。')
        raise


if __name__ == '__main__':
    main()
