"""Zero-update, fixed-seed sampling audit of a completed package checkpoint.

The primary training endpoint remains its original greedy count-mode result.
This secondary audit was designed after observing its sparse1000 output.
"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from beatfcn.model.full_event_model import FullEventConfig
from beatfcn.model.local_event_package import EventPackageCanvas
from beatfcn.model.local_package_sampling import sample_package
from beatfcn.model.sample import match_f1
from scripts.run_frontend_canvas_comparison import setup
from scripts.run_local_portfolio_candidate_quality import batch
from scripts.run_local_portfolio_mamba_health import sha, write
from scripts.run_local_portfolio_scale import row_factory
from scripts.run_local_typed_canvas import score
from scripts.review_frontend_canvas_comparison import metrics


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'bench/rhythm-v2/local-package-readout-v1.json'
REPORT = ROOT / 'reports/local_package_readout_v1.json'
PARENT = ROOT / 'bench/rhythm-v2/local-portfolio-package-quality-v1.json'
ART = ROOT / 'artifacts/local-package-readout-v1'


def prepare():
    if PLAN.exists():
        raise FileExistsError('Preserve readout precommitment')
    parent = json.loads(PARENT.read_text())
    plan = dict(id='local-package-readout-v1', parent_plan_sha256=sha(PARENT),
        model=parent['model'], indices={k: parent['indices'][k] for k in ('dev', 'empty_dev')},
        density_reference=parent['density_reference'], shifts_us=parent['shifts_us'],
        samples=4, seed_formula='20261041 + 17*index + sample_id',
        permutation_seed_formula='20261043 + index',
        wall_s=1200, optimizer_updates=0, cloud_compute_calls=0, confirmation_scored=False,
        scope='Secondary frozen2000 readout audit designed after observing1000 sparse outputs; not a retroactive change of the primary greedy quality endpoint.',
        hypothesis='The learned normalized count distribution may retain temporally located probability below its per-group mode. Test temperature1 whole-package ancestral generation against fixed frame-permutation and count-preserving onset shifts.',
        decision='Report sample0 and all four equally weighted; no sample/temperature/threshold search, no quality-training authorization implied. A gain only in event count is insufficient evidence of acoustic localization.',
        source_sha256={name: sha(ROOT / name) for name in (
            'scripts/probe_local_package_readout.py', 'beatfcn/model/local_package_sampling.py',
            'tests/test_local_package_sampling.py', 'beatfcn/model/local_event_package.py',
            'beatfcn/model/canvas_diffusion.py', 'scripts/run_local_portfolio_candidate_quality.py',
            'scripts/review_frontend_canvas_comparison.py',
            'scripts/run_local_typed_canvas.py', str(PARENT.relative_to(ROOT)))})
    write(PLAN, plan)
    with (ROOT / 'EXPERIMENT_LOG.md').open('a') as f:
        f.write('\n\nlocal-package-readout-v1预注册：在看到事件包1000步稀疏后，次要零更新T1全包采样检查；只用未来固定2000终态，不替换原mode主终点。4固定种子均报、帧置换及7保数量位移零模型；开发32窗+2空窗，1200秒、确认未评分、云0。不能以多发音符代替声学增益，不能择样本/温度或改变训练。\n')
    print('prepared', PLAN.relative_to(ROOT))


@torch.no_grad()
def run():
    if REPORT.exists():
        raise FileExistsError('Preserve the readout audit')
    plan = json.loads(PLAN.read_text())
    for path, expected in plan['source_sha256'].items():
        if sha(ROOT / path) != expected:
            raise ValueError('Readout source drift: ' + path)
    quality_path = ROOT / 'reports/local_portfolio_package_quality_v1.json'
    quality = json.loads(quality_path.read_text())
    if quality['status'] != 'completed' or quality['updates'] != 2000:
        raise ValueError('Require completed fixed2000 quality result')
    if quality['plan_sha256'] != plan['parent_plan_sha256'] or quality['health_weights_used']:
        raise ValueError('Parent provenance mismatch')
    checkpoint = ROOT / 'artifacts/local-portfolio-v1/package-quality/quality-2000.pt'
    if sha(checkpoint) != quality['final_weight_sha256']:
        raise ValueError('Parent weight identity mismatch')
    torch.set_num_threads(4)
    torch.mps.set_per_process_memory_fraction(.5)
    started = time.monotonic()
    deadline = started + plan['wall_s']
    model = EventPackageCanvas(FullEventConfig(**plan['model']), density_reference=plan['density_reference'])
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if saved['updates'] != 2000 or saved['plan_sha256'] != plan['parent_plan_sha256']:
        raise ValueError('Checkpoint metadata mismatch')
    model.load_state_dict(saved['model'])
    del saved
    model = model.to('mps').eval()
    panel, store = setup(json.loads((ROOT / 'bench/rhythm-v2/local-frontend-comparison-v2.json').read_text()))
    row = row_factory(panel, store)
    report = dict(id=plan['id'], status='running', optimizer_updates=0,
        checkpoint_sha256=sha(checkpoint), plan_sha256=sha(PLAN), cases=[], summaries={},
        failures=[], cloud_compute_calls=0, confirmation_scored=False)
    write(REPORT, report)
    collections = {f'{task}-{kind}-{s}': [] for task in ('draft', 'empty')
                   for kind in ('sample', 'permuted') for s in range(plan['samples'])}
    try:
        for task, indices in [('draft', plan['indices']['dev']), ('empty', plan['indices']['empty_dev'])]:
            for index in indices:
                if time.monotonic() > deadline:
                    raise TimeoutError('Frozen readout wall')
                example = row(index, 'draft', 0)
                inputs, target = batch(example, 'package')
                output = model(inputs)
                probability = output['count'][0].float().cpu().numpy()
                valid = inputs['canvas_valid'][0].cpu().numpy()
                values = np.arange(probability.shape[-1])
                expected = float((probability[valid] * values).sum())
                mode = int(probability[valid].argmax(-1).sum())
                case = dict(index=index, task=task, component=panel.records[index]['component'],
                    expected_count=expected, mode_count=mode, target_count=len(example['targets']))
                report['cases'].append(case)
                ART.mkdir(parents=True, exist_ok=True)
                torch.save({k: v.float().cpu() for k, v in output.items()}, ART / f'probability-{index}.pt')
                permutation = np.arange(len(valid))
                addresses = np.flatnonzero(valid)
                permutation[addresses] = np.random.default_rng(20261043 + index).permutation(addresses)
                for sample_id in range(plan['samples']):
                    for kind in ('sample', 'permuted'):
                        failure = None
                        try:
                            events = sample_package(model, inputs, example['request'],
                                seed=20261041 + 17 * index + sample_id, output=output,
                                frame_permutation=permutation if kind == 'permuted' else None,
                                deadline=deadline)
                        except (ValueError, RuntimeError, TimeoutError) as exc:
                            failure = f'{type(exc).__name__}: {exc}'
                            events = []
                        request = example['request']
                        width = request.end_us - request.start_us
                        shifts = [match_f1([request.start_us + (e.onset_us - request.start_us + d) % width
                            for e in events], [e.onset_us for e in example['targets']], 10000)[3]
                            for d in plan['shifts_us']]
                        record = dict(index=index, component=panel.records[index]['component'],
                            request=asdict(request), prediction=dict(complete=failure is None,
                                failure=failure, events=[asdict(e) for e in events]),
                            target=[asdict(e) for e in example['targets']],
                            scores=score(events, example['targets']), shifted_onset10=shifts)
                        key = f'{task}-{kind}-{sample_id}'
                        collections[key].append(record)
                        write(ART / f'{key}.json', collections[key])
                write(REPORT, report)
                print(json.dumps(case), flush=True)
        report['summaries'] = {k: metrics(rows) for k, rows in collections.items()}
        report['four_sample_means'] = {kind: {metric: sum(report['summaries'][f'draft-{kind}-{s}'][metric]
            for s in range(plan['samples'])) / plan['samples']
            for metric in ('full_note_f1', 'onset10', 'onset50', 'density_ratio', 'shift_null10')}
            for kind in ('sample', 'permuted')}
        report.update(status='completed', wall_s=time.monotonic() - started)
        report['artifact_sha256'] = {str(p.relative_to(ROOT)): sha(p) for p in ART.iterdir() if p.is_file()}
        write(REPORT, report)
    except BaseException as exc:
        report.update(status='stopped', wall_s=time.monotonic() - started)
        report['failures'].append(f'{type(exc).__name__}: {exc}')
        write(REPORT, report)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'run'))
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '1':
        raise ValueError('No silent backend fallback')
    (prepare if parser.parse_args().stage == 'prepare' else run)()
