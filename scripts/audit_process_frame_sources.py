"""Frozen feature-source permutation for the two complete group processes.

CPU only, no optimizer or primary-score replacement. Physical address/support,
explicit phase, noise schedule/history and RNG seed stay fixed. Encoded frame
sources are permuted before probabilistic generation, not emitted events.
"""
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from beatfcn.model.full_event_model import FullEventConfig
from beatfcn.model.grouped_chart_process import GroupedChartProcess, generate_grouped_process
from beatfcn.model.local_group_diffusion import GroupDiffusionCanvas, sample_group_diffusion
from beatfcn.model.sample import match_f1
from beatfcn.train.canonical_packages import canonical_batch, explicit_reference_row
from beatfcn.train.canonical_probability import probability_batch, explicit_row
from scripts.run_frontend_canvas_comparison import setup
from scripts.run_local_portfolio_scale import row_factory, sha, write
from scripts.run_local_typed_canvas import score
from scripts.review_frontend_canvas_comparison import metrics


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'bench/rhythm-v2/local-process-frame-sources-v1.json'
REPORT = ROOT / 'reports/local_process_frame_sources_v1.json'
ART = ROOT / 'artifacts/local-process-frame-sources-v1'


def frame_permutation(inputs, seed):
    valid = inputs['canvas_valid'][0].cpu().numpy()
    allowed = inputs['canvas_head_allowed'][0].cpu().numpy()
    groups = defaultdict(list)
    for f in np.flatnonzero(valid):
        groups[allowed[f].tobytes()].append(int(f))
    result = np.arange(len(valid))
    rng = np.random.default_rng(seed)
    for indices in groups.values():
        result[indices] = rng.permutation(indices)
    return result


def install_permutation(model, kind, permutation):
    if kind == 'diffusion':
        original = model.encode_audio
        model.encode_audio = lambda inputs: original(inputs)[:, permutation]
        return lambda: setattr(model, 'encode_audio', original)
    original = model.context
    def changed(inputs, request):
        value = original(inputs, request)
        x = value['x'][permutation]
        head, frame = value['head'], value['frame']
        seconds = (head - request.start_us).float() / 1e6
        phase = torch.stack((seconds / 8, torch.sin(seconds * np.pi), torch.cos(seconds * np.pi),
                             ((head - request.start_us) % 1000).float() / 1000), -1)
        return dict(value, x=x, keys=model.address_key(x[frame]) + model.physical_phase(phase),
                    pooled=(x * inputs['canvas_valid'][0, :, None]).sum(0) / inputs['canvas_valid'].sum().clamp_min(1))
    model.context = changed
    return lambda: setattr(model, 'context', original)


@torch.no_grad()
def main():
    if PLAN.exists() or REPORT.exists():
        raise FileExistsError('No null seed/temperature search or retry')
    torch.set_num_threads(2)
    paths = dict(diffusion='local-group-diffusion-quality-v1', grouped='local-grouped-chart-v2')
    plan = dict(id='local-process-frame-sources-v1', seed=20261076, sample_id=0,
        model_forwards='64requests permodel: matchedCPUcorrect+permuted, existing sample0 fixed before this null',
        optimizer_updates=0, device='cpu', wall_s=1200,
        source_sha256={n: sha(ROOT / n) for n in ('scripts/audit_process_frame_sources.py',
            'beatfcn/model/local_group_diffusion.py', 'beatfcn/model/grouped_chart_process.py')},
        contracts={k: dict(plan_sha256=sha(ROOT / f'bench/rhythm-v2/{ident}.json'),
            checkpoint_sha256=sha(ROOT / f'artifacts/{ident}/quality-1000.pt')) for k, ident in paths.items()},
        interpretation='Address-source dependence, not a selective raw-audio intervention: encoded features include positional/context information. Conditional generation can change global count/type marginals, which are reported rather than assumed preserved. Compare matched CPUcorrect and CPUpermuted sample0; do not compare a single draw to the fourdraw primary mean.',
        primary_decisions_unchanged=True, confirmation_scored=False, cloud_compute_calls=0)
    write(PLAN, plan)
    panel, store = setup(json.loads((ROOT / 'bench/rhythm-v2/local-frontend-comparison-v2.json').read_text()))
    row = row_factory(panel, store)
    report = dict(id=plan['id'], status='running', models={}, optimizer_updates=0,
                  device='cpu', cloud_compute_calls=0, confirmation_scored=False)
    write(REPORT, report)
    started = time.monotonic()
    try:
        for kind, ident in paths.items():
            source = json.loads((ROOT / f'bench/rhythm-v2/{ident}.json').read_text())
            cls = GroupDiffusionCanvas if kind == 'diffusion' else GroupedChartProcess
            model = cls(FullEventConfig(**source['model']), density_reference=source['density_reference']).eval()
            saved = torch.load(ROOT / f'artifacts/{ident}/quality-1000.pt', map_location='cpu', weights_only=False)
            if saved['updates'] != 1000 or saved['plan_sha256'] != plan['contracts'][kind]['plan_sha256']:
                raise ValueError('Exact fixed checkpoint required')
            model.load_state_dict(saved['model']); del saved
            current = dict(correct=[], permuted=[])
            for index in source['indices']['dev']:
                e = (explicit_reference_row if kind == 'diffusion' else explicit_row)(row(index, 'draft'))
                inputs, _ = (canonical_batch if kind == 'diffusion' else probability_batch)([{**e, 'targets': []}], 'cpu')
                permutation = frame_permutation(inputs, plan['seed'] + 17 * index)
                for variant in ('correct', 'permuted'):
                    if time.monotonic() - started > plan['wall_s']:
                        raise TimeoutError('Source permutation wall')
                    undo = install_permutation(model, kind, permutation) if variant == 'permuted' else lambda: None
                    try:
                        generate = sample_group_diffusion if kind == 'diffusion' else generate_grouped_process
                        events, _ = generate(model, inputs, e['request'], seed=source['seed'] + 17 * index,
                                             deadline=started + plan['wall_s'])
                        prediction = dict(complete=True, events=[asdict(v) for v in events])
                    except (ValueError, RuntimeError, TimeoutError) as exc:
                        events = []; prediction = dict(complete=False, events=[], failure=str(exc))
                    finally:
                        undo()
                    start = e['request'].start_us; width = e['request'].end_us - start
                    shifts = [match_f1([start + (v.onset_us - start + d) % width for v in events],
                                      [v.onset_us for v in e['targets']], 10000)[3] for d in source['shifts_us']]
                    current[variant].append(dict(index=index, component=panel.records[index]['component'],
                        request=asdict(e['request']), prediction=prediction, target=[asdict(v) for v in e['targets']],
                        scores=score(events, e['targets']), shifted_onset10=shifts,
                        frame_source_permutation=permutation.tolist() if variant == 'permuted' else None))
                    write(ART / f'{kind}-{variant}.json', current[variant])
            values = {name: metrics(records) for name, records in current.items()}
            report['models'][kind] = dict(results=values,
                onset10_delta=values['correct']['onset10'] - values['permuted']['onset10'],
                positive_components=sum(values['correct']['components'][c]['onset10'] > v['onset10']
                    for c, v in values['permuted']['components'].items()))
            write(REPORT, report)
        report.update(status='completed', wall_s=time.monotonic() - started)
        write(REPORT, report)
    except BaseException as exc:
        report.update(status='stopped', failure=f'{type(exc).__name__}: {exc}', wall_s=time.monotonic() - started)
        write(REPORT, report)
        raise


if __name__ == '__main__':
    main()
