"""Paired/interleaved final evaluation of immutable local-bundle checkpoints.

Same frozen cases, conditions, token limits and total wall bound as the training
runner. Interleaving avoids spending the shared evaluation budget on one arm
before the other arm can be attempted. No training or checkpoint mutation.
"""
import json
import time

import torch

from scripts.run_local_bundle import (ROOT, PLAN, REPORT, ART, ARMS, configure,
    panel_for, write, sha, log, generation, summarize)
from beatfcn.model.full_event_model import FullEventConfig, FullEventModel
from beatfcn.model.local_bundle import LocalBundleModel


def main():
    plan = json.loads(PLAN.read_text()); report = json.loads(REPORT.read_text())
    if report['status'] != 'trained':
        raise ValueError('Final paired evaluation requires completed matched training')
    configure(); panel = panel_for(plan); models = {}
    for arm in ARMS:
        saved = torch.load(ART / arm / f"step-{plan['steps']:04d}.pt", map_location='cpu', weights_only=False)
        if saved['updates'] != plan['steps'] or saved['plan_sha256'] != sha(PLAN):
            raise ValueError('Checkpoint identity mismatch')
        model = (LocalBundleModel if arm == 'bundle' else FullEventModel)(FullEventConfig(**plan['model']))
        model.load_state_dict(saved['model']); models[arm] = model.to('mps').eval()
        del saved
    cases = {
        'draft': (plan['indices']['val'], 'draft', None),
        'completion': (plan['indices']['val'][::2], 'middle', None),
        'density_low': (plan['indices']['val'][::4], 'draft', {'density_notes_per_second': [2., 3.]}),
        'density_high': (plan['indices']['val'][::4], 'draft', {'density_notes_per_second': [6., 8.]})}
    outputs = {arm: {task: [] for task in cases} for arm in ARMS}
    if any((ART / arm / f'{task}-final.json').exists() for arm in ARMS for task in cases):
        raise FileExistsError('Preserve previous final evaluation artifacts')
    report['evaluation_protocol'] = 'Frozen cases/caps/bands, round-robin tasks and alternating arm order; no outcome-based selection'
    report['evaluation_source_sha256'] = {'scripts/evaluate_local_bundle.py': sha(__file__)}
    report['status'] = 'evaluating'; write(REPORT, report)
    started = time.monotonic(); deadline = started + plan['evaluation_wall_s']
    pair_number = 0
    try:
        for position in range(max(len(v[0]) for v in cases.values())):
            for task, (indices, view, bands) in cases.items():
                if position >= len(indices):
                    continue
                index = indices[position]
                order = ARMS if pair_number % 2 == 0 else tuple(reversed(ARMS))
                for arm in order:
                    result = generation(models[arm], panel, [index], view=view, manual_bands=bands,
                        token_cap=plan['generation_token_cap'], deadline=deadline)
                    outputs[arm][task].extend(result)
                    write(ART / arm / f'{task}-final.json', outputs[arm][task])
                pair_number += 1
                log(dict(stage='paired_evaluation', task=task, index=index,
                    results={arm: summarize(outputs[arm][task][-1:]) for arm in ARMS},
                    wall_s=time.monotonic() - started))
                report['evaluation_progress'] = {arm: {task: len(rows) for task, rows in tasks.items()}
                                                 for arm, tasks in outputs.items()}
                write(REPORT, report)
        report['final'] = {arm: {task: summarize(rows) for task, rows in tasks.items()}
                           for arm, tasks in outputs.items()}
        report['evaluation_wall_s'] = time.monotonic() - started
        report['status'] = 'evaluated'; write(REPORT, report)
    except BaseException as exc:
        report['status'] = 'evaluation_stopped'
        report['evaluation_wall_s'] = time.monotonic() - started
        report['failures'].append(f'{type(exc).__name__}: {exc}')
        write(REPORT, report)
        raise


if __name__ == '__main__':
    main()
