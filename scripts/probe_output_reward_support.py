"""Read-only T=1 outcome-reward support on a declared archived policy.

No optimizer exists. Every selected request and capped/empty outcome is retained.
Reward variation is not a quality result or permission to resume this policy.
"""
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import torch

from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.full_event_model import FullEventConfig, FullEventModel
from beatfcn.model.full_event_generate import generate_batch
from beatfcn.model.local_song_rollout import canonical
from beatfcn.train.condition_outcome import score_condition_outcome
from beatfcn.train.condition_scst import sample_trajectory
from beatfcn.train.local_request_panel import LocalRequestPanel, event_request, request_inputs
from beatfcn.train.local_shapes import pad_input_shapes
from scripts.audit_local_candidate_conditions import request_legality
from scripts.prepare_local_data_scale_probe import sha, padding_shapes
from scripts.run_local_song_baseline import write_json

ROOT = Path(__file__).resolve().parents[1]
PLAN = 'bench/rhythm-v2/d100-output-reward-support-v1.json'


def run():
    plan = json.loads((ROOT / PLAN).read_text())
    output = ROOT / plan['output']
    if output.exists():
        raise FileExistsError('Existing real support evidence must not be overwritten')
    started = time.monotonic()
    for path, digest in plan['source_sha256'].items():
        if sha(path) != digest:
            raise ValueError('Frozen support input changed: ' + path)
    saved = torch.load(ROOT / plan['checkpoint'], map_location='cpu', weights_only=False)
    if saved['updates'] != 1000 or saved['plan_sha256'] != sha(plan['training_plan']):
        raise ValueError('Archived policy provenance mismatch')
    torch.set_num_threads(2)
    model = FullEventModel(FullEventConfig(**saved['config'])).eval()
    model.load_state_dict(saved['model'])
    del saved
    torch.manual_seed(plan['seed'])
    panel = LocalRequestPanel(ROOT, plan['tape'], plan['features'])
    support = json.loads((ROOT / plan['support_panel']).read_text())
    if len(support['records']) != 32 or support['sample_temperature'] != 1.:
        raise ValueError('Original real32/T1 support contract changed')
    report = dict(id=plan['id'], complete=False, execution_complete=False, signal_gate_passed=None,
        model_training_allowed=False, quality_passed=False, converged_candidates=0,
        optimizer_updates=0, new_cloud_compute_calls=0, rows=[], plan_sha256=sha(PLAN),
        source_sha256=dict(plan['source_sha256']), policy_disposition='D94 remains rejected for quality',
        current_index=None, status='sampling_real_training_requests',
        limitations=['A condition reward is not a music-quality reward.',
            'T1 sampling support does not establish deterministic condition response on unseen songs.',
            'This diagnostic cannot resurrect the rejected training entrypoint or authorize scaling.',
            'No probabilities or gradients have yet been audited on these actual trajectories.'])
    verified = set()

    def save():
        report['wall_s'] = time.monotonic() - started
        write_json(output, report)
        print(json.dumps(dict(status=report['status'], requests=len(report['rows']),
                             current_index=report['current_index'], wall_s=report['wall_s'])), flush=True)

    def score(trajectory, request, bands):
        events = [RhythmEvent(**e) for e in trajectory['events']]
        legality = request_legality(events, request)
        outcome = score_condition_outcome(request.mode, canonical([*request.known_carry, *events], request.mode),
            (request.start_us, request.end_us), bands, complete=trajectory['complete'], legality_errors=legality)
        return outcome

    try:
        for case in support['records']:
            if time.monotonic() - started >= plan['max_wall_s'] or time.time() >= plan['deadline_unix']:
                raise TimeoutError('Frozen support wall/absolute deadline reached')
            i = case['record_index']
            row = panel.records[i]
            if row['split'] != 'train' or row['chart_id'] != case['chart_id'] or row['request'] != case['request']:
                raise ValueError('Real support request identity changed')
            for path, digest in panel.features.records[row['audio_sha256']]['sha256'].items():
                if path not in verified:
                    if sha(path) != digest:
                        raise ValueError('Audio feature changed')
                    report['source_sha256'][path] = digest
                    verified.add(path)
            request = event_request(row['request'])
            values, _ = request_inputs(panel.features, row['audio_sha256'], request, case['controls'])
            inputs = {k: torch.from_numpy(np.asarray(v))[None] for k, v in values.items()}
            inputs = pad_input_shapes(inputs, padding_shapes(inputs['song'].shape[1]))
            report['current_index'] = i
            save()
            deadline = min(started + plan['max_wall_s'], time.monotonic() + max(0, plan['deadline_unix'] - time.time()))
            greedy = generate_batch(model, inputs, [request], plan['token_cap'], deadline=deadline)[0]
            sampled = sample_trajectory(model, inputs, request, plan['token_cap'], deadline=deadline)
            # Process-local versions are not a durable identity. The frozen
            # checkpoint SHA is recorded; later replay must verify every weight.
            sampled.pop('parameter_versions')
            outcomes = [score(p, request, case['bands']) for p in (greedy, sampled)]
            if any(x['reward'] is None for x in outcomes):
                raise ValueError('Support request has no explicit reward field')
            advantage = outcomes[1]['reward'] - outcomes[0]['reward']
            both_usable = all(x['eligible'] and x['audit']['heads'] > 0 for x in outcomes)
            report['rows'].append(dict(record_index=i, component=case['component'], chart_id=case['chart_id'],
                mode=case['mode'], bands=case['bands'], request=case['request'],
                greedy=greedy, sampled=sampled, outcomes=outcomes, advantage=advantage,
                both_complete_legal_nonempty=both_usable, identical_tokens=greedy['tokens'] == sampled['tokens']))
            save()
        rows = report['rows']
        nonzero = [r for r in rows if abs(r['advantage']) > 1e-12]
        usable_nonzero = [r for r in nonzero if r['both_complete_legal_nonempty']]
        complete_nonempty = sum(r['outcomes'][1]['eligible'] and r['outcomes'][1]['audit']['heads'] > 0 for r in rows)
        summary = dict(requests=len(rows), nonzero_advantage=len(nonzero),
            both_complete_legal_nonempty_nonzero=len(usable_nonzero), complete_legal_nonempty_samples=complete_nonempty,
            sampled_better=sum(r['advantage'] > 1e-12 for r in rows), sampled_worse=sum(r['advantage'] < -1e-12 for r in rows),
            sampled_types=dict(Counter(e['type'] for r in rows for e in r['sampled']['events'])),
            greedy_types=dict(Counter(e['type'] for r in rows for e in r['greedy']['events'])),
            sample_tokens=sum(len(r['sampled']['tokens']) for r in rows),
            greedy_tokens=sum(len(r['greedy']['tokens']) for r in rows),
            per_field={field: dict(requests=sum(field in r['bands'] for r in rows),
                independent_songs=len({r['component'] for r in rows if field in r['bands']}),
                scalar_nonzero=sum(field in r['bands'] and abs(r['advantage']) > 1e-12 for r in rows),
                own_violation_changed=sum(field in r['bands'] and abs(r['outcomes'][0]['fields'][field]['violation'] -
                    r['outcomes'][1]['fields'][field]['violation']) > 1e-12 for r in rows))
                for field in support['panel_training_song_support']})
        gate = (len(rows) == 32 and len(nonzero) >= 8 and complete_nonempty >= 1 and len(usable_nonzero) >= 1
                and all(x['independent_songs'] >= 8 for x in summary['per_field'].values()))
        report.update(complete=True, execution_complete=True, status='support_gate_passed' if gate else 'support_gate_failed',
            signal_gate_passed=gate, summary=summary, current_index=None,
            checkpoint_unchanged=sha(plan['checkpoint']) == plan['source_sha256'][plan['checkpoint']])
        if not report['checkpoint_unchanged']:
            raise ValueError('Read-only diagnostic changed the policy')
        save()
    except Exception as exc:
        report.update(execution_complete=True, status='failed', error=f'{type(exc).__name__}: {exc}')
        save()
        raise


if __name__ == '__main__':
    run()
