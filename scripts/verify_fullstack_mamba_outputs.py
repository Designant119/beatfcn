"""Independent identities, initialization and complete artifact replay checks."""
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from beatfcn.model.full_event_codec import EventState
from beatfcn.train.local_request_panel import LocalRequestPanel, event_request
from scripts.run_reference_completion import read, write
from scripts.run_fullstack_mamba_quality import factory
from scripts.run_completion_quality import batches

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    digest = hashlib.sha256()
    with (ROOT / path).open('rb') as stream:
        for part in iter(lambda: stream.read(8 << 20), b''):
            digest.update(part)
    return digest.hexdigest()


def main():
    output = ROOT / 'reports/fullstack_mamba_quality_verification_v1.json'
    if output.exists():
        raise FileExistsError(output)
    started = time.monotonic()
    plan = read('bench/rhythm-v2/fullstack-mamba-quality-v2.json')
    report = read(plan['report'])
    if not report['complete'] or not report['training_complete']:
        raise ValueError('Terminal complete training evidence required')
    assert report['shared_initial_parameters_equal']
    assert report['updates'] == dict(transformer=1000, fullstack_mamba=1000)
    assert report['tokens']['transformer'] == report['tokens']['fullstack_mamba']
    assert report['plan_sha256'] == sha('bench/rhythm-v2/fullstack-mamba-quality-v2.json')
    for path, expected in plan['source_sha256'].items():
        assert sha(path) == expected, path
    trained_cache_parity = {}
    parity_panel = LocalRequestPanel(ROOT, plan['tape'], plan['features'])
    parity_item = next(x for x in plan['steps'][0] if x['task'] is not None)
    parity_inputs, parity_sequence = batches(parity_panel, parity_item, plan)['reference']
    n = min(32, int(parity_sequence['valid'].sum()))
    short = {k: v[:, :n] for k, v in parity_sequence.items()}
    torch.set_num_threads(2)
    for arm, entry in report['checkpoints'].items():
        assert sha(entry['path']) == entry['sha256']
        checkpoint = torch.load(ROOT / entry['path'], map_location='cpu', weights_only=False)
        assert checkpoint['family'] == arm and checkpoint['updates'] == 1000 and not checkpoint['auto_resume']
        assert {float(v['step']) for v in checkpoint['optimizer']['state'].values()} == {1000.}
        model = factory(arm, checkpoint['config']).eval()
        model.load_state_dict(checkpoint['model'])
        with torch.no_grad():
            memory = model.encode(parity_inputs)
            full = model.decode(*(short[k] for k in ('tokens', 'fields', 'times', 'valid')), memory)
            cache = model.start_cache(memory)
            pieces = [model.decode(*(short[k][:, i:i + 1] for k in ('tokens', 'fields', 'times', 'valid')),
                                   memory, cache=cache) for i in range(n)]
            error = float((full - torch.cat(pieces, 1)).abs().max())
            trained_cache_parity[arm] = error
            assert error <= 1e-4, (arm, error)
        del model
        del checkpoint
    panel = LocalRequestPanel(ROOT, plan['tape'], plan['features'])
    sig = lambda e: (e['onset_us'], e['end_us'], e['type'], e['lane'])
    trajectories, notes, hidden_types = 0, Counter(), Counter()
    minimum_choices = {arm: Counter() for arm in report['updates']}
    decision_counts = {arm: Counter() for arm in report['updates']}
    for row in report['songs']:
        outputs = {}
        for arm, entry in row['outputs'].items():
            artifact = read(entry['artifact'])
            assert artifact['song']['chart_id'] == row['chart_id']
            assert artifact['merged_unmodified_visible_and_prediction'] == artifact['visible'] + artifact['prediction']
            original, replayed = [], []
            for index in artifact['song']['indices']:
                assert panel.records[index]['split'] == 'val'
                original.extend(asdict(e) for e in panel.targets(index))
            assert Counter(map(sig, original)) == Counter(map(sig, artifact['visible'] + artifact['hidden']))
            assert not ({e['onset_us'] for e in artifact['visible']} & {e['onset_us'] for e in artifact['hidden']})
            for core in artifact['cores']:
                state = EventState(event_request(core['request']))
                for token in core['prediction']['tokens']:
                    field, value = state.vocab.decision(token)
                    decision_counts[arm][field] += 1
                    minimum_choices[arm][field] += value == min(state.allowed_values())
                    state.apply(token)
                assert state.done == core['prediction']['complete']
                assert Counter(map(sig, [asdict(e) for e in state.events])) == Counter(map(sig, core['prediction']['events']))
                replayed.extend(core['prediction']['events']); trajectories += 1
            assert Counter(map(sig, replayed)) == Counter(map(sig, artifact['prediction']))
            assert entry['pred_count'] == len(artifact['prediction'])
            notes[arm] += len(artifact['prediction'])
            outputs[arm] = artifact
            if arm == 'fullstack_mamba':
                hidden_types.update(e['type'] for e in artifact['hidden'])
        assert outputs['transformer']['visible'] == outputs['fullstack_mamba']['visible']
        assert outputs['transformer']['hidden'] == outputs['fullstack_mamba']['hidden']
    gains = [r['outputs']['fullstack_mamba']['full_note_f1'] - r['outputs']['transformer']['full_note_f1'] for r in report['songs']]
    assert abs(np.mean(gains) - report['summary']['paired_song_mean_full_note_f1_gain']) < 1e-12
    result = dict(passed=True, source_identities=len(plan['source_sha256']), matched_updates_each=1000,
        matched_target_tokens_each=report['tokens']['transformer'], fresh_adam_final_steps_each=1000,
        observed_songs=len(report['songs']), unattempted_songs=20 - len(report['songs']),
        replayed_core_trajectories=trajectories, predicted_notes=dict(notes), hidden_type_support=dict(hidden_types),
        visible_hidden_partition_exact=True, protected_objects_unchanged=True,
        trained_actual_full_cache_max_abs=trained_cache_parity,
        quality_passed=report['quality_passed'], actual_listening=False, native_playtest=False,
        smallest_legal_choice_fraction={a: {f: minimum_choices[a][f] / n for f, n in decision_counts[a].items()}
                                         for a in decision_counts},
        smallest_legal_scope='Exploratory replay descriptor; does not alter preregistered quality gates.',
        wall_s=time.monotonic() - started)
    write(output, result); print(json.dumps(result))


if __name__ == '__main__':
    main()
