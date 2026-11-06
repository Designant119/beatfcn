"""Verify rejected quality artifacts without running inference or training."""
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

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT / path).read_text())


def sha(path):
    h = hashlib.sha256()
    with (ROOT / path).open('rb') as stream:
        for b in iter(lambda: stream.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    started = time.monotonic()
    plan = read('bench/rhythm-v2/completion-quality-v1.json')
    result = read(plan['report'])
    assert result['complete'] and result['training_complete'] and result['parent_unchanged']
    assert result['updates'] == {'no_reference': 1000, 'reference': 1000}
    assert result['tokens']['no_reference'] == result['tokens']['reference']
    for path, expected in plan['source_sha256'].items():
        assert sha(path) == expected, path
    for arm, entry in result['checkpoints'].items():
        assert sha(entry['path']) == entry['sha256']
        state = torch.load(ROOT / entry['path'], map_location='cpu', weights_only=False)
        assert state['arm'] == arm and state['updates'] == 1000 and state['parent_updates'] == 2
        assert {float(v['step']) for v in state['optimizer']['state'].values()} == {1002.}
        assert not state['auto_resume']
        del state
    panel = LocalRequestPanel(ROOT, plan['tape'], plan['features'])
    sig = lambda e: (e['onset_us'], e['end_us'], e['type'], e['lane'])
    cores, notes, targets = 0, 0, Counter()
    for row in result['songs']:
        arms = {}
        for arm, entry in row['outputs'].items():
            artifact = read(entry['artifact'])
            assert artifact['song']['chart_id'] == row['chart_id']
            assert artifact['merged_unmodified_visible_and_prediction'] == artifact['visible'] + artifact['prediction']
            replayed = []
            for core in artifact['cores']:
                state = EventState(event_request(core['request']))
                for token in core['prediction']['tokens']:
                    state.apply(token)
                assert state.done == core['prediction']['complete']
                assert Counter(map(sig, [asdict(e) for e in state.events])) == Counter(map(sig, core['prediction']['events']))
                replayed.extend(core['prediction']['events'])
                cores += 1
            assert Counter(map(sig, replayed)) == Counter(map(sig, artifact['prediction']))
            original = []
            for index in artifact['song']['indices']:
                assert panel.records[index]['split'] == 'val'
                original.extend(asdict(e) for e in panel.targets(index))
            assert Counter(map(sig, original)) == Counter(map(sig, artifact['visible'] + artifact['hidden']))
            assert not ({e['onset_us'] for e in artifact['visible']} & {e['onset_us'] for e in artifact['hidden']})
            assert not artifact['metrics']['legality']
            arms[arm] = artifact
            if arm == 'reference':
                notes += len(artifact['prediction'])
                targets.update(e['type'] for e in artifact['hidden'])
        assert arms['reference']['hidden'] == arms['no_reference']['hidden']
        assert arms['reference']['visible'] == arms['no_reference']['visible']
    assert result['stop_reason'] == 'single_type_collapse_sentinels'
    assert len(result['songs']) == 4 and not result['quality_passed']
    assert notes == result['summary']['reference_types']['tap'] == 12233
    # Recompute the diagnostic mean, without treating the other sixteen as observed.
    gains = [r['outputs']['reference']['full_note_f1'] - r['outputs']['no_reference']['full_note_f1'] for r in result['songs']]
    assert abs(np.mean(gains) - result['summary']['paired_song_mean_f1_gain']) < 1e-12
    images = list((ROOT / 'artifacts/completion-review-v2').glob('*.svg.png'))
    out = dict(passed=True, source_identities=len(plan['source_sha256']),
        checkpoint_optimizer_step=1002, matched_updates_each=1000,
        matched_target_tokens_each=result['tokens']['reference'], replayed_core_trajectories=cores,
        observed_songs=4, unattempted_songs=16, reference_predicted_taps=notes,
        hidden_type_support=dict(targets), visible_hidden_partition_exact=True,
        protected_objects_unchanged=True, quality_passed=False,
        rendered_review_images=[dict(path=str(p.relative_to(ROOT)), sha256=sha(p.relative_to(ROOT))) for p in images],
        actual_listening=False, native_playtest=False, wall_s=time.monotonic() - started)
    path = ROOT / 'reports/completion_quality_verification_v1.json'
    with path.open('x') as stream:
        stream.write(json.dumps(out, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(out))


if __name__ == '__main__':
    main()
