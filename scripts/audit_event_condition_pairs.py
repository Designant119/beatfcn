"""Audit existing real pairs for the new event-conditioning path, without a GPU.

This is a support inventory and model-view conversion, not new corpus admission.
No cached audio, source chart, previous probe, or evaluation target is modified.
"""
from collections import Counter, defaultdict
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from beatfcn.events.canonical import Event, quantize_us
from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.full_event_codec import EventRequest, EventState, encode_events, teacher_sequence
from beatfcn.train.event_conditions import CONTROL_FIELDS, paired_control_views, training_labels

ROOT = Path(__file__).resolve().parents[1]


def stats(values):
    x = np.asarray(values, dtype=np.float64)
    return dict(shape=list(x.shape), nan=int(np.isnan(x).sum()), inf=int(np.isinf(x).sum()),
                minimum=float(x.min()), maximum=float(x.max()), mean=float(x.mean()), std=float(x.std()))


def audit():
    source = ROOT/'inputs/r09_positive_pairs/catalog.pt'
    catalog = torch.load(source, map_location='cpu', weights_only=False)
    support, unique = Counter(), defaultdict(set)
    records, rejects, encoded, input_modes = [], [], [], Counter()
    for pair in catalog['train_pairs']:
        input_modes[pair['mode']] += 1
        try:
            entries = pair['entries']
            if len(entries) != 2:
                raise ValueError('Pair must have two real targets')
            reqs = [entry['request'] for entry in entries]
            ordinary = [{k:v for k,v in r.items() if k != 'controls'} for r in reqs]
            if ordinary[0] != ordinary[1]:
                raise ValueError('Non-condition request fields differ')
            if reqs[0]['observed_events'] or reqs[0]['references']:
                raise ValueError('This inventory only certifies blank-context pairs')
            scopes = [(r['start_us'], r['end_us']) for r in reqs[0]['regions'] if r['state'] == 'editable']
            if len(scopes) != 1:
                raise ValueError('Multiple edit scopes require a separate encoding contract')
            scope, mode = scopes[0], pair['mode']
            converted, truth, identities, label_changes = [], [], [], []
            for entry in entries:
                row = entry['row']
                if (row['split'] != 'train' or row['audio_sha256'] != pair['audio_sha256']
                        or row['mode'] != mode or entry['request']['audio_id'] != pair['audio_sha256']
                        or entry['request']['mode'] != mode):
                    raise ValueError('Mode/audio/split identity mismatch')
                if any(e.get('fake') for e in entry['truth_events']):
                    raise ValueError('Fake event in supervision')
                canonical = [Event.from_json(e) for e in entry['truth_events']]
                model_events = [replace(e, onset_us=quantize_us(e.onset_us), end_us=quantize_us(e.end_us))
                                for e in canonical]
                old_bands, _ = training_labels(mode, canonical, scope)
                new_bands, _ = training_labels(mode, model_events, scope)
                label_changes.append([k for k in CONTROL_FIELDS if old_bands.get(k) != new_bands.get(k)])
                request = EventRequest(mode, *scope, entry['request']['audio_duration_us'],
                                       allowed_types=tuple(entry['request']['controls']['allowed_types']))
                rhythm = [RhythmEvent.from_event(e, mode) for e in model_events]
                tokens = encode_events(rhythm, request)
                seq = teacher_sequence(tokens, request)
                state = EventState(request)
                for token in tokens:
                    state.apply(token)
                if not state.done or Counter(state.events) != Counter(rhythm):
                    raise ValueError('Event codec changed target multiplicity or timing')
                if not np.all(seq['allowed'][np.arange(len(tokens)), seq['targets']]):
                    raise ValueError('Target outside prefix-only support')
                identities.append(dict(chart_id=row['chart_id'], target_events=[asdict(e) for e in rhythm],
                                       target_tokens=len(tokens), target_types=dict(Counter(e.type for e in rhythm)),
                                       original_target_sha256=entry['target_sha256']))
                converted.append(model_events)
                truth.append(tokens)
            views = []
            for field in CONTROL_FIELDS:
                try:
                    view = paired_control_views(mode, *converted, scope, field, origin_us=scope[0])
                except ValueError:
                    continue  # Same bands/undefined/inapplicable fields are not negatives.
                key = f'{mode}:{field}'
                support[key] += 1
                unique[key].add(pair['audio_sha256'])
                encoded.extend(view['controls'])
                views.append({k:v.tolist() if isinstance(v, np.ndarray) else v for k,v in view.items()})
            if not views:
                raise ValueError('No symmetric single-field contrast after current model conversion')
            records.append(dict(pair_id=pair['pair_id'], mode=mode, audio_sha256=pair['audio_sha256'],
                scope_us=list(scope), duration_us=reqs[0]['audio_duration_us'], targets=identities,
                control_views=views, quantization_band_changes=label_changes,
                non_condition_request_equal=True, model_event_roundtrip_exact=True,
                newly_admitted_to_training=False))
        except (ValueError, KeyError, TypeError) as exc:
            rejects.append(dict(pair_id=pair['pair_id'], reason=str(exc)))
    out = ROOT/'inputs/d64_condition_training'
    out.mkdir(exist_ok=True)
    data = out/'audited_pairs.jsonl'
    data.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))
    paths = [source, data, Path(__file__).resolve(), ROOT/'beatfcn/train/event_conditions.py',
             ROOT/'beatfcn/model/full_event_codec.py', ROOT/'beatfcn/events/controls.py']
    report = dict(id='d64-condition-support-v1', complete=True, input_pairs=dict(input_modes),
        accepted_pairs=dict(Counter(r['mode'] for r in records)), rejected_pairs=rejects,
        independent_audio={mode:len({r['audio_sha256'] for r in records if r['mode']==mode}) for mode in input_modes},
        support={k:dict(real_pairs=support[k], independent_audio=len(unique[k])) for k in sorted(support)},
        encoded_controls=stats(encoded) if encoded else None,
        quantization_changed_pairs=sum(any(r['quantization_band_changes']) for r in records),
        inputs_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        new_cloud_spend_usd=0, optimizer_updates=0, newly_admitted_to_training=0,
        scope='Old train-pair subset; CPU target/label/identity audit only. Not new content admission or conditional generation.')
    (ROOT/'reports/d64_condition_support.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['accepted_pairs','independent_audio','support',
        'quantization_changed_pairs','encoded_controls']}, ensure_ascii=False))
    print('rejects', len(rejects), rejects[:3])


if __name__ == '__main__':
    audit()
