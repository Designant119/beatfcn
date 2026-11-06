"""Replay a saved event-model implementation probe without optimizer updates."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from scripts.probe_full_event_model import load_batch, metrics, generate
from beatfcn.model.full_event_codec import EventRequest, EventVocabulary
from beatfcn.model.full_event_model import FullEventConfig, FullEventModel

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    plan_path, output = ROOT / args.plan, ROOT / args.output
    if output.exists():
        raise FileExistsError(output)
    plan = json.loads(plan_path.read_text())
    result_path, checkpoint_path = ROOT / plan['result'], ROOT / plan['checkpoint']
    result = json.loads(result_path.read_text())
    resources = json.loads((ROOT / plan['resource_report']).read_text())
    if not (result['complete'] and resources['complete'] and resources['exit_code'] == 0
            and not resources.get('terminated_by')):
        raise ValueError('Probe has not completed normally')
    expected = dict(plan['input_sha256'])
    expected[args.plan] = result['plan_sha256']
    expected[plan['checkpoint']] = result['checkpoint_sha256']
    for name, digest in expected.items():
        if sha(ROOT / name) != digest:
            raise ValueError(f'Evidence identity changed: {name}')
    torch.set_num_threads(plan['threads'])
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if not state['diagnostic_only'] or state['plan_sha256'] != result['plan_sha256']:
        raise ValueError('Checkpoint identity or scope mismatch')
    manifest, inputs, sequence = load_batch(ROOT / plan['data'], 'cpu')
    model = FullEventModel(FullEventConfig(**state['config']), EventVocabulary(**state['vocabulary']))
    model.load_state_dict(state['model'], strict=True)
    del state
    model.eval()
    with torch.no_grad():
        logits = model(inputs, sequence)
        logits_match = hashlib.sha256(logits.numpy().tobytes()).hexdigest() == result['teacher_logits_sha256']
        final = metrics(logits, sequence)
    streams = 0
    events = 0
    for index, row in enumerate(manifest['records']):
        replay = generate(model, {k:v[index:index+1] for k,v in inputs.items()},
                          EventRequest(**row['request']), plan['generation_token_cap'])
        original = result['generations'][index]
        if any(replay[k] != original[k] for k in ('complete', 'events', 'tokens')):
            raise ValueError('Saved generation does not replay')
        streams += 1
        events += len(replay['events'])
    if not logits_match:
        raise ValueError('Saved teacher logits do not replay exactly')
    verification = dict(id=plan['id'] + '-independent-replay', complete=True, passed=True,
        probe_fit_passed=result['passed'], parameters=sum(p.numel() for p in model.parameters()),
        optimizer_updates=0, verified_training_updates=result['optimizer_updates'],
        input_files=len(plan['input_sha256']), teacher_logits_identical=logits_match,
        generated_streams_identical=streams, generated_events=events,
        final_ce=final['ce'], all_fields_accuracy=min(v['accuracy'] for v in final['fields'].values()),
        peak_rss_gib=resources['maximum_rss_gib'],
        source_sha256={name:sha(ROOT / name) for name in (
            args.plan, plan['result'], plan['checkpoint'], 'scripts/verify_full_event_probe.py')},
        scope='Replay of two fitted training cores; does not certify fit gate or independent-song quality.',
        new_cloud_spend_usd=0)
    output.write_text(json.dumps(verification, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k:v for k,v in verification.items() if k != 'source_sha256'}))


if __name__ == '__main__':
    main()
