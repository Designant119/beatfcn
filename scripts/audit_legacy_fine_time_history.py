"""Finish C00-02 on its original D97 weights, without training or generation.

The intervention removes both token embedding content and every decoder
history-mixing contribution. Only head_cell/head_ms logits are replaced;
all other logits, clocks, masks and legal teacher supports stay unchanged.
This is an OOD dependence test, not an architecture comparison.
"""
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from beatfcn.model.full_event_codec import FIELDS
from beatfcn.model.full_event_model import FullEventConfig, FullEventModel
from beatfcn.train.causal_diagnosis import decision_rows
from beatfcn.train.local_request_panel import LocalRequestPanel
from scripts.run_local_portfolio_scale import sha, write


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'bench/rhythm-v2/local-legacy-fine-time-history-v1.json'
REPORT = ROOT / 'reports/local_legacy_fine_time_history_v1.json'


@torch.inference_mode()
def main():
    if PLAN.exists() or REPORT.exists():
        raise FileExistsError('Preserve original zero-update dependence test')
    torch.set_num_threads(2)
    oldpath = ROOT / 'bench/rhythm-v2/root-cause-v2.json'
    old = json.loads(oldpath.read_text())
    reference = ROOT / 'reports/root_cause_v2.json'
    prior = json.loads(reference.read_text())
    weight = ROOT / old['checkpoint']
    if sha(weight) != old['source_sha256'][old['checkpoint']]:
        raise ValueError('Original D97 checkpoint changed')
    plan = dict(id='local-legacy-fine-time-history-v1', checkpoint=old['checkpoint'],
        checkpoint_sha256=sha(weight), original_plan_sha256=sha(oldpath),
        prior_audio_test_sha256=sha(reference), records=old['records'],
        fields=['head_cell', 'head_ms'], wall_s=900, optimizer_updates=0,
        history_gate=dict(song_mean_nll_increase=.05, positive_song_fraction=.6),
        intervention='Zero token embeddings and all decoder self-attention/history outputs. Replace only fine-time output rows; leave all other logits exactly baseline.',
        scope='Teacher forced legal support and query times remain true. This measures conditional dependence, not free generation or causal superiority of a retrained architecture.',
        source_sha256={p:sha(ROOT/p) for p in ('scripts/audit_legacy_fine_time_history.py', 'beatfcn/model/full_event_model.py', 'beatfcn/train/causal_diagnosis.py')},
        cloud_compute_calls=0)
    write(PLAN, plan)
    saved = torch.load(weight, map_location='cpu', weights_only=False)
    model = FullEventModel(FullEventConfig(**saved['config'])).eval()
    model.load_state_dict(saved['model'])
    del saved
    panel = LocalRequestPanel(ROOT, old['tape'], old['features'])
    report = dict(id=plan['id'], status='running', rows=[], optimizer_updates=0,
        plan_sha256=sha(PLAN), cloud_compute_calls=0)
    write(REPORT, report)
    started = time.monotonic()
    try:
        for item in plan['records']:
            if time.monotonic()-started > plan['wall_s']:
                raise TimeoutError('Fine-time dependence wall')
            index = item['index']; record = panel.records[index]
            controls = np.asarray(record['controls'], np.float32).copy()
            controls[[i for i in range(len(controls)) if i not in item['specified_fields']], :3] = 0
            inputs, seq = panel.batch([index], controls=[controls])
            baseline = model(inputs, seq)
            hooks = [model.token_embedding.register_forward_hook(lambda _m,_a,out:torch.zeros_like(out))]
            for layer in model.decoder:
                mixer = layer.history if layer.is_ssm else layer.self_attn
                hooks.append(mixer.register_forward_hook(lambda _m,_a,out:torch.zeros_like(out)))
            try:
                removed = model(inputs, seq)
            finally:
                for hook in hooks: hook.remove()
            mask = torch.zeros_like(seq['valid'])
            for name in plan['fields']: mask |= seq['fields'] == FIELDS.index(name)
            combined = torch.where(mask[...,None], removed, baseline)
            if not torch.equal(combined[~mask], baseline[~mask]):
                raise AssertionError('Non-time heads changed')
            if not torch.isfinite(combined).all(): raise ValueError('Nonfinite dependence logits')
            actual = decision_rows(baseline, seq, model.vocab)
            altered = decision_rows(combined, seq, model.vocab)
            report['rows'].append(dict(index=index, component=record['component'], split=record['split'],
                baseline={k:actual[k] for k in plan['fields'] if k in actual},
                history_removed={k:altered[k] for k in plan['fields'] if k in altered},
                untouched_logits_exact=True))
            write(REPORT, report)
        summaries = {}
        for name in plan['fields']:
            values = defaultdict(list)
            for entry in report['rows']:
                if entry['split']=='val' and name in entry['baseline']:
                    delta = np.mean(entry['history_removed'][name]['nll'])-np.mean(entry['baseline'][name]['nll'])
                    values[entry['component']].append(float(delta))
            songs = {k:float(np.mean(v)) for k,v in values.items()}
            mean = float(np.mean(list(songs.values())))
            positive = float(np.mean(np.asarray(list(songs.values()))>0))
            summaries[name] = dict(song_deltas=songs, song_mean_nll_increase=mean,
                positive_song_fraction=positive, detected=mean>=.05 and positive>=.6)
        report.update(status='completed', history_effect=summaries,
            prior_local_audio_effect=prior['verdict']['local_timing'],
            original_checkpoint_unchanged=sha(weight)==plan['checkpoint_sha256'],
            wall_s=time.monotonic()-started)
        write(REPORT, report)
    except BaseException as exc:
        report.update(status='stopped', failure=f'{type(exc).__name__}: {exc}', wall_s=time.monotonic()-started)
        write(REPORT, report)
        raise


if __name__=='__main__': main()
