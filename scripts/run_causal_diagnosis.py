"""Frozen causal input tests and genuine cached prefix-rescue trajectories."""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.full_event_codec import BOS, EventState, FIELDS, encode_events, teacher_sequence
from beatfcn.model.full_event_model import FullEventConfig, FullEventModel
from beatfcn.model.local_song_rollout import canonical, independent_legality
from beatfcn.model.evaluation import score_events
from beatfcn.train.causal_diagnosis import audio_intervention, decision_rows, legal_probabilities
from beatfcn.train.local_request_panel import LocalRequestPanel, event_request
from beatfcn.train.local_panel import collate
from beatfcn.train.local_audio_audit import tensor_health

ROOT = Path(__file__).resolve().parents[1]
PLAN = 'bench/rhythm-v2/root-cause-v2.json'
VARIANTS = ('correct', 'local_plus200', 'local_minus200', 'remote_reverse', 'remote_rotate')


def read(path):
    return json.loads((ROOT / path).read_text())


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def write(path, payload):
    path = ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.partial')
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def prepare():
    if (ROOT / PLAN).exists():
        raise FileExistsError(PLAN)
    old = read('bench/rhythm-v2/d98-teacher-decisions-v1.json')
    panel = LocalRequestPanel(ROOT, old['tape'], old['features'])
    records = [r for r in old['records'] if panel.records[r['index']]['mode'] == 'phira']
    val = [r['index'] for r in records if r['split'] == 'val']
    counts = {i: panel.records[i]['target_stop'] - panel.records[i]['target_start'] for i in val}
    active = sorted((i for i in val if counts[i] > 0), key=lambda i: (counts[i], i))
    picked = [active[int(k)] for k in np.linspace(0, len(active) - 1, min(8, len(active)))]
    sources = {k: v for k, v in old['source_sha256'].items()
               if k in (old['checkpoint'], old['tape'], old['features'],
                        'beatfcn/model/full_event_codec.py', 'beatfcn/model/full_event_model.py',
                        'beatfcn/train/local_request_panel.py', 'beatfcn/train/local_features.py')}
    for p in ('scripts/run_causal_diagnosis.py', 'beatfcn/train/causal_diagnosis.py',
              'design/CAUSAL_DIAGNOSIS_AND_COMPLETION_v1.md'):
        sources[p] = sha(p)
    # Every actual selected feature and chart archive is frozen before inference.
    for item in records:
        row = panel.records[item['index']]
        chart = panel.charts[row['chart_index']]
        sources.update(panel.features.records[row['audio_sha256']]['sha256'])
        sources.update({chart['target_path']: chart['target_sha256'], chart['token_path']: chart['token_sha256']})
    plan = dict(id='root-cause-v2', checkpoint=old['checkpoint'], tape=old['tape'],
                features=old['features'], records=records, rollout_indices=picked,
                source_sha256=sources, seed=20261026, max_wall_s=900,
                deadline_unix=1793037775., token_cap=1024,
                output='reports/root_cause_v2.json', artifacts='artifacts/root-cause-v2',
                gates=dict(local_nll=.05, global_nll=.02, global_tv=.01,
                           history_non_tap_probability=.05, independent_song_fraction=.6))
    write(PLAN, plan)
    print(json.dumps(dict(prepared=True, records=len(records), rollout_indices=picked)))


def aggregate(rows):
    result = {}
    for split in ('train', 'val'):
        selected = [r for r in rows if r['split'] == split]
        out = {}
        for variant in VARIANTS:
            fields = {}
            for name in FIELDS:
                parts = [r['variants'][variant][name] for r in selected if name in r['variants'][variant]]
                if not parts:
                    continue
                n = sum(p['count'] for p in parts)
                entry = dict(count=n, accuracy=sum(p['correct'] for p in parts) / n,
                             nll=float(np.mean([v for p in parts for v in p['nll']])))
                nf = sum(p['nonforced_count'] for p in parts)
                entry['nonforced_accuracy'] = sum(p['nonforced_correct'] for p in parts) / nf if nf else None
                errors = [abs(v) for p in parts for v in p.get('conditional_error_ms', [])]
                if errors:
                    entry['conditional_absolute_error_ms_p50_p90'] = np.quantile(errors, [.5, .9]).tolist()
                    entry['within_ms'] = {str(t): float(np.mean(np.asarray(errors) <= t)) for t in (10, 20, 40, 100)}
                if variant != 'correct':
                    entry['mean_tv'] = float(np.mean([v for p in parts for v in p['tv']]))
                fields[name] = entry
            out[variant] = fields
        result[split] = dict(requests=len(selected), songs=len({r['component'] for r in selected}), variants=out)
    return result


def paired_verdict(rows, plan):
    """Aggregate paired effects by independent song, rather than counting tokens as songs."""
    selected = [r for r in rows if r['split'] == 'val']
    local, remote, hist = defaultdict(list), defaultdict(list), defaultdict(list)
    remote_tv = []
    for row in selected:
        base = row['variants']['correct']
        for dest, variants, fields in (
                (local, ('local_plus200', 'local_minus200'), ('head_cell', 'head_ms')),
                (remote, ('remote_reverse', 'remote_rotate'), tuple(base))):
            relevant = [f for f in fields if f in base]
            baseline = np.mean([v for f in relevant for v in base[f]['nll']])
            altered = np.mean([v for name in variants for f in relevant for v in row['variants'][name][f]['nll']])
            dest[row['component']].append(float(altered - baseline))
        remote_tv.extend(v for name in ('remote_reverse', 'remote_rotate')
                         for field in row['variants'][name].values() for v in field['tv'])
        if row['history'].get('non_tap_probability_drop') is not None:
            hist[row['component']].append(row['history']['non_tap_probability_drop'])
    def judge(values, minimum):
        songs = {k: float(np.mean(v)) for k, v in values.items()}
        mean = float(np.mean(list(songs.values()))) if songs else None
        positive = float(np.mean(np.asarray(list(songs.values())) > 0)) if songs else None
        return dict(song_deltas=songs, songs=len(songs), song_mean_delta=mean,
                    positive_song_fraction=positive,
                    detected=bool(mean is not None and mean >= minimum and positive >= plan['gates']['independent_song_fraction']))
    verdict = dict(local_timing=judge(local, plan['gates']['local_nll']),
                   remote_task_relevance=judge(remote, plan['gates']['global_nll']),
                   type_duration_history=judge(hist, plan['gates']['history_non_tap_probability']))
    verdict['remote_order_sensitivity'] = dict(mean_token_tv=float(np.mean(remote_tv)),
                                              detected=float(np.mean(remote_tv)) >= plan['gates']['global_tv'])
    return verdict


@torch.inference_mode()
def rollout(model, inputs, request, seed, cap, deadline):
    state = EventState(request, model.vocab)
    memory = model.encode(inputs)
    cache = model.start_cache(memory)
    trace = []
    for step in range(len(seed) + cap):
        if time.monotonic() >= deadline:
            return dict(complete=False, failure='deadline', tokens=state.tokens,
                        events=[asdict(e) for e in state.events], trace=trace)
        field, when = state.query()
        previous = state.tokens[-1] if state.tokens else BOS
        logits = model.decode(torch.tensor([[previous]]), torch.tensor([[field]]),
                              torch.tensor([[when]], dtype=torch.float32),
                              torch.ones((1, 1), dtype=torch.bool), memory, cache=cache)[0, 0]
        allowed = state.allowed_tokens()
        legal = logits[list(allowed)].softmax(-1)
        selected = int(legal.argmax())
        choice = allowed[selected]
        forced = step < len(seed)
        if forced:
            choice = int(seed[step])
        entry = dict(step=step, field=FIELDS[field], query_time_s=when, token=choice,
                     forced_oracle=forced, argmax=allowed[selected], max_probability=float(legal[selected]),
                     event_count=len(state.events))
        if state.phase in ('group', 'member'):
            end_token = model.vocab.token(state.phase, 0)
            entry['end_probability'] = float(legal[allowed.index(end_token)]) if end_token in allowed else 0.
        if state.phase == 'kind':
            entry['kind_probabilities'] = legal.tolist()
        trace.append(entry)
        state.apply(choice)
        if state.done:
            break
    return dict(complete=state.done, failure=None if state.done else 'token_cap',
                tokens=state.tokens, events=[asdict(e) for e in state.events], trace=trace)


@torch.inference_mode()
def run():
    plan = read(PLAN)
    if (ROOT / plan['output']).exists():
        raise FileExistsError(plan['output'])
    started = time.monotonic()
    deadline = min(started + plan['max_wall_s'], started + plan['deadline_unix'] - time.time())
    report = dict(id=plan['id'], complete=False, status='verifying', plan_sha256=sha(PLAN),
                  rows=[], rollouts=[], tensor_stats=[], optimizer_updates=0, new_cloud_compute_calls=0,
                  quality_passed=False, limitations=['Teacher/oracle experiments do not measure product quality.',
                  'Circular local shifts change boundary content; both signs are reported.',
                  'Remote reordering is an off-distribution diagnostic, not evidence of musical understanding.'])
    try:
        for p, expected in plan['source_sha256'].items():
            if sha(p) != expected:
                raise ValueError('Frozen identity changed: ' + p)
        torch.set_num_threads(2)
        torch.manual_seed(plan['seed'])
        saved = torch.load(ROOT / plan['checkpoint'], map_location='cpu', weights_only=False)
        model = FullEventModel(FullEventConfig(**saved['config'])).eval()
        model.load_state_dict(saved['model'])
        del saved
        panel = LocalRequestPanel(ROOT, plan['tape'], plan['features'])
        for item in plan['records']:
            if time.monotonic() >= deadline:
                raise TimeoutError('Diagnostic wall limit')
            index = item['index']
            row = panel.records[index]
            control = np.asarray(row['controls'], np.float32).copy()
            control[[i for i in range(len(control)) if i not in item['specified_fields']], :3] = 0
            inputs, seq = panel.batch([index], controls=[control])
            if row['split'] == 'val' and inputs['controls'][..., 2].any():
                raise ValueError('Hidden conditions leaked')
            if not report['tensor_stats']:
                report['tensor_stats'] = [{s: {k: tensor_health(v.numpy()) for k, v in values.items()}
                                           for s, values in (('inputs', inputs), ('sequence', seq))}]
            result = dict(index=index, split=row['split'], chart_id=row['chart_id'],
                          component=row['component'], stratum=row['stratum'],
                          scope_us=[row['request']['start_us'], row['request']['end_us']], variants={})
            baseline = None
            for name in VARIANTS:
                logits = model(audio_intervention(inputs, name), seq)
                if not torch.isfinite(logits).all():
                    raise ValueError('Nonfinite logits')
                result['variants'][name] = decision_rows(logits, seq, model.vocab, baseline)
                if baseline is None:
                    baseline = legal_probabilities(logits, seq)[0]
            # Legal alternate type/duration history, exact same onset groups and multiplicities.
            events = panel.targets(index)
            taps = [RhythmEvent(e.onset_us, e.onset_us, 'tap', -1) for e in events]
            request = event_request(row['request'])
            altseq = teacher_sequence(encode_events(taps, request), request)
            _, alt = collate([dict(inputs=panel.row(index, controls=control)['inputs'], sequence=altseq)])
            altlogits = model(inputs, alt)
            pm = legal_probabilities(altlogits, alt)[0]
            ids = (alt['valid'][0] & (alt['fields'][0] == FIELDS.index('kind'))).nonzero().flatten()
            correct = result['variants']['correct'].get('kind')
            history = dict(non_tap_probability_drop=None)
            if correct:
                if len(ids) != correct['count']:
                    raise ValueError('Kind alignment changed')
                true_tokens = torch.tensor([model.vocab.token('kind', k) for k in correct['truth']])
                p = pm[ids].gather(1, true_tokens[:, None]).squeeze(1).tolist()
                differences = [a - b for a, b, k in zip(correct['target_probability'], p, correct['truth']) if k != 0]
                history.update(non_tap_probability_drop=float(np.mean(differences)) if differences else None,
                               original_truth=correct['truth'], altered_target_probability=p,
                               altered_predicted=[model.vocab.decision(int(t))[1] for t in pm[ids].argmax(-1)],
                               aligned_kind_count=len(ids))
            result['history'] = history
            report['rows'].append(result)
            report.update(status='teacher_interventions', wall_s=time.monotonic() - started)
            write(plan['output'], report)
        report['summary'] = aggregate(report['rows'])
        report['verdict'] = paired_verdict(report['rows'], plan)
        print(json.dumps(dict(status='teacher_complete', verdict=report['verdict'])), flush=True)
        for index in plan['rollout_indices']:
            row = panel.records[index]
            request = event_request(row['request'])
            controls = np.asarray(row['controls'], np.float32).copy()
            controls[:, :3] = 0
            inputs, _ = panel.batch([index], controls=[controls])
            events = panel.targets(index)
            midpoint = (request.start_us + request.end_us) // 2
            seed = encode_events([e for e in events if e.onset_us < midpoint], request)[:-1]
            outputs = {}
            for name, forced in (('free', ()), ('oracle_prefix', seed)):
                prediction = rollout(model, inputs, request, forced, plan['token_cap'], deadline)
                pred_events = [RhythmEvent(**e) for e in prediction['events'] if e['onset_us'] >= midpoint]
                truth = [e for e in events if e.onset_us >= midpoint]
                scores = score_events(canonical(pred_events, 'phira'), canonical(truth, 'phira'), 'phira')
                output = f"{plan['artifacts']}/{index}-{name}.json"
                write(output, dict(index=index, chart_id=row['chart_id'], audio_sha256=row['audio_sha256'],
                                   request=row['request'], midpoint_us=midpoint, prediction=prediction,
                                   truth=[asdict(e) for e in events], scores_after_midpoint=scores,
                                   legality=independent_legality([RhythmEvent(**e) for e in prediction['events']], 'phira', request.duration_us)))
                full = scores['edit_utility']['full_note']
                denom = full['pred'] + full['true']
                outputs[name] = dict(artifact=output, complete=prediction['complete'],
                                     failure=prediction['failure'], suffix_types=dict(Counter(e.type for e in pred_events)),
                                     suffix_full_note_f1=2 * full['matched'] / denom if denom else 1.,
                                     suffix_notes=len(pred_events), suffix_truth=len(truth))
                if prediction['failure'] == 'deadline':
                    raise TimeoutError('Rollout wall limit')
            report['rollouts'].append(dict(index=index, chart_id=row['chart_id'], stratum=row['stratum'], outputs=outputs))
            report.update(status='prefix_rollouts', wall_s=time.monotonic() - started)
            write(plan['output'], report)
            print(json.dumps(dict(status='prefix_rollouts', completed=len(report['rollouts']), wall_s=report['wall_s'])), flush=True)
        report.update(complete=True, status='completed', checkpoint_unchanged=sha(plan['checkpoint']) == plan['source_sha256'][plan['checkpoint']])
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        report['wall_s'] = time.monotonic() - started
        write(plan['output'], report)
    print(json.dumps(dict(complete=True, wall_s=report['wall_s'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    prepare() if args.prepare else run()
