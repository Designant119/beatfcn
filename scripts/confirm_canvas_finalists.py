"""One confirmation opening after fixed identities and all second-seed outcomes."""
import gc
import json
from pathlib import Path
import shutil
import time

import torch

from beatfcn.train.canonical_probability import explicit_row
from beatfcn.train.mert_canvas_features import MertCanvasFeatures
from scripts.replicate_canvas_finalist import make_model, evaluate, ROOT, SELECTION, POLICY
from scripts.run_frontend_canvas_comparison import setup
from scripts.run_local_portfolio_scale import row_factory, sha, write


def gates(metrics, limit):
    return dict(complete=metrics['attempted'] == metrics['complete'] == limit['complete'],
        nonempty=metrics['nonempty'] >= limit['min_nonempty'],
        count=limit['count_ratio'][0] <= metrics['density_ratio'] <= limit['count_ratio'][1],
        full=metrics['full_note_f1'] >= limit['full_note_f1'],
        onset=metrics['onset10'] >= limit['onset10'],
        null=metrics['onset10'] - metrics['shift_null10'] >= limit['onset10_minus_shift'],
        components=sum(v['onset10'] > v['shift_null10'] for v in metrics['components'].values()) >= limit['min_positive_components'])


def main():
    ident = 'local-portfolio-confirmation-v1'
    report_path = ROOT / 'reports/local_portfolio_confirmation_v1.json'
    plan_path = ROOT / f'bench/rhythm-v2/{ident}.json'
    art = ROOT / 'artifacts' / ident
    if report_path.exists() or plan_path.exists():
        raise FileExistsError('Confirmation is opened once, never reselected')
    selection = json.loads(SELECTION.read_text())
    policy = json.loads(POLICY.read_text())
    if sha(POLICY) != selection['policy_sha256']:
        raise ValueError('Selection policy drift')
    qualified = []
    terminal = []
    for rank, entry in enumerate(selection['finalist_identities'], 1):
        rp = ROOT / f'reports/local_finalist_r{rank}_seed2_v1.json'
        result = json.loads(rp.read_text())
        terminal.append(dict(rank=rank, candidate=entry['id'], status=result['status'], sha256=sha(rp)))
        if result['status'] not in ('completed', 'health_failed', 'stopped'):
            raise ValueError('Record every second-seed terminal before confirmation')
        if result['status'] == 'completed' and result['reproduced']:
            qualified.append((rank, entry, result))
    # Refuse to silently replace a missing short-context confirmation cache.
    # If needed it is extracted from raw audio after identity freeze first.
    short_cache = ROOT / 'reports/local_mert_confirmation_context_cache_v1.json'
    if any(entry['arm'] == 'short32' for _, entry, _ in qualified):
        cache = json.loads(short_cache.read_text())
        if not cache['complete']:
            raise ValueError('Short-context confirmation cache incomplete')
    identities = []
    for rank, entry, result in qualified:
        replica = ROOT / f'artifacts/local-finalist-r{rank}-seed2-v1/quality-{entry["steps"]:04d}.pt'
        for weight, digest in [(ROOT / entry['checkpoint'], entry['checkpoint_sha256']),
                               (replica, result['final_weight_sha256'])]:
            if sha(weight) != digest:
                raise ValueError('Frozen confirmation checkpoint changed')
        identities.append(dict(rank=rank, candidate=entry['id'], original=entry['checkpoint'],
            original_sha256=entry['checkpoint_sha256'], seed2=str(replica.relative_to(ROOT)), seed2_sha256=result['final_weight_sha256']))
    payloads = {}
    if qualified:
        panel, store = setup(json.loads((ROOT / 'bench/rhythm-v2/local-frontend-comparison-v2.json').read_text()))
        cohort = json.loads((ROOT / 'bench/rhythm-v2/local-portfolio-cohort-v1.json').read_text())
        reference = json.loads((ROOT / qualified[0][1]['plan']).read_text())
        confirm = reference['indices']['confirm']
        for index in confirm:
            audio = panel.records[index]['audio_sha256']
            rec = cohort['mert_cache_identities'][audio]
            if rec['path'] != store.mert[audio]['path']:
                raise ValueError('Original full-cache route drift')
            for name, info in rec['files'].items():
                path = ROOT / rec['path'] / name
                actual = sha(path)
                if actual != info['sha256']:
                    raise ValueError('Confirmation full-cache payload drift')
                payloads[str(path.relative_to(ROOT))] = actual
            raw = store.pupu.records[audio]
            path = ROOT / raw['complete_mel']
            actual = sha(path)
            if actual != raw['sha256'][raw['complete_mel']]:
                raise ValueError('Confirmation shared mel drift')
            payloads[str(path.relative_to(ROOT))] = actual
            name = panel.charts[panel.records[index]['chart_index']]['target_path']
            actual = sha(ROOT / name)
            if actual != cohort['target_sha256'][name]:
                raise ValueError('Frozen confirmation target identity drift')
            payloads[name] = actual
        if any(e['arm'] == 'short32' for _, e, _ in qualified):
            needed = {panel.records[i]['audio_sha256'] for i in confirm}
            for rec in cache['records']:
                if rec['audio_sha256'] not in needed:
                    continue
                for name, info in rec['files'].items():
                    path = ROOT / rec['path'] / name
                    actual = sha(path)
                    if actual != info['sha256']:
                        raise ValueError('Short-context confirmation payload drift')
                    payloads[str(path.relative_to(ROOT))] = actual
    plan = dict(id=ident, selection_sha256=sha(SELECTION), policy_sha256=sha(POLICY),
        second_seed_terminals=terminal, identities=identities,
        short_confirmation_cache_sha256=sha(short_cache) if any(e['arm'] == 'short32' for _, e, _ in qualified) else None,
        confirmation_payload_sha256=payloads,
        gates=policy['confirmation']['local_followup_support'],
        interpretation='Held out from this local campaign, not all-project blind. Both seed gates and their mean are reported. No threshold/seed/checkpoint choice or retraining after opening.',
        per_seed_wall_s=900, source_sha256={n: sha(ROOT / n) for n in
            ('scripts/confirm_canvas_finalists.py', 'scripts/replicate_canvas_finalist.py')}, cloud_compute_calls=0)
    write(plan_path, plan)
    for name in plan['source_sha256']:
        dest = art / 'source' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dest)
    shutil.copy2(plan_path, art / 'source/frozen-plan.json')
    report = dict(id=ident, status='running', plan_sha256=sha(plan_path), finalists=[],
                  confirmation_scored=False, cloud_compute_calls=0)
    write(report_path, report)
    if not qualified:
        report.update(status='not_opened_no_reproduced_finalist', reason='No substitute finalist, third seed or retuning; conditional protocol ends here.')
        write(report_path, report)
        return
    torch.set_num_threads(4)
    torch.mps.set_per_process_memory_fraction(.5)
    started = time.monotonic()
    try:
        for (rank, entry, result), identity in zip(qualified, identities):
            original = json.loads((ROOT / entry['plan']).read_text())
            features = MertCanvasFeatures(ROOT, store.pupu, str(short_cache.relative_to(ROOT))) if entry['arm'] == 'short32' else store
            base = row_factory(panel, features)
            def row(*args, **kwargs):
                return explicit_row(base(*args, **kwargs))
            confirm = original['indices']['confirm']
            if len(confirm) != 16 or len({panel.records[i]['component'] for i in confirm}) != 8:
                raise ValueError('Frozen confirmation identity mismatch')
            if set(confirm) & set(original['indices']['dev']):
                raise ValueError('Development/confirmation overlap')
            item = dict(rank=rank, candidate=entry['id'], seeds={})
            report['finalists'].append(item)
            for seed_name, weight in [('original', identity['original']), ('seed2', identity['seed2'])]:
                tick = time.monotonic()
                model = make_model(entry, original, 0, 'cpu')
                saved = torch.load(ROOT / weight, map_location='cpu', weights_only=False)
                if saved['updates'] != entry['steps']:
                    raise ValueError('Confirmation terminal update mismatch')
                model.load_state_dict(saved['model'])
                del saved
                model.to('mps').eval()
                folder = art / f'rank{rank}' / seed_name
                folder.mkdir(parents=True, exist_ok=True)
                tasks = dict(
                    draft=(confirm, {}),
                    completion=(confirm[::2], dict(view='middle')),
                    reference_masked=(confirm[::2], dict(view='middle', masked=True)),
                    density_low=(confirm[::4], dict(bands=(2., 3.))),
                    density_high=(confirm[::4], dict(bands=(6., 8.))),
                    seen_empty_diagnostic=(original['indices']['empty_dev'], {}),
                )
                outcome = {}
                item['seeds'][seed_name] = outcome
                for task, (indices, kwargs) in tasks.items():
                    outcome[task] = evaluate(model, entry, original, row, panel, indices, folder, task,
                        tick + plan['per_seed_wall_s'], **kwargs)
                    report['confirmation_scored'] = True
                    write(report_path, report)
                outcome['support_gates'] = gates(outcome['draft'], plan['gates'])
                outcome['support'] = all(outcome['support_gates'].values())
                outcome['wall_s'] = time.monotonic() - tick
                write(report_path, report)
                del model
                gc.collect()
                torch.mps.empty_cache()
            item['both_seeds_support'] = all(v['support'] for v in item['seeds'].values())
            item['mean_draft'] = {k: sum(v['draft'][k] for v in item['seeds'].values()) / 2
                for k in ('full_note_f1', 'onset10', 'onset50', 'shift_null10', 'density_ratio')}
            write(report_path, report)
        report.update(status='completed', wall_s=time.monotonic() - started)
        write(report_path, report)
    except BaseException as exc:
        report.update(status='stopped', error=f'{type(exc).__name__}: {exc}', wall_s=time.monotonic() - started)
        write(report_path, report)
        raise


if __name__ == '__main__':
    main()
