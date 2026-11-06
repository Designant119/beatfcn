"""Append scoped decisions only after verified real terminals.

Never closes prepared scripts or turns an implementation/health interruption
into a quality verdict. It does not select models, change gates or run ML.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import subprocess
import time

from scripts.run_local_portfolio_scale import sha, write


ROOT = Path(__file__).resolve().parents[1]
COUNCIL = ROOT / 'reports/council-archive/decisions.jsonl'
RESEARCH = ROOT / 'reports/local-portfolio-v1/research-suggestions.jsonl'
REPORT = ROOT / 'reports/local-portfolio-v1/terminal-closeout-watch-v1.json'


def read(path):
    return json.loads((ROOT / path).read_text())


def latest(path):
    events = [json.loads(v) for v in path.read_text().splitlines()]
    return events, {v['id']: v for v in events}


def record(ident, status, reason, evidence):
    ledger = RESEARCH if ident.startswith('R-') else COUNCIL
    for path in evidence:
        if not (ROOT / path).exists():
            raise ValueError('Missing closure evidence: ' + path)
    with ledger.open('r+') as out:
        fcntl.flock(out.fileno(), fcntl.LOCK_EX)
        events = [json.loads(v) for v in out.read().splitlines()]
        old = [v for v in events if v['id'] == ident][-1]
        if old['status'] != 'deferred':
            return False
        value = dict(old)
        value.update(event_id=ident + '.' + str(sum(v['id'] == ident for v in events) + 1),
            status=status, rationale=reason, evidence=evidence,
            recorded_at=datetime.now(timezone.utc).isoformat(),
            recording_basis='Verified fixed terminal, source/weight identities and preregistered gates; automatic receipt, not a new council opinion',
            context='Finite119item local pool, no partial closures, no newtraining/council/cloud',
            revisit='No automatic extra steps/seed/checkpoint/threshold selection; narrow complete-recipe or diagnostic conclusion only.')
        out.seek(0, 2)
        out.write(json.dumps(value, ensure_ascii=False) + '\n')
        out.flush()
    return True


def completed(path):
    p = ROOT / path
    if not p.exists():
        return None
    value = read(path)
    return value if value.get('status') == 'completed' else None


def arm_verified(value, path, art, arm):
    r = value['arms'][arm]
    if r['status'] != 'completed' or r['health_updates'] != 100 or r['quality_updates'] != 1000:
        raise ValueError('No quality closure for incomplete/failed-health arm: ' + path + ':' + arm)
    if r['final_weight_sha256'] != sha(ROOT / art / arm / 'quality-1000.pt'):
        raise ValueError('Terminal weight identity drift')
    if r['evaluations']['draft']['attempted'] != 32:
        raise ValueError('Missing primary requests')
    return r


def metric(r):
    d = r['evaluations']['draft']
    return dict(full=d['full_note_f1'], onset10=d['onset10'], count=d['density_ratio'],
                complete=d['complete'], attempted=d['attempted'])


def once():
    changes = []
    path = 'reports/local_mert_context_pair_v1.json'
    value = completed(path)
    _, decisions = latest(COUNCIL)
    if value and any(decisions[i]['status'] == 'deferred' for i in ('C03-19', 'C04-S07')):
        rows = {a: arm_verified(value, path, 'artifacts/local-mert-context-pair-v1', a) for a in ('full', 'short32')}
        if not value.get('actual_supervision_and_common_mel_exact'):
            raise ValueError('The paired actual supervision was not verified')
        if rows['full']['initial_weight_sha256'] != rows['short32']['initial_weight_sha256']:
            raise ValueError('Paired initial tensors differ')
        summary = dict(id='local-mert-context-pair-review-v1', status='completed',
            metrics={a: metric(r) for a, r in rows.items()}, comparison=value['full_context_comparison'],
            initial_tensors_exact=True, actual_supervision_and_common_mel_exact=True,
            scope='Same frozenMERT weights/layers/runtime, actual full versus32s extraction, same DirectSong routing. Relative context hypothesis and absolute candidate eligibility remain separate; no emotion/product inference.',
            source_report_sha256=sha(ROOT / path), cloud_compute_calls=0)
        review = 'reports/local_mert_context_pair_review_v1.json'
        write(ROOT / review, summary)
        supported = bool(value['full_context_comparison']['supported'])
        if record('C03-19', 'tested_supported' if supported else 'tested_failed',
            f'Actual paired100health/fresh1000each completed. Full/32s metrics={summary["metrics"]}; fixed relativecontext comparison={summary["comparison"]}. Initial tensors and all4000actual supervision/commonmel inputs match. This closes the exact extraction-context hypothesis; both downstream summaries still cover the whole song. Not a proof about pretraining necessity, semantic preservation or250M scaling.', [review, path, 'reports/local_mert_context_cache_audit_v1.json']):
            changes.append('C03-19')
        if any(r['exposure']['windows'] != 4000 for r in rows.values()):
            raise ValueError('Batch4 actual exposure incomplete')
        if record('C04-S07', 'adopted',
            'The complete batch4/50-25-25/warm100/lr3e-4/WD.01 recipe actually executed1000updates perfrontend,4000requests each; paired actual targets/references/controls/commonmel identical. Density-only count tilt and original compositefieldloss/expected_tail_rank preserved; generated-prefix replay off. Absolute metrics and relative context result are linked. This is a complete recipe test, not a causal claim that batch size or any one optimizer setting independently improved quality.', [review, path, 'bench/rhythm-v2/local-mert-context-pair-v1.json']):
            changes.append('C04-S07')
    path = 'reports/local_attention_control_quality_v1.json'
    value = completed(path)
    if value and any(decisions[i]['status'] == 'deferred' for i in ('C04-G01', 'C04-S01')):
        r = arm_verified(value, path, 'artifacts/local-attention-control-quality-v1', 'full')
        supported = r['local_screen_passed']
        for ident in ('C04-G01', 'C04-S01'):
            if record(ident, 'tested_supported' if supported else 'tested_failed',
                f'Actual standalone10.14M attention control ran100health discarded/fresh1000 on4000requests, fixed compositeobjective/relativeendpoint/expectedtailrank. Metrics={metric(r)}; frozen local gates={r["local_screen"]}. The old100step F1/type health failure is retained; this is the separate explicit fitting/numeric/legal admission followed by real independent-song quality. Not substituted by the16.7M access experiment, and not an isolated width/attention-family verdict.', [path, 'bench/rhythm-v2/local-attention-control-quality-v1.json']):
                changes.append(ident)
    path = 'reports/local_raw_audio_translation_v1.json'
    value = completed(path)
    if value and decisions['C03-18']['status'] == 'deferred':
        if not value['checkpoint_unchanged'] or len(value['rows']) != 4:
            raise ValueError('Raw translation diagnostic identity/coverage incomplete')
        if record('C03-18', 'adopted',
            f'Actual137ms raw waveform translation/re-extraction completed on4fixed development components with zero-shift MERT/mel parity, same checkpoint unchanged, zero updates. Mean onset-only scores={value["mean_scores"]}; component directions={value["positive_translation_components"]}/4. Input values re-extracted from audio, not outputshift/feature roll. Previous7branch interventions remain separate. This is a bounded reused-song dependence/clock diagnostic, not emotion, duration or new-song confirmation.', [path, 'reports/local_raw_audio_translation_features_v1.json', 'reports/local_ssd_acoustic_routes_v1.json']):
            changes.append('C03-18')
    _, research = latest(RESEARCH)
    for kind, ident in [('depth', 'R-LFM3-02'), ('ssl', 'R-LFM3-03')]:
        path = f'reports/local_music_{kind}_pair_v1.json'
        value = completed(path)
        if not value or research[ident]['status'] != 'deferred':
            continue
        arms = ('wide8', 'deep16') if kind == 'depth' else ('supervised', 'matching')
        rows = {a: arm_verified(value, path, f'artifacts/local-music-{kind}-pair-v1', a) for a in arms}
        supported = bool(value['comparison']['hypothesis_supported'])
        if record(ident, 'tested_supported' if supported else 'tested_failed',
            f'Actual paired100health/fresh1000each complete. Metrics={ {a:metric(r) for a,r in rows.items()} }; fixed hypothesis comparison={value["comparison"]}. Same frozen data/objective/readout and declared pair controls; actual matched600second envelope, source-audio andcoarsecontext counterfactuals archived. This closes the implemented budget-allocation/SSL mechanism at this local budget, not all depth/pretraining/semantic hypotheses. No human motif/transition/emotion annotations or playtest claim.', [path, f'bench/rhythm-v2/local-music-{kind}-pair-v1.json', 'bench/rhythm-v2/local-music-research-contract-v3.json']):
            changes.append(ident)
    path = 'reports/local_portfolio_confirmation_v1.json'
    if (ROOT / path).exists() and any(decisions[i]['status'] == 'deferred' for i in ('C00-04', 'C04-A11', 'C04-S08')):
        value = read(path)
        if value['status'] in ('completed', 'not_opened_no_reproduced_finalist'):
            selection_path = 'reports/local_portfolio_finalist_selection_v1.json'
            selection = read(selection_path)
            if len(selection['finalist_identities']) > 2:
                raise ValueError('Finalist cap violated')
            seeds = []
            evidence = [path, selection_path, 'reports/local_portfolio_data_scale_review_v1.json']
            for rank, entry in enumerate(selection['finalist_identities'], 1):
                rp = f'reports/local_finalist_r{rank}_seed2_v1.json'
                r = read(rp)
                if r['status'] not in ('completed', 'health_failed', 'stopped'):
                    raise ValueError('Missing second-seed terminal')
                seeds.append(dict(candidate=entry['id'], status=r['status'], reproduced=r.get('reproduced'), development=r.get('development')))
                evidence.append(rp)
            detail = dict(candidates=[v['id'] for v in selection['finalist_identities']],
                second_seed_outcomes=[{k:v for k,v in r.items() if k != 'development'} for r in seeds],
                confirmation_status=value['status'], confirmation_scored=value['confirmation_scored'],
                confirmed=[dict(candidate=r['candidate'], both_seeds_support=r['both_seeds_support'], mean=r['mean_draft']) for r in value.get('finalists', [])])
            for ident in ('C00-04', 'C04-A11', 'C04-S08'):
                if record(ident, 'adopted',
                    f'Actual bounded exposure/curve/secondseed/conditionalconfirm protocol finished. Fixed selection and outcomes={detail}. Original24/64 curves retained. No bestcheckpoint/draw/seed, thirdseed, replacement finalist or post-confirm training. Confirm8 means8components held out only from this local campaign, not all-project blind. If no reproduced finalist, the declared conditional branch leaves confirmation sealed; it is not falsely reported as tested. Local support does not authorize cloud/product claims.', evidence):
                    changes.append(ident)
    if changes:
        for args in [('python3', 'scripts/archive_council.py', 'sync'), ('python3', 'scripts/archive_council.py', 'check'),
                     (str(ROOT / '.venv/bin/python'), '-m', 'scripts.write_council_closeout')]:
            subprocess.run(args, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
        print(json.dumps(dict(closed_actual_terminals=changes, at=datetime.now(timezone.utc).isoformat())), flush=True)
    return changes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    started = time.monotonic()
    while True:
        try:
            changes = once()
            ce, c = latest(COUNCIL); re, r = latest(RESEARCH)
            open_ids = [k for k, v in {**c, **r}.items() if v['status'] == 'deferred']
            status = dict(status='completed' if not open_ids else 'watching_real_terminals',
                last_checked=datetime.now(timezone.utc).isoformat(), open_ids=open_ids, changes=changes,
                council_events=len(ce), research_events=len(re), cloud_compute_calls=0, no_training_or_gate_mutation=True)
            write(REPORT, status)
            if not args.watch or not open_ids:
                break
        except (json.JSONDecodeError, FileNotFoundError):
            # A producer may be committing a report; this never supplies a pass.
            if not args.watch:
                raise
        except BaseException as exc:
            write(REPORT, dict(status='requires_explicit_review', error=f'{type(exc).__name__}: {exc}'))
            raise
        if time.monotonic() - started > 72000:
            raise TimeoutError('Closeout observation bound')
        time.sleep(30)


if __name__ == '__main__':
    main()
