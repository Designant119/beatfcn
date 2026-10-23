"""Reproducible offset migration and audits of the production sampler and collator.

All outputs are separate artifacts. Migration preserves song splits and fitted train constants.
No raw audio, charts or live cloud manifests are modified.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import numpy as np
from ..ingest import offset as O
from .data import Corpus, dup_only, selected_for_training
from .loop import CoreSampler
from ..model.batch import STATIC, STATIC_CORE, collate_meta, SongInfo
from ..model.features import SkipExample


def fingerprint(rows):
    body = '\n'.join(json.dumps(r, sort_keys=True, separators=(',', ':'))
                     for r in sorted(rows, key=lambda r: r['chart_id']))
    return hashlib.sha256(body.encode()).hexdigest()


def eligible(row):
    return bool(row.get('kept') or dup_only(row)) and selected_for_training(row)


def refresh_offsets(rows, diagnostics):
    """diagnostics(chart row) -> original full offset diagnostic, including agree_counts."""
    updated, changes = [], []
    for original in rows:
        row = deepcopy(original)
        if row['status'] == 'parsed':
            diagnostic = diagnostics(row)
            old_diagnostic = original.get('offset', {})
            if any(diagnostic.get(k) != old_diagnostic.get(k) for k in ('status', 'delta_ms', 'n')):
                raise ValueError(f"diagnostic identity mismatch for {row['chart_id']}")
            if diagnostic.get('status') == 'ok' and 'agree_counts' not in diagnostic:
                raise ValueError(f"full diagnostic missing for {row['chart_id']}")
            previous = original['offset_decision']
            decision = O.decide(diagnostic, previous['c_ms'])
            if 'c_basis' in previous:
                decision['c_basis'] = previous['c_basis']
            row['offset_decision'] = decision
            reasons = list(dict.fromkeys(row['p0_quarantine'] + row['validation_quarantine'] + decision['reasons']))
            row.update(quarantine_reasons=reasons, quarantined=bool(reasons), kept=not reasons)
            if (previous['action'], previous['shift_ms']) != (decision['action'], decision['shift_ms']):
                changes.append({'chart_id': row['chart_id'], 'source': row['source'], 'split': row['split'],
                                'old': previous, 'new': decision,
                                'eligible_before': eligible(original), 'eligible_after': eligible(row)})
        updated.append(row)
    return updated, {'schema': 'beatfcn.offset_migration.v2', 'before_sha256': fingerprint(rows),
                     'after_sha256': fingerprint(updated), 'rows': len(rows), 'changed': len(changes),
                     'transitions': dict(Counter(f"{r['source']}:{r['old']['action']}->{r['new']['action']}" for r in changes)),
                     'selection_changes': dict(Counter(f"{r['split']}:{r['source']}:{r['eligible_before']}->{r['eligible_after']}"
                                                        for r in changes)), 'changes': changes}


def sample_rows(rows, per_mode=128, seed=20261023):
    """Uniform chart sample per mode plus explicit density/hold stress samples; report strata separately."""
    selected = []
    for mode in sorted({r['mode'] for r in rows}):
        group = [r for r in rows if r['mode'] == mode]
        order = sorted(group, key=lambda r: hashlib.sha256(f"{seed}:{r['chart_id']}".encode()).hexdigest())
        taken = {r['chart_id'] for r in order[:per_mode]}
        selected += [{**r, '_audit_stratum': 'uniform_chart'} for r in order[:per_mode]]
        for label, key in [('dense_tail', lambda r: r['n_events'] / max(r['audio_sec'], 1)),
                           ('hold_tail', lambda r: r.get('types', {}).get('hold', 0) / max(r['audio_sec'], 1))]:
            tail = [r for r in sorted(group, key=key, reverse=True) if r['chart_id'] not in taken][:16]
            selected += [{**r, '_audit_stratum': label} for r in tail]
            taken.update(r['chart_id'] for r in tail)
    return selected


def summarize(traces):
    out = {}
    for key in ('mode', 'stratum', 'kind', 'density_band', 'hold_band', 'difficulty'):
        groups = defaultdict(list)
        for t in traces:
            groups[str(t.get(key, 'unknown'))].append(t)
        out[key] = {}
        for label, records in sorted(groups.items()):
            accepted = sum(t['accepted'] for t in records)
            out[key][label] = {'attempted': len(records), 'accepted': accepted,
                               'accept_rate': round(accepted / len(records), 5),
                               'reject_stages': dict(Counter(t['stage'] for t in records if not t['accepted'])),
                               'reasons': dict(Counter(t.get('reason', '') for t in records if not t['accepted']))}
    return out


def audit_cores(root, rows, requests_per_chart=4, seed=20261023, batch_size=32):
    """Runs real requests, salient peaks, target checks and metadata collation on cached real charts.

    Stratified diagnostic probes, NOT an estimate of the song-weighted training distribution.
    Only audio array dimensions use duration-derived SongInfo; real mel is loaded for reference peaks.
    """
    corpus = Corpus(root=str(root))
    trows = corpus.train_rows(rows)
    rejected_charts = [r['chart_id'] for r in rows if not corpus.usable(r)]
    trows = [r for r in trows if r['chart_id'] not in set(rejected_charts)]
    by_id = {r['chart_id']: r for r in rows}
    results, details = {}, []
    for static in (False, True):
        traces, pending, pending_trace = [], [], []
        sampler = CoreSampler(trows, corpus.chart, None, seed, salient=corpus.salient_us,
                              limits=STATIC_CORE if static else None, audit=traces.append)
        batches, collate_rejects = 0, 0
        def flush():
            nonlocal batches, collate_rejects
            if not pending:
                return
            infos = [SongInfo(t['song'], -(-t['duration_us']//40_000), -(-t['duration_us']//2_000_000)) for t in pending_trace]
            try:
                collate_meta(pending, infos, (9, 16, 24), 24, 1024, 1024, True, STATIC if static else None)
                batches += 1
                for trace in pending_trace:
                    trace['collated'] = True
            except SkipExample as exc:
                collate_rejects += 1
                for trace in pending_trace:
                    trace.update(collated=False, collate_reason=str(exc))
            pending.clear()
            pending_trace.clear()
        for row_number, row in enumerate(trows, 1):
            source = by_id[row['chart_id']]
            density = source['n_events']/max(source['audio_sec'], 1)
            hold_ratio = source.get('types', {}).get('hold', 0)/max(source['n_events'], 1)
            for j in range(requests_per_chart):
                request_seed = int.from_bytes(hashlib.sha256(f"{seed}:{row['chart_id']}:{j}".encode()).digest()[:4], 'little')
                try:
                    ex = sampler.core(row, request_seed)
                    pending.append(ex)
                    pending_trace.append(traces[-1])
                except (SkipExample, ValueError, AssertionError):
                    pass
                traces[-1].update(stratum=source.get('_audit_stratum', 'all'),
                                  density_band='<4' if density < 4 else '4-8' if density < 8 else '8-16' if density < 16 else '>=16',
                                  hold_band='0' if hold_ratio == 0 else '<0.2' if hold_ratio < .2 else '>=0.2',
                                  difficulty=source.get('difficulty_label', 'unknown'), static=static)
                if len(pending) == batch_size:
                    flush()
            if row_number % 40 == 0:
                print(f"audit {'static' if static else 'dynamic'}: {row_number}/{len(trows)} charts", flush=True)
        flush()
        results['static' if static else 'dynamic'] = {'stages': summarize(traces), 'batches': batches,
            'collate_rejected_batches': collate_rejects, 'attempted': len(traces),
            'accepted': sum(t['accepted'] for t in traces), 'collated': sum(t.get('collated', False) for t in traces)}
        details += traces
    return {'schema': 'beatfcn.data_audit.v1', 'rows_sha256': fingerprint(rows), 'seed': seed,
            'charts': len(rows), 'usable_charts': len(trows), 'corpus_rejected_charts': rejected_charts,
            'requests_per_chart': requests_per_chart,
            'sampling': 'uniform charts per mode plus separately labelled stress tails; not production-weighted',
            'audio_dimensions': 'duration-derived for collate; actual mel for salient evidence',
            **results}, details
