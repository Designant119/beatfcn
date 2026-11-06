#!/usr/bin/env python3
"""Read-only training monitor: live console + committed JSONL + campaign ledger.

Run with project Python and --watch. It never launches/stops training or changes
guards. State, readable status and deduplicated alerts live in reports/.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports'
TARGET = REPORTS / 'training_monitor_target.json'
CLOUD = REPORTS / 'training_monitor_cloud.json'
STATE = REPORTS / 'training_monitor.json'
STATUS = REPORTS / 'training_monitor.md'
ALERTS = REPORTS / 'training_monitor_alerts.jsonl'
LEDGER = 'evolution/capability-evolution-20261024/campaign.json'
PROJECT_PYTHON = ROOT / '.venv' / 'bin' / 'python'


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, str):
        return re.sub(r'\b(?:ak|sk)-[A-Za-z0-9_-]+', '<REDACTED>', value)
    return value


def save(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def read_json(path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def condition_campaign_status(snapshot, budget, reports=REPORTS, *, now=None, stale_s=180):
    """Join saved stage outcomes and generation results without treating validation as a benchmark."""
    now = time.time() if now is None else now
    stages = budget.get('paid_stages_dispatched', [])
    training = next((s for s in reversed(stages) if s['name'].startswith('matched-training') or s['name'] == 'matched_training'), {})
    if training.get('result', {}).get('complete'):
        for run in [snapshot, *snapshot.get('paired_runs', [])]:
            run['status'] = 'review_required'
            run['alerts'] = [a for a in run.get('alerts', []) if not a.startswith('日志超过')]
        snapshot['training_complete'] = True
    stage = next((s for s in stages if s['name'] == 'paired_evaluation'), None)
    if stage is None:
        return
    console = reports / 'evolution_condition_evaluation_console.log'
    progress = None
    if console.exists():
        for line in console.read_text(errors='replace').splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict) and row.get('event') == 'condition_benchmark':
                progress = row
    age = max(0, now - (console.stat().st_mtime if console.exists() else stage.get('reserved_at', 0)))
    result = stage.get('result') or read_json(reports / 'condition_v1_evaluation.json', {})
    summary = read_json(reports / 'condition_v1_benchmark_summary.json', {})
    snapshot['condition_evaluation'] = {'stage_status': stage['status'], 'progress': progress,
                                        'result': result, 'summary': summary, 'log_age_s': age}
    if stage['status'] == 'stopped_by_user':
        snapshot['status'] = 'evaluation_stopped_by_user'
    elif stage['status'] in ('complete', 'incomplete'):
        snapshot['status'] = 'review_required' if result.get('complete') else 'evaluation_incomplete'
    elif stage['status'] == 'observation_error':
        snapshot['status'] = 'evaluation_observation_error'
    else:
        snapshot['status'] = 'evaluation_log_stale' if age > stale_s else 'evaluating'


def r03_campaign_status(snapshot, budget, target, reports=REPORTS):
    """Keep the current round separate from terminal R02 stages and old metrics."""
    phase = target['phase']
    stage = next((s for s in budget.get('paid_stages_dispatched', [])
                  if s['name'] == f'r03-window-{phase}'), {})
    console = reports / target['console_file']
    counts = re.findall(r'"event": "r03_metadata", "batches": (\d+)',
                        console.read_text(errors='replace')) if console.exists() else []
    snapshot['r03'] = dict(phase=phase, stage_status=stage.get('status', 'not_dispatched'),
                          review=read_json(reports / 'r03_window_review.json', {}),
                          preflight=read_json(reports / 'r03_window_preflight.json', {}))
    if read_json(reports / 'r03_window_train.json', {}).get('complete'):
        snapshot['training_complete'] = True
        for run in [snapshot, *snapshot.get('paired_runs', [])]:
            run['alerts'] = [a for a in run.get('alerts', []) if not a.startswith('日志超过')]
    snapshot['current_work'] = dict(round='R03 全部窗口条件标签',
        status={'prepare': 'metadata_prepare', 'preflight': '真实探针与修复后基线',
                'train': 'train', 'evaluate': 'benchmark'}[phase],
        prepared_batches=int(counts[-1]) if counts else None)
    if stage.get('status') in ('complete', 'incomplete'):
        snapshot['status'] = 'review_required'
        snapshot['current_work']['status'] = 'awaiting_review'
        if phase == 'train':
            result = read_json(reports / 'r03_window_train.json', {})
            snapshot['training_complete'] = bool(result.get('complete'))
    elif stage.get('status') == 'observation_error':
        snapshot['status'] = 'evaluation_observation_error'
    elif phase in ('prepare', 'preflight', 'evaluate'):
        snapshot['status'] = {'prepare': 'preparing', 'preflight': 'probing', 'evaluate': 'evaluating'}[phase]
    if phase in ('prepare', 'preflight'):
        snapshot['alerts'] = [a for a in snapshot['alerts'] if not a.startswith('日志超过')]
    if target.get('active_diagnostic'):
        diagnostic = next((s for s in budget.get('paid_stages_dispatched', [])
                           if s['name'] == target['active_diagnostic']), {})
        snapshot['active_diagnostic'] = diagnostic
        if diagnostic.get('status') == 'running':
            snapshot['status'] = 'diagnosing'
            snapshot['current_work'] = dict(round=diagnostic['name'], status='diagnostic')


def r04_campaign_status(snapshot, budget, target, reports=REPORTS):
    phase = target['phase']
    stage = next((s for s in budget.get('paid_stages_dispatched', [])
                  if s['name'] == target.get('stage_name', f'r04-initial-{phase}')), {})
    snapshot['r03'] = dict(review=read_json(reports / 'r03_window_review.json', {}))
    snapshot['r04'] = dict(phase=phase, stage_status=stage.get('status', 'not_dispatched'),
        preflight=read_json(reports / 'r04_preflight.json', {}), review=read_json(reports / 'r04_review.json', {}))
    snapshot['current_work'] = dict(round='R04 初始决策监督',
        status={'preflight':'真实 batch 拟合与旧对照等价检查', 'train':'train', 'evaluate':'benchmark'}[phase])
    snapshot['training_complete'] = bool(read_json(reports / 'r04_train.json', {}).get('complete'))
    snapshot['historical_training_metrics'] = phase == 'preflight'
    if stage.get('status') in ('complete', 'incomplete'):
        snapshot['status'] = 'review_required'
        snapshot['current_work']['status'] = 'awaiting_review'
    elif stage.get('status') == 'failed':
        snapshot['status'] = 'failed'
        snapshot['current_work']['status'] = 'needs_diagnosis'
        snapshot.setdefault('alerts', []).append(stage.get('error', 'R04 前置检查或执行失败'))
    elif stage.get('status') == 'observation_error':
        snapshot['status'] = 'evaluation_observation_error'
    else:
        snapshot['status'] = {'preflight':'probing', 'train':'training', 'evaluate':'evaluating'}[phase]
    if phase != 'train' or snapshot['training_complete']:
        snapshot['alerts'] = [a for a in snapshot.get('alerts', []) if not a.startswith('日志超过')]
    verdict = snapshot['r04']['review'].get('comparison', {})
    if phase == 'evaluate' and verdict.get('short_window_gate_passed') is False:
        snapshot['status'] = 'experiment_rejected'
        snapshot['current_work']['status'] = 'rejected'


def r05_campaign_status(snapshot, budget, target, reports=REPORTS):
    phase=target['phase']
    stage=next((s for s in budget.get('paid_stages_dispatched', [])
                if s['name']==f'r05-direct-condition-{phase}'),{})
    snapshot['r05']=dict(phase=phase,stage_status=stage.get('status','not_dispatched'),
        preflight=read_json(reports/'r05_preflight.json',{}),review=read_json(reports/'r05_review.json',{}))
    snapshot['r04']=dict(review=read_json(reports/'r04_review.json',{}))
    snapshot['training_complete']=bool(read_json(reports/'r05_train.json',{}).get('complete'))
    snapshot['historical_training_metrics']=phase=='preflight'
    snapshot['round_label']='R05'
    snapshot['current_work']=dict(round='R05 直接条件残差',status={
        'preflight':'真实 batch 探针与零初始化等价检查','train':'train','evaluate':'benchmark'}[phase])
    status=stage.get('status')
    snapshot['status']= {'complete':'review_required','incomplete':'evaluation_incomplete',
        'failed':'failed','observation_error':'evaluation_observation_error'}.get(status,
        {'preflight':'probing','train':'training','evaluate':'evaluating'}[phase] if status in ('reserved','running') else 'preparing')
    if phase!='train' or snapshot['training_complete']:
        snapshot['alerts']=[a for a in snapshot.get('alerts',[]) if not a.startswith('日志超过')]
    if status=='failed':snapshot.setdefault('alerts',[]).append(stage.get('error','R05 阶段未通过，请检查结果。'))
    if phase=='evaluate' and snapshot['r05']['review'].get('comparison',{}).get('short_window_gate_passed') is False:
        snapshot['status']='experiment_rejected';snapshot['current_work']['status']='rejected'


def r06_campaign_status(snapshot, budget, target, reports=REPORTS):
    phase=target['phase']
    stage=next((s for s in budget.get('paid_stages_dispatched', [])
                if s['name']==f'r06-ordered-time-{phase}'),{})
    snapshot['r06']=dict(phase=phase,stage_status=stage.get('status','not_dispatched'),
        preflight=read_json(reports/'r06_preflight.json',{}),review=read_json(reports/'r06_review.json',{}),
        rollback=read_json(reports/'r06_rollback.json',{}))
    snapshot['r05']=dict(review=read_json(reports/'r05_review.json',{}))
    snapshot['training_complete']=bool(read_json(reports/'r06_train.json',{}).get('complete'))
    snapshot['historical_training_metrics']=phase=='preflight'
    snapshot['round_label']='R06'
    snapshot['current_work']=dict(round='R06 有序时间风险残差',status={
        'preflight':'真实 batch 探针与零初始化等价检查','train':'train','evaluate':'benchmark'}[phase])
    status=stage.get('status')
    snapshot['status']={'complete':'review_required','incomplete':'evaluation_incomplete',
        'failed':'failed','observation_error':'evaluation_observation_error'}.get(status,
        {'preflight':'probing','train':'training','evaluate':'evaluating'}[phase] if status in ('reserved','running') else 'preparing')
    if phase!='train' or snapshot['training_complete']:
        snapshot['alerts']=[a for a in snapshot.get('alerts',[]) if not a.startswith('日志超过')]
    if status=='failed':snapshot.setdefault('alerts',[]).append(stage.get('error','R06 阶段未通过，请检查结果。'))
    verdict=snapshot['r06']['review'].get('comparison',{}).get('short_window_gate_passed')
    if phase=='evaluate' and verdict is not None:
        snapshot['status']='research_retained' if verdict else 'experiment_rejected'
        snapshot['current_work']['status']='research_retained' if verdict else 'rejected'
        snapshot['alerts']=[a for a in snapshot.get('alerts',[])
                            if not a.startswith('已到 1000 次有效更新复盘点')]


def r07_campaign_status(snapshot, budget, target, reports=REPORTS):
    phase=target['phase']
    prefix=target.get('report_prefix','r07')
    if prefix not in ('r07','r07b'):raise ValueError('unknown context round')
    stage=next((s for s in budget.get('paid_stages_dispatched', [])
                if s['name']==f'{prefix}-blank-context-{phase}'),{})
    snapshot['r07']=dict(phase=phase,stage_status=stage.get('status','not_dispatched'),
        preflight=read_json(reports/f'{prefix}_preflight.json',{}),review=read_json(reports/f'{prefix}_review.json',{}),
        rollback=read_json(reports/f'{prefix}_rollback.json',{}),
        infrastructure=read_json(reports/f'{prefix}_infrastructure_review.json',{}))
    snapshot['r06']=dict(review=read_json(reports/'r06_review.json',{}), rollback=read_json(reports/'r06_rollback.json',{}))
    snapshot['training_complete']=bool(read_json(reports/f'{prefix}_train.json',{}).get('complete'))
    snapshot['historical_training_metrics']=phase=='preflight'
    snapshot['round_label']=prefix.upper()
    snapshot['current_work']=dict(round=f'{prefix.upper()} 外部上下文移除',status={
        'preflight':'真实 batch 探针与独立请求重建检查','train':'train','evaluate':'benchmark'}[phase])
    status=stage.get('status')
    snapshot['status']={'complete':'review_required','incomplete':'evaluation_incomplete',
        'failed':'failed','observation_error':'evaluation_observation_error'}.get(status,
        {'preflight':'probing','train':'training','evaluate':'evaluating'}[phase] if status in ('reserved','running') else 'preparing')
    if phase!='train' or snapshot['training_complete']:
        snapshot['alerts']=[a for a in snapshot.get('alerts',[]) if not a.startswith('日志超过')]
    if status=='failed':snapshot.setdefault('alerts',[]).append(stage.get('error','R07 阶段未通过，请检查结果。'))
    if snapshot['r07']['infrastructure'].get('status') == 'infrastructure_interrupted':
        snapshot['status'] = 'infrastructure_interrupted'
        snapshot['current_work']['status'] = 'data_loading_interrupted'
        snapshot['alerts'] = [a for a in snapshot.get('alerts', []) if not a.startswith('日志超过')]
    verdict=snapshot['r07']['review'].get('comparison',{}).get('short_window_gate_passed')
    if phase=='evaluate' and verdict is not None:
        snapshot['status']='research_retained' if verdict else 'experiment_rejected'
        snapshot['current_work']['status']='research_retained' if verdict else 'rejected'
        snapshot['alerts']=[a for a in snapshot.get('alerts',[])
                            if not a.startswith('已到 1000 次有效更新复盘点')]


def active_diagnostic_status(snapshot, budget, target):
    """Show the running diagnostic while retaining the latest model scorecard."""
    name = target.get('active_diagnostic')
    if not name:
        return
    stage = next((s for s in budget.get('paid_stages_dispatched', []) if s['name'] == name), {})
    snapshot['active_diagnostic'] = stage
    status = stage.get('status')
    if status in ('reserved', 'running', 'complete', 'incomplete', 'failed', 'observation_error'):
        snapshot['status'] = {'reserved':'diagnosing', 'running':'diagnosing', 'complete':'review_required',
            'incomplete':'evaluation_incomplete', 'failed':'failed',
            'observation_error':'evaluation_observation_error'}[status]
        snapshot['current_work'] = dict(round=name, status=status)


def console_record(line, observed_at):
    """Recover the failure header from old 600-character console records."""
    run_id, _, payload = line.partition(' ')
    try:
        return run_id, json.loads(payload)
    except ValueError:
        if '"event": "nonfinite_gradient"' not in payload:
            return None
        record = {'event': 'nonfinite_gradient', 'ts': observed_at}
        for key in ('attempted_step', 'completed_updates'):
            match = re.search(r'"' + key + r'":\s*(\d+)', payload)
            if match:
                record[key] = int(match.group(1))
        return run_id, record


class Monitor:
    def __init__(self):
        self.runs, self.seen = {}, set()

    def ingest(self, run_id, record):
        event = record.get('event')
        if not event:
            return
        identity = (run_id, event, record.get('ts'), record.get('step'), record.get('attempted_step'))
        if identity in self.seen:
            return
        self.seen.add(identity)
        r = self.runs.setdefault(run_id, {'run_id': run_id, 'status': 'starting', 'step': None,
                                         'start_step': None, 'metrics': {}, 'alerts': []})
        r['last_event_at'] = max(r.get('last_event_at', 0), record.get('ts', 0))
        if event == 'data':
            r['validation_prep'] = record.get('prep')
        if event == 'data_identity':
            r['rows_sha256'] = record.get('rows_sha256')
        if event == 'start':
            r['model_config'] = record.get('model')
        if event in ('warm_start', 'resume') and record.get('ts', 0) >= r['last_event_at']:
            r['status'], r['resumed_at'], r['alerts'] = 'starting', record.get('ts', 0), []
            if r.get('failure'):
                r['previous_failure'] = r.pop('failure')
        if event in ('warm_start', 'resume', 'round_baseline'):
            r['start_step'] = record.get('round_start_step', record['step'])
            r['step'] = max(r.get('step') or 0, record['step'])
        if event == 'train' and record.get('step', 0) >= (r['step'] or 0):
            previous = r['metrics']
            r['status'], r['step'] = 'training', record['step']
            fields = record.get('fields', {})
            losses = [v['loss'] for v in fields.values() if isinstance(v, dict) and isinstance(v.get('loss'), (int, float))]
            alloc = record.get('alloc', {})
            elapsed = record.get('elapsed_s')
            interval = elapsed - previous['elapsed_s'] if elapsed is not None and previous.get('elapsed_s') is not None else None
            examples_delta = record['examples'] - previous['examples'] if 'examples' in record and previous.get('examples') is not None else None
            r['metrics'] = {
                'loss': sum(losses) if losses else None,
                'grad_norm': record.get('grad_norm'), 'gpu_util_percent': record.get('gpu_util'),
                'allocated_gib': alloc.get('allocated_bytes.all.current', 0) / 2**30 if 'allocated_bytes.all.current' in alloc else None,
                'reserved_gib': alloc.get('reserved_gb'),
                'peak_gb': record.get('peak_gb'),
                'pool_device_gib': record.get('pool_memory', {}).get('total_device_bytes', 0) / 2**30 if record.get('pool_memory') else None,
                'data_wait_percent': 100 * record['data_wait_s'] / elapsed if elapsed and 'data_wait_s' in record else None,
                'elapsed_s': elapsed, 'fields': fields,
                'examples': record.get('examples'), 'recorded_at': record.get('ts'),
                'cores_per_s': examples_delta / interval if interval and interval > 0 and examples_delta is not None and examples_delta >= 0 else None,
                'input_timing': record.get('data', {}),
                'accepted_training_exposure': record.get('accepted_training_exposure', {}),
            }
            if any(not math.isfinite(x) for x in losses + [record.get('grad_norm', 0)] if isinstance(x, (int, float))):
                r['status'] = 'failed'
                r['alerts'].append('loss 或梯度出现非有限值')
        if event in ('val', 'round_baseline'):
            r.setdefault('validations', {})[record['step']] = record
        if event in ('nonfinite_gradient', 'diagnostic_failure', 'abort', 'aborted') and record.get('ts', 0) >= r.get('resumed_at', 0):
            r['status'] = 'stopped_by_guard' if event in ('abort', 'aborted') else 'failed'
            r['failure'] = record
            if 'step' in record:
                r['step'] = record['step']
            if r['start_step'] is not None and 'completed_updates' in record:
                r['step'] = r['start_step'] + record['completed_updates']
            r['alerts'].append(f"{event}: {record.get('error') or record.get('reason') or '训练已中止'}")

    def model_progress(self, run_id):
        """Compare only matching fixed validation data and model/loss configurations."""
        current = self.runs.get(run_id, {})
        def identity(run):
            parts = [run.get('validation_prep'), run.get('rows_sha256'), run.get('model_config')]
            return json.dumps(parts, sort_keys=True) if all(parts) else None
        key = identity(current)
        comparable = [r for r in self.runs.values()
                      if r is current or (key is not None and identity(r) == key
                                          and r.get('start_step') == current.get('start_step'))]
        points = {}
        for run in comparable:
            for step, record in run.get('validations', {}).items():
                if run is not current and step != current.get('start_step'):
                    continue  # Identical architecture/data do not make different trained weights interchangeable.
                if record.get('ts', 0) >= points.get(step, {}).get('ts', 0):
                    points[step] = dict(record, run_id=run['run_id'])
        latest = max(current.get('validations', {}).values(), key=lambda v: (v['step'], v.get('ts', 0)), default=None)
        baseline = points.get(current.get('start_step'))
        return {'latest_validation': latest, 'baseline': baseline,
                'history': [points[s] for s in sorted(points)],
                'comparison': 'matched' if baseline and latest else 'baseline_or_validation_missing'}

    def snapshot(self, run_id, *, now=None, stale_s=180, ledger=None, benchmark=None):
        now = time.time() if now is None else now
        r = dict(self.runs.get(run_id, {'run_id': run_id, 'status': 'waiting_for_log', 'metrics': {}}))
        alerts = list(r.get('alerts', []))
        start, step = r.get('start_step'), r.get('step')
        r['effective_updates'] = max(0, step - start) if step is not None and start is not None else None
        r['log_age_s'] = round(max(0, now - r['last_event_at']), 1) if r.get('last_event_at') else None
        r['model_progress'] = self.model_progress(run_id)
        r.pop('validations', None)
        r['benchmark'] = benchmark
        if ledger:
            all_rounds = ledger.get('rounds', {})
            if all_rounds:
                name, work = max(all_rounds.items(), key=lambda item: item[1].get('started_at', 0))
                result = work.get('diagnostic_result', {})
                r['current_work'] = {'round': name, 'status': work['status'], 'finished_at': work.get('finished_at'),
                                     'prepared_batches': result.get('batches'), 'complete': result.get('complete')}
            rounds = [v for v in ledger.get('rounds', {}).values()
                      if v.get('plan', {}).get('training', {}).get('run_id') == run_id
                      and not v['plan'].get('diagnostic_only') and not v['plan'].get('gradient_replay')]
            if rounds:
                current = max(rounds, key=lambda v: v.get('started_at', 0))
                r['campaign_stage'] = current['status']
                if current['status'] == 'needs_diagnosis':
                    guard = current.get('training_result', {}).get('aborted')
                    training_complete = current.get('training_result', {}).get('round_effective_updates') == 1000
                    r['status'] = 'stopped_by_guard' if guard else 'evaluation_failed' if training_complete else 'failed'
                    reason = current.get('error') or guard or '训练未完成，等待诊断'
                    if not any(a.endswith(reason) for a in alerts):
                        alerts.append(reason)
                elif current['status'] == 'awaiting_review':
                    r['status'] = 'review_required'
                elif current['status'] == 'benchmark':
                    r['status'] = 'evaluating'
            r['budget'] = {'approved_usd': ledger.get('approved_usd'),
                           'reserved_usd': sum(s.get('reserved_usd', 0) for s in ledger.get('stages', {}).values()),
                           'estimated_completed_usd': sum(s.get('estimated_usd', 0) for s in ledger.get('stages', {}).values()),
                           'note': '预留是保守资源上限；已结束阶段估算不是账单，未含正在运行的阶段。'}
        if r['status'] in ('starting', 'training') and r['log_age_s'] is not None and r['log_age_s'] > stale_s:
            r['status'] = 'log_stale'
            alerts.append(f'日志超过 {stale_s:g} 秒未更新，需核实运行状态')
        if (r.get('effective_updates') or 0) >= 1000:
            alerts.append('已到 1000 次有效更新复盘点：先审查 benchmark 再继续')
        r.update(observed_at=now, alerts=list(dict.fromkeys(alerts)))
        return clean(r)


def pull_cloud():
    # Run in a bounded subprocess so network/auth failures cannot hang the local watcher.
    import modal
    volume = modal.Volume.from_name('beatfcn-data')
    target = read_json(TARGET, {}) or {}
    if target.get('campaign_ledger') == 'rhythm_v2_budget.json':
        logs, errors = {}, {}
        for run_id in target['run_ids']:
            try:
                logs[run_id] = b''.join(volume.read_file(f'runs/{run_id}/log.jsonl')).decode()
            except Exception as exc:
                if target.get('phase') not in ('prepare', 'preflight'):
                    errors[run_id] = f'{type(exc).__name__}: {exc}'
        save(CLOUD, {'fetched_at': time.time(), 'run_id': target['run_id'], 'ledger': None,
                     'log': logs.get(target['run_id'], ''), 'paired_logs': logs, 'log_error': str(errors) if errors else None})
        return
    ledger = json.loads(b''.join(volume.read_file(LEDGER)))
    eligible = [v for v in ledger['rounds'].values()
                if not any(v['plan'].get(k) for k in ('diagnostic_only', 'gradient_replay', 'allocation_probe', 'transport_probe', 'live_input_probe', 'metadata_prepare', 'metadata_compress'))]
    current = max(eligible, key=lambda v: v['started_at']) if eligible else None
    run_id = current['plan']['training']['run_id'] if current else (read_json(TARGET, {}) or {}).get('run_id')
    log, error = '', None
    if run_id:
        try:
            log = b''.join(volume.read_file(f'runs/{run_id}/log.jsonl')).decode()
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
    benchmark, benchmark_error = None, None
    if current and (current['status'] == 'benchmark' or current.get('benchmark_result')):
        try:
            path = f"evolution/capability-evolution-20261024/{current['plan']['round']}/benchmark.json"
            benchmark = json.loads(b''.join(volume.read_file(path)))
        except Exception as exc:
            benchmark_error = f'{type(exc).__name__}: {exc}'
    save(CLOUD, {'fetched_at': time.time(), 'run_id': run_id, 'ledger': ledger, 'log': log, 'log_error': error,
                 'benchmark': benchmark, 'benchmark_error': benchmark_error})


def local_probe_status(snapshot, target, reports=REPORTS, command_lookup=None):
    """Keep disposable local fitting separate from the latest formal model."""
    target = target.get('local_probe')
    if not target:
        return
    result = read_json(reports / target['result'], {}) or {}
    resource = read_json(reports / target['resources'], {}) or {}
    pid = resource.get('pid')
    if command_lookup is None:
        def command_lookup(pid):
            if not isinstance(pid, int) or pid <= 0:
                return ''
            try:
                p = subprocess.run(['ps', '-p', str(pid), '-o', 'command='], capture_output=True,
                                   text=True, timeout=2)
                return p.stdout.strip() if p.returncode == 0 else ''
            except subprocess.TimeoutExpired:
                return ''
    command = command_lookup(pid)
    alive = bool(command and target['script'] in command and target['plan'] in command)
    if resource.get('complete'):
        status = ('passed' if result.get('probe_passed') else 'failed') if result.get('complete') else 'interrupted'
    else:
        status = 'running' if alive else 'process_not_confirmed'
    history = result.get('history') or []
    snapshot['local_probe'] = dict(status=status, process_confirmed_live=alive, pid=pid,
        optimizer_updates=result.get('optimizer_updates', 0), expected_updates=target['steps'],
        latest_fixed_batch=history[-1] if history else None, wall_s=result.get('wall_s'),
        peak_rss_gib=max(result.get('peak_rss_gib', 0), resource.get('maximum_rss_gib', 0)),
        phase=result.get('phase'), probe_passed=result.get('probe_passed'), model_promotion=False,
        scope_label=target.get('scope_label', '这些是两个固定训练片段的记忆测试'),
        supervised_gate=result.get('supervised_gate'), binding_gate=result.get('binding_gate'),
        generation_gate=result.get('generation_gate'),
        generation_summary=result.get('summary') if result.get('complete') else None)


def external_baseline_status(snapshot, target, reports=REPORTS):
    """Show completed external inference without promoting the trained model."""
    from hashlib import sha256
    config = target.get('external_baseline')
    if not config:
        return
    review = read_json(reports / config['review'], {}) or {}
    path = reports / config['benchmark']
    expected = review.get('source_sha256', {}).get('reports/' + config['benchmark'])
    valid = bool(review.get('complete') and path.is_file() and expected
                 and sha256(path.read_bytes()).hexdigest() == expected)
    snapshot['external_baseline'] = dict(status='complete' if valid else 'evidence_incomplete',
        model_promotion=False, optimizer_updates=0, scope_label=config['scope_label'],
        review=config['review'])
    if valid:
        benchmark = read_json(path, {})
        snapshot['external_baseline'].update(charts=benchmark['charts'],
            screen_passed=review['screen_passed'],
            hold_control=review['hold_control_observation'],
            new_cloud_spend_usd=review['new_cloud_spend_usd'])


def audio_coordinate_status(snapshot, target, reports=REPORTS):
    """Keep representation measurements separate from generated-chart quality."""
    from hashlib import sha256
    config = target.get('audio_coordinate_diagnostic')
    if not config:
        return
    review = read_json(reports / config['review'], {}) or {}
    path = reports / config['audit']
    expected = review.get('source_sha256', {}).get('reports/' + config['audit'])
    valid = bool(review.get('complete') and path.is_file() and expected
                 and sha256(path.read_bytes()).hexdigest() == expected)
    snapshot['audio_coordinate_diagnostic'] = dict(
        status='complete' if valid else 'evidence_incomplete',
        model_promotion=False, optimizer_updates=0, review=config['review'])
    if valid:
        snapshot['audio_coordinate_diagnostic'].update(
            charts=review['charts'], summary=review['monitor_summary'],
            coarse_screen_passed=review['coordinate_screen_passed'],
            new_cloud_spend_usd=review['new_cloud_spend_usd'])


def paired_state_probe_status(snapshot, target, reports=REPORTS):
    """Show the new paired fixed-batch result without replacing model benchmarks."""
    from hashlib import sha256
    config = target.get('paired_state_probe')
    if not config:
        return
    review = read_json(reports / config['review'], {}) or {}
    path = reports / config['result']
    expected = review.get('source_sha256', {}).get('reports/' + config['result'])
    valid = bool(review.get('complete') and expected and path.is_file()
                 and sha256(path.read_bytes()).hexdigest() == expected)
    snapshot['paired_state_probe'] = dict(status='evidence_incomplete', model_promotion=False)
    if valid:
        snapshot['paired_state_probe'].update(
            status='passed' if review['passed'] else 'failed',
            summary=review['summary'], optimizer_updates=review['optimizer_updates'],
            scope=review['scope'], review=config['review'],
            wall_s=review['bounded_process_s'], peak_rss_gib=review['peak_rss_gib'],
            new_cloud_spend_usd=review['new_cloud_spend_usd'],
            interpretation_note=review.get('interpretation_note'))


def candidate_probe_status(snapshot, target, reports=REPORTS):
    """Keep current candidate health visible without calling it a benchmark."""
    from hashlib import sha256
    config = target.get('candidate_probes')
    if not config:
        return
    result = read_json(reports / config['report'], {}) or {}
    sources = result.get('sources_sha256', {})
    if not result.get('complete') or not sources:
        return
    for name, expected in sources.items():
        path = reports.parent / name
        if not path.is_file() or sha256(path.read_bytes()).hexdigest() != expected:
            snapshot['candidate_probes'] = {'status': 'evidence_incomplete'}
            return
    snapshot['candidate_probes'] = dict(status='available', **result)


def local_multifidelity_status(snapshot, target, reports=REPORTS, command_lookup=None):
    """Keep local implementation fit separate from song-disjoint generation."""
    from hashlib import sha256
    config = target.get('local_multifidelity')
    if not config:
        return
    item = dict(status='pending', health=[], curve=[], model_promotion=False,
                missing=['整曲生成', '七项Phira条件干预', '参考交换', '音频错配', '300首以上规模曲线'])
    snapshot['local_multifidelity'] = item
    source_root = reports.parent / config.get('source_snapshot', '.')
    for name in config.get('health_reports', []):
        result = read_json(reports / name, {}) or {}
        sources = result.get('sources_sha256', {})
        if not sources or not all((source_root / p).is_file()
                and sha256((source_root / p).read_bytes()).hexdigest() == h for p, h in sources.items()):
            item['health'].append(dict(report=name, status='evidence_incomplete'))
            continue
        item['health'].append(dict(report=name, status='passed' if result.get('passed') else 'failed',
            parameters=result['parameters'], updates=result['updates'], train_s=result.get('train_s'),
            request_mean_ce=result.get('final', {}).get('request_mean_ce')))
    result_path = reports / config['result']
    result = read_json(result_path, {}) or {}
    if not result:
        return
    plan_path = reports.parent / config['plan']
    if not plan_path.is_file() or result.get('plan_sha256') != sha256(plan_path.read_bytes()).hexdigest():
        item['status'] = 'evidence_incomplete'
        return
    sources = result.get('source_sha256', {})
    if not sources or not all((source_root / p).is_file()
            and sha256((source_root / p).read_bytes()).hexdigest() == h for p, h in sources.items()):
        item['status'] = 'evidence_incomplete'
        return
    pid = config.get('pid')
    if command_lookup is None:
        def command_lookup(p):
            return subprocess.run(['ps', '-p', str(p), '-o', 'command='],
                capture_output=True, text=True, timeout=5).stdout
    command = command_lookup(pid) if pid else ''
    alive = config['plan'] in command and 'scripts.run_local_learning_curve' in command
    status = ('failed' if result.get('error') else 'complete') if result.get('complete') else (
        'running' if alive else 'interrupted')
    item.update(status=status, updates=result.get('updates'), curve=result.get('curve', []),
        valid_tokens=result.get('valid_tokens'), requests=result.get('requests'),
        latest_train_ce=result.get('latest_train_ce'), panel=result.get('panel_summary'),
        train_s=result.get('train_s'), wall_s=result.get('wall_s'), memory=result.get('memory'),
        report_age_s=max(0, time.time()-result_path.stat().st_mtime), error=result.get('error'),
        early_stop=result.get('early_stop'), scope=result.get('scope'))
    for key in ('audio_audit', 'condition_audit'):
        evidence = config.get(key)
        if not evidence:
            continue
        path = reports / evidence['report']
        if path.is_file() and sha256(path.read_bytes()).hexdigest() == evidence['sha256']:
            item[key] = read_json(path, {})
    if item.get('audio_audit', {}).get('complete'):
        item['missing'].remove('音频错配')
    if item.get('condition_audit', {}).get('complete'):
        item['missing'].remove('七项Phira条件干预')
        item['missing'].append('其余五项Phira条件及已测条件的多歌泛化')
    snapshot['status'] = 'local_' + status


def local_song_training_status(snapshot, target, reports=REPORTS, command_lookup=None):
    """Current complete-song baseline, without relabeling the old D65 results."""
    from hashlib import sha256
    config = target.get('local_song_training')
    if not config:
        return
    item = dict(status='pending', whole_song_review='pending', condition_response='unmeasured',
                human_style_continuation='untrained', model_promotion=False)
    snapshot['local_song_training'] = item
    path = reports/config['result']
    result = read_json(path, {}) or {}
    if not result:
        return
    plan = reports.parent/config['plan']
    source = reports.parent/config.get('source_snapshot', '.')
    if (not plan.is_file() or sha256(plan.read_bytes()).hexdigest() != result.get('plan_sha256')
            or not result.get('source_sha256') or not all((source/p).is_file()
            and sha256((source/p).read_bytes()).hexdigest() == h for p, h in result['source_sha256'].items())):
        item['status'] = 'evidence_incomplete'
        return
    if command_lookup is None:
        command_lookup = lambda pid: subprocess.run(['ps', '-p', str(pid), '-o', 'command='],
            capture_output=True, text=True, timeout=5).stdout
    command = command_lookup(config['pid']) if config.get('pid') else ''
    alive = config.get('script', 'scripts.run_local_song_baseline') in command and config['plan'] in command
    generation = read_json(reports/config['generation'], {}) or {}
    generation_bound = bool(result.get('checkpoint_sha256')) and generation.get('checkpoint_sha256') == result.get('checkpoint_sha256')
    runner = config.get('generation_runner') or {}
    if runner:
        generation_plan = reports.parent/runner['plan']
        generation_bound = generation_bound and generation_plan.is_file() and sha256(generation_plan.read_bytes()).hexdigest() == generation.get('plan_sha256')
        gen_command = command_lookup(runner['pid']) if runner.get('pid') else ''
        alive |= runner['script'] in gen_command and runner['plan'] in gen_command
    generation_error = generation.get('error') if generation_bound else None
    status = ('failed' if result.get('error') or generation_error else 'awaiting_review' if result.get('complete') or
              (generation_bound and generation.get('execution_complete')) else
              'running' if alive else 'yielded' if result.get('phase') in
              ('explicit_driver_probe_yield', 'bounded_training_yield') else 'interrupted')
    item.update(status=status, label=config.get('label', 'D68'), updates=result['updates'], curve=result['curve'],
        latest_train_ce=result.get('latest_train_ce'), parameters=result['parameters'],
        valid_tokens=result['valid_tokens'], requests=result['requests'], phase=result.get('phase'),
        train_s=result['train_s'], preparation_s=result['preparation_s'], evaluation_s=result['evaluation_s'],
        memory=result.get('memory'), error=result.get('error') or generation_error, scope=result.get('scope'),
        report_age_s=max(0, time.time()-path.stat().st_mtime))
    if 'auxiliary_pair_updates' in result:
        item['auxiliary_supervision'] = {k: result.get(k) for k in (
            'auxiliary_pair_updates', 'auxiliary_prefix_views', 'auxiliary_prefix_tokens',
            'auxiliary_mode_exposure', 'auxiliary_field_exposure', 'latest_auxiliary_loss', 'latest_interaction_gap')}
    if generation_bound:
        item['whole_songs'] = dict(complete=generation.get('complete'),
            backend=generation.get('backend', config.get('generation_backend', 'unrecorded')),
            completed=generation.get('completed_songs', 0), expected=generation.get('expected_songs', 20),
            attempted=len(generation.get('song_files', {})), failures=generation.get('failures', {}))
    review_config = config.get('review') or {}
    if review_config:
        review_path = reports/review_config['report']
        if review_path.is_file() and sha256(review_path.read_bytes()).hexdigest() == review_config.get('sha256'):
            review = read_json(review_path, {}) or {}
            records = review.get('records', [])
            if (result.get('checkpoint_sha256') and review.get('checkpoint_sha256') == result['checkpoint_sha256']
                    and len(records) == 20):
                f1 = [r['full_note_f1'] for r in records if r.get('full_note_f1') is not None]
                item['review_summary'] = dict(songs=len(records), completed=sum(bool(r['complete']) for r in records),
                    raw_full_note_f1_mean=sum(f1)/len(f1) if f1 else None, f1_songs=len(f1),
                    empty_songs=sum(r['notes']['model'] == 0 for r in records),
                    model_notes=sum(r['notes']['model'] for r in records), human_notes=sum(r['notes']['human'] for r in records),
                    assistant_review_complete=bool(review.get('assistant_review_complete')),
                    scope='All raw outputs, including incomplete attempts; reference matching is descriptive')
    for key in ('timeline_review', 'condition_audit', 'audio_audit'):
        evidence = config.get(key) or {}
        evidence_path = reports/evidence.get('report', '__missing__')
        if evidence_path.is_file() and sha256(evidence_path.read_bytes()).hexdigest() == evidence.get('sha256'):
            audited = read_json(evidence_path, {}) or {}
            if audited.get('checkpoint_sha256') != result.get('checkpoint_sha256'):
                continue
            if (key == 'timeline_review' and audited.get('songs') == 20 and audited.get('timeline_visual_review_complete')
                    and 'review_summary' in item and audited.get('automatic_evidence_sha256') == review_config.get('sha256')):
                item['whole_song_review'] = 'timeline_reviewed_audition_pending'
                item['timeline_review'] = dict(report=evidence['report'],
                    aggregate=audited['aggregate'], audio_audition_complete=audited['audio_audition_complete'],
                    native_playtest_complete=audited['native_playtest_complete'], artifact_index=config.get('artifact_index'))
                if audited.get('quality_passed') is False:
                    item['status'] = status = 'reviewed_failed'
            elif (key == 'condition_audit' and audited.get('execution_complete')
                  and audited.get('expansion_allowed') is False
                  and any(r.get('sentinel_passed') is False for r in audited.get('by_field', {}).values())):
                item['condition_response'] = 'required_field_failed_expansion_stopped'
                item['condition_audit'] = dict(by_field=audited['by_field'],
                    combined_gate=None, stop_reason=audited.get('stop_reason'),
                    unmeasured_fields=audited.get('unmeasured_fields', []))
            elif (key == 'condition_audit' and audited.get('complete') and set(audited.get('by_field', {})) ==
                  {'density_notes_per_second', 'hold_head_ratio'}):
                combined = audited.get('combined_gate')
                item['condition_response'] = ('combined_sentinel_only' if combined.get('passed') else
                    'combined_sentinel_failed') if combined is not None else ('two_field_sentinel_failed' if not any(
                    r.get('sentinel_passed') for r in audited.get('by_field', {}).values()) else 'sentinel_only')
                item['condition_audit'] = dict(by_field=audited.get('by_field', {}),
                    combined_gate=combined, stop_reason=audited.get('stop_reason'))
            elif key == 'audio_audit' and audited.get('complete') and len(audited.get('rows', [])) == 4:
                item['audio_audit'] = dict(pairs=4, identical_pairs=audited.get('identical_pairs'),
                    scope=audited.get('scope'), quality_passed=False)
    # Early-stop segment evidence must never masquerade as twenty whole songs.
    disposition_config = config.get('early_stop_disposition') or {}
    disposition_path = reports/disposition_config.get('report', '__missing__')
    if (disposition_path.is_file() and
            sha256(disposition_path.read_bytes()).hexdigest() == disposition_config.get('sha256')):
        disposition = read_json(disposition_path, {}) or {}
        if (disposition.get('complete') and disposition.get('quality_passed') is False
                and disposition.get('early_stop_triggered') is True
                and disposition.get('checkpoint_sha256') == result.get('checkpoint_sha256')
                and disposition.get('plan_sha256') == result.get('plan_sha256')
                and disposition.get('updates') == result['updates']):
            item['status'] = status = 'early_stopped_quality_failed'
            item['stage_failure'] = dict(reason=disposition['stop_reason'],
                stages=disposition['stage_summary'], report=disposition_config['report'])
            item['whole_song_review'] = 'not_run_after_registered_early_stop'
    snapshot['status'] = 'local_song_'+status
    snapshot['run_id'] = config.get('run_id', result.get('id', snapshot.get('run_id')))
    snapshot['training_complete'] = bool(result.get('complete'))
    snapshot['effective_updates'] = snapshot['step'] = result['updates']
    snapshot['metrics'].update(loss=result.get('latest_train_ce'), grad_norm=result.get('gradient_norm'))
    prior = target.get('prior_song_quality')
    if prior:
        historical = {'metrics': {}}
        local_song_training_status(historical, {'local_song_training': prior}, reports, lambda pid: '')
        item['previous_quality'] = historical.get('local_song_training', {})


def full_event_probe_status(snapshot, target, reports=REPORTS):
    """Display the full-size implementation probe without promoting a quality baseline."""
    from hashlib import sha256
    config = target.get('full_event_probe')
    if not config:
        return
    item = dict(status='evidence_incomplete', model_promotion=False)
    snapshot['full_event_probe'] = item
    review = read_json(reports / config['verification'], {}) or {}
    result_path = reports / config['result']
    plan_path = reports.parent / config['plan']
    expected = review.get('source_sha256', {}).get('reports/' + config['result'])
    if not (review.get('complete') and review.get('passed') and expected
            and result_path.is_file() and plan_path.is_file()
            and sha256(result_path.read_bytes()).hexdigest() == expected):
        return
    result = read_json(result_path, {})
    if not result.get('complete') or result.get('plan_sha256') != sha256(plan_path.read_bytes()).hexdigest():
        return
    item.update(status='passed' if result['passed'] else 'failed',
        parameters=result['parameters'], optimizer_updates=result['optimizer_updates'],
        initial_ce=result['initial']['ce'], final_ce=result['final']['ce'],
        fields=result['final']['fields'], generated_streams=review['generated_streams_identical'],
        generated_events=review['generated_events'], wall_s=result['wall_s'],
        scope='固定的两段训练样本；未验证独立歌曲、参考理解或条件控制',
        verification=config['verification'])


def training_horizon_status(snapshot, target, reports=REPORTS, command_lookup=None):
    """Keep a bounded learning curve and its held-out predictions separate."""
    from hashlib import sha256
    config = target.get('training_horizon')
    if not config:
        return
    result = read_json(reports / config['result'], {}) or {}
    plan = reports.parent / config['plan']
    item = dict(status='evidence_incomplete', model_promotion=False)
    snapshot['training_horizon'] = item
    if not plan.is_file() or result.get('plan_sha256') != sha256(plan.read_bytes()).hexdigest():
        return
    process = {}
    local_probe_status(process, {'local_probe': config}, reports, command_lookup)
    item.update(process['local_probe'])
    item['scope_label'] = '6首训练、5首不同音频验证的有限诊断；无留出低难mania，条件未启用'
    if result.get('complete') and item['status'] in ('passed', 'failed'):
        item['status'] = 'passed' if result.get('diagnostic_horizon_benefit') else 'failed'
    item['checkpoints'] = {}
    for step, point in result.get('checkpoints', {}).items():
        predictions = [(path, expected) for path, expected in point.get('assets_sha256', {}).items()
                       if Path(path).name.startswith('predictions-')]
        if len(predictions) != 1:
            continue
        path, expected = predictions[0]
        file = reports.parent / path
        if file.is_file() and sha256(file.read_bytes()).hexdigest() == expected:
            item['checkpoints'][step] = point['summary']
    item['new_cloud_spend_usd'] = result.get('new_cloud_spend_usd')
    item['gates'] = result.get('gates')


def local_data_status(snapshot, target, reports=REPORTS, command_lookup=None):
    snapshot['training_plan_label'] = target.get('training_plan_label')
    snapshot['current_work_label'] = target.get('current_work_label')
    native_config = target.get('native_content_audit')
    if native_config:
        from hashlib import sha256
        path = reports / native_config['report']
        if path.is_file() and sha256(path.read_bytes()).hexdigest() == native_config['sha256']:
            native = read_json(path, {}) or {}
            if native.get('complete'):
                snapshot['native_content_audit'] = native['summary']
    strata_path = target.get('phira_strata_audit')
    if strata_path:
        strata = read_json(reports / strata_path, {}) or {}
        if strata:
            snapshot['phira_strata'] = dict(summary=strata.get('summary', {}),
                matrix=strata.get('matrices', {}).get('screened_metadata_candidates', {}),
                quantity_gate_waived_for_this_run=strata.get('quantity_gate_waived_for_this_run', False))
    scale_path = target.get('data_scale_audit')
    if scale_path:
        scale = read_json(reports / scale_path, {}) or {}
        if scale:
            snapshot['data_scale'] = dict(formal_training_ready=scale.get('formal_training_ready', False),
                candidates=scale.get('candidate_train_upper_bounds', {}),
                minimums=scale.get('policy', {}).get('minimum_unique_train_charts_per_tier', {}))
    config = target.get('local_data')
    if not config:
        return
    transfer = read_json(reports / config['transfer'], {}) or {}
    processing = read_json(reports / config['processing'], {}) or {}
    audit = read_json(reports / config['audit'], {}) or {}
    pid = config.get('pid')
    if command_lookup is None:
        def command_lookup(pid):
            if not pid:
                return ''
            return subprocess.run(['ps', '-p', str(pid), '-o', 'command='],
                                  capture_output=True, text=True).stdout
    command = command_lookup(pid)
    alive = bool(command and config['script'] in command and config['plan'] in command
                 and f"--phase {config['phase']}" in command)
    status = ('complete' if audit.get('complete') else 'processing' if alive and config['phase'] == 'process'
              else 'downloading' if alive else 'downloaded' if transfer.get('complete')
              else 'process_not_confirmed')
    snapshot['local_data'] = dict(status=status, process_confirmed_live=alive,
        downloaded=sum(x.get('status') == 'downloaded' for x in transfer.get('records', [])),
        failed=sum(x.get('status') == 'failed' for x in transfer.get('records', [])),
        expected_archives=config['expected_archives'], bytes=transfer.get('transferred_bytes', 0),
        processed_items=len(processing.get('items', [])), summary=audit.get('summary'),
        training_started=False, model_promotion=False)


def render_text(r):
    m = r['metrics']
    def num(x, unit=''):
        return '尚无数据' if x is None else f'{x:.2f}{unit}'
    labels = {'infrastructure_interrupted': '数据加载阻塞，训练中断；质量未裁决',
              'failed': '异常停止，待修复', 'experiment_rejected': '本轮质量未通过，模型不晋升', 'evaluation_failed': '训练轮次完成，生成评测失败', 'stopped_by_guard': '守卫停训，待复盘', 'training': '训练中', 'starting': '启动中',
              'evaluating': '评测中', 'review_required': '等待复盘', 'log_stale': '日志停滞，待核实',
              'evaluation_incomplete': '评测已结束，存在未完成或不合法请求',
              'evaluation_observation_error': '评测连接异常，需核实原任务',
              'evaluation_log_stale': '评测日志停滞，需核实原任务',
              'evaluation_stopped_by_user': '评测已按要求停止，分析部分结果并优化执行',
              'waiting_for_log': '等待日志', 'preparing': 'CPU 数据准备', 'probing': '实际数据探针与基线评测',
              'diagnosing': '已有模型的只读诊断中，零训练更新'}
    headline = labels.get(r['status'], r['status'])
    local_probe = r.get('local_probe')
    if local_probe:
        local_labels = {'running': '本机固定批诊断中', 'passed': '本机固定批探针通过',
                        'failed': '本机固定批探针未通过', 'interrupted': '本机探针中断',
                        'process_not_confirmed': '本机探针进程待核实'}
        headline = local_labels[local_probe['status']] + '；最近正式模型未晋升'
    local_data = r.get('local_data')
    if local_data:
        names = {'downloading': '扩充真实谱面数据中', 'processing': '新谱音频、标签与划分审计中',
                 'complete': '首批扩库审计完成', 'downloaded': '首批下载完成，待审计',
                 'process_not_confirmed': '扩库进程待核实'}
        headline = names[local_data['status']] + '；未启动新训练，模型未晋升'
    if r.get('data_scale'):
        headline = '三层训练数据准备中；正式训练尚未就绪，模型未晋升'
    if r.get('user_paused') and r.get('evaluation_review'):
        coverage = r['evaluation_review'].get('coverage', {})
        headline = '训练已暂停；整曲评测结果已取回，等待复盘' if coverage and all(c.get('missing', 1) == 0 for c in coverage.values()) else '训练已暂停；正在补齐检查点评测'
    external = r.get('external_baseline')
    if external and external['status'] == 'complete':
        headline = '外部整曲对照已完成；长按控制失败，正式模型未晋升'
    audio_coordinate = r.get('audio_coordinate_diagnostic')
    if audio_coordinate and audio_coordinate['status'] == 'complete':
        headline = '音频时间坐标诊断完成；未启动新训练，正式模型未晋升'
    paired_state = r.get('paired_state_probe')
    if paired_state:
        headline = {'failed': '新主干配对探针未通过；Phira释放错误仍在，候选已隔离',
                    'passed': '新主干固定批探针通过；正式模型未晋升',
                    'evidence_incomplete': '新主干探针证据待核验；正式模型未晋升'}[paired_state['status']]
    horizon = r.get('training_horizon')
    if horizon:
        headline = {'running': '固定主干学习曲线诊断进行中；正式模型未晋升',
                    'passed': '训练长度诊断改善了小样本留出指标；正式模型未晋升',
                    'failed': '训练长度诊断未通过；保留训练与留出曲线复盘',
                    'interrupted': '训练长度诊断中断；待核实已保存结果',
                    'process_not_confirmed': '训练长度诊断进程待核实；不自动重启',
                    'evidence_incomplete': '训练长度诊断证据待核验'}[horizon['status']]
    if r.get('current_work_label'):
        headline = r['current_work_label']
    lines = ['# BeatFCN 训练监视器', '',
             f"更新时间：{dt.datetime.fromtimestamp(r['observed_at']).astimezone().isoformat(timespec='seconds')}", '',
             f"**{headline}** · `{r.get('run_id') or '未指定运行'}`", '',
             '| 项目 | 最近正式训练记录（历史） |' if local_probe else '| 项目 | 当前值 |', '|---|---|',
             f"| 步数 / 本轮有效更新 | {r.get('step')} / {r.get('effective_updates')} |",
             f"| 训练 loss / 梯度范数 | {num(m.get('loss'))} / {num(m.get('grad_norm'))} |",
             f"| GPU 利用率 | {num(m.get('gpu_util_percent'), '%')} |",
             f"| 最近区间训练核心数 / 秒 | {num(m.get('cores_per_s'))} |",
             f"| 已分配 / 保留显存 | {num(m.get('allocated_gib'), ' GiB')} / {num(m.get('reserved_gib'), ' GiB')} |",
             f"| 歌曲池设备内存 | {num(m.get('pool_device_gib'), ' GiB')} |",
             f"| 数据等待占累计时间 | {num(m.get('data_wait_percent'), '%')} |",
             f"| 距离最后完整日志 | {num(r.get('log_age_s'), ' s')} |", '']
    whole = r.get('local_song_training')
    if whole:
        lines += [f"当前整曲候选（{whole.get('label', 'D68')}；实现探针另列）", '',
                  f"状态：{whole['status']}；有效更新 {whole.get('updates', 0)}；训练CE {num(whole.get('latest_train_ce'))}。", '',
                  '| 更新 | Phira 留出请求 CE | mania 留出请求 CE | 有效 token |', '|---|---:|---:|---:|']
        for point in whole.get('curve', []):
            ce = point.get('mode_val_ce', {})
            lines.append(f"| {point['step']} | {num(ce.get('phira'))} | {num(ce.get('mania_4k'))} | {point.get('valid_tokens')} |")
        if whole.get('stage_failure'):
            failure = whole['stage_failure']
            lines += ['', f"阶段早停：{failure['reason']}。", 
                      '片段审查已触发停止条件；未进行本候选的1000步20首整曲评测。',
                      f"证据：{failure['report']}。", '']
        songs = whole.get('whole_songs', {})
        review_state = '20首时序图已审阅，听音与原生试玩未完成' if whole.get('timeline_review') else '逐曲审阅待完成'
        lines += ['', f"整曲生成：{songs.get('completed', 0)}/{songs.get('expected', 20)}（{songs.get('backend', '待启动')}）；{review_state}。",
                  '上表为 teacher-forced 诊断，不能当作整曲质量或 conditions 响应。授权人类风格续写尚未训练。', '']
        auxiliary = whole.get('auxiliary_supervision')
        if auxiliary:
            lines += [f"附加真实配对监督：{auxiliary.get('auxiliary_pair_updates', 0)}对，"
                      f"{auxiliary.get('auxiliary_prefix_views', 0)}个条件前缀；"
                      f"模式曝光 {auxiliary.get('auxiliary_mode_exposure', {})}。",
                      '附加交互loss与gap只诊断训练目标，不能代替生成条件响应。', '']
        prior = whole.get('previous_quality', {})
        previous = prior.get('review_summary')
        if previous:
            lines += [f"上一份已审整曲结果（{prior.get('label', 'D68')}，不属于当前候选）："
                      f"{previous['completed']}/{previous['songs']}完整，原始音符F1={num(previous['raw_full_note_f1_mean'])}，"
                      f"条件状态={prior.get('condition_response')}，晋升={prior.get('model_promotion')}。", '']
        reviewed = whole.get('review_summary')
        if reviewed:
            lines += ['1000步实际整曲输出（包含不完整尝试，不排除坏例）', '',
                      '| 项目 | 实测 |', '|---|---:|',
                      f"| 完成整曲 | {reviewed['completed']}/{reviewed['songs']} |",
                      f"| 空谱 | {reviewed['empty_songs']} |",
                      f"| 原始完整音符F1均值 | {num(reviewed['raw_full_note_f1_mean'])} |",
                      f"| 模型 / 人谱音符总数 | {reviewed['model_notes']} / {reviewed['human_notes']} |", '',
                      'F1只是与这组人谱的匹配诊断；不等于音乐理解、控制有效、可玩性或模型晋升。', '']
        if whole.get('timeline_review'):
            evidence = whole['timeline_review']
            counts = evidence['aggregate']['model_types']
            lines += [f"实际输出类型：{counts}。结论：未晋升；停止此基线自动续训。",
                      f"逐曲观察：{evidence['report']}；原始产物：{evidence.get('artifact_index') or '未绑定'}。", '']
        if whole.get('condition_audit'):
            lines += ['条件单字段干预（四首独立歌曲，非整曲控制）：', '']
            for name, field in whole['condition_audit']['by_field'].items():
                lines.append(f"- {name}：{field['identical_pairs']}/{field['pairs']}对输出完全相同；在带 {field['in_band_outputs']}/{2*field['pairs']}；哨兵通过={field['sentinel_passed']}。")
            lines += ['', '其余五项未测；人类参考续写未训练。失败后停止扩测，不补造通过项。', '']
        if whole.get('audio_audit'):
            audit = whole['audio_audit']
            lines += [f"音频错配：{audit['identical_pairs']}/{audit['pairs']}对输出完全相同；此项只诊断音频依赖，不等于音乐理解通过。", '']
        if whole.get('error'):
            lines += [whole['error'], '']
    local_curves = r.get('local_multifidelity')
    if local_curves:
        lines += ['D65 历史小面板（不代表当前候选结果）' if whole else 'M3 当前实验（固定批拟合与留出生成分开）', '',
                  '| 实现探针 | 参数 | 更新 | 请求CE | 训练秒 |', '|---|---:|---:|---:|---:|']
        for h in local_curves['health']:
            if h['status'] == 'evidence_incomplete':
                lines += [f"| {h['report']}证据缺失 | — | — | — | — |"]
            else:
                lines += [f"| {h['status']} | {h['parameters']/1e6:.2f}M | {h['updates']} | "
                          f"{num(h['request_mean_ce'])} | {num(h['train_s'])} |"]
        lines += ['', '以上仅为两段训练谱的记忆检查，不能排名模型质量。', '',
                  f"独立歌曲短片段曲线：{local_curves['status']}；更新 {local_curves.get('updates')}；"
                  f"有效token {local_curves.get('valid_tokens')}；请求 {local_curves.get('requests')}。", '',
                  '| 更新 | Phira验证CE | Phira完整音符F1 | Phira完成条数 | mania完整音符F1 |',
                  '|---|---:|---:|---:|---:|']
        for point in local_curves['curve']:
            g = point.get('generation_summary', {})
            ph = g.get('phira', {}); ma = g.get('mania_4k', {})
            lines += [f"| {point['step']} | {num(point['mode_val_ce'].get('phira'))} | "
                      f"{num(ph.get('full_note_f1'))} | {ph.get('complete')}/{ph.get('requests')} | "
                      f"{num(ma.get('full_note_f1'))} |"]
        lines += ['', '训练13首、验证7首的旧诊断子集；生成仅三首Phira和一首mania的8秒片段。'
                  '不完整输出的F1也仅是部分预测，不能视为成功。', '',
                  '尚缺：' + '、'.join(local_curves['missing']) + '。正式质量模型未晋升。', '']
        if local_curves.get('audio_audit', {}).get('complete'):
            lines += ['音频换歌：三条输出均改变，完整F1两降一升；只证实敏感，有效音乐理解仍未证实。', '']
        condition_audit = local_curves.get('condition_audit', {})
        if condition_audit.get('complete'):
            for row in condition_audit['rows']:
                values = [r[row['field']] for r in row['realized']]
                lines += [f"条件 {row['field']}：请求 {row['low']} / {row['high']}，实际 {values}；"
                          f"完整token流相同={row['identical_token_stream']}。", '']
            lines += ['已停止扩测此检查点；输入标签接通尚未形成控制能力。', '']
        if local_curves.get('error') or local_curves.get('early_stop'):
            lines += [str(local_curves.get('error') or local_curves.get('early_stop')), '']
    candidates = r.get('candidate_probes')
    if candidates and candidates['status'] == 'available':
        lines += ['D64 当前候选健康检查（不是整曲 benchmark）', '',
                  '| 主干 | 参数 | 固定批更新 | 请求均值 CE | 精确重建音符 |',
                  '|---|---:|---:|---:|---:|']
        for c in candidates['candidates']:
            lines.append(f"| {c['name']} | {c['parameters']/1e6:.2f}M | {c['updates']} | "
                         f"{c['request_mean_ce']:.8f} | {c['exact_events']} |")
        transport = candidates['transport']
        lines += ['', candidates['scope'], '',
                  f"优化传输：输入/标签一致，logits最大差 {transport['logits_max_error']:.1e}；"
                  f"短探针总用时 {transport['wall_s']:.2f} 秒，已保存 CPU/CUDA trace。"
                  ' 尚未完成全数据换曲与480秒守卫验证。', '',
                  '正式训练更新：0；当前没有质量胜出者。', '']
    elif candidates:
        lines += ['D64候选报告身份不符，暂不展示分数。', '']
    full_event = r.get('full_event_probe')
    if full_event:
        if full_event['status'] == 'evidence_incomplete':
            lines += ['完整事件新主干：探针证据缺失或身份不符，暂不展示分数。', '']
        else:
            label = '历史完整事件实现探针' if candidates else '最新完整事件主干'
            lines += [f"{label}：**{full_event['parameters'] / 1e6:.2f}M 参数，"
                      f"{full_event['optimizer_updates']} 次固定批更新**；实现探针"
                      + ('通过。' if full_event['status'] == 'passed' else '未通过。'), '',
                      f"监督 CE：{full_event['initial_ce']:.6f} → {full_event['final_ce']:.8f}；"
                      f"模型运行 {full_event['wall_s']:.1f} 秒。独立检查点回放复现 "
                      f"{full_event['generated_streams']} 段、{full_event['generated_events']} 个音符。", '',
                      full_event['scope'] + '；不是新整曲 benchmark，诊断权重不晋升。', '',
                      '| 输出字段 | 有效监督数 | 固定批准确率 |', '|---|---:|---:|']
            labels = dict(group='继续/结束', head_block='起点粗区间', head_cell='起点帧',
                          head_ms='起点毫秒', mania='四键联合动作', member='组内成员',
                          kind='Phira类型', tail_block='长按结束粗区间',
                          tail_cell='长按结束帧', tail_ms='长按结束毫秒')
            for field, stats in full_event['fields'].items():
                lines.append(f"| {labels.get(field, field)} | {stats['count']} | {stats['accuracy']:.1%} |")
            lines += ['']
    if local_data:
        lines += [f"扩库：下载 **{local_data['downloaded']}/{local_data['expected_archives']} 个谱包**，"
                  f"失败 {local_data['failed']}；已传输 {local_data['bytes'] / 2**20:.1f} MiB，"
                  f"已处理 {local_data['processed_items']} 个包。低难度与质量证据分别统计。", '',
                  '新增数据尚未计作训练曝光；原模型指标继续保留，数据量增长不代表能力提升。', '']
        for mode, row in (local_data.get('summary') or {}).items():
            lines += [f"{mode}：技术检查通过 {row['technical_pass']} 张，训练候选 {row['train_candidates']} 张；"
                      f"密度分布 {row['density_buckets']}。", '']
    if r.get('data_scale'):
        scale = r['data_scale']
        waiver = r.get('phira_strata', {}).get('quantity_gate_waived_for_this_run')
        lines += [('原三层数量配额本轮已由用户豁免；下表保留旧候选上界与真实缺额，不代表新准入。'
                   if waiver else '三层配额：每层 mania 1,500 张、Phira 700 张。下表为候选上界，尚待内容去重、难度复核及特征验收。'), '',
                  '| 模式 / 难度层 | 已有与新增训练候选 | 至少缺口 |', '|---|---:|---:|']
        for key, row in scale['candidates'].items():
            if not key.endswith('/unknown'):
                lines.append(f"| {key} | {row['candidate_sum_before_content_dedup']} | {row['known_minimum_shortfall']} |")
        plan_label = r.get('training_plan_label') or '保留3500作对照；按最新实验复盘选择下一候选。数据/探针/预算达标后再正式训练，每1000有效更新复盘。'
        lines += ['', f'训练方案：{plan_label}', '']
    if r.get('phira_strata'):
        strata = r['phira_strata']
        s = strata['summary']
        screen = s.get('screen_status_within_previous', {})
        lines += [f"Phira 元数据清洗：原候选 {s.get('previous_metadata_candidates')} 张，"
                  f"排除 {screen.get('excluded', 0)}、待审隔离 {screen.get('review_required', 0)}；"
                  f"筛查通过且可分层 {s.get('screened_metadata_candidates')}。这是候选数，不是已核验训练谱。", '',
                  '| 标签 | (0,4) | [4,8) | [8,11) | [11,13) | [13,15) | [15,16) | ≥16 |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for tag in ('EZ', 'HD', 'IN', 'AT'):
            counts = strata['matrix'].get(tag, {})
            lines.append('| ' + tag + ' | ' + ' | '.join(str(counts.get(band, 0))
                for band in ('0-4', '4-8', '8-11', '11-13', '13-15', '15-16', '16+')) + ' |')
        lines += ['', '元数据按 ID 连接不证明谱包版本一致；原谱玩法、时钟、去重与特征审核继续执行。', '']
    native = r.get('native_content_audit')
    if native:
        status = native['status_counts']
        lines += [f"本地原包内容检查：{native['archive_files']} 份文件、"
                  f"{native['unique_archive_bytes']} 份不同原包；"
                  f"{native['object_counts']['canonical_fakes']} 个显式假音符与监督目标分开。"
                  f"{status.get('native_semantics_review_required', 0)} 份原生字段待审、"
                  f"{status.get('parse_or_accounting_quarantine', 0)} 份解析问题、"
                  f"{status.get('unsupported_or_invalid_package', 0)} 份格式暂不支持。", '',
                  '静态检查无异常不等于完整可玩性通过；保留正常演出和艺术drag，不擅自删点。', '']
    if local_probe:
        fixed = local_probe.get('latest_fixed_batch') or {}
        fixed_loss = '尚无数据' if fixed.get('loss') is None else f"{fixed['loss']:.6f}"
        lines += ['上表是最近正式模型的历史训练记录，不代表当前 CPU 探针的步数、速度或资源占用。', '',
            f"本机诊断：**{local_probe['optimizer_updates']}/{local_probe['expected_updates']} 次更新**；"
            f"固定批监督 loss {fixed_loss}；峰值进程内存 {num(local_probe.get('peak_rss_gib'), ' GiB')}。", '',
            local_probe.get('scope_label', '固定训练片段的记忆测试') + '；探针通过也不等于独立泛化或制谱能力通过。拟合权重不晋升。', '']
        binding = local_probe.get('binding_gate')
        if binding is not None:
            if binding.get('kind') == 'generation_swap':
                lines += [f"记忆配对条件移除/交换：**{len(binding['passed_fields'])}/{binding['total_fields']} 达标**。仅覆盖本批指定字段，不是独立条件控制 benchmark。", '']
            else:
                lines += [f"固定批条件评分间隔：**{len(binding['passed_fields'])}/{binding['total_fields']} 达标**。这是正确/错误条件的似然判别，不是生成控制通过数。", '']
        generation = local_probe.get('generation_gate')
        if generation is not None:
            scores = '；'.join(f"{row['chart_id']}={row['full_note_f1']:.4f}" for row in generation['cores'])
            lines += [f"记忆样本自由生成完整音符 F1：{scores}。独立泛化与整曲实用性仍未验收。", '']
        generation_summary = local_probe.get('generation_summary')
        if generation_summary:
            lines += ['最新本机探针的自由生成结果（训练片段，非整曲 benchmark）：', '',
                      '| 模式 | 首轮完整 F1 | 第四轮完整 F1 | 第四轮非 tap：预测 / 目标 |',
                      '|---|---:|---:|---|']
            for mode, row in generation_summary.items():
                first = num(row.get('first_f1'))
                complete = bool(row.get('all_four_rounds')) and row.get('final_f1') is not None
                final = num(row['final_f1']) if complete else '未完成'
                counts = '未完成'
                if complete and row.get('per_type_scope') == 'fourth_round':
                    counts = '；'.join(f"{kind} {v['pred']}/{v['true']}"
                        for kind, v in row.get('per_type', {}).items() if kind != 'tap' and (v['true'] or v['pred'])) or '无适用目标'
                lines.append(f'| {mode} | {first} | {final} | {counts} |')
            lines += ['', '音符数量不等于匹配成功；类型缺失和未完成轮次均保留，不能用最后可用轮替代第四轮。', '']
    elif r.get('user_paused'):
        lines += ['已按用户要求暂停后续训练。' + ('当前另行授权的工作仅为既有检查点评测。' if r.get('evaluation_review') else '等待方法复盘。'), '']
        lines += ['上面的 GPU、显存与速度来自最后训练日志，不代表当前资源占用。', '']
    elif r.get('historical_training_metrics'):
        lines += [f"上方是已有模型的最后训练记录；GPU、显存与速度不代表当前资源占用。{r.get('round_label','R04')} 尚未完成正式训练。", '']
    elif r.get('training_complete'):
        lines += ['本轮训练已完成。上面的 GPU、显存与速度来自最后训练日志，不代表当前评测资源占用。', '']
    if horizon:
        lines += ['', f"训练长度诊断：{horizon.get('scope_label', '证据待核验')}。", '',
                  f"有效更新 {horizon.get('optimizer_updates', 0)} / {horizon.get('expected_updates', 1000)}；"
                  f"状态 {horizon['status']}。", '',
                  '| 步数 | 训练完整F1 | 留出完整F1 | 留出起音F1 | 训练loss | 留出loss |',
                  '|---|---:|---:|---:|---:|---:|']
        for step, point in sorted(horizon.get('checkpoints', {}).items(), key=lambda pair: int(pair[0])):
            train, held = point['train']['all'], point['heldout']['all']
            lines.append(f"| {step} | {train['full_note_f1']:.1%} | {held['full_note_f1']:.1%} | "
                         f"{held['timing_point_f1']:.1%} | {train['teacher_loss']:.4f} | {held['teacher_loss']:.4f} |")
        lines += ['', '这些是原始自由生成结果；全部检查点和音符时间表保留，未作吸附、补尾或删音符。', '']
    if paired_state:
        lines += ['', '最新主干探针（固定训练片段，不能视作新歌benchmark）：', '']
        if paired_state['status'] == 'evidence_incomplete':
            lines += ['结果缺失或摘要不匹配，暂不展示模型指标。', '']
        else:
            lines += [f"两组共 {paired_state['optimizer_updates']} 次有效更新；本机 {paired_state['wall_s']:.2f} 秒，"
                      f"峰值内存 {paired_state['peak_rss_gib']:.3f} GiB，新增云费用 ${paired_state['new_cloud_spend_usd']:.2f}。", '',
                      '| 版本 | 监督loss | mania完整F1 | Phira完整F1 | Phira hold匹配 | Phira drag匹配 |',
                      '|---|---:|---:|---:|---:|---:|']
            for arm, row in paired_state['summary'].items():
                lines.append(f"| {arm} | {row['loss']:.4f} | {row['mania_f1']:.1%} | {row['phira_f1']:.1%} | "
                             f"{row['phira_hold_matched']}/{row['phira_hold_true']} | {row['phira_drag_matched']}/{row['phira_drag_true']} |")
            lines += ['', 'A使用完整已生成动作历史，B再加入显式占用摘要。两段共126个真实音符；'
                      '一条Phira长按被提前416.667ms释放，B还漏掉3个drag。近零拟合不等于全部类型或独立泛化通过。', '']
            if paired_state.get('interpretation_note'):
                lines += [paired_state['interpretation_note'], '']
    if audio_coordinate:
        lines += ['', '最新表示诊断（不是生成质量提升）：', '']
        if audio_coordinate['status'] != 'complete':
            lines += ['证据不完整或摘要不匹配，暂不展示坐标覆盖结果。', '']
        else:
            lines += [f"{audio_coordinate['charts']} 首真实歌曲；原始及既有修正时钟分别统计。下表是修正时钟下混合细分网格的20ms覆盖。", '',
                      '| 模式 | 独立起音组 | 长按尾点 | 连续 drag | 起音组相对移相对照超额 |',
                      '|---|---:|---:|---:|---:|']
            for mode, row in audio_coordinate['summary'].items():
                burst = '无适用目标' if row['drag_burst_20ms'] is None else f"{row['drag_burst_20ms'] * 100:.1f}%"
                lines.append(f"| {mode} | {row['heads_20ms'] * 100:.1f}% | {row['hold_ends_20ms'] * 100:.1f}% | "
                             f"{burst} | {row['heads_phase_excess_20ms'] * 100:.1f}pp |")
            lines += ['', '粗网格未通过预注册筛选；更密网格的移相对照也接近100%，不能将高覆盖当作节拍理解。'
                      '连续时间出口保留，原谱和模型均未修改。', '']
    if external:
        lines += ['', '最新外部预训练模型对照（零训练更新）：', '', external['scope_label'], '']
        if external['status'] != 'complete':
            lines += ['证据不完整或摘要不匹配，暂不展示结果。', '']
        else:
            lines += ['| 版本 | 音符 / 长按 | 音频20ms锚定率 | 最大内部起音间隔 | 每秒峰值 |',
                      '|---|---:|---:|---:|---:|']
            for key, label in [('previous_model', '旧模型'), ('external_2p5', '外部请求2.5'),
                               ('external_4p5', '外部请求4.5'), ('human_reference', '人谱参照')]:
                row = external['charts'][key]
                lines.append(f"| {label} | {row['notes']} / {row['holds']} | "
                    f"{100 * row['music']['anchor_20ms_precision']:.1f}% | "
                    f"{row['max_internal_head_gap_s']:.3f}s | {row['peak_notes_per_fixed_second']} |")
            lines += ['', '较高请求的长按占用未达到系统自己的目标。一次整曲对照与训练片段探针分别报告；'
                      '外部推理不代表BeatFCN训练提升，也没有证明Phira或全部条件有效。', '']
    if r.get('current_work'):
        work = r['current_work']
        work_labels = {'metadata_prepare': 'CPU 预生成训练批次', 'metadata_compress': 'CPU 无损压缩训练批次', 'prepared': '训练批次已准备完成',
                       'live_input_probe': '验证持续供数与复制路径', 'transport_probe': '固定输入短对照',
                       'diagnosed': '短对照完成，分析中', 'train': '训练中', 'benchmark': '生成 benchmark 中',
                       'needs_diagnosis': '等待排障', 'awaiting_review': '等待能力复盘',
                       'rejected': '本轮已否决；下一步转向架构与生成路径诊断',
                       'diagnostic': '定位选时、前文与动作预测错误'}
        work_prefix = '最近正式轮次状态' if local_probe else '原训练应用最后状态' if r.get('evaluation_review') else '当前工作'
        lines += [f"{work_prefix}：**{work_labels.get(work['status'], work['status'])}** · `{work['round']}`。" +
                  (f"已准备 {work['prepared_batches']} 个微批次。" if work.get('prepared_batches') is not None else ''), '']
    timing = m.get('input_timing', {})
    if timing:
        lines += [f"最近 {timing.get('batches', '未知')} 个微批次：加载等待 {num(timing.get('loader_wait_s'), ' s')}；内存暂存 {num(timing.get('pin_stage_s'), ' s')}；音频装配 {num(timing.get('attach_s'), ' s')}。", '']
    if r.get('budget'):
        b = r['budget']
        prefix = '原训练账本（历史快照）：' if r.get('evaluation_review') else ''
        lines += [prefix + f"总预算 ${b['approved_usd']:.2f}；阶段保守预留 ${b['reserved_usd']:.2f}；已结束阶段估算 ${b['estimated_completed_usd']:.3f}。", '', b['note'], '']
    review_budget = r.get('evaluation_review_budget') or {}
    if review_budget:
        upper = review_budget['prior_cost_upper_usd'] + review_budget['other_allowance_usd'] + sum(s['reserved_usd'] for s in review_budget['stages'].values())
        lines += [f"本次复盘账本：原总上限 ${review_budget['approved_total_usd']:.2f}；含历史费用及本次评测预留的保守上界 ${upper:.3f}。预留不等于实付。", '']
    if r.get('rhythm_budget'):
        b = r['rhythm_budget']
        lines += [f"当前累计预算上限 ${b['approved_total_usd']:.2f}；历史已计入上界 ${b['historical_accounted_upper_usd']:.3f}；"
                  f"含后续评测预留的累计保守上界 ${b['campaign_committed_upper_usd']:.3f}。预留不等于实付。", '']
    if r.get('paired_runs'):
        lines += ['同数据训练对照（1000 次有效更新后复盘）：', '',
                  '| 训练组 | 步数 | 有效更新 | 最新固定验证 loss | 状态 |', '|---|---:|---:|---:|---|']
        for paired in r['paired_runs']:
            val = paired.get('model_progress', {}).get('latest_validation') or {}
            lines.append(f"| {paired['run_id']} | {paired.get('step')} | {paired.get('effective_updates')} | "
                         f"{num(val.get('total'))} | {labels.get(paired['status'], paired['status'])} |")
        lines += ['']
    if r.get('condition_diagnostic'):
        lines += ['已完成的历史模型密度诊断（局部核心，尚非整曲条件验收）：', '',
                  '| 检查点 / 模式 | 低密度请求 | 中密度请求 | 高密度请求 | 单调响应 |', '|---|---:|---:|---:|---:|']
        for label, item in r['condition_diagnostic'].get('controls', {}).items():
            means = [num(v.get('mean')) for v in item['achieved_by_band'].values()]
            lines.append(f"| {label} | " + ' | '.join(means) + f" | {item.get('positive_monotonic_songs', '未测')}/{item.get('paired_songs', '未测')} |")
        lines += ['新训练组必须重新做同条件生成评测；上述结果不能当作新模型指标。', '']
    progress = r.get('model_progress', {})
    val, base = progress.get('latest_validation') or {}, progress.get('baseline') or {}
    lines += ['## 模型学习进展', '',
              f"训练指标：第 {r.get('step', '未知')} 步最近一个日志区间；固定验证：第 {val.get('step', '尚未评测')} 步。",
              '训练准确率受请求难度和模式配比影响；下面的变化只比较同一固定验证集。', '']
    if val:
        delta = val.get('total', 0) - base['total'] if 'total' in val and 'total' in base else None
        lines += [f"验证 loss：**{num(val.get('total'))}**；相对第 {base.get('step', '未知')} 步：{num(delta)}。", '']
    heads = [('time', '起音时间格'), ('offset', '毫秒偏移'), ('cont', '同组继续'), ('type', '音符类型'),
             ('lane_choice', '键道（有选择空间）'), ('lane', '键道（全部）'),
             ('lane_first_free', '首个空闲键道启发式'), ('lane_class_0', '键道 1'), ('lane_class_1', '键道 2'),
             ('lane_class_2', '键道 3'), ('lane_class_3', '键道 4'),
             ('end_block', '长按结束块'), ('end_cell', '长按结束格'), ('end_off', '长按结束毫秒')]
    def accuracy(record, head):
        entry = record.get(head, {})
        return entry.get('acc') if entry.get('n', 0) > 0 else None
    def percent(value):
        return '未测' if value is None else f'{100 * value:.2f}%'
    lines += ['| 预测头 | 训练准确率 | 固定验证准确率 | 相对本轮起点 | 验证目标数 |', '|---|---:|---:|---:|---:|']
    regressions = []
    for head, label in heads:
        train_acc, val_acc, base_acc = (accuracy(record, head) for record in (m.get('fields', {}), val, base))
        change = val_acc - base_acc if val_acc is not None and base_acc is not None else None
        diff = '不可比较' if change is None else f'{100 * change:+.2f} 个百分点'
        if change is not None and change < 0 and head not in ('lane_first_free',):
            regressions.append(f'{label} {100 * change:+.2f}')
        lines.append(f"| {label} | {percent(train_acc)} | {percent(val_acc)} | {diff} | {val.get(head, {}).get('n', '—')} |")
    lines += ['', '验证点估计下降：' + ('；'.join(regressions) + ' 个百分点。' if regressions else '当前可比较项未见下降，或尚无可比较数据。'),
              '这些是教师强制预测指标，不能单独证明整曲谱面质量；小幅变化也未做显著性判定。', '']
    history = progress.get('history', [])
    if history:
        lines += ['| 验证步数 | loss | 起音格准确率 | 毫秒准确率 | 键道选择准确率 |', '|---:|---:|---:|---:|---:|']
        for point in history[-8:]:
            lines.append(f"| {point['step']} | {num(point.get('total'))} | {percent(accuracy(point, 'time'))} | {percent(accuracy(point, 'offset'))} | {percent(accuracy(point, 'lane_choice'))} |")
        lines.append('')
    lines += ['## 实际生成 benchmark', '']
    benchmark = r.get('benchmark') or {}
    evaluation = r.get('condition_evaluation') or {}
    reviewed_generation = any((((r.get(key) or {}).get('review') or {}).get('results') or {}).get('candidate')
                              for key in ('r04', 'r05', 'r06', 'r07'))
    if r.get('r07'):
        label=r.get('round_label','R07')
        current=r['r07'];probe=current.get('preflight') or {};review=current.get('review') or {}
        lines+=[f'{label}：目标、条件、模型和损失固定，仅移除安全完整窗口的外部真人上下文。','']
        if current.get('infrastructure', {}).get('status') == 'infrastructure_interrupted':
            lines += [f"本轮完成 {current['infrastructure']['effective_updates']}/1000 次有效更新后被原守卫停止；"
                      '未运行生成评测，不能裁决模型质量。上下文改动已归档，入口关闭。', '']
        if probe.get('history'):
            lines += [f"真实固定 batch：{probe.get('optimizer_updates')} 次更新，loss {probe['history'][0]['loss']:.4f} → "
                f"{probe['history'][-1]['loss']:.4f}；前置关{'通过' if probe.get('probe_passed') else '未通过'}。拟合权重未保存。", '']
        row=review.get('results',{}).get('candidate')
        if row:
            lines += [f"{label} 实际生成：完整音符 F1 {percent(row['macro_full_note_f1'])}，时间点 F1 "
                f"{percent(row['macro_timing_point_f1'])}，首点 ±20ms {row['first_onset_matches']}/8；"
                f"条件 {row['passed_fields']}/13，空窗 {row['empty_windows']}/8。",'']
            if 'short_window_gate_passed' in review.get('comparison',{}):
                lines += [('研究保留关通过，仍非产品验收。' if review['comparison']['short_window_gate_passed']
                           else f'{label} 不通过，不晋升该模型。'),'']
            if current.get('rollback',{}).get('source_restored_exactly'):
                lines += ['本轮上下文改动已撤回，训练入口已关闭；以上是被否决检查点的实际结果。','']
        else:
            lines += [f'{label} 尚无训练后生成 benchmark；下列 R06 是已否决的历史模型，不能当作 {label} 指标。','']
    if r.get('r06'):
        current=r['r06'];probe=current.get('preflight') or {};review=current.get('review') or {}
        lines+=['R06：数据、损失和解码固定，只增加按时间格顺序学习的风险残差。','']
        if probe.get('history'):
            lines += [f"真实固定 batch：{probe.get('optimizer_updates')} 次更新，loss {probe['history'][0]['loss']:.4f} → "
                f"{probe['history'][-1]['loss']:.4f}；前置关{'通过' if probe.get('probe_passed') else '未通过'}。拟合权重未保存。", '']
        row=review.get('results',{}).get('candidate')
        if row:
            lines += [f"R06 实际生成：完整音符 F1 {percent(row['macro_full_note_f1'])}，时间点 F1 "
                f"{percent(row['macro_timing_point_f1'])}，首点 ±20ms {row['first_onset_matches']}/8；"
                f"条件 {row['passed_fields']}/13，空窗 {row['empty_windows']}/8。",'']
            if 'short_window_gate_passed' in review.get('comparison',{}):
                lines += [('研究保留关通过，仍非产品验收。' if review['comparison']['short_window_gate_passed']
                           else 'R06 不通过，不晋升该模型。'),'']
            if current.get('rollback',{}).get('source_restored_exactly'):
                lines += ['本轮架构改动已撤回，训练入口已关闭；以上是被否决检查点的实际结果。','']
        else:
            lines += ['R06 尚无训练后生成 benchmark；下列 R05 是已否决的历史模型，不能当作 R06 指标。','']
    if r.get('r05'):
        current=r['r05'];probe=current.get('preflight') or {};review=current.get('review') or {}
        lines+=['R05：数据与全条件标签对照相同，仅加入直接条件残差；默认关闭，零初始化。','']
        if probe.get('history'):
            lines += [f"真实固定 batch：{probe.get('optimizer_updates')} 次更新，loss {probe['history'][0]['loss']:.4f} → "
                f"{probe['history'][-1]['loss']:.4f}；前置关{'通过' if probe.get('probe_passed') else '未通过'}。拟合权重未保存。", '']
        row=review.get('results',{}).get('candidate')
        if row:
            lines += [f"R05 实际生成：完整音符 F1 {percent(row['macro_full_note_f1'])}，时间点 F1 "
                f"{percent(row['macro_timing_point_f1'])}，条件 {row['passed_fields']}/13；空窗 {row['empty_windows']}/8。",'']
            if 'short_window_gate_passed' in review.get('comparison',{}):
                lines += [('研究保留关通过，仍非产品验收。' if review['comparison']['short_window_gate_passed']
                           else 'R05 不通过，不晋升该模型。'),'']
        else:
            lines += ['R05 尚无训练后生成 benchmark；下列 R04 是已否决的历史模型，不能当作 R05 指标。','']
    if r.get('r04'):
        current = r['r04']
        lines += ['R04 只改变 time/type 头中初始与后续决策的损失聚合。新旧 total loss 口径不同；质量以自由生成指标比较。', '']
        probe = current.get('preflight') or {}
        if probe.get('history'):
            lines += [f"真实固定 batch：{probe.get('optimizer_updates')} 次更新，loss {probe['history'][0]['loss']:.4f} → "
                f"{probe['history'][-1]['loss']:.4f}；前置关{'通过' if probe.get('probe_passed') else '未通过'}。拟合权重未保存。", '']
        review = current.get('review') or {}
        row = review.get('results', {}).get('candidate')
        if row:
            lines += [f"R04 实际生成：完整音符 F1 {percent(row['macro_full_note_f1'])}，时间点 F1 "
                      f"{percent(row['macro_timing_point_f1'])}，条件 {row['passed_fields']}/13。", '']
            lines += [f"首起音 ±20ms：{row['first_onset_matches']}/8；空窗：{row['empty_windows']}/8；"
                      f"总体音符数量比：{row['aggregate_count_ratio']:.3f}。", '']
            verdict = review.get('comparison', {})
            if 'short_window_gate_passed' in verdict:
                lines += ['短窗验收：' + ('通过，仍需整曲及制谱实用性验证。' if verdict['short_window_gate_passed']
                                          else '不通过，不晋升该模型。'), '']
        else:
            lines += ['R04 尚无训练后生成 benchmark；下表仍为此前 R03 的失败对照。', '']
    if r.get('r03'):
        review = r['r03'].get('review') or {}
        lines += ['R03：完整窗口全部条件标签；只展示同一协议下已取回的结果。', '']
        if review.get('results'):
            lines += ['| 模型 | 通过条件组合 | 完整音符 F1 | 时间点 F1 | 参考编辑节省代理 |',
                      '|---|---:|---:|---:|---:|']
            for name, row in review['results'].items():
                lines.append(f"| {name} | {row['passed_fields']}/13 | {percent(row['macro_full_note_f1'])} | "
                             f"{percent(row['macro_timing_point_f1'])} | {percent(row['macro_reference_edit_saved_fraction'])} |")
            lines += ['', '这是固定短窗诊断，未通过整曲和人类实用性验收。未出现候选行表示尚未评估候选，不能把基线当新模型指标。', '']
        else:
            lines += ['修复后基线尚未完成回收；暂无 R03 候选生成指标。', '']
        probe = r['r03'].get('preflight') or {}
        if probe.get('history'):
            lines += [f"固定 batch 探针：{probe.get('optimizer_updates')} 步，loss {probe['history'][0]['loss']:.4f} → "
                      f"{probe['history'][-1]['loss']:.4f}；门槛{'通过' if probe.get('probe_passed') else '未通过'}。该拟合权重不保存、不晋升。", '']
    elif evaluation:
        ev_progress, result, summary = (evaluation.get(k) or {} for k in ('progress', 'result', 'summary'))
        if ev_progress:
            lines += [f"新一轮已写入 {ev_progress['records']}/{ev_progress['expected']} 条请求记录；写入数量包括失败，不等于通过数量。", '']
        for label, coverage in result.get('coverage', {}).items():
            lines += [f"`{label}`：合法完成 {coverage['complete_legal']}/{coverage['requested']}。"]
        lines += ['']
        if summary:
            lines += [f"已取回 {summary['received']}/{summary['expected']} 条记录；"
                      + ('全部请求合法完成，仍需审查质量。' if summary['complete'] else
                         f"其中已记录失败 {summary.get('failed_recorded', '未知')} 条，未形成记录 {summary.get('not_recorded', summary['expected'] - summary['received'])} 条；不能作完整验收。"), '',
                      '| 模型 / 模式 / 请求 | 合法完成 / 请求 | 音符 F1 ±20 ms | 音符数量比 | 时间点编辑节省代理 | 完整音符编辑节省代理 |',
                      '|---|---:|---:|---:|---:|---:|']
            for label, cell in summary.get('cells', {}).items():
                editing = cell['20ms']
                lines.append(f"| {label} | {cell['complete_legal']}/{cell['requested']} | {percent(cell['note_f1_20ms']['mean'])} | "
                             f"{num(cell['count_ratio']['mean'])} | {percent(editing['timing_point']['mean'])} | {percent(editing['full_note']['mean'])} |")
            lines += ['', '编辑节省是相对人谱重建的计算代理，可能惩罚合理的不同写法；不是实测人工工时。', '',
                      '| 模型 / 模式 / 条件作用范围 | 有效歌曲 / 请求歌曲 | 低 / 中 / 高实际密度 | 正向单调歌曲 | 在带请求 |',
                      '|---|---:|---:|---:|---:|']
            for label, cell in summary.get('controls', {}).items():
                means = ' / '.join(num(v['mean']) for v in cell['achieved_by_band'].values())
                lines.append(f"| {label} | {cell['complete_feasible_sweeps']}/{cell['requested_songs']} | {means} | "
                             f"{cell['positive_monotonic_songs']}/{cell['complete_feasible_sweeps']} | {cell['in_band']}/{cell['in_band_denominator']} |")
            lines += ['', '均值只包含合法完成请求；条件统计排除已被锁定音符违反的密度要求，并保留请求歌曲分母。',
                      '配对差值、歌曲级区间与音频基线对照见 condition_v1_benchmark_summary.json。', '']
        else:
            lines += ['新模型生成指标尚未汇总；固定验证 loss 不能替代本轮 benchmark。', '']
    elif not benchmark:
        if reviewed_generation:
            lines += ['上述为已完成的短窗生成诊断；整曲制谱实用性尚未验收。', '']
        else:
            lines += ['原训练应用当时未完成生成评测；当前补跑结果见下方“3500 / 4500 整曲复盘”。' if r.get('evaluation_review') else '**尚无本轮生成评测结果，模型能力提升尚未验收。**', '']
    else:
        lines += [f"覆盖状态：{'已完成全部请求' if benchmark.get('complete_for_review') else '部分完成，不能作完整验收'}。", '']
        for path in benchmark.get('checkpoints', []):
            coverage = benchmark.get('coverage', {}).get(path, {})
            records = benchmark.get('records', {}).get(path, [])
            completed = coverage.get('completed', sum(bool(x.get('complete')) for x in records))
            legal = coverage.get('legal_completed', sum(bool(x.get('complete')) and not x.get('legality') for x in records))
            requested = coverage.get('requested', benchmark.get('expected_per_checkpoint', '未知'))
            lines += [f"检查点 `{path}`：完成 {completed}/{requested}，合法完成 {legal}/{requested}。", '',
                      '| 模式 / 请求 / 分层 | 完成 / 请求 | 音频锚定率 ±20 ms | 强起音召回 | 密度与起音强度相关 | IOI 距离 |',
                      '|---|---:|---:|---:|---:|---:|']
            for cell in benchmark.get('scorecards', {}).get(path, {}).get('cells', []):
                metrics = cell.get('metrics', {})
                mean = lambda key: metrics.get(key, {}).get('generated_mean')
                distance = cell.get('reference_distances', {}).get('ioi_js_bits', {}).get('mean')
                lines.append(f"| {cell['mode']} / {cell['request_kind']} / {cell['stratum']} | {cell['completed']}/{cell['requests']} | {percent(mean('anchor_20ms_precision'))} | {percent(mean('anchor_20ms_strong_recall'))} | {num(mean('density_flux_spearman'))} | {num(distance)} |")
            lines.append('')
    review = r.get('evaluation_review') or {}
    if review:
        coverage = review.get('coverage', {})
        all_attempted = bool(coverage) and all(c.get('missing', 1) == 0 for c in coverage.values())
        lines += ['## 3500 / 4500 整曲复盘', '', f"本地结果汇总时间：{review.get('summarized_at', '未记录')}。", '',
                  '全部计划请求已尝试；失败另计。' if all_attempted else '评测进行中；以下只代表已取回样本，不能作完整验收。', '',
                  '| 检查点 | 已尝试 / 计划 | 完成 | 未执行 |', '|---|---:|---:|---:|']
        for ck, c in coverage.items():
            lines.append(f"| {ck} | {c['attempted']}/{c['expected']} | {c['completed']} | {c['missing']} |")
        lines += ['', '| 模式 / 请求 | 3500 音符 F1 ±20 ms | 4500 音符 F1 ±20 ms | 完成数 3500 / 4500 |', '|---|---:|---:|---:|']
        for mode in ('mania_4k', 'phira'):
            for kind in ('from_scratch', 'middle_holes', 'reference'):
                a, b = [review.get('groups', {}).get(f'{ck}/baseline/{mode}/{kind}', {}) for ck in ('3500', '4500')]
                av, bv = [g.get('note_f1_20ms', {}).get('mean_complete_only') for g in (a, b)]
                lines.append(f"| {mode} / {kind} | {percent(av)} | {percent(bv)} | {a.get('completed', 0)} / {b.get('completed', 0)} |")
        lines += ['', '上表均值仅包括完成请求；配对变化、音频错配、历史移除和参考交换见 capability_review_summary.json。',
                  'Phira 空间和连续 drag 手势尚未验收。控制量响应与人评校准也不能由本次输入诊断代替。', '']
    elif not evaluation:
        lines += [('条件响应以各模型上报结果为准；音频/参考交换和人评校准仍需另行验收。'
                   if reviewed_generation else '条件响应、音频/参考交换和人评校准：尚未完成；不计为通过。'), '']
    else:
        lines += ['本轮针对密度条件；其他七项数值条件、参考交换与人评校准不能由这些结果代替。', '']
    if r.get('evaluation_review') and r['alerts']:
        lines += ['原训练应用保留的告警：', '']
    lines += [f'- {a}' for a in r['alerts']]
    if r.get('cloud_error'):
        error = r['cloud_error'].strip().splitlines()[-1]
        lines += ['', f"云端读取暂不可用：{error}。以上为最后已知指标；本地日志仍在监视。"]
    lines += ['', '监视器只读日志；缺失或过期数据不会被显示成正常。']
    return '\n'.join(lines) + '\n'


def render(r):
    STATUS.write_text(render_text(r))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--pull', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--local-only', action='store_true')
    parser.add_argument('--interval', type=float, default=5)
    parser.add_argument('--cloud-every', type=float, default=60)
    parser.add_argument('--stale-seconds', type=float, default=180)
    parser.add_argument('--max-hours', type=float, default=6)
    args = parser.parse_args()
    REPORTS.mkdir(exist_ok=True)
    if args.pull:
        pull_cloud()
        return
    if min(args.interval, args.cloud_every, args.stale_seconds, args.max_hours) <= 0:
        parser.error('intervals and runtime must be positive')
    lock = (REPORTS / 'training_monitor.lock').open('w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('已有 monitor 在运行')
    lock.write(str(os.getpid()))
    lock.flush()
    mon, offsets, pending = Monitor(), {}, {}
    emitted = {json.loads(l)['key'] for l in ALERTS.read_text().splitlines()} if ALERTS.exists() else set()
    started, next_cloud, cloud_error, last_print = time.time(), 0., None, None
    while True:
        now = time.time()
        target = read_json(TARGET, {}) or {}
        consoles = set(REPORTS.glob('evolution*console.log'))
        if target.get('console_file') and (REPORTS / target['console_file']).is_file():
            consoles.add(REPORTS / target['console_file'])
        for path in sorted(consoles):
            size = path.stat().st_size
            offset = offsets.get(path, 0)
            if size < offset:
                offset, pending[path] = 0, ''
            with path.open(errors='replace') as file:
                file.seek(offset)
                chunk = pending.get(path, '') + file.read()
                offsets[path] = file.tell()
            parts = chunk.split('\n')
            pending[path] = parts.pop()
            for line in parts:
                if ' {"event":' not in line:
                    continue
                try:
                    parsed = console_record(line, path.stat().st_mtime)
                    if parsed:
                        mon.ingest(*parsed)
                except (ValueError, KeyError):
                    pass  # legacy console lines are truncated; committed JSONL is the fallback
        if not args.local_only and now >= next_cloud:
            try:
                python = str(PROJECT_PYTHON) if PROJECT_PYTHON.exists() else sys.executable
                result = subprocess.run([python, __file__, '--pull'], capture_output=True, text=True, timeout=25, cwd=ROOT)
                if result.returncode:
                    raise RuntimeError(clean((result.stderr or result.stdout)[-600:]))
                cloud_error = None
            except (subprocess.TimeoutExpired, RuntimeError) as exc:
                cloud_error = clean(str(exc))
            next_cloud = time.time() + (max(args.cloud_every, 180) if cloud_error else args.cloud_every)
        cloud = read_json(CLOUD, {}) or {}
        target = read_json(TARGET, {}) or {}
        run_id = target.get('run_id') or cloud.get('run_id')
        for paired_id, log in cloud.get('paired_logs', {}).items():
            for line in log.splitlines():
                try:
                    mon.ingest(paired_id, json.loads(line))
                except (ValueError, KeyError):
                    pass
        if cloud.get('run_id'):
            for line in cloud.get('log', '').splitlines():
                try:
                    mon.ingest(cloud['run_id'], json.loads(line))
                except (ValueError, KeyError):
                    pass
        if not run_id and mon.runs:
            run_id = max(mon.runs, key=lambda key: mon.runs[key].get('last_event_at', 0))
        benchmark = cloud.get('benchmark') if cloud.get('run_id') == run_id else None
        snap = mon.snapshot(run_id, now=time.time(), stale_s=args.stale_seconds, ledger=cloud.get('ledger'), benchmark=benchmark)
        snap.update(pid=os.getpid(), cloud_error=cloud_error or cloud.get('log_error'),
                    cloud_fetched_at=cloud.get('fetched_at'), monitor_uptime_s=round(time.time() - started, 1))
        local_campaign = read_json(REPORTS / 'evolution_campaign_local.json', {}) or {}
        snap['user_paused'] = local_campaign.get('status') == 'paused_for_review_by_user'
        if local_campaign.get('evaluation_review', {}).get('user_authorized'):
            snap['evaluation_review'] = read_json(REPORTS / 'capability_review_summary.json', {}) or {}
            snap['evaluation_review_budget'] = read_json(REPORTS / 'capability_review_budget.json', {}) or {}
        if target.get('campaign_ledger') == 'rhythm_v2_budget.json':
            snap['user_paused'] = False
            snap.pop('evaluation_review', None)
            snap.pop('evaluation_review_budget', None)
            snap['rhythm_budget'] = read_json(REPORTS / target['campaign_ledger'], {})
            snap['paired_runs'] = [mon.snapshot(rid, now=time.time(), stale_s=args.stale_seconds) for rid in target['run_ids']]
            if target.get('round') == 'd65-local-multifidelity':
                snap['status'] = 'local_pending'
                snap['current_work'] = dict(round=target['round'], status='local_exploration')
            elif target.get('round') == 'd64-controlled-selection':
                snap['status'] = 'preflight'
                snap['current_work'] = dict(round=target['round'], status='implementation_and_pipeline_checks')
            elif target.get('round') == 'r03-window-conditions-v1':
                r03_campaign_status(snap, snap['rhythm_budget'], target)
            elif target.get('round') == 'r04-initial-loss-v1':
                r04_campaign_status(snap, snap['rhythm_budget'], target)
            elif target.get('round') == 'r05-direct-condition-v1':
                r05_campaign_status(snap, snap['rhythm_budget'], target)
            elif target.get('round') in ('r07-blank-context-v1','r07b-blank-context-local-v1'):
                r07_campaign_status(snap, snap['rhythm_budget'], target)
            elif target.get('round') == 'r06-ordered-time-v1':
                r06_campaign_status(snap, snap['rhythm_budget'], target)
            else:
                snap['condition_diagnostic'] = read_json(REPORTS / 'history-density-v1_summary.json', {})
                condition_campaign_status(snap, snap['rhythm_budget'], stale_s=args.stale_seconds)
            active_diagnostic_status(snap, snap['rhythm_budget'], target)
            if target.get('phase') == 'prepare' and target.get('round') != 'r03-window-conditions-v1':
                console = REPORTS / 'evolution_condition_prepare_console.log'
                counts = re.findall(r'"event": "condition_metadata", "batches": (\d+)', console.read_text(errors='replace')) if console.exists() else []
                snap['current_work'] = {'round': 'condition-v1', 'status': 'metadata_prepare',
                                        'prepared_batches': int(counts[-1]) if counts else 0}
        local_probe_status(snap, target)
        local_data_status(snap, target)
        external_baseline_status(snap, target)
        audio_coordinate_status(snap, target)
        paired_state_probe_status(snap, target)
        training_horizon_status(snap, target)
        candidate_probe_status(snap, target)
        full_event_probe_status(snap, target)
        local_multifidelity_status(snap, target)
        local_song_training_status(snap, target)
        save(STATE, snap)
        render(snap)
        fingerprint = (snap['run_id'], snap['status'], snap.get('step'), tuple(snap['alerts']), snap['cloud_error'])
        if fingerprint != last_print:
            # The saved snapshot retains full evidence; repeated console lines need only live state.
            console = {key: snap[key] for key in ('run_id', 'status', 'effective_updates', 'metrics',
                'observed_at', 'alerts', 'cloud_error', 'current_work') if key in snap}
            print(json.dumps(clean(console), ensure_ascii=False, allow_nan=False), flush=True)
            last_print = fingerprint
        for alert in snap['alerts']:
            key = f'{run_id}:{alert}'
            if key not in emitted:
                with ALERTS.open('a') as file:
                    file.write(json.dumps({'key': key, 'at': time.time(), 'run_id': run_id, 'alert': alert}, ensure_ascii=False) + '\n')
                emitted.add(key)
        if not args.watch or time.time() - started >= args.max_hours * 3600:
            break
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
