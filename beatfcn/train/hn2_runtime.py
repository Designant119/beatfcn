"""Bounded HN2 CUDA execution and pure, CPU-testable launch/resource gates.

Importing this module never constructs a model, imports Modal, reads credentials
or reserves money. Only the explicitly gated cloud worker calls ``run_cuda``.
"""
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time


GRANT = 'owner-modal120m-20261030'
REQUIRED_SOURCES = (
    'beatfcn/model/hn2_native.py', 'beatfcn/train/hn2_native.py',
    'beatfcn/train/hn2_runtime.py', 'beatfcn/cloud/hn2_app.py',
    'beatfcn/train/modal_event_dataset.py', 'scripts/run_hn2_modal.py',
    'beatfcn/train/owner_candidate_budget.py',
)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f'.{threading.get_ident()}.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def rooted(root, name):
    root = Path(root).resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Path escapes the frozen root')
    return path


@dataclass(frozen=True)
class RuntimeConfig:
    applied_updates: int = 1000
    effective_batch: int = 32
    workers: int = 4
    log_every: int = 25
    evaluate_every: int = 250
    evaluation_batch: int = 4
    seed: int = 20261030
    learning_rate: float = 0.0002
    minimum_learning_rate: float = 0.00002
    warmup_updates: int = 100
    weight_decay: float = 0.01
    clip: float = 1.0
    evaluation_wall_s: float = 600.0
    min_util: float = 70.0
    util_intervals: int = 8
    check_after_s: float = 480.0
    max_data_wait: float = 0.8
    check_data_after_s: float = 480.0

    def __post_init__(self):
        if self.applied_updates < 1 or self.effective_batch != 32 or self.workers < 1:
            raise ValueError('Positive endpoint/workers and effective batch32 required')
        if self.log_every != 25 or self.evaluate_every != 250:
            raise ValueError('Frozen 25-step logging and 250-step actual evaluation required')
        if (self.min_util, self.util_intervals, self.check_after_s,
            self.max_data_wait, self.check_data_after_s) != (70., 8, 480., .8, 480.):
            raise ValueError('Original resource guards cannot be relaxed')
        if self.evaluation_batch < 1 or self.evaluation_wall_s <= 0:
            raise ValueError('Positive bounded evaluation resources required')
        if not 0 < self.minimum_learning_rate <= self.learning_rate or not 0 < self.clip or self.warmup_updates < 1:
            raise ValueError('Invalid optimizer contract')


class ResourceGuard:
    """Last-eight interval means, original thresholds, missing samples stop."""
    def __init__(self, config=None):
        self.config = config or RuntimeConfig()
        self.utilization = deque(maxlen=8)

    def observe(self, util, *, elapsed_s, data_wait_s, training_elapsed_s=None):
        if util is None or not math.isfinite(float(util)) or not 0 <= float(util) <= 100:
            raise RuntimeError('GPU utilization observation unavailable; fail closed')
        if not all(math.isfinite(v) and v >= 0 for v in (elapsed_s, data_wait_s)):
            raise RuntimeError('Invalid resource elapsed/wait meter')
        self.utilization.append(float(util))
        c = self.config
        if elapsed_s >= c.check_data_after_s and data_wait_s > c.max_data_wait * elapsed_s:
            raise RuntimeError(f'Data wait {data_wait_s:.3f}s of {elapsed_s:.3f}s exceeds80%')
        training_elapsed_s = elapsed_s if training_elapsed_s is None else training_elapsed_s
        mean = sum(self.utilization) / len(self.utilization)
        if training_elapsed_s >= c.check_after_s and mean < c.min_util:
            raise RuntimeError(f'GPU util {mean:.3f}% mean of {len(self.utilization)} intervals <70%')
        return dict(util_rolling_mean=mean, util_intervals=len(self.utilization))


def evaluation_due(applied_updates, config=None):
    c = config or RuntimeConfig()
    return applied_updates > 0 and (applied_updates % c.evaluate_every == 0 or applied_updates == c.applied_updates)


def validate_envelope(contract, authorization, review, local_gate):
    """No IO/API calls; user budget and independent review are both mandatory."""
    if contract.get('grant_id') != GRANT or authorization.get('grant_id') != GRANT:
        raise ValueError('Independent Owner candidate grant required; old pools are ineligible')
    try:
        cap = Decimal(str(authorization.get('cap_usd')))
    except InvalidOperation as exc:
        raise ValueError('Concrete Owner dollar cap is still pending') from exc
    if not cap.is_finite() or cap <= 0 or authorization.get('authorized_by_owner') is not True:
        raise ValueError('Concrete positive Owner-authorized cap is absent')
    if contract.get('actual_trainable_parameters', 0) < 120_000_000:
        raise ValueError('Actual >=120M trainable candidate required')
    if review.get('passed') is not True or local_gate.get('passed') is not True:
        raise ValueError('Independent review and actual local gate must pass')
    sources = contract.get('source_sha256', {})
    if not set(REQUIRED_SOURCES) <= set(sources):
        raise ValueError('Frozen runtime/model/data source closure is incomplete')
    for receipt in (authorization, review):
        if (receipt.get('candidate_id') != contract.get('candidate_id')
                or receipt.get('source_sha256') != sources
                or receipt.get('data_manifest_sha256') != contract.get('data_manifest_sha256')):
            raise ValueError('Candidate/source/data identity differs from authorization or review')
    if (local_gate.get('candidate_id') != contract.get('candidate_id')
            or local_gate.get('source_sha256') != sources
            or local_gate.get('data_manifest_sha256') != contract.get('data_manifest_sha256')):
        raise ValueError('Local admission identity differs from this exact candidate/source/data')
    if not isinstance(contract.get('data_manifest_sha256'), dict) or not contract['data_manifest_sha256']:
        raise ValueError('Freeze a data/prior/upload manifest path-to-SHA mapping')
    if contract.get('density_rule') != 'all-events-1s-gt12-drag-only-v2' or contract.get('evaluate_every') != 250:
        raise ValueError('Owner v2 data rule and250-applied-update evaluation required')
    if authorization.get('review_passed') is not True:
        raise ValueError('Authorization does not acknowledge the passing independent review')
    RuntimeConfig(**contract.get('training', {}))
    resources = contract.get('resources', {})
    if (resources.get('gpu') != 'H100' or resources.get('max_containers') != 1
            or resources.get('retries') != 0):
        raise ValueError('Exactly one H100 and zero retries required')
    for key in ('cpu', 'memory_gib', 'timeout_s', 'startup_allowance_s', 'stop_allowance_s'):
        if not math.isfinite(float(resources.get(key, 0))) or float(resources.get(key, 0)) <= 0:
            raise ValueError('Missing positive lifecycle resource: ' + key)
    if resources['timeout_s'] <= resources['stop_allowance_s']:
        raise ValueError('Runtime must preserve external-stop allowance')
    if not contract.get('evaluation_panel'):
        raise ValueError('Freeze explicit 250-step evaluation panel before dispatch')
    health=contract.get('cloud_health',{})
    if (health.get('updates')!=100 or health.get('effective_batch')!=5
            or len(health.get('fit_draft_indices',[]))!=5):
        raise ValueError('Separate full-model discarded100-update/five-request fit contract required')
    acceptance=health.get('acceptance',{})
    for key in ('maximum_total_loss','maximum_loss_ratio','maximum_field_ratio'):
        if not isinstance(acceptance.get(key),(int,float)) or not 0<float(acceptance[key]):
            raise ValueError('Unfrozen full-model fit acceptance: '+key)
    return cap


def check_fresh_connection(connection, *, now=None):
    now = time.time() if now is None else now
    age = now - float(connection.get('checked_at_unix', 0))
    results = connection.get('results', {})
    if (not connection.get('fresh_check_succeeded') or not 0 <= age < 600
            or not connection.get('evidence_sha256') or not connection.get('full_lifecycle_bound_verified')
            or (results and not all(results.get(k, {}).get('ok') for k in ('apps', 'rates', 'billing')))):
        raise ValueError('Fresh app/rates/billing evidence less than600s old required')


def validate_lifecycle_costs(contract, rates):
    """Explicit non-GPU bounds; a single H100 wall rate is insufficient."""
    life=contract.get('lifecycle',{})
    for key in ('gpu_usd_s','cpu_usd_s','memory_gib_usd_s'):
        if Decimal(str(rates.get(key,0)))<=0:raise ValueError('Positive current normalized rate required: '+key)
    cpu=life.get('cpu_asset_verification',{})
    if (cpu.get('containers'),cpu.get('cpu'),cpu.get('memory_gib'),cpu.get('timeout_s'))!=(1,4,4,1200):
        raise ValueError('Bound the extra CPU admission at1x4core/4GiB/1200s')
    cpu_upper=Decimal(1200)*(4*Decimal(str(rates['cpu_usd_s']))+4*Decimal(str(rates['memory_gib_usd_s'])))
    other=cpu_upper
    for key in ('image_build_upper_usd','upload_upper_usd','failure_cleanup_upper_usd'):
        value=Decimal(str(life.get(key,'NaN')))
        if not value.is_finite() or value<0:raise ValueError('Explicit lifecycle bound missing: '+key)
        other+=value
    storage=life.get('volume_storage',{});egress=life.get('egress',{})
    if storage.get('gib',0)<20 or storage.get('retention_days',0)<=0 or not storage.get('cleanup_plan'):
        raise ValueError('20GiB Volume retention duration and actual cleanup plan required')
    if egress.get('gib',-1)<0:raise ValueError('Explicit egress byte bound required')
    for item in (storage,egress):
        value=Decimal(str(item.get('upper_usd','NaN')))
        if not value.is_finite() or value<0:raise ValueError('Storage/egress upper cost missing')
        other+=value
    if Decimal(str(contract['resources'].get('other_upper_usd',0)))<other:
        raise ValueError('other_upper_usd does not cover CPU/build/upload/storage/egress/cleanup')
    return dict(extra_cpu_upper_usd=str(cpu_upper),minimum_other_upper_usd=str(other),
        storage_cleanup_execution='Must be verified before full_lifecycle_bound_verified;not an automatic Volume TTL')


def local_launch_gate(root, contract_path, authorization_path=None, *, now=None, verify_assets=False):
    root = Path(root)
    contract_path = rooted(root, contract_path)
    contract = json.loads(contract_path.read_text())
    review_path = rooted(root, contract['independent_review_receipt'])
    local_path = rooted(root, contract['local_gate_receipt'])
    review, local_gate = json.loads(review_path.read_text()), json.loads(local_path.read_text())
    ledger_path = rooted(root, contract.get('ledger_path','reports/rhythm_v2_budget.json'))
    ledger = json.loads(ledger_path.read_text())
    grant = ledger.get('conditional_candidate_authorizations',{}).get(GRANT,{})
    # This ephemeral summary is derived from the sole authoritative ledger.
    # It creates no grant and does not replace reserve_owner_candidate's check.
    authorization = dict(grant_id=GRANT,cap_usd=grant.get('new_usd_cap'),
        authorized_by_owner=grant.get('owner_authorized') is True,review_passed=review.get('passed') is True,
        candidate_id=contract.get('candidate_id'),source_sha256=contract.get('source_sha256'),
        data_manifest_sha256=contract.get('data_manifest_sha256'),ledger_sha256=sha(ledger_path))
    if authorization_path is not None:
        supplied=json.loads(rooted(root,authorization_path).read_text())
        if supplied != authorization:raise ValueError('Optional authorization summary differs from authoritative ledger')
    validate_envelope(contract, authorization, review, local_gate)
    sdk=contract.get('local_modal_sdk_sha256',{})
    if not all(any(name.endswith('/modal/'+file) for name in sdk) for file in ('app.py','runner.py')):
        raise ValueError('Freeze actual SDK pre-build app-id assignment evidence')
    for name,expected in sdk.items():
        if not name.startswith('.venv/') or not name.endswith('.py') or sha(rooted(root,name))!=expected:
            raise ValueError('Local Modal construction/stop SDK source drift')
    if review.get('unresolved_blockers') or not review.get('independent_original_reports'):
        raise ValueError('Independent originals and no unresolved blockers required')
    for item in review['independent_original_reports']:
        if sha(rooted(root,item['path'])) != item['sha256']:
            raise ValueError('Independent original report drift')
    for name, expected in contract['source_sha256'].items():
        if Path(name).suffix != '.py' or Path(name).parts[0] not in ('beatfcn', 'scripts', 'tests'):
            raise ValueError('Source inventory may contain only reviewed Python source')
        if sha(rooted(root, name)) != expected:
            raise ValueError('Frozen source drift: ' + name)
    for path,expected in contract['data_manifest_sha256'].items():
        if sha(rooted(root,path)) != expected:
            raise ValueError('Frozen data/upload manifest drift')
    if not all(contract.get(k) in contract['data_manifest_sha256'] for k in ('data_manifest','upload_manifest','duration_prior_path')):
        raise ValueError('Data/upload/fixed duration prior must all have frozen SHA identities')
    data = json.loads(rooted(root, contract['data_manifest']).read_text())
    if not data.get('complete') or data.get('target_slots') != 512:
        raise ValueError('Complete CPU-prepared512-slot dataset required')
    known = {r['array_index']: r for r in data['records'] if r['split'] == 'dev'}
    keys = set()
    for spec in contract['evaluation_panel']:
        key = (spec['array_index'], spec['audio'])
        if spec['array_index'] not in known or spec['audio'] not in ('real', 'zero') or key in keys:
            raise ValueError('Evaluation panel is not a fixed unique development/zero panel')
        keys.add(key)
    if verify_assets:
        upload = json.loads(rooted(root, contract['upload_manifest']).read_text())
        for name, info in upload['files'].items():
            path = rooted(root, name)
            if path.stat().st_size != info['bytes'] or sha(path) != info['sha256']:
                raise ValueError('Data payload drift: ' + name)
    connection = json.loads(rooted(root, contract['connection_path']).read_text())
    check_fresh_connection(connection, now=now)
    rates = json.loads(rooted(root, contract['rates_path']).read_text())
    validate_lifecycle_costs(contract,rates)
    return dict(contract=contract, authorization=authorization, review=review, local_gate=local_gate,
        connection=connection, rates=rates, contract_json=contract_path.read_text(),
        authorization_json=json.dumps(authorization), review_json=review_path.read_text(),
        local_gate_json=local_path.read_text())


class DeadlineWatchdog:
    """Independent of training progress; injected hard-stop enables CPU tests."""
    def __init__(self, deadline_unix, stop_allowance_s, *, on_soft_stop, hard_stop=None):
        self.deadline = float(deadline_unix)
        self.allowance = float(stop_allowance_s)
        self.on_soft_stop = on_soft_stop
        self.hard_stop = hard_stop or (lambda: os._exit(124))
        self.closed, self.stop = threading.Event(), threading.Event()
        self.thread = threading.Thread(target=self._watch, daemon=True)

    def _watch(self):
        if self.closed.wait(max(0, self.deadline - self.allowance - time.time())):
            return
        self.stop.set()
        self.on_soft_stop()
        if not self.closed.wait(max(0, self.deadline - time.time())):
            self.hard_stop()

    def start(self):
        self.thread.start()
        return self

    def check(self):
        if self.stop.is_set() or time.time() >= self.deadline - self.allowance:
            raise TimeoutError('Lifecycle deadline preserves independent stop allowance')

    def close(self):
        self.closed.set()


class ConstructionWatchdog:
    """Pure state machine: discover owned app before app.run yields/build ends."""
    def __init__(self,hard_deadline,stop_allowance):
        self.hard_deadline=hard_deadline;self.soft_deadline=hard_deadline-stop_allowance
        self.app_id=None;self.last_meter=-float('inf')

    def tick(self,app_id,now):
        if app_id:
            if self.app_id and app_id!=self.app_id:raise RuntimeError('Owned app identity changed')
            self.app_id=app_id
        due=self.app_id is not None and now-self.last_meter>=10
        if due:self.last_meter=now
        return dict(app_id=self.app_id,meter_due=due,stop_required=now>=self.soft_deadline,
                    hard_stop_required=now>=self.hard_deadline)


class JsonLog:
    def __init__(self, run_id, directory):
        self.run_id = run_id
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.file = (self.directory / 'log.jsonl').open('a')
        self.lock = threading.Lock()

    def emit(self, event, **fields):
        record = dict(event=event, ts=time.time(), **fields)
        line = json.dumps(record, allow_nan=False)
        with self.lock:
            self.file.write(line + '\n')
            self.file.flush()
            print(self.run_id, line, flush=True)

    def close(self):
        self.file.close()


def zero_audio_inputs(inputs):
    """All four audio value streams change; public clocks/support remain fixed."""
    import torch
    import numpy as np
    streams = {'local', 'mel', 'song', 'whole_mel'}
    if not streams <= inputs.keys():
        raise ValueError('All-stream zero control requires local/mel/song/whole_mel')
    return {k: (torch.zeros_like(v) if torch.is_tensor(v) else np.zeros_like(v)) if k in streams else v
            for k, v in inputs.items()}


def assert_public_inputs(inputs):
    forbidden = {'targets', 'target_valid', 'target_count', 'counts', 'slot_valid', 'hidden_events'}
    if forbidden & set(inputs):
        raise ValueError('Hidden supervision entered model inputs')


def run_evaluation_schedule(specs, free, proper, persist, *, check=lambda: None):
    """All fixed-denominator free outcomes persist before any proper call."""
    outcomes = [dict(spec=spec, status='not_attempted') for spec in specs]
    persist('expected', outcomes)
    for i, spec in enumerate(specs):
        check()
        outcomes[i] = dict(spec=spec, status='free_completed', free=free(spec))
        persist('free', outcomes)
    persist('free_complete', outcomes)
    for i, spec in enumerate(specs):
        check()
        outcomes[i]['proper'] = proper(spec, outcomes[i]['free'])
        outcomes[i]['status'] = 'completed'
        persist('proper', outcomes)
    return outcomes


def _to_device(batch, device):
    return {k: {n: v.to(device, non_blocking=True) for n, v in value.items()} if k == 'inputs'
            else value.to(device, non_blocking=True) if k in ('targets', 'target_valid') else value
            for k, value in batch.items()}


def _score(events, truth):
    import numpy as np
    from beatfcn.model.event_canvas import RhythmEvent
    from beatfcn.model.local_song_rollout import canonical, independent_legality
    from beatfcn.model.evaluation import score_events
    kinds = ('tap', 'hold', 'flick', 'drag')
    pred = [RhythmEvent(int(h), int(t), kinds[int(k)], int(l)) for h,t,k,l in events]
    target = [RhythmEvent(int(h), int(t), kinds[int(k)], int(l)) for h,t,k,l in truth]
    pp, tt = canonical(pred,'phira'), canonical(target,'phira')
    all_scores = score_events(pp,tt,'phira')
    types = {}
    for kind in kinds:
        typed = score_events([e for e in pp if e.type==kind],[e for e in tt if e.type==kind],'phira')
        types[kind] = typed['hold_joint_50ms' if kind=='hold' else 'note_50ms']
    holds_p, holds_t = [e for e in pred if e.type=='hold'], [e for e in target if e.type=='hold']
    candidates = sorted((abs(p.onset_us-t.onset_us),i,j) for i,p in enumerate(holds_p)
                        for j,t in enumerate(holds_t) if abs(p.onset_us-t.onset_us)<=50000)
    used_p, used_t, tails = set(), set(), []
    for _,i,j in candidates:
        if i not in used_p and j not in used_t:
            used_p.add(i);used_t.add(j);tails.append(abs(holds_p[i].end_us-holds_t[j].end_us)/1000)
    note=all_scores['note_50ms'];head=all_scores['hold_head_50ms'];joint=all_scores['hold_joint_50ms']
    return dict(onset10=all_scores['onset_10ms'],full50=2*(note['tp']-head['tp']+joint['tp'])/max(len(pred)+len(target),1),
        per_type=types,predicted=len(pred),truth=len(target),empty_false_events=len(pred) if not target else None,
        hold=dict(predicted=len(holds_p),truth=len(holds_t),head_matches=len(tails),tail_abs_ms=tails,
                  median_tail_ms=float(np.median(tails)) if tails else None,joint50=sum(v<=50 for v in tails),
                  matching='Greedy nearest unused head within50ms;duplicates remain separate'),
        output_multiset=events)


def _proper_record(output, metrics, batch, prior):
    import torch
    result=[]
    probabilities=metrics['type_logits'].float().softmax(-1).cpu()
    labels=metrics['targets'].cpu();valid=metrics['target_valid'].cpu()
    supported=metrics['type_support'].cpu();active=output['active_logits'].float().sigmoid().cpu()
    for i in range(len(labels)):
        mask=valid[i];kinds=labels[i,mask,2];p=probabilities[i,mask];support=supported[i,mask]
        base=torch.tensor(prior['global_probability'])[None]*support
        base=base/base.sum(-1,keepdim=True).clamp_min(1e-30)
        truth_p=p.gather(1,kinds[:,None]).squeeze(1).clamp_min(1e-30)
        prior_p=base.gather(1,kinds[:,None]).squeeze(1).clamp_min(1e-30)
        first=None if not len(kinds) else dict(kind=int(kinds[0]),model_nll=float(-truth_p[0].log()),
            train_prior_nll=float(-prior_p[0].log()),scope='First canonical target,all-mask p0 or same free-machine z p1;not free type quality')
        type_stats={}
        for k,name in enumerate(('tap','hold','flick','drag')):
            positive=kinds==k;negative=(kinds!=k)&support[:,k]
            type_stats[name]=dict(positive=int(positive.sum()),negative=int(negative.sum()),
                positive_nll=(-truth_p[positive].log()).tolist(),prior_positive_nll=(-prior_p[positive].log()).tolist(),
                negative_nll=(-(1-p[negative,k]).clamp_min(1e-30).log()).tolist(),
                prior_negative_nll=(-(1-base[negative,k]).clamp_min(1e-30).log()).tolist())
        hold=mask&(labels[i,:,2]==1)
        result.append(dict(first=first,types=type_stats,type_probability=p.tolist(),
            type_allowed=support.tolist(),head_nll=metrics['head_nll'][i].cpu()[mask].tolist(),
            hold_tail_nll=metrics['tail_nll'][i].cpu()[hold].tolist(),
            occupancy=dict(expected=float(active[i].sum()),marginal_mode=int((active[i]>=.5).sum()),
                expected_on_NULL=float(active[i,~mask].sum()),true_slots_below_half=int((active[i,mask]<.5).sum()),
                probability=active[i].tolist()),supported_members=metrics['supported_members'][i].cpu().tolist()))
    return result


def diagnostic_context_states(machine,targets,target_valid):
    """Evaluation-only NULL/TRUE arms; TRUE is explicitly an oracle."""
    import torch
    from beatfcn.train.hn2_native import canonical_targets
    truth,valid=canonical_targets(targets,target_valid)
    return dict(own_z=machine,
        null=dict(events=torch.zeros_like(machine['events']),active=torch.zeros_like(machine['active'])),
        true_oracle=dict(events=torch.where(valid[...,None],truth,torch.zeros_like(truth)),active=valid))


def fixed_z_cuda_parity(cpu_batch,prior,seed):
    """Actual FP32 CPU/CUDA values/gradients before any optimized update."""
    import torch
    from beatfcn.model.hn2_native import HN2Native,HN2Config
    from beatfcn.train.hn2_native import loss_pass,sample_pass
    torch.manual_seed(seed)
    c=HN2Config(d=64,heads=4,ffn=256,fine=32)
    base=HN2Native(c).eval()
    with torch.no_grad():base.tail_residual.weight.fill_(.01)
    weights={k:v.detach().clone() for k,v in base.state_dict().items()}
    with torch.no_grad():
        first=base(cpu_batch['inputs'],pass_index=0)
        z=sample_pass(base,first,prior,generator=torch.Generator().manual_seed(seed+1))
    del first,base
    results=[]
    for device in ('cpu','cuda'):
        model=HN2Native(c).to(device).eval();model.load_state_dict(weights)
        batch=_to_device(cpu_batch,device);state={k:v.to(device) for k,v in z.items()}
        out=model(batch['inputs'],state=state,pass_index=1)
        loss,details=loss_pass(model,out,batch['targets'],batch['target_valid'],prior,details=True)
        loss.backward()
        results.append(dict(loss=float(loss.detach().cpu()),head=model.head_logits(out).detach().cpu(),
            type=details['type_logits'].cpu(),tail=[r['logits'][r['valid']].cpu() for r in details['tail_factors']],
            gradients={n:p.grad.detach().cpu() for n,p in model.named_parameters() if p.grad is not None}))
        del model,out,batch,details,loss,state
    left,right=results;errors=dict(loss=abs(left['loss']-right['loss']))
    for key in ('head','type'):
        a,b=left[key],right[key];valid=torch.isfinite(a)&torch.isfinite(b)
        if not torch.equal(torch.isfinite(a),torch.isfinite(b)):raise ValueError('CUDA parity support/finite mask mismatch')
        errors[key]=float((a[valid]-b[valid]).abs().max()) if valid.any() else 0.
    for i,(a,b) in enumerate(zip(left['tail'],right['tail'])):
        errors['tail'+str(i)]=float((a-b).abs().max()) if a.numel() else 0.
    if left['gradients'].keys()!=right['gradients'].keys():raise ValueError('CUDA gradient presence differs')
    errors['gradient']=max(float((v-right['gradients'][n]).abs().max()) for n,v in left['gradients'].items())
    if not all(math.isfinite(v) and v<1e-4 for v in errors.values()):raise ValueError('Actual fixed-z CPU/CUDA parity failed: '+str(errors))
    return dict(passed=True,errors=errors,optimizer_updates=0,torch_version=torch.__version__,
        scope='Narrow same-module FP32 CUDA runtime gate;not full-fit or cross-version local2.14 parity')


def full_fit(model,dataset,contract,prior,micro,config,watchdog,logger):
    """Discarded full-model100 updates, five distinct draft rows each update."""
    import torch
    from beatfcn.train.modal_event_dataset import collate_event_batch
    from beatfcn.train.hn2_native import training_step,generate
    h=contract['cloud_health'];draft=[i for i,r in enumerate(dataset.records) if r['view']=='draft']
    if len(set(h['fit_draft_indices']))!=5 or any(not 0<=i<len(draft) for i in h['fit_draft_indices']):
        raise ValueError('Health rows must be five distinct prepared train/draft indices')
    rows=[dataset[draft[i]] for i in h['fit_draft_indices']]
    batches=[_to_device(collate_event_batch(rows[i:i+micro]),'cuda') for i in range(0,5,micro)]
    def measure():
        model.eval();value=0.;fields=Counter()
        with torch.no_grad():
            g=torch.Generator(device='cuda').manual_seed(config.seed+900)
            for batch in batches:
                with torch.autocast('cuda',dtype=torch.bfloat16):loss,metrics,_,_=training_step(model,batch,prior,generator=g)
                weight=len(batch['targets'])/5
                value+=float(loss)*weight
                for p,items in metrics.items():
                    for key in ('occupancy','head','kind','tail'):fields[p+'_'+key]+=float(items[key])*weight
        return dict(loss=value,fields=dict(fields))
    before=measure();opt=torch.optim.AdamW(model.parameters(),lr=config.learning_rate,weight_decay=config.weight_decay,fused=True)
    generator=torch.Generator(device='cuda').manual_seed(config.seed+901);curve=[]
    from beatfcn.cloud.gpu_util import GpuUtil
    monitor=GpuUtil();guard=ResourceGuard(config);start=time.monotonic()
    try:
        for step in range(1,101):
            watchdog.check();model.train();opt.zero_grad(set_to_none=True);total=0.
            for batch in batches:
                with torch.autocast('cuda',dtype=torch.bfloat16):loss,_,_,_=training_step(model,batch,prior,generator=generator)
                if not bool(torch.isfinite(loss)):raise FloatingPointError('Full-fit loss nonfinite')
                weight=len(batch['targets'])/5
                (loss*weight).backward();total+=float(loss.detach())*weight
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config.clip,error_if_nonfinite=True);opt.step()
            write_json(logger.directory/'phase.json',dict(phase='full_fit',health_applied_updates=step,quality_applied_updates=0))
            if step%25==0:
                sample=monitor.take();guard.observe(sample['util_mean'],elapsed_s=time.monotonic()-start,data_wait_s=0.)
                curve.append(dict(step=step,loss=total,grad_norm=float(norm),gpu_util=sample['util_mean']))
                logger.emit('health',step=step,loss=total,unique_requests=5,effective_batch=5)
        after=measure();model.eval();free=[]
        with torch.no_grad():
            for row in rows:
                batch=_to_device(collate_event_batch([row]),'cuda')
                with torch.autocast('cuda',dtype=torch.bfloat16):z,y=generate(model,batch['inputs'],prior,generator=generator)
                truth=row['targets'].tolist()
                free.append({p:_score(state['events'][0,state['active'][0]].cpu().tolist(),truth)
                             for p,state in [('p0',z),('p1',y)]})
        a=h['acceptance'];checks=dict(finite=math.isfinite(after['loss']),
            absolute=after['loss']<=a['maximum_total_loss'],ratio=after['loss']<=a['maximum_loss_ratio']*before['loss'],
            fields=all(v<=a['maximum_field_ratio']*before['fields'][k]+1e-6 for k,v in after['fields'].items()))
        counters=[int(v['step']) for v in opt.state.values()]
        checks['actual_optimizer_steps']=len(counters)==len(list(model.parameters())) and min(counters)==max(counters)==100
        return dict(passed=all(checks.values()),checks=checks,before=before,after=after,curve=curve,free=free,
            applied_updates=100,unique_requests=5,effective_batch=5,per_request_exposures=100,
            fit_rows=[dict(draft_index=i,array_index=r['metadata']['array_index'],source_index=r['metadata']['index'])
                      for i,r in zip(h['fit_draft_indices'],rows)],
            optimizer_state_steps=dict(parameters=len(counters),minimum=min(counters),maximum=max(counters)),
            weights_and_optimizer_discarded=True)
    finally:monitor.close();del opt,batches


def evaluate_checkpoint(model, dataset, specs, prior, directory, *, step, config, watchdog, logger):
    """CUDA batches for free p0/p1 first; matching proper uses the saved free z."""
    import torch
    from beatfcn.train.modal_event_dataset import collate_event_batch
    from beatfcn.train.hn2_native import generate,loss_pass
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    lookup={r['array_index']:i for i,r in enumerate(dataset.records)}
    expected=[]
    for spec in specs:
        record=dataset.records[lookup[spec['array_index']]]
        expected.append(dict(key=f"{spec['array_index']}:{spec['audio']}",spec=spec,status='not_attempted',
            component=record['component'],index=record['index'],view=record['view'],
            target_count=record['target_slice'][1]-record['target_slice'][0],request=record['request']))
    write_json(directory/'expected.json',expected)
    # Group only by public fine-audio bucket; preserve each frozen request.
    buckets=defaultdict(list)
    for position,spec in enumerate(specs):buckets[dataset.public_audio_bucket(lookup[spec['array_index']])].append(position)
    batches=[v[i:i+config.evaluation_batch] for v in buckets.values() for i in range(0,len(v),config.evaluation_batch)]
    started=time.monotonic();model.eval();saved={};free_records=[]
    generator=torch.Generator(device='cuda').manual_seed(config.seed+step)
    def get_batch(positions):
        rows=[dataset[lookup[specs[p]['array_index']]] for p in positions]
        for p,row in zip(positions,rows):
            if specs[p]['audio']=='zero':row['inputs']=zero_audio_inputs(row['inputs'])
        batch=collate_event_batch(rows)
        assert_public_inputs(batch['inputs'])
        return batch,_to_device(batch,'cuda')
    def check():
        watchdog.check()
        if time.monotonic()-started>config.evaluation_wall_s:raise TimeoutError('Fixed evaluation wall exhausted')
    with torch.no_grad():
        for positions in batches:
            check();cpu,batch=get_batch(positions)
            with torch.autocast('cuda',dtype=torch.bfloat16):z,y=generate(model,batch['inputs'],prior,generator=generator)
            zs={k:v.detach().cpu() for k,v in z.items()};ys={k:v.detach().cpu() for k,v in y.items()}
            for j,p in enumerate(positions):
                truth=cpu['targets'][j,cpu['target_valid'][j]].tolist()
                record=dict(position=p,spec=specs[p],component=cpu['metadata'][j]['component'],view=cpu['metadata'][j]['view'],
                    status='free_completed',p0=_score(zs['events'][j,zs['active'][j]].tolist(),truth),
                    p1=_score(ys['events'][j,ys['active'][j]].tolist(),truth))
                saved[p]={k:v[j].clone() for k,v in zs.items()}
                free_records.append(record);expected[p]['status']='free_completed'
            write_json(directory/'free.json',sorted(free_records,key=lambda r:r['position']))
            write_json(directory/'expected.json',expected)
            del z,y,batch,cpu
        write_json(directory/'free-complete.json',dict(expected=len(specs),completed=len(free_records),passes=['p0','p1']))
        records={r['position']:r for r in free_records}
        for positions in batches:
            check();cpu,batch=get_batch(positions)
            state={k:torch.stack([saved[p][k] for p in positions]).to('cuda',non_blocking=True) for k in ('events','active')}
            with torch.autocast('cuda',dtype=torch.bfloat16):
                first=model(batch['inputs'],pass_index=0)
                _,m0=loss_pass(model,first,batch['targets'],batch['target_valid'],prior,details=True)
                second=model(batch['inputs'],state=state,memory=first['memory'],pass_index=1)
                _,m1=loss_pass(model,second,batch['targets'],batch['target_valid'],prior,details=True)
            p0,p1=_proper_record(first,m0,batch,prior),_proper_record(second,m1,batch,prior)
            contexts=diagnostic_context_states(state,batch['targets'],batch['target_valid'])
            context_records={}
            for context in ('null','true_oracle'):
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    output=model(batch['inputs'],state=contexts[context],memory=first['memory'],pass_index=1)
                    _,detail=loss_pass(model,output,batch['targets'],batch['target_valid'],prior,details=True)
                context_records[context]=_proper_record(output,detail,batch,prior)
                del output,detail
            for j,p in enumerate(positions):
                records[p].update(proper_p0=p0[j],proper_p1=p1[j],
                    proper_p1_null=context_records['null'][j],proper_p1_true_oracle=context_records['true_oracle'][j],
                    context_scope='Matchedaudio/head/targets/support;own-z fixed from persisted free p0;TRUE is oracle,never deployment evidence',
                    status='completed');expected[p]['status']='completed'
            write_json(directory/'proper.json',[records[p] for p in sorted(records)])
            write_json(directory/'expected.json',expected)
            del first,second,m0,m1,batch,cpu,state,contexts,context_records
    summary={}
    for pass_name in ('p0','p1'):
        rows=[r for r in records.values() if r['spec']['audio']=='real' and r['view']=='draft']
        components=defaultdict(list)
        for r in rows:components[r['component']].append(r[pass_name]['full50'])
        types={}
        for kind in ('tap','hold','flick','drag'):
            counters={key:sum(r[pass_name]['per_type'][kind][key] for r in rows) for key in ('tp','pred','true')}
            counters['f1']=2*counters['tp']/max(counters['pred']+counters['true'],1);types[kind]=counters
        summary[pass_name]=dict(full50_component_macro=sum(sum(v)/len(v) for v in components.values())/max(len(components),1),
            per_type=types,empty_false_events=sum(r[pass_name]['empty_false_events'] or 0 for r in rows))
    summary.update(step=step,expected_requests=len(specs),completed_requests=len(records),
        audio_controls_done=['all_stream_zero'],audio_controls_unmeasured=['shift','foreign'],
        context_proper_done=['fixed_own_z','NULL','TRUE_oracle'],
        scope='Fixed8s development windows;no whole-song/product/generalization claim',wall_s=time.monotonic()-started)
    write_json(directory/'summary.json',summary);logger.emit('val',step=step,**{k:v for k,v in summary.items() if k!='step'})
    return summary


class ContinuousPublicBatches:
    """Retain all rows and remainders; batch membership never inspects labels."""
    def __init__(self,dataset,batch_size,batches,seed):
        self.dataset,self.batch_size,self.batches,self.seed=dataset,batch_size,batches,seed

    def __len__(self):return self.batches

    def __iter__(self):
        from beatfcn.train.modal_event_dataset import PublicDurationBatchSampler
        pending=[];emitted=0;epoch=0
        while emitted<self.batches:
            sampler=PublicDurationBatchSampler(self.dataset,self.batch_size,seed=self.seed+epoch,shuffle=True)
            for batch in sampler:
                pending.extend(batch)
                while len(pending)>=self.batch_size and emitted<self.batches:
                    yield pending[:self.batch_size];pending=pending[self.batch_size:];emitted+=1
            epoch+=1


def run_cuda(contract, data_root, output_root, *, dispatch_deadline_unix, commit=lambda:None):
    """Called only inside the independently authorized, reviewed H100 worker."""
    import torch
    from beatfcn.cloud.gpu_util import GpuUtil
    from beatfcn.model.hn2_native import HN2Native,HN2Config
    from beatfcn.train.hn2_native import training_step,validate_targets
    from beatfcn.train.modal_event_dataset import ModalEventDataset,collate_event_batch
    config=RuntimeConfig(**contract.get('training',{}));resources=contract['resources']
    if not torch.cuda.is_available():raise RuntimeError('CUDA H100 admission required')
    if torch.__version__!='2.6.0+cu126':raise RuntimeError('Pinned historical CUDA runtime identity differs')
    torch.set_num_threads(4);torch.manual_seed(config.seed);torch.cuda.manual_seed_all(config.seed)
    directory=Path(output_root)/contract['run_id'];logger=JsonLog(contract['run_id'],directory)
    watchdog=DeadlineWatchdog(dispatch_deadline_unix,resources['stop_allowance_s'],
        on_soft_stop=lambda:logger.emit('aborted',reason='Independent lifecycle cutoff',step=0)).start()
    mon=None;model=None;optimizer=None;applied=0;function_start=time.monotonic()
    def checkpoint(reason):
        if model is None:return
        temporary=directory/'last.pt.partial'
        torch.save(dict(model=model.state_dict(),optimizer=None if optimizer is None else optimizer.state_dict(),
            applied_updates=applied,reason=reason,source_sha256=contract['source_sha256'],
            data_manifest_sha256=contract['data_manifest_sha256'],model_config=contract.get('model',{})),temporary)
        temporary.replace(directory/'last.pt');commit()
    try:
        train=ModalEventDataset(data_root,contract['data_manifest'],split='train')
        dev=ModalEventDataset(data_root,contract['data_manifest'],split='dev')
        type_prior=json.loads(rooted(data_root,train.manifest['priors']).read_text())
        prior=dict(json.loads(rooted(data_root,contract['duration_prior_path']).read_text())['duration'])
        prior['global_probability']=type_prior['global_probability']
        model=HN2Native(HN2Config(**contract.get('model',{}))).cuda()
        actual=sum(p.numel() for p in model.parameters() if p.requires_grad)
        if actual!=contract['actual_trainable_parameters'] or actual<120_000_000:
            raise ValueError('Actual CUDA trainable parameter identity differs')
        logger.emit('start',model=contract.get('model',{}),actual_trainable_parameters=actual,
            data_manifest_sha256=contract['data_manifest_sha256'],effective_batch=32,
            fixed_endpoint=config.applied_updates,calibration_optimizer_updates=0,
            torch_version=torch.__version__,cuda_version=torch.version.cuda,
            local_probe_torch_version='2.14.0',runtime_version_change=True)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        cpu_parity=collate_event_batch([train[0]])
        parity=fixed_z_cuda_parity(cpu_parity,prior,config.seed)
        write_json(directory/'cuda-fp32-parity.json',parity);del cpu_parity
        # Largest PUBLIC duration bucket; targets are not used to choose a size.
        largest=max(train.public_audio_bucket(i) for i in range(len(train)))
        pool=[i for i in range(len(train)) if train.public_audio_bucket(i)==largest]
        max_reference=max(range(len(train)),key=lambda i:train.records[i]['reference_slice'][1]-train.records[i]['reference_slice'][0])
        # Include both public maxima so B2 already pads to the joint worst
        # audio/reference shape. No hidden event count selects calibration.
        pool=[pool[0],max_reference,*pool[1:]]
        calibration=[];generator=torch.Generator(device='cuda').manual_seed(config.seed)
        for size in (2,8,16,32):
            watchdog.check();cpu=collate_event_batch([train[pool[i%len(pool)]] for i in range(size)])
            validate_targets(cpu['inputs'],cpu['targets'],cpu['target_valid'])
            batch=_to_device(cpu,'cuda');torch.cuda.reset_peak_memory_stats();start=time.monotonic()
            loss=None;norm=None
            try:
                model.train();model.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):loss,_,_,_=training_step(model,batch,prior,generator=generator)
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config.clip,error_if_nonfinite=True)
                torch.cuda.synchronize();peak=torch.cuda.max_memory_allocated()
                passed=peak<=.9*torch.cuda.get_device_properties(0).total_memory
                calibration.append(dict(micro_batch=size,passed=passed,peak_bytes=peak,wall_s=time.monotonic()-start,
                    finite_gradient=bool(torch.isfinite(norm)),optimizer_updates=0,public_bucket=largest))
            except torch.cuda.OutOfMemoryError:
                calibration.append(dict(micro_batch=size,passed=False,reason='CUDA_OOM',optimizer_updates=0,public_bucket=largest))
            finally:
                loss=None;norm=None;model.zero_grad(set_to_none=True);del batch,cpu;torch.cuda.empty_cache()
        admitted=[r['micro_batch'] for r in calibration if r['passed']]
        write_json(directory/'cuda-admission.json',dict(passed=bool(admitted),calibration=calibration,
            quality_updates=0,selection='Largest passing public-duration memory bucket;no dev/quality selection'))
        if not admitted:raise RuntimeError('CUDA numerical/resource admission failed;no quality updates')
        micro=max(admitted);accumulation=32//micro
        def tensor_digest(value):
            digest=hashlib.sha256()
            for name,tensor in sorted(value.state_dict().items()):
                digest.update(name.encode());digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
            return digest.hexdigest()
        initial=tensor_digest(model)
        health=full_fit(model,train,contract,prior,micro,config,watchdog,logger)
        write_json(directory/'full-fit.json',health);commit()
        if not health['passed']:raise RuntimeError('Full-model fixed-batch fit failed;quality prohibited')
        del model;torch.cuda.empty_cache();torch.manual_seed(config.seed);torch.cuda.manual_seed_all(config.seed)
        model=HN2Native(HN2Config(**contract.get('model',{}))).cuda()
        fresh=tensor_digest(model)
        if fresh!=initial:raise ValueError('Fresh quality initialization differs from discarded health initialization')
        write_json(directory/'fresh-quality-entry.json',dict(passed=True,health_initial_digest=initial,
            quality_initial_digest=fresh,health_optimizer_reused=False,health_weights_reused=False))
        generator.manual_seed(config.seed)
        sampler=ContinuousPublicBatches(train,micro,config.applied_updates*accumulation,config.seed)
        loader=torch.utils.data.DataLoader(train,batch_sampler=sampler,num_workers=config.workers,
            pin_memory=True,prefetch_factor=2,persistent_workers=True,collate_fn=collate_event_batch)
        iterator=iter(loader)
        optimizer=torch.optim.AdamW(model.parameters(),lr=config.learning_rate,weight_decay=config.weight_decay,fused=True)
        mon=GpuUtil();mon.take();guard=ResourceGuard(config);loop_start=time.monotonic();wait_s=0.;examples=0
        metrics_acc=defaultdict(float);exposure=Counter()
        while applied<config.applied_updates:
            watchdog.check();model.train();optimizer.zero_grad(set_to_none=True)
            progress=max(0,applied-config.warmup_updates)/max(1,config.applied_updates-config.warmup_updates)
            lr=config.learning_rate*(applied+1)/config.warmup_updates if applied<config.warmup_updates else config.minimum_learning_rate+.5*(config.learning_rate-config.minimum_learning_rate)*(1+math.cos(math.pi*progress))
            for group in optimizer.param_groups:group['lr']=lr
            for _ in range(accumulation):
                before=time.monotonic();cpu=next(iterator);batch=_to_device(cpu,'cuda');wait_s+=time.monotonic()-before
                assert_public_inputs(batch['inputs'])
                with torch.autocast('cuda',dtype=torch.bfloat16):loss,metrics,_,_=training_step(model,batch,prior,generator=generator)
                if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite HN2 training loss')
                (loss/accumulation).backward();examples+=len(cpu['targets'])
                kinds=cpu['targets'][cpu['target_valid']][:,2]
                for k,n in enumerate(torch.bincount(kinds,minlength=4).tolist()):exposure[('tap','hold','flick','drag')[k]]+=n
                for p,fields in metrics.items():
                    for k,v in fields.items():
                        if v.numel()==1:metrics_acc[p+'_'+k]+=v.detach()/accumulation
                del loss,metrics,batch,cpu
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config.clip,error_if_nonfinite=True)
            optimizer.step();applied+=1
            write_json(directory/'phase.json',dict(phase='quality',health_applied_updates=100,quality_applied_updates=applied))
            if applied%config.log_every==0:
                elapsed=time.monotonic()-loop_start;sample=mon.take();util=sample['util_mean']
                fields={k:dict(loss=float(v/config.log_every)) for k,v in metrics_acc.items() if k.rsplit('_',1)[-1] in ('occupancy','head','kind','tail')}
                logger.emit('train',step=applied,examples=examples,elapsed_s=elapsed,data_wait_s=wait_s,
                    fields=fields,grad_norm=float(norm),lr=lr,gpu_util=util,gpu_util_samples=sample['samples'],
                    peak_gb=torch.cuda.max_memory_allocated()/1e9,
                    alloc={'allocated_bytes.all.current':torch.cuda.memory_allocated(),'reserved_gb':torch.cuda.memory_reserved()/2**30},
                    accepted_training_exposure=dict(exposure),micro_batch=micro,accumulation=accumulation)
                metrics_acc.clear();guard.observe(util,elapsed_s=elapsed,data_wait_s=wait_s)
            if evaluation_due(applied,config):
                checkpoint('fixed250evaluation')
                evaluate_checkpoint(model,dev,contract['evaluation_panel'],prior,directory/f'eval-{applied:04d}',
                    step=applied,config=config,watchdog=watchdog,logger=logger)
                commit();mon.take()  # Never put evaluation time in training-util samples.
        checkpoint('fixed_endpoint_complete')
        logger.emit('complete',step=applied,applied_updates=applied,function_wall_s=time.monotonic()-function_start,
            no_best_checkpoint_selection=True)
        return dict(status='complete',applied_updates=applied,output=str(directory),function_wall_s=time.monotonic()-function_start)
    except BaseException as exc:
        logger.emit('aborted',step=applied,reason=f'{type(exc).__name__}:{exc}',applied_updates=applied)
        checkpoint('aborted')
        raise
    finally:
        watchdog.close()
        if mon is not None:mon.close()
        commit();logger.close()
