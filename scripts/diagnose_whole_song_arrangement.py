"""Frozen-weight mechanism audit, never a training or quality-selection run."""
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from beatfcn.model.arrangement_iohmm import forward_log_probability, prior_path
from beatfcn.model.canonical_event_package import package_support
from beatfcn.model.local_event_package import GRU_WIDTH
from beatfcn.model.local_typed_canvas import FRAMES
from beatfcn.train.canonical_probability import explicit_row
from beatfcn.train.whole_song_mel import WholeMelStore
from scripts.run_frontend_canvas_comparison import setup
from scripts.run_local_portfolio_scale import row_factory, write, weight_digest
from scripts.run_local_typed_canvas import score
from scripts.run_whole_song_arrangement import fresh


ROOT = Path(__file__).resolve().parents[1]
IDENT = 'local-whole-song-mechanism-v2'
PLAN = ROOT / f'bench/rhythm-v2/{IDENT}.json'
REPORT = ROOT / 'reports/local_whole_song_mechanism_v2.json'
ART = ROOT / 'artifacts' / IDENT


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def first_components(panel, indices, number):
    seen = set(); result = []
    for i in indices:
        c = panel.records[i]['component']
        if c not in seen:
            seen.add(c); result.append(i)
        if len(result) == number:
            break
    return result


def marginal(initial, transitions, emissions):
    alpha = [initial + emissions[0]]
    for j in range(1, len(emissions)):
        alpha.append(emissions[j] + torch.logsumexp(alpha[-1][:, None] + transitions[j], 0))
    z = alpha[-1].logsumexp(0)
    beta = [torch.zeros_like(initial) for _ in emissions]
    for j in range(len(emissions)-2, -1, -1):
        beta[j] = torch.logsumexp(transitions[j+1] + emissions[j+1][None] + beta[j+1][None], 1)
    q = torch.stack([(a+b-z).exp() for a, b in zip(alpha, beta)])
    return z, q


def viterbi(initial, transition, emission):
    a = initial + emission[0]; back = []
    for j in range(1, len(emission)):
        values = a[:, None] + transition[j]
        best, previous = values.max(0)
        a = best + emission[j]; back.append(previous)
    out = [int(a.argmax())]
    for previous in reversed(back):
        out.append(int(previous[out[-1]]))
    return np.asarray(out[::-1])


def teacher(model, inputs, target, request, output=None, detailed=False):
    output = model(inputs) if output is None else output
    support = package_support(inputs, request)[0]
    count = target['count'][0]
    block = (request.start_us + torch.arange(FRAMES)*40000)//2000000
    valid = inputs['canvas_valid'][0]
    denom = support.sum().clamp_min(1).float()
    emissions = []; parts = []; head_details = []; tail_details = []
    for state in range(4):
        per = -output['count_log_probs'][state].gather(-1, count[..., None]).squeeze(-1)
        per = (per*support).sum(-1)
        attr = per * 0
        onset_total = per.sum()*0; tail_total = per.sum()*0
        for frame, kind in ((count > 0) & support).nonzero().tolist():
            n = int(count[frame, kind]); logits = []; conditions = []
            hook = model.onset_read.register_forward_hook(lambda _, args, value: logits.append(value)) if detailed else None
            thook = model.tail_condition.register_forward_hook(lambda _, args, value: conditions.append(value)) if detailed else None
            try:
                members, onset, tail = model.ordered_members(output['x'][state], frame, kind, n, inputs, request,
                    teacher={k: target[k][0, frame, kind, :n] for k in ('offset', 'tail_us')})
            finally:
                if hook: hook.remove()
                if thook: thook.remove()
            attr = attr.index_add(0, torch.tensor([frame]), (onset+tail)[None])
            onset_total = onset_total + onset; tail_total = tail_total + tail
            if detailed:
                previous_head = previous_end = None
                for slot, member in enumerate(members):
                    allowed = inputs['canvas_head_allowed'][0, frame].clone()
                    heads = inputs['canvas_head_us'][0, frame]
                    if previous_head is not None: allowed &= heads >= previous_head
                    if kind == 1: allowed &= heads < request.duration_us//1000*1000
                    lp = logits[slot].masked_fill(~allowed, -1e9).log_softmax(-1)
                    label = int(target['offset'][0, frame, kind, slot])
                    predicted = int(lp.argmax())
                    p = lp.exp()
                    head_details.append(dict(state=state, kind=kind, frame=frame, count=n,
                        nll=float(-lp[label].detach()), uniform_nll=math.log(int(allowed.sum())),
                        entropy=float(-(p[allowed]*lp[allowed]).sum().detach()),
                        correct=predicted == label, error_ms=(int(heads[predicted])-member.onset_us)/1000))
                    if kind == 1:
                        minimum = max(member.onset_us+1, previous_end if previous_head == member.onset_us else member.onset_us+1)
                        best, _, _ = model.tail_law(conditions[slot], request, request.start_us+frame*40000, minimum, greedy=True)
                        legal = (request.duration_us//1000 - (minimum+999)//1000 + 1)
                        legal += sum(t % 1000 != 0 and minimum <= t <= request.duration_us for t in set(request.copy_times))
                        tail_details.append(dict(state=state, kind=kind, count=n,
                            true_duration_ms=(member.end_us-member.onset_us)/1000,
                            teacher_head_tail_error_ms=(best-member.end_us)/1000,
                            uniform_nll=math.log(max(1,legal))))
                    previous_head, previous_end = member.onset_us, member.end_us
        emission = per.new_zeros(len(output['transition'])).index_add(0, block[valid], -(per+attr)[valid])
        emissions.append(emission)
        parts.append(dict(count=float(per.sum().detach()/denom), onset=float(onset_total.detach()/denom),
                          tail=float(tail_total.detach()/denom)))
    emission = torch.stack(emissions, -1)
    z, q = marginal(output['initial'], output['transition'], emission)
    _, prior = marginal(output['initial'], output['transition'], torch.zeros_like(emission))
    posterior_kl = (q*emission).sum() - z
    active = torch.unique(block[valid])
    probs = output['count_log_probs'].exp()
    mixture = probs.mean(0)
    js = (probs * (output['count_log_probs']-mixture.clamp_min(1e-30).log())).sum(-1).mean()
    stats = dict(nll=float((-z/denom).detach()), groups=int(denom), events=int(count.sum()),
        state_parts=parts, posterior_joint_kl_nats=float(posterior_kl.detach()),
        posterior_active_entropy=float((-(q[active]*q[active].clamp_min(1e-30).log()).sum(-1)).mean().detach()),
        prior_active_entropy=float((-(prior[active]*prior[active].clamp_min(1e-30).log()).sum(-1)).mean().detach()),
        active_prior=prior[active].detach().tolist(), active_posterior=q[active].detach().tolist(),
        count_state_JS_nats=float(js.detach()), state_count_mean=[float((p*torch.arange(17)).sum(-1)[support].sum().detach()) for p in probs],
        head_details=head_details, tail_details=tail_details)
    return -z/denom, stats, emission, output


@torch.no_grad()
def decode(model, output, inputs, request, seed, *, override_path=None, counts_override=None, greedy_members=False):
    rng = np.random.default_rng(seed)
    original_path = prior_path(output['initial'], output['transition'], rng=rng)
    path = original_path if override_path is None else override_path
    support = package_support(inputs, request)[0]; addresses = support.nonzero().tolist()
    block = (request.start_us+np.arange(FRAMES)*40000)//2000000
    probabilities = np.asarray([output['count_log_probs'][path[block[f]],f,k].exp().double().numpy() for f,k in addresses])
    sampled = np.asarray([rng.choice(17,p=p/p.sum()) for p in probabilities])
    counts = sampled if counts_override is None else np.asarray(counts_override)
    events = []
    for (frame,kind), n in zip(addresses,counts):
        if n:
            members,_,_ = model.ordered_members(output['x'][path[block[frame]]],frame,kind,int(n),inputs,request,
                rng=None if greedy_members else rng,greedy=greedy_members)
            events.extend(members)
    return sorted(events), dict(path=path.tolist(),counts=counts.tolist())


def compact_scores(rows):
    components = {}
    for r in rows:
        components.setdefault(r['component'], []).append(r)
    by = {c: {k:float(np.mean([v['scores'][k] for v in values])) for k in ('full_note_f1','onset_f1')}
          for c,values in components.items()}
    return dict(full_note_f1=float(np.mean([v['full_note_f1'] for v in by.values()])),
        onset50=float(np.mean([v['onset_f1'] for v in by.values()])),
        onset10=float(np.mean([r['scores']['onset_10ms']['f1'] for r in rows])),
        predicted=sum(len(r['events']) for r in rows),target=sum(r['scores']['true'] for r in rows),components=by)


def main():
    if REPORT.exists() or PLAN.exists(): raise FileExistsError('No automatic diagnostic rerun')
    torch.set_num_threads(2)
    original_path = ROOT/'bench/rhythm-v2/local-whole-song-arrangement-v1.json'
    original = json.loads(original_path.read_text()); old = json.loads((ROOT/'reports/local_whole_song_arrangement_v1.json').read_text())
    drift = {}
    for name,digest in original['source_sha256'].items():
        if sha(ROOT/name) != digest:
            # A later collator added an unused exact-ms auxiliary target. Use
            # the original byte-verified data source for this old model, rather
            # than silently assuming the current collator is equivalent.
            if name != 'beatfcn/train/canonical_probability.py':
                raise ValueError('Scientific source drift: '+name)
            snapshot=ROOT/'artifacts/local-whole-song-arrangement-v1/source'/name
            if sha(snapshot)!=digest:raise ValueError('Original data snapshot drift')
            spec=importlib.util.spec_from_file_location('beatfcn.train._arrangement_frozen_probability',snapshot)
            frozen=importlib.util.module_from_spec(spec);spec.loader.exec_module(frozen)
            import beatfcn.train.whole_song_mel as store_module
            store_module.probability_batch=frozen.probability_batch
            drift[name]=dict(original_sha256=digest,current_sha256=sha(ROOT/name),
                action='Load original snapshot into this diagnostic process only; unchanged files on disk',
                difference='Current code adds target tail_ms_native, unused by the old arrangement model')
    weight = ROOT/'artifacts/local-whole-song-arrangement-v1/quality-1000.pt'
    if sha(weight) != old['final_weight_sha256']: raise ValueError('Weight identity drift')
    panel, features = setup(json.loads((ROOT/'bench/rhythm-v2/local-frontend-comparison-v2.json').read_text()))
    factory = row_factory(panel, features)
    store = WholeMelStore(ROOT,panel.features.records,ROOT/'artifacts/local-whole-song-arrangement-v1/whole-mel')
    train = first_components(panel,original['indices']['train64'],8)
    dev = first_components(panel,original['indices']['dev'],16)
    checkpoints = [0,250,500,1000]
    variants = ['baseline','remote_zero','whole_context_zero','fine_zero','all_audio_zero','state_film_zero',
                'fixed_counts_greedy_members','oracle_posterior_state']
    plan = dict(id=IDENT,created_at=datetime.now(timezone.utc).isoformat(),optimizer_updates=0,device='cpu',threads=2,
        wall_s=1800,checkpoints=checkpoints,train_indices=train,dev_indices=dev,variants=variants,
        primary_unchanged=True,oracle_excluded_from_free_generation=True,
        controls='Same index/request/reference/control/clock/mask/native support and seed. Route-zero OOD interventions are not silence. Remotezero preserves whole-summary blocks touching the observedlocal16s. Filmzero changes only the4state-FiLM embedding in memory and restores it. Fixed-count membergreedy holds originalbaseline path/counts. Oraclepath useshiddenlabels only in explicitdiagnostics.',
        original_plan_sha256=sha(original_path),original_report_sha256=sha(ROOT/'reports/local_whole_song_arrangement_v1.json'),
        source_sha256=sha(Path(__file__)),checkpoint_sha256={str(n):sha(weight.parent/f'quality-{n:04d}.pt') for n in checkpoints if n},
        source_drift_resolved=drift,
        mechanisms=['State-emitter collapse versus prior/posterior mismatch','Diffuse conditionaltime/tail laws versus count failure',
                    'Remotecontext ignored versus localacoustic path failure','Fixed-train/dev convergence versus unsupported capacity claim'],cloud_compute_calls=0)
    write(PLAN,plan); ART.mkdir(parents=True,exist_ok=True)
    result=dict(id=IDENT,status='running',plan_sha256=sha(PLAN),checkpoints={},counterfactuals={},gradients=[],failures=[],optimizer_updates=0,cloud_compute_calls=0)
    write(REPORT,result); started=time.monotonic()
    def bound():
        if time.monotonic()-started > plan['wall_s']: raise TimeoutError('Frozen diagnostic wall')
    def example(index):
        e=explicit_row(factory(index,'draft')); inp,tgt=store.batch(e,panel.records[index]['audio_sha256'],device='cpu')
        return e,inp,tgt
    try:
        for step in checkpoints:
            model=fresh(original,'cpu').eval()
            if step:
                saved=torch.load(weight.parent/f'quality-{step:04d}.pt',map_location='cpu',weights_only=False)
                if saved['updates']!=step:raise ValueError('Checkpoint stage identity')
                model.load_state_dict(saved['model']);del saved
            elif weight_digest(model)!=old['initial_weight_sha256']:raise ValueError('Initial identity')
            rows=[]
            with torch.no_grad():
                for split,indices in [('train',train),('dev',dev)]:
                    for index in indices:
                        bound();e,inp,tgt=example(index)
                        _,stats,_,_=teacher(model,inp,tgt,e['request'],detailed=step==1000)
                        rows.append(dict(index=index,component=panel.records[index]['component'],split=split,**stats))
            result['checkpoints'][str(step)]=rows;write(REPORT,result)
            print(json.dumps(dict(stage='fixed_teacher',checkpoint=step,requests=len(rows),wall_s=time.monotonic()-started)),flush=True)
        # The terminal model is still loaded, with no health reuse or optimizer.
        before=weight_digest(model);raw={v:[] for v in variants};replays=[]
        original_raw={r['index']:r for r in json.loads((weight.parent/'draft-sample0.json').read_text())}
        for index in dev:
            bound();e,inp,tgt=example(index);seed=original['seed']+17*index
            with torch.no_grad():
                output=model(inp);base,decision=decode(model,output,inp,e['request'],seed)
                expected=original_raw[index]['prediction']['events']
                replays.append(dict(index=index,events_exact=[asdict(v) for v in base]==expected))
                _,_,emission,_=teacher(model,inp,tgt,e['request'],output=output)
                oracle=viterbi(output['initial'],output['transition'],emission)
                for variant in variants:
                    bound();used=inp;out=output;override=None;counts=None;greedy=False
                    if variant in ('remote_zero','whole_context_zero','all_audio_zero'):
                        changed=inp['whole_song'].clone()
                        if variant=='remote_zero':
                            t=inp['whole_song_times'][0];fixed=(t+1>-4)&(t-1<12);changed[:,~fixed]=0
                        else:changed.zero_()
                        used={**used,'whole_song':changed}
                    if variant in ('fine_zero','all_audio_zero'):used={**used,'mel_flux':torch.zeros_like(inp['mel_flux'])}
                    if variant in ('remote_zero','whole_context_zero','fine_zero','all_audio_zero'):out=model(used)
                    if variant=='state_film_zero':
                        saved=model.state_film.weight.detach().clone()
                        try:model.state_film.weight.zero_();out=model(inp)
                        finally:model.state_film.weight.copy_(saved)
                    if variant=='fixed_counts_greedy_members':counts=decision['counts'];override=np.asarray(decision['path']);greedy=True
                    if variant=='oracle_posterior_state':override=oracle
                    pred,_=decode(model,out,used,e['request'],seed,override_path=override,counts_override=counts,greedy_members=greedy)
                    raw[variant].append(dict(index=index,component=panel.records[index]['component'],oracle=variant.startswith('oracle'),
                        events=[asdict(v) for v in pred],scores=score(pred,e['targets']),
                        max_count_probability_delta=float((out['count_log_probs'].exp()-output['count_log_probs'].exp()).abs().max())))
            write(ART/'counterfactuals.json',raw);write(ART/'cpu-replay.json',replays)
        result['counterfactuals']={v:compact_scores(rows) for v,rows in raw.items()}
        result['cpu_replay_exact']=sum(r['events_exact'] for r in replays)
        result['cpu_replay_requests']=len(replays)
        # True-label gradients are diagnostic only and perform no update.
        for index in train[:3]+dev[:3]:
            bound();model.zero_grad(set_to_none=True);e,inp,tgt=example(index)
            loss,_,_,_=teacher(model,inp,tgt,e['request']);loss.backward()
            groups={}
            for name,p in model.named_parameters():
                if name.startswith(('initial_read','transition_read','transition_bias')):key='state_prior'
                elif name.startswith('state_film'):key='state_film'
                elif name.startswith(('song_input','song_blocks','context_read')):key='whole_context'
                elif name.startswith(('fine_projection','fine_blocks','frame_read')):key='fine_audio'
                elif name.startswith(('count_gate','count_positive')):key='count'
                else:key='members_and_other'
                row=groups.setdefault(key,dict(squared=0.,parameters=0,nonzero_parameters=0))
                row['parameters']+=p.numel()
                if p.grad is not None:
                    row['squared']+=float(p.grad.square().sum());row['nonzero_parameters']+=int((p.grad!=0).sum())
            for row in groups.values():
                row['l2']=math.sqrt(row.pop('squared'));row['rms']=row['l2']/math.sqrt(row['parameters'])
            result['gradients'].append(dict(index=index,loss=float(loss.detach()),groups=groups))
        model.zero_grad(set_to_none=True)
        if weight_digest(model)!=before or sha(weight)!=old['final_weight_sha256']:raise ValueError('Intervention mutated terminal weights')
        result.update(status='completed',wall_s=time.monotonic()-started,weights_unchanged=True,
            raw_sha256={p.name:sha(p) for p in ART.glob('*.json')})
        write(REPORT,result);print(json.dumps(dict(status='completed',wall_s=result['wall_s'],counterfactuals=result['counterfactuals'])),flush=True)
    except BaseException as exc:
        result.update(status='stopped',wall_s=time.monotonic()-started);result['failures'].append(f'{type(exc).__name__}: {exc}')
        write(REPORT,result);raise


if __name__=='__main__':main()
