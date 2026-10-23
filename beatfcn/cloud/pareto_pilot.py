"""Bounded paired pilot on corrected real samples. Not a population-quality baseline."""
from beatfcn.cloud.train_app import app, image, data, DATA


@app.function(image=image, gpu='H100', volumes={DATA:data}, cpu=4, memory=49152, timeout=600,
              retries=0, max_containers=1, scaledown_window=2)
def run_pilot(batches, manifest_name, graph_compare=True, steps=100, run_tag='h100', wall_budget_s=580,
              compact_inner=False):
    import copy
    import gc
    import json
    import os
    import time
    from dataclasses import asdict
    from types import SimpleNamespace
    import torch
    from beatfcn.model.p1 import P1Config, P1Model
    from beatfcn.model.batch import attach_audio, prepare_inner_indices
    from beatfcn.model.evaluation import evaluate_songs
    from beatfcn.train.data import Corpus, load_manifest
    from beatfcn.train.graph import GraphedStep
    from beatfcn.train.loop import to_device, merge, summarize
    from beatfcn.train.pool import SongPool
    from beatfcn.cloud.gpu_util import GpuUtil
    started = time.perf_counter()
    if steps < 5:
        raise ValueError('need warmup, capture and replay steps')
    z = torch.load(f'{DATA}/{batches}', weights_only=False)
    if not z.get('rows_sha256') or z.get('static') is not False or z.get('manifest_name') != manifest_name:
        raise ValueError('pilot requires corrected dynamic metadata from this manifest')
    metas, taps = z['batches'], tuple(z['taps'])
    if len(metas) != 4:
        raise ValueError('pilot expects four fixed microbatches')
    cfg = P1Config(taps=taps, top_tap=max(taps), dropout=0., packed=False, selection=False,
                   compact_inner=compact_inner)
    corpus = Corpus(DATA, z['taps_key'], taps, pca=z['pca'])
    songs = sorted({s for b in metas for s in b['_audio']['keys']})
    pool = SongPool(corpus.load_numpy, taps, 'cuda', capacity=len(songs))
    out = {'schema':'beatfcn.pareto_pilot.v1','gpu':torch.cuda.get_device_name(),'torch':torch.__version__,
           'config':asdict(cfg),'rows_sha256':z['rows_sha256'],'manifest_name':manifest_name,
           'sample_sequence':[b['_provenance'] for b in metas], 'steps':steps,
           'protocol':'100 updates repeating the same 32 real cores; no claim about population quality',
           'variants':{}, 'quality_nonregression_established':False}
    root = f'{DATA}/runs/budget2-{run_tag}-20261023'
    os.makedirs(root, exist_ok=True)
    def save():
        out['function_wall_s_before_commit'] = time.perf_counter() - started
        with open(f'{root}/report.json','w') as f:
            json.dump(out,f,indent=1)
        data.commit()
        print(json.dumps({k:v for k,v in out.items() if k not in ('sample_sequence','config','variants')})[:2000],flush=True)
    mon = GpuUtil()
    try:
        pool.request(songs)
        full = []
        for meta in metas:
            bt = copy.deepcopy(meta)
            for key in ('_provenance','_sampling_audit'):
                bt.pop(key,None)
            if compact_inner:
                prepare_inner_indices(bt)
            attach_audio(bt,pool.get,'cuda',corpus.scale if corpus.quant else None)
            full.append(to_device(bt,torch.device('cuda')))
        pool.wait_idle()
        torch.cuda.synchronize()
        out['prepare_s'] = time.perf_counter()-started
        out['effective_batch'] = sum(int(b['mode'].shape[0]) for b in full)
        out['input_counts'] = {k:sum(int(b[k].sum()) for b in full)
                               for k in ('grp_valid','hold_valid','known_valid','refs_valid')}
        out['group_slots'] = sum(b['grp_valid'].numel() for b in full)
        out['note_slots'] = sum(b['note_type'].numel() for b in full)
        out['valid_notes'] = sum(int(b['grp_n'].sum()) for b in full)
        val_rows = load_manifest(DATA,split='val',require_offset_version=2,manifest_name=manifest_name)
        # Same fixed held-out selector in all three variants. Missing caches remain explicit failures.
        reference_state, reference_moments = None, None
        for label in (['eager','graph'] if graph_compare else ['eager']):
            if time.perf_counter()-started > wall_budget_s-80:
                out['budget_stop'] = f'insufficient time to start {label}'
                break
            variant_start = time.perf_counter()
            torch.manual_seed(20261023)
            model = P1Model(cfg).cuda().train()
            groups = [{'params':[p for p in model.parameters() if p.ndim>=2],'weight_decay':.01},
                      {'params':[p for p in model.parameters() if p.ndim<2],'weight_decay':0.}]
            optimizer = torch.optim.AdamW(groups,lr=torch.tensor(2e-4,device='cuda'),betas=(.9,.95),
                                          fused=True,capturable=True)
            runner = GraphedStep([SimpleNamespace(model=model,opt=optimizer)],1.,'cuda',capture=label=='graph',
                                 before_capture=pool.wait_idle)
            record = {'capture':label=='graph','completed_steps':0}
            out['variants'][label] = record
            measured_start = None
            first_fields = None
            mon.take()
            for step in range(steps):
                if time.perf_counter()-started >= wall_budget_s-70:
                    record['budget_stopped'] = True
                    break
                runner.fill_many(full,0.)
                mets,norms = runner.step()
                record['completed_steps'] = step+1
                if step == 0:
                    acc = {}
                    merge(acc,mets[0])
                    first_fields = summarize(acc)
                if step == 3:
                    torch.cuda.synchronize()
                    record['setup_and_four_steps_s'] = time.perf_counter()-variant_start
                    measured_start = time.perf_counter()
                if (step+1)%25 == 0:
                    print(json.dumps({'variant':label,'step':step+1,'wall_s':time.perf_counter()-started}),flush=True)
            torch.cuda.synchronize()
            record['training_wall_s'] = time.perf_counter()-variant_start
            record['steady_s'] = time.perf_counter()-measured_start if measured_start is not None else None
            record['steady_steps'] = max(record['completed_steps']-4,0)
            record['seconds_per_step'] = record['steady_s']/max(record['steady_steps'],1) if record['steady_s'] is not None else None
            record['gpu_util'] = mon.take()
            record['initial_fields'] = first_fields
            acc = {}
            merge(acc,mets[0])
            record['final_fields'] = summarize(acc)
            state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            moments = {k:{s:v.detach().cpu().clone() for s,v in optimizer.state[p].items() if torch.is_tensor(v)}
                       for k,p in model.named_parameters() if p in optimizer.state}
            if label=='eager':
                reference_state,reference_moments = state,moments
                if compact_inner:
                    previous = torch.load(f'{DATA}/runs/budget2-h100-20261023/eager.pt',weights_only=False)
                    record['vs_padded_max_abs_parameter_diff'] = max(float((state[k]-v).abs().max())
                                                                     for k,v in previous['model'].items())
                    record['vs_padded_same_steps'] = previous['step']==record['completed_steps']
                    del previous
            else:
                pdiff=max(float((state[k]-v).abs().max()) for k,v in reference_state.items())
                mdiff=max(float((moments[k][s]-v).abs().max()) for k,d in reference_moments.items() for s,v in d.items())
                record['equivalence']={'max_abs_parameter_diff':pdiff,'max_abs_optimizer_state_diff':mdiff,
                                       'threshold':1e-4,'ok':pdiff<1e-4 and mdiff<1e-4 and
                                       record['completed_steps']==out['variants']['eager']['completed_steps']}
            torch.save({'model':state,'config':asdict(cfg),'step':record['completed_steps'],
                        'rows_sha256':z['rows_sha256'],'protocol':out['protocol']},f'{root}/{label}.pt')
            save()
            remaining = wall_budget_s-(time.perf_counter()-started)
            if record['completed_steps']==steps and remaining>65:
                model.eval()
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    record['whole_songs'] = evaluate_songs(model,corpus,val_rows,n_songs=2,seed=20261023,
                                                           device='cuda',greedy=True,per_song_timeout_s=25.)
            else:
                record['whole_songs_skipped'] = 'time budget or incomplete training'
            record['variant_wall_s'] = time.perf_counter()-variant_start
            save()
            del runner,optimizer,model,state,moments
            gc.collect()
            torch.cuda.empty_cache()
    except Exception as exc:
        out['failure'] = f'{type(exc).__name__}:{exc}'
        save()
        raise
    finally:
        mon.close()
        pool.pool.shutdown(wait=True)
    out['function_wall_s'] = time.perf_counter()-started
    save()
    return out


@app.function(image=image, gpu='H100', volumes={DATA:data}, cpu=4, memory=49152, timeout=120,
              retries=0, max_containers=1, scaledown_window=2)
def graph_diagnostic(batches):
    """Separate eager variability, side-stream warmup, capture and replay on the same inputs."""
    import copy
    import json
    import time
    from types import SimpleNamespace
    import torch
    from beatfcn.model.p1 import P1Config, P1Model
    from beatfcn.model.batch import attach_audio
    from beatfcn.train.data import Corpus
    from beatfcn.train.graph import GraphedStep
    from beatfcn.train.loop import to_device
    from beatfcn.train.pool import SongPool
    started = time.perf_counter()
    z = torch.load(f'{DATA}/{batches}',weights_only=False)
    taps = tuple(z['taps'])
    corpus = Corpus(DATA,z['taps_key'],taps,pca=z['pca'])
    songs = sorted({s for b in z['batches'] for s in b['_audio']['keys']})
    pool = SongPool(corpus.load_numpy,taps,'cuda',capacity=len(songs))
    pool.request(songs)
    full = []
    for meta in z['batches']:
        bt = copy.deepcopy(meta)
        bt.pop('_provenance',None)
        bt.pop('_sampling_audit',None)
        attach_audio(bt,pool.get,'cuda',corpus.scale if corpus.quant else None)
        full.append(to_device(bt,torch.device('cuda')))
    pool.wait_idle()
    cfg = P1Config(taps=taps,top_tap=max(taps),dropout=0.)
    runs, runners = [], []
    for capture in (False,True,False):
        torch.manual_seed(20261023)
        model = P1Model(cfg).cuda().train()
        groups = [{'params':[p for p in model.parameters() if p.ndim>=2],'weight_decay':.01},
                  {'params':[p for p in model.parameters() if p.ndim<2],'weight_decay':0.}]
        optimizer = torch.optim.AdamW(groups,lr=torch.tensor(2e-4,device='cuda'),betas=(.9,.95),
                                      fused=True,capturable=True)
        run = SimpleNamespace(model=model,opt=optimizer)
        runs.append(run)
        runners.append(GraphedStep([run],1.,'cuda',capture=capture,before_capture=pool.wait_idle))
    def diff(a,b):
        return float(torch.stack([(p-q).abs().max() for p,q in zip(a.parameters(),b.parameters())]).max())
    out = {'rows_sha256':z['rows_sha256'],'steps':[],
           'initial_diff':diff(runs[0].model,runs[1].model)}
    try:
        for i in range(10):
            fields, norms = [], []
            for runner in runners:
                runner.fill_many(full,0.)
                met, gn = runner.step()
                # Copy now: the next graph replay overwrites its static output storage.
                fields.append({k:float(v[0]) for k,v in met[0].items() if k!='_illegal'})
                norms.append(float(gn[0]))
            record = {'step':i+1,'phase':'warmup' if i<3 else ('capture' if i==3 else 'replay'),
                      'eager_graph_max_abs':diff(runs[0].model,runs[1].model),
                      'eager_eager_max_abs':diff(runs[0].model,runs[2].model),
                      'losses':fields,'grad_norms':norms,
                      'optimizer_steps':[[float(v['step']) for v in r.opt.state.values()][:1] for r in runs]}
            out['steps'].append(record)
            print(json.dumps(record),flush=True)
        out['function_wall_s'] = time.perf_counter()-started
        with open(f'{DATA}/bench/offset_v2_graph_diagnostic.json','w') as f:
            json.dump(out,f,indent=1)
        data.commit()
        return out
    finally:
        pool.pool.shutdown(wait=True)
