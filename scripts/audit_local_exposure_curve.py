"""D89 zero-update exposure/conditional-field audit of the original baseline."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import resource
import time

import numpy as np
import torch
import torch.nn.functional as F

from beatfcn.events.canonical import TYPES
from beatfcn.model.full_event_codec import FIELDS, EventVocabulary
from beatfcn.model.full_event_model import FullEventConfig, FullEventModel
from beatfcn.train.event_conditions import encode_controls
from beatfcn.train.local_request_panel import LocalRequestPanel
from beatfcn.train.local_shapes import pad_input_shapes
from scripts.audit_relative_teacher_fields import summarize
from scripts.run_local_song_baseline import sha, write_json

ROOT = Path(__file__).resolve().parents[1]


@torch.no_grad()
def run(path):
    plan = json.loads(path.read_text())
    output = ROOT/plan['output']
    if output.exists():
        raise FileExistsError(output)
    started = time.monotonic()
    report = dict(id=plan['id'], complete=False, optimizer_updates=0, new_cloud_compute_calls=0,
        plan_sha256=sha(path), exposure=[], rows=[], summary=[], quality_passed=False,
        scope='Fixed train-song and heldout-song true-prefix diagnostic. Unknown controls on both sides; original teacher carry. Not free-generation quality or a new candidate.')
    def guard():
        if time.monotonic()-started > plan['max_wall_s']:
            raise TimeoutError('Exposure audit exceeded frozen deadline')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > plan['max_rss_bytes']:
            raise MemoryError('Exposure audit exceeded frozen macOS RSS limit')
    try:
        for p,h in plan['source_sha256'].items():
            if sha(ROOT/p) != h:
                raise ValueError('Audit input changed: '+p)
        source = json.loads((ROOT/plan['quality_plan']).read_text())
        panel = LocalRequestPanel(ROOT, source['tape'], source['features'])
        exposure = json.loads((ROOT/source['exposure']).read_text())
        selected = plan['train_indices'] + plan['validation_indices']
        if len({panel.records[i]['component'] for i in selected}) != 129:
            raise ValueError('Expected 129 distinct song components')
        paths = {}
        for i in selected:
            row=panel.records[i]; chart=panel.charts[row['chart_index']]
            paths.update(panel.features.records[row['audio_sha256']]['sha256'])
            paths.update({chart['token_path']:chart['token_sha256'],chart['target_path']:chart['target_sha256']})
        for p,h in paths.items():
            guard()
            if sha(ROOT/p) != h:
                raise ValueError('Admitted feature or target changed: '+p)
        torch.set_num_threads(plan['cpu_threads'])
        for step in plan['steps']:
            guard()
            counts = Counter(i for x in exposure['steps'][:step] for i in x['indices'])
            views = Counter((i,tuple(fields)) for x in exposure['steps'][:step]
                for i,fields in zip(x['indices'],x['specified_condition_fields'],strict=True))
            by_mode={}
            for mode in ('phira','mania_4k'):
                c={i:n for i,n in counts.items()if panel.records[i]['mode']==mode}
                pool=[i for i,r in enumerate(panel.records)if r['split']=='train' and r['mode']==mode]
                by_mode[mode]=dict(requests=sum(c.values()), distinct_requests=len(c),
                    total_pool_requests=len(pool), covered_pool_fraction=len(c)/len(pool),
                    exposure_count_distribution=dict(Counter(c.values())),
                    distinct_song_components=len({panel.records[i]['component'] for i in c}),
                    distinct_charts=len({panel.records[i]['chart_id'] for i in c}),
                    valid_tokens=sum(panel.records[i]['tokens']*n for i,n in c.items()),
                    empty_requests=sum(n for i,n in c.items()if panel.records[i]['event_count']==0))
            report['exposure'].append(dict(step=step, by_mode=by_mode,distinct_request_condition_views=len(views)))
            checkpoint=ROOT/source['checkpoint_dir']/f'step-{step:04d}.pt'
            before=sha(checkpoint)
            saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
            if saved['updates']!=step or saved['plan_sha256']!=sha(ROOT/plan['quality_plan']):
                raise ValueError('Wrong original-baseline checkpoint')
            model=FullEventModel(FullEventConfig(**saved['config'])).eval()
            model.load_state_dict(saved['model']);del saved
            for i in selected:
                guard()
                row=panel.records[i];q=row['request'];a,b=q['start_us'],q['end_us']
                controls=encode_controls(row['mode'],{},(a,b),origin_us=a)
                inputs,seq=panel.batch([i],controls=[controls])
                logits=model(pad_input_shapes(inputs,source['input_shapes']),seq)[0]
                valid=seq['valid'][0];target=seq['targets'][0];allowed=seq['allowed'][0]
                assert torch.isfinite(logits[valid]).all()
                assert allowed.gather(-1,target[:,None])[valid].all()
                masked=logits.masked_fill(~allowed,-1e9)
                loss=F.cross_entropy(masked,target,reduction='none')
                pred=masked.argmax(-1);correct=pred==target;choice=allowed.sum(-1)>1
                stats={name:summarize(valid & (seq['fields'][0]==j),loss,correct,choice)
                    for j,name in enumerate(FIELDS)}
                vocab=EventVocabulary()
                kinds={kind:summarize(valid & (target==vocab.token('kind',j)),loss,correct,choice)
                    for j,kind in enumerate(TYPES)}
                report['rows'].append(dict(step=step,record_index=i,split=row['split'],mode=row['mode'],
                    component=row['component'],chart_id=row['chart_id'],stratum=row['stratum'],
                    scope_us=[a,b],actual_window_exposures=counts[i],
                    exact_unknown_condition_exposures=views[(i,())],target_notes=row['event_count'],
                    checkpoint_sha256=before,request_ce=float(loss[valid].mean()),fields=stats,kinds=kinds,
                    bos_target=int(target[0]),bos_prediction=int(pred[0]),bos_end_probability=float(masked[0].softmax(-1)[2])))
                if len(report['rows'])%20==0:write_json(output,report)
            assert sha(checkpoint)==before
            del model
            for split in ('train','val'):
                for mode in ('phira','mania_4k'):
                    rows=[r for r in report['rows']if (r['step'],r['split'],r['mode'])==(step,split,mode)]
                    fields={}
                    for field in FIELDS:
                        vals=[r['fields'][field]for r in rows];n=sum(v['tokens']for v in vals);c=sum(v['choice_tokens']for v in vals)
                        fields[field]=dict(tokens=n,choice_tokens=c,
                            choice_accuracy=sum(v['choice_correct']for v in vals)/c if c else None,
                            ce=sum(v['ce_sum']for v in vals)/n if n else None)
                    report['summary'].append(dict(step=step,split=split,mode=mode,songs=len(rows),
                        request_ce_mean=float(np.mean([r['request_ce']for r in rows])),
                        window_exposure_median=float(np.median([r['actual_window_exposures']for r in rows])),
                        never_exposed_windows=sum(r['actual_window_exposures']==0 for r in rows),
                        exact_unknown_condition_exposures=sum(r['exact_unknown_condition_exposures']for r in rows),
                        fields=fields))
            print(json.dumps(dict(step=step,rows=len(report['rows']),wall_s=time.monotonic()-started)),flush=True)
            write_json(output,report)
        report.update(complete=True,wall_s=time.monotonic()-started,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,checkpoint_unchanged=True)
        write_json(output,report)
    except Exception as exc:
        report.update(error=f'{type(exc).__name__}: {exc}',wall_s=time.monotonic()-started)
        write_json(output,report)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True)
    run(parser.parse_args().plan)
