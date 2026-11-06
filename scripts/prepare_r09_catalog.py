"""Reconstruct every positive pair and audit prospective validation before inference."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import torch

from beatfcn.model import features as F
from beatfcn.model.batch import MAX_NOTES, SongInfo
from beatfcn.train.condition_pairs import blank_context_excerpt, assert_same_generation_inputs
from beatfcn.train.data import chart_from_canon, model_shift_us, dup_only, check_stacks
from beatfcn.train.positive_pairs import matched_schedules

ROOT=Path(__file__).resolve().parents[1]


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    started=time.perf_counter()
    plan_path=ROOT/'bench/rhythm-v2/r09-positive-pairs-prepare-v1.json'
    plan=json.loads(plan_path.read_text())
    for path,expected in plan['input_sha256'].items():
        if sha(ROOT/path)!=expected: raise ValueError(f'frozen input changed: {path}')
    receipt=json.loads((ROOT/'reports/r09_validation_canon_transfer.json').read_text())
    if not receipt['complete'] or receipt['plan_sha256']!=sha(plan_path): raise ValueError('canonical transfer incomplete')
    validation_hashes={r['chart_id']:r['sha256'] for r in receipt['files']}
    rows={r['chart_id']:r for r in plan['train_rows']}
    rows.update({x['row']['chart_id']:x['row'] for x in plan['validation_rows']})
    charts,canonical_hashes={},{}
    for cid,row in rows.items():
        train=row['split']=='train'
        path=ROOT/plan['train_canonical_root' if train else 'validation_canonical_root']/row['source']/f'{row["native_id"]}.json'
        expected=(plan['train_canonical_sha256'] if train else validation_hashes)[cid]
        if sha(path)!=expected: raise ValueError(f'canonical changed: {cid}')
        doc=json.loads(path.read_text())
        if any(doc[k]!=row[k] for k in ('chart_id','audio_sha256','mode')): raise ValueError('source identity differs')
        chart=chart_from_canon(doc,model_shift_us(row,True))
        if dup_only(row): check_stacks(chart.events)
        charts[cid]=chart;canonical_hashes[cid]=expected

    def build(cid,c0):
        row,chart=rows[cid],charts[cid]
        duration=round(row['audio_sec']*F.SEC)
        sample=blank_context_excerpt(chart,c0,duration,row['audio_sha256'],20261025)
        ex=F.build_core(sample,chart.events,c0,duration,stage_a_arm='legacy')
        bad=F.check_targets(ex)
        if bad or any(len(g['notes'])>MAX_NOTES for g in ex.groups):
            raise ValueError(f'illegal or over-capacity target: {bad}')
        rhythm=[(g['t_rel_us'],sorted((n['type'],n['lane'] if row['mode']=='mania_4k' else -1,
            n.get('end_rel_us',0)) for n in g['notes'])) for g in ex.groups]
        target_sha=hashlib.sha256(json.dumps(rhythm).encode()).hexdigest()
        # Audio lengths here are metadata estimates for pair isolation only.
        # Actual feature headers and tensors must be checked before model use.
        ncells=(duration+F.CELL_US-1)//F.CELL_US
        info=SongInfo(row['audio_sha256'],ncells,(ncells+49)//50)
        return dict(row=row,request=sample.request,raw_controls=sample.raw_controls,
            truth_events=[e.to_json() for g in sample.model_view.target_groups for e in g.events],
            example=ex,target_sha256=target_sha,info=info,
            counts=dict(Counter(e.type for e in sample.hidden)))

    pairs=[]
    for index,source in enumerate(plan['train_pairs']):
        entries=[build(cid,source['c0_us']) for cid in source['chart_ids']]
        for entry,expected in zip(entries,source['items']):
            if entry['target_sha256']!=expected['rhythm_target_sha256']:
                raise ValueError('D09 target reconstruction differs')
            if entry['request']['controls']!=expected['controls']:
                raise ValueError('D09 condition labels differ')
        isolation=assert_same_generation_inputs(entries[0]['example'],entries[1]['example'],
            entries[0]['info'],(9,16,24),24,1024,1024)
        if set(isolation['changed_fields'])!=set(source['changed_fields']): raise ValueError('changed fields differ')
        pairs.append(dict(pair_id=f'r09-pair-{index:04d}',mode=source['mode'],
            audio_sha256=source['audio_sha256'],c0_us=source['c0_us'],
            changed_fields=source['changed_fields'],entries=entries,isolation=isolation))
    schedule=matched_schedules(pairs,steps=1000,seed=20261025)
    validation=[];rejected=[];coverage=[]
    for selected in plan['validation_rows']:
        row=selected['row'];cid=row['chart_id'];duration=round(row['audio_sec']*F.SEC)
        accepted=[]
        for c0 in range(0,duration-F.CORE_US+1,F.CORE_US):
            try: accepted.append(build(cid,c0))
            except (ValueError,AssertionError,F.SkipExample) as exc:
                rejected.append(dict(chart_id=cid,c0_us=c0,error=f'{type(exc).__name__}: {exc}'))
        accepted.sort(key=lambda e:hashlib.sha256(f'r09-val-window:{row["audio_sha256"]}:{e["example"].c0_us}'.encode()).hexdigest())
        chosen=accepted[:3]
        coverage.append(dict(chart_id=cid,mode=row['mode'],bucket=selected['selection_bucket'],
            legal_windows=len(accepted),selected_windows=len(chosen)))
        for entry in chosen:
            entry['selection_bucket']=selected['selection_bucket']
            entry['condition_source']='Target-derived fields are audit labels only; evaluation requests must replace them before inference.'
            validation.append(entry)
    package=dict(schema='beatfcn.r09_positive_pair_catalog.v1',plan_sha256=sha(plan_path),
        train_pairs=pairs,validation_candidates=validation,schedules=schedule)
    output=ROOT/'inputs/r09_positive_pairs/catalog.pt'
    if output.exists(): raise FileExistsError('catalog already exists; inspect before replacing')
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_suffix('.partial')
    torch.save(package,temporary);temporary.replace(output)
    fields={}
    for mode in ('mania_4k','phira'):
        for field in sorted({f for p in pairs if p['mode']==mode for f in p['changed_fields']}):
            subset=[p for p in pairs if p['mode']==mode and field in p['changed_fields']]
            fields[f'{mode}:{field}']=dict(pair_windows=len(subset),independent_audio=len({p['audio_sha256'] for p in subset}))
    report=dict(schema='beatfcn.r09_positive_pair_catalog_audit.v1',complete=True,
        plan_sha256=sha(plan_path),catalog=str(output.relative_to(ROOT)),catalog_sha256=sha(output),
        canonical_sha256=canonical_hashes,train_pairs=len(pairs),train_audio=len({p['audio_sha256'] for p in pairs}),
        train_target_cores=2*len(pairs),all_targets_match_prior_audit=True,
        all_initial_noncondition_metadata_equal=True,field_support=fields,
        schedule=dict(steps_per_arm=1000,cores_per_update=2,cores_per_arm=2000,
            matching_exposure_blocks=schedule['blocks'],exact_global_and_100_update_exposure=True),
        validation_coverage=coverage,validation_windows=len(validation),validation_rejections=rejected,
        missing_validation_cells=plan['missing_validation_cells'],
        metadata_ready=all(r['selected_windows']==3 for r in coverage),
        actual_audio_tensor_probe_completed=False,trained_model_baseline_measured=False,
        training_started=False,model_promotion=False,wall_s=time.perf_counter()-started)
    (ROOT/'reports/r09_catalog_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('complete','train_pairs','train_audio','validation_windows','metadata_ready','wall_s')}))


if __name__=='__main__': main()
