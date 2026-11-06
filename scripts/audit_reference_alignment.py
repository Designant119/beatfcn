"""D14: independently inspect reference clock uncertainty, not new model scores."""
import asyncio
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import modal
import numpy as np
from beatfcn.events.canonical import Event
from beatfcn.ingest.offset import flux_envelope, _peaks, agree_counts
from beatfcn.model.edit_utility import edit_utility

ROOT=Path(__file__).resolve().parents[1]


async def main():
    plan_path=ROOT/'bench/rhythm-v2/d14-reference-clock-audit-v1.json'
    plan=json.loads(plan_path.read_text())
    paths=dict(panel=ROOT/'reports/r03_window_raw/panel.json',candidate=ROOT/'reports/r07b_evaluate.json',
               control=ROOT/'reports/r03_window_raw/r03-window-all.benchmark.json')
    for key,path in paths.items():
        if hashlib.sha256(path.read_bytes()).hexdigest()!=plan[key+'_sha256']:
            raise ValueError('registered artifact changed')
    panel=json.loads(paths['panel'].read_text())
    records={name:{r['case_id']:r for r in json.loads(paths[name].read_text())['records'] if r['field'] is None}
             for name in ('candidate','control')}
    cache=ROOT/'inputs/d14_alignment/mel10';cache.mkdir(parents=True,exist_ok=True)
    volume=modal.Volume.from_name('beatfcn-data',create_if_missing=False)
    receipt=dict(started_at_unix=time.time(),transferred_bytes=0,complete=False,files=[],new_remote_compute=False)
    receipt_path=ROOT/'reports/d14_read_receipt.json'
    semaphore=asyncio.Semaphore(plan['concurrency'])
    async def fetch(sha):
        async with semaphore:
            path=cache/f'{sha}.npy';raw=bytearray()
            async for chunk in volume.read_file.aio(f'mel10/{sha}.npy'):
                receipt['transferred_bytes']+=len(chunk)
                if receipt['transferred_bytes']>plan['maximum_download_bytes']:raise ValueError('mel download limit')
                raw.extend(chunk)
            path.write_bytes(raw)
            receipt['files'].append(dict(audio_sha256=sha,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
            receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
    try:
        await asyncio.wait_for(asyncio.gather(*(fetch(sha) for sha in sorted({c['row']['audio_sha256'] for c in panel['cases']}))),plan['overall_timeout_s'])
        receipt['complete']=True
    finally:
        receipt['ended_at_unix']=time.time();receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
    result=[]
    for case in panel['cases']:
        row=case['row'];decision=row['offset_decision'];key=case['case_id']
        mel=np.load(cache/f'{row["audio_sha256"]}.npy',allow_pickle=False)
        if mel.ndim!=2 or mel.shape[1]!=128 or not np.isfinite(mel).all():raise ValueError('bad cached mel')
        peaks=_peaks(flux_envelope(mel));del mel
        truth=[Event.from_json(e) for e in records['candidate'][key]['truth_events']]
        if records['candidate'][key]['truth_events']!=records['control'][key]['truth_events']:
            raise ValueError('baseline and candidate reference changed')
        remaining=decision['r_ms']-decision['shift_ms']
        aligned=[replace(e,onset_us=e.onset_us+remaining*1000,end_us=e.end_us+remaining*1000) for e in truth]
        def anchoring(events):
            heads=np.unique([e.onset_us/1e6 for e in events])
            hits=int(agree_counts(heads,peaks,shifts_ms=[0])[0])
            return dict(unique_heads=len(heads),anchored_heads=hits,fraction=hits/len(heads) if len(heads) else None)
        item=dict(case_id=key,chart_id=row['chart_id'],mode=row['mode'],offset_decision=decision,
            unapplied_residual_ms=remaining,reference_audio_anchoring=anchoring(truth),
            residual_adjusted_reference_audio_anchoring=anchoring(aligned),models={})
        for name in ('control','candidate'):
            pred=[Event.from_json(e) for e in records[name][key]['generated_events']]
            scores={}
            for tol in plan['tolerances_ms']:
                counts=edit_utility(pred,truth,row['mode'],tolerance_us=tol*1000)['timing_point']
                scores[str(tol)]=dict(f1=2*counts['matched']/(counts['pred']+counts['true']) if counts['pred']+counts['true'] else 1.,**counts)
            counts=edit_utility(pred,aligned,row['mode'])['timing_point']
            item['models'][name]=dict(tolerance_curve=scores,audio_anchoring=anchoring(pred),
                anchoring_by_type={kind:anchoring([e for e in pred if e.type==kind]) for kind in sorted({e.type for e in pred})},
                metadata_residual_adjusted_f1_20ms=2*counts['matched']/(counts['pred']+counts['true']) if counts['pred']+counts['true'] else 1.,
                first_error_original_ms=(min(e.onset_us for e in pred)-min(e.onset_us for e in truth))/1000 if pred else None,
                first_error_residual_adjusted_ms=(min(e.onset_us for e in pred)-min(e.onset_us for e in aligned))/1000 if pred else None)
        result.append(item)
    report=dict(schema='beatfcn.reference_clock_audit.v1',complete=True,records=result,
        plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),acceptance_changed=False,
        model_quality_promotion=False,training_updates=0,new_remote_compute=False,limitations=plan['limitations'])
    (ROOT/'reports/d14_reference_alignment.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps([dict(case=r['case_id'],residual=r['unapplied_residual_ms'],
        reference_anchor=r['reference_audio_anchoring']['fraction'],candidate_anchor=r['models']['candidate']['audio_anchoring']['fraction'],
        fixed_f1=r['models']['candidate']['tolerance_curve']['20']['f1'],
        metadata_adjusted_diagnostic_f1=r['models']['candidate']['metadata_residual_adjusted_f1_20ms']) for r in result]))


if __name__=='__main__':asyncio.run(main())
